import os
from pathlib import Path

import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from sarvamai import SarvamAI


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Use the exact Sarvam model IDs available in your account/docs.
SARVAM_LLM = "sarvam-105b"
SARVAM_STT = "saaras:v4"

TOP_K = 5

DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "ashishproject"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "12345678"),
}


# ---------------------------------------------------------
# INITIALIZE CLIENTS
# ---------------------------------------------------------

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

sarvam = SarvamAI(
    api_subscription_key="<Insert your Sarvam API key here>",
)


# ---------------------------------------------------------
# DATABASE CONNECTION
# ---------------------------------------------------------

def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)

    # Allows psycopg2 to understand pgvector values.
    register_vector(conn)

    return conn


# ---------------------------------------------------------
# VOICE -> TEXT
# ---------------------------------------------------------

def transcribe_audio(audio_filename: str) -> str:

    audio_path = Path(audio_filename)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path.resolve()}"
        )

    with audio_path.open("rb") as audio_file:

        response = sarvam.speech_to_text.transcribe(
            file=audio_file,
            model=SARVAM_STT,
            mode="transcribe",
        )

    query = response.transcript.strip()

    if not query:
        raise ValueError("Could not extract any text from audio.")

    return query


# ---------------------------------------------------------
# SEMANTIC SEARCH
# ---------------------------------------------------------

def semantic_search(query: str, top_k: int = TOP_K):

    # IMPORTANT:
    # E5 expects "query:" for search queries.
    query_embedding = embedding_model.encode(
        f"query: {query}",
        normalize_embeddings=True
    )

    conn = get_db_connection()

    try:

        sql = """
            SELECT
                chunk_id,
                crop,
                stage,
                condition,
                language,
                content,
                metadata,

                1 - (embedding <=> %s) AS similarity

            FROM knowledge_chunks

            WHERE embedding IS NOT NULL

            ORDER BY embedding <=> %s

            LIMIT %s;
        """

        with conn.cursor() as cursor:

            cursor.execute(
                sql,
                (
                    query_embedding,
                    query_embedding,
                    top_k,
                )
            )

            rows = cursor.fetchall()

        results = []

        for row in rows:

            results.append({
                "chunk_id": row[0],
                "crop": row[1],
                "stage": row[2],
                "condition": row[3],
                "language": row[4],
                "content": row[5],
                "metadata": row[6],
                "similarity": float(row[7]),
            })

        return results

    finally:
        conn.close()


# ---------------------------------------------------------
# BUILD CONTEXT FOR LLM
# ---------------------------------------------------------

def build_context(chunks):

    if not chunks:
        return "No relevant information was found."

    context = []

    for i, chunk in enumerate(chunks, start=1):

        context.append(
            f"""
[Knowledge Chunk {i}]

Crop: {chunk["crop"]}
Stage: {chunk["stage"]}
Condition: {chunk["condition"]}
Language: {chunk["language"]}

Content:
{chunk["content"]}
"""
        )

    return "\n".join(context)


# ---------------------------------------------------------
# SARVAM-105B
# ---------------------------------------------------------

def generate_answer(query: str, chunks):

    context = build_context(chunks)

    system_prompt = """
You are an agricultural fertilizer knowledge-base assistant.

Answer the user's question using the provided knowledge-base context.

Rules:

1. Use the knowledge base as the primary source of truth.
2. Do not invent fertilizer names, doses, timings, or application methods.
3. Preserve fertilizer/product names accurately.
4. Preserve numeric doses accurately.
5. If the knowledge base does not contain enough information,
   explicitly say that the available knowledge base does not provide
   enough information.
6. If multiple chunks are relevant, combine them carefully.
7. Give a concise and practical answer.
"""

    user_prompt = f"""
Knowledge Base:

{context}

User Question:

{query}

Answer the question using the knowledge base.
"""

    response = sarvam.chat.completions(
        model=SARVAM_LLM,

        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        reasoning_effort=None,
        temperature=0.2,
        max_tokens=1000,
    )

    return response.choices[0].message.content.strip()


# ---------------------------------------------------------
# MAIN METHOD
# ---------------------------------------------------------

def answer_query(
    text: str | None = None,
    audio_filename: str | None = None,
    top_k: int = TOP_K,
):

    # Exactly one input should be supplied.
    if (text is None) == (audio_filename is None):

        raise ValueError(
            "Provide exactly one of text or audio_filename."
        )

    # ---------------------------------------------
    # STEP 1: Convert input -> text
    # ---------------------------------------------

    if audio_filename:

        query = transcribe_audio(audio_filename)

        input_type = "voice"

    else:

        query = text.strip()

        if not query:
            raise ValueError("Text query cannot be empty.")

        input_type = "text"

    # ---------------------------------------------
    # STEP 2: Semantic retrieval
    # ---------------------------------------------

    chunks = semantic_search(
        query=query,
        top_k=top_k,
    )

    # ---------------------------------------------
    # STEP 3: Send context to Sarvam-105B
    # ---------------------------------------------

    answer = generate_answer(
        query=query,
        chunks=chunks,
    )

    # ---------------------------------------------
    # Return everything useful
    # ---------------------------------------------

    return {
        "input_type": input_type,
        "query": query,
        "chunks": chunks,
        "answer": answer,
    }