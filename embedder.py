import json
import os
from pathlib import Path

import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "intfloat/multilingual-e5-large"
JSON_PATH = Path(__file__).parent / "chunks.json"

DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "ashishproject"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "12345678"),
}


# -----------------------------
# Database setup
# -----------------------------
CREATE_TABLE_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id BIGSERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    document_id TEXT NOT NULL,
    image_id TEXT,
    crop TEXT,
    stage TEXT,
    condition TEXT,
    language TEXT,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding VECTOR(1024),
    created_at TIMESTAMP DEFAULT NOW()
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS knowledge_chunks_embedding_idx
ON knowledge_chunks
USING hnsw (embedding vector_cosine_ops);
"""


def get_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn


def load_chunks():
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"chunks.json not found at: {JSON_PATH}")

    with JSON_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def setup_database(conn):
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
        conn.commit()

        cur.execute(CREATE_INDEX_SQL)
        conn.commit()


def create_embeddings(model, chunks):
    """
    E5 models work best when documents are prefixed with 'passage:'.
    User queries should later be prefixed with 'query:'.
    """
    texts = [f"passage: {chunk['content']}" for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings


def insert_chunks(conn, chunks, embeddings):
    sql = """
    INSERT INTO knowledge_chunks (
        chunk_id,
        document_id,
        image_id,
        crop,
        stage,
        condition,
        language,
        content,
        metadata,
        embedding
    )
    VALUES (
        %(chunk_id)s,
        %(document_id)s,
        %(image_id)s,
        %(crop)s,
        %(stage)s,
        %(condition)s,
        %(language)s,
        %(content)s,
        %(metadata)s,
        %(embedding)s
    )
    ON CONFLICT (chunk_id)
    DO UPDATE SET
        document_id = EXCLUDED.document_id,
        image_id = EXCLUDED.image_id,
        crop = EXCLUDED.crop,
        stage = EXCLUDED.stage,
        condition = EXCLUDED.condition,
        language = EXCLUDED.language,
        content = EXCLUDED.content,
        metadata = EXCLUDED.metadata,
        embedding = EXCLUDED.embedding;
    """

    with conn.cursor() as cur:
        for chunk, embedding in zip(chunks, embeddings):
            params = {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "image_id": chunk.get("image_id"),
                "crop": chunk.get("crop"),
                "stage": chunk.get("stage"),
                "condition": chunk.get("condition"),
                "language": chunk.get("language"),
                "content": chunk["content"],
                "metadata": json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
                "embedding": embedding,
            }

            cur.execute(sql, params)

        conn.commit()


def semantic_search(conn, query, top_k=5, crop=None, language=None):
    """
    Semantic search using cosine distance.

    E5 convention:
      query -> 'query: ...'
      knowledge-base passages -> 'passage: ...'
    """
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode(
        f"query: {query}",
        normalize_embeddings=True,
    )

    sql = """
    SELECT
        id,
        chunk_id,
        crop,
        stage,
        condition,
        language,
        content,
        metadata,
        1 - (embedding <=> %s) AS similarity
    FROM knowledge_chunks
    WHERE 1 = 1
    """

    params = [query_embedding]

    if crop:
        sql += " AND crop = %s"
        params.append(crop)

    if language:
        sql += " AND language = %s"
        params.append(language)

    sql += """
    ORDER BY embedding <=> %s
    LIMIT %s;
    """

    params.extend([query_embedding, top_k])

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    return rows


def main():
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Loading chunks from: {JSON_PATH}")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    conn = get_connection()

    try:
        print("Setting up PostgreSQL + pgvector...")
        setup_database(conn)

        print("Generating embeddings...")
        embeddings = create_embeddings(model, chunks)

        print("Inserting chunks + embeddings...")
        insert_chunks(conn, chunks, embeddings)

        print("Done.")
        print(f"Inserted/updated {len(chunks)} chunks.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
