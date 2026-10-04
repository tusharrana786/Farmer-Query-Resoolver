# Agriculture RAG with PostgreSQL + pgvector

This project builds a vector database for agricultural knowledge and answers user questions by combining semantic retrieval with a language model.

The main flow is:

1. Install and configure PostgreSQL with the pgvector extension
2. Ingest knowledge chunks into the database using `embedder.py`
3. Query the database and generate answers using `rag.py`
4. Run example usage from `query.py`

---

## 1) Vector DB installation

This project stores embeddings in PostgreSQL using the pgvector extension.

### Install PostgreSQL and pgvector

If you are using Windows, install PostgreSQL and then enable the pgvector extension.

You can follow the official pgvector installation steps for your OS:

- PostgreSQL: https://www.postgresql.org/download/
- pgvector: https://github.com/pgvector/pgvector

### Create the database

After PostgreSQL is installed, connect to psql and run:

```sql
CREATE DATABASE ashishproject;
\c ashishproject;
CREATE EXTENSION IF NOT EXISTS vector;
```

### Configure environment variables

The project reads database details from environment variables:

```bash
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=ashishproject
export PGUSER=postgres
export PGPASSWORD=12345678
```

On Windows PowerShell:

```powershell
$env:PGHOST = "localhost"
$env:PGPORT = "5432"
$env:PGDATABASE = "ashishproject"
$env:PGUSER = "postgres"
$env:PGPASSWORD = "12345678"
```

Make sure your PostgreSQL server is running before continuing.

---

## 2) Install Python dependencies

Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you do not have a `requirements.txt` file yet, install the required packages manually:

```bash
pip install psycopg2-binary pgvector sentence-transformers sarvamai
```

> Note: `embedder.py` and `rag.py` use the `SentenceTransformer` model and the Sarvam AI API, so your environment must have internet access for model downloads and API calls.

---

## 3) Run ingestion with `embedder.py`

The `embedder.py` file loads the knowledge chunks from `chunks.json`, creates embeddings using the multilingual E5 model, and stores them in PostgreSQL.

### What it does

- Reads `chunks.json`
- Creates a table named `knowledge_chunks` if it does not exist
- Generates vector embeddings for each document chunk
- Stores the chunk content, metadata, and embedding in PostgreSQL
- Creates a vector index for similarity search

### Run the ingestion

```bash
python embedder.py
```

Example output will look like:

```bash
Loading embedding model: intfloat/multilingual-e5-large
Loading chunks from: ...\chunks.json
Loaded 123 chunks
Setting up PostgreSQL + pgvector...
Generating embeddings...
Inserting chunks + embeddings...
Done.
Inserted/updated 123 chunks.
```

This is the step that prepares the vector database for retrieval.

---

## 4) How to use `rag.py`

The `rag.py` file contains the retrieval and answer-generation logic.

### Main purpose

It does the following:

- connects to PostgreSQL
- converts a user question into an embedding
- finds the most similar knowledge chunks using cosine similarity
- builds a context from retrieved chunks
- sends that context to the Sarvam LLM to generate an answer

### Core functions

- `semantic_search(query, top_k=5)`
  - Finds relevant chunks in the vector DB
- `build_context(chunks)`
  - Creates a readable prompt context from the retrieved documents
- `generate_answer(query, chunks)`
  - Calls the Sarvam LLM with the context and question
- `answer_query(text=None, audio_filename=None)`
  - Accepts either text or a voice file, performs retrieval, and returns the final output

### Example usage

You can call the logic directly in Python:

```python
from rag import answer_query

result = answer_query(
    text="What fertilizer should I use for tomato during flowering?"
)

print(result["query"])
print(result["answer"])
```

You can also provide an audio file:

```python
from rag import answer_query

result = answer_query(
    audio_filename="voicequery.mp3"
)

print(result["query"])
print(result["answer"])
```

---

## 5) Example workflow in `query.py`

The `query.py` file shows exactly how to use the RAG pipeline with both text and audio input.

```python
from rag import answer_query

result = answer_query(
    text="What fertilizer should I use for tomato during flowering?"
)

print("Text Query:")
print(result["query"])
print(result["answer"])
print("-----------------------------------")

result = answer_query(
    audio_filename="voicequery.mp3"
)

print("Voice Query:")
print(result["query"])
print(result["answer"])
```

To run it:

```bash
python query.py
```

This will:

1. run a semantic search on the vector DB
2. retrieve the most relevant agriculture knowledge
3. pass the context to the LLM
4. print the final answer

---

## 6) Project files

- `embedder.py` — creates embeddings and stores them in PostgreSQL
- `rag.py` — retrieval + answer generation pipeline
- `query.py` — example usage with text and audio queries
- `chunks.json` — source data used for ingestion

---

## 7) Typical usage flow

```bash
# 1. Start PostgreSQL and create the database
# 2. Set the PG* environment variables
# 3. Ingest the data
python embedder.py

# 4. Ask a question
python query.py
```

This project is ready for semantic search over agricultural knowledge and answer generation using the vector database plus the Sarvam model.
