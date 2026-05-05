# Multi-User Conversational RAG

Production-style Retrieval-Augmented Generation system that supports user-scoped document retrieval, persistent conversational memory, vector search, and LLM-based evaluation.

This project demonstrates how to build a practical multi-user RAG backend using LangChain, OpenAI models, ChromaDB, SQLite-backed chat history, and DeepEval metrics. It is designed around a realistic product requirement: every user can query shared knowledge while also retrieving from their own uploaded documents, without leaking private context across users.

## Why This Project Matters

Most simple RAG demos answer one-off questions over a single document set. This project goes a step further by handling the problems that appear in real applications:

- Multi-user retrieval isolation through metadata filters.
- Shared plus private knowledge access in the same retrieval pipeline.
- Conversation-aware question reformulation for follow-up questions.
- Persistent chat memory across sessions using SQLite.
- Persistent vector storage using ChromaDB.
- Batch ingestion for larger corpora.
- Evaluation coverage for answer relevance, faithfulness, contextual precision, contextual recall, contextual relevancy, and hallucination risk.

## Core Capabilities

- **Conversational RAG**: Maintains dialogue context and rewrites follow-up questions into standalone retrieval queries.
- **Multi-user document access**: Retrieves from both global SimpleWiki content and documents uploaded by the active user.
- **User-level access control at retrieval time**: Applies Chroma metadata filters so each user only sees their own private documents plus shared public knowledge.
- **Persistent memory**: Stores conversation history in `memory.db` using LangChain's SQL chat history integration.
- **Persistent vector database**: Stores embeddings in `data/vector_store` using ChromaDB.
- **OpenAI-powered generation and embeddings**: Uses `gpt-4o-mini` for answer generation and `text-embedding-3-small` for embeddings.
- **LLM evaluation harness**: Uses DeepEval with `gpt-4o` as evaluator model for retrieval and answer-quality metrics.

## Architecture

```text
User Prompt
    |
    v
ConversationChain
    |
    |-- loads user-specific chat history from SQLite
    |
    v
History-Aware Retriever
    |
    |-- rewrites follow-up questions using chat history
    |-- searches Chroma vector store
    |-- filters context by:
        - shared SimpleWiki documents
        - documents belonging to the current user_id
    |
    v
RAG Answer Chain
    |
    |-- injects retrieved context
    |-- includes recent chat history window
    |-- generates grounded response
    |
    v
Answer + Retrieved Context
```

## Repository Structure

```text
Project_Resume_Multi_User_Rag/
|-- scripts/
|   |-- download_data.py             # Downloads source dataset
|   |-- run_ingestion.py             # Ingests shared SimpleWiki corpus
|   |-- ingest_user_documents.py     # Ingests documents for a specific user
|   |-- inference.py                 # CLI conversational RAG interface
|   `-- run_evaluation.py            # Runs DeepEval quality checks
|-- src/
|   `-- multi_user_rag/
|       |-- config.py                # Models, chunking, paths, vector settings
|       |-- chains/
|       |   |-- conversation_chain.py # Persistent multi-user conversation wrapper
|       |   `-- rag_chain.py          # History-aware RAG chain
|       |-- ingestion/
|       |   |-- ingestion.py          # Shared and user ingestion orchestration
|       |   |-- loader.py             # SimpleWiki loader
|       |   |-- splitter.py           # Recursive text splitter factory
|       |   |-- user_loader.py        # Per-user document loader
|       |   `-- vector_store.py       # Chroma vector store abstraction
|       |-- retrieval/
|       |   `-- retriever.py          # User-filtered retriever service
|       `-- evaluation/
|           `-- evaluator.py          # DeepEval metric wrapper
|-- data/
|   |-- raw/                          # Downloaded source corpus
|   |-- user_uploads/                 # User-specific uploaded documents
|   `-- vector_store/                 # Persisted ChromaDB index
|-- memory.db                         # SQLite chat history database
`-- README.md
```

## Technical Design

### 1. Ingestion Pipeline

The ingestion service loads documents, splits them into overlapping chunks, embeds them using OpenAI embeddings, and stores them in ChromaDB.

Configuration:

- Chunk size: `2000`
- Chunk overlap: `200`
- Embedding model: `text-embedding-3-small`
- Vector store: ChromaDB
- Distance metric: cosine similarity
- Collection name: `multi_user_collection`

Shared corpus ingestion is handled by:

```bash
python scripts/run_ingestion.py
```

User-specific document ingestion is handled by:

```bash
python scripts/ingest_user_documents.py
```

User files should be placed under:

```text
data/user_uploads/<user_id>/
```

The current user loader supports `.txt` files and attaches metadata:

```python
{
    "source": "user",
    "user_id": "<user_id>",
    "filename": "<filename>"
}
```

### 2. Retrieval Strategy

The retriever uses a metadata filter to combine shared and private knowledge:

```python
{
    "$or": [
        {"source": "simplewiki"},
        {"user_id": user_id}
    ]
}
```

This allows every user to access common knowledge while preserving isolation for user-uploaded documents.

### 3. Conversational Memory

Conversation history is stored in SQLite through `SQLChatMessageHistory`, keyed by `session_id`. In this implementation, the `user_id` is passed as the session identifier, which allows each user to continue prior conversations.

To keep prompts efficient, the RAG chain uses a memory window over the most recent conversation turns.

### 4. History-Aware RAG

Follow-up questions are reformulated into standalone questions before retrieval. For example:

```text
User: What is machine learning?
User: How is it different from deep learning?
```

The second question is rewritten with conversation context before vector search, improving retrieval quality for multi-turn dialogue.

### 5. Evaluation

The evaluation module wraps DeepEval metrics for both answer quality and retrieval quality:

- Contextual precision
- Contextual recall
- Contextual relevancy
- Answer relevancy
- Faithfulness
- Hallucination score derived from faithfulness

Run evaluation with:

```bash
python scripts/run_evaluation.py
```

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

Install the core packages used by the project:

```bash
pip install langchain langchain-core langchain-community langchain-openai langchain-chroma langchain-text-splitters chromadb openai python-dotenv deepeval requests tqdm
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Running The Project

### Download a dataset

```bash
python scripts/download_data.py --url <dataset_url>
```

By default, the file is saved to:

```text
data/raw/simplewiki-2020-11-01.jsonl.gz
```

### Ingest shared documents

```bash
python scripts/run_ingestion.py
```

### Add and ingest user documents

Place user files in:

```text
data/user_uploads/<user_id>/
```

Then run:

```bash
python scripts/ingest_user_documents.py
```

### Start interactive inference

```bash
python scripts/inference.py
```

Example session:

```text
User_Id: Ari_08

Conversational RAG System (type 'exit' to quit)

Prompt: What is machine learning?
Assistant: Machine learning is ...
```

## Example Use Case

A resume-review or enterprise knowledge assistant can use this pattern to serve many users from one application:

- Shared knowledge base: company policies, public documentation, FAQs, domain corpus.
- Private user knowledge: resumes, notes, uploaded files, personal documents.
- Conversation memory: ongoing advisory sessions that preserve context.
- Retrieval isolation: user A cannot retrieve user B's private documents.

## Model Configuration

The default model settings live in `src/multi_user_rag/config.py`:

```python
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-4o-mini"
EVALUATOR_MODEL = "gpt-4o"
TEMPERATURE = 0
```

These defaults favor deterministic, grounded answers while keeping inference cost practical.

## Current Limitations And Next Improvements

- User document ingestion currently supports `.txt` files. PDF, DOCX, HTML, and Markdown loaders would make this more production-ready.
- Authentication and authorization are represented by `user_id`; a production deployment should integrate identity management.
- The CLI interface can be extended into a FastAPI service or Streamlit app.
- SimpleWiki document metadata can be enriched further for better source attribution and stricter shared-corpus filtering.
- Automated regression tests can be added around ingestion, retrieval filtering, and conversation memory.

## Skills Demonstrated

- Retrieval-Augmented Generation architecture
- LangChain chain composition
- Multi-user metadata filtering
- ChromaDB vector persistence
- OpenAI embeddings and chat models
- Conversation memory with SQLite
- RAG evaluation with DeepEval
- Modular Python project design
- Practical LLM application engineering

## Summary

This project is a strong foundation for a multi-user AI assistant that combines shared knowledge, user-private retrieval, persistent memory, and measurable answer quality. It shows not only how to build a working RAG pipeline, but also how to think about the operational concerns that matter in real-world LLM applications: isolation, persistence, evaluation, and maintainable system design.
