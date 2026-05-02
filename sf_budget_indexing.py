"""Load, split, and index San Francisco budget documents for RAG."""

from __future__ import annotations

import os
from typing import List, Tuple

os.environ.setdefault("USER_AGENT", "sf-budget-rag-agent/1.0")

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import RateLimitError

from rag_embeddings_store import create_vector_store

SF_BUDGET_URLS: Tuple[str, ...] = (
    "https://sf.gov/topics/budget",
    "https://sf.gov/budget-process-documents-for-fiscal-years-2026-2027",
    "https://sf.gov/information/mayors-office-finance",
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_documents() -> List[Document]:
    """Load San Francisco budget pages."""
    loader = WebBaseLoader(web_paths=SF_BUDGET_URLS)
    return loader.load()


def split_documents(docs: List[Document]) -> List[Document]:
    """Split loaded pages into retrieval-friendly chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def build_budget_vector_store() -> tuple[InMemoryVectorStore, List[Document]]:
    """Load, split, and index SF budget documents into an in-memory vector store."""
    docs = load_documents()
    splits = split_documents(docs)
    vector_store = create_vector_store()
    vector_store.add_documents(documents=splits)
    return vector_store, splits


def _preview_retrieval(vector_store: InMemoryVectorStore, k: int = 3) -> None:
    """Run a sample retrieval query to verify indexing worked."""
    query = "What is the timeline for the San Francisco budget process?"
    retrieved_docs = vector_store.similarity_search(query, k=k)
    print(f"Sample query: {query}")
    print(f"Top {k} retrieved chunks:")
    for idx, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "unknown")
        excerpt = doc.page_content[:220].replace("\n", " ")
        print(f"{idx}. Source: {source}")
        print(f"   Excerpt: {excerpt}...")


if __name__ == "__main__":
    print("Loading SF budget documents...")
    docs = load_documents()
    print(f"Loaded documents: {len(docs)}")

    print("Splitting documents...")
    splits = split_documents(docs)
    print(f"Total chunks: {len(splits)}")
    print(f"Chunk config: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")

    print("Indexing into InMemoryVectorStore...")
    vector_store = create_vector_store()
    try:
        vector_store.add_documents(documents=splits)
    except RateLimitError:
        print(
            "Embedding request failed: OpenAI returned insufficient quota (429). "
            "Your API key is recognized, but billing/quota needs to be enabled."
        )
        print(
            "After updating quota, rerun: source .venv/bin/activate && "
            "python sf_budget_indexing.py"
        )
        raise SystemExit(1)
    print("Indexing complete.")

    _preview_retrieval(vector_store)
