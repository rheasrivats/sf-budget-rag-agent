"""Set up embeddings + vector store for the LangChain RAG tutorial."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

EMBEDDING_MODEL = "text-embedding-3-large"


def create_embeddings(model: str = EMBEDDING_MODEL) -> OpenAIEmbeddings:
    """Create the OpenAI embeddings client used by the RAG tutorial."""
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")
    return OpenAIEmbeddings(model=model)


def create_vector_store(
    embeddings: OpenAIEmbeddings | None = None,
) -> InMemoryVectorStore:
    """Create an in-memory vector store backed by the selected embeddings model."""
    embeddings = embeddings or create_embeddings()
    return InMemoryVectorStore(embeddings)


if __name__ == "__main__":
    embeddings = create_embeddings()
    vector_store = create_vector_store(embeddings)
    print(f"Embeddings model: {EMBEDDING_MODEL}")
    print(f"Vector store: {type(vector_store).__name__}")
    print("Embeddings + vector store initialized.")
