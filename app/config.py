from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = "SF Budget Research Agent"
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./research_agent.db")
    seed_urls: tuple[str, ...] = (
        "https://sf.gov/topics/budget",
        "https://sf.gov/budget-process-documents-for-fiscal-years-2026-2027",
        "https://sf.gov/information/mayors-office-finance",
    )
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
    retriever_k: int = int(os.getenv("RAG_RETRIEVER_K", "8"))
    chat_model_mini: str = os.getenv("RAG_CHAT_MINI_MODEL", "gpt-5.4-mini")
    planner_model: str = os.getenv("RAG_PLANNER_MODEL", "gpt-5.4")
    planner_reasoning_effort: str = os.getenv("RAG_PLANNER_REASONING_EFFORT", "high")
    qa_reasoning_effort: str = os.getenv("RAG_QA_REASONING_EFFORT", "low")
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-large")
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")
    rerank_model: str = os.getenv("RAG_RERANK_MODEL", "rerank-v4.0-pro")
    rerank_candidate_pool: int = int(os.getenv("RAG_RERANK_CANDIDATE_POOL", "8"))
    stale_hours: int = int(os.getenv("SOURCE_STALE_HOURS", "24"))


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
