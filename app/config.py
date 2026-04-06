from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(slots=True)
class Settings:
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "samsung_audit")
    postgres_user: str = os.getenv("POSTGRES_USER", "postgres")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "postgres")

    html_dir: str = os.getenv("HTML_DIR", "./data/html")
    parsed_dir: str = os.getenv("PARSED_DIR", "./data/parsed")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))
    note_chunk_size: int = int(os.getenv("NOTE_CHUNK_SIZE", "1100"))
    note_chunk_overlap: int = int(os.getenv("NOTE_CHUNK_OVERLAP", "150"))
    text_chunk_size: int = int(os.getenv("TEXT_CHUNK_SIZE", "1000"))
    text_chunk_overlap: int = int(os.getenv("TEXT_CHUNK_OVERLAP", "120"))
    table_row_group_size: int = int(os.getenv("TABLE_ROW_GROUP_SIZE", "12"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    hybrid_candidate_k: int = int(os.getenv("HYBRID_CANDIDATE_K", "15"))
    semantic_weight: float = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))
    keyword_weight: float = float(os.getenv("KEYWORD_WEIGHT", "0.3"))
    use_reranker: bool = os.getenv("USE_RERANKER", "false").lower() in {"1", "true", "yes", "y"}
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "20"))
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    llm_model: str = os.getenv("LLM_MODEL", "Qwen/Qwen3-0.6B")
    llm_max_new_tokens: int = int(os.getenv("LLM_MAX_NEW_TOKENS", "384"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    llm_top_p: float = float(os.getenv("LLM_TOP_P", "0.9"))
    retrieval_profile: str = os.getenv("RETRIEVAL_PROFILE", "local")
    pipeline_id: str = os.getenv("PIPELINE_ID", "default")

    @property
    def db_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
