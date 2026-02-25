"""
Application configuration loaded from environment variables.
All secrets are injected via environment (never hard-coded).
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    # "chat" model: topics / sentiment / customer_status
    llm_chat_model: str = "openai/gpt-4o-2025-11-20"
    # "reasoning" model: priority / triage (reduces under-triage 20-30%)
    llm_reasoning_model: str = "openai/o1-2025-12-01"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_temperature: float = 0.1
    llm_max_retries: int = 3
    llm_timeout_seconds: int = 60

    # ── Embedding / NER ───────────────────────────────────────────────────────
    embedding_model: str = "paraphrase-multilingual-mpnet-base-v2"
    spacy_model: str = "it_core_news_lg"

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://triage:triage@localhost:5432/email_triage"
    database_url_sync: str = "postgresql://triage:triage@localhost:5432/email_triage"
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # ── Redis / Celery ────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # ── Privacy / GDPR ────────────────────────────────────────────────────────
    # Fernet key (base64-encoded 32 bytes). Generate: Fernet.generate_key()
    encryption_key: str = ""
    pii_retention_days: int = 90

    # ── Monitoring ────────────────────────────────────────────────────────────
    enable_metrics: bool = True
    metrics_port: int = 9090

    # ── Pipeline behaviour ───────────────────────────────────────────────────
    canonicalization_version: str = "1.2.0"
    stoplist_version: str = "stopwords-it-2025.1"
    ner_model_version: str = "it_core_news_lg-3.7.1"
    schema_version: str = "json-schema-v3.0"
    default_dictionary_version: int = 1
    # Number of top candidates sent to LLM
    max_candidates_to_llm: int = 100
    # Max body length (chars) sent to LLM
    max_body_chars: int = 8000

    # CRM integration
    crm_base_url: str = ""
    crm_api_key: str = ""

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" | "console"


@lru_cache
def get_settings() -> Settings:
    return Settings()
