# Email Triage Pipeline

Production-ready email triage system that classifies emails into multi-label topics, priority, sentiment, and customer status using deterministic dictionary-based NER + LLM (tool calling) with versioned lexicons and Observation Storage.

## Architecture

```
Email Ingestion → Preprocessing → Candidate Generation → LLM Classification
       ↓                                                         ↓
Entity Extraction ←─────────────────────────────── Post-Processing
       ↓
Observation Storage → Dictionary Promoter → Dictionary Version Management
```

## Quick Start

```bash
# 1. Copy env template
cp config/.env.example .env

# 2. Start infrastructure
docker compose up -d postgres redis

# 3. Install dependencies
pip install -e ".[dev]"
python -m spacy download it_core_news_lg

# 4. Run DB migrations
alembic upgrade head

# 5. Start API
uvicorn src.api.app:app --reload

# 6. Start Celery worker (in a separate terminal)
celery -A src.api.worker worker --loglevel=info
```

## Running Tests

```bash
pytest --cov=src --cov-report=html
```

## Project Structure

```
email-triage/
├── src/
│   ├── contracts.py          # Pydantic data contracts (v3.0)
│   ├── config.py             # Settings & environment
│   ├── ingestion/            # Email parsing & canonicalization
│   ├── candidates/           # Deterministic keyword candidates
│   ├── classification/       # LLM classification (tool calling)
│   ├── enrichment/           # Customer status & priority scorer
│   ├── entity_extraction/    # NER pipeline (RegEx + spaCy + LLM)
│   ├── dictionary/           # Observation Storage + Promoter
│   ├── evaluation/           # Metrics & drift detection
│   ├── monitoring/           # Prometheus metrics
│   ├── privacy/              # PII handler (GDPR)
│   └── api/                  # FastAPI app + Celery worker
├── tests/
│   ├── unit/
│   ├── integration/
│   └── performance/
├── scripts/                  # DB init, maintenance scripts
├── config/                   # .env.example, alerts.yml
├── alembic/                  # DB migrations
└── docker-compose.yml
```

## Pipeline Invariants

- **Determinism**: same `dictionary_version + model_version + parser_version` → same output
- **Idempotency**: re-processing same message_id is safe (upsert via natural keys)
- **Audit trail**: every run stores `PipelineVersion` + `RemovedSection` log
- **Freeze/update**: dictionaries frozen during a run, updated only end-of-batch
