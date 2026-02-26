# Observation Layer — Avanzamento Lavori

> Ultimo aggiornamento: 2026-02-26
> Scope: Observation Storage + Promoter (il triage LLM è già esistente)
> Stack: FastAPI · PostgreSQL · SQLAlchemy async · Celery + Redis · Prometheus + Grafana

---

## Legenda
- [ ] Non iniziato
- [~] In corso
- [x] Completato

---

## FASE A — Infrastruttura & Contratti Dati

- [x] A1 — Struttura repository (`src/`, `tests/`, `scripts/`, `config/`, `docker-compose.yml` con PostgreSQL + Redis + API + Worker)
- [x] A2 — Pydantic contracts: `EntityExtractionOutput`, `TriageOutput`, `ObservationBatch` (allineati a `message_envelope.json`)
- [x] A3 — Dataclass `PipelineVersion` v3 (con campo `model_type: Literal["chat", "reasoning"]`)

## FASE B — Schema DB PostgreSQL

- [x] B1 — Migration Alembic: tabella `pipeline_runs`
- [x] B2 — Migration Alembic: tabella `messages`
- [x] B3 — Migration Alembic: tabella `keyword_observations` (UNIQUE su `message_id, label_id, candidate_id, dict_version`)
- [x] B4 — Migration Alembic: tabella `entity_observations` (UNIQUE su `message_id, text_hash, type, start, end, source, extractor_version`)
- [x] B5 — Migration Alembic: tabelle `label_registry` e `lexicon_entries`
- [x] B6 — Migration Alembic: tabella `promotion_events` (append-only)
- [x] B7 — Indici su `(label_id, status)`, `(lemma)`, `(observed_at)`

## FASE C — Observation Storage Engine

- [x] C1 — `src/observation_store/writer.py`: `persist_batch()` con upsert idempotente in singola transazione
- [x] C2 — `src/observation_store/reader.py`: query per promoter (`get_doc_freq_by_label_lemma`, `get_collision_index`, `get_quarantined_above_threshold`)
- [x] C3 — FastAPI route `POST /observations/batch` → risponde con `ObservationBatchAck` (created, skipped, errors)

## FASE D — Promoter

- [x] D1 — `src/dictionary/promoter.py`: `KeywordPromoter` con config da `config/promoter_config.yaml`
- [x] D2 — `src/dictionary/versioning.py`: `create_new_version()` con PostgreSQL advisory lock (single-writer)
- [x] D3 — `src/dictionary/collision_detector.py`: `get_collision_index()` e `get_collision_rate()`
- [x] D4 — `scripts/batch_promote.py`: batch job Celery/cron per promozione fine-run

## FASE E — Privacy & Compliance

- [x] E1 — Verifica no PII raw in `entity_observations` per tipi sensibili (CODICEFISCALE, EMAIL, TELEFONO, IBAN): solo `value_hash` (SHA-256) / `value_enc`
- [x] E2 — Campo `expires_at` su `keyword_observations` ed `entity_observations` + task `scripts/apply_retention.py`

## FASE F — Monitoring & Alerting

- [x] F1 — `src/api/metrics.py`: metriche Prometheus (`collision_rate`, `promotion_rate`, `quarantined_total`, `dict_size_by_label`, `unknown_topic_rate`)
- [x] F2 — Alert rules: `collision_rate > 0.15`, `churn_rate > 0.30`, `promotion_rate < 0.02 | > 0.50`, `UNKNOWN_TOPIC > 20%`
- [x] F3 — FastAPI route `GET /dictionaries/health` con snapshot metriche

## FASE G — Test

- [x] G1 — `tests/test_idempotency.py`: re-run stessa mail non crea duplicati
- [x] G2 — `tests/test_spans.py`: ogni span in `[0, len(text_canonical)]`, substring ↔ quote
- [x] G3 — `tests/test_versioning.py`: ogni observation ha `pipeline_version` completa, NOT NULL constraints
- [x] G4 — `tests/test_performance.py`: budget per step (regex, NER, merge), fallback se NER supera soglia
- [x] G5 — `tests/test_promoter.py`: promozione, quarantena, reject, lock single-writer

---

## Smoke test finali

- [ ] `POST /observations/batch` con payload `message_envelope.json` → `observations_created=7`, `entities_created=1`, `errors=[]`
- [ ] Stessa POST ripetuta → `observations_created=0`, `skipped_idempotent=7`
- [ ] `batch_promote.py` su 3+ messaggi con stessa keyword → keyword passa da `candidate` ad `active`
- [ ] `GET /dictionaries/health` ritorna metriche coerenti
- [ ] Dashboard Grafana visualizza `collision_rate` e `dict_size_by_label`

---

## Note & Decisioni

| # | Decisione |
|---|-----------|
| 1 | Scope limitato a Observation Storage + Promoter; pipeline triage non modificata |
| 2 | Idempotenza via chiavi naturali + `ON CONFLICT DO NOTHING` |
| 3 | PII: mai `value` raw per tipi sensibili, solo `value_hash` SHA-256 |
| 4 | Versioning single-writer con PostgreSQL advisory lock |
| 5 | Nessuna stima temporale — sequenza logica A→B→C→D→E→F→G |
