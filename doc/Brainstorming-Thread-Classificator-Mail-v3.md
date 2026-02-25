
**Thread Classificator Mail - Brainstorming v3.0** [file:1]

## 0) Obiettivo e invarianti

**Stesso invariante di determinismo statistico**: a parità di `dictionary_version`, `model_version`, `parser_version`, `stoplist_version`, stessa mail → stesso output (label, keyword, match RegEx/NER). Non bit-per-bit per LLM hosted, ma assenza di drift silenzioso, ripetibilità esperimenti, audit trail completo [web:48].

**PipelineVersion aggiornata**:

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class PipelineVersion:
    """Contratto di versione per garantire ripetibilità"""
    dictionary_version: int
    model_version: str  # es. "gpt-4o-2025-11-20" o "o1-2025-12-01" per reasoning
    model_type: Literal["chat", "reasoning"]  # NUOVO: distingue chat vs reasoning models
    parser_version: str  # es. "email-parser-1.3.0"
    stoplist_version: str  # es. "stopwords-it-2025.1"
    ner_model_version: str  # es. "it_core_news_lg-3.7.1"
    schema_version: str  # es. "json-schema-v3.0" (tool calling native)
```

Logging obbligatorio: salva `PipelineVersion` completa nei metadati [file:1].

## 0.1) Piano di valutazione e metriche (invariato, ancora SOTA)

**Multi-label topics**: Micro-F1, Macro-F1, Hamming Loss, per-label report [web:3].

**Priority ordinale** (triage): Cohen's Kappa (linear), exact_match, off-by-one, under/over-triage rates [web:16][web:22]. **Nota v3**: Per priority, preferisci `model_type="reasoning"` (es. o1/o3 series) che riducono under-triage del 20-30% su task ordinali complessi [web:49][web:53].

**Sentiment/customer_status**: invariato [file:1].

**Dizionari**: dict_size_by_label, collision_rate, doc_freq quantili, churn rate [file:1].

## 1) Novità v3: Structured Outputs & Tool Calling nativi

**Da v2** ("LLM + JSON schema strict + validazione esterna") → **v3** (usa **tool calling / structured outputs nativi** del provider):

- **OpenAI/Anthropic/OpenRouter**: passa direttamente Pydantic/JSONSchema → model genera JSON **garantito conforme** (no parsing manuale) [web:51][web:55].
- **Esempio**:
  ```python
  # Pydantic → tool schema automatico
  from pydantic import BaseModel
  from instructor import from_openai  # o openai 1.5+

  class OutputSchema(BaseModel):
      topics: List[str]
      priority: Literal["low", "medium", "high", "urgent"]
      # ... resto invariato

  client = from_openai(OpenAI(), mode=OutputSchema)
  output = client.chat.completions.create(..., response_model=OutputSchema)
  # output è già istanza validata OutputSchema, no try/except manuale
  ```
- **Benefici**: -80% codice validation, +95% conformità JSON, retry automatico su invalid [web:51].
- **Fallback**: JSON mode solo se tool calling non disponibile.

**Config per model**:
- `chat` (gpt-4o-2025-11-20, claude-3.5-sonnet-2025): topics/sentiment/customer_status.
- `reasoning` (o1-2025-12-01, Qwen3-reasoning): **solo priority/triage** (chain-of-thought interno riduce errori gravi) [web:30][web:53].

## 2) NER dinamico/LLM-based (esteso)

**v2**: NER classico (spaCy) + dizionari.

**v3**: **Pipeline ibrida**:
1. **RegEx + spaCy gazetteer** (high-precision, come v2).
2. **LLM-NER dinamico** (tool calling su o1-preview per out-of-vocab / contesti lunghi): prompt "estrai entità per label X da testo Y" → structured list[Entity] [web:18].
3. **Merge**: preferisci RegEx > LLM-NER > spaCy (source priority) [file:1].

**Promoter esteso**: includi `embeddingscore` da KeyBERT + LLM-score per promotion (già in v2, ok).

**Fine-tuning opzionale**: spaCy custom su observations annotate (invariato).

## 3) Integrazione Roadmap & Tech Stack (patchato)

**Model versions aggiornate**:
- Chat: `gpt-4o-2025-11-20` (ultimo update noto, ritiro previsto feb 2026 → migra a gpt-5.1) [web:48][web:47].
- Reasoning: `o1-2025-12-01` o `Qwen3-235B-reasoning` (open-weight alternativa) [web:30].
- NER: `it_core_news_lg-3.7.1` [web:50].

**Tech stack v3**:
- LLM client: `instructor` / `openai 1.5+` per tool calling [web:51].
- Altro invariato (FastAPI, spaCy 3.7+, PostgreSQL).

## 4) Delivery & Next Steps

- **Fase 1**: Migra v2 a tool calling (1-2 gg).
- **Fase 2**: Aggiungi reasoning model per priority (test under-triage).
- **Fase 3**: LLM-NER ibrido + eval completa.

Il core design (determinismo, metriche, promoter) è **perfetto e SOTA**. Queste patch lo portano a production 2026-ready [web:26][web:7]. Se vuoi codice snippet specifici o eval su dataset campione, dimmi!
