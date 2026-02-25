# Observation Storage & Triage Mail — Documento consolidato

Questo file consolida: (a) i ragionamenti e le proposte emerse in chat sull’Observation Storage, (b) un’analisi “production-ready” punto per punto con sezioni esplicative, (c) l’inclusione integrale delle sezioni dei brainstorming v2 e v3, (d) i dettagli rilevanti dell’esempio `message_envelope.json`.

## 1. Contesto e obiettivi
Il problema: costruire una pipeline di triage email che produca classificazioni (topics multi-label, priority, sentiment, customer status) e segnali spiegabili (evidence/spans, keyword selezionate, entità estratte), e che migliori nel tempo via dizionari/lexicon versionati.
La soluzione proposta: introdurre un layer “Observation Storage” come source-of-truth persistente e interrogabile, che rende possibile audit, backtesting, monitoring, e aggiornamento deterministico dei dizionari (freeze in-run, update end-of-run).

## 2. Cos’è l’Observation Storage (ruolo operativo)
**Definizione operativa**: un layer di persistenza versionata delle “osservazioni” prodotte dal triage (keyword per topic + evidence/spans, segnali di priority, entità estratte, metadati di run/versioni), scritto in modo idempotente e interrogabile.
Serve a due funzioni operative: (1) tracciabilità/quality monitoring (per capire cosa è successo e perché), (2) alimentare il promoter per aggiornare dizionari/lexicon in modo controllato e ripetibile.

## 3. Ragionamenti punto per punto (con miglioramenti)
### 3.1 Determinismo ripetibile (freeze in-run / update end-of-run)
Ragionamento: se un run usa un dizionario che può cambiare “mentre sta elaborando”, non puoi più riprodurre gli esiti e perdi audit trail.
Miglioramento: ogni osservazione deve essere agganciata a `dictionaryversion_used` e ad un contratto di versioni (modelversion, parserversion, stoplistversion, nermodelversion, schemaversion, ecc.), così la stessa mail rielaborata con la stessa versione produce lo stesso output “operativamente deterministico”.
Implementazione: lo store salva eventi/record immutabili (append-only log o tabelle con vincoli) e il promoter crea la nuova versione dizionario solo a fine batch con una transazione atomica e lock (single-writer).

### 3.2 Observation Storage come “source of truth” per monitoring
Ragionamento: senza un dataset di osservazioni persistito, non puoi misurare drift, collisioni, churn, né verificare l’effetto delle nuove keyword nel tempo.
Miglioramento: materializzare viste/aggregati (es. docfreq per lemma/label, collision index cross-label, distribuzioni di confidence) e collegarle a run_id e dictionaryversion.
Esempi di metriche operative: `collisionrate`, `promotionrate`, `churnrate`, percentuale `UNKNOWNTOPIC`, tassi di under-triage per priority, error rate di validazione LLM.

### 3.3 Contratti dati: separare input/output in tre oggetti versionati
Ragionamento: un “envelope blob” unico tende a crescere e a mescolare responsabilità; rende difficili evoluzioni e compatibilità.
Miglioramento: formalizzare 3 contratti versionati e stabili:
- `EntityExtractionOutput`: entità + testo canonico + hash + versioni estrattori.
- `TriageOutput`: decisioni di triage + evidence/spans + signals di priority + keywords selezionate (con `candidateid`).
- `ObservationBatch` (ack): esito di persistenza, conteggi, dedup, errori/warnings, e riferimento a `dictionaryversion_used` (ed eventualmente `dictionaryversion_new`).
Conseguenza: Observation Storage diventa un layer componibile (può essere usato anche se promoter gira separatamente).

### 3.4 Idempotenza: “exactly-once logico” con chiavi naturali
Ragionamento: in produzione reprocess e retry sono inevitabili; lo store deve evitare duplicati logici anche se riceve lo stesso evento più volte.
Miglioramento: usare vincoli UNIQUE + upsert su chiavi naturali. Esempi:
- Keyword observation: `(messageid, labelid, candidateid, dictversion)`; opzionale includere `text_hash` per evitare conflitti se lo stesso messageid viene “mutato”.
- Entity observation: `(messageid, text_hash, type, start, end, source, extractor_version)`; oppure includere `value_hash` se si persiste il valore minimizzato.
Trade-off: più campi nella chiave = più robustezza ma più complessità; la scelta va coerente con le regole di canonicalizzazione (vedi spans).

### 3.5 Coerenza spans/offset: una sola stringa canonica + [start,end)
Ragionamento: evidenze e entità devono essere comparabili/deduplicabili e utilizzabili per UI highlight e training set; se gli offset si riferiscono a testi diversi o convenzioni diverse, tutto si rompe.
Miglioramento: definire `text_canonical` come unica base per offsets e adottare la convenzione `[start, end)` (end escluso), salvando sempre `text_hash` per verifiche.
Regola pratica: `spanllm` e “span corrected server-side” possono restare in diagnostica, ma lo store persistente deve contenere solo spans risolti sulla stringa canonica.

### 3.6 Bug di contratto: source=subject con subject nullo
Ragionamento: se `source` dichiara “subject” ma il subject è null, o c’è un fallback non tracciato o un’etichetta sbagliata; in entrambi i casi l’audit trail diventa inaffidabile.
Miglioramento: rendere `source` un enum coerente con i campi disponibili (`subject`, `body`, `header:<name>`, `attachment:<id>`), e imporre vincoli applicativi: se `source=subject` allora `subject` deve essere non-null e gli spans devono cadere sul testo del subject canonico (o su un campo canonico separato).

### 3.7 NER: costo/rumore e strategie “NER a budget”
Ragionamento: NER può dominare la latenza e introdurre rumore; salvare tali entità nello store senza gate rischia di degradare il promoter.
Miglioramenti:
- Eseguire NER solo quando serve (feature flag per dominio/topic, sampling, oppure batch).
- Dare priorità a RegEx per PII e pattern strutturati (CF, IBAN, importi, email, telefono) perché sono ad alta precisione e spesso più economici.
- Definire regole di merge/priorità tra fonti (`regex > lexicon > ner > llm_ner`) e un quality gate (es. quarantena per entità NER a bassa qualità o ad alta collisione).

### 3.8 Governance del promoter: quarantena + human-in-the-loop
Ragionamento: un promoter che promuove automaticamente tutto “avvelena” il dizionario; serve un circuito deterministico con soglie e revisione umana sui casi ambigui.
Miglioramento: stati per entry (`candidate`, `quarantined`, `active`, `rejected`) e regole deterministiche di promozione basate su docfreq, collision index e segnali (embedding score, total count, stabilità temporale).
Operativo: promuovere automaticamente solo entry con alta evidenza e bassa collisione; il resto va in quarantena con coda di review.

### 3.9 Privacy/compliance: minimizzazione PII nello store
Ragionamento: lo store tende a diventare un “data lake” di testo/PII; se non minimizzi, aumenti rischio GDPR e superficie d’attacco.
Miglioramenti:
- Persistenza minimizzata per PII: `type`, `start/end`, `value_hash` e (se serve) cifratura del valore; retention breve e access control forte.
- Separazione tra store operativo (osservazioni) e store di audit (payload completi) con policy diverse.

## 4. Evidenze dall’esempio message_envelope.json
Di seguito alcuni punti “osservabili” nel JSON di esempio, utili come requisiti e test di coerenza del layer:
- `messageid`: None
- `subject`: None
- `language`: None
- `observations_count`: 0
- `entities_count`: 0
- `processing_time_ms`: None
- `ner_step_ms`: None
- `dictionaryversion`: None
- `schemaversion`: None

Aspetti importanti emersi: presenza di `observations` con `dictversion` e `observedat`; presenza di entità sia `regex` che `ner`; diagnostica con timing step-by-step e warning su “span corrected server-side”.

## 5. Proposta di schema PostgreSQL (outline)
Questa sezione traduce i requisiti in un modello relazionale tipico (indicazioni; da adattare ai nomi effettivi).

Tabelle chiave (minimo):
- `pipeline_runs`: run_id, started_at, finished_at, pipelineversion (jsonb), dictionaryversion_used, schemaversion, status, metrics sintetiche.
- `messages`: messageid, text_hash, text_canonical (opz. fuori DB), metadata email (from/to/date), pii_flags.
- `keyword_observations`: osservazioni keyword per (messageid,labelid,candidateid,dictversion), spans (jsonb), evidence_quote_hash, embeddingscore, observedat.
- `entity_observations`: entità per (messageid,text_hash,type,start,end,source,extractor_version), confidence, value_hash/value_enc (opz.), observedat.
- `lexicon_entries`: entries per label e dicttype con status, surfaceforms, regexpattern, docfreq, first_seen_at/last_seen_at, dictversion_added/deprecated.
- `promotion_events`: cosa è stato promosso/rejected/quarantined, con reason codes, e riferimento a run_id e dictversion_new.

## 6. Appendice A — Contenuto integrale brainstorming v2
> Nota: incluso integralmente come richiesto, per preservare tutte le sezioni e i dettagli operativi.

Enhanced with SOTA best practices, evaluation framework, and
dynamic NER integration

Vuoi prendere una mail (.eml o via IMAP), normalizzarla, produrre:
(a) topics[] multi-label piatto, (b) customer_status (new/old), (c)
priority, (d) sentiment, e soprattutto una lista di keyword presenti nel
testo che vengono auto-inserite nei dizionari per RegEx + NER per le
analisi successive.

L'invariante di determinismo si ottiene con: dizionari versionati
(freeze in-run, update end-of-run), candidati keyword deterministici,
LLM usato come "selettore vincolato" con output JSON validato da
schema strict.

Il determinismo garantito è statistico e a parità di versione, non bit-
per-bit:

Determinismo garantito: a parità di dictionary_version,
model_version, parser_version, stoplist_version, stessa mail
produce stesso output (label, keyword, match RegEx/NER).
Non garantito: stabilità bit-per-bit con LLM hosted (variabilità
intrinseca del modello, infrastruttura).
Obiettivo operativo: assenza di drift silenzioso, ripetibilità degli
esperimenti, audit trail completo.
Thread Classicator Mail -
Brainstorming v2.
0) Obiettivo e invarianti
Precisazione sul determinismo
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class PipelineVersion:
"""Contratto di versione per garantire ripetibilità"""
dictionary_version: int
model_version: str # es. "gpt-4o-2024-11-20"
parser_version: str # es. "email-parser-1.2.0"
stoplist_version: str # es. "stopwords-it-2024.1"
ner_model_version: str # es. "it_core_news_lg-3.7.0"
schema_version: str # es. "json-schema-v2.1"

def __repr__(self):
return f"Pipeline-{self.dictionary_version}-{self.model_version}"
Logging obbligatorio: ogni run salva PipelineVersion completa nei
metadati per audit e backtesting[1][2].

Problema: senza piano di valutazione, impossibile misurare stabilità,
drift, qualità del promoter e collisioni keyword[3][4].

Standard per multi-label:

Micro-F1: media globale su tutte le label, domina performance
su label frequenti
Macro-F1: media non pesata, rileva problemi su label rare
Hamming Loss: frazione di label sbagliate per documento
Label-wise Precision/Recall/F1: per analizzare singole label
deboli[5][6]
Schema di garanzia
0.1) Piano di valutazione e metriche
(NUOVO)
Metriche per classicazione multi-label topics
from sklearn.metrics import hamming_loss, f1_score,
classication_report
import numpy as np

def evaluate_multilabel_topics(y_true, y_pred, label_names):
"""
y_true, y_pred: array [n_samples, n_labels] binari
"""
micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
macro_f1 = f1_score(y_true, y_pred, average='macro',
zero_division=0)
hamming = hamming_loss(y_true, y_pred)

# Per-label breakdown
report = classication_report(
y_true, y_pred,
target_names=label_names,
zero_division=0,
output_dict=True
)
return {
"micro_f1": micro_f1,
"macro_f1": macro_f1,
"hamming_loss": hamming,
"per_label": report
}
Problema specico triage: penalizzare errori gravi (urgent→low)
più degli errori minori (high→medium)[7][8].

from sklearn.metrics import cohen_kappa_score, confusion_matrix
import numpy as np

PRIORITY_ORDER = {"low": 0, "medium": 1, "high": 2, "urgent": 3}

Metriche per priority (ordinale)
def evaluate_priority(y_true, y_pred):
"""Metriche per classicazione ordinale priority"""

# Cohen's Kappa (agreement oltre il caso)
kappa = cohen_kappa_score(y_true, y_pred, weights='linear')
# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=["low","medium","high","urgen
# Weighted accuracy: penalizza distanza nell'ordinamento
y_true_ord = np.array([PRIORITY_ORDER[p] for p in y_true])
y_pred_ord = np.array([PRIORITY_ORDER[p] for p in y_pred])
distances = np.abs(y_true_ord - y_pred_ord)
exact_match = np.mean(distances == 0)
o_by_one = np.mean(distances <= 1)
# Triage error rates (critico per help-desk/clinical)
under_triage = np.mean((y_true_ord - y_pred_ord) >= 2) # predetto troppo ba
over_triage = np.mean((y_pred_ord - y_true_ord) >= 2) # predetto troppo alto
return {
"kappa": kappa,
"exact_match": exact_match,
"o_by_one": o_by_one,
"under_triage_rate": under_triage,
"over_triage_rate": over_triage,
"confusion_matrix": cm
}
Riferimenti: lavori su triage clinico e bug priority mostrano che
Cohen's Kappa e tassi di under/over-triage sono metriche chiave[7][8]
[9].

Standard: accuracy, F1 macro/micro, confusion matrix.

from sklearn.metrics import accuracy_score, f1_score,
confusion_matrix

def evaluate_sentiment(y_true, y_pred):
acc = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average='macro',
zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=
["positive","neutral","negative"])

return {
"accuracy": acc,
"f1_macro": f1_macro,
"confusion_matrix": cm
}
Se deterministico (lookup CRM), verica solo:

Match rate: % email con identicatore trovato in CRM
Condence distribution: distribuzione dei punteggi di
condenza
False new rate: % clienti esistenti classicati come "new" (da
audit manuale)
def evaluate_customer_status(df):
"""df: DataFrame con 'customer_status_value' e
'customer_status_condence'"""
match_rate = (df['customer_status_value'] != 'unknown').mean()

conf_dist = df.groupby('customer_status_value')['customer_status_condence
# Richiede ground truth da audit manuale
if 'customer_status_true' in df.columns:
Metriche per sentiment (3-class)
Metriche per customer_status (deterministico)
false_new = ((df['customer_status_true'] == 'existing') &
(df['customer_status_value'] == 'new')).mean()
else:
false_new = None
return {
"match_rate": match_rate,
"condence_distribution": conf_dist,
"false_new_rate": false_new
}
Problema: dizionari dinamici possono:

crescere senza controllo (keyword noise),
creare collisioni cross-label,
avere drift semantico[10][11].
def evaluate_dictionaries(lexicon_db, observations, version_current,
version_previous=None):
"""Analizza salute dei dizionari per label"""

metrics = {}
# 1. Crescita dizionari
metrics['dict_size_by_label'] = lexicon_db.groupby('label_id').size().to_dict()
metrics['total_active_keywords'] = len(lexicon_db[lexicon_db['status'] == 'activ
metrics['quarantined_keywords'] = len(lexicon_db[lexicon_db['status'] == 'qua
# 2. Collisioni cross-label
collision_count = lexicon_db.groupby('lemma')['label_id'].nunique()
metrics['multi_label_keywords'] = (collision_count > 1).sum()
metrics['collision_rate'] = (collision_count > 1).mean()
# 3. Doc frequency distribution (rileva keyword troppo rare/troppo comuni)
metrics['doc_freq_p25'] = lexicon_db['doc_freq'].quantile(0.25)
metrics['doc_freq_median'] = lexicon_db['doc_freq'].median()
Metriche per keyword/dizionari (stabilità e qualità)
metrics['doc_freq_p75'] = lexicon_db['doc_freq'].quantile(0.75)
# 4. Drift tra versioni
if version_previous:
added = set(lexicon_db[lexicon_db['dict_version_added'] == version_curren
removed = set() # keyword deprecate
metrics['keywords_added'] = len(added)
metrics['keywords_removed'] = len(removed)
metrics['churn_rate'] = (len(added) + len(removed)) / len(lexicon_db)
# 5. Promoter eciency (da observations recenti)
promoted = observations[observations['promoted_to_active'] == True]
metrics['promotion_rate'] = len(promoted) / len(observations)
metrics['avg_doc_freq_at_promotion'] = promoted['doc_freq'].mean()
return metrics
Alert automatici:

collision_rate > 0.15: troppa ambiguità cross-label
churn_rate > 0.3: dizionario troppo instabile tra versioni
promotion_rate < 0.02: soglie troppo restrittive (nessuna
keyword entra)
promotion_rate > 0.5: soglie troppo permissive (rumore)[10][11]
from scipy.stats import chi2_contingency

def detect_label_drift(df_current, df_baseline, alpha=0.05):
"""
Test chi-quadro per rilevare shift nella distribuzione delle label
df: DataFrame con colonna 'topics_assigned' (lista di label per email)
"""
from collections import Counter

# Flatten topics
topics_current = [t for topics in df_current['topics_assigned'] for t in topics]
topics_baseline = [t for topics in df_baseline['topics_assigned'] for t in topics]
Test di stabilità temporale (drift detection)
labels_union = set(topics_current) | set(topics_baseline)
counts_current = Counter(topics_current)
counts_baseline = Counter(topics_baseline)
# Contingency table
table = [
[counts_current.get(label, 0) for label in labels_union],
[counts_baseline.get(label, 0) for label in labels_union]
]
chi2, p_value, dof, expected = chi2_contingency(table)
return {
"chi2_statistic": chi2,
"p_value": p_value,
"drift_detected": p_value < alpha,
"interpretation": "Distribuzione label è cambiata signicativamente" if p_v
}
Uso: esegui ogni 7-14 giorni confrontando batch corrente con
baseline (es. primo mese)[12][13].

Requisiti minimi:

Train/validation: almeno 500-1000 email annotate
manualmente con:
topics (multi-label),
priority ground truth,
sentiment,
customer_status vericato (da CRM).
Test set: 200-300 email straticate per label, priority,
sentiment[3][4].
Processo di annotazione:

Dataset e annotazione
Linee guida chiare con esempi positivi/negativi per ogni label
Doppia annotazione indipendente su subset (100-200 email)
Calcolo inter-annotator agreement (Cohen's Kappa,
Krippendor's alpha)
Risoluzione divergenze e ranamento linee guida
Annotazione completa del dataset[14][15]
Riferimenti: pratiche standard da letteratura NLP e email
classication[3][4][14].

Backtesting (oine): ri-processare email storiche con versioni
diverse di dizionari/modelli per confrontare output.

def backtest_pipeline(emails, pipeline_versions):
"""
Esegue pipeline su stesse email con versioni diverse
"""
results = {}

for version in pipeline_versions:
outputs = []
for email in emails:
output = run_pipeline(email, version=version)
outputs.append(output)
results[str(version)] = {
"outputs": outputs,
"metrics": compute_all_metrics(outputs, ground_truth=None) # se dispo
}
# Confronta stabilità
stability = compare_outputs(results)
return results, stability
def compare_outputs(results):
"""Confronta accordo tra versioni (stabilità)"""

Backtesting e A/B testing
versions = list(results.keys())

if len(versions) < 2:
return None
# Esempio: % email con stessi topics tra v1 e v
v1_topics = [set(o['topics']) for o in results[versions[0]]['outputs']]
v2_topics = [set(o['topics']) for o in results[versions[1]]['outputs']]
agreement = sum(t1 == t2 for t1, t2 in zip(v1_topics, v2_topics)) / len(v1_topics
return {
"topic_agreement": agreement,
"interpretation": f"{agreement*100:.1f}% email con stessi topics"
}
A/B testing (online): in produzione, assegnare random 10-20%
traco a versione sperimentale, confrontare metriche business
(tempo risoluzione ticket, soddisfazione cliente, ecc.)[16].

Una mail conforme a Internet Message Format ha una header section
seguita (opzionalmente) dal body, separati da una riga vuota; gli
header sono righe Field-Name: eld-body e possono essere "foldati"
su più righe, da "unfoldare" prima dell'uso[17][18].

from email import policy
from email.parser import BytesParser

def parse_eml_bytes(raw: bytes):
msg = BytesParser(policy=policy.default).parsebytes(raw)
return msg

def get_basic_headers(msg):
return {
"message_id": msg.get("Message-ID", ""),

1) Ingestion e normalizzazione email
1.1 Parsing .eml (Python)
"date": msg.get("Date", ""),
"from": msg.get("From", ""),
"to": msg.get("To", ""),
"subject": msg.get("Subject", ""),
}

import re

def html_to_text(html: str) -> str:

Minimalista: in produzione usa BeautifulSoup + sanitizer
html = re.sub(r"(?is)<(script|style).?>.?</\1>", " ", html)
text = re.sub(r"(?is)<br\s/?>", "\n", html)
text = re.sub(r"(?is)</p\s>", "\n", text)
text = re.sub(r"(?is)<.*?>", " ", text)
return re.sub(r"[ \t]+", " ", text).strip()

def extract_body_text(msg) -> str:
if msg.is_multipart():
parts = list(msg.walk())
else:
parts = [msg]

text_plain = []
text_html = []
for p in parts:
ctype = (p.get_content_type() or "").lower()
if ctype in ("text/plain", "text/html"):
try:
content = p.get_content()
except Exception:
continue
if not isinstance(content, str):
continue
if ctype == "text/plain":
text_plain.append(content)
else:
1.2 Estrarre testo "canonico" (plain > html)
text_html.append(content)
if text_plain:
return "\n".join(text_plain).strip()
if text_html:
return html_to_text("\n".join(text_html))
return ""
Best practice: applica una pipeline di "strip" sempre uguale (stesse
regex, stesso ordine) e conserva anche il testo originale per
audit[19][20].

Rischio: stripping aggressivo può rimuovere contesto critico per
topic/sentiment. Necessario:

Versioning delle regole di pulizia
(CANONICALIZATION_VERSION)
Switch per audit: ag keep_full_body_for_audit
Logging delle porzioni rimosse con tipo
(quote/rma/disclaimer) e span[19][20]
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class RemovedSection:
"""Traccia cosa è stato rimosso per audit"""
type: str # "quote" | "signature" | "disclaimer" | "reply_header"
span_start: int
span_end: int
content: str

QUOTE_PATTERNS = [
(r"(?m)^\s*>.", "signature"), # rma "standard"
(r"(?is)\nIl giorno .? ha scritto:.", "reply_header"), # reply header

1.3 Pulizia deterministica (quote, rme, disclaimer) -
AGGIORNATO
comune
(r"(?is)\n_{10,}.$", "disclaimer"), # disclaimer lungo
]

CANONICALIZATION_VERSION = "1.2.0" # Versionato per ripetibilità

def canonicalize_text(s: str, keep_audit_info: bool = True) -> Tuple[str,
List[RemovedSection]]:
"""
Canonicalizza testo rimuovendo quote/rme/disclaimer

Returns:
(testo_pulito, sezioni_rimosse)
"""
s = s.replace("\r\n", "\n").replace("\r", "\n")
removed_sections = []
for pat, section_type in QUOTE_PATTERNS:
for match in re.nditer(pat, s):
if keep_audit_info:
removed_sections.append(RemovedSection(
type=section_type,
span_start=match.start(),
span_end=match.end(),
content=match.group(0)
))
s = re.sub(pat, "\n", s)
s = re.sub(r"\n{3,}", "\n\n", s)
s = s.strip()
return s, removed_sections
def test_canonicalization_stability():
"""Assicura che stessi input producano stesso output"""
sample = "Hello\n\n> quoted line\n--\nSignature"

clean1, removed1 = canonicalize_text(sample)
clean2, removed2 = canonicalize_text(sample)
assert clean1 == clean2, "Canonicalizzazione non deterministica!"
assert len(removed1) == len(removed2)
Riferimento: lavori su triage clinico evidenziano che context
stripping errato degrada classicazione[19][20].

from dataclasses import dataclass
from typing import List, Optional

@dataclass(frozen=True)
class EmailDocument:
message_id: str
from_raw: str
subject: str
body: str
body_canonical: str
removed_sections: List[RemovedSection] # NUOVO: per audit
parser_version: str # NUOVO: per tracciabilità
canonicalization_version: str # NUOVO

def build_doc(msg, parser_version: str = "email-parser-1.2.0") ->
EmailDocument:
h = get_basic_headers(msg)
body = extract_body_text(msg)
body_canonical, removed = canonicalize_text(body,
keep_audit_info=True)

Test unit obbligatorio
1.4 Record "EmailDocument"
return EmailDocument(
message_id=h["message_id"] or "",
from_raw=h["from"] or "",
subject=h["subject"] or "",
body=body,
body_canonical=body_canonical,
removed_sections=removed,
parser_version=parser_version,
canonicalization_version=CANONICALIZATION_VERSION
)
Qui costruisci candidate_keywords[] riproducibile: stessi input ⇒
stessa lista e stessi candidate_id.

Regola d'oro: l'LLM non deve "inventare" keyword da inserire in
dizionario; può solo selezionare dai candidati[21][22][23].

import hashlib
import re

def tokenize_it(text: str):

Semplicata: in produzione valuta tokenizer spaCy, ma pinna
versione
text = text.lower()
tokens = re.ndall(r"[a-zàèéìòù]+(?:'[a-zàèéìòù]+)?|\d+", text,
ags=re.I)
return tokens

def ngrams(tokens, n):
return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def stable_id(*parts: str) -> str:
raw = "|".join(parts).encode("utf-8")

2) Candidati keyword deterministici (prima
dell'LLM)
2.1 Tokenizzazione + n-gram (semplice, stabile)
return hashlib.sha1(raw).hexdigest()[:12]

from collections import Counter

def build_candidates(doc: EmailDocument):
subj = doc.subject or ""
body = doc.body_canonical or ""

subj_tokens = tokenize_it(subj)
body_tokens = tokenize_it(body)
items = []
for source, toks in [("subject", subj_tokens), ("body", body_tokens)]:
uni = toks
bi = ngrams(toks, 2)
tri = ngrams(toks, 3)
for term in uni + bi + tri:
if len(term) < 3:
continue
items.append((source, term))
counts = Counter(items)
candidates = []
for (source, term), cnt in counts.items():
cid = stable_id(source, term)
candidates.append({
"candidate_id": cid,
"source": source,
"term": term,
"count": cnt,
# lemma: se non hai lemmatizzazione pronta, inizialmente usa term
"lemma": term,
})
2.2 Costruzione candidati con contatori e "source"
# Ordinamento stabile (fondamentale)
candidates.sort(key=lambda x: (x["source"], -x["count"], x["term"]))
return candidates
Motivazione: n-gram puri possono perdere keyword
semanticamente rilevanti ma sintatticamente variabili. KeyBERT e
sentence embeddings aggiungono segnale semantico[24][25][26].

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import numpy as np

EMBEDDING_MODEL_VERSION = "paraphrase-multilingual-mpnet-
base-v2"
kw_model =
KeyBERT(model=SentenceTransformer(EMBEDDING_MODEL_VERSI
ON))

def enrich_candidates_with_embeddings(doc: EmailDocument,
candidates: List[dict]) -> List[dict]:
"""
Aggiunge score semantico ai candidati usando KeyBERT
"""
full_text = f"{doc.subject}\n{doc.body_canonical}"

if not full_text.strip():
return candidates
# KeyBERT estrae keyword con score cosine similarity
try:
2.3 Integrazione KeyBERT/Embeddings per scoring
candidati - NUOVO
Setup modello (pinna versione
per ripetibilità)
keybert_keywords = kw_model.extract_keywords(
full_text,
keyphrase_ngram_range=(1, 3),
stop_words='italian',
top_n=50,
use_mmr=True, # diversicazione
diversity=0.
)
except Exception as e:
print(f"KeyBERT failed: {e}")
keybert_keywords = []
# Mappa keyword → score
keybert_scores = {kw: score for kw, score in keybert_keywords}
# Arricchisci candidati
for cand in candidates:
term = cand["term"]
lemma = cand["lemma"]
# Match esatto o lemma
score = keybert_scores.get(term, keybert_scores.get(lemma, 0.0))
cand["embedding_score"] = oat(score)
return candidates
def score_candidate_composite(cand: dict, weights: dict = None) ->
oat:
"""
Score composito per ranking candidati

Combina:
count (frequenza nel testo)
embedding_score (rilevanza semantica)
source_bonus (subject pesa di più)
"""
if weights is None:
weights = {
"count": 0.3,
"embedding": 0.5,
"subject_bonus": 0.
}
count_norm = np.log1p(cand["count"]) / 5.0 # normalizzato
embedding_score = cand.get("embedding_score", 0.0)
subject_bonus = 1.0 if cand["source"] == "subject" else 0.
score = (
weights["count"] * count_norm +
weights["embedding"] * embedding_score +
weights["subject_bonus"] * subject_bonus
)
return score
Uso nel promoter: embedding_score diventa feature aggiuntiva per
decidere promozione a active[24][25][26].

Riferimenti:

KeyBERT: keyword extraction basato su embeddings e cosine
similarity[24]
Sentence transformers multilingua per IT[26]
Lavori su lexicon expansion mostrano che scoring semantico
riduce rumore[27][28]
Stopword italiane + blacklist (es. "grazie", "cordiali", "buongiorno",
"saluti").

Token troppo generici (numeri isolati, date senza contesto, "re",
"fw", "ciao").

Whitelist di pattern utili (codici pratica, numeri fattura, IBAN,
P.IVA/CF), ma questi spesso li tratti meglio con RegEx dedicata.

2.4 Filtri "hard" per evitare inquinamento dizionario
STOPLIST_VERSION = "stopwords-it-2024.1"
STOPWORDS_IT = set([
"grazie", "cordiali", "saluti", "buongiorno", "buonasera",
"ciao", "distinti", "gentile", "egregio", "spett",

+ lista completa da stopwords-iso o NLTK[29][30]
])

BLACKLIST_PATTERNS = [
r"^re: ", r"^fwd:", # numeri isolati
]

def lter_candidates(candidates: List[dict]) -> List[dict]:
"""Applica ltri hard deterministici"""
ltered = []

for cand in candidates:
term = cand["term"].lower()
lemma = cand["lemma"].lower()
# Stopword
if lemma in STOPWORDS_IT:
continue
# Blacklist pattern
if any(re.match(pat, term) for pat in BLACKLIST_PATTERNS):
continue
# Troppo corto (già ltrato in build_candidates, ma double-check)
if len(term) < 3:
continue
ltered.append(cand)
return ltered
Stoplist italiana (versione pinna)
Usa Structured Outputs con response_format.type = json_schema,
strict: true e additionalProperties: false per ottenere un JSON stabile e
validabile[31][32][33].

Denisci un registry (DB o le) con label_id stabili e stato:

active: usabili in topics[]
proposed: candidate (non usabili in produzione)
deprecated/merged: storicizzazione
Esempio seed (solo per partire):

TOPICS_ENUM = [
"FATTURAZIONE", "ASSISTENZA_TECNICA", "RECLAMO",
"INFO_COMMERCIALI", "DOCUMENTI", "APPUNTAMENTO",
"CONTRATTO", "GARANZIA", "SPEDIZIONE",
"UNKNOWN_TOPIC" # sempre disponibile
]

Qui l'idea chiave è: label_id ∈ enum, keywords_in_text[].candidate_id
deve riferirsi ai candidati (lo controlli tu lato validazione), e niente
campi extra[31][32][33].

Miglioramento: schema ottimizzato seguendo principi PARSE per
ridurre ambiguità e migliorare aderenza LLM[34].

RESPONSE_SCHEMA = {
"name": "email_triage_v2",
"strict": True,
"schema": {
"type": "object",
"additionalProperties": False,
"required": ["dictionary_version", "topics", "sentiment", "priority",
"customer_status"],

3) LLM vincolato: multi-label + sentiment +
segnali
3.1 Registry label (topics at)
3.2 JSON Schema (estratto pratico) - AGGIORNATO
"properties": {
"dictionary_version": {
"type": "integer",
"description": "Version of keyword dictionary used"
},

"customer_status": {
"type": "object",
"additionalProperties": False,
"required": ["value", "condence"],
"properties": {
"value": {
"type": "string",
"enum": ["new", "existing", "unknown"],
"description": "Customer status: new=rst contact, existing=in CRM, unk
},
"condence": {
"type": "number",
"minimum": 0,
"maximum": 1,
"description": "Condence in status determination (0-1)"
}
},
"description": "Customer status based on CRM lookup, not LLM opinion"
},
"sentiment": {
"type": "object",
"additionalProperties": False,
"required": ["value", "condence"],
"properties": {
"value": {
"type": "string",
"enum": ["positive", "neutral", "negative"],
"description": "Overall email sentiment"
},
"condence": {
"type": "number",
"minimum": 0,
"maximum": 1
}
}
},

"priority": {
"type": "object",
"additionalProperties": False,
"required": ["value", "condence", "signals"],
"properties": {
"value": {
"type": "string",
"enum": ["low", "medium", "high", "urgent"],
"description": "Priority level: urgent=immediate action, high=same day, m
},
"condence": {
"type": "number",
"minimum": 0,
"maximum": 1
},
"signals": {
"type": "array",
"maxItems": 6,
"items": {"type": "string"},
"description": "Phrases/keywords that justify priority (for audit)"
}
}
},

"topics": {
"type": "array",
"maxItems": 5, # RIDOTTO da 8 (PARSE: schemi più semplici = migliore ade
"minItems": 1, # NUOVO: almeno un topic (anche UNKNOWN_TOPIC)
"items": {
"type": "object",
"additionalProperties": False,
"required": ["label_id", "condence", "keywords_in_text", "evidence"],

"properties": {
"label_id": {
"type": "string",
"enum": TOPICS_ENUM,
"description": "Topic label from closed taxonomy"
},
"condence": {
"type": "number",
"minimum": 0,
"maximum": 1
},
"keywords_in_text": {
"type": "array",
"maxItems": 15, # RIDOTTO da 20
"minItems": 1, # NUOVO: almeno 1 keyword per topic
"items": {
"type": "object",
"additionalProperties": False,
"required": ["candidate_id", "lemma", "count"], # spans opzionale
"properties": {
"candidate_id": {
"type": "string",
"description": "MUST match a candidate_id from input. Do not inven
},
"lemma": {"type": "string"},
"count": {"type": "integer", "minimum": 1},
"spans": {
"type": "array",
"items": {
"type": "array",
"minItems": 2,
"maxItems": 2,
"items": {"type": "integer"}
},
"description": "[start, end] positions in text"
}
}
},

"description": "Keywords ONLY from candidate list that support this top
},
"evidence": {
"type": "array",
"maxItems": 2, # RIDOTTO da 3 (semplicazione schema)
"minItems": 1, # NUOVO: almeno 1 evidenza
"items": {
"type": "object",
"additionalProperties": False,
"required": ["quote"], # span opzionale
"properties": {
"quote": {
"type": "string",
"maxLength": 200, # NUOVO: limita lunghezza
"description": "Exact quote from email supporting this topic"
},
"span": {
"type": "array",
"minItems": 2,
"maxItems": 2,
"items": {"type": "integer"}
}
}
},
"description": "Text evidence justifying topic assignment"
} } } } } } }

Miglioramenti rispetto a v1 (seguendo PARSE[34]):

Riduzione limiti: maxItems ridotti per topics (5 invece di 8),
keywords (15 invece di 20), evidence (2 invece di 3) → schema
più semplice = migliore aderenza LLM
Vincoli minimi: minItems: 1 per topics, keywords, evidence →
forza almeno un elemento
Descrizioni chiare: ogni campo ha description esplicita su cosa
ci aspettiamo
Limiti stringhe: maxLength su quote per evitare output troppo
verbosi
Campi obbligatori ridotti: spans opzionale (non sempre
disponibile)
Riferimento: PARSE mostra che ottimizzazione schema può
migliorare extraction accuracy di 10-15%[34].

import json

def build_llm_prompt(doc: EmailDocument, candidates: List[dict],
dictionary_version: int) -> dict:
"""
Costruisce payload per LLM con istruzioni chiare e candidate list
"""

# Top candidati per score composito
candidates_sorted = sorted(
candidates,
key=lambda c: score_candidate_composite(c),
reverse=True
)[:100] # Limita a top 100 per non saturare context
user_payload = {
"dictionary_version": dictionary_version,
"subject": doc.subject,
"from": doc.from_raw,
"body": doc.body_canonical[:8000], # Limita lunghezza
"allowed_topics": TOPICS_ENUM,
"candidate_keywords": [
3.3 Prompting "vincolato" (pattern) - MIGLIORATO
{
"candidate_id": c["candidate_id"],
"term": c["term"],
"lemma": c["lemma"],
"count": c["count"],
"source": c["source"],
"score": round(score_candidate_composite(c), 3)
}
for c in candidates_sorted
],

"instructions": [
"You are a precise email triage classier for Italian customer service.",
"",
"TOPICS (multi-label):",
"- Select 1-5 topic labels from 'allowed_topics' enum",
"- Use UNKNOWN_TOPIC if content is unclear or doesn't t existing labe
"- For each topic, provide evidence quotes and keywords",
"",
"KEYWORDS:",
"- Select keywords ONLY from 'candidate_keywords' list by candidate_id"
"- DO NOT invent new keywords not in the candidate list",
"- Each keyword must have count >= 1 and appear in the text",
"",
"SENTIMENT:",
"- positive: customer expresses satisfaction, gratitude, positive tone",
"- neutral: informational, no emotional charge",
"- negative: complaints, frustration, dissatisfaction, anger",
"",
"PRIORITY:",
"- urgent: immediate action required (SLA breach, blocking issue, legal)"
"- high: same-day response needed",
"- medium: 1-2 days acceptable",
"- low: routine, no time pressure",
"- Provide 'signals' with phrases justifying priority level",
"",
"CUSTOMER_STATUS:",
"- Will be computed deterministically from CRM lookup",

"- You can suggest evidence in text ('ho già un contratto...') but nal deci
"",
"OUTPUT:",
"- Return valid JSON matching the strict schema",
"- All elds required",
"- Condence values between 0-1"
]
}
return user_payload
Miglioramenti:

Istruzioni più strutturate e esplicite per ogni asse
Denizioni operative chiare (es. urgent = SLA breach)
Reminder multipli su "non inventare keyword"
Limiti espliciti (1-5 topics, non di più)[34][35]
import requests
import os
import json

def call_llm_openrouter(doc: EmailDocument, candidates: List[dict],
dictionary_version: int):
api_key = os.environ["OPENROUTER_API_KEY"]
model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o")

user_payload = build_llm_prompt(doc, candidates, dictionary_version)
# System prompt separato
system_prompt = (
"You are an expert email triage system for Italian customer service. "
"You MUST return valid JSON matching the provided schema. "
"Select keywords ONLY from the candidate list provided. "
"Do not invent keywords, labels, or elds not in the schema."
)
3.4 Chiamata OpenRouter (Python, esempio)
req = {
"model": model,
"messages": [
{"role": "system", "content": system_prompt},
{"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
],
"response_format": {
"type": "json_schema",
"json_schema": RESPONSE_SCHEMA
},
"temperature": 0.1, # Bassa temperatura per ridurre variabilità
"stream": False
}
r = requests.post(
"https://openrouter.ai/api/v1/chat/completions",
headers={
"Authorization": f"Bearer {api_key}",
"Content-Type": "application/json"
},
json=req,
timeout=60
)
r.raise_for_status()
content = r.json()["choices"][0]["message"]["content"]
return content # dovrebbe essere JSON conforme
Valida JSON con jsonschema e riuta/ritenta se non valido.

Verica che ogni candidate_id in output esista nella tua lista
candidati (check deterministico lato codice).

Rimuovi topic duplicati e keyword duplicate (dedup stabile).

NUOVO: Guardrail multi-stadio (ispirato a PARSE[34]):

3.5 Validazione e "guardrail" applicativi - ESTESO
from jsonschema import validate, ValidationError
import json

class ValidationResult:
def init(self, valid: bool, errors: List[str], warnings: List[str]):
self.valid = valid
self.errors = errors
self.warnings = warnings

def validate_llm_output_multistage(
output_json: str,
candidates: List[dict],
allowed_topics: List[str]
) -> ValidationResult:
"""
Validazione multi-stadio:

JSON parse
Schema conformance
Business rules (candidate_id exists, topics in enum, etc.)
Quality checks (condence reasonable, evidence non-empty, etc.)
"""
errors = []
warnings = []
# Stage 1: Parse JSON
try:
data = json.loads(output_json)
except json.JSONDecodeError as e:
errors.append(f"Invalid JSON: {e}")
return ValidationResult(False, errors, warnings)
# Stage 2: Schema validation
try:
validate(instance=data, schema=RESPONSE_SCHEMA["schema"])
except ValidationError as e:
errors.append(f"Schema violation: {e.message}")
return ValidationResult(False, errors, warnings)
# Stage 3: Business rules
candidate_ids = {c["candidate_id"] for c in candidates}

for topic in data.get("topics", []):

Check label_id in enum
if topic["label_id"] not in allowed_topics:
errors.append(f"Invalid label_id: {topic['label_id']}")

Check candidate_ids exist
for kw in topic.get("keywords_in_text", []):
cid = kw.get("candidate_id")
if cid not in candidate_ids:
errors.append(f"Invented candidate_id: {cid}")

Stage 4: Quality checks (warnings, not errors)
for topic in data.get("topics", []):
if topic["condence"] < 0.2:
warnings.append(f"Very low condence for {topic['label_id']}: {topic['con

if len(topic.get("keywords_in_text", [])) == 0:
warnings.append(f"No keywords for topic {topic['label_id']}")

if len(topic.get("evidence", [])) == 0:
warnings.append(f"No evidence for topic {topic['label_id']}")

Dedup topics
seen_labels = set()
unique_topics = []
for topic in data.get("topics", []):
if topic["label_id"] not in seen_labels:
unique_topics.append(topic)
seen_labels.add(topic["label_id"])
else:
warnings.append(f"Duplicate topic removed: {topic['label_id']}")

data["topics"] = unique_topics

valid = len(errors) == 0

return ValidationResult(valid, errors, warnings)
def call_llm_with_retry(doc, candidates, dictionary_version,
max_retries=3):
"""Chiama LLM con retry su validation failure"""

for attempt in range(max_retries):
output = call_llm_openrouter(doc, candidates, dictionary_version)
validation = validate_llm_output_multistage(
output,
candidates,
TOPICS_ENUM
)
if validation.valid:
if validation.warnings:
print(f"Warnings: {validation.warnings}")
return json.loads(output)
else:
print(f"Attempt {attempt+1} failed: {validation.errors}")
if attempt == max_retries - 1:
raise ValueError(f"LLM output validation failed after {max_retries} at
return None
Riferimento: PARSE mostra che guardrail multi-stadio riducono
drasticamente errori di extraction[34].

Retry logic con backo
Non delegare "new vs existing" all'LLM: fallo con lookup
CRM/anagraca (email, dominio, P.IVA/CF estratta)[36][37].

L'LLM può aiutare a segnalare evidenze ("ho già un vostro
contratto...") ma la decisione nale deve essere una funzione
ripetibile.

def compute_customer_status(from_email: str, crm_lookup,
text_body: str = "") -> dict:
"""
crm_lookup: funzione deterministica che ritorna match
exact/weak/none
"""
match_type, match_condence = crm_lookup(from_email)

if match_type == "exact":
return {"value": "existing", "condence": 1.0, "source": "crm_exact_match"}
if match_type == "domain":
return {"value": "existing", "condence": 0.7, "source": "crm_domain_match
if match_type == "none":
# Check se nel testo ci sono segnali di cliente esistente
existing_signals = ["ho già un contratto", "cliente dal", "vostro cliente"]
has_signal = any(sig in text_body.lower() for sig in existing_signals)
if has_signal:
return {"value": "existing", "condence": 0.5, "source": "text_signal"}
else:
return {"value": "new", "condence": 0.8, "source": "no_crm_no_signal"}
return {"value": "unknown", "condence": 0.2, "source": "lookup_failed"}
4) Customer status, priority e condence:
calcolo deterministico
4.1 Customer status (esempio)
Esempio: punteggio = urgenza (regex) + sentiment negativo + "cliente
nuovo" + presenza scadenze[38][39].

Estensione: priority come funzione parametrica, con pesi
apprendibili da dati storici[40].

import re
from datetime import datetime, timedelta

URGENT_TERMS = [
"urgente", "bloccante", "dida", "reclamo", "rimborso",
"disdetta", "guasto", "fermo", "critico", "SLA"
]

HIGH_TERMS = [
"problema", "errore", "non funziona", "assistenza", "supporto"
]

def extract_deadline_signals(text: str) -> int:
"""
Cerca menzioni di scadenze imminenti
Returns: urgency boost (0-3)
"""
today = datetime.now()

# Pattern date comuni: "entro il 15/02", "scadenza 2026-02-15"
date_patterns = [
r"entro il (\d{1,2}/\d{1,2})",
r"scadenza:?\s*(\d{4}-\d{2}-\d{2})",
r"entro (\d{1,2}) giorni"
]
for pattern in date_patterns:
matches = re.ndall(pattern, text, re.IGNORECASE)
if matches:
# Analisi semplicata: se menzione scadenza = urgency boost
return 2
4.2 Priority (rule-based score) - MIGLIORATO
return 0
class PriorityScorer:
"""
Priority scorer parametrico
Può essere calibrato su dati storici
"""
def init(self, weights: dict = None):
if weights is None:

Pesi di default (possono essere learned)
self.weights = {
"urgent_terms": 3.0,
"high_terms": 1.5,
"sentiment_negative": 2.0,
"customer_new": 1.0,
"deadline_signal": 2.0,
"vip_customer": 2.5
}
else:
self.weights = weights

def score(self, doc: EmailDocument, sentiment_value: str, customer_value: st
vip_status: bool = False) -> dict:
"""
Calcola priority score e bucket
Returns: {"value": "urgent"|"high"|"medium"|"low", "condence": oat, "si
"""
t = (doc.subject + "\n" + doc.body_canonical).lower()
score = 0.0
signals = []
# Urgent terms
urgent_count = sum(1 for term in URGENT_TERMS if term in t)
if urgent_count > 0:
score += self.weights["urgent_terms"] * urgent_count
signals.append(f"urgent_keywords:{urgent_count}")
High priority terms
high_count = sum(1 for term in HIGH_TERMS if term in t)
if high_count > 0:
score += self.weights["high_terms"] * high_count
signals.append(f"high_keywords:{high_count}")

Sentiment
if sentiment_value == "negative":
score += self.weights["sentiment_negative"]
signals.append("negative_sentiment")

Customer status
if customer_value == "new":
score += self.weights["customer_new"]
signals.append("new_customer")

Deadline
deadline_boost = extract_deadline_signals(t)
if deadline_boost > 0:
score += self.weights["deadline_signal"] * deadline_boost
signals.append("deadline_mentioned")

VIP
if vip_status:
score += self.weights["vip_customer"]
signals.append("vip_customer")

Bucketing
if score >= 7.0:
priority_val = "urgent"
condence = 0.95
elif score >= 4.0:
priority_val = "high"
condence = 0.85
elif score >= 2.0:
priority_val = "medium"
condence = 0.75
else:

priority_val = "low"
condence = 0.70
return {
"value": priority_val,
"condence": condence,
"signals": signals,
"raw_score": score
}
def calibrate_from_data(self, training_data):
"""
Apprendi pesi ottimali da dati storici con logistic regression
training_data: DataFrame con features + priority_true
"""
from sklearn.linear_model import LogisticRegression
# Implementazione semplicata
# In produzione: estrai feature da training_data, t modello, aggiorna weig
pass
priority_scorer = PriorityScorer()

Riferimenti: lavori su multi-target classication per bug priority
mostrano benet di pesi learned[2][9].

Consiglio: usa l'LLM-condence solo come segnale, ma calcola una
condenza applicativa da:

somma score keyword selezionate (TF/TF‑IDF/KeyBERT)
numero evidenze e copertura (keyword con span)
penalità collisione (keyword ambigue già viste in più label)[41]
[42]
Usage
4.3 Condence "vera" per topic (non solo auto-
dichiarata) - NUOVO
import numpy as np

def compute_topic_condence_adjusted(
topic: dict,
candidates: List[dict],
collision_index: dict, # lemma -> set di label_id dove appare
llm_condence: oat
) -> oat:
"""
Calcola condence "vera" per topic combinando:

LLM condence (peso 0.3)
Keyword quality score (peso 0.4)
Evidence coverage (peso 0.2)
Collision penalty (peso 0.1)
"""
# 1. Keyword quality
keywords_in_text = topic.get("keywords_in_text", [])
if not keywords_in_text:
return 0.1 # Condenza bassissima se nessuna keyword
# Mappa candidate_id -> candidate
cand_map = {c["candidate_id"]: c for c in candidates}
keyword_scores = []
for kw in keywords_in_text:
cand = cand_map.get(kw["candidate_id"])
if cand:
# Score composito del candidato
kw_score = score_candidate_composite(cand)
keyword_scores.append(kw_score)
avg_kw_score = np.mean(keyword_scores) if keyword_scores else 0.0
# 2. Evidence coverage
evidence = topic.get("evidence", [])
evidence_score = min(len(evidence) / 2.0, 1.0) # Normalizzato: 2+ evidence =
# 3. Collision penalty
label_id = topic["label_id"]
collision_penalties = []
for kw in keywords_in_text:
cand = cand_map.get(kw["candidate_id"])
if cand:
lemma = cand["lemma"]
labels_with_lemma = collision_index.get(lemma, {label_id})
if len(labels_with_lemma) > 1:
# Keyword ambigua: penalità proporzionale a quante label la usano
penalty = 1.0 / len(labels_with_lemma)
collision_penalties.append(penalty)
else:
collision_penalties.append(1.0) # Nessuna penalità
avg_collision_penalty = np.mean(collision_penalties) if collision_penalties el
# 4. Combine
condence_adjusted = (
0.3 * llm_condence +
0.4 * avg_kw_score +
0.2 * evidence_score +
0.1 * avg_collision_penalty
)
condence_adjusted = np.clip(condence_adjusted, 0.0, 1.0)
return condence_adjusted
def adjust_all_topic_condences(output: dict, candidates: List[dict],
collision_index: dict) -> dict:
"""Ricalcola condence per tutti i topics"""
for topic in output.get("topics", []):
llm_conf = topic["condence"]
adjusted_conf = compute_topic_condence_adjusted(

topic, candidates, collision_index, llm_conf
)
topic["condence_llm"] = llm_conf # Salva originale
topic["condence"] = adjusted_conf # Sovrascrive con adjusted

return output
Riferimento: lavori su multi-label condence calibration mostrano
che segnali compositi battono condence native[43][44].

Aggiorna due dizionari per label: regex_lexicon[label] (precision-rst)
e ner_lexicon[label] (più ampio), con versioning "X → X+1" a ne
batch[45][46][47].

"""
label_registry(
label_id VARCHAR PRIMARY KEY,
name VARCHAR,
status VARCHAR, -- active | proposed | deprecated | merged
created_at TIMESTAMP,
merged_into VARCHAR NULL
)

lexicon_entries(
entry_id VARCHAR PRIMARY KEY,
label_id VARCHAR,
dict_type VARCHAR, -- 'regex' | 'ner'
lemma VARCHAR,
surface_forms JSON, -- ["fattura", "Fattura", "FATTURA"]
regex_pattern VARCHAR NULL,
status VARCHAR, -- active | candidate | quarantined | rejected

5) Auto-aggiornamento dizionari per label
(RegEx + NER) e merge risultati
5.1 Data model minimo (per label)
Tabelle/schema DB
doc_freq INT,
total_count INT,
embedding_score FLOAT NULL, -- NUOVO: score medio KeyBERT
rst_seen_at TIMESTAMP,
last_seen_at TIMESTAMP,
dict_version_added INT,

INDEX(label_id, status),
INDEX(lemma),
UNIQUE(label_id, dict_type, lemma)
)
keyword_observations(
obs_id VARCHAR PRIMARY KEY,
message_id VARCHAR,
label_id VARCHAR,
lemma VARCHAR,
count INT,
embedding_score FLOAT, -- NUOVO
dict_version INT,
promoted_to_active BOOLEAN DEFAULT FALSE,
observed_at TIMESTAMP,

INDEX(label_id, lemma),
INDEX(observed_at)
)
"""
Inserisci in observations tutte le keywords_in_text per ogni topic
assegnato.

Promuovi a regex_lexicon.active solo se: doc_freq(label, lemma) >= 3
e collisione cross-label bassa (es. la stessa lemma non è "alta" anche
su altre label).

5.2 Promoter: regole di promozione (deterministiche) -
ESTESO
Promuovi a ner_lexicon.active con soglie più basse (es. doc_freq>=2),
ma lascia quarantined se ambigua[45][46][47].

NUOVO: Integrazione embedding_score nel promoter:

from collections import defaultdict
from typing import Dict, Set

class KeywordPromoter:
"""
Gestisce promozione keyword nei dizionari con regole
deterministiche
"""
def init(self, cong: dict = None):
if cong is None:
self.cong = {
"regex_min_doc_freq": 3,
"regex_min_embedding_score": 0.35,
"ner_min_doc_freq": 2,
"ner_min_embedding_score": 0.25,
"max_collision_labels": 2, # Keyword in >2 label = quarantena
"min_total_count": 5 # Count cumulativo minimo
}
else:
self.cong = cong

def compute_stats(self, observations: List[dict]) -> Dict[str, Dict[str, dict]]:
"""
Calcola statistiche aggregate per (label_id, lemma)
Returns:
{label_id: {lemma: {"doc_freq": int, "total_count": int, "avg_embedding_sc
"""
stats = defaultdict(lambda: defaultdict(lambda: {
"doc_freq": 0,
"total_count": 0,
"embedding_scores": []
}))
Group by (label, lemma, message_id) per calcolare doc_freq
message_lemma_label = defaultdict(set)

for obs in observations:
label = obs["label_id"]
lemma = obs["lemma"]
msg_id = obs["message_id"]
count = obs["count"]
emb_score = obs.get("embedding_score", 0.0)

Doc freq
message_lemma_label[(label, lemma)].add(msg_id)

Total count
stats[label][lemma]["total_count"] += count

Embedding score
if emb_score > 0:
stats[label][lemma]["embedding_scores"].append(emb_score)

Finalizza
for (label, lemma), msg_ids in message_lemma_label.items():
stats[label][lemma]["doc_freq"] = len(msg_ids)

for label in stats:
for lemma in stats[label]:
scores = stats[label][lemma]["embedding_scores"]
stats[label][lemma]["avg_embedding_score"] = (
sum(scores) / len(scores) if scores else 0.0
)
del stats[label][lemma]["embedding_scores"] # Cleanup

return dict(stats)

def compute_collision_index(self, stats: Dict[str, Dict[str, dict]]) -> Dict[str, Set[
"""
Per ogni lemma, trova tutti i label_id dove appare

Returns:
{lemma: {label_id1, label_id2, ...}}
"""
collision_index = defaultdict(set)

for label_id, lemmas in stats.items():
for lemma in lemmas.keys():
collision_index[lemma].add(label_id)

return dict(collision_index)

def promote_keywords(
self,
observations: List[dict],
existing_lexicon: Dict[str, List[dict]] = None # {label_id: [entries]}
) -> Dict[str, List[dict]]:
"""
Decide promozioni per regex e ner lexicon

Returns:
{
"regex_active": [{label_id, lemma, surface_forms, ...}],
"ner_active": [...],
"quarantined": [...],
"rejected": [...]
}
"""
stats = self.compute_stats(observations)
collision_index = self.compute_collision_index(stats)

updates = {
"regex_active": [],
"ner_active": [],
"quarantined": [],
"rejected": []
}

for label_id, lemmas in stats.items():

for lemma, stat in lemmas.items():
doc_freq = stat["doc_freq"]
total_count = stat["total_count"]
avg_emb_score = stat["avg_embedding_score"]

collision_labels = collision_index.get(lemma, {label_id})
num_collisions = len(collision_labels)

Collect surface forms from observations
surface_forms = list(set([
obs["term"] for obs in observations
if obs["label_id"] == label_id and obs["lemma"] == lemma
]))

entry = {
"label_id": label_id,
"lemma": lemma,
"surface_forms": surface_forms,
"doc_freq": doc_freq,
"total_count": total_count,
"avg_embedding_score": avg_emb_score,
"collision_labels": list(collision_labels)
}

Decision tree
Reject se total count troppo basso (rumore)
if total_count < self.cong["min_total_count"]:
updates["rejected"].append({**entry, "reason": "low_count"})
continue

Quarantena se troppe collisioni
if num_collisions > self.cong["max_collision_labels"]:
updates["quarantined"].append({**entry, "reason": "high_collision"}
continue

Promozione a regex_active (high precision)
if (doc_freq >= self.cong["regex_min_doc_freq"] and

avg_emb_score >= self.cong["regex_min_embedding_score"] and
num_collisions <= 1):

Genera regex pattern
regex_pattern = self._generate_regex_pattern(surface_forms)
updates["regex_active"].append({
**entry,
"dict_type": "regex",
"regex_pattern": regex_pattern,
"status": "active"
})

Promuovi anche in NER (se in regex, anche in NER)
updates["ner_active"].append({
**entry,
"dict_type": "ner",
"status": "active"
})

Promozione a ner_active solo (lower precision ok)
elif (doc_freq >= self.cong["ner_min_doc_freq"] and
avg_emb_score >= self.cong["ner_min_embedding_score"]):

if num_collisions <= 2:
updates["ner_active"].append({
entry,
"dict_type": "ner",
"status": "active"
})
else:
updates["quarantined"].append({entry, "reason": "moderate_col

Altrimenti quarantena (candidato, serve più evidenza)
else:
updates["quarantined"].append({**entry, "reason": "insucient_evi

return updates

def _generate_regex_pattern(self, surface_forms: List[str]) -> str:
"""
Genera regex pattern sicuro da surface forms
Escape caratteri speciali, usa word boundaries
"""
import re
escaped = [re.escape(sf) for sf in surface_forms]
pattern = r"\b(" + "|".join(escaped) + r")\b"
return pattern
Uso:

promoter = KeywordPromoter()

observations = []
for email_output in batch_outputs:
for topic in email_output["topics"]:
label_id = topic["label_id"]
for kw in topic["keywords_in_text"]:
observations.append({
"message_id": email_output["message_id"],
"label_id": label_id,
"lemma": kw["lemma"],
"term": kw["term"], # surface form
"count": kw["count"],
"embedding_score": kw.get("embedding_score", 0.0)
})

Dopo classicazione batch di
email
Raccogli observations da output
LLM
updates = promoter.promote_keywords(observations)

print(f"Promoted to regex: {len(updates['regex_active'])}")
print(f"Promoted to NER: {len(updates['ner_active'])}")
print(f"Quarantined: {len(updates['quarantined'])}")
print(f"Rejected: {len(updates['rejected'])}")

Riferimenti: dynamic lexicon + embedding scoring per NER mostrato
ecace in DyLex, VecNER[46][47][48].

Escape sempre dei caratteri speciali.

Usa boundary sensati per IT (spaziatura/punteggiatura), e mantieni
varianti (singolare/plurale) solo quando le hai osservate come
surface form (o le generi con regole morfologiche controllate).

import re

def generate_safe_regex(surface_forms: List[str], case_sensitive: bool
= False) -> str:
"""
Genera pattern regex sicuro da lista di surface forms
"""

Escape caratteri speciali
escaped = [re.escape(sf) for sf in surface_forms]

# Word boundary per IT (include apostro)
pattern = r"\b(" + "|".join(escaped) + r")\b"
if not case_sensitive:
pattern = "(?i)" + pattern # Case insensitive ag
Promuovi
Applica a DB creando nuova
dictionary_version
5.3 Generazione regex "safe"
return pattern
pattern = generate_safe_regex(["fattura", "Fattura", "FATTURA"],
case_sensitive=False)
regex = re.compile(pattern)

test_text = "Invio fattura n. 12345"
matches = regex.ndall(test_text)
print(matches) # ['fattura']

Approccio robusto e ripetibile:

RegEx matching con dizionario regex_lexicon (alta precisione),
produce entità con span.
NER (es. spaCy IT) sul testo canonicalizzato per estrarre entità
generiche; poi ltri/arricchisci con ner_lexicon[label] come
gazetteer (ad es. validazione "soft").
Merge deterministico: se due entità sovrappongono, preferisci
RegEx (più preciso) oppure "longest span wins", ma la regola
deve essere ssa[49][50][51].
NUOVO: Integrazione NER con dizionari come lexicon layer:

import spacy
from typing import List, Tuple

NER_MODEL_VERSION = "it_core_news_lg-3.7.0"
nlp = spacy.load("it_core_news_lg")

Compile e testa
5.4 NER + RegEx: strategia di combinazione - ESTESO
Load modello NER italiano
(pinna versione)
class Entity:
def init(self, text: str, label: str, start: int, end: int, source: str,
condence: oat = 1.0):
self.text = text
self.label = label
self.start = start
self.end = end
self.source = source # "regex" | "ner" | "lexicon"
self.condence = condence

def __repr__(self):
return f"Entity('{self.text}', {self.label}, [{self.start},{self.end}], {self.source}
def overlaps(self, other: 'Entity') -> bool:
"""Check se due entità si sovrappongono"""
return not (self.end <= other.start or other.end <= self.start)
def extract_entities_regex(text: str, regex_lexicon: dict, label_id: str) ->
List[Entity]:
"""
Estrai entità con RegEx da dizionario per label

regex_lexicon: {label_id: [{"lemma": ..., "regex_pattern": ..., "label": ...}]}
"""
entities = []
entries = regex_lexicon.get(label_id, [])
for entry in entries:
pattern = entry["regex_pattern"]
lemma = entry["lemma"]
try:
regex = re.compile(pattern, re.IGNORECASE)
except Exception as e:
print(f"Invalid regex pattern: {pattern}, error: {e}")
continue
for match in regex.nditer(text):
entities.append(Entity(
text=match.group(0),
label=lemma,
start=match.start(),
end=match.end(),
source="regex",
condence=0.95
))
return entities
def extract_entities_ner(text: str, nlp_model) -> List[Entity]:
"""
Estrai entità con spaCy NER
"""
doc = nlp_model(text)

entities = []
for ent in doc.ents:
entities.append(Entity(
text=ent.text,
label=ent.label_,
start=ent.start_char,
end=ent.end_char,
source="ner",
condence=0.75 # NER generalmente meno preciso di regex
))
return entities
def enhance_ner_with_lexicon(ner_entities: List[Entity], ner_lexicon:
dict, label_id: str, text: str) -> List[Entity]:
"""
Arricchisci entità NER con match da ner_lexicon (gazetteer)

"""
enhanced = list(ner_entities)

entries = ner_lexicon.get(label_id, [])
for entry in entries:
lemma = entry["lemma"]
surface_forms = entry.get("surface_forms", [lemma])
for sf in surface_forms:
# Simple substring search (case insensitive)
lower_text = text.lower()
lower_sf = sf.lower()
start = 0
while True:
pos = lower_text.nd(lower_sf, start)
if pos == -1:
break
# Check word boundary (semplicato)
if (pos == 0 or not lower_text[pos-1].isalnum()) and \
(pos + len(lower_sf) == len(lower_text) or not lower_text[pos + len(lo
enhanced.append(Entity(
text=text[pos:pos+len(sf)],
label=lemma,
start=pos,
end=pos + len(sf),
source="lexicon",
condence=0.85
))
start = pos + 1
return enhanced
def merge_entities_deterministic(entities: List[Entity]) -> List[Entity]:
"""
Merge entità sovrapposte con regole deterministiche:

RegEx > Lexicon > NER (per source)
Se stesso source, longest span wins
Se stessa lunghezza, higher condence wins
"""
if not entities:
return []
# Sort per priorità
source_priority = {"regex": 0, "lexicon": 1, "ner": 2}
entities_sorted = sorted(
entities,
key=lambda e: (e.start, -e.end, source_priority[e.source], -e.condence)
)
merged = []
for entity in entities_sorted:
# Check overlap con entità già in merged
has_overlap = False
for existing in merged:
if entity.overlaps(existing):
has_overlap = True
# Decide quale tenere
# Priorità: source > length > condence
if source_priority[entity.source] < source_priority[existing.source]:
# entity ha priorità più alta, sostituisci
merged.remove(existing)
merged.append(entity)
elif source_priority[entity.source] == source_priority[existing.source]:
# Stesso source, longest wins
entity_len = entity.end - entity.start
existing_len = existing.end - existing.start
if entity_len > existing_len:
merged.remove(existing)
merged.append(entity)
elif entity_len == existing_len and entity.condence > existing.cond
merged.remove(existing)
merged.append(entity)
break
if not has_overlap:
merged.append(entity)
# Sort nale per posizione
merged.sort(key=lambda e: e.start)
return merged
def extract_all_entities(
text: str,
label_id: str,
regex_lexicon: dict,
ner_lexicon: dict,
nlp_model
) -> List[Entity]:
"""
Pipeline completa: RegEx + NER + Lexicon + Merge
"""

1. RegEx (alta precisione)
regex_entities = extract_entities_regex(text, regex_lexicon, label_id)

# 2. NER generico
ner_entities = extract_entities_ner(text, nlp_model)
# 3. Enhance NER con lexicon
enhanced_entities = enhance_ner_with_lexicon(ner_entities, ner_lexicon, lab
Pipeline completa
# 4. Combine
all_entities = regex_entities + enhanced_entities
# 5. Merge deterministico
merged = merge_entities_deterministic(all_entities)
return merged
Riferimenti:

Lexicon-infused NER: migliora recall senza degradare
precision[52][53]
Gazetteer dinamici in spaCy: pattern comune per domain-
specic NER[49][50]
DyLex: integrazione dynamic lexicon in BERT per sequence
labeling, SOTA su più task[46][47]
Scenario: dopo fase iniziale con RegEx+NER+LLM, hai accumulato
dati annotati. Puoi ne-tunare un NER specico per i tuoi
topics/label[54][55][56].

"""
Accumula training data da observations + manual review:
testo annotato con span per ogni label/topic
formato spaCy: (text, {"entities": [(start, end, label)]})
Fine-tune modello base it_core_news_lg:
import spacy
from spacy.training import Example
nlp = spacy.load("it_core_news_lg")
ner = nlp.get_pipe("ner")
5.5 Integrazione modello NER ne-tuned (opzionale,
avanzato) - NUOVO
Conceptual: ne-tuning spaCy
NER su nuovi label
for label in TOPICS_ENUM:
ner.add_label(label)
optimizer = nlp.resume_training()
for epoch in range(20):
for text, annotations in training_data:
doc = nlp.make_doc(text)
example = Example.from_dict(doc, annotations)
nlp.update([example], sgd=optimizer)
nlp.to_disk("./models/ner_custom_v1")
Deploy modello custom al posto di it_core_news_lg base
Ridurre ruolo LLM a:
fallback per out-of-distribution
explanation/justication
gestione UNKNOWN_TOPIC
Re-train periodicamente (es. mensile) con nuovi dati
"""
Beneci:

Latenza ridotta (NER locale più veloce di LLM API)
Costo ridotto (meno chiamate LLM)
Maggiore controllo e ripetibilità
Trade-o:

Richiede infrastruttura ML (training, deploy)
Necessita dataset annotato (500-1000+ esempi)
Manutenzione modello[54][55][56]
Riferimenti:

Incremental NER: gestione nuovi tipi senza catastrophic
forgetting[57][58]
Dynamic NER: formalizzazione del problema con tipi
evolventi[59]
Aggiungi nuove label
Training loop
Transfer learning per NER domain-specic: best practice[60]
[ ] Pinna versioni di: tokenizer/lemmatizzatore, stoplist, modelli
NER, e soprattutto dictionary_version (freeze)[61]
[ ] Ogni run salva PipelineVersion completa nei metadati
[ ] Dictionary freeze in-run: ogni run usa versione X, update solo
a ne batch genera X+1
[ ] Test di stabilità: stessa mail + stessa versione = stesso output
[ ] Logga sempre: input canonicalizzato, sezioni rimosse
(quote/rma), output LLM validato, regole priority applicate,
delta-dizionario generato
[ ] Trace completa per ogni email: message_id → topics assigned
→ keywords used → entities extracted
[ ] Validation errors loggati con retry count
[ ] Collision warnings loggati separatamente per review
[ ] Se gestisci dati personali, minimizza e pseudonimizza (hash di
email/ID) nei log e nei prompt dove possibile
[ ] Body email troncato nei log (max 500 char preview)
[ ] PII detection prima di inviare a LLM esterno (opzionale)
[ ] Retention policy su observations e logs (GDPR compliance)
[ ] Dashboard con metriche real-time:
Distribution topics assigned
Priority distribution
Sentiment distribution
LLM validation error rate
Dictionary size growth rate
6) Checklist best practice (operativa) -
ESTESO
Versioning e determinismo
Logging e audit
Privacy e compliance
Monitoring e alerting
Collision rate
[ ] Alert se:
Validation error rate > 5%
Collision rate > 15%
Dictionary churn > 30% tra versioni
UNKNOWN_TOPIC > 20% delle email
Under-triage rate > 10%
[ ] Weekly drift detection report (chi-quadro test su label
distribution)
[ ] Test set annotato (200-300 email) con ground truth
[ ] Evaluation mensile completa:
Multi-label metrics (micro/macro F1, Hamming loss)
Priority metrics (Kappa, under/over-triage)
Sentiment accuracy
Dictionary quality metrics
[ ] Backtesting prima di deploy nuova versione dizionari
[ ] A/B testing su 10-20% traco per major changes
[ ] Batch processing giornaliero con:
Classicazione email
Observations collection
Promoter execution
Dictionary update (new version)
[ ] Human review queue per:
Email con condence < 0.3
UNKNOWN_TOPIC con condence > 0.6 (possibili nuove
label)
Keyword in quarantine con doc_freq > 10 (potential
promotion)
[ ] Weekly review meeting con sample di misclassications
Evaluation e testing
Operazioni quotidiane
┌─────────────────────────────────────────────
────────────────┐
│ Email Ingestion Layer │
│ (.eml parser, IMAP client, API endpoint) │
└────────────────┬────────────────────────────
────────────────┘
│
v
┌─────────────────────────────────────────────
────────────────┐
│ Preprocessing & Normalization │
│ (RFC5322 parsing, MIME decode, canonicalization) │
└────────────────┬────────────────────────────
────────────────┘
│
v
┌─────────────────────────────────────────────
────────────────┐
│ Candidate Generation (Deterministic) │
│ (n-gram, KeyBERT, header extraction, ltering) │
└────────────────┬────────────────────────────
────────────────┘
│
v
┌─────────────────────────────────────────────
────────────────┐
│ LLM Classication Layer │
│ (OpenRouter/LLM API, structured output, validation) │
└────────────────┬────────────────────────────
────────────────┘
│
v
┌─────────────────────────────────────────────
────────────────┐
│ Post-Processing & Enrichment │

7) Architettura sistema completo - NUOVO
7.1 Componenti
│ (Customer status lookup, priority scoring, condence adj) │
└────────────────┬────────────────────────────
────────────────┘
│
v
┌─────────────────────────────────────────────
────────────────┐
│ Entity Extraction (RegEx + NER) │
│ (Dictionary matching, spaCy NER, lexicon enhancement) │
└────────────────┬────────────────────────────
────────────────┘
│
v
┌─────────────────────────────────────────────
────────────────┐
│ Observation Storage & Promoter │
│ (keyword observations, promotion rules, versioning) │
└────────────────┬────────────────────────────
────────────────┘
│
v
┌─────────────────────────────────────────────
────────────────┐
│ Dictionary Update (Batch, nightly) │
│ (Promoter execution, new version creation, deployment) │
└─────────────────────────────────────────────
────────────────┘

Backend:

Python 3.10+
FastAPI (API server)
PostgreSQL (metadati, dizionari, observations)
Redis (cache, job queue)
Celery (batch processing)
ML/NLP:

spaCy 3.7+ (NER italiano)
7.2 Tech stack proposto
sentence-transformers (embeddings)
KeyBERT (keyword extraction)
OpenRouter / OpenAI API (LLM)
Infra:

Docker + Docker Compose
Prometheus + Grafana (monitoring)
Logstash (centralized logging)
Code structure:
email-triage/
├── src/
│ ├── ingestion/
│ │ ├── parser.py # RFC5322, MIME parsing
│ │ ├── canonicalization.py
│ │ └── document.py
│ ├── candidates/
│ │ ├── tokenizer.py
│ │ ├── ngram.py
│ │ ├── keybert_extractor.py
│ │ └── lters.py
│ ├── classication/
│ │ ├── llm_client.py # OpenRouter integration
│ │ ├── schema.py # JSON Schema denitions
│ │ ├── validation.py
│ │ └── condence.py
│ ├── enrichment/
│ │ ├── customer_status.py
│ │ ├── priority_scorer.py
│ │ └── sentiment.py
│ ├── entity_extraction/
│ │ ├── regex_matcher.py
│ │ ├── ner_extractor.py
│ │ ├── lexicon_enhancer.py
│ │ └── merger.py
│ ├── dictionary/
│ │ ├── models.py # DB models
│ │ ├── promoter.py
│ │ ├── versioning.py

│ │ └── collision_detector.py
│ ├── evaluation/
│ │ ├── metrics.py
│ │ ├── drift_detection.py
│ │ └── backtesting.py
│ └── api/
│ ├── main.py # FastAPI app
│ ├── routes.py
│ └── schemas.py
├── tests/
│ ├── test_parser.py
│ ├── test_candidates.py
│ ├── test_classication.py
│ ├── test_promoter.py
│ └── test_evaluation.py
├── scripts/
│ ├── batch_process.py # Daily batch job
│ ├── dictionary_update.py
│ └── evaluate.py
├── cong/
│ ├── pipeline_cong.yaml
│ ├── promoter_cong.yaml
│ └── model_versions.yaml
├── docker-compose.yml
├── Dockerle
└── README.md

Obiettivo: Pipeline funzionante end-to-end con evaluation baseline

[ ] Setup infra base (Docker, DB, API)
[ ] Parsing & canonicalization
[ ] Candidate generation (n-gram + ltri hard)
[ ] LLM integration con schema v1
[ ] Customer status lookup (mock CRM)
[ ] Priority scorer rule-based
8) Roadmap implementazione suggerita
Fase 1: MVP (4-6 settimane)
[ ] Output storage
[ ] Annotazione 200 email per test set
[ ] Evaluation baseline (F1, accuracy)
Deliverable: Sistema che classica email e produce metriche iniziali

Obiettivo: Auto-espansione controllata dizionari

[ ] Schema DB per dizionari (label_registry, lexicon_entries,
observations)
[ ] Promoter v1 (regole base doc_freq + collision)
[ ] Versioning dizionari
[ ] RegEx + NER pipeline base
[ ] Batch job nightly per update dizionari
[ ] Dashboard dizionari (size, growth, collisions)
Deliverable: Dizionari aggiornati automaticamente, versioning
funzionante

Obiettivo: Embedding, condence adjustments, monitoring

[ ] Integrazione KeyBERT per candidate scoring
[ ] Embedding score in promoter
[ ] Adjusted condence per topics
[ ] Schema optimization (PARSE-inspired)
[ ] Guardrail multi-stadio
[ ] Drift detection automatico
[ ] Alerting system
[ ] Annotazione 500+ email per training set completo
Deliverable: Sistema production-ready con monitoring

Obiettivo: Performance, costo, qualità

[ ] A/B testing infra
[ ] Fine-tuning NER custom (opzionale)
Fase 2: Dictionary Management (3-4 settimane)
Fase 3: Advanced Features (4-6 settimane)
Fase 4: Optimization & Scale (ongoing)
[ ] Priority scorer con pesi learned
[ ] Human-in-the-loop review queue
[ ] Backtesting automatico
[ ] Cost optimization (caching, batch API)
[ ] Latency optimization (<2s per email)
Deliverable: Sistema ottimizzato per produzione ad alto volume

[1] Abbasi, A., et al. (2024). "Ecient multi-target classication for bug
priority and resolution time prediction". Multimedia Tools and
Applications.

[2] Zhou, Y., et al. (2012). "Where should the bugs be xed? More
accurate information retrieval-based bug localization". ICSE.

[3] Aberdeen, J., et al. (2010). "MITRE system for clinical assertion
status classication". Journal of the American Medical Informatics
Association.

[4] Smith, V., et al. (2012). "Multi-domain active learning for text
classication". KDD.

[5] Tsoumakas, G., & Katakis, I. (2007). "Multi-label classication: An
overview". International Journal of Data Warehousing and Mining.

[6] Madjarov, G., et al. (2012). "An extensive experimental comparison
of methods for multi-label learning". Pattern Recognition.

[7] Chen, Y., et al. (2023). "A Disease-Prediction Protocol Integrating
Triage Priority and BERT-Based Transfer Learning". Bioengineering,
10(4), 420.

[8] Cohen, J. (1968). "Weighted kappa: Nominal scale agreement with
provision for scaled disagreement". Psychological Bulletin.

[9] Giger, E., et al. (2010). "Predicting the x time of bugs". MSR.

9) Riferimenti e paper utilizzati
Email Triage & Classication
[10] Gao, J., et al. (2022). "Dynamic lexicon: improving neural dialogue
generation with dynamic lexicon". ICASSP.

[11] Zhang, Y., & Yang, J. (2018). "Chinese NER using lattice LSTM".
ACL.

[12] Kumar, A., et al. (2022). "A Comparison of Machine Learning
Models to Prioritise Emails using Emotion Analysis". Knowledge
Engineering and Data Science, 5(2).

[13] Li, X., et al. (2024). "Large language model triaging of simulated
nephrology patient inbox messages". Frontiers in Articial Intelligence.

[14] Aberdeen, D., et al. (2019). "A Content-Based Approach to Email
Triage Action Prediction". arXiv:1905.01991.

[15] Krippendor, K. (2004). "Reliability in content analysis". Human
Communication Research.

[16] Ibrahim, O., & Yen, D. C. (2014). "Goal-oriented Email Stream
Classier with A Multi-agent System Approach". International Journal
of Advanced Computer Science.

[17] Li, X., et al. (2024). "Large language model triaging of simulated
nephrology patient inbox messages". Frontiers in AI, 2:1452469.

[18] Chen, Y., et al. (2023). "A Disease-Prediction Protocol Integrating
Triage Priority and BERT-Based Transfer Learning for Intelligent
Triage". Bioengineering, 10(4).

[19] Hemphill, L., et al. (2015). "Email stripping and content analysis".
CSCW.

[20] Bird, C., et al. (2011). "The promises and perils of mining email
data". MSR.

[21] Campos, R., et al. (2020). "YAKE! Keyword extraction from single
documents using multiple local features". Information Sciences.

[22] Rose, S., et al. (2010). "Automatic keyword extraction from
individual documents". Text Mining: Applications and Theory.

Keyword Extraction & Embeddings
[23] El-Beltagy, S. R., & Rafea, A. (2009). "KP-Miner: A keyphrase
extraction system for English and Arabic documents". Information
Systems.

[24] Grootendorst, M. (2020). "KeyBERT: Minimal keyword extraction
with BERT". GitHub repository.
https://github.com/MaartenGr/KeyBERT

[25] Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence
Embeddings using Siamese BERT-Networks". EMNLP.

[26] Conneau, A., et al. (2020). "Unsupervised Cross-lingual
Representation Learning at Scale". ACL.

[27] Li, Y., et al. (2023). "Taxonomy Expansion for Named Entity
Recognition". arXiv:2305.13191.

[28] Ju, M., et al. (2021). "GOLD: Improving Out-of-Scope Detection in
Dialogues using Data Augmentation". EMNLP.

[29] Stopwords ISO (2024). "Italian stopwords list". https://github.com/
stopwords-iso/stopwords-it

[30] NLTK (2024). "Natural Language Toolkit - Italian stopwords".
https://www.nltk.org/

[31] OpenAI (2024). "Structured Outputs with JSON Schema". OpenAI
Documentation.

[32] OpenRouter (2024). "Structured Outputs Guide". https://openrout
er.ai/docs/guides/features/structured-outputs

[33] Anthropic (2024). "Tool Use and JSON Mode". Claude API
Documentation.

[34] Li, Y., et al. (2025). "PARSE: LLM Driven Schema Optimization for
Reliable Structured Extraction". arXiv:2510.08623.

[35] Instructor (2024). "Bad Schemas could break your LLM
Structured Outputs". https://python.useinstructor.com/blog/

Dynamic NER & Lexicon Expansion
[36] Salesforce (2024). "CRM Best Practices for Customer Data
Management". Salesforce Research.

[37] HubSpot (2024). "Customer Identication and Deduplication
Strategies". HubSpot Academy.

[38] Atlassian (2024). "How to prioritize customer support tickets".
Atlassian Support Guide.

[39] Zendesk (2024). "Ticket Priority Best Practices". Zendesk Support.

[40] He, H., & Garcia, E. A. (2009). "Learning from imbalanced data".
IEEE TKDE.

[41] Guo, C., et al. (2017). "On calibration of modern neural
networks". ICML.

[42] Platt, J. (1999). "Probabilistic outputs for support vector
machines". Advances in Large Margin Classiers.

[43] Read, J., et al. (2011). "Classier chains for multi-label
classication". Machine Learning, 85(3).

[44] Zhang, M. L., & Zhou, Z. H. (2014). "A review on multi-label
learning algorithms". IEEE TKDE.

[45] Liu, Y., et al. (2021). "CNER: A COVID-19 Named Entity
Recognition System". EMNLP.

[46] Ding, X., et al. (2020). "CNER-DyLex: Incorporating Dynamic
Lexicons into BERT for Sequence Labeling". arXiv.

[47] Wang, X., et al. (2021). "VecNER: A novel method for Chinese
clinical named entity recognition". AMIA.

[48] Yang, H., et al. (2020). "Incorporating lexicon knowledge into
BERT-based Chinese named entity recognition". NLPCC.

[49] spaCy (2024). "Rule-based matching and gazetteers". https://spac
y.io/usage/rule-based-matching

Lexicon & NER Integration
[50] Explosion AI (2024). "Adding a custom NER model with gazetteer
features". spaCy documentation.

[51] Akbik, A., et al. (2018). "Contextual string embeddings for
sequence labeling". COLING.

[52] Passos, A., et al. (2014). "Lexicon Infused Phrase Embeddings for
Named Entity Resolution". CoNLL.

[53] Ratinov, L., & Roth, D. (2009). "Design challenges and
misconceptions in named entity recognition". CoNLL.

[54] Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional
Transformers". NAACL.

[55] Yamada, I., et al. (2020). "LUKE: Deep Contextualized Entity
Representations with Entity-aware Self-attention". EMNLP.

[56] Li, J., et al. (2020). "A Unied MRC Framework for Named Entity
Recognition". ACL.

[57] Monaikul, N., et al. (2021). "Incremental Few-shot Named Entity
Recognition". ACL Findings.

[58] Wang, Z., et al. (2025). "Dynamic Prex as Instructor for
Incremental Named Entity Recognition". ACL Findings.

[59] Wang, Z., et al. (2023). "Dynamic Named Entity Recognition".
arXiv:2302.10314.

[60] Lee, J., et al. (2020). "BioBERT: a pre-trained biomedical language
representation model". Bioinformatics.

[61] Bender, E. M., & Friedman, B. (2018). "Data Statements for
Natural Language Processing". TACL.

## 7. Appendice B — Contenuto integrale brainstorming v3
> Nota: incluso integralmente come richiesto, per preservare tutte le sezioni e le differenze rispetto a v2 (tool calling nativo, reasoning model per priority, LLM-NER ibrido, patch roadmap).


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


## 8. Appendice C — Estratto (raw) message_envelope.json
> Incluso per completezza e per allineare i contratti ai campi effettivi osservati.

```json
{
  "email_context": {
    "message_id": "<abcd1234-5678-90ef-ghij-klmnopqrstuv@example.it>",
    "id_conversazione": "<abcd1234-5678-90ef-ghij-klmnopqrstuv@example.it>",
    "testo_normalizzato": "Gentile team, Volevo confermare che i dati sono corretti: Codice Fiscale: RSSMRA80A01H501U, come discusso. Ho verificato tutti i dettagli e sono d'accordo con i termini proposti: durata 24 mesi, rata mensile di € 450,00. Potete inviarmi il contratto definitivo in formato PDF editabile per apporre la firma digitale? Potete fornirmi un preventivo aggiornato con le nuove tariffe 2024? In allegato trovate il documento firmato e i dati aggiuntivi richiesti (IBAN per domiciliazione, delega amministrativa). Contatto: mario.rossi@example.it, tel. +39 02 98765432. Cordiali saluti, Mario Rossi",
    "mittente": "mario.rossi@example.it",
    "destinatario": "supporto@banca.it",
    "timestamp": "2026-02-24T12:02:09Z",
    "lingua": "it",
    "oggetto": null,
    "allegati": []
  },
  "triage": {
    "topics": [
      {
        "labelid": "CONTRATTO",
        "confidence": 0.75592,
        "keywordsintext": [
          {
            "candidateid": "L4CD0keGl10i4l43",
            "lemma": "contrattare",
            "term": "contratto",
            "count": 1,
            "source": "subject",
            "embeddingscore": 0.449
          },
          {
            "candidateid": "t9lOmUSIi6dXny_-",
            "lemma": "richiedere informazione contratto",
            "term": "richiesta informazioni contratto",
            "count": 1,
            "source": "subject",
            "embeddingscore": 0.4981
          },
          {
            "candidateid": "msyzHoS_Evg6f97C",
            "lemma": "contratto abc",
            "term": "contratto abc",
            "count": 1,
            "source": "subject",
            "embeddingscore": 0.449
          }
        ],
        "evidence": [
          {
            "quote": "Volevo confermare che i dati sono corretti: Codice Fiscale: RSSMRA80A01H501U",
            "span": [
              13,
              89
            ],
            "span_llm": [
              42,
              64
            ],
            "span_status": "exact_match",
            "text_hash": "b65ab80ee3f135041394d6aa56f7b6bca8cafaabb21d1410caaf6466470b4d43"
          },
          {
            "quote": "come discusso. Ho verificato tutti i dettagli e sono d'accordo con i termini proposti: durata 24 mesi",
            "span": [
              90,
              191
            ],
            "span_llm": [
              86,
              126
            ],
            "span_status": "exact_match",
            "text_hash": "b65ab80ee3f135041394d6aa56f7b6bca8cafaabb21d1410caaf6466470b4d43"
          }
        ],
        "confidence_llm": 0.95,
        "confidence_adjusted": 0.75592,
        "keywords": [
          {
            "candidateid": "L4CD0keGl10i4l43",
            "term": "contratto",
            "lemma": "contrattare",
            "count": 1,
            "source": "subject",
            "embeddingscore": 0.449
          },
          {
            "candidateid": "t9lOmUSIi6dXny_-",
            "term": "richiesta informazioni contratto",
            "lemma": "richiedere informazione contratto",
            "count": 1,
            "source": "subject",
            "embeddingscore": 0.4981
          },
          {
            "candidateid": "msyzHoS_Evg6f97C",
            "term": "contratto abc",
            "lemma": "contratto abc",
            "count": 1,
            "source": "subject",
            "embeddingscore": 0.449
          }
        ]
      },
      {
        "labelid": "INFO_COMMERCIALI",
        "confidence": 0.6950599999999999,
        "keywordsintext": [
          {
            "candidateid": "R-DXoOrrXpiqaMc6",
            "lemma": "informazione",
            "term": "informazioni",
            "count": 1,
            "source": "subject",
            "embeddingscore": 0.4235
          },
          {
            "candidateid": "Os2IJycAYHGeHnrN",
            "lemma": "informazione contratto",
            "term": "informazioni contratto",
            "count": 1,
            "source": "subject",
            "embeddingscore": 0.3985
          }
        ],
        "evidence": [
          {
            "quote": "Potete inviarmi il contratto definitivo in formato PDF editabile per apporre la firma digitale?",
            "span": [
              192,
              287
            ],
            "span_llm": [
              153,
              193
            ],
            "span_status": "exact_match",
            "text_hash": "b65ab80ee3f135041394d6aa56f7b6bca8cafaabb21d1410caaf6466470b4d43"
          },
          {
            "quote": "Potete fornirmi un preventivo aggiornato con le nuove tariffe 2024?",
            "span": [
              288,
              355
            ],
            "span_llm": [
              223,
              246
            ],
            "span_status": "exact_match",
            "text_hash": "b65ab80ee3f135041394d6aa56f7b6bca8cafaabb21d1410caaf6466470b4d43"
          }
        ],
        "confidence_llm": 0.8,
        "confidence_adjusted": 0.6950599999999999,
        "keywords": [
          {
            "candidateid": "R-DXoOrrXpiqaMc6",
            "term": "informazioni",
            "lemma": "informazione",
            "count": 1,
            "source": "subject",
            "embeddingscore": 0.4235
          },
          {
            "candidateid": "Os2IJycAYHGeHnrN",
            "term": "informazioni contratto",
            "lemma": "informazione contratto",
            "count": 1,
            "source": "subject",
            "embeddingscore": 0.3985
          }
        ]
      },
      {
        "labelid": "DOCUMENTI",
        "confidence": 0.5476599999999999,
        "keywordsintext": [
          {
            "candidateid": "3nagICgXcL2XHUkS",
            "lemma": "documento",
            "term": "documento",
            "count": 1,
            "source": "body",
            "embeddingscore": 0.4202
          },
          {
            "candidateid": "CL-6JNMnyweVxxT-",
            "lemma": "richiesta",
            "term": "richiesta",
            "count": 1,
            "source": "subject",
            "embeddingscore": 0.4178
          }
        ],
        "evidence": [
          {
            "quote": "In allegato trovate il documento firmato e i dati aggiuntivi richiesti (IBAN per domiciliazione, delega amministrativa).",
            "span": [
              356,
              476
            ],
            "span_llm": [
              252,
              283
            ],
            "span_status": "exact_match",
            "text_hash": "b65ab80ee3f135041394d6aa56f7b6bca8cafaabb21d1410caaf6466470b4d43"
          }
        ],
        "confidence_llm": 0.7,
        "confidence_adjusted": 0.5476599999999999,
        "keywords": [
          {
            "candidateid": "3nagICgXcL2XHUkS",
            "term": "documento",
            "lemma": "documento",
            "count": 1,
            "source": "body",
            "embeddingscore": 0.4202
          },
          {
            "candidateid": "CL-6JNMnyweVxxT-",
            "term": "richiesta",
            "lemma": "richiesta",
            "count": 1,
            "source": "subject",
            "embeddingscore": 0.4178
          }
        ]
      }
    ],
    "sentiment": {
      "value": "neutral",
      "confidence": 0.7
    },
    "priority": {
      "value": "low",
      "confidence": 0.7,
      "signals": [],
      "rawscore": 0.0
    },
    "customerstatus": {
      "value": "existing",
      "confidence": 1.0,
      "source": "crm_exact_match"
    }
  },
  "postprocessing": {
    "message_id": "<abcd1234-5678-90ef-ghij-klmnopqrstuv@example.it>",
    "pipeline_version": {
      "dictionaryversion": 42,
      "modelversion": "gemma3:4b-it-q8_0",
      "model_type": "chat",
      "parserversion": "email-parser-1.3.0",
      "stoplistversion": "stopwords-it-2025.2",
      "nermodelversion": "it_core_news_lg-3.8.2",
      "schemaversion": "json-schema-v3.3",
      "toolcallingversion": "openai-tool-calling-2026"
    },
    "triage": {
      "topics": [
        {
          "labelid": "CONTRATTO",
          "confidence": 0.75592,
          "keywordsintext": [
            {
              "candidateid": "L4CD0keGl10i4l43",
              "lemma": "contrattare",
              "term": "contratto",
              "count": 1,
              "source": "subject",
              "embeddingscore": 0.449
            },
            {
              "candidateid": "t9lOmUSIi6dXny_-",
              "lemma": "richiedere informazione contratto",
              "term": "richiesta informazioni contratto",
              "count": 1,
              "source": "subject",
              "embeddingscore": 0.4981
            },
            {
              "candidateid": "msyzHoS_Evg6f97C",
              "lemma": "contratto abc",
              "term": "contratto abc",
              "count": 1,
              "source": "subject",
              "embeddingscore": 0.449
            }
          ],
          "evidence": [
            {
              "quote": "Volevo confermare che i dati sono corretti: Codice Fiscale: RSSMRA80A01H501U",
              "span": [
                13,
                89
              ],
              "span_llm": [
                42,
                64
              ],
              "span_status": "exact_match",
              "text_hash": "b65ab80ee3f135041394d6aa56f7b6bca8cafaabb21d1410caaf6466470b4d43"
            },
            {
              "quote": "come discusso. Ho verificato tutti i dettagli e sono d'accordo con i termini proposti: durata 24 mesi",
              "span": [
                90,
                191
              ],
              "span_llm": [
                86,
                126
              ],
              "span_status": "exact_match",
              "text_hash": "b65ab80ee3f135041394d6aa56f7b6bca8cafaabb21d1410caaf6466470b4d43"
            }
          ],
          "confidence_llm": 0.95,
          "confidence_adjusted": 0.75592,
          "keywords": [
            {
              "candidateid": "L4CD0keGl10i4l43",
              "term": "contratto",
              "lemma": "contrattare",
              "count": 1,
              "source": "subject",
              "embeddingscore": 0.449
            },
            {
              "candidateid": "t9lOmUSIi6dXny_-",
              "term": "richiesta informazioni contratto",
              "lemma": "richiedere informazione contratto",
              "count": 1,
              "source": "subject",
              "embeddingscore": 0.4981
            },
            {
              "candidateid": "msyzHoS_Evg6f97C",
              "term": "contratto abc",
              "lemma": "contratto abc",
              "count": 1,
              "source": "subject",
              "embeddingscore": 0.449
            }
          ]
        },
        {
          "labelid": "INFO_COMMERCIALI",
          "confidence": 0.6950599999999999,
          "keywordsintext": [
            {
              "candidateid": "R-DXoOrrXpiqaMc6",
              "lemma": "informazione",
              "term": "informazioni",
              "count": 1,
              "source": "subject",
              "embeddingscore": 0.4235
            },
            {
              "candidateid": "Os2IJycAYHGeHnrN",
              "lemma": "informazione contratto",
              "term": "informazioni contratto",
              "count": 1,
              "source": "subject",
              "embeddingscore": 0.3985
            }
          ],
          "evidence": [
            {
              "quote": "Potete inviarmi il contratto definitivo in formato PDF editabile per apporre la firma digitale?",
              "span": [
                192,
                287
              ],
              "span_llm": [
                153,
                193
              ],
              "span_status": "exact_match",
              "text_hash": "b65ab80ee3f135041394d6aa56f7b6bca8cafaabb21d1410caaf6466470b4d43"
            },
            {
              "quote": "Potete fornirmi un preventivo aggiornato con le nuove tariffe 2024?",
              "span": [
                288,
                355
              ],
              "span_llm": [
                223,
                246
              ],
              "span_status": "exact_match",
              "text_hash": "b65ab80ee3f135041394d6aa56f7b6bca8cafaabb21d1410caaf6466470b4d43"
            }
          ],
          "confidence_llm": 0.8,
          "confidence_adjusted": 0.6950599999999999,
          "keywords": [
            {
              "candidateid": "R-DXoOrrXpiqaMc6",
              "term": "informazioni",
              "lemma": "informazione",
              "count": 1,
              "source": "subject",
              "embeddingscore": 0.4235
            },
            {
              "candidateid": "Os2IJycAYHGeHnrN",
              "term": "informazioni contratto",
              "lemma": "informazione contratto",
              "count": 1,
              "source": "subject",
              "embeddingscore": 0.3985
            }
          ]
        },
        {
          "labelid": "DOCUMENTI",
          "confidence": 0.5476599999999999,
          "keywordsintext": [
            {
              "candidateid": "3nagICgXcL2XHUkS",
              "lemma": "documento",
              "term": "documento",
              "count": 1,
              "source": "body",
              "embeddingscore": 0.4202
            },
            {
              "candidateid": "CL-6JNMnyweVxxT-",
              "lemma": "richiesta",
              "term": "richiesta",
              "count": 1,
              "source": "subject",
              "embeddingscore": 0.4178
            }
          ],
          "evidence": [
            {
              "quote": "In allegato trovate il documento firmato e i dati aggiuntivi richiesti (IBAN per domiciliazione, delega amministrativa).",
              "span": [
                356,
                476
              ],
              "span_llm": [
                252,
                283
              ],
              "span_status": "exact_match",
              "text_hash": "b65ab80ee3f135041394d6aa56f7b6bca8cafaabb21d1410caaf6466470b4d43"
            }
          ],
          "confidence_llm": 0.7,
          "confidence_adjusted": 0.5476599999999999,
          "keywords": [
            {
              "candidateid": "3nagICgXcL2XHUkS",
              "term": "documento",
              "lemma": "documento",
              "count": 1,
              "source": "body",
              "embeddingscore": 0.4202
            },
            {
              "candidateid": "CL-6JNMnyweVxxT-",
              "term": "richiesta",
              "lemma": "richiesta",
              "count": 1,
              "source": "subject",
              "embeddingscore": 0.4178
            }
          ]
        }
      ],
      "sentiment": {
        "value": "neutral",
        "confidence": 0.7
      },
      "priority": {
        "value": "low",
        "confidence": 0.7,
        "signals": [],
        "rawscore": 0.0
      },
      "customerstatus": {
        "value": "existing",
        "confidence": 1.0,
        "source": "crm_exact_match"
      }
    },
    "entities": [
      {
        "text": "RSSMRA80A01H501U",
        "label": "CODICEFISCALE",
        "start": 73,
        "end": 89,
        "source": "regex",
        "confidence": 0.95
      }
    ],
    "observations": [
      {
        "obs_id": "09e65b2c-41bd-42e9-8d60-df1ac821e993",
        "message_id": "<abcd1234-5678-90ef-ghij-klmnopqrstuv@example.it>",
        "labelid": "CONTRATTO",
        "candidateid": "L4CD0keGl10i4l43",
        "lemma": "contrattare",
        "term": "contratto",
        "count": 1,
        "embeddingscore": 0.449,
        "dict_version": 42,
        "promoted_to_active": false,
        "observed_at": "2026-02-25T14:29:34.351696+00:00"
      },
      {
        "obs_id": "f28f5d59-be80-4102-a3ab-4369511b9d69",
        "message_id": "<abcd1234-5678-90ef-ghij-klmnopqrstuv@example.it>",
        "labelid": "CONTRATTO",
        "candidateid": "t9lOmUSIi6dXny_-",
        "lemma": "richiedere informazione contratto",
        "term": "richiesta informazioni contratto",
        "count": 1,
        "embeddingscore": 0.4981,
        "dict_version": 42,
        "promoted_to_active": false,
        "observed_at": "2026-02-25T14:29:34.351725+00:00"
      },
      {
        "obs_id": "1926398b-529c-4901-96c7-a496e96acd27",
        "message_id": "<abcd1234-5678-90ef-ghij-klmnopqrstuv@example.it>",
        "labelid": "CONTRATTO",
        "candidateid": "msyzHoS_Evg6f97C",
        "lemma": "contratto abc",
        "term": "contratto abc",
        "count": 1,
        "embeddingscore": 0.449,
        "dict_version": 42,
        "promoted_to_active": false,
        "observed_at": "2026-02-25T14:29:34.351733+00:00"
      },
      {
        "obs_id": "0e2d872b-243e-4953-8286-69c4432abdec",
        "message_id": "<abcd1234-5678-90ef-ghij-klmnopqrstuv@example.it>",
        "labelid": "INFO_COMMERCIALI",
        "candidateid": "R-DXoOrrXpiqaMc6",
        "lemma": "informazione",
        "term": "informazioni",
        "count": 1,
        "embeddingscore": 0.4235,
        "dict_version": 42,
        "promoted_to_active": false,
        "observed_at": "2026-02-25T14:29:34.351740+00:00"
      },
      {
        "obs_id": "da2b613a-1b58-42f6-b247-51399fae0823",
        "message_id": "<abcd1234-5678-90ef-ghij-klmnopqrstuv@example.it>",
        "labelid": "INFO_COMMERCIALI",
        "candidateid": "Os2IJycAYHGeHnrN",
        "lemma": "informazione contratto",
        "term": "informazioni contratto",
        "count": 1,
        "embeddingscore": 0.3985,
        "dict_version": 42,
        "promoted_to_active": false,
        "observed_at": "2026-02-25T14:29:34.351746+00:00"
      },
      {
        "obs_id": "fde5d1f1-8231-4817-8baf-4385d786dfee",
        "message_id": "<abcd1234-5678-90ef-ghij-klmnopqrstuv@example.it>",
        "labelid": "DOCUMENTI",
        "candidateid": "3nagICgXcL2XHUkS",
        "lemma": "documento",
        "term": "documento",
        "count": 1,
        "embeddingscore": 0.4202,
        "dict_version": 42,
        "promoted_to_active": false,
        "observed_at": "2026-02-25T14:29:34.351752+00:00"
      },
      {
        "obs_id": "ccf4961f-3227-45ed-9b4d-75de3cbc12f2",
        "message_id": "<abcd1234-5678-90ef-ghij-klmnopqrstuv@example.it>",
        "labelid": "DOCUMENTI",
        "candidateid": "CL-6JNMnyweVxxT-",
        "lemma": "richiesta",
        "term": "richiesta",
        "count": 1,
        "embeddingscore": 0.4178,
        "dict_version": 42,
        "promoted_to_active": false,
        "observed_at": "2026-02-25T14:29:34.351757+00:00"
      }
    ],
    "diagnostics": {
      "warnings": [
        "labelid alias resolved: 'INFOCOMMERCIALI' → 'INFO_COMMERCIALI'",
        "LLM span corrected server-side: 5 span(s) recomputed from quote"
      ],
      "errors": [],
      "validation_retries": 0,
      "fallback_applied": false
    },
    "processing_metadata": {
      "postprocessing_duration_ms": 11,
      "entities_extracted": 1,
      "observations_created": 7,
      "confidence_adjustments_applied": 3,
      "span_exact_match_count": 5,
      "span_fuzzy_match_count": 0,
      "span_not_found_count": 0
    }
  },
  "ner_entities": {
    "entities": [
      {
        "type": "MISC",
        "value": "Volevo",
        "span": {
          "start": 14,
          "end": 20
        },
        "confidence": 0.75,
        "source": "ner",
        "version": "it_core_news_lg"
      },
      {
        "type": "MISC",
        "value": "Codice Fiscale",
        "span": {
          "start": 58,
          "end": 72
        },
        "confidence": 0.75,
        "source": "ner",
        "version": "it_core_news_lg"
      },
      {
        "type": "CODICEFISCALE",
        "value": "RSSMRA80A01H501U",
        "span": {
          "start": 74,
          "end": 90
        },
        "confidence": 0.95,
        "source": "regex",
        "version": "regex-v1.0"
      },
      {
        "type": "IMPORTO",
        "value": "450.00",
        "span": {
          "start": 211,
          "end": 219
        },
        "confidence": 0.95,
        "source": "regex",
        "version": "regex-v1.0"
      },
      {
        "type": "MISC",
        "value": "PDF",
        "span": {
          "start": 272,
          "end": 275
        },
        "confidence": 0.75,
        "source": "ner",
        "version": "it_core_news_lg"
      },
      {
        "type": "PER",
        "value": "Contatto",
        "span": {
          "start": 506,
          "end": 514
        },
        "confidence": 0.75,
        "source": "ner",
        "version": "it_core_news_lg"
      },
      {
        "type": "EMAIL",
        "value": "mario.rossi@example.it",
        "span": {
          "start": 516,
          "end": 538
        },
        "confidence": 0.95,
        "source": "regex",
        "version": "regex-v1.0"
      },
      {
        "type": "TELEFONO",
        "value": "+39 02 98765432",
        "span": {
          "start": 545,
          "end": 560
        },
        "confidence": 0.95,
        "source": "regex",
        "version": "regex-v1.0"
      },
      {
        "type": "PER",
        "value": "Mario Rossi",
        "span": {
          "start": 579,
          "end": 590
        },
        "confidence": 0.75,
        "source": "ner",
        "version": "it_core_news_lg"
      }
    ],
    "meta": {
      "id_conversazione": "<abcd1234-5678-90ef-ghij-klmnopqrstuv@example.it>",
      "id_messaggio": "<abcd1234-5678-90ef-ghij-klmnopqrstuv@example.it>",
      "status": "ok",
      "layer_version": "1.0.0",
      "processing_time_ms": 5213.505,
      "component_timings_ms": {
        "step1_validation": 0.166,
        "step2_normalisation": 0.132,
        "step3_regex": 2.155,
        "step4_ner": 5210.473,
        "step5_lexicon": 0.014,
        "step6_merge": 0.035,
        "step7_filters": 0.021
      },
      "feature_flags": {
        "engine_regex": true,
        "engine_ner": true,
        "engine_lexicon": true
      },
      "fallbacks": [],
      "entity_count": 9
    },
    "errors": []
  }
}
```

## 9. Appendice D — Test e checklist consigliati
- Test di idempotenza: re-run stessa mail (stessa text_hash e dictionaryversion) non deve cambiare conteggi né creare duplicati.
- Test di coerenza spans: ogni span deve ricadere in [0, len(text_canonical)] e la substring deve matchare la quote (o essere marcata fuzzy con ragione).
- Test di versioning: ogni record di observation deve avere pipelineversion completa, e i vincoli NOT NULL devono impedire scritture “senza versioni”.
- Test performance: budget per step (regex, ner, merge) e fallback se NER supera soglia (es. disabilitazione dinamica o deferred extraction).
- Test privacy: PII non deve essere loggata in chiaro fuori dai percorsi autorizzati; hashing deterministico e encryption at rest verificati.