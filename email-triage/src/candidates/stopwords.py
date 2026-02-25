"""
Italian stopword list + blacklist patterns.
Version: stopwords-it-2025.1
"""
STOPWORDS_IT: set[str] = {
    # generic greetings / closings
    "grazie", "cordiali", "saluti", "buongiorno", "buonasera",
    "ciao", "distinti", "gentile", "egregio", "spett",
    "prego", "arrivederci", "buona", "giornata", "sera",
    "buona giornata", "buona sera",
    # articles
    "il", "lo", "la", "gli", "le", "un", "uno", "una",
    # prepositions / conjunctions
    "di", "da", "in", "con", "su", "per", "tra", "fra",
    "che", "non", "del", "della", "dei", "degli", "dal",
    "nel", "nella", "nei", "agli", "alle", "all",
    # common verbs (high-frequency, low-signal)
    "sono", "sei", "ha", "ho", "avere", "essere",
    "fare", "stare", "volere", "potere", "dovere",
    "può", "può", "può", "puoi", "vuole", "vuoi",
    # common function words
    "anche", "come", "quando", "dove", "cosa",
    "questo", "questa", "questi", "queste",
    "quello", "quella", "quelli", "quelle",
    "suo", "sua", "suoi", "sue", "mio", "mia",
    # email threading tokens
    "re", "fw", "fwd", "rispondi", "risposta",
    # months / weekdays (low signal)
    "gennaio", "febbraio", "marzo", "aprile", "maggio",
    "giugno", "luglio", "agosto", "settembre", "ottobre",
    "novembre", "dicembre",
    "lunedì", "martedì", "mercoledì", "giovedì", "venerdì",
    "sabato", "domenica",
}

# Regex patterns — if any match, the candidate is rejected
BLACKLIST_PATTERNS: list[str] = [
    r"^re:\s",
    r"^fwd?:\s",
    r"^\d{1,2}$",           # isolated 1-2 digit numbers
    r"^[a-z]{1,2}$",        # single/double chars
    r"^[\W_]+$",            # only punctuation / special chars
]
