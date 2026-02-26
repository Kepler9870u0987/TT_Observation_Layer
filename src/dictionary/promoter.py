"""
KeywordPromoter — sezione 5.2 del documento.

Decisioni deterministiche di promozione per regex e NER lexicon,
basate su doc_freq, embedding_score e collision index.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import yaml


def _load_config(path: str = "config/promoter_config.yaml") -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("promoter", {})
    except FileNotFoundError:
        return {}


class KeywordPromoter:
    DEFAULT_CONFIG = {
        "regex_min_doc_freq": 3,
        "regex_min_embedding_score": 0.35,
        "regex_min_total_count": 5,
        "regex_max_collision_labels": 1,
        "ner_min_doc_freq": 2,
        "ner_min_embedding_score": 0.25,
        "ner_min_total_count": 3,
        "ner_max_collision_labels": 2,
        "max_collision_labels": 2,
        "min_total_count": 3,
    }

    def __init__(self, config: dict | None = None, config_path: str = "config/promoter_config.yaml"):
        base = dict(self.DEFAULT_CONFIG)
        base.update(_load_config(config_path))
        if config:
            base.update(config)
        self.config = base

    # ─────────────────────────────────────────────────────────────────────
    # compute_stats
    # ─────────────────────────────────────────────────────────────────────

    def compute_stats(
        self, observations: list[dict]
    ) -> dict[str, dict[str, dict]]:
        """
        Aggrega observations per (label_id, lemma).
        Ritorna {label_id: {lemma: {doc_freq, total_count, avg_embedding_score, surface_forms}}}.
        """
        stats: dict[str, dict[str, dict]] = defaultdict(
            lambda: defaultdict(lambda: {
                "doc_freq": 0,
                "total_count": 0,
                "msg_ids": set(),
                "embedding_scores": [],
                "surface_forms": set(),
            })
        )
        for obs in observations:
            label = obs["label_id"]
            lemma = obs["lemma"]
            s = stats[label][lemma]
            s["msg_ids"].add(obs["message_id"])
            s["total_count"] += obs.get("count", 1)
            score = obs.get("embedding_score")
            if score is not None and score > 0:
                s["embedding_scores"].append(float(score))
            term = obs.get("term", lemma)
            if term:
                s["surface_forms"].add(term)

        # Finalise
        result: dict[str, dict[str, dict]] = {}
        for label, lemmas in stats.items():
            result[label] = {}
            for lemma, s in lemmas.items():
                scores = s["embedding_scores"]
                result[label][lemma] = {
                    "doc_freq": len(s["msg_ids"]),
                    "total_count": s["total_count"],
                    "avg_embedding_score": sum(scores) / len(scores) if scores else 0.0,
                    "surface_forms": list(s["surface_forms"]),
                }
        return result

    # ─────────────────────────────────────────────────────────────────────
    # compute_collision_index
    # ─────────────────────────────────────────────────────────────────────

    def compute_collision_index(
        self, stats: dict[str, dict[str, dict]]
    ) -> dict[str, set[str]]:
        """Ritorna {lemma: {label_id1, label_id2, …}}."""
        index: dict[str, set[str]] = defaultdict(set)
        for label_id, lemmas in stats.items():
            for lemma in lemmas:
                index[lemma].add(label_id)
        return dict(index)

    # ─────────────────────────────────────────────────────────────────────
    # promote_keywords — decision tree
    # ─────────────────────────────────────────────────────────────────────

    def promote_keywords(
        self,
        observations: list[dict],
        existing_lexicon: dict[str, list[dict]] | None = None,
    ) -> dict[str, list[dict]]:
        """
        Decide promozioni.
        Ritorna {"regex_active": [...], "ner_active": [...], "quarantined": [...], "rejected": [...]}.
        """
        cfg = self.config
        stats = self.compute_stats(observations)
        collision_index = self.compute_collision_index(stats)

        updates: dict[str, list[dict]] = {
            "regex_active": [],
            "ner_active": [],
            "quarantined": [],
            "rejected": [],
        }

        for label_id, lemmas in stats.items():
            for lemma, stat in lemmas.items():
                doc_freq = stat["doc_freq"]
                total_count = stat["total_count"]
                avg_emb = stat["avg_embedding_score"]
                surface_forms = stat["surface_forms"]
                collision_labels = collision_index.get(lemma, {label_id})
                num_collisions = len(collision_labels)

                entry: dict[str, Any] = {
                    "label_id": label_id,
                    "lemma": lemma,
                    "surface_forms": surface_forms,
                    "doc_freq": doc_freq,
                    "total_count": total_count,
                    "avg_embedding_score": avg_emb,
                    "collision_labels": list(collision_labels),
                }

                # ── Reject: rumore puro ────────────────────────────────────
                if total_count < cfg["min_total_count"]:
                    updates["rejected"].append({**entry, "reason": "low_count"})
                    continue

                # ── Quarantena: troppe collisioni ─────────────────────────
                if num_collisions > cfg["max_collision_labels"]:
                    updates["quarantined"].append({**entry, "reason": "high_collision"})
                    continue

                # ── Promozione regex (alta precisione) ────────────────────
                if (
                    doc_freq >= cfg["regex_min_doc_freq"]
                    and avg_emb >= cfg["regex_min_embedding_score"]
                    and total_count >= cfg["regex_min_total_count"]
                    and num_collisions <= cfg["regex_max_collision_labels"]
                ):
                    regex_pattern = self._generate_regex_pattern(surface_forms)
                    updates["regex_active"].append({
                        **entry,
                        "dict_type": "regex",
                        "regex_pattern": regex_pattern,
                        "status": "active",
                    })
                    # Se promosso a regex → promosso anche a NER
                    updates["ner_active"].append({
                        **entry,
                        "dict_type": "ner",
                        "status": "active",
                    })
                    continue

                # ── Promozione NER (precisione minore OK) ─────────────────
                if (
                    doc_freq >= cfg["ner_min_doc_freq"]
                    and avg_emb >= cfg["ner_min_embedding_score"]
                    and total_count >= cfg["ner_min_total_count"]
                ):
                    if num_collisions <= cfg["ner_max_collision_labels"]:
                        updates["ner_active"].append({
                            **entry,
                            "dict_type": "ner",
                            "status": "active",
                        })
                    else:
                        updates["quarantined"].append({**entry, "reason": "moderate_collision"})
                    continue

                # ── Candidato: non abbastanza evidenza ───────────────────
                updates["quarantined"].append({**entry, "reason": "insufficient_evidence"})

        return updates

    # ─────────────────────────────────────────────────────────────────────
    # Regex pattern generation
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _generate_regex_pattern(surface_forms: list[str]) -> str:
        """Genera regex safe con word boundary da surface forms."""
        if not surface_forms:
            return ""
        escaped = [re.escape(sf) for sf in sorted(set(surface_forms))]
        return r"(?i)\b(" + "|".join(escaped) + r")\b"
