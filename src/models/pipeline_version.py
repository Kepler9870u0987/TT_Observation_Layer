"""PipelineVersion — contratto di versione immutabile per garantire ripetibilità deterministica."""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PipelineVersion:
    """
    Contratto di versione v3.
    Ogni run salva l'istanza completa nei metadati (pipeline_runs.pipeline_version).
    Stessa mail + stessa PipelineVersion → stesso output (determinismo statistico).
    """
    dictionary_version: int
    model_version: str          # es. "gpt-4o-2025-11-20"
    model_type: Literal["chat", "reasoning"]   # chat=topics/sentiment, reasoning=priority
    parser_version: str         # es. "email-parser-1.3.0"
    stoplist_version: str       # es. "stopwords-it-2025.2"
    ner_model_version: str      # es. "it_core_news_lg-3.8.2"
    schema_version: str         # es. "json-schema-v3.3"
    tool_calling_version: str   # es. "openai-tool-calling-2026"

    def to_dict(self) -> dict:
        return {
            "dictionaryversion": self.dictionary_version,
            "modelversion": self.model_version,
            "model_type": self.model_type,
            "parserversion": self.parser_version,
            "stoplistversion": self.stoplist_version,
            "nermodelversion": self.ner_model_version,
            "schemaversion": self.schema_version,
            "toolcallingversion": self.tool_calling_version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineVersion":
        return cls(
            dictionary_version=d["dictionaryversion"],
            model_version=d["modelversion"],
            model_type=d.get("model_type", "chat"),
            parser_version=d["parserversion"],
            stoplist_version=d["stoplistversion"],
            ner_model_version=d["nermodelversion"],
            schema_version=d["schemaversion"],
            tool_calling_version=d.get("toolcallingversion", ""),
        )

    def __repr__(self) -> str:
        return (
            f"Pipeline-dict{self.dictionary_version}"
            f"-{self.model_version}"
            f"-{self.model_type}"
        )
