# src/ingestion/__init__.py
from .parser import parse_eml_bytes, parse_eml_file
from .preprocessor import canonicalize_text, build_email_document

__all__ = ["parse_eml_bytes", "parse_eml_file", "canonicalize_text", "build_email_document"]
