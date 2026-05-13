"""
Text builders for the classification pipeline.

Two separate text representations optimized for different purposes:
  - Retrieval text: compact, field-tagged, for embedding + BM25 matching
  - LLM context:   rich, complete, for the classification prompt

Which fields feed into each is controlled by config.py
(RETRIEVAL_FIELDS, LLM_CONTEXT_FIELDS, LLM_RAW_CONTEXT_COL).

Usage:
    from src.classification.text_builder import build_retrieval_text, build_llm_context

    df["retrieval_text"] = df.apply(build_retrieval_text, axis=1)
    df["llm_context"]    = df.apply(build_llm_context, axis=1)
"""
import pandas as pd
from config import RETRIEVAL_FIELDS, RETRIEVAL_SEPARATOR, LLM_CONTEXT_FIELDS, LLM_RAW_CONTEXT_COL


def build_retrieval_text(row):
    """
    Build compact text for retrieval (embedding + BM25).

    Concatenates extracted fields with prefix tags and [SEP] markers.
    Skips any field that is null or empty.

    Returns None if no fields have content.
    """
    parts = []
    for col, prefix in RETRIEVAL_FIELDS:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            text = str(val).strip()
            parts.append(f"{prefix}: {text}" if prefix else text)

    return RETRIEVAL_SEPARATOR.join(parts) if parts else None


#def build_llm_context(row):
    """
    Build rich context text for the LLM classification prompt.

    Prefers raw remote_remarks_en for maximum completeness —
    the LLM can handle boilerplate and extract what matters.
    Falls back to structured extracted fields if raw text is missing.

    Returns None if no text is available.
    """
    # prefer raw remarks (>20 chars to skip near-empty entries)
    raw = row.get(LLM_RAW_CONTEXT_COL)
    if pd.notna(raw) and len(str(raw).strip()) > 20:
        return str(raw).strip()

    # fallback: structured extracted fields
    parts = []
    for label, col in LLM_CONTEXT_FIELDS:
        val = row.get(col)
        if pd.notna(val) and str(val).strip():
            parts.append(f"{label}: {str(val).strip()}")

    return "\n".join(parts) if parts else None

def build_llm_context(row):
    """ 
    Build rich context text for the LLM classification prompt.
    Add also structured input to explicit signal presentation 
    to improve fine-grained classification decisions.
    """
    raw = str(row.get(LLM_RAW_CONTEXT_COL, "") or "").strip()

    if len(raw) > 20:
        structured = "\n".join(
            f"{name}: {str(row.get(col,'') or '')[:300]}"
            for name, col in LLM_CONTEXT_FIELDS if row.get(col)
        )
        return f"{structured}\n\n{raw}" if structured else raw

    parts = [
        f"{name}: {str(row.get(col,'') or '')}"
        for name, col in LLM_CONTEXT_FIELDS if row.get(col)
    ]
    return "\n".join(parts) if parts else None