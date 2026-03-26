"""
Text Preparation for Topic Modeling.

Sits between the existing extraction pipeline (text.py) and BERTopic.
Input:  cfr_savings_processed.parquet (with extracted_* columns)
Output: clean document lists ready for embedding

Two stages:
  A) clean_text()         — cleans ONE extracted field
  B) prepare_documents()  — combines cleaned fields into one doc per case
"""
import re
import pandas as pd
import ftfy


# ── Boilerplate phrases from service form templates ───────────────────
# These are field labels that leak through the text extraction.
# They add noise to embeddings and must be stripped BEFORE embedding.
# Add more as you spot them in topic_terms.csv.

BOILERPLATE_PHRASES = [
    "Problem description by engineer",
    "Problem description",
    "Information to support the complaint handling process",
    "Customer Function/Role",
    "How was the device being used",
    "Expected and actual behavior of the product",
    "User Impact",
    "Patient Impact",
    "Current Software Version",
    "Malfunction area",
    "Error # and/or description of error",
    "Troubleshooting Action",
    "Repair Action",
    "occurrence malfunction",
    "complaint handling",
    "process customer",
    "support complaint",
    "diagnostic expected",
    "diagnostics expected",
    "device used",
    "customer function",
    "patient involved",
    "information support",
    "provided device",
    "provided patient",
    "failure issue",
    "issue occurring",
    "problem confirmed",
    "description engineer",
]


def strip_boilerplate(text):
    """Remove form-template phrases that add noise to embeddings."""
    for phrase in BOILERPLATE_PHRASES:
        text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)
    return text


# ── Stage A: clean a single text field ────────────────────────────────

def clean_text(text):
    """
    Minimal cleaning for one extracted field.

    Does:
      - Fix encoding artifacts (ftfy)
      - Remove URLs and emails
      - Strip leftover [SEP] markers from extraction phase
      - Strip form-template boilerplate phrases
      - Collapse whitespace

    Does NOT (by design — BERTopic best practices):
      - Stem or lemmatize    → embeddings handle synonymy
      - Lowercase            → "MR" vs "mr" matters in medical text
      - Remove stopwords     → CountVectorizer handles this in Stage 4
    """
    if pd.isna(text) or not isinstance(text, str):
        return None

    text = ftfy.fix_text(text)
    text = re.sub(r"https?://\S+", "", text)        # URLs
    text = re.sub(r"\S+@\S+\.\S+", "", text)        # emails
    text = text.replace("[SEP]", "")                 # old separators
    text = strip_boilerplate(text)                   # form labels
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else None


# ── Stage B: combine fields into one document ─────────────────────────

def combine_fields(row, columns, separator):
    """
    Join multiple extracted columns into a single document string.

    Skips null/empty fields so the separator only appears between
    real content.
    """
    parts = []
    for col in columns:
        cleaned = clean_text(row.get(col))
        if cleaned:
            parts.append(cleaned)

    return separator.join(parts) if parts else None


def get_first_valid(row, columns):
    """Return the first non-null cleaned value from a list of columns."""
    for col in columns:
        cleaned = clean_text(row.get(col))
        if cleaned:
            return cleaned
    return None


def build_problem_doc(row, prefix_columns, text_columns, separator):
    """
    Build a problem document with malfunction area as a strong prefix.

    Instead of treating malfunction_area equally with problem_description,
    we prepend it as a structured label. This gives the embedding model
    a strong categorical signal that pulls similar problems together.

    Examples:
        prefix="Chiller", text="LCC error overnight, cooling issue"
        → "MALFUNCTION: Chiller. LCC error overnight, cooling issue"

        prefix=None, text="LCC error overnight"
        → "LCC error overnight"   (no prefix if malfunction area is empty)

        prefix="Gradient", text=None
        → "MALFUNCTION: Gradient"  (prefix alone is still useful)
    """
    prefix = get_first_valid(row, prefix_columns)
    body = combine_fields(row, text_columns, separator)

    if prefix and body:
        return f"MALFUNCTION: {prefix}. {body}"
    elif prefix:
        return f"MALFUNCTION: {prefix}"
    elif body:
        return body
    return None


def prepare_problem_documents(df, cfg):
    """
    Prepare documents for the Problem topic model.

    Uses malfunction area as a prefix for stronger clustering signal.

    Returns DataFrame with doc_index and doc_text.
    """
    docs = df.apply(
        lambda row: build_problem_doc(
            row,
            cfg.text_prep.problem_prefix_columns,
            cfg.text_prep.problem_text_columns,
            cfg.text_prep.separator,
        ),
        axis=1,
    )

    result = pd.DataFrame({"doc_index": df.index, "doc_text": docs})
    result = result.dropna(subset=["doc_text"])
    mask = (
        (result["doc_text"].str.len() >= cfg.text_prep.min_doc_length) &
        (result["doc_text"].str.split().str.len() >= cfg.text_prep.min_word_count)
    )
    return result[mask].reset_index(drop=True)


def prepare_documents(df, columns, separator,
                      min_length=10, min_words=3):
    """
    Prepare documents for the Resolution topic model (no prefix).

    Returns DataFrame with doc_index and doc_text.
    """
    docs = df.apply(lambda row: combine_fields(row, columns, separator), axis=1)

    result = pd.DataFrame({"doc_index": df.index, "doc_text": docs})
    result = result.dropna(subset=["doc_text"])
    mask = (
        (result["doc_text"].str.len() >= min_length) &
        (result["doc_text"].str.split().str.len() >= min_words)
    )
    return result[mask].reset_index(drop=True)