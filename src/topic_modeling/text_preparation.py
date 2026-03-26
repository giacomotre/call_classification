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


# ── Stage A: clean a single text field ────────────────────────────────

def clean_text(text: str) -> str | None:
    """
    Minimal cleaning for one extracted field.

    Does:
      - Fix encoding artifacts (ftfy)
      - Remove URLs and emails
      - Strip leftover [SEP] markers from extraction phase
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
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else None


# ── Stage B: combine fields into one document ─────────────────────────

def combine_fields(row: pd.Series, columns: list, separator: str) -> str | None:
    """
    Join multiple extracted columns into a single document string.

    Skips null/empty fields so the separator only appears between
    real content. Example:

        columns = ["extracted_problem_description_remote",
                    "extracted_malfunction_area_remote"]
        row vals: ["Can't scan, disk full", "Storage System"]
        → "Can't scan, disk full [SEP] Storage System"

        row vals: [None, "Storage System"]
        → "Storage System"   (no leading separator)
    """
    parts = []
    for col in columns:
        cleaned = clean_text(row.get(col))
        if cleaned:
            parts.append(cleaned)

    return separator.join(parts) if parts else None


def prepare_documents(df: pd.DataFrame, columns: list, separator: str,
                      min_length: int = 10,
                      min_words: int = 3) -> pd.DataFrame:
    """
    Full text preparation: combine fields → filter short docs.

    Parameters
    ----------
    df          : source dataframe (the parquet)
    columns     : which extracted_* columns to combine
    separator   : join string, typically " [SEP] "
    min_length  : drop docs shorter than this (chars)
    min_words   : drop docs with fewer words than this

    Returns
    -------
    DataFrame with:
      - doc_index : original row index (to map topics back later)
      - doc_text  : combined cleaned text, ready for embedding
    """
    docs = df.apply(lambda row: combine_fields(row, columns, separator), axis=1)

    result = pd.DataFrame({
        "doc_index": df.index,
        "doc_text": docs,
    })

    # drop nulls and too-short documents
    result = result.dropna(subset=["doc_text"])
    mask = (
        (result["doc_text"].str.len() >= min_length) &
        (result["doc_text"].str.split().str.len() >= min_words)
    )
    result = result[mask].reset_index(drop=True)

    return result