"""
Topic Modeling Configuration.

Lives alongside the project's main config.py (loading & text extraction).
All BERTopic tuning knobs are here — no need to dig into module code.

Follows BERTopic best practices from Maarten Grootendorst's official docs:
  - Explicit sub-models (UMAP, HDBSCAN, CountVectorizer)
  - Pre-computed embeddings for fast iteration
  - random_state on UMAP for reproducibility
  - KeyBERTInspired for better topic representations
"""
from dataclasses import dataclass, field
from pathlib import Path


# ── Paths (topic-modeling specific) ────────────────────────────────────
MODELS_DIR = Path("data/models")
REPORTS_DIR = Path("data/reports")
EMBEDDINGS_DIR = Path("data/embeddings")

INPUT_PARQUET = Path("data/processed/cfr_savings_processed.parquet")
OUTPUT_PARQUET = Path("data/processed/cfr_savings_with_topics.parquet")


# ── Stage configs ──────────────────────────────────────────────────────

@dataclass
class EmbeddingConfig:
    """Stage 1: Sentence-Transformers embedding.

    model_name accepts either:
      - a HuggingFace model name: "all-MiniLM-L6-v2"  (needs internet)
      - a local folder path:      "models/all-MiniLM-L6-v2"  (offline)
    """
    model_name: str = "models/all-MiniLM-L6-v2"   # local path — no internet needed
    batch_size: int = 32
    show_progress: bool = True


@dataclass
class UMAPConfig:
    """Stage 2: Dimensionality reduction."""
    n_neighbors: int = 15
    n_components: int = 5
    min_dist: float = 0.0
    metric: str = "cosine"
    random_state: int = 42       # critical for reproducibility


@dataclass
class HDBSCANConfig:
    """Stage 3: Density-based clustering."""
    min_cluster_size: int = 150   # was 30 → forces broader clusters
    min_samples: int = 10
    metric: str = "euclidean"
    prediction_data: bool = True  # needed for .predict() later


@dataclass
class VectorizerConfig:
    """Stage 4: Tokenization for c-TF-IDF topic representation."""
    ngram_range: tuple = (1, 2)   # unigrams + bigrams
    min_df: int = 5               # ignore terms in < 5 docs
    max_df: float = 0.5           # ignore terms in > 50% of docs (was 0.95)

    # English stopwords + form template boilerplate only.
    # Keep domain words (even common ones) — let max_df handle frequency filtering.
    # Add more ONLY if you see obvious form labels in topic_terms.csv.
    custom_stop_words: list = field(default_factory=lambda: [
        # form template phrases that leak through extraction
        "problem", "description", "problem description",
        "complaint", "complaint handling", "handling",
        "process", "customer", "process customer",
        "device", "device used", "used",
        "customer function", "function", "role",
        "support", "support complaint",
        "occurrence", "occurrence malfunction",
        "diagnostic", "diagnostic expected", "expected",
        "diagnostics", "diagnostics expected",
        "used diagnostic", "used diagnostics",
        "patient impact", "impact", "user impact",
        "provided", "provided device", "provided patient",
        "patient involved", "involved",
        "information", "information support",
        "product", "product failure",
        "failure issue",
        "malfunction", "malfunction area", "area",
        "issue occurring", "occurring",
        "engineer", "description engineer",
        "technician", "technologist",
        "tech device", "technician device", "technologist device",
        "nurse device", "sonographer device",
        "device usage", "behavior product",
        "version", "confirmed", "problem confirmed",
        "sep",  # leftover separator token
    ])


# ── Text preparation ──────────────────────────────────────────────────

@dataclass
class TextPrepConfig:
    """Controls how extracted fields are combined and cleaned."""
    separator: str = " [SEP] "
    min_doc_length: int = 5      # chars — skip very short docs like "ok" (was 10)
    min_word_count: int = 2       # words — need enough content to embed (was 2)

    # columns to combine for the Problem model
    problem_columns: list = field(default_factory=lambda: [
        "extracted_problem_description_remote",
        "extracted_malfunction_area_remote",
        "extracted_problem_description_field",
        "extracted_malfunction_area_field",
    ])

    # columns to combine for the Resolution model
    resolution_columns: list = field(default_factory=lambda: [
        "extracted_repair_action_remote",
        "extracted_repair_action_field",
    ])


# ── Master config ─────────────────────────────────────────────────────

@dataclass
class TopicModelConfig:
    """Full pipeline configuration combining all stages."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    umap: UMAPConfig = field(default_factory=UMAPConfig)
    hdbscan: HDBSCANConfig = field(default_factory=HDBSCANConfig)
    vectorizer: VectorizerConfig = field(default_factory=VectorizerConfig)
    text_prep: TextPrepConfig = field(default_factory=TextPrepConfig)

    top_n_words: int = 15              # terms per topic in output
    nr_topics: str | int = "auto"      # "auto" or fixed int to merge down
    use_keybertinspired: bool = True   # recommended by BERTopic docs
    save_embeddings: bool = True       # cache embeddings for fast re-runs