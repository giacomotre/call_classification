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
    """Stage 1: Sentence-Transformers embedding."""
    model_name: str = "all-MiniLM-L6-v2"
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
    min_cluster_size: int = 30
    min_samples: int = 10
    metric: str = "euclidean"
    prediction_data: bool = True  # needed for .predict() later


@dataclass
class VectorizerConfig:
    """Stage 4: Tokenization for c-TF-IDF topic representation."""
    ngram_range: tuple = (1, 2)   # unigrams + bigrams
    stop_words: str = "english"
    min_df: int = 5               # ignore terms in < 5 docs
    max_df: float = 0.95          # ignore terms in > 95% of docs


# ── Text preparation ──────────────────────────────────────────────────

@dataclass
class TextPrepConfig:
    """Controls how extracted fields are combined and cleaned."""
    separator: str = " [SEP] "
    min_doc_length: int = 10      # chars — skip very short docs like "ok"
    min_word_count: int = 3       # words — need enough content to embed

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