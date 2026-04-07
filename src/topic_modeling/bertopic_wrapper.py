"""
BERTopic Wrapper.

Thin layer around BERTopic that:
  - Explicitly defines each sub-model (no hidden defaults)
  - Pre-computes embeddings separately (fast iteration on clustering params)
  - Saves/loads models and embeddings to disk
  - Follows Maarten Grootendorst's recommended patterns

Usage:
    from topic_modeling.bertopic_wrapper import TopicModel
    from topic_config import TopicModelConfig

    cfg = TopicModelConfig()
    model = TopicModel(cfg)
    topics, probs = model.fit(docs)
    model.save("data/models/problem_model")
"""
import numpy as np
import pandas as pd
from pathlib import Path

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired


def build_topic_model(cfg, seed_topic_list=None):
    """
    Construct a BERTopic instance with explicit sub-models from config.

    Each stage from our pipeline diagram becomes a named component:
      Stage 1 → embedding_model  (SentenceTransformer)
      Stage 2 → umap_model       (UMAP)
      Stage 3 → hdbscan_model    (HDBSCAN)
      Stage 4 → vectorizer_model (CountVectorizer)
      Stage 5 → c-TF-IDF         (handled internally by BERTopic)

    Parameters
    ----------
    cfg              : TopicModelConfig
    seed_topic_list  : optional list of seed word lists for guided BERTopic.
                       Example: [["chiller", "cooling"], ["coil", "body coil"]]
                       Seeds nudge the model but don't force topic count.
    """
    # Stage 1: embedding model
    embedding_model = SentenceTransformer(cfg.embedding.model_name)

    # Stage 2: dimensionality reduction
    umap_model = UMAP(
        n_neighbors=cfg.umap.n_neighbors,
        n_components=cfg.umap.n_components,
        min_dist=cfg.umap.min_dist,
        metric=cfg.umap.metric,
        random_state=cfg.umap.random_state,
    )

    # Stage 3: clustering
    hdbscan_model = HDBSCAN(
        min_cluster_size=cfg.hdbscan.min_cluster_size,
        min_samples=cfg.hdbscan.min_samples,
        metric=cfg.hdbscan.metric,
        prediction_data=cfg.hdbscan.prediction_data,
    )

    # Stage 4: tokenization for topic representation
    # Merge sklearn's English stopwords with our custom boilerplate list
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    all_stop_words = list(ENGLISH_STOP_WORDS | set(cfg.vectorizer.custom_stop_words))

    vectorizer_model = CountVectorizer(
        ngram_range=cfg.vectorizer.ngram_range,
        stop_words=all_stop_words,
        min_df=cfg.vectorizer.min_df,
        max_df=cfg.vectorizer.max_df,
    )

    # Optional: KeyBERTInspired for better topic terms
    representation_model = KeyBERTInspired() if cfg.use_keybertinspired else None

    # Wire everything together
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        top_n_words=cfg.top_n_words,
        nr_topics=cfg.nr_topics if cfg.nr_topics != "auto" else None,
        seed_topic_list=seed_topic_list,
        verbose=True,
    )

    return topic_model, embedding_model


def compute_embeddings(docs, embedding_model, batch_size=32, show_progress=True):
    """
    Pre-compute embeddings separately from BERTopic.

    Why? This is the slow step (~5-10 min for 13K docs on CPU).
    By computing once and saving, you can re-run clustering with
    different UMAP/HDBSCAN params without re-embedding.
    """
    embeddings = embedding_model.encode(
        docs,
        batch_size=batch_size,
        show_progress_bar=show_progress,
    )
    return embeddings


def save_embeddings(embeddings, path):
    """Save embeddings as numpy array for fast re-loading."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    print(f"Embeddings saved to {path}")


def load_embeddings(path):
    """Load pre-computed embeddings."""
    path = Path(path)
    embeddings = np.load(path)
    print(f"Embeddings loaded from {path}: shape {embeddings.shape}")
    return embeddings


def fit_model(topic_model, docs, embeddings):
    """
    Train BERTopic on documents using pre-computed embeddings.

    Returns:
        topics : list of topic assignments (one per doc, -1 = outlier)
        probs  : topic probability per doc (confidence)
    """
    topics, probs = topic_model.fit_transform(docs, embeddings)
    return topics, probs


def get_topic_summary(topic_model):
    """
    Extract a clean summary of discovered topics.

    Returns a DataFrame with topic_id, count, and top terms.
    This is what you'll read to decide on labels.
    """
    info = topic_model.get_topic_info()

    # build a cleaner terms column from the raw representation
    summaries = []
    for topic_id in info["Topic"]:
        if topic_id == -1:
            continue
        terms = topic_model.get_topic(topic_id)
        term_list = [word for word, score in terms]
        summaries.append({
            "topic_id": topic_id,
            "count": info[info["Topic"] == topic_id]["Count"].values[0],
            "top_terms": ", ".join(term_list),
        })

    summary_df = pd.DataFrame(summaries)

    # add outlier count as context
    outlier_count = info[info["Topic"] == -1]["Count"].values
    if len(outlier_count) > 0:
        print(f"Outlier documents (Topic -1): {outlier_count[0]}")

    return summary_df


def save_model(topic_model, path):
    """Save trained BERTopic model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    topic_model.save(path, serialization="pickle")
    print(f"Model saved to {path}")


def load_model(path):
    """Load a trained BERTopic model."""
    path = Path(path)
    topic_model = BERTopic.load(path)
    print(f"Model loaded from {path}")
    return topic_model