"""
Retriever for RAG-based classification.

Embeds labeled cases and retrieves the most similar ones
for a given unlabeled case. All processing runs locally.

Usage:
    from classification.retriever import build_index, retrieve_examples

    index = build_index(labeled_df, text_col, embedding_model)
    examples = retrieve_examples("chiller temp high", index, n=5)
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def build_index(labeled_df, text_col, label_cols, embedding_model, batch_size=32):
    """
    Build a retrieval index from labeled cases.

    Embeds all labeled texts and stores them alongside their labels
    for fast nearest-neighbor lookup.

    Parameters
    ----------
    labeled_df       : dataframe with labeled cases
    text_col         : column name containing the text to embed
    label_cols       : list of column names containing labels
                       e.g., ["nam_main_category", "nam_sub_category"]
    embedding_model  : SentenceTransformer model (already loaded)
    batch_size       : batch size for embedding

    Returns
    -------
    dict with:
        "embeddings" : numpy array of shape (n_cases, 384)
        "texts"      : list of text strings
        "labels"     : list of dicts with label values
    """
    # filter to rows that have both text and labels
    valid = labeled_df[text_col].notna()
    for col in label_cols:
        valid = valid & labeled_df[col].notna()

    subset = labeled_df[valid].copy()
    texts = subset[text_col].tolist()

    print(f"  Embedding {len(texts)} labeled cases...")
    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    # store labels as list of dicts for easy access
    labels = []
    for _, row in subset.iterrows():
        labels.append({col: row[col] for col in label_cols})

    print(f"  Index built: {len(texts)} cases, {embeddings.shape[1]} dimensions")

    return {
        "embeddings": embeddings,
        "texts": texts,
        "labels": labels,
    }


def save_index(index, path):
    """Save the index to disk for fast re-loading."""
    from pathlib import Path
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "embeddings.npy", index["embeddings"])
    pd.DataFrame({
        "text": index["texts"],
        **{k: [lab[k] for lab in index["labels"]] for k in index["labels"][0]}
    }).to_csv(path / "labeled_cases.csv", index=False)
    print(f"  Index saved to {path}")


def load_index(path, embedding_shape=384):
    """Load a previously saved index."""
    from pathlib import Path
    path = Path(path)
    embeddings = np.load(path / "embeddings.npy")
    df = pd.read_csv(path / "labeled_cases.csv")
    texts = df["text"].tolist()
    label_cols = [c for c in df.columns if c != "text"]
    labels = []
    for _, row in df.iterrows():
        labels.append({col: row[col] for col in label_cols})

    print(f"  Index loaded: {len(texts)} cases from {path}")
    return {
        "embeddings": embeddings,
        "texts": texts,
        "labels": labels,
    }


def retrieve_examples(query_text, index, embedding_model, n=5):
    """
    Find the n most similar labeled cases for a given text.

    Parameters
    ----------
    query_text       : the unlabeled case text to classify
    index            : the retrieval index from build_index()
    embedding_model  : SentenceTransformer model
    n                : number of examples to retrieve

    Returns
    -------
    list of dicts, each with:
        "text"       : the labeled case text
        "labels"     : dict of label values
        "similarity" : cosine similarity score (0-1)
    """
    # embed the query
    query_embedding = embedding_model.encode([query_text])[0]

    # cosine similarity against all indexed embeddings
    # cos_sim = (A · B) / (||A|| * ||B||)
    index_embeddings = index["embeddings"]
    norms = np.linalg.norm(index_embeddings, axis=1) * np.linalg.norm(query_embedding)
    similarities = np.dot(index_embeddings, query_embedding) / norms

    # get top-n indices
    top_indices = np.argsort(similarities)[::-1][:n]

    results = []
    for idx in top_indices:
        results.append({
            "text": index["texts"][idx],
            "labels": index["labels"][idx],
            "similarity": float(similarities[idx]),
        })

    return results


def retrieve_batch(query_texts, index, embedding_model, n=5, batch_size=32):
    """
    Retrieve examples for multiple cases at once. More efficient than
    calling retrieve_examples in a loop.

    Returns list of lists (one list of examples per query).
    """
    query_embeddings = embedding_model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    index_embeddings = index["embeddings"]
    # normalize for cosine similarity
    index_norms = np.linalg.norm(index_embeddings, axis=1, keepdims=True)
    query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)

    index_normed = index_embeddings / index_norms
    query_normed = query_embeddings / query_norms

    # similarity matrix: (n_queries, n_indexed)
    sim_matrix = np.dot(query_normed, index_normed.T)

    all_results = []
    for i in range(len(query_texts)):
        top_indices = np.argsort(sim_matrix[i])[::-1][:n]
        results = []
        for idx in top_indices:
            results.append({
                "text": index["texts"][idx],
                "labels": index["labels"][idx],
                "similarity": float(sim_matrix[i][idx]),
            })
        all_results.append(results)

    return all_results
