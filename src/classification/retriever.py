"""
Retriever for RAG-based classification.

Hybrid retrieval: BM25 (sparse, word-match) + dense embeddings (semantic),
fused via Reciprocal Rank Fusion. All processing runs locally.

Usage:
    from classification.retriever import build_index, retrieve_examples

    index = build_index(labeled_df, text_col, label_cols, embedding_model)
    examples = retrieve_examples("chiller temp high", index, embedding_model, n=5)
"""
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


def build_index(labeled_df, text_col, label_cols, embedding_model, batch_size=32):
    """
    Build a hybrid retrieval index from labeled cases.

    Computes dense embeddings and a BM25 sparse index over the same texts.
    Both are stored for hybrid retrieval.

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
        "embeddings" : numpy array of dense embeddings
        "bm25"       : BM25Okapi sparse scorer
        "texts"      : list of text strings
        "labels"     : list of dicts with label values
    """
    # filter to rows that have both text and labels
    valid = labeled_df[text_col].notna()
    for col in label_cols:
        valid = valid & labeled_df[col].notna()

    subset = labeled_df[valid].copy()
    texts = subset[text_col].tolist()

    # dense embeddings
    print(f"  Embedding {len(texts)} labeled cases...")
    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    # BM25 sparse index — simple lowercase tokenization keeps error codes intact
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    # store labels as list of dicts for easy access
    labels = []
    for _, row in subset.iterrows():
        labels.append({col: row[col] for col in label_cols})

    print(f"  Index built: {len(texts)} cases, "
          f"{embeddings.shape[1]} dense dims + BM25 sparse")

    return {
        "embeddings": embeddings,
        "bm25": bm25,
        "texts": texts,
        "labels": labels,
    }


def save_index(index, path):
    """Save the index to disk. BM25 is rebuilt from texts on load."""
    from pathlib import Path
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "embeddings.npy", index["embeddings"])
    pd.DataFrame({
        "text": index["texts"],
        **{k: [lab[k] for lab in index["labels"]] for k in index["labels"][0]}
    }).to_csv(path / "labeled_cases.csv", index=False)
    print(f"  Index saved to {path}")


def load_index(path):
    """Load a previously saved index. Rebuilds BM25 from saved texts."""
    from pathlib import Path
    path = Path(path)
    embeddings = np.load(path / "embeddings.npy")
    df = pd.read_csv(path / "labeled_cases.csv")
    texts = df["text"].tolist()
    label_cols = [c for c in df.columns if c != "text"]
    labels = []
    for _, row in df.iterrows():
        labels.append({col: row[col] for col in label_cols})

    # rebuild BM25 from loaded texts
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    print(f"  Index loaded: {len(texts)} cases from {path}")
    return {
        "embeddings": embeddings,
        "bm25": bm25,
        "texts": texts,
        "labels": labels,
    }


def rrf_fuse(dense_indices, bm25_indices, k=60):
    """
    Combine two ranked lists using Reciprocal Rank Fusion.

    For each document, its fused score is:
        score = 1/(k + rank_in_dense) + 1/(k + rank_in_bm25)

    Documents that appear in only one list still score from that list alone.

    Parameters
    ----------
    dense_indices : array of doc indices, sorted by dense similarity (best first)
    bm25_indices  : array of doc indices, sorted by BM25 score (best first)
    k             : RRF constant (60 is the standard value)

    Returns list of document indices sorted by fused score (best first).
    """
    scores = {}
    for rank, doc_idx in enumerate(dense_indices):
        scores[doc_idx] = scores.get(doc_idx, 0) + 1 / (k + rank + 1)
    for rank, doc_idx in enumerate(bm25_indices):
        scores[doc_idx] = scores.get(doc_idx, 0) + 1 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)


def retrieve_examples(query_text, index, embedding_model, n=5, n_candidates=50):
    """
    Find the n most similar labeled cases using hybrid retrieval.

    Gets top n_candidates from dense and BM25 separately,
    fuses with RRF, returns top-n.

    Parameters
    ----------
    query_text       : the unlabeled case text to classify
    index            : the retrieval index from build_index()
    embedding_model  : SentenceTransformer model
    n                : number of examples to return
    n_candidates     : candidates pulled from each retriever before fusion

    Returns
    -------
    list of dicts, each with:
        "text"       : the labeled case text
        "labels"     : dict of label values
        "similarity" : dense cosine similarity (kept for evaluation metrics)
    """
    # dense: cosine similarity against all indexed embeddings
    query_embedding = embedding_model.encode([query_text])[0]
    index_embeddings = index["embeddings"]
    norms = np.linalg.norm(index_embeddings, axis=1) * np.linalg.norm(query_embedding)
    similarities = np.dot(index_embeddings, query_embedding) / norms
    dense_top = np.argsort(similarities)[::-1][:n_candidates]

    # BM25: word-match scoring
    query_tokens = query_text.lower().split()
    bm25_scores = index["bm25"].get_scores(query_tokens)
    bm25_top = np.argsort(bm25_scores)[::-1][:n_candidates]

    # fuse rankings
    fused = rrf_fuse(dense_top, bm25_top)

    results = []
    for idx in fused[:n]:
        results.append({
            "text": index["texts"][idx],
            "labels": index["labels"][idx],
            "similarity": float(similarities[idx]),
        })

    return results


def retrieve_batch(query_texts, index, embedding_model, n=5,
                   n_candidates=50, batch_size=32):
    """
    Retrieve examples for multiple cases using hybrid retrieval.

    More efficient than calling retrieve_examples in a loop — embeds all
    queries in one batch, then fuses per query.

    Returns list of lists (one list of examples per query).
    """
    # dense: batch embed + full similarity matrix
    query_embeddings = embedding_model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    index_embeddings = index["embeddings"]
    index_norms = np.linalg.norm(index_embeddings, axis=1, keepdims=True)
    query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)

    index_normed = index_embeddings / index_norms
    query_normed = query_embeddings / query_norms

    sim_matrix = np.dot(query_normed, index_normed.T)

    # per-query: BM25 scoring + RRF fusion
    all_results = []
    for i in range(len(query_texts)):
        dense_top = np.argsort(sim_matrix[i])[::-1][:n_candidates]

        query_tokens = query_texts[i].lower().split()
        bm25_scores = index["bm25"].get_scores(query_tokens)
        bm25_top = np.argsort(bm25_scores)[::-1][:n_candidates]

        fused = rrf_fuse(dense_top, bm25_top)

        results = []
        for idx in fused[:n]:
            results.append({
                "text": index["texts"][idx],
                "labels": index["labels"][idx],
                "similarity": float(sim_matrix[i][idx]),
            })
        all_results.append(results)

    return all_results