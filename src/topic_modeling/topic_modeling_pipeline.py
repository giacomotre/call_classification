"""
Topic Modeling Pipeline — Orchestrator.

Wires together: load → prepare → embed → train → validate → save.
Runs two independent models: Problem topics and Resolution topics.

Usage:
    from pipeline.topic_modeling_pipeline import run_pipeline
    from topic_config import TopicModelConfig

    cfg = TopicModelConfig()
    run_pipeline(cfg)

Or from command line:
    python -m pipeline.topic_modeling_pipeline
"""
import pandas as pd
import numpy as np
from pathlib import Path

from topic_config import (
    TopicModelConfig, INPUT_PARQUET, OUTPUT_PARQUET,
    MODELS_DIR, REPORTS_DIR, EMBEDDINGS_DIR,
)
from topic_modeling.text_preparation import prepare_documents
from topic_modeling.bertopic_wrapper import (
    build_topic_model, compute_embeddings,
    save_embeddings, load_embeddings,
    fit_model, get_topic_summary,
    save_model,
)
from topic_modeling.validation import validate_model, save_report


# ── Single model run ──────────────────────────────────────────────────

def run_single_model(df, cfg, model_name, columns):
    """
    Run the full pipeline for one model (problem OR resolution).

    Steps:
      1. Prepare documents from extracted columns
      2. Compute or load cached embeddings
      3. Build and train BERTopic model
      4. Validate results
      5. Save model, embeddings, and reports

    Parameters
    ----------
    df          : source dataframe with extracted_* columns
    cfg         : TopicModelConfig instance
    model_name  : "problem" or "resolution" (used for file naming)
    columns     : which extracted_* columns to combine

    Returns
    -------
    doc_df      : DataFrame with doc_index and doc_text
    topics      : list of topic assignments
    probs       : topic probabilities
    topic_model : trained BERTopic model
    """
    print(f"\n{'='*60}")
    print(f"  RUNNING: {model_name.upper()} TOPIC MODEL")
    print(f"{'='*60}")

    # ── Step 1: Prepare documents ──
    print(f"\n[1/5] Preparing {model_name} documents...")
    doc_df = prepare_documents(
        df,
        columns=columns,
        separator=cfg.text_prep.separator,
        min_length=cfg.text_prep.min_doc_length,
        min_words=cfg.text_prep.min_word_count,
    )
    docs = doc_df["doc_text"].tolist()
    print(f"  {len(df)} input rows → {len(docs)} valid documents")

    # ── Step 2: Embeddings (compute or load cached) ──
    embeddings_path = EMBEDDINGS_DIR / f"{model_name}_embeddings.npy"

    if embeddings_path.exists():
        print(f"\n[2/5] Loading cached embeddings from {embeddings_path}...")
        embeddings = load_embeddings(embeddings_path)

        # sanity check: cached embeddings must match doc count
        if len(embeddings) != len(docs):
            print(f"  ⚠ Cache mismatch ({len(embeddings)} vs {len(docs)} docs). Re-computing...")
            topic_model, embedding_model = build_topic_model(cfg)
            embeddings = compute_embeddings(
                docs, embedding_model,
                batch_size=cfg.embedding.batch_size,
                show_progress=cfg.embedding.show_progress,
            )
            save_embeddings(embeddings, embeddings_path)
        else:
            topic_model, _ = build_topic_model(cfg)
    else:
        print(f"\n[2/5] Computing embeddings (this takes a few minutes)...")
        topic_model, embedding_model = build_topic_model(cfg)
        embeddings = compute_embeddings(
            docs, embedding_model,
            batch_size=cfg.embedding.batch_size,
            show_progress=cfg.embedding.show_progress,
        )
        if cfg.save_embeddings:
            save_embeddings(embeddings, embeddings_path)

    # ── Step 3: Train ──
    print(f"\n[3/5] Training BERTopic model...")
    topics, probs = fit_model(topic_model, docs, embeddings)

    # ── Step 4: Validate ──
    print(f"\n[4/5] Validating model...")
    results = validate_model(topic_model, docs, topics, probs)
    save_report(results, REPORTS_DIR / model_name)

    # Topic terms summary
    summary = get_topic_summary(topic_model)
    summary_path = REPORTS_DIR / model_name / "topic_terms.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n  Topic terms saved to {summary_path}")
    print(f"\n  ── Topic Terms ──")
    for _, row in summary.iterrows():
        print(f"  Topic {row['topic_id']:>2} ({row['count']:>5} docs): {row['top_terms']}")

    # ── Step 5: Save model ──
    print(f"\n[5/5] Saving model...")
    save_model(topic_model, MODELS_DIR / f"{model_name}_model")

    return doc_df, topics, probs, topic_model


# ── Map topics back to original dataframe ─────────────────────────────

def assign_topics_to_df(df, doc_df, topics, probs, prefix):
    """
    Map topic assignments back to the original dataframe rows.

    Documents that were filtered out (too short, null) get topic_id = -1
    and confidence = 0.0.

    Parameters
    ----------
    df      : original full dataframe
    doc_df  : output from prepare_documents (has doc_index column)
    topics  : topic assignments from fit_model
    probs   : probabilities from fit_model
    prefix  : "problem" or "resolution" — column name prefix

    Adds columns:
      - {prefix}_topic_id
      - {prefix}_confidence
    """
    # default: unassigned
    df[f"{prefix}_topic_id"] = -1
    df[f"{prefix}_confidence"] = 0.0

    # map back using doc_index
    probs_array = np.array(probs)
    if probs_array.ndim == 2:
        max_probs = probs_array.max(axis=1)
    else:
        max_probs = probs_array

    for i, row in doc_df.iterrows():
        orig_idx = row["doc_index"]
        df.at[orig_idx, f"{prefix}_topic_id"] = topics[i]
        df.at[orig_idx, f"{prefix}_confidence"] = float(max_probs[i])

    assigned = (df[f"{prefix}_topic_id"] != -1).sum()
    print(f"  {prefix}: {assigned}/{len(df)} rows assigned to a topic")


# ── Full pipeline ─────────────────────────────────────────────────────

def run_pipeline(cfg=None):
    """
    Run the complete topic modeling pipeline.

    1. Load processed parquet
    2. Train Problem model
    3. Train Resolution model
    4. Map topics back to dataframe
    5. Save enriched parquet
    """
    if cfg is None:
        cfg = TopicModelConfig()

    # ── Load data ──
    print(f"Loading data from {INPUT_PARQUET}...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # ── Problem model ──
    problem_doc_df, problem_topics, problem_probs, problem_model = run_single_model(
        df, cfg,
        model_name="problem",
        columns=cfg.text_prep.problem_columns,
    )

    # ── Resolution model ──
    resolution_doc_df, resolution_topics, resolution_probs, resolution_model = run_single_model(
        df, cfg,
        model_name="resolution",
        columns=cfg.text_prep.resolution_columns,
    )

    # ── Map topics back ──
    print(f"\n{'='*60}")
    print(f"  MAPPING TOPICS TO DATASET")
    print(f"{'='*60}")
    assign_topics_to_df(df, problem_doc_df, problem_topics, problem_probs, "problem")
    assign_topics_to_df(df, resolution_doc_df, resolution_topics, resolution_probs, "resolution")

    # ── Save enriched parquet ──
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nEnriched dataset saved to {OUTPUT_PARQUET}")
    print(f"New columns: problem_topic_id, problem_confidence, "
          f"resolution_topic_id, resolution_confidence")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Problem topics:    {len(set(problem_topics) - {-1})}")
    print(f"  Resolution topics: {len(set(resolution_topics) - {-1})}")
    print(f"  Output: {OUTPUT_PARQUET}")
    print(f"  Reports: {REPORTS_DIR}/")
    print(f"  Models: {MODELS_DIR}/")

    return df


# ── CLI entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()
