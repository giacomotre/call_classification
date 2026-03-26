"""
Topic Model Validation.

No ground truth available, so we rely on three approaches:
  1. Topic distribution — are clusters balanced? Any single topic > 40%?
  2. Outlier analysis   — what % of docs landed in Topic -1?
  3. Manual spot-check   — sample N random docs per topic for you to eyeball

Usage:
    from topic_modeling.validation import validate_model

    report = validate_model(topic_model, docs, topics, probs)
"""
import pandas as pd
import numpy as np
from pathlib import Path


def topic_distribution(topic_model, topics):
    """
    Check how documents are spread across topics.

    Healthy pattern:  no single topic holds > 40% of docs
    Warning sign:     one mega-topic (usually means min_cluster_size too low)
    Warning sign:     many tiny topics (min_cluster_size too high)

    Returns a DataFrame with topic_id, count, and percentage.
    """
    info = topic_model.get_topic_info()
    total = len(topics)

    rows = []
    for _, row in info.iterrows():
        topic_id = row["Topic"]
        count = row["Count"]
        pct = round(100 * count / total, 1)

        # flag issues
        flag = ""
        if topic_id == -1 and pct > 30:
            flag = "⚠ high outlier rate"
        elif topic_id != -1 and pct > 40:
            flag = "⚠ dominant topic"
        elif topic_id != -1 and count < 20:
            flag = "⚠ very small cluster"

        rows.append({
            "topic_id": topic_id,
            "count": count,
            "pct": pct,
            "flag": flag,
        })

    return pd.DataFrame(rows)


def outlier_analysis(topics):
    """
    Summarize outlier (Topic -1) statistics.

    Returns a dict with counts and percentages.
    """
    topics_array = np.array(topics)
    total = len(topics_array)
    outliers = int((topics_array == -1).sum())
    assigned = total - outliers

    return {
        "total_docs": total,
        "assigned": assigned,
        "outliers": outliers,
        "outlier_pct": round(100 * outliers / total, 1),
        "num_topics": len(set(topics_array)) - (1 if -1 in topics_array else 0),
    }


def sample_docs_per_topic(docs, topics, n_samples=5, max_length=200):
    """
    Pull random document samples from each topic for manual review.

    This is your primary validation tool without ground truth.
    Read the samples and ask: "do these documents belong together?"

    Parameters
    ----------
    docs        : list of document strings
    topics      : list of topic assignments from fit_model
    n_samples   : how many docs to sample per topic
    max_length  : truncate long docs for readability

    Returns a DataFrame with topic_id, doc_index, and truncated text.
    """
    df = pd.DataFrame({"topic": topics, "text": docs})

    samples = []
    for topic_id in sorted(df["topic"].unique()):
        topic_docs = df[df["topic"] == topic_id]
        n = min(n_samples, len(topic_docs))
        sampled = topic_docs.sample(n, random_state=42)

        for idx, row in sampled.iterrows():
            text = row["text"]
            truncated = text[:max_length] + "..." if len(text) > max_length else text
            samples.append({
                "topic_id": topic_id,
                "doc_index": idx,
                "sample_text": truncated,
            })

    return pd.DataFrame(samples)


def confidence_summary(probs):
    """
    Summarize topic assignment confidence scores.

    Low average confidence → topics may be poorly separated.
    High confidence → documents clearly belong to their assigned topic.
    """
    probs_array = np.array(probs)

    # probs can be 1-D (single prob per doc) or 2-D (prob per topic)
    if probs_array.ndim == 2:
        max_probs = probs_array.max(axis=1)
    else:
        max_probs = probs_array

    return {
        "mean_confidence": round(float(np.mean(max_probs)), 3),
        "median_confidence": round(float(np.median(max_probs)), 3),
        "low_confidence_pct": round(
            100 * float((max_probs < 0.3).sum()) / len(max_probs), 1
        ),
    }


def validate_model(topic_model, docs, topics, probs, n_samples=5):
    """
    Run all validation checks and print a readable report.

    Returns a dict with all results for programmatic use.
    """
    print("=" * 60)
    print("TOPIC MODEL VALIDATION REPORT")
    print("=" * 60)

    # 1. Outlier analysis
    outliers = outlier_analysis(topics)
    print(f"\n── Outlier Analysis ──")
    print(f"  Total documents:  {outliers['total_docs']}")
    print(f"  Assigned to topic: {outliers['assigned']}")
    print(f"  Outliers (Topic -1): {outliers['outliers']} ({outliers['outlier_pct']}%)")
    print(f"  Topics discovered: {outliers['num_topics']}")

    # 2. Topic distribution
    dist = topic_distribution(topic_model, topics)
    print(f"\n── Topic Distribution ──")
    for _, row in dist.iterrows():
        label = f"  Topic {row['topic_id']:>3}: {row['count']:>5} docs ({row['pct']:>5}%)"
        if row["flag"]:
            label += f"  {row['flag']}"
        print(label)

    # 3. Confidence
    conf = confidence_summary(probs)
    print(f"\n── Confidence Scores ──")
    print(f"  Mean confidence:  {conf['mean_confidence']}")
    print(f"  Median confidence: {conf['median_confidence']}")
    print(f"  Low confidence (<0.3): {conf['low_confidence_pct']}%")

    # 4. Document samples
    samples = sample_docs_per_topic(docs, topics, n_samples=n_samples)
    print(f"\n── Sample Documents (first {n_samples} per topic) ──")
    for topic_id in sorted(samples["topic_id"].unique()):
        topic_samples = samples[samples["topic_id"] == topic_id]
        print(f"\n  Topic {topic_id}:")
        for _, row in topic_samples.iterrows():
            print(f"    [{row['doc_index']}] {row['sample_text']}")

    print("\n" + "=" * 60)

    return {
        "outliers": outliers,
        "distribution": dist,
        "confidence": conf,
        "samples": samples,
    }


def save_report(results, path):
    """
    Save validation results to CSV files in a reports directory.

    Creates:
      - {path}/topic_distribution.csv
      - {path}/doc_samples.csv
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    results["distribution"].to_csv(path / "topic_distribution.csv", index=False)
    results["samples"].to_csv(path / "doc_samples.csv", index=False)

    print(f"Reports saved to {path}/")
