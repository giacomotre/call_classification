"""
Evaluation Framework for RAG Classification.

Splits labeled data into index + test, runs the classification pipeline
on the test set, and reports accuracy at each stage:
  1. Retrieval quality — do retrieved examples have the right labels?
  2. Classification accuracy — does the LLM output match the true label?
  3. Per-category breakdown — which categories work, which don't?

Usage:
    from classification.evaluation import run_evaluation

    results = run_evaluation(
        labeled_df=nam_labeled,
        text_col="extracted_problem_description_remote",
        label_cols=["nam_main_category", "nam_sub_category"],
        embedding_model=embedding_model,
        taxonomy_text=taxonomy_text,
        test_size=0.2,
    )
"""
import pandas as pd
import numpy as np
from collections import Counter


def split_labeled_data(labeled_df, test_size=0.2, random_state=42):
    """
    Split labeled cases into index (training) and test sets.

    Uses stratified-like splitting: shuffles, then splits.
    Prints the distribution of labels in both sets.

    Parameters
    ----------
    labeled_df   : dataframe with labeled cases
    test_size    : fraction to hold out for testing (0.2 = 20%)
    random_state : seed for reproducibility

    Returns (index_df, test_df)
    """
    shuffled = labeled_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split_point = int(len(shuffled) * (1 - test_size))

    index_df = shuffled.iloc[:split_point].copy()
    test_df = shuffled.iloc[split_point:].copy()

    print(f"  Split: {len(index_df)} index / {len(test_df)} test "
          f"({100*(1-test_size):.0f}% / {100*test_size:.0f}%)")

    return index_df, test_df


def evaluate_retrieval(test_df, all_examples, text_col, main_label_col):
    """
    Check whether retrieved examples have the correct label.

    For each test case, looks at the majority label among retrieved
    examples and compares it to the true label.

    Parameters
    ----------
    test_df        : test set dataframe
    all_examples   : list of lists from retrieve_batch()
    text_col       : column name with the text
    main_label_col : column name with the main category label

    Returns dict with retrieval metrics.
    """
    true_labels = test_df[main_label_col].tolist()
    total = len(true_labels)

    majority_correct = 0
    any_correct = 0
    avg_top1_similarity = 0

    details = []

    for i, (true_label, examples) in enumerate(zip(true_labels, all_examples)):
        # labels of retrieved examples
        retrieved_labels = [ex["labels"][main_label_col] for ex in examples]
        similarities = [ex["similarity"] for ex in examples]

        # majority vote
        label_counts = Counter(retrieved_labels)
        majority_label = label_counts.most_common(1)[0][0]

        # metrics
        is_majority_correct = majority_label == true_label
        is_any_correct = true_label in retrieved_labels

        majority_correct += int(is_majority_correct)
        any_correct += int(is_any_correct)
        avg_top1_similarity += similarities[0]

        details.append({
            "true_label": true_label,
            "majority_label": majority_label,
            "majority_correct": is_majority_correct,
            "any_correct": is_any_correct,
            "top1_similarity": similarities[0],
            "retrieved_labels": retrieved_labels,
        })

    return {
        "majority_accuracy": majority_correct / total,
        "any_correct_rate": any_correct / total,
        "avg_top1_similarity": avg_top1_similarity / total,
        "total": total,
        "details": details,
    }


def evaluate_classification(test_df, predictions, main_label_col, sub_label_col=None):
    """
    Compare predicted labels against true labels.

    Parameters
    ----------
    test_df        : test set with true labels
    predictions    : DataFrame from classify_batch() with predicted labels
    main_label_col : column with true main category
    sub_label_col  : column with true sub category (optional)

    Returns dict with classification metrics.
    """
    true_main = test_df[main_label_col].tolist()
    pred_main = predictions["main_category"].tolist()
    total = len(true_main)

    main_correct = sum(t == p for t, p in zip(true_main, pred_main))

    result = {
        "main_accuracy": main_correct / total,
        "main_correct": main_correct,
        "total": total,
    }

    if sub_label_col:
        true_sub = test_df[sub_label_col].tolist()
        pred_sub = predictions["sub_category"].tolist()
        sub_correct = sum(t == p for t, p in zip(true_sub, pred_sub))
        both_correct = sum(
            tm == pm and ts == ps
            for tm, pm, ts, ps in zip(true_main, pred_main, true_sub, pred_sub)
        )

        result["sub_accuracy"] = sub_correct / total
        result["sub_correct"] = sub_correct
        result["both_accuracy"] = both_correct / total
        result["both_correct"] = both_correct

    return result


def confusion_report(test_df, predictions, main_label_col):
    """
    Per-category breakdown: accuracy, count, and common misclassifications.

    Returns a DataFrame sorted by accuracy (worst first) so you know
    where to focus improvement efforts.
    """
    true_main = test_df[main_label_col].tolist()
    pred_main = predictions["main_category"].tolist()

    # per-category stats
    category_stats = {}
    for true, pred in zip(true_main, pred_main):
        if true not in category_stats:
            category_stats[true] = {"correct": 0, "total": 0, "misclassified_as": []}

        category_stats[true]["total"] += 1
        if true == pred:
            category_stats[true]["correct"] += 1
        else:
            category_stats[true]["misclassified_as"].append(pred)

    rows = []
    for cat, stats in category_stats.items():
        total = stats["total"]
        correct = stats["correct"]
        accuracy = correct / total if total > 0 else 0

        # most common misclassification
        if stats["misclassified_as"]:
            top_confusion = Counter(stats["misclassified_as"]).most_common(1)[0]
            confused_with = f"{top_confusion[0]} ({top_confusion[1]}x)"
        else:
            confused_with = ""

        rows.append({
            "category": cat,
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 3),
            "most_confused_with": confused_with,
        })

    report_df = pd.DataFrame(rows).sort_values("accuracy", ascending=True)
    return report_df.reset_index(drop=True)


def diagnose_failures(test_df, predictions, retrieval_details,
                      main_label_col, n_show=10):
    """
    Diagnose WHY cases were misclassified.

    For each failure, determines:
      - Was retrieval wrong? (majority of retrieved examples had wrong label)
      - Was the LLM wrong? (retrieval was right but LLM disagreed)

    Prints the top-n failures with context for debugging.
    """
    true_main = test_df[main_label_col].tolist()
    pred_main = predictions["main_category"].tolist()
    texts = test_df.iloc[:, 0].tolist()  # first column as fallback

    retrieval_failures = 0
    llm_failures = 0
    failures = []

    for i, (true, pred) in enumerate(zip(true_main, pred_main)):
        if true == pred:
            continue

        detail = retrieval_details[i]
        retrieval_was_correct = detail["majority_correct"]

        if not retrieval_was_correct:
            retrieval_failures += 1
            failure_type = "RETRIEVAL"
        else:
            llm_failures += 1
            failure_type = "LLM"

        failures.append({
            "index": i,
            "true_label": true,
            "predicted": pred,
            "failure_type": failure_type,
            "majority_retrieved": detail["majority_label"],
            "retrieved_labels": detail["retrieved_labels"],
            "top1_similarity": detail["top1_similarity"],
        })

    total_failures = len(failures)
    print(f"\n── Failure Diagnosis ──")
    print(f"  Total failures: {total_failures}")
    if total_failures > 0:
        print(f"  Retrieval failures: {retrieval_failures} "
              f"({100*retrieval_failures/total_failures:.0f}%) "
              f"— retrieved examples had wrong labels")
        print(f"  LLM failures: {llm_failures} "
              f"({100*llm_failures/total_failures:.0f}%) "
              f"— retrieval was correct but LLM disagreed")

        print(f"\n  Top {min(n_show, total_failures)} failures:")
        for f in failures[:n_show]:
            print(f"    [{f['failure_type']}] true={f['true_label']}, "
                  f"pred={f['predicted']}, "
                  f"retrieved={f['retrieved_labels']}, "
                  f"sim={f['top1_similarity']:.3f}")

    return failures


def run_evaluation(labeled_df, text_col, label_cols, embedding_model,
                   taxonomy_text, test_size=0.2, n_examples=5,
                   random_state=42):
    """
    Run the full evaluation pipeline.

    Steps:
      1. Split labeled data into index + test
      2. Build retrieval index from index set
      3. Retrieve examples for all test cases
      4. Evaluate retrieval quality
      5. Classify all test cases via LLM
      6. Evaluate classification accuracy
      7. Per-category confusion report
      8. Diagnose failures

    Returns dict with all results for further analysis.
    """
    from classification.retriever import build_index, retrieve_batch
    from classification.classifier import classify_batch

    main_label_col = label_cols[0]
    sub_label_col = label_cols[1] if len(label_cols) > 1 else None

    print("=" * 60)
    print("  CLASSIFICATION EVALUATION")
    print("=" * 60)

    # 1. Split
    print("\n── Step 1: Splitting data ──")
    index_df, test_df = split_labeled_data(labeled_df, test_size, random_state)

    # 2. Build index
    print("\n── Step 2: Building retrieval index ──")
    index = build_index(
        index_df, text_col, label_cols,
        embedding_model, batch_size=32,
    )

    # 3. Retrieve examples for test cases
    print("\n── Step 3: Retrieving examples for test cases ──")
    test_texts = test_df[text_col].tolist()
    all_examples = retrieve_batch(
        test_texts, index, embedding_model,
        n=n_examples, batch_size=32,
    )

    # 4. Evaluate retrieval
    print("\n── Step 4: Retrieval quality ──")
    retrieval_results = evaluate_retrieval(test_df, all_examples, text_col, main_label_col)
    print(f"  Majority vote accuracy: {retrieval_results['majority_accuracy']:.1%}")
    print(f"  Any correct in top-{n_examples}: {retrieval_results['any_correct_rate']:.1%}")
    print(f"  Avg top-1 similarity: {retrieval_results['avg_top1_similarity']:.3f}")

    # 5. Classify test cases
    print("\n── Step 5: Classifying test cases via LLM ──")
    predictions = classify_batch(
        test_texts, index, embedding_model,
        taxonomy_text, n_examples=n_examples,
    )

    # 6. Classification accuracy
    print("\n── Step 6: Classification accuracy ──")
    class_results = evaluate_classification(
        test_df, predictions, main_label_col, sub_label_col,
    )
    print(f"  Main category accuracy: {class_results['main_accuracy']:.1%} "
          f"({class_results['main_correct']}/{class_results['total']})")
    if sub_label_col:
        print(f"  Sub category accuracy:  {class_results['sub_accuracy']:.1%} "
              f"({class_results['sub_correct']}/{class_results['total']})")
        print(f"  Both correct:           {class_results['both_accuracy']:.1%} "
              f"({class_results['both_correct']}/{class_results['total']})")

    # 7. Confusion report
    print("\n── Step 7: Per-category breakdown (worst first) ──")
    confusion = confusion_report(test_df, predictions, main_label_col)
    print(confusion.to_string(index=False))

    # 8. Diagnose failures
    diagnose_failures(
        test_df, predictions, retrieval_results["details"],
        main_label_col, n_show=10,
    )

    # save reports
    print(f"\n{'=' * 60}")

    return {
        "index_df": index_df,
        "test_df": test_df,
        "predictions": predictions,
        "retrieval": retrieval_results,
        "classification": class_results,
        "confusion": confusion,
    }
