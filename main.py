import pandas as pd
import sys
from pathlib import Path
from src.utils.loading import csv_loader, cast_column_type
from src.utils.text import text_section_parser, count_resolutions, extract_all_subfields
from src.utils.features import get_resolution_path, get_parts_used_flag
from src.utils.nam_labels import load_label_table, apply_nam_labels
from sentence_transformers import SentenceTransformer
from src.classification.taxonomy import load_taxonomy, format_taxonomy_for_prompt
from src.classification.retriever import build_index
from src.classification.classifier import classify_case

ROOT = Path(__file__).parent
OUTPUT_PATH = ROOT / "data" / "processed" / "cfr_savings_processed.parquet"



if __name__ == "__main__":

    raw_data = "services_cases_final_v2.csv"

    print("Loading data...")
    df = csv_loader(raw_data)
    if df is None:
        print("Pipeline stopped: could not load data.")
        sys.exit(1)
    df = cast_column_type(df)
    print(f"Loaded {len(df)} rows.")

    # Layer 1 — field remarks
    print("\nParsing Field Remarks (Layer 1)...")
    parsed_field = df["field_remarks_en"].apply(
        lambda x: text_section_parser(x) if pd.notna(x) else {})
    parsed_field_df = pd.DataFrame(parsed_field.tolist())
    df = pd.concat([df, parsed_field_df], axis=1)
    print("Done.")

    # Layer 1 — remote remarks
    print("\nParsing Remote Remarks (Layer 1)...")
    parsed_remote = df["remote_remarks_en"].apply(
        lambda x: text_section_parser(x) if pd.notna(x) else {})
    parsed_remote_df = pd.DataFrame(parsed_remote.tolist())
    parsed_remote_df = parsed_remote_df.add_suffix("_remote")
    df = pd.concat([df, parsed_remote_df], axis=1)
    print("Done.")

    # Layer 2 — sub-field extraction field remarks
    print("\nExtracting sub-fields from Field Remarks (Layer 2)...")
    extracted_field = df.apply(
        lambda row: extract_all_subfields(row, suffix=""),
        axis=1, result_type="expand")
    extracted_field = extracted_field.add_suffix("_field")
    df = pd.concat([df, extracted_field], axis=1)
    print("Done.")

    # Layer 2 — sub-field extraction remote remarks
    print("\nExtracting sub-fields from Remote Remarks (Layer 2)...")
    extracted_remote = df.apply(
        lambda row: extract_all_subfields(row, suffix="_remote"),
        axis=1, result_type="expand")
    extracted_remote = extracted_remote.add_suffix("_remote")
    df = pd.concat([df, extracted_remote], axis=1)
    print("Done.")

    #Metadata features
    print("\nComputing metadata features...")
    df["resolution_count"] = df["resolution_text"].apply(count_resolutions)
    df["resolution_path"] = df["field_remarks"].apply(get_resolution_path)
    df["parts_used_flag"] = df["parts_consumed_list"].apply(get_parts_used_flag)
    print("Done.")

    # NAM label Extraction     
    label_table = load_label_table("data/raw/nam_label.xlsx")
    df = apply_nam_labels(df, label_table)

    print("\nSaving to parquet...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nProcessing complete. Rows processed: {len(df)} | Columns: {len(df.columns)}")

        # ── Layer 3 — AI Classification (evaluation mode) ──────────────────
    #
    # Evaluates classification accuracy on NAM-labeled cases:
    #   1. Build enriched text columns (retrieval text + LLM context)
    #   2. Split labeled cases into index (80%) and test (20%)
    #   3. Retrieve similar cases using enriched retrieval text
    #   4. Classify test cases using LLM with rich context
    #   5. Report accuracy and save results
    #
    # Config: all parameters in config.py under CLASSIFICATION section.
    # ──────────────────────────────────────────────────────────────────────

    from sentence_transformers import SentenceTransformer
    from src.classification.taxonomy import load_taxonomy, format_taxonomy_for_prompt
    from src.classification.text_builder import build_retrieval_text, build_llm_context
    from src.classification.retriever import build_index, retrieve_batch
    from src.classification.classifier import classify_batch
    from src.utils.case_handler import compute_info_score
    from src.classification.evaluation import (
        split_labeled_data, evaluate_retrieval, evaluate_classification,
        confusion_report, diagnose_failures,
    )
    from config import (
        EMBEDDING_MODEL_PATH, TAXONOMY_PATH, LABEL_COLS,
        MAIN_LABEL_COL, SUB_LABEL_COL,
        N_RETRIEVAL_EXAMPLES, RETRIEVAL_BATCH_SIZE, TEST_SIZE,
    )

    print("\n" + "=" * 60)
    print("  LAYER 3 — AI CLASSIFICATION (evaluation)")
    print("=" * 60)

    # ── 1. Build enriched text columns ──
    print("\nBuilding enriched text columns...")
    df["retrieval_text"] = df.apply(build_retrieval_text, axis=1)
    df["llm_context"] = df.apply(build_llm_context, axis=1)

    filled_ret = df["retrieval_text"].notna().sum()
    filled_llm = df["llm_context"].notna().sum()
    print(f"  retrieval_text: {filled_ret}/{len(df)} rows")
    print(f"  llm_context:    {filled_llm}/{len(df)} rows")

    # ── 2. Load taxonomy + embedding model ──
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    taxonomy_text = format_taxonomy_for_prompt(taxonomy)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH)

    # ── 2.5 Filter low-information cases ──
    print("\nFiltering low-information cases...")

    # compute information score
    df["info_score"] = df.apply(compute_info_score, axis=1)

    # create mask
    info_mask = df["info_score"] >= 40   # <-- threshold (tune later if needed)

    # split dataset
    df_valid = df[info_mask].copy()
    df_invalid = df[~info_mask].copy()

    print(f" Valid cases: {len(df_valid)}")
    print(f" Low-information cases: {len(df_invalid)}")

    # store invalid cases for review
    review_cols = [
        "case_number",
        "info_score",
        "remote_remarks_en",
        "extracted_problem_description_remote",
        "extracted_error_remote",
        "extracted_malfunction_area_remote",
        "extracted_diagnostic_remote",
    ]

    review_path = ROOT / "data" / "reports" / "low_information_cases.csv"
    review_path.parent.mkdir(parents=True, exist_ok=True)

    df_invalid[review_cols].to_csv(review_path, index=False)
    print(f"Rejected cases: {len(df_invalid)} ({len(df_invalid)/len(df):.1%})")
    print(f" Low-information cases saved to: {review_path}")


    # ── 3. Filter to labeled cases with text ──
    nam_labeled = df_valid[
        df[MAIN_LABEL_COL].notna() &
        df["retrieval_text"].notna() &
        df["llm_context"].notna()
    ].copy()
    print(f"\nLabeled cases with text: {len(nam_labeled)}")

    # ── 4. Split: 80% for index, 20% for testing ──
    index_df, test_df = split_labeled_data(nam_labeled, test_size=TEST_SIZE)

    # ── 5. Build or load retrieval index ──

    # Embedding ~1,000 labeled cases takes a few minutes.
    # After the first run, the index is saved to disk:
    #   data/processed/retrieval_index/embeddings.npy    (dense vectors)
    #   data/processed/retrieval_index/labeled_cases.csv  (texts + labels)
    #
    # Subsequent runs load these files instantly, skipping the embedding step.
    #
    # The cache is ONLY valid if the underlying data hasn't changed.
    # Re-build when: labeled cases change, retrieval fields change (config),
    # or the embedding model changes.
    #
    # To force a fresh build, either:
    #   - delete the files (or the folder) at data/processed/retrieval_index/
    #   - run:  python main.py --rebuild-index

    from src.classification.retriever import save_index, load_index
    from config import INDEX_CACHE_PATH

    index_cache = ROOT / INDEX_CACHE_PATH
    index_ready = (index_cache / "embeddings.npy").exists() and (index_cache / "labeled_cases.csv").exists()
    if index_ready and "--rebuild-index" not in sys.argv:
        print("\nLoading cached retrieval index...")
        index = load_index(index_cache)
    else:
        print("\nBuilding retrieval index...")
        index = build_index(
            index_df,
            text_col="retrieval_text",
            label_cols=LABEL_COLS,
            embedding_model=embedding_model,
            batch_size=RETRIEVAL_BATCH_SIZE,
            extra_cols=["llm_context"],
        )
        save_index(index, index_cache)

    # ── 6. Retrieve examples for test cases ──
    print(f"\nRetrieving {N_RETRIEVAL_EXAMPLES} examples per test case...")
    test_retrieval_texts = test_df["retrieval_text"].tolist()
    all_examples = retrieve_batch(
        test_retrieval_texts, index, embedding_model,
        n=N_RETRIEVAL_EXAMPLES, batch_size=RETRIEVAL_BATCH_SIZE,
    )

    # ── 7. Evaluate retrieval quality ──
    retrieval_results = evaluate_retrieval(
        test_df, all_examples, "retrieval_text", MAIN_LABEL_COL,
    )
    print(f"\n── Retrieval Quality ──")
    print(f"  Majority vote accuracy: {retrieval_results['majority_accuracy']:.1%}")
    print(f"  Any correct in top-{N_RETRIEVAL_EXAMPLES}:   {retrieval_results['any_correct_rate']:.1%}")

    # ── 8. Classify test cases via LLM ──
    test_contexts = test_df["llm_context"].tolist()
    print(f"\n── Classifying {len(test_contexts)} test cases via LLM ──")
    predictions = classify_batch(test_contexts, all_examples, taxonomy_text)

    # ── 9. Evaluate classification accuracy ──
    class_results = evaluate_classification(
        test_df, predictions, MAIN_LABEL_COL, SUB_LABEL_COL,
    )
    print(f"\n── Classification Accuracy ──")
    print(f"  Main category: {class_results['main_accuracy']:.1%} "
          f"({class_results['main_correct']}/{class_results['total']})")
    print(f"  Sub category:  {class_results['sub_accuracy']:.1%} "
          f"({class_results['sub_correct']}/{class_results['total']})")
    print(f"  Both correct:  {class_results['both_accuracy']:.1%} "
          f"({class_results['both_correct']}/{class_results['total']})")

    # ── 10. Per-category breakdown (worst first) ──
    print(f"\n── Per-Category Breakdown ──")
    confusion = confusion_report(test_df, predictions, MAIN_LABEL_COL)
    print(confusion.to_string(index=False))

    # ── 11. Diagnose failures ──
    diagnose_failures(
        test_df, predictions, retrieval_results["details"],
        MAIN_LABEL_COL, n_show=10,
    )

    # ── 12. Save results ──
    # Include retrieval majority label for failure analysis in evaluation notebook
    from collections import Counter

    retrieval_majority_labels = []
    for examples in all_examples:
        labels = [ex["labels"][MAIN_LABEL_COL] for ex in examples]
        majority = Counter(labels).most_common(1)[0][0]
        retrieval_majority_labels.append(majority)

    test_output = test_df[["retrieval_text", "llm_context",
                           MAIN_LABEL_COL, SUB_LABEL_COL]].copy()
    test_output["predicted_main"] = predictions["main_category"].values
    test_output["predicted_sub"] = predictions["sub_category"].values
    test_output["retrieval_majority_label"] = retrieval_majority_labels
    test_output["main_correct"] = test_output[MAIN_LABEL_COL] == test_output["predicted_main"]
    test_output["sub_correct"] = test_output[SUB_LABEL_COL] == test_output["predicted_sub"]

    test_output_path = ROOT / "data" / "reports" / "classification_test_results.csv"
    test_output_path.parent.mkdir(parents=True, exist_ok=True)
    test_output.to_csv(test_output_path, index=False)
    print(f"\nTest results saved to {test_output_path}")
