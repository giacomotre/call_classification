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

    # ── Layer 4 — AI Classification (test mode) ──────────────────────────
    from src.classification.classifier import classify_batch
    from src.classification.evaluation import (
        split_labeled_data, evaluate_retrieval, evaluate_classification,
        confusion_report, diagnose_failures,
    )
    from src.classification.retriever import retrieve_batch

    # load taxonomy
    taxonomy = load_taxonomy("data/raw/nam_label.xlsx")
    taxonomy_text = format_taxonomy_for_prompt(taxonomy)

    # load embedding model (same one used for retrieval evaluation)
    embedding_model = SentenceTransformer("models/bge-base-en-v1.5")

    # filter to labeled cases with text
    nam_labeled = df[
        df["nam_main_category"].notna() &
        df["extracted_problem_description_remote"].notna()
    ].copy()
    print(f"\nLabeled cases with text: {len(nam_labeled)}")

    # split: 80% for index, 20% for testing
    index_df, test_df = split_labeled_data(nam_labeled, test_size=0.2)

    # build retrieval index from the 80%
    index = build_index(
        index_df,
        text_col="extracted_problem_description_remote",
        label_cols=["nam_main_category", "nam_sub_category"],
        embedding_model=embedding_model,
    )

    # retrieve examples for test cases (for retrieval diagnostics)
    test_texts = test_df["extracted_problem_description_remote"].tolist()
    all_examples = retrieve_batch(test_texts, index, embedding_model, n=5)

    # evaluate retrieval quality
    retrieval_results = evaluate_retrieval(
        test_df, all_examples,
        "extracted_problem_description_remote", "nam_main_category",
    )
    print(f"\n── Retrieval Quality ──")
    print(f"  Majority vote accuracy: {retrieval_results['majority_accuracy']:.1%}")
    print(f"  Any correct in top-5:   {retrieval_results['any_correct_rate']:.1%}")

    # classify the 20% test set via LLM
    print(f"\n── Classifying {len(test_texts)} test cases via LLM ──")
    predictions = classify_batch(
        test_texts, index, embedding_model, taxonomy_text, n_examples=5,
    )

    # evaluate classification accuracy
    class_results = evaluate_classification(
        test_df, predictions,
        main_label_col="nam_main_category",
        sub_label_col="nam_sub_category",
    )
    print(f"\n── Classification Accuracy ──")
    print(f"  Main category: {class_results['main_accuracy']:.1%} "
        f"({class_results['main_correct']}/{class_results['total']})")
    print(f"  Sub category:  {class_results['sub_accuracy']:.1%} "
        f"({class_results['sub_correct']}/{class_results['total']})")
    print(f"  Both correct:  {class_results['both_accuracy']:.1%} "
        f"({class_results['both_correct']}/{class_results['total']})")

    # per-category breakdown (worst first)
    print(f"\n── Per-Category Breakdown ──")
    confusion = confusion_report(test_df, predictions, "nam_main_category")
    print(confusion.to_string(index=False))

    # diagnose failures: retrieval vs LLM
    diagnose_failures(
        test_df, predictions, retrieval_results["details"],
        "nam_main_category", n_show=10,
    )

    # save predictions alongside true labels for manual inspection
    test_output = test_df[["extracted_problem_description_remote",
                            "nam_main_category", "nam_sub_category"]].copy()
    test_output["predicted_main"] = predictions["main_category"].values
    test_output["predicted_sub"] = predictions["sub_category"].values
    test_output["main_correct"] = test_output["nam_main_category"] == test_output["predicted_main"]
    test_output["sub_correct"] = test_output["nam_sub_category"] == test_output["predicted_sub"]

    test_output_path = ROOT / "data" / "reports" / "classification_test_results.csv"
    test_output_path.parent.mkdir(parents=True, exist_ok=True)
    test_output.to_csv(test_output_path, index=False)
    print(f"\nTest results saved to {test_output_path}")