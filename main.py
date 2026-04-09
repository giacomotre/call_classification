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

    # ── Filter for topic modeling ─────────────────────────────────────
    # Remove pure remote-only cases (no field visit AND no parts used).
    # These are typically simple restarts that add noise to the model.

    before = len(df)
    # AFTER (keeps remote-only and US):

    topic_df = df[
    (df["field_remarks"].isna() | (df["field_remarks"].astype(str).str.strip() == "")) &
    (df["parts_consumed_list"].isna()) &
    (df["country"].isin(["United States", "Canada"])) # "India", "Ireland", "United Kingdom", 
    ].copy()
    print(f"\nFiltered for topic modeling: {before} → {len(topic_df)} rows")
    print(f"  Removed {before - len(topic_df)} field visit / parts consumed cases")

    # ── Layer 3 — Topic Modeling ──────────────────────────────────────
    #
    # Runs two BERTopic models (problem + resolution) on the extracted
    # text fields. First run computes embeddings (~5-10 min), subsequent
    # runs use cached embeddings and are much faster.
    #
    # To skip topic modeling, comment out or use: --skip-topics flag
    # To re-run with different params, edit topic_config.py
    # ──────────────────────────────────────────────────────────────────

    run_topics = "--skip-topics" not in sys.argv

    if run_topics:
        print("\n" + "=" * 60)
        print("  LAYER 3 — TOPIC MODELING")
        print("=" * 60)

        from src.topic_modeling.text_preparation import prepare_documents, prepare_problem_documents
        from src.topic_modeling.bertopic_wrapper import (
            build_topic_model, compute_embeddings,
            save_embeddings, load_embeddings,
            fit_model, get_topic_summary, save_model,
        )
        from src.topic_modeling.validation import validate_model, save_report
        from topic_config import TopicModelConfig, MODELS_DIR, REPORTS_DIR, EMBEDDINGS_DIR

        cfg = TopicModelConfig()

        # ── Problem model ──
        print("\nPreparing Problem documents...")
        problem_docs_df = prepare_problem_documents(topic_df, cfg)
        problem_docs = problem_docs_df["doc_text"].tolist()
        print(f"  {len(df)} rows → {len(problem_docs)} valid problem documents")

        print("\nBuilding Problem topic model...")
        problem_model, problem_emb_model = build_topic_model(cfg, seed_topic_list=cfg.problem_seed_topics)

        emb_path = EMBEDDINGS_DIR / "problem_embeddings.npy"
        if emb_path.exists():
            print("Loading cached problem embeddings...")
            problem_embeddings = load_embeddings(emb_path)
            if len(problem_embeddings) != len(problem_docs):
                print("  Cache mismatch — recomputing...")
                problem_embeddings = compute_embeddings(
                    problem_docs, problem_emb_model,
                    cfg.embedding.batch_size, cfg.embedding.show_progress)
                save_embeddings(problem_embeddings, emb_path)
        else:
            print("Computing problem embeddings (first run, takes a few minutes)...")
            problem_embeddings = compute_embeddings(
                problem_docs, problem_emb_model,
                cfg.embedding.batch_size, cfg.embedding.show_progress)
            if cfg.save_embeddings:
                save_embeddings(problem_embeddings, emb_path)

        print("\nTraining Problem topic model...")
        problem_topics, problem_probs = fit_model(problem_model, problem_docs, problem_embeddings)

        print("\nValidating Problem model...")
        problem_results = validate_model(problem_model, problem_docs, problem_topics, problem_probs)
        save_report(problem_results, REPORTS_DIR / "problem")

        problem_summary = get_topic_summary(problem_model)
        problem_summary.to_csv(REPORTS_DIR / "problem" / "topic_terms.csv", index=False)
        save_model(problem_model, MODELS_DIR / "problem_model")

        # ── Resolution model ──
        print("\nPreparing Resolution documents...")
        resolution_docs_df = prepare_documents(
            topic_df,
            columns=cfg.text_prep.resolution_columns,
            separator=cfg.text_prep.separator,
            min_length=cfg.text_prep.min_doc_length,
            min_words=cfg.text_prep.min_word_count,
        )
        resolution_docs = resolution_docs_df["doc_text"].tolist()
        print(f"  {len(df)} rows → {len(resolution_docs)} valid resolution documents")

        print("\nBuilding Resolution topic model...")
        resolution_model, resolution_emb_model = build_topic_model(cfg, seed_topic_list=cfg.resolution_seed_topics)

        emb_path = EMBEDDINGS_DIR / "resolution_embeddings.npy"
        if emb_path.exists():
            print("Loading cached resolution embeddings...")
            resolution_embeddings = load_embeddings(emb_path)
            if len(resolution_embeddings) != len(resolution_docs):
                print("  Cache mismatch — recomputing...")
                resolution_embeddings = compute_embeddings(
                    resolution_docs, resolution_emb_model,
                    cfg.embedding.batch_size, cfg.embedding.show_progress)
                save_embeddings(resolution_embeddings, emb_path)
        else:
            print("Computing resolution embeddings (first run, takes a few minutes)...")
            resolution_embeddings = compute_embeddings(
                resolution_docs, resolution_emb_model,
                cfg.embedding.batch_size, cfg.embedding.show_progress)
            if cfg.save_embeddings:
                save_embeddings(resolution_embeddings, emb_path)

        print("\nTraining Resolution topic model...")
        resolution_topics, resolution_probs = fit_model(
            resolution_model, resolution_docs, resolution_embeddings)

        print("\nValidating Resolution model...")
        resolution_results = validate_model(
            resolution_model, resolution_docs, resolution_topics, resolution_probs)
        save_report(resolution_results, REPORTS_DIR / "resolution")

        resolution_summary = get_topic_summary(resolution_model)
        resolution_summary.to_csv(REPORTS_DIR / "resolution" / "topic_terms.csv", index=False)
        save_model(resolution_model, MODELS_DIR / "resolution_model")

        # ── Map topics back to dataframe ──
        print("\nMapping topics back to dataset...")
        import numpy as np

        for prefix, docs_df, topics, probs in [
            ("problem", problem_docs_df, problem_topics, problem_probs),
            ("resolution", resolution_docs_df, resolution_topics, resolution_probs),
        ]:
            df[f"{prefix}_topic_id"] = -1
            df[f"{prefix}_confidence"] = 0.0

            probs_array = np.array(probs)
            max_probs = probs_array.max(axis=1) if probs_array.ndim == 2 else probs_array

            for i, row in docs_df.iterrows():
                df.at[row["doc_index"], f"{prefix}_topic_id"] = topics[i]
                df.at[row["doc_index"], f"{prefix}_confidence"] = float(max_probs[i])

            assigned = (df[f"{prefix}_topic_id"] != -1).sum()
            print(f"  {prefix}: {assigned}/{len(df)} rows assigned to a topic")

        # ── Save enriched parquet ──
        enriched_path = ROOT / "data" / "processed" / "cfr_savings_with_topics.parquet"
        df.to_parquet(enriched_path, index=False)
        print(f"\nEnriched dataset saved to {enriched_path}")
        print(f"New columns: problem_topic_id, problem_confidence, "
              f"resolution_topic_id, resolution_confidence")

        print(f"\n{'='*60}")
                # ── Save case-level topic assignments ──
        case_topics = df[["case_number", 
                          "problem_topic_id", "problem_confidence",
                          "resolution_topic_id", "resolution_confidence"]].copy()
        case_topics_path = ROOT / "data" / "reports" / "case_topic_assignments.csv"
        case_topics.to_csv(case_topics_path, index=False)
        print(f"Case-level assignments saved to {case_topics_path}")
        print(f"  TOPIC MODELING COMPLETE")
        print(f"  Problem topics:    {len(set(problem_topics) - {-1})}")
        print(f"  Resolution topics: {len(set(resolution_topics) - {-1})}")
        print(f"  Reports: data/reports/")
        print(f"  Models:  data/models/")
        print(f"{'='*60}")
    else:
        print("\nTopic modeling skipped (--skip-topics flag detected).")


    # ── Layer 4 — AI Classification ──
    
    # load taxonomy
    taxonomy = load_taxonomy("data/raw/nam_label.xlsx")
    taxonomy_text = format_taxonomy_for_prompt(taxonomy)

    # load embedding model (same one used in BERTopic)
    embedding_model = SentenceTransformer("models/all-MiniLM-L6-v2")

    # build retrieval index from NAM labeled cases
    nam_labeled = df[df["nam_main_category"].notna()]
    index = build_index(
        nam_labeled,
        text_col="extracted_problem_description_remote",
        label_cols=["nam_main_category", "nam_sub_category"],
        embedding_model=embedding_model,
    )

    # classify a single case (test)
    test_text = "chiller temp high, cooling water out of spec"
    result = classify_case(test_text, index, embedding_model, taxonomy_text)
    print(result)