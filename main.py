import pandas as pd
import sys
from pathlib import Path
from src.utils.loading import csv_loader, cast_column_type
from src.utils.text import text_section_parser, count_resolutions, extract_all_subfields
from src.utils.features import get_resolution_path, get_parts_used_flag

ROOT = Path(__file__).parent
OUTPUT_PATH = ROOT / "data" / "processed" / "cfr_savings_processed.parquet"



if __name__ == "__main__":

    raw_data = "services_cases_final.csv"

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

    print("\nSaving to parquet...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nProcessing complete. Rows processed: {len(df)} | Columns: {len(df.columns)}")