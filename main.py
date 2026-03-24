import pandas as pd
import sys
from pathlib import Path
from src.utils.loading import csv_loader, cast_column_type
from src.utils.text import text_section_parser, count_resolutions
from src.utils.features import get_resolution_path, get_parts_used_flag

ROOT = Path(__file__).parent
OUTPUT_PATH = ROOT / "data" / "processed" / "cfr_savings_processed.parquet"



if __name__ == "__main__":

    raw_data = "cfr_savings_2026.csv"

    df = csv_loader(raw_data)
    if df is None:
        print("Pipeline stopped: could not load data.")
        sys.exit(1)
    df = cast_column_type(df)
    parsed = df["field_remarks"].apply(text_section_parser)
    parsed_df = pd.DataFrame(parsed.tolist())
    df = pd.concat([df, parsed_df], axis=1)
    df["resolution_count"] = df["resolution_text"].apply(count_resolutions)
    df["resolution_path"] = df["field_remarks"].apply(get_resolution_path)
    df["parts_used_flag"] = df["parts_consumed_list"].apply(get_parts_used_flag)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Processing complete. Rows processed: {len(df)}")