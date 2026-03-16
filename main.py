import pandas as pd
import sys
from pathlib import Path
from src.utils.loading import csv_loader, cast_column_type
from src.utils.text import text_section_parser

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
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Processing complete. Rows processed: {len(df)}")