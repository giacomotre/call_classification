import pandas as pd
from pathlib import Path
from config import COL_TYPES, DATE_COLS, KEEP_COLS, RENAME_COLS

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data" / "raw"

# --- Loading the csv and selecting columns ---
def csv_loader(filename):
    path = DATA_DIR / filename
    try:
        df_raw = pd.read_csv(path, encoding="latin1", na_values=["-"])
        df_raw = df_raw[KEEP_COLS]
        df_raw = df_raw.rename(columns=RENAME_COLS)
        return df_raw
    except FileNotFoundError:
        print(f"No file found in {path}")

    
# --- cast column types ---
def cast_column_type(df, col_types=COL_TYPES, date_cols=DATE_COLS):
     df = df.astype(col_types)
     
     for col, fmt in date_cols.items():
          df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")
          
     return df

