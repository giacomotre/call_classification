"""
NAM Label Extraction.

Extracts structured labels from NAM market subject codes.
The first 6 characters of the subject line encode three levels:
  - chars 1-2: main category code   (e.g., "11" → softwares)
  - chars 3-4: sub-category code    (e.g., "08" → not_all_scan_resources_avail)
  - chars 5-6: event type code      (e.g., "CF" → corrective_maintenance_remote_fix)

Usage in main.py:
    from src.utils.nam_labels import load_label_table, extract_nam_labels
    
    label_table = load_label_table("data/raw/nam_label.xlsx")
    nam_mask = df["market"] == "NAM"
    df.loc[nam_mask, ["nam_main_category", "nam_sub_category", "nam_event_type"]] = (
        df.loc[nam_mask, "subject"].apply(lambda x: extract_nam_labels(x, label_table))
    )
"""
import pandas as pd


def load_label_table(path):
    """Load the NAM lookup table once. Pass the result to extract functions."""
    return pd.read_excel(path)


def parse_code(subject):
    """
    Extract the 6-character code from a NAM subject line.

    Returns (main_code, sub_code, event_code) as strings,
    or None if the subject doesn't start with 2 digits.

    Example:
        "1108CF System Not Scanning" → ("11", "08", "CF")
        "System down" → None
    """
    if pd.isna(subject) or not isinstance(subject, str) or len(subject) < 6:
        return None
    if not (subject[0].isdigit() and subject[1].isdigit()):
        return None

    main_code = subject[0:2]
    sub_code = subject[2:4]
    event_code = subject[4:6]

    return main_code, sub_code, event_code


def lookup_main_category(main_code, label_table):
    """Look up main category: '11' → 'softwares'."""
    match = label_table[label_table["category_main_code"] == int(main_code)]
    if match.empty:
        return None
    return match["category_main"].values[0]


def lookup_sub_category(main_code, sub_code, label_table):
    """
    Look up sub category: '11', '08' → 'not_all_scan_resources_avail'.

    Finds the right columns by prefix pattern (e.g., '11_software_code')
    because column names don't always match category_main exactly.
    """
    # find columns by prefix: "11_*_code" and "11_*"
    code_cols = [c for c in label_table.columns
                 if c.startswith(f"{main_code}_") and c.endswith("_code")]
    if not code_cols:
        return None

    code_col = code_cols[0]
    label_col = code_col.replace("_code", "")

    match = label_table[label_table[code_col] == int(sub_code)]
    if match.empty:
        return None
    return match[label_col].values[0]


def lookup_event_type(event_code, label_table):
    """Look up event type: 'CF' → 'corrective_maintenance_remote_fix'."""
    match = label_table[label_table["event_type_code"] == event_code.lower()]
    if match.empty:
        return None
    return match["event_type"].values[0]


def extract_nam_labels(subject, label_table):
    """
    Extract all three label levels from a NAM subject code.

    Returns a pd.Series with three values so it can be used with .apply()
    to create multiple columns at once.

    Example:
        extract_nam_labels("1108CF System Not Scanning", label_table)
        → pd.Series({
            "nam_main_category": "softwares",
            "nam_sub_category": "not_all_scan_resources_avail",
            "nam_event_type": "corrective_maintenance_remote_fix"
          })
    """
    empty = pd.Series({
        "nam_main_category": None,
        "nam_sub_category": None,
        "nam_event_type": None,
    })

    codes = parse_code(subject)
    if codes is None:
        return empty

    main_code, sub_code, event_code = codes

    main_category = lookup_main_category(main_code, label_table)
    sub_category = lookup_sub_category(main_code, sub_code, label_table)
    event_type = lookup_event_type(event_code, label_table)

    return pd.Series({
        "nam_main_category": main_category,
        "nam_sub_category": sub_category,
        "nam_event_type": event_type,
    })

#wrapper function
def apply_nam_labels(df, label_table):
    """
    Apply NAM label extraction to a dataframe.
    
    Filters for NAM market, extracts labels, and prints a summary.
    Adds three columns: nam_main_category, nam_sub_category, nam_event_type.
    
    Returns the dataframe with new columns added.
    """
    nam_mask = df["market"] == "NAM"
    nam_count = nam_mask.sum()
    print(f"  NAM cases found: {nam_count}")

    if nam_count == 0:
        df["nam_main_category"] = None
        df["nam_sub_category"] = None
        df["nam_event_type"] = None
        return df

    df.loc[nam_mask, ["nam_main_category", "nam_sub_category", "nam_event_type"]] = (
        df.loc[nam_mask, "subject"].apply(lambda x: extract_nam_labels(x, label_table))
    )

    # Stats
    has_code = df.loc[nam_mask, "nam_main_category"].notna().sum()
    has_sub = df.loc[nam_mask, "nam_sub_category"].notna().sum()
    has_event = df.loc[nam_mask, "nam_event_type"].notna().sum()
    no_code = nam_count - has_code

    print(f"  Parsed successfully: {has_code}/{nam_count} ({100*has_code/nam_count:.1f}%)")
    print(f"  No valid code:      {no_code}/{nam_count} ({100*no_code/nam_count:.1f}%)")
    print(f"  Main category found: {has_code} | Sub category found: {has_sub} | Event type found: {has_event}")

    return df