"""
Taxonomy definitions for AI classification.

Loads the NAM label Excel file and builds a clean taxonomy
that can be inserted into LLM prompts.

Usage:
    from classification.taxonomy import load_taxonomy, format_taxonomy_for_prompt
    
    taxonomy = load_taxonomy("data/raw/nam_label.xlsx")
    prompt_text = format_taxonomy_for_prompt(taxonomy)
"""
import pandas as pd


def load_taxonomy(path):
    """
    Load the NAM label Excel and extract the category hierarchy.

    Returns a dict: {main_category: [sub_categories]}
    
    Example:
        {
            "coil": ["analog_non_synergy", "analog_synergy", "digital", "dstream", ...],
            "magnet": ["magnet", "coldhead", "rmmu", "meu", ...],
            ...
        }
    """
    df = pd.read_excel(path)

    taxonomy = {}

    for _, row in df[["category_main_code", "category_main"]].dropna().iterrows():
        code = int(row["category_main_code"])
        main_name = row["category_main"]
        code_str = f"{code:02d}"

        # find the sub-category column for this main code
        code_cols = [c for c in df.columns
                     if c.startswith(f"{code_str}_") and c.endswith("_code")]
        if not code_cols:
            taxonomy[main_name] = []
            continue

        label_col = code_cols[0].replace("_code", "")

        # get all non-null sub-category values
        subs = df[label_col].dropna().tolist()
        taxonomy[main_name] = subs

    return taxonomy


def format_taxonomy_for_prompt(taxonomy):
    """
    Format the taxonomy as a readable string for LLM prompts.
    
    Output:
        MAIN CATEGORY: coil
          Sub-categories: analog_non_synergy, analog_synergy, digital, dstream, ...
        
        MAIN CATEGORY: magnet
          Sub-categories: magnet, coldhead, rmmu, meu, ...
        ...
    """
    lines = []
    for main_cat, sub_cats in taxonomy.items():
        lines.append(f"MAIN CATEGORY: {main_cat}")
        if sub_cats:
            lines.append(f"  Sub-categories: {', '.join(sub_cats)}")
        else:
            lines.append(f"  Sub-categories: (none defined)")
        lines.append("")

    return "\n".join(lines)
