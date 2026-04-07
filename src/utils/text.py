import re
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (FIELD_REMARKS_SECTIONS, SEPARATOR, PARENT_PREFIX, 
                    SECTION_COLUMN_MAP, PROBLEM_SUBFIELD_PATTERNS,
                    ERROR_SUBFIELD_PATTERNS, MALFUNCTION_SUBFIELD_PATTERNS,
                    TROUBLESHOOTING_SUBFIELD_PATTERNS, REPAIR_ACTION_SUBFIELD_PATTERNS,
                    EXTRACTED_COLUMNS, BOILERPLATE_STRIP)


# ── Markers that signal "form fields start here" ─────────────────────
# Used by truncate_at_boilerplate() to cut off fallback text.
# These appear in the intermediate columns AFTER the actual content.

FALLBACK_CUTOFF_MARKERS = [
    "Information to support the complaint handling process",
    "Customer Function/Role",
    "How was the device being used",
    "When was the first occurrence",
    "How often is issue occurring",
    "Where in the end-user workflow",
    "What recent events have occurred",
    "Has the customer created a specific system event report",
    "Expected and actual behavior of the product",
    "User Impact:",
    "Patient Impact:",
    "Current Software Version",
]


def text_section_parser(remark_field_string):
    
    column_dictionary = {
        "diagnostic_text": None, "diagnostic_date": None,
        "follow_up_text": None, "follow_up_date": None,
        "problem_description_text": None, "problem_description_date": None,
        "resolution_text": None, "resolution_date": None,
        "internal_comments_text": None, "internal_comments_date": None,
        "internal_remarks_text": None, "internal_remarks_date": None,
        "external_remarks_text": None, "external_remarks_date": None,
        "parent_diagnostic_text": None, "parent_diagnostic_date": None,
        "parent_follow_up_text": None, "parent_follow_up_date": None,
        "parent_problem_description_text": None, "parent_problem_description_date": None,
        "parent_resolution_text": None, "parent_resolution_date": None,
        "parent_internal_comments_text": None, "parent_internal_comments_date": None,
        "parent_internal_remarks_text": None, "parent_internal_remarks_date": None,
        "parent_external_remarks_text": None, "parent_external_remarks_date": None,
        "t2_activities_text": None, "t2_activities_date": None,
        "onems_internal_text": None, "onems_internal_date": None,
        "pfq_malfunction_text": None, "pfq_malfunction_date": None,
        "parent_t2_activities_text": None, "parent_t2_activities_date": None,
        "parent_onems_internal_text": None, "parent_onems_internal_date": None,
        "parent_pfq_malfunction_text": None, "parent_pfq_malfunction_date": None,
    }
        
    for pattern_prefix, key_prefix in [("", ""), (PARENT_PREFIX, "parent_")]:
        for section in FIELD_REMARKS_SECTIONS:
            pattern = rf"\*\*\* {pattern_prefix}{section} \[(.*?)\](.*?)(?=\*\*\*|$)"
            chunk = re.findall(pattern, remark_field_string, re.DOTALL)

            if not chunk:
                continue

            base_key = key_prefix + SECTION_COLUMN_MAP[section]
            text_key = base_key + "_text"
            date_key = base_key + "_date"

            if len(chunk) > 1:
                all_texts = []
                all_dates = []
                for tuple in chunk:
                    all_texts.append(tuple[1].strip())
                    all_dates.append(tuple[0])
                column_dictionary[text_key] = SEPARATOR.join(all_texts)
                column_dictionary[date_key] = all_dates[-1]
            else:
                column_dictionary[date_key] = chunk[0][0]
                column_dictionary[text_key] = chunk[0][1].strip()
                
    return column_dictionary

def count_resolutions(resolution_text):
    if pd.isna(resolution_text):
        return None
    return resolution_text.count(SEPARATOR) + 1

def strip_boilerplate(text):
    for phrase in BOILERPLATE_STRIP:
        idx = text.find(phrase)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def truncate_at_boilerplate(text):
    """
    Cut intermediate-column text at the first form-template marker.

    The intermediate columns (problem_description_text, diagnostic_text)
    often contain real content followed by form fields like:
    
        Problem description by engineer :
        Chiller is not functioning              ← we want this
        
        How often is issue occurring? :         ← cut here
        First occurrence
        
        Malfunction area :                      ← form field
        Chiller
        
        Information to support the complaint... ← boilerplate block
        Customer Function/Role: ...

    This function finds the earliest cutoff marker and returns only
    the text before it. Used as a safer fallback than the raw column.
    """
    if pd.isna(text) or not isinstance(text, str):
        return None

    earliest_cut = len(text)
    for marker in FALLBACK_CUTOFF_MARKERS:
        idx = text.lower().find(marker.lower())
        if idx != -1 and idx < earliest_cut:
            earliest_cut = idx

    truncated = text[:earliest_cut].strip()
    return truncated if truncated else None


def extract_subfield(text, patterns):
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return None
    for pattern in patterns:
        regex = rf"{re.escape(pattern)}(.*?)(?=\n\S.{{0,60}}:|$)"
        match = re.search(regex, text, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            extracted = strip_boilerplate(extracted)
            return extracted if extracted else None
    return None

def extract_all_subfields(row, suffix=""):
    result = {col: None for col in EXTRACTED_COLUMNS}

    result["extracted_problem_description"] = (
        extract_subfield(row.get(f"problem_description_text{suffix}"), PROBLEM_SUBFIELD_PATTERNS) or
        extract_subfield(row.get(f"t2_activities_text{suffix}"), PROBLEM_SUBFIELD_PATTERNS) or
        extract_subfield(row.get(f"onems_internal_text{suffix}"), PROBLEM_SUBFIELD_PATTERNS) or
        truncate_at_boilerplate(row.get(f"problem_description_text{suffix}")) or
        truncate_at_boilerplate(row.get(f"diagnostic_text{suffix}")) or
        None
    )

    result["extracted_diagnostic"] = (
        truncate_at_boilerplate(row.get(f"diagnostic_text{suffix}")) or
        None
    )

    result["extracted_error"] = (
        extract_subfield(row.get(f"problem_description_text{suffix}"), ERROR_SUBFIELD_PATTERNS) or
        extract_subfield(row.get(f"internal_comments_text{suffix}"), ERROR_SUBFIELD_PATTERNS) or
        extract_subfield(row.get(f"follow_up_text{suffix}"), ERROR_SUBFIELD_PATTERNS) or
        None
    )

    result["extracted_malfunction_area"] = (
        extract_subfield(row.get(f"problem_description_text{suffix}"), MALFUNCTION_SUBFIELD_PATTERNS) or
        extract_subfield(row.get(f"t2_activities_text{suffix}"), MALFUNCTION_SUBFIELD_PATTERNS) or
        None
    )

    result["extracted_troubleshooting"] = (
        extract_subfield(row.get(f"t2_activities_text{suffix}"), TROUBLESHOOTING_SUBFIELD_PATTERNS) or
        extract_subfield(row.get(f"external_remarks_text{suffix}"), TROUBLESHOOTING_SUBFIELD_PATTERNS) or
        None
    )

    result["extracted_repair_action"] = (
        extract_subfield(row.get(f"t2_activities_text{suffix}"), REPAIR_ACTION_SUBFIELD_PATTERNS) or
        extract_subfield(row.get(f"external_remarks_text{suffix}"), REPAIR_ACTION_SUBFIELD_PATTERNS) or
        extract_subfield(row.get(f"internal_comments_text{suffix}"), REPAIR_ACTION_SUBFIELD_PATTERNS) or
        extract_subfield(row.get(f"problem_description_text{suffix}"), REPAIR_ACTION_SUBFIELD_PATTERNS) or
        extract_subfield(row.get(f"onems_internal_text{suffix}"), REPAIR_ACTION_SUBFIELD_PATTERNS) or
        row.get(f"resolution_text{suffix}") or
        None
    )

    return result

if __name__ == "__main__":
    test_text = """*** Problem Description by Engineer [2025-11-17 20:08:23]
Problem description by engineer :
Noise in patient table
Malfunction area :
Patient Support
*** Resolution [2026-01-02 11:57:06]
Engineer replaced the gradient board."""

    parsed = text_section_parser(test_text)
    print("--- Section parser ---")
    print(parsed)
    
    print("\n--- Sub-field extractor ---")
    import pandas as pd
    row = pd.Series(parsed)
    result = extract_all_subfields(row)
    print(result)