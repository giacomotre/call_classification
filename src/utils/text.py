import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pathlib import Path
from config import FIELD_REMARKS_SECTIONS, SEPARATOR, PARENT_PREFIX, SECTION_COLUMN_MAP




def text_section_parser(remark_field_string):
    
    column_dictionary = {
        "diagnostic_text": "", "diagnostic_date": "",
        "follow_up_text": "", "follow_up_date": "",
        "problem_description_text": "", "problem_description_date": "",
        "resolution_text": "", "resolution_date": "",
        "internal_comments_text": "", "internal_comments_date": "",
        "internal_remarks_text": "", "internal_remarks_date": "",
        "external_remarks_text": "", "external_remarks_date": "",
        "parent_diagnostic_text": "", "parent_diagnostic_date": "",
        "parent_follow_up_text": "", "parent_follow_up_date": "",
        "parent_problem_description_text": "", "parent_problem_description_date": "",
        "parent_resolution_text": "", "parent_resolution_date": "",
        "parent_internal_comments_text": "", "parent_internal_comments_date": "",
        "parent_internal_remarks_text": "", "parent_internal_remarks_date": "",
        "parent_external_remarks_text": "", "parent_external_remarks_date": "",
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
    if not resolution_text:
        return None
    return resolution_text.count(SEPARATOR) + 1

if __name__ == "__main__":
    test_text = test_text = """*** Diagnostic performed by Engineer [2025-11-17 20:08:23]
    Mantenimiento Correctivo
    *** Resolution [2026-01-02 11:57:06]
    Se realiza cambio de parte.
    *** Parent Resolution [2025-12-31 15:26:22]
    Parent resolution text here."""

    result = text_section_parser(test_text)
    print(result)