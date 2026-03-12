import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pathlib import Path
from config import FIELD_REMARKS_SECTIONS, SEPARATOR, PARENT_PREFIX, SECTION_COLUMN_MAP




def text_section_parser(remark_field_string):
    column_dictionary = {"diagnostic_text": "","diagnostic_date": "",
                    "follow_up_text": "", "follow_up_date": "",
                    "problem_description_text": "", "problem_description_date": "",
                    "resolution_text": "", "resolution_date":"",
                    "internal_comments_text": "", "internal_comments_date": "" }
        
    for section in FIELD_REMARKS_SECTIONS:
        pattern = rf"\*\*\* {section} \[(.*?)\](.*?)(?=\*\*\*|$)"
        chunk = re.findall(pattern, remark_field_string, re.DOTALL)

        if not chunk:
            continue
        
        #creating the key of the dictionary
        base_key = SECTION_COLUMN_MAP[section]
        text_key = base_key + "_text"
        date_key = base_key + "_date"

    print(section)
    print(base_key)
        


    
    return column_dictionary

if __name__ == "__main__":
    test_text = """*** Diagnostic performed by Engineer [2025-11-17 20:08:23]
Mantenimiento Correctivo
- Se realiza inspeccion y evaluación de alineación de la mesa de pacientes.
*** Resolution [2026-01-02 11:57:06]
Se realiza cambio de parte de acuerdo a protocolos de fábrica.
*** Resolution [2026-01-09 16:07:35]
Al realizar el reemplazo el ruido desaparece."""

    result = text_section_parser(test_text)
    print(result)