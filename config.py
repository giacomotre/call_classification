# --- Data type raw file --- 

COL_TYPES = {
    "case_number": "int64",
    "subject": "string",
    'field_remarks': "string"
    }

DATE_COLS = {
    "creation_data": "%Y-%m-%d",
    "disposition_date" :"%Y-%m-%d"
    }

# --- List on field_remarks section ---

FIELD_REMARKS_SECTIONS = [
    "Diagnostic performed by Engineer",
    "Follow Up Required",
    "Problem Description by Engineer",
    "Resolution",
    "Internal Comments"]

SECTION_COLUMN_MAP = {
    "Diagnostic performed by Engineer": "diagnostic",
    "Follow Up Required": "follow_up",
    "Problem Description by Engineer": "problem_description",
    "Resolution": "resolution",
    "Internal Comments": "internal_comments"
}

SEPARATOR = " [SEP] "

PARENT_PREFIX = "Parent "