# --- Data type raw file --- 

# --- Column renaming ---
RENAME_COLS = {
    "Case Number":         "case_number",
    "Case Prio":           "case_priority",
    "Case Type":           "case_type",
    "Subject":             "subject",
    "Creation Date":       "creation_date",
    "Disposition Date":    "disposition_date",
    "TECO Date":           "teco_date",
    "Market":              "market",
    "Country":             "country",
    "IB Service Team":     "ib_service_team",
    "Remote Remarks":      "remote_remarks",
    "Field Remarks":       "field_remarks",
    "Equipment":           "equipment",
    "System Code":         "system_code",
    "Parts Consumed List": "parts_consumed_list",
    "Remote Hours":        "remote_hours",
    "Travel Hours":        "travel_hours",
    "Onsite Hours":        "onsite_hours",
    "Offsite Hours":       "offsite_hours",
    "Total Hours":         "total_hours",
    "TTSU Days":           "ttsu_days",
    "TTSU Bucket":         "ttsu_bucket"
}

# --- Data types ---
COL_TYPES = {
    "case_number":          "Int64",
    "case_priority":        "Int64",
    "case_type":            "string",
    "subject":              "string",
    "market":               "string",
    "country":              "string",
    "ib_service_team":      "string",
    "equipment":            "Int64",
    "system_code":          "Int64",
    "parts_consumed_list":  "string",
    "remote_hours":         "float64",
    "travel_hours":         "float64",
    "onsite_hours":         "float64",
    "offsite_hours":        "float64",
    "total_hours":          "float64",
    "ttsu_days":            "float64",
    "ttsu_bucket":          "string"
}

# --- Date columns ---
DATE_COLS = {
    "creation_date":    "%Y-%m-%d",
    "disposition_date": "%Y-%m-%d",
    "teco_date":        "%Y-%m-%d"
}

# --- Columns to load
KEEP_COLS = [
    "Case Number", "Case Prio", "Case Type",
    "Subject",
    'Creation Date', 'Disposition Date', 'TECO Date',
    "Market", "Country", "IB Service Team",
    "Remote Remarks",
    "Field Remarks",
    "Equipment", "System Code", 
    'Parts Consumed List',
    'Remote Hours',
    'Travel Hours', 'Onsite Hours', 'Offsite Hours', 'Total Hours',
    'TTSU Days', 'TTSU Bucket'
]

# --- List on field_remarks section ---

FIELD_REMARKS_SECTIONS = [
    "Diagnostic performed by Engineer",
    "Follow Up Required",
    "Problem Description by Engineer",
    "Resolution",
    "Internal Comments",
    "Internal Remarks",
    "External Remarks"]

SECTION_COLUMN_MAP = {
    "Diagnostic performed by Engineer": "diagnostic",
    "Follow Up Required": "follow_up",
    "Problem Description by Engineer": "problem_description",
    "Resolution": "resolution",
    "Internal Comments": "internal_comments", 
    "Internal Remarks": "internal_remarks",
    "External Remarks": "external_remarks"
}

SEPARATOR = " [SEP] "

PARENT_PREFIX = "Parent "

# --- Metadata Feature labels ---

RESOLUTION_PATH_LABELS = {
    "remote": "remote_only",
    "field":  "remote_plus_field"
}

PARTS_FLAG_LABELS = {
    "no_parts":   "no_parts",
    "parts_used": "parts_used"
}