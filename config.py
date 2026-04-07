# --- LOADING --- 

# Column renaming 
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
    "Remote Remarks_EN":  "remote_remarks_en",
    "Field Remarks_EN":   "field_remarks_en",
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

# Data types 
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

# Date columns
DATE_COLS = {
    "creation_date":    "%Y-%m-%d",
    "disposition_date": "%Y-%m-%d",
    "teco_date":        "%Y-%m-%d"
}

# --- Columns with comma decimal separator ---
COMMA_DECIMAL_COLS = [
    "Remote Hours", "Travel Hours", "Onsite Hours",
    "Offsite Hours", "Total Hours"
]

# Columns to load
KEEP_COLS = [
    "Case Number", "Case Prio", "Case Type",
    "Subject",
    'Creation Date', 'Disposition Date', 'TECO Date',
    "Market", "Country", "IB Service Team",
    "Remote Remarks",        # keep original for reference
    "Field Remarks",         # keep original for reference
    "Remote Remarks_EN",     # translated version
    "Field Remarks_EN",      # translated version
    "Equipment", "System Code", 
    'Parts Consumed List',
    'Remote Hours',
    'Travel Hours', 'Onsite Hours', 'Offsite Hours', 'Total Hours',
    'TTSU Days', 'TTSU Bucket'
]

# --- TEXT EXTRACTION ---

SEPARATOR = " [SEP] "

PARENT_PREFIX = "Parent "

FIELD_REMARKS_SECTIONS = [
    # Family A — standard
    "Diagnostic performed by Engineer",
    "Follow Up Required",
    "Problem Description by Engineer",
    "Resolution",
    "Internal Comments",
    "Internal Remarks",
    "External Remarks",
    # Family B — T2/OneEMS
    "T2 Activities",
    "OneEMS internal remarks",
    # Occasional
    "PFQ: Malfunction",
]

SECTION_COLUMN_MAP = {
    "Diagnostic performed by Engineer":  "diagnostic",
    "Follow Up Required":                "follow_up",
    "Problem Description by Engineer":   "problem_description",
    "Resolution":                        "resolution",
    "Internal Comments":                 "internal_comments",
    "Internal Remarks":                  "internal_remarks",
    "External Remarks":                  "external_remarks",
    "T2 Activities":                     "t2_activities",
    "OneEMS internal remarks":           "onems_internal",
    "PFQ: Malfunction":                  "pfq_malfunction",
}

# --- Sub-field extraction patterns (Layer 2) ---

PROBLEM_SUBFIELD_PATTERNS = [
    "Problem description by engineer :",
    "Problem Description:",
]

ERROR_SUBFIELD_PATTERNS = [
    "Error # and/or description of error :",
    "Error # and/or description of error:",
]

MALFUNCTION_SUBFIELD_PATTERNS = [
    "Malfunction area :",
    "Malfunction area:"
]

TROUBLESHOOTING_SUBFIELD_PATTERNS = [
    "Troubleshooting Action:",
]

REPAIR_ACTION_SUBFIELD_PATTERNS = [
    "Repair Action:",
    "Repair Action :",
    'Repair Action "Internal" :',
    'Repair Action "External" :',
]

BOILERPLATE_STRIP = [
    "Information to support the complaint handling process",
    "Customer Function/Role:",
    "How was the device being used?",
    "Expected and actual behavior of the product:",
    "User Impact:",
    "Patient Impact:",
    "Current Software Version:",
]

# --- Layer 2 output column names ---

EXTRACTED_COLUMNS = [
    "extracted_problem_description",
    "extracted_error",
    "extracted_malfunction_area",
    "extracted_troubleshooting",
    "extracted_repair_action",
    "extracted_diagnostic"
]


# --- METADATA FEATURES ---

RESOLUTION_PATH_LABELS = {
    "remote": "remote_only",
    "field":  "remote_plus_field"
}

PARTS_FLAG_LABELS = {
    "no_parts":   "no_parts",
    "parts_used": "parts_used"
}