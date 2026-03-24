import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import RESOLUTION_PATH_LABELS, PARTS_FLAG_LABELS


def get_resolution_path(field_remarks):
    if pd.isna(field_remarks) or not str(field_remarks).strip():
        return RESOLUTION_PATH_LABELS["remote"]
    return RESOLUTION_PATH_LABELS["field"]


def get_parts_used_flag(parts_consumed_list):
    if pd.isna(parts_consumed_list) or not str(parts_consumed_list).strip():
        return PARTS_FLAG_LABELS["no_parts"]
    return PARTS_FLAG_LABELS["parts_used"]