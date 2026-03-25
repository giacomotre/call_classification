import re
import pandas as pd
import ftfy  # fixes text encoding issues
from config import SEPARATOR

def preprocess_text(text):
    if pd.isna(text):
        return None
    text = ftfy.fix_text(text)  # fixes encoding artifacts
    cleaned_text = text.replace(SEPARATOR, "")
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text if cleaned_text else None