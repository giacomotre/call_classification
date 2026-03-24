import re
from config import SEPARATOR

def preprocess_text(text):
    if not text or not text.strip():
        return None
    cleaned_text = text.replace(SEPARATOR, "")
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text