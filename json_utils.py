import re
import json5

def extract_json_from_text(text):
    """
    Extract the first JSON array or object from a string.
    Returns the parsed JSON, or raises if not found.
    Uses json5 for tolerant parsing.
    """
    match = re.search(r'(\[\s*{.*?}\s*\])', text, re.DOTALL)
    if not match:
        match = re.search(r'({.*})', text, re.DOTALL)
    if match:
        json_str = match.group(1)
        return json5.loads(json_str)
    raise ValueError("No valid JSON found in text")
