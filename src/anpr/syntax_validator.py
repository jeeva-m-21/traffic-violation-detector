import re
from typing import Optional, Tuple

# Indian State Codes
VALID_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HR",
    "HP", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ",
    "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB"
}

# OCR character disambiguation dictionaries
CHAR_TO_DIGIT = {
    'O': '0', 'D': '0', 'Q': '0',
    'I': '1', 'L': '1',
    'Z': '2',
    'E': '3',
    'A': '4',
    'S': '5',
    'G': '6',
    'B': '8'
}

DIGIT_TO_CHAR = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '3': 'E',
    '4': 'A',
    '5': 'S',
    '6': 'G',
    '8': 'B'
}

INDIAN_PLATE_REGEX = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$')

def clean_ocr_text(raw_text: str) -> str:
    """Removes non-alphanumeric characters, spaces, and converts to uppercase."""
    if not raw_text:
        return ""
    return re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()

def format_indian_plate(text: str) -> Optional[str]:
    """
    Attempts to validate and format a license plate string into standard Indian format.
    Standard formats:
      - 2 State letters + 2 District digits + 2 Series letters + 4 Unique digits (e.g. MH12DE1433)
      - 2 State letters + 1 District digit + 2 Series letters + 4 Unique digits (e.g. DL1CAA1111)
      - 2 State letters + 2 District digits + 4 Unique digits (e.g. MH121433)
    """
    clean = clean_ocr_text(text)
    if len(clean) < 7 or len(clean) > 11:
        return None

    chars = list(clean)

    # Position 0 & 1: Must be letters (State Code)
    for i in range(2):
        if chars[i].isdigit() and chars[i] in DIGIT_TO_CHAR:
            chars[i] = DIGIT_TO_CHAR[chars[i]]

    # Last 4 characters: Must be digits (Unique ID)
    for i in range(len(chars) - 4, len(chars)):
        if not chars[i].isdigit() and chars[i] in CHAR_TO_DIGIT:
            chars[i] = CHAR_TO_DIGIT[chars[i]]

    # Characters 2 & 3: District code (Digits)
    if chars[2].isalpha() and chars[2] in CHAR_TO_DIGIT:
        chars[2] = CHAR_TO_DIGIT[chars[2]]

    candidate = "".join(chars)

    # Match against standard regex
    if INDIAN_PLATE_REGEX.match(candidate):
        state = candidate[:2]
        if state in VALID_STATE_CODES or state in {"IND", "BH"}:  # Bharat series or standard
            return candidate

    return candidate if (len(candidate) >= 8 and candidate[-4:].isdigit()) else None
