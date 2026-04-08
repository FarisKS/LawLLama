import re

# Patterns to detect and mask PII
PII_PATTERNS = [
    # Aadhaar number (12 digits, sometimes with spaces)
    (r'\b\d{4}\s?\d{4}\s?\d{4}\b', '[AADHAAR MASKED]'),
    # PAN card
    (r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', '[PAN MASKED]'),
    # Phone numbers (Indian)
    (r'\b(?:\+91[\s-]?)?[6-9]\d{9}\b', '[PHONE MASKED]'),
    # Email addresses
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL MASKED]'),
    # Indian bank account numbers (9-18 digits)
    (r'\b\d{9,18}\b', '[ACCOUNT# MASKED]'),
    # IFSC codes
    (r'\b[A-Z]{4}0[A-Z0-9]{6}\b', '[IFSC MASKED]'),
    # Passport numbers
    (r'\b[A-Z][1-9][0-9]{7}\b', '[PASSPORT MASKED]'),
    # Voter ID
    (r'\b[A-Z]{3}[0-9]{7}\b', '[VOTER-ID MASKED]'),
    # Pincode (standalone 6 digits)
    (r'\bPIN[\s:]*\d{6}\b', '[PINCODE MASKED]'),
]

# Common Indian name patterns (after "name:", "s/o", "d/o", "w/o")
NAME_PATTERNS = [
    (r'(?i)(name\s*[:\-]\s*)([A-Z][a-z]+ [A-Z][a-z]+)', r'\1[NAME MASKED]'),
    (r'(?i)(s/o|d/o|w/o)\s+([A-Z][a-z]+ ?[A-Z]?[a-z]*)', r'\1 [NAME MASKED]'),
    (r'(?i)(complainant|accused|victim|petitioner|respondent)\s*[:\-]\s*([A-Z][a-z]+ [A-Z][a-z]+)', r'\1: [NAME MASKED]'),
]


def mask_pii(text: str) -> tuple[str, list[str]]:
    """
    Mask PII in text before sending to external API.
    Returns (masked_text, list_of_what_was_masked)
    """
    masked = text
    found = []

    for pattern, replacement in PII_PATTERNS:
        matches = re.findall(pattern, masked)
        if matches:
            label = replacement.strip('[]').replace(' MASKED', '')
            found.append(f"{label} ({len(matches)} instance{'s' if len(matches)>1 else ''})")
            masked = re.sub(pattern, replacement, masked)

    for pattern, replacement in NAME_PATTERNS:
        matches = re.findall(pattern, masked)
        if matches:
            if 'NAME' not in str(found):
                found.append(f"Name (possible)")
            masked = re.sub(pattern, replacement, masked)

    return masked, found


def get_privacy_report(found_items: list[str]) -> str:
    if not found_items:
        return None
    return "Masked before sending to AI: " + ", ".join(found_items)
