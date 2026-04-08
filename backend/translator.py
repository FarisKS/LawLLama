from deep_translator import GoogleTranslator
from langdetect import detect


SUPPORTED_LANGS = {
    "en": "English",
    "hi": "Hindi",
    "ml": "Malayalam"
}

TARGET_CODES = {
    "English": "en",
    "Hindi (हिन्दी)": "hi",
    "Malayalam (മലയാളം)": "ml"
}


def detect_language(text: str) -> tuple[str, str]:
    """Returns (lang_code, lang_name)"""
    try:
        code = detect(text)
        name = SUPPORTED_LANGS.get(code, code.upper())
        return code, name
    except:
        return "en", "English"


def translate_to_english(text: str) -> tuple[str, str]:
    """Translate any supported language to English. Returns (translated_text, source_lang_code)"""
    try:
        code, _ = detect_language(text)
        if code != "en":
            # chunk to stay within translator limits
            chunks = [text[i:i+4500] for i in range(0, len(text), 4500)]
            translated = " ".join(
                GoogleTranslator(source="auto", target="en").translate(c)
                for c in chunks
            )
            return translated, code
        return text, "en"
    except:
        return text, "en"


def translate_text(text: str, target_lang_code: str) -> str:
    """Translate English text to any target language."""
    if target_lang_code == "en":
        return text
    try:
        chunks = [text[i:i+4500] for i in range(0, len(text), 4500)]
        translated = " ".join(
            GoogleTranslator(source="en", target=target_lang_code).translate(c)
            for c in chunks
        )
        return translated
    except Exception as e:
        return text
