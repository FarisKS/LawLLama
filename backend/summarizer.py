"""
BART Summarizer — optional toggle, GPU-accelerated on RTX 3050
Uses facebook/bart-base (~550MB VRAM — safe alongside FAISS + sentence-transformers)
"""
import torch
from transformers import BartForConditionalGeneration, BartTokenizer

_instance = None


class BartSummarizer:
    def __init__(self):
        print("[BART] Loading facebook/bart-base...")
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.model     = BartForConditionalGeneration.from_pretrained(
                            "facebook/bart-base").to(self.device)
        print(f"[BART] Loaded on {self.device}")

    def summarize(self, text: str) -> str:
        text = text[:3000].strip()
        if not text:
            return ""
        framed = (
            "Legal document summary: The following is a legal document. "
            "Summarize the key facts, parties involved, and main legal issues. "
            + text
        )
        inputs = self.tokenizer(
            framed, return_tensors="pt",
            max_length=1024, truncation=True
        ).to(self.device)
        ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=180, min_length=50,
            length_penalty=2.0, num_beams=4,
            early_stopping=True, no_repeat_ngram_size=3
        )
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)


def get_summarizer():
    """Lazy singleton — BART only loads when first needed."""
    global _instance
    if _instance is None:
        _instance = BartSummarizer()
    return _instance


def summarize_if_enabled(text: str, enabled: bool) -> str:
    """Returns BART summary if toggle is on, empty string otherwise."""
    if not enabled:
        return ""
    try:
        return get_summarizer().summarize(text)
    except Exception as e:
        print(f"[BART] Error: {e}")
        return ""
