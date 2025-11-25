"""
text_cleaner.py
Utility for cleaning and normalizing OCR text before sending to Gemini.

Features:
- Basic normalization (spacing, newlines, case)
- Removal of known OCR noise patterns
- Regex cleanup for medicine strips (batch numbers, dates, MRP, etc.)
- Deduplication of repeated words
- Configurable cleaning pipeline

Usage:
>>> from ocr.text_cleaner import TextCleaner
>>> cleaner = TextCleaner()
>>> clean = cleaner.clean_text(ocr_text)
>>> print(clean)
"""

import re
import unicodedata
from typing import List


class TextCleaner:
    """
    Cleans OCR text for downstream LLM parsing.
    """

    # Common noisy tokens seen in OCR outputs
    NOISE_PATTERNS = [
        r"[^A-Za-z0-9\s\+\-/(),.]",
        r"\bMFG\.?\b.*?\d{4}",                # manufacturing dates
        r"\bEXP\.?\b.*?\d{4}",                # expiry dates
        r"\bBATCH\.?\b.*?[A-Z0-9]+",          # batch codes
        r"\bB\.?No\.?.*?[A-Z0-9]+",           # B.No patterns
        r"\bMRP.*?\d+\.?\d*",                 # MRP lines
        r"\bSCHEDULE\b.*",                    # schedule descriptors
        r"\bKEEP OUT OF REACH.*",             # common warnings
        r"\bSTRIP OF\b.*",                    # packaging text
    ]

    MULTISPACE_REGEX = re.compile(r"\s+")
    REPEATED_WORD_REGEX = re.compile(r"\b(\w+)( \1\b)+", flags=re.IGNORECASE)

    def __init__(self, lowercase: bool = False):
        """
        :param lowercase: If True, convert text to lowercase; else keep original case.
        """
        self.lowercase = lowercase

    # ---------------------------------------------------------
    # Public Function
    # ---------------------------------------------------------
    def clean_text(self, text: str) -> str:
        """
        Clean OCR text using predefined regex and normalization steps.
        """
        if not text:
            return ""

        # Normalize Unicode
        text = unicodedata.normalize("NFKD", text)

        # Basic cleanup
        text = text.replace("\r", " ").replace("\n", " ")

        # Remove noisy patterns
        for pattern in self.NOISE_PATTERNS:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

        # Collapse multiple spaces
        text = self.MULTISPACE_REGEX.sub(" ", text).strip()

        # Remove repeated words: "tablet tablet", "mg mg"
        text = self._remove_repeated_words(text)

        # Lowercase if required
        if self.lowercase:
            text = text.lower()

        return text.strip()

    # ---------------------------------------------------------
    # Internal Helpers
    # ---------------------------------------------------------
    def _remove_repeated_words(self, text: str) -> str:
        """
        Replace consecutive repeated words "word word" with a single "word".
        """
        def repl(match: re.Match) -> str:
            return match.group(1)
        return self.REPEATED_WORD_REGEX.sub(repl, text)


# Convenience function
_default_cleaner = TextCleaner()

def clean_text(text: str) -> str:
    return _default_cleaner.clean_text(text)
