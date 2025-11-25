"""
ai_router.py
Text-based AI router that uses the final GeminiClient (text).
Provides parse_medicine(cleaned_text) -> dict with keys: raw_text, json, duration.
"""

import json
from typing import Dict, Any

from ai.gemini_client import get_gemini_client   # <-- FIXED


# Logger
try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger("ai_router")
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)


def _load_prompt_text() -> str:
    try:
        with open("ai/prompts/medicine_prompt.txt", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        # fallback minimal prompt
        return "Extract structured medicine info in JSON. Return only valid JSON."


class AIRouter:
    def __init__(self):
        self.prompt_template = _load_prompt_text()
        self.gemini = get_gemini_client()   # <-- FIXED
        logger.info("AI Router initialized successfully.")

    def parse_medicine(self, cleaned_text: str) -> Dict[str, Any]:
        """
        Send cleaned_text to Gemini (text mode) and return the gemini generate_json output.
        Ensures the returned JSON has "medicines" list and simple cheaper_alternatives.
        """
        prompt = self.prompt_template + "\n\nOCR_TEXT:\n" + (cleaned_text or "")

        try:
            response = self.gemini.generate_json(prompt)
        except Exception as e:
            logger.error(f"AI Router: generate_json failed: {e}")
            # Return safe, empty structure
            return {"raw_text": "", "json": {"medicines": []}, "duration": 0}

        out_json = response.get("json", {}) or {}
        # Ensure structure
        if "medicines" not in out_json:
            out_json["medicines"] = []

        # Guarantee at least one medicine entry (best-effort empty)
        if not out_json["medicines"]:
            out_json["medicines"].append({
                "brand_name": "",
                "generic_name": "",
                "composition": "",
                "form": "",
                "uses": "",
                "side_effects": "",
                "precautions": "",
                "interactions": "",
                "confidence": 0.0,
                "cheaper_alternatives": [],
                "notes": ""
            })

        # Ensure cheaper_alternatives exists (Gemini will often include but ensure)
        for med in out_json["medicines"]:
            if "cheaper_alternatives" not in med:
                med["cheaper_alternatives"] = []
        response["json"] = out_json

        return response


# Singleton accessor
_ai_router = None


def get_ai_router() -> AIRouter:
    global _ai_router
    if _ai_router is None:
        _ai_router = AIRouter()
    return _ai_router
