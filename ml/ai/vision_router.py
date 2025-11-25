"""
vision_router.py
Vision fallback using Gemini's vision generate_json_vision method.
Sends a clear vision prompt and the raw image bytes.
"""

from typing import Any, Dict
from ai.gemini_client import get_gemini_client

# Logger
try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger("vision_router")
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)


VISION_PROMPT = """
You are MediCure Vision — identify the medicine from this IMAGE.
Use the IMAGE ONLY (ignore any OCR text). Provide one or more medicines in simple language.
Return ONLY valid JSON using the same schema as below.

JSON Schema:
{
  "medicines": [
    {
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
    }
  ],
  "notes": ""
}
Rules:
- If you can identify the generic molecule, include it.
- If you must guess, set confidence lower (0.2-0.6).
- Always try to include at least one cheaper alternative (Jan Aushadhi if applicable).
- Return only JSON.
"""

def analyze_medicine_image(image_bytes: bytes) -> Dict[str, Any]:
    gemini = get_gemini_client()
    try:
        # Use the gemini client's vision method
        resp = gemini.generate_json_vision(VISION_PROMPT, image_bytes)
    except Exception as e:
        logger.error(f"VisionRouter: generate_json_vision failed: {e}")
        return {
            "ocr_text": "",
            "clean_text": "",
            "ai_parsed": {"medicines": []},
            "avg_confidence": 0.0,
            "ai_confidence": 0.0,
            "total_duration": 0.0
        }

    # Normalize response
    parsed = resp.get("json", {}) or {}
    if "medicines" not in parsed:
        parsed["medicines"] = []

    avg_conf = 0.0
    if parsed["medicines"]:
        avg_conf = float(parsed["medicines"][0].get("confidence", 0.0))

    return {
        "ocr_text": "",
        "clean_text": "",
        "ai_parsed": parsed,
        "avg_confidence": avg_conf,
        "ai_confidence": avg_conf,
        "total_duration": resp.get("duration", 0.0)
    }
