"""
pipeline.py
MediCure pipeline (Option B: no local DB fallback).
Flow:
- OCR (with preprocessing)
- Calculate Data Quality Rating (DQR)
- If OCR short → TEXT AI
- If OCR LONG → skip text AI (Gemini blocks long medical OCR)
- Vision fallback always available
"""

import time
from typing import Dict, Any

# Ensure correct imports based on your existing structure
from ocr.ocr_engine import get_default_ocr_engine
from ai.ai_router import get_ai_router
from ai.vision_router import analyze_medicine_image

# Logger setup
try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except:
    import logging
    logger = logging.getLogger("pipeline")
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)


def calculate_dqr(ocr_confidence: float, ocr_text: str) -> float:
    """
    Calculates the Data Quality Rating (DQR) based on OCR confidence and text density.
    
    Formula: (0.7 * Avg OCR Confidence) + (0.3 * Normalized Text Length)
    """
    
    # 1. OCR Confidence Score (Weight: 70%)
    conf_score = ocr_confidence * 0.7

    # 2. Text Length/Density Score (Weight: 30%)
    # Target length is a typical expectation for a well-captured medicine label
    TARGET_LENGTH = 150 
    text_length = len(ocr_text)
    
    # Normalize: Clamp the length score at 1.0 (if length >= TARGET_LENGTH)
    length_score = min(1.0, text_length / TARGET_LENGTH) * 0.3
    
    # 3. Final DQR (0.0 to 1.0)
    dqr = round(conf_score + length_score, 4)
    
    return dqr


class MediCurePipeline:
    def __init__(self):
        self.ocr = get_default_ocr_engine()
        self.ai = get_ai_router()
        logger.info("MediCure Pipeline initialized.")


    def process_image_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        total_start = time.time()

        # 1) OCR
        logger.info("Running OCR...")
        try:
            ocr_result = self.ocr.extract(image_bytes)
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            ocr_result = {"ocr_text": "", "clean_text": "", "avg_confidence": 0.0}

        ocr_text = (ocr_result.get("ocr_text") or "").strip()
        clean_text = (ocr_result.get("clean_text") or "").strip()
        ocr_conf = ocr_result.get("avg_confidence", 0.0)

        logger.info(f"OCR Text Extracted (len={len(ocr_text)})")

        # 🛑 NEW: Calculate DQR 🛑
        data_quality_rating = calculate_dqr(ocr_conf, clean_text)
        logger.info(f"Calculated DQR: {data_quality_rating:.4f}")

        
        # 2) Determine TEXT or VISION
        if len(ocr_text) > 200000:
            logger.info("OCR text too long → using Vision AI (skipping TEXT).")
            try_text_ai = False
        else:
            try_text_ai = (len(ocr_text) >= 30 and len(clean_text.split()) > 3)

        # 3) TEXT AI if allowed
        if try_text_ai:
            logger.info("OCR looks useful → TEXT AI parsing...")
            try:
                ai_resp = self.ai.parse_medicine(clean_text)
                ai_json = ai_resp.get("json", {})
                meds = ai_json.get("medicines", [])

                if meds and meds[0].get("confidence", 0) > 0:
                    ai_conf = float(meds[0].get("confidence", 0))
                    return {
                        "ocr_text": ocr_text,
                        "clean_text": clean_text,
                        "ai_parsed": ai_json,
                        "avg_confidence": round((ocr_conf + ai_conf) / 2, 4),
                        "ocr_confidence": ocr_conf,
                        "ai_confidence": ai_conf,
                        "total_duration": round(time.time() - total_start, 3),
                        # 🛑 NEW DQR RETURN 🛑
                        "data_quality_rating": data_quality_rating,
                    }
                else:
                    logger.warning("TEXT AI returned low-confidence → trying Vision fallback...")

            except Exception as e:
                logger.warning(f"TEXT AI failed: {e}")


        # 4) VISION fallback
        logger.info("Using Vision fallback...")
        try:
            vision_resp = analyze_medicine_image(image_bytes)
            ai_json = vision_resp.get("ai_parsed", {})
            meds = ai_json.get("medicines", [])

            if meds:
                ai_conf = float(meds[0].get("confidence", 0))
                return {
                    "ocr_text": ocr_text,
                    "clean_text": clean_text,
                    "ai_parsed": ai_json,
                    "avg_confidence": round((ocr_conf + ai_conf) / 2, 4),
                    "ocr_confidence": ocr_conf,
                    "ai_confidence": ai_conf,
                    "total_duration": round(time.time() - total_start, 3),
                    # 🛑 NEW DQR RETURN 🛑
                    "data_quality_rating": data_quality_rating, 
                }
            else:
                logger.warning("Vision returned no medicines.")

        except Exception as e:
            logger.error(f"Vision AI failed completely: {e}")

        # 5) Total Failure → return safe empty structure
        logger.error("All methods failed → returning FALLBACK JSON.")

        return {
            "ocr_text": ocr_text,
            "clean_text": clean_text,
            "ai_parsed": {
                "medicines": [],
                "notes": "Unable to identify medicine with confidence.",
            },
            "avg_confidence": 0,
            "ocr_confidence": ocr_conf,
            "ai_confidence": 0,
            "total_duration": round(time.time() - total_start, 3),
            # 🛑 NEW DQR RETURN FOR FAILURE 🛑
            "data_quality_rating": data_quality_rating,
        }


# Singleton
_pipeline = None
def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = MediCurePipeline()
    return _pipeline