"""
gemini_client.py
Final production version for MediCure ML.
Supports:
- TEXT JSON (OCR → prompt → Gemini)
- VISION JSON (image → prompt → Gemini Vision)
- SIMPLE CHAT (for Streamlit chatbot)
"""

import os
import json
import time
import google.generativeai as genai
from typing import Any, Dict, Optional

# Logger
try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except:
    import logging
    logger = logging.getLogger("gemini_client")
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)


class GeminiClient:
    """
    Handles ALL Gemini interactions (text + vision + chat).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 25,
        temperature: float = 0.0,
    ):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY environment variable is not set!")

        genai.configure(api_key=api_key)

        self.model_name = model or "models/gemini-2.5-flash"
        self.max_retries = max_retries
        self.timeout = timeout
        self.temperature = temperature

        logger.info(f"GeminiClient initialized with model: {self.model_name}")

    # ---------------------------------------------------------------------------
    # SIMPLE TEXT API (FOR CHATBOT USE)
    # ---------------------------------------------------------------------------
    def generate_response(self, prompt: str) -> str:
        """
        Simple wrapper for chatbot's text-only responses.
        """
        attempt = 0
        last_error = None
        model = genai.GenerativeModel(self.model_name)

        # Higher temperature for conversations
        chat_temp = 0.7

        while attempt < self.max_retries:
            attempt += 1
            logger.info(f"[Gemini CHAT] Attempt {attempt}/{self.max_retries}")

            try:
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": chat_temp,
                        "max_output_tokens": 2048,
                    }
                )

                # Safe check
                if hasattr(response, "text") and response.text.strip():
                    return response.text.strip()

                raise RuntimeError("Gemini returned EMPTY response text.")

            except Exception as e:
                last_error = e
                logger.warning(f"Gemini CHAT error: {e} — retrying...")
                time.sleep(1)

        raise RuntimeError(f"❌ Gemini CHAT failed after retries: {last_error}")

    # ---------------------------------------------------------------------------
    # SAFE TEXT JSON API
    # ---------------------------------------------------------------------------
    def generate_json(self, prompt: str) -> Dict[str, Any]:

        attempt = 0
        last_error = None
        model = genai.GenerativeModel(self.model_name)

        while attempt < self.max_retries:
            attempt += 1
            logger.info(f"[Gemini TEXT] Attempt {attempt}/{self.max_retries}")

            try:
                start = time.time()

                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": 4000,
                    }
                )

                duration = time.time() - start

                # -------- SAFE EXTRACTION --------
                text = ""

                try:
                    if hasattr(response, "text") and isinstance(response.text, str):
                        if response.text.strip():
                            text = response.text.strip()
                except Exception:
                    pass

                # fallback extraction
                if not text:
                    try:
                        candidate = response.candidates[0]
                        parts = candidate.content.parts
                        chunks = []
                        for p in parts:
                            if hasattr(p, "text") and isinstance(p.text, str):
                                if p.text.strip():
                                    chunks.append(p.text.strip())
                        text = "\n".join(chunks).strip()
                    except Exception:
                        text = ""

                if not text:
                    raise RuntimeError("Gemini returned EMPTY response. No text extracted.")

                text = self._strip_markdown(text)

                try:
                    parsed = json.loads(text)
                except:
                    parsed = self._fix_json(text)

                return {
                    "raw_text": text,
                    "json": parsed,
                    "duration": round(duration, 3),
                    "attempts": attempt,
                }

            except Exception as e:
                last_error = e
                logger.warning(f"Gemini TEXT error: {e} — retrying...")

        raise RuntimeError(f"❌ Gemini TEXT failed after retries: {last_error}")

    # ---------------------------------------------------------------------------
    # SAFE VISION JSON API
    # ---------------------------------------------------------------------------
    def generate_json_vision(self, prompt_text: str, image_bytes: bytes):

        attempt = 0
        last_error = None
        model = genai.GenerativeModel(self.model_name)

        while attempt < self.max_retries:
            attempt += 1
            logger.info(f"[Gemini VISION] Attempt {attempt}/{self.max_retries}")

            try:
                start = time.time()

                response = model.generate_content(
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt_text},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": image_bytes,
                                }
                            },
                        ],
                    },
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": 4000,
                    }
                )

                duration = time.time() - start

                # -------- SAFE EXTRACTION --------
                text = ""

                try:
                    if hasattr(response, "text") and isinstance(response.text, str):
                        if response.text.strip():
                            text = response.text.strip()
                except:
                    pass

                if not text:
                    try:
                        candidate = response.candidates[0]
                        parts = candidate.content.parts
                        chunks = []
                        for p in parts:
                            if hasattr(p, "text") and isinstance(p.text, str):
                                if p.text.strip():
                                    chunks.append(p.text.strip())
                        text = "\n".join(chunks).strip()
                    except:
                        text = ""

                if not text:
                    raise RuntimeError("Gemini Vision returned EMPTY response.")

                text = self._strip_markdown(text)

                try:
                    parsed = json.loads(text)
                except:
                    parsed = self._fix_json(text)

                return {
                    "raw_text": text,
                    "json": parsed,
                    "duration": round(duration, 3),
                    "attempts": attempt,
                }

            except Exception as e:
                last_error = e
                logger.warning(f"Gemini VISION error: {e} — retrying...")

        raise RuntimeError(f"❌ Gemini VISION failed after retries: {last_error}")

    # ---------------------------------------------------------------------------
    # Markdown Removal
    # ---------------------------------------------------------------------------
    def _strip_markdown(self, text: str) -> str:
        if "```" in text:
            parts = text.split("```")
            if len(parts) > 1:
                text = parts[1]
            else:
                text = parts[0]
        text = text.replace("json", "").replace("JSON", "")
        return text.replace("`", "").strip()

    # ---------------------------------------------------------------------------
    # JSON Repair
    # ---------------------------------------------------------------------------
    def _fix_json(self, text: str) -> Dict[str, Any]:
        try:
            fixed = text.replace(",}", "}").replace(",]", "]").strip()
            if not fixed.startswith("{"):
                fixed = "{" + fixed
            if not fixed.endswith("}"):
                fixed = fixed + "}"
            return json.loads(fixed)
        except:
            logger.error("❌ JSON repair failed. Returning empty.")
            return {"medicines": []}


# Singleton
_client = None
def get_gemini_client() -> GeminiClient:
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client
