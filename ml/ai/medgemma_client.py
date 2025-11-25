import os
import json
import time
import io
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from PIL import Image
from typing import Dict, Any


class MedGemmaClient:

    def __init__(
        self,
        model_name: str = "google/med-gemma-2b-it",
        max_retries: int = 3,
        device: str = None,
        temperature: float = 0.1,
    ):
        self.max_retries = max_retries
        self.temperature = temperature

        # Use CPU or GPU automatically
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[MedGemma] Loading 4-bit quantized model on {self.device} ...")

        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0
        )

        # Load processor (small file)
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Load 4-bit model
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )

        print("[MedGemma] 4-bit quantized model loaded successfully.")

    # ---------------------------------------------------------
    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """
        TEXT-only mode (for OCR-cleaned text)
        """
        attempt = 0
        last_error = None

        while attempt < self.max_retries:
            attempt += 1

            try:
                start = time.time()

                inputs = self.processor(
                    text=prompt,
                    return_tensors="pt"
                ).to(self.device)

                output_ids = self.model.generate(
                    **inputs,
                    max_length=2048,
                    temperature=self.temperature
                )

                text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

                try:
                    parsed = json.loads(text)
                except:
                    parsed = self._fix_json(text)

                return {
                    "raw_text": text,
                    "json": parsed,
                    "duration": round(time.time() - start, 3),
                }

            except Exception as e:
                last_error = e

        raise RuntimeError(f"MedGemma TEXT generation failed: {last_error}")

    # ---------------------------------------------------------
    def generate_json_vision(self, prompt: str, image_bytes: bytes) -> Dict[str, Any]:
        """
        Vision + text mode (image interpretation)
        """
        attempt = 0
        last_error = None

        while attempt < self.max_retries:
            attempt += 1

            try:
                start = time.time()

                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                ).to(self.device)

                output_ids = self.model.generate(
                    **inputs,
                    max_length=2048,
                    temperature=self.temperature
                )

                text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

                try:
                    parsed = json.loads(text)
                except:
                    parsed = self._fix_json(text)

                return {
                    "raw_text": text,
                    "json": parsed,
                    "duration": round(time.time() - start, 3),
                }

            except Exception as e:
                last_error = e

        raise RuntimeError(f"MedGemma VISION generation failed: {last_error}")

    # ---------------------------------------------------------
    def _fix_json(self, text: str):
        """
        Best-effort fix for malformed JSON.
        """
        try:
            t = text.replace(",}", "}").replace(",]", "]").strip()
            if not t.startswith("{"):
                t = "{" + t
            if not t.endswith("}"):
                t = t + "}"
            return json.loads(t)
        except:
            return {"medicines": [], "notes": "json_parse_failed", "raw": text}


# Singleton
_client = None

def get_medgemma_client() -> MedGemmaClient:
    global _client
    if _client is None:
        _client = MedGemmaClient()
    return _client
