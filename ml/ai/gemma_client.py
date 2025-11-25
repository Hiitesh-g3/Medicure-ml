import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class GemmaTextClient:
    def __init__(self,
                 model_name: str = "google/gemma-2-2b-it",
                 temperature: float = 0.1):

        self.temperature = temperature

        print("[Gemma] Loading text model:", model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        print("[Gemma] Text model loaded.")

    def generate_json(self, prompt: str):
        start = time.time()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=self.temperature
        )

        text = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

        # Try parse JSON
        try:
            parsed = json.loads(text)
        except:
            parsed = {"error": "json_parse_failed", "raw": text}

        return {
            "raw_text": text,
            "json": parsed,
            "duration": round(time.time() - start, 3),
        }


# Singleton
_client = None

def get_gemma_client():
    global _client
    if _client is None:
        _client = GemmaTextClient()
    return _client
