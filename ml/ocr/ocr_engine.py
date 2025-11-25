import io
import time
import re
import numpy as np
from PIL import Image, ImageOps
import cv2
import pytesseract

try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger("ocr_engine")
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

class OCREngine:
    """
    v3 Aggressive Engine:
    - Uses Otsu thresholding for white-on-dark text (Brand names).
    - Uses Dilation to thicken white text.
    - Runs multi-angle (0, 90, 270).
    """

    def __init__(self):
        logger.info("Initializing v3 Aggressive OCREngine...")

    def extract(self, image_bytes: bytes) -> dict:
        start = time.time()
        try:
            pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            logger.error(f"OCR - failed to load image: {e}")
            return self._empty_result()

        # 1. Prepare Base Images
        # Standard grayscale for black text (ingredients)
        gray_std = self._preprocess_standard(pil)
        
        # High-contrast binary for white text (Brand name "Qoaliderma")
        # We invert, then Otsu threshold, then thicken lines.
        gray_inv = self._preprocess_inverted_heavy(pil)

        # 2. Define Variations (Image, Description)
        # We need 0 degrees and 90 degrees (Counter-Clockwise) mostly.
        # 270 (Clockwise) is added just in case.
        
        variations = [
            (gray_std, "Std_0"),
            (gray_inv, "Inv_0"),
            
            # 90 deg CCW (Vertical text reading bottom-to-top)
            (cv2.rotate(gray_std, cv2.ROTATE_90_COUNTERCLOCKWISE), "Std_90"),
            (cv2.rotate(gray_inv, cv2.ROTATE_90_COUNTERCLOCKWISE), "Inv_90"),

            # 90 deg CW (Vertical text reading top-to-bottom)
            (cv2.rotate(gray_std, cv2.ROTATE_90_CLOCKWISE), "Std_270"),
            (cv2.rotate(gray_inv, cv2.ROTATE_90_CLOCKWISE), "Inv_270"),
        ]

        detected_words = []

        # 3. Run OCR
        # We use two configs:
        # PSM 6: Assume a uniform block of text (good for ingredients)
        # PSM 11: Sparse text (good for isolated Brand names next to logos)
        configs = [r"--oem 3 --psm 6", r"--oem 3 --psm 11"]

        for img, desc in variations:
            for cfg in configs:
                try:
                    text = pytesseract.image_to_string(img, config=cfg)
                    cleaned = self._clean_and_split(text)
                    detected_words.extend(cleaned)
                except Exception:
                    pass

        # 4. Dedup and Reconstruct
        # We preserve order somewhat by using dict.fromkeys
        unique_words = list(dict.fromkeys(detected_words))
        final_text = " ".join(unique_words)

        duration = round(time.time() - start, 3)
        
        return {
            "ocr_text": final_text, 
            "clean_text": final_text,
            "duration": duration,
            "engine_used": "tesseract_v3_aggressive"
        }

    def _preprocess_standard(self, pil_img: Image.Image) -> np.ndarray:
        """Standard CLAHE grayscale for normal black-on-white text."""
        pil_img = ImageOps.exif_transpose(pil_img)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Upscale logic
        h, w = img.shape[:2]
        if min(h, w) < 1000:
            img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        # Contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _preprocess_inverted_heavy(self, pil_img: Image.Image) -> np.ndarray:
        """
        Aggressive processing for white-on-color text (The Brand Name).
        Invert -> Otsu Threshold -> Dilate.
        """
        pil_img = ImageOps.exif_transpose(pil_img)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Upscale
        h, w = img.shape[:2]
        if min(h, w) < 1000:
            img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        # Invert colors (White text becomes dark, Red background becomes Cyan/Light)
        img_inv = cv2.bitwise_not(img)
        gray = cv2.cvtColor(img_inv, cv2.COLOR_BGR2GRAY)

        # Otsu's Thresholding
        # This forces every pixel to be either Black or White. 
        # It removes the "gray" noise of the red background.
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Dilate (Thicken the text)
        # White text on dark often looks "thin" to Tesseract. We make it bolder.
        # Since we inverted, the text is now BLACK on WHITE.
        # To thicken Black text, we actually use ERODE (eats away white).
        kernel = np.ones((2, 2), np.uint8)
        thickened = cv2.erode(thresh, kernel, iterations=1)

        return thickened

    def _clean_and_split(self, text: str) -> list:
        """
        Splits text into words and removes garbage.
        """
        if not text: return []
        
        # Replace common OCR confusion chars
        t = text.replace("|", "I").replace("\n", " ")
        
        # Allow alphanumeric, %, -, +, and basic punctuation
        # We strictly filter out short garbage words like "pa", "Le" unless they are "mg" or "ml"
        words = t.split(" ")
        valid_words = []
        
        for w in words:
            # Remove non-ascii visual noise
            w_clean = re.sub(r"[^a-zA-Z0-9\-\%\(\)\.]", "", w)
            
            if len(w_clean) < 3:
                # Keep units or very short known words, skip garbage like 'es' 'pa'
                if w_clean.lower() in ['mg', 'ml', 'g', 'gm', 'ip', 'bp', 'usp', '%']:
                    valid_words.append(w_clean)
                continue
                
            valid_words.append(w_clean)
            
        return valid_words

    def _empty_result(self) -> dict:
        return {
            "ocr_text": "",
            "clean_text": "",
            "duration": 0.0,
            "engine_used": "none"
        }

# Singleton accessor
_default_engine = None

def get_default_ocr_engine() -> OCREngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = OCREngine()
    return _default_engine