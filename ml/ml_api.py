"""
ml_api.py
FastAPI microservice for MediCure ML pipeline.
Updates: Now generates scan_id and saves context to DB for Chatbot.
"""

import time
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from pipeline import get_pipeline
from database import save_scan_result

# Logger
try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger("ml_api")
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

app = FastAPI(title="MediCure ML Service", version="1.1")

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = get_pipeline()


# -------------------------------------------------------------------
# Root Test Route
# -------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "MediCure ML microservice running."}


# -------------------------------------------------------------------
# MAIN ENDPOINT: /analyze-image
# -------------------------------------------------------------------
@app.post("/analyze-image")
async def analyze_medicine_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded image and runs the MediCure pipeline.
    Saves result to DB and returns scan_id for Chatbot integration.
    """

    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    filename = file.filename.lower()
    if not (
        filename.endswith(".jpg") or 
        filename.endswith(".jpeg") or 
        filename.endswith(".png")
    ):
        raise HTTPException(status_code=400, detail="Unsupported file format. Use JPG/PNG.")

    try:
        logger.info(f"Received file: {file.filename}")

        img_bytes = await file.read()
        start = time.time()

        # 1. Run the ML Pipeline
        result = pipeline.process_image_bytes(img_bytes)
        duration = round(time.time() - start, 4)

        # 2. Generate a Unique ID for this scan
        scan_id = str(uuid.uuid4())

        # --------------------------------------------------------------
        # 3. Extract Medicine Name from AI Parsed JSON
        # --------------------------------------------------------------
        ai_parsed_data: Optional[dict] = result.get("ai_parsed", {})

        medicines_list = ai_parsed_data.get("medicines", [])
        final_med_name = "Unknown Medicine"

        if (
            medicines_list 
            and isinstance(medicines_list, list) 
            and len(medicines_list) > 0
        ):
            first_medicine = medicines_list[0]

            brand = first_medicine.get("brand_name")
            generic = first_medicine.get("generic_name")

            if brand and brand != "":
                final_med_name = brand
            elif generic and generic != "":
                final_med_name = generic

        # --------------------------------------------------------------
        # 4. Save Scan Result to Database (Chatbot memory)
        # --------------------------------------------------------------
        try:
            save_scan_result(scan_id, final_med_name, result)
            logger.info(f"Saved scan context to DB: {scan_id} ({final_med_name})")
        except Exception as db_err:
            logger.error(f"Failed to save to DB (Chatbot won't work for this scan): {db_err}")
            # User still receives result, so do not raise

        logger.info(f"Pipeline completed in {duration}s")

        # 5. Return API Response
        return {
            "success": True,
            "scan_id": scan_id,
            "duration": duration,
            "data": result
        }

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))
