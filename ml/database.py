import sqlite3
import json
import datetime
from typing import List, Dict, Any, Optional

DB_NAME = "medicure_memory.db"

def init_db():
    """Creates the database tables if they don't exist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Table to store Scan Results (Short term context)
    c.execute('''CREATE TABLE IF NOT EXISTS scans 
                (scan_id TEXT PRIMARY KEY, 
                 medicine_name TEXT, 
                 full_details TEXT, 
                 timestamp DATETIME)''')
    
    # Table to store Chat History (Long term memory)
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history 
                (user_id TEXT, 
                 role TEXT, 
                 message TEXT, 
                 timestamp DATETIME)''')
    
    conn.commit()
    conn.close()

def save_scan_result(scan_id: str, medicine_name: str, details_dict: Dict[str, Any]):
    """Saves the OCR/AI result so the chatbot can read it later."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Convert dictionary to JSON string for storage
    c.execute(
        "INSERT OR REPLACE INTO scans VALUES (?, ?, ?, ?)", 
        (scan_id, medicine_name, json.dumps(details_dict), datetime.datetime.now())
    )
    conn.commit()
    conn.close()

def get_scan_context(scan_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves the medicine details for the chatbot.
    Returns a merged dict consisting of medicine_name + parsed details.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT medicine_name, full_details FROM scans WHERE scan_id=?", (scan_id,))
    result = c.fetchone()
    conn.close()
    
    if result:
        medicine_name = result[0]
        full_details_json = result[1]
        
        context_data = {"medicine_name": medicine_name}
        try:
            details_dict = json.loads(full_details_json) if full_details_json else {}
            if isinstance(details_dict, dict):
                context_data.update(details_dict)
        except (json.JSONDecodeError, TypeError):
            # Malformed JSON — return only the name
            pass
            
        return context_data
    return None

def save_chat_message(user_id: str, role: str, message: str):
    """Remembers what the user said for future reference."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO chat_history VALUES (?, ?, ?, ?)", 
              (user_id, role, message, datetime.datetime.now()))
    conn.commit()
    conn.close()

def get_chat_history(user_id: str, limit: int = 10):
    """Gets previous conversation context (oldest-first for AI)."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT role, message FROM chat_history WHERE user_id=? ORDER BY timestamp DESC LIMIT ?", (user_id, limit))
    rows = c.fetchall()
    conn.close()
    # Reverse to oldest-first
    return [{"role": r[0], "content": r[1]} for r in rows][::-1]

# ---------------------------------------------------------------------
# Metrics extractor used by the dashboard. Extracts many fields with defaults,
# so older records (without new keys) won't break.
# ---------------------------------------------------------------------
def get_all_scan_metrics() -> List[Dict[str, Any]]:
    """
    Retrieves a list of metric dictionaries for each scan.
    Each dict contains:
      - medicine_name, duration, confidence, dqr, timestamp
      - ocr_text, ocr_text_length, ai_parsed (dict if present)
      - brand_name, generic_name, alternatives (list)
      - status, user_id, blur_score, brightness, resolution
      - flags: is_unknown, is_low_confidence, ocr_contains_name
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute("SELECT medicine_name, full_details, timestamp FROM scans ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    
    metrics_list: List[Dict[str, Any]] = []
    for row in rows:
        name, details_json, timestamp = row
        try:
            details_dict = json.loads(details_json) if details_json else {}
        except (json.JSONDecodeError, TypeError):
            details_dict = {}

        # Normalize / fallback keys
        duration = details_dict.get("total_duration") or details_dict.get("duration") or 0.0
        confidence = details_dict.get("ai_confidence") or details_dict.get("confidence") or 0.0
        dqr = details_dict.get("data_quality_rating") or details_dict.get("dqr") or 0.0

        # OCR text and length
        ocr_text = details_dict.get("ocr_text") or details_dict.get("extracted_text") or ""
        ocr_text_length = len(ocr_text) if isinstance(ocr_text, str) else 0

        # ai_parsed can be stored as dict or stringified dict
        ai_parsed_raw = details_dict.get("ai_parsed") or details_dict.get("parsed") or {}
        ai_parsed = {}
        if isinstance(ai_parsed_raw, str) and ai_parsed_raw.strip():
            try:
                ai_parsed = json.loads(ai_parsed_raw)
            except (json.JSONDecodeError, TypeError):
                ai_parsed = {}
        elif isinstance(ai_parsed_raw, dict):
            ai_parsed = ai_parsed_raw

        # Extract brand/generic/alternatives from parsed JSON
        brand_name = ai_parsed.get("brand_name") or ai_parsed.get("brand") or details_dict.get("brand_name") or ""
        generic_name = ai_parsed.get("generic_name") or ai_parsed.get("generic") or details_dict.get("generic_name") or ""
        alternatives = ai_parsed.get("alternatives") or details_dict.get("alternatives") or []
        if not isinstance(alternatives, list):
            alternatives = []

        # Other optional/system fields
        status = details_dict.get("status") or details_dict.get("scan_status") or "success"
        user_id = details_dict.get("user_id") or details_dict.get("uid") or None

        # Optional image quality scores if you saved them earlier
        blur_score = details_dict.get("blur_score")
        brightness = details_dict.get("brightness")
        resolution = details_dict.get("resolution")  # e.g., width*height or tuple

        # Flags
        name_norm = (name or "").strip().lower()
        is_unknown = name_norm.startswith("unknown") or name_norm in ("", "unknown medicine")
        is_low_confidence = (confidence is None) and True or (confidence < 0.5)
        ocr_contains_name = False
        try:
            if isinstance(ocr_text, str) and isinstance(name, str) and name.strip():
                ocr_contains_name = name.strip().lower() in ocr_text.lower()
        except Exception:
            ocr_contains_name = False

        metrics_list.append({
            "medicine_name": name or "",
            "duration": float(duration or 0.0),
            "confidence": float(confidence or 0.0),
            "timestamp": timestamp,
            "is_unknown": bool(is_unknown),
            "is_low_confidence": bool(is_low_confidence),
            "dqr": float(dqr or 0.0),
            "ocr_text": ocr_text,
            "ocr_text_length": ocr_text_length,
            "ai_parsed": ai_parsed,
            "brand_name": brand_name,
            "generic_name": generic_name,
            "alternatives": alternatives,
            "status": status,
            "user_id": user_id,
            "blur_score": blur_score,
            "brightness": brightness,
            "resolution": resolution,
            "ocr_contains_name": bool(ocr_contains_name),
        })
        
    return metrics_list

# Initialize DB on import
init_db()
