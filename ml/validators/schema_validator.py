"""
schema_validator.py
Ensures AI JSON output follows a strict & safe schema for MediCure.

This validator:
- Ensures required keys exist
- Fills missing fields with defaults
- Normalizes data types
- Handles cheaper_alternatives schema
- Guarantees safe output for the ML pipeline
"""

from typing import Any, Dict, List


class SchemaValidator:
    """
    Validates and normalizes AI-generated JSON output.
    """

    # Required fields for each medicine entry
    MED_FIELDS = [
        "brand_name",
        "generic_name",
        "composition",
        "form",
        "uses",
        "side_effects",
        "precautions",
        "interactions",
        "confidence",
        "cheaper_alternatives",
        "notes",
    ]

    # Schema for cheaper alternative item
    ALT_FIELDS = ["name", "availability", "price_comparison", "notes"]

    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry to validate entire JSON object.
        """
        if not isinstance(data, dict):
            return {"medicines": [], "notes": ""}

        # Ensure medicines list exists
        meds = data.get("medicines")
        if not isinstance(meds, list):
            meds = []

        validated_meds = []

        for item in meds:
            validated_meds.append(self._validate_medicine(item))

        # Validate top-level notes
        top_notes = data.get("notes", "")
        if not isinstance(top_notes, str):
            top_notes = ""

        return {
            "medicines": validated_meds,
            "notes": top_notes,
        }

    # ------------------------------------------------------------------
    # Validate each medicine entry
    # ------------------------------------------------------------------
    def _validate_medicine(self, med: Any) -> Dict[str, Any]:
        """
        Ensure a single medicine dict has all required keys.
        """
        if not isinstance(med, dict):
            med = {}

        validated = {}

        for field in self.MED_FIELDS:
            if field == "confidence":
                validated[field] = self._parse_confidence(med.get(field))

            elif field == "cheaper_alternatives":
                validated[field] = self._validate_alternatives(
                    med.get("cheaper_alternatives")
                )

            else:
                value = med.get(field, "")
                validated[field] = value if isinstance(value, str) else ""

        return validated

    # ------------------------------------------------------------------
    # Validate alternatives list
    # ------------------------------------------------------------------
    def _validate_alternatives(self, alts: Any) -> List[Dict[str, str]]:
        """
        Validates the cheaper alternatives list.
        """
        if not isinstance(alts, list):
            return []

        validated_list = []

        for alt in alts:
            if not isinstance(alt, dict):
                continue

            clean_alt = {}

            for key in self.ALT_FIELDS:
                val = alt.get(key, "")
                clean_alt[key] = val if isinstance(val, str) else ""

            validated_list.append(clean_alt)

        return validated_list

    # ------------------------------------------------------------------
    # Parse confidence float safely
    # ------------------------------------------------------------------
    def _parse_confidence(self, val: Any) -> float:
        """
        Converts confidence into float between 0 and 1.
        """
        try:
            f = float(val)
            if f < 0:
                return 0.0
            if f > 1:
                return 1.0
            return f
        except Exception:
            return 0.0
