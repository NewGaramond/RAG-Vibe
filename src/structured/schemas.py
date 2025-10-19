# src/structured/schemas.py
from __future__ import annotations
from typing import Optional, Dict, Any
from datetime import date
from pydantic import BaseModel, EmailStr, Field, field_validator
from dateutil import parser as dateparser
import json

class PersonRecord(BaseModel):
    first_name: Optional[str] = Field(None, description="Given name")
    last_name: Optional[str] = Field(None, description="Surname / family name")
    date_of_birth: Optional[date] = Field(None, description="YYYY-MM-DD")
    id_number: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    address_line: Optional[str] = None
    city: Optional[str] = None
    state_region: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None

    @field_validator("date_of_birth")
    def _parse_date(cls, v):
        if not v or isinstance(v, date):
            return v
        try:
            # Accept 12/05/1990, 1990-05-12, "May 12 1990", etc.
            return dateparser.parse(str(v), dayfirst=True).date()
        except Exception:
            return None

def model_to_dict(m: BaseModel) -> Dict[str, Any]:
    """Return only JSON-safe primitives (dates -> ISO strings)."""
    try:
        # Pydantic v2
        return json.loads(m.model_dump_json())
    except Exception:
        # Pydantic v1
        return json.loads(m.json())


SCHEMAS = {
    "person": (PersonRecord, {
        "description": (
            "Extract basic personal identification fields. "
            "Required keys in the output JSON: "
            "first_name, last_name, date_of_birth (YYYY-MM-DD), id_number, email, phone, "
            "address_line, city, state_region, postal_code, country. "
            "Use null for missing values."
        )
    }),
}

