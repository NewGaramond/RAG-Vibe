# src/structured/multifor_extractor.py
from __future__ import annotations
import json, os, re
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import fitz  # PyMuPDF
from langchain_openai import ChatOpenAI

from src.structured.schemas import SCHEMAS, PersonRecord, model_to_dict

# ========= Config =========
@dataclass
class MultiFormConfig:
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    chat_model: str = os.getenv("OPENAI_MODEL_CHAT", "gpt-4o-mini")
    allow_llm: bool = True
    use_acroform: bool = True
    max_pages: Optional[int] = None  # limit for huge PDFs

    # --- OCR knobs ---
    use_ocr: bool = False            # enable OCR fallback
    ocr_force_all: bool = False      # OCR every page (even if text exists)
    ocr_lang: str = "eng"            # tesseract language(s), e.g., "eng" or "eng+spa"
    ocr_min_chars: int = 25          # if extracted text has fewer chars than this, OCR the page
    ocr_dpi: int = 300               # rasterization DPI for OCR
    ocr_psm: str = "6"               # tesseract page segmentation mode (e.g., "4", "6", "3")
# ========= PDF IO =========
def _person_field_names():
    return (
        list(getattr(PersonRecord, "model_fields", {}).keys())  # pydantic v2
        or list(getattr(PersonRecord, "__fields__", {}).keys())  # pydantic v1
    )
def _pixmap_to_pil(pix) -> "Image.Image":
    from PIL import Image
    mode = "RGB" if pix.alpha == 0 else "RGBA"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if mode == "RGBA":
        img = img.convert("RGB")
    return img

def _ocr_page_with_tesseract(page, cfg: MultiFormConfig) -> str:
    import pytesseract
    pix = page.get_pixmap(dpi=cfg.ocr_dpi, alpha=False)
    img = _pixmap_to_pil(pix)
    config = f"--psm {cfg.ocr_psm}"
    return pytesseract.image_to_string(img, lang=cfg.ocr_lang, config=config) or ""

def read_all_pages(pdf_path: str, max_pages: Optional[int] = None, cfg: Optional[MultiFormConfig] = None) -> List[Tuple[int, str]]:
    """
    Returns [(page_number, text)] using native text when possible.
    If OCR is enabled and text is scarce (< ocr_min_chars) or force flag is on, OCR the page.
    """
    pages: List[Tuple[int, str]] = []
    use_ocr = bool(cfg and cfg.use_ocr)
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            if max_pages and i > max_pages:
                break
            text = (page.get_text("text") or "").strip()

            if use_ocr:
                need_ocr = (cfg.ocr_force_all if cfg else False) or (len(text) < (cfg.ocr_min_chars if cfg else 25))
                if need_ocr:
                    try:
                        text_ocr = _ocr_page_with_tesseract(page, cfg)
                        # If OCR found something, prefer it; else keep the original text
                        if text_ocr and len(text_ocr.strip()) > len(text):
                            text = text_ocr.strip()
                    except Exception:
                        # silently ignore OCR errors; keep whatever text we have
                        pass

            if text:
                pages.append((i, text))
    return pages




def read_acroform_fields(pdf_path: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    # PyMuPDF widgets
    try:
        with fitz.open(pdf_path) as doc:
            for p in doc:
                for w in (p.widgets() or []):
                    name = (w.field_name or w.name or "").strip()
                    if name:
                        val = w.field_value
                        data[name] = val.strip() if isinstance(val, str) else val
    except Exception:
        pass
    if data:
        return data
    # pypdf fallback
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        fields = reader.get_fields() or {}
        out = {}
        for k, v in fields.items():
            val = v.get("/V") or v.get("/AS")
            out[str(k)] = val if isinstance(val, str) else (str(val) if val is not None else None)
        return out
    except Exception:
        return {}

# ========= Heuristics =========
# Label → target key (English + some Spanish fallbacks because many public forms are bilingual)
LABELS = {
    r"(?:^|\b)(first\s*name|given\s*name|nombre)\b": "first_name",
    r"(?:^|\b)(last\s*name|surname|family\s*name|apellidos?)\b": "last_name",
    r"(?:^|\b)(date\s*of\s*birth|dob|fecha\s*de\s*nacimiento)\b": "date_of_birth",
    r"(?:^|\b)(id\s*number|dni|nie|passport|documento)\b": "id_number",
    r"(?:^|\b)(email|correo)\b": "email",
    r"(?:^|\b)(phone|tel[eé]fono)\b": "phone",
    r"(?:^|\b)(address|direcci[oó]n)\b": "address_line",
    r"(?:^|\b)(city|ciudad|poblaci[oó]n)\b": "city",
    r"(?:^|\b)(state|province|region|estado|provincia|regi[oó]n)\b": "state_region",
    r"(?:^|\b)(postal\s*code|zip|c[oó]digo\s*postal)\b": "postal_code",
    r"(?:^|\b)(country|pa[ií]s)\b": "country",
}

DATE_RE   = re.compile(r"\b(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}|\d{4}[\/\-.]\d{1,2}[\/\-.]\d{1,2}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})\b")
EMAIL_RE  = re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.I)
PHONE_RE  = re.compile(r"\+?\d[\d\s().-]{6,}\d")
POSTAL_RE = re.compile(r"\b\d{4,6}(?:-\d{3,4})?\b")

def harvest_candidates(pages: List[Tuple[int, str]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {k: [] for k in _person_field_names()}
    for page_no, text in pages:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            # 1) Label-based "<Label>: <value>"
            for lbl_re, key in LABELS.items():
                if re.search(lbl_re, line, re.I):
                    parts = re.split(r"[:：]\s*", line, maxsplit=1)
                    value = parts[1].strip() if len(parts) > 1 else ""
                    if value:
                        out[key].append({"value": value, "page": page_no, "line": raw_line, "method": "label", "score": 0.9})

            # 2) Regex catches for lines without explicit labels
            if EMAIL_RE.search(line):
                for m in EMAIL_RE.findall(line):
                    out["email"].append({"value": m, "page": page_no, "line": raw_line, "method": "regex", "score": 0.6})
            if PHONE_RE.search(line):
                ph = max(PHONE_RE.findall(line), key=len)
                out["phone"].append({"value": ph, "page": page_no, "line": raw_line, "method": "regex", "score": 0.55})
            if DATE_RE.search(line):
                out["date_of_birth"].append({"value": DATE_RE.search(line).group(1), "page": page_no, "line": raw_line, "method": "regex", "score": 0.5})
            if "zip" in line.lower() or "postal" in line.lower():
                z = POSTAL_RE.search(line)
                if z:
                    out["postal_code"].append({"value": z.group(0), "page": page_no, "line": raw_line, "method": "regex", "score": 0.5})

            # 3) Address-ish lines: number + street tokens
            if re.search(r"\d{1,5}\s+\S+", line) and any(tok in line.lower() for tok in ["calle", "av", "avenida", "street", "st.", "road", "rd.", "ave", "blvd", "boulevard"]):
                out["address_line"].append({"value": line, "page": page_no, "line": raw_line, "method": "heuristic", "score": 0.5})
    return out

def pick_best_heuristic(cands: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    chosen = {k: None for k in PersonRecord.__fields__.keys()}
    for field, lst in cands.items():
        if not lst:
            continue
        best = sorted(lst, key=lambda x: x.get("score", 0.0), reverse=True)[0]
        chosen[field] = best["value"]
    return chosen

# ========= LLM Adjudication =========
def llm_select_fields(pages: List[Tuple[int,str]], cands: Dict[str, List[Dict[str, Any]]], cfg: MultiFormConfig) -> Dict[str, Any]:
    if not (cfg.allow_llm and cfg.openai_api_key):
        return pick_best_heuristic(cands)

    # Slim evidence (cap per field)
    evidence = {}
    for field, lst in cands.items():
        if lst:
            evidence[field] = [{"value": x["value"], "page": x["page"]} for x in lst[:5]]

    # A few page snippets helps the LLM disambiguate
    sample = "\n\n".join([f"[PAGE {p}] {t[:800]}" for p, t in pages[:4]])

    system = (
        "You are a data extractor. Return a VALID JSON object only, with EXACT keys: "
        "first_name, last_name, date_of_birth (YYYY-MM-DD), id_number, email, phone, "
        "address_line, city, state_region, postal_code, country. "
        "Use null for missing values. Do not include any commentary."
    )
    user = (
        "Below are pre-extracted candidates and short text snippets from the document. "
        "Pick the best value per field (or null if absent). Verify against the snippets when possible.\n\n"
        f"CANDIDATES (JSON):\n{json.dumps(evidence, ensure_ascii=False)}\n\n"
        f"SNIPPETS:\n{sample}\n"
    )

    llm = ChatOpenAI(model=cfg.chat_model, api_key=cfg.openai_api_key, temperature=0)
    raw = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}]).content
    try:
        data = json.loads(raw)
    except Exception:
        import re as _re
        m = _re.search(r"\{.*\}", raw, _re.S)
        data = json.loads(m.group(0)) if m else {}

    try:
        return model_to_dict(PersonRecord(**data))
    except Exception:
        # fall back to heuristics if validation fails
        return pick_best_heuristic(cands)

# ========= Public API =========
def extract_person_from_pdf(pdf_path: str, cfg: MultiFormConfig) -> Dict[str, Any]:
    """
    Strategy:
      1) Try AcroForm → map to target keys → validate.
      2) Else read all pages → harvest candidates → LLM (or heuristic) select → validate.
    Returns: {"mode": "...", "data": {...}, "evidence": {...}}
    """
    # 1) AcroForm
    if cfg.use_acroform:
        fields = read_acroform_fields(pdf_path)
        if fields:
            mapped = _map_acroform_to_person(fields)
            try:
                return {"mode": "acroform", "data": model_to_dict(PersonRecord(**mapped)), "evidence": {"acroform": fields}}
            except Exception:
                # continue to heuristics if mapping partial
                pass

    # 2) Heuristics across all pages
    pages = read_all_pages(pdf_path, max_pages=cfg.max_pages, cfg=cfg)
    cands = harvest_candidates(pages)

    # 3) LLM adjudication or pure heuristic
    selected = llm_select_fields(pages, cands, cfg)
    return {"mode": "hybrid_llm" if cfg.allow_llm else "heuristic_only", "data": selected, "evidence": {"candidates": cands}}

def _map_acroform_to_person(fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Light mapping from common field labels to PersonRecord keys.
    Extend with your own form-specific mappings if needed.
    """
    aliases = {
        # left: lowercased form label; right: PersonRecord key
        "first name": "first_name",
        "firstname": "first_name",
        "given name": "first_name",
        "nombre": "first_name",

        "last name": "last_name",
        "lastname": "last_name",
        "surname": "last_name",
        "apellidos": "last_name",

        "date of birth": "date_of_birth",
        "dob": "date_of_birth",
        "fecha de nacimiento": "date_of_birth",

        "id": "id_number",
        "id number": "id_number",
        "dni": "id_number",
        "nie": "id_number",
        "passport": "id_number",
        "documento": "id_number",

        "email": "email",
        "correo": "email",

        "phone": "phone",
        "teléfono": "phone",
        "telefono": "phone",

        "address": "address_line",
        "dirección": "address_line",
        "direccion": "address_line",

        "city": "city",
        "ciudad": "city",

        "state": "state_region",
        "province": "state_region",
        "region": "state_region",
        "estado": "state_region",
        "provincia": "state_region",
        "región": "state_region",

        "postal code": "postal_code",
        "zip": "postal_code",
        "código postal": "postal_code",
        "codigo postal": "postal_code",

        "country": "country",
        "país": "country",
        "pais": "country",
    }
    out = {k: None for k in PersonRecord.__fields__.keys()}
    for k, v in fields.items():
        lk = str(k).strip().lower()
        # exact alias first
        if lk in aliases:
            out[aliases[lk]] = v if v is None else str(v).strip()
            continue
        # soft match by contains
        for a, target in aliases.items():
            if a in lk and (out[target] is None):
                out[target] = v if v is None else str(v).strip()
                break
    return out
