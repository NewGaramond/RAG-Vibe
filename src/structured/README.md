# Structured Extraction (Multi-Form Person Extractor)

Cross-PDF structured data extraction for **person records**. The pipeline tries **AcroForm** fields first, then falls back to **heuristics + regex** over page text, with optional **LLM adjudication** and **OCR** for low-text PDFs.

```
PDF ──┬── AcroForm fields → light mapping → validate (Pydantic)
      └── Read pages → harvest candidates (labels + regex + heuristics)
                      └── (optional) LLM picks best → validate
```

---

## Files

* `multiform_extractor.py` — core logic (PDF I/O, heuristics, optional OCR, optional LLM adjudication).
* `run_multi.py` — CLI driver (batch over files/dirs; writes JSON to `storage/structured/`).
* `schemas.py` — Pydantic models and helpers (currently `PersonRecord`).

---

## Quick start (CLI)

```bash
# Heuristics only (no LLM), process a folder of PDFs
python -m src.structured.run_multi --pdf-dir data/raw/forms --no-llm

# Use LLM adjudication (requires OPENAI_API_KEY)
python -m src.structured.run_multi --pdf-dir data/raw/forms

# Enable OCR fallback (needs Tesseract installed)
python -m src.structured.run_multi --pdf-dir data/raw/forms --ocr --ocr-lang "eng+spa"

# Limit scanned pages for large docs
python -m src.structured.run_multi --pdf-dir data/raw/forms --max-pages 6
```

Outputs one JSON per input PDF:

```
storage/structured/<pdf-stem>.person.json
```

---

## Environment

```env
OPENAI_API_KEY=sk-...        # needed only if you don't pass --no-llm
OPENAI_MODEL_CHAT=gpt-4o-mini
```

---

## CLI flags (`run_multi.py`)

| Flag              |       Type | Default              | Description                                    |
| ----------------- | ---------: | -------------------- | ---------------------------------------------- |
| `--pdf`           | repeatable | —                    | Path(s) to individual PDFs.                    |
| `--pdf-dir`       |        str | —                    | Process all `*.pdf` in directory.              |
| `--out-dir`       |        str | `storage/structured` | Where JSON outputs are written.                |
| `--no-llm`        |       flag | false                | Disable LLM adjudication (heuristics only).    |
| `--max-pages`     |        int | `None`               | Upper bound of pages per PDF to scan.          |
| `--ocr`           |       flag | false                | Enable OCR fallback per page.                  |
| `--ocr-force-all` |       flag | false                | OCR **every** page (ignores native text).      |
| `--ocr-lang`      |        str | `eng`                | Tesseract languages (`"eng"`, `"eng+spa"`, …). |
| `--ocr-min-chars` |        int | `25`                 | If native text length < N, OCR that page.      |
| `--ocr-dpi`       |        int | `300`                | Rasterization DPI for OCR.                     |
| `--ocr-psm`       |        str | `6`                  | Tesseract PSM (e.g., `3`, `4`, `6`).           |

---

## Programmatic use

```python
from src.structured.multiform_extractor import MultiFormConfig, extract_person_from_pdf

cfg = MultiFormConfig(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    allow_llm=True,        # or False for heuristics-only
    use_acroform=True,
    use_ocr=True,          # enable OCR fallback if needed
    ocr_lang="eng+spa",
    max_pages=8,
)

res = extract_person_from_pdf("data/raw/forms/sample.pdf", cfg)
print(res["mode"])   # 'acroform' | 'hybrid_llm' | 'heuristic_only'
print(res["data"])   # normalized PersonRecord dict
```

---

## Output format

Each run returns (and the CLI writes) a JSON like:

```json
{
  "pdf": "data/raw/forms/sample.pdf",
  "mode": "hybrid_llm",
  "data": {
    "first_name": "María",
    "last_name": "García",
    "date_of_birth": "1990-05-12",
    "id_number": "X1234567",
    "email": "maria.garcia@example.com",
    "phone": "+34 600 123 456",
    "address_line": "Calle Mayor 10, 3ºB",
    "city": "Madrid",
    "state_region": "Comunidad de Madrid",
    "postal_code": "28013",
    "country": "España"
  },
  "evidence": {
    "acroform": { "...": "..." },     // present if mode='acroform'
    "candidates": { "...": [ ... ] }  // present for heuristic/LLM modes
  }
}
```

### Validation & normalization

* `schemas.PersonRecord` (Pydantic) ensures fields are well-formed.
* `date_of_birth` accepts many formats; parsed to `YYYY-MM-DD` (day-first enabled).
* `model_to_dict(...)` returns JSON-safe primitives.

---

## How it works

### 1) AcroForm path

* Reads form widgets via **PyMuPDF**; if empty, falls back to **pypdf**’s `get_fields()`.
* Maps common labels/aliases (EN/ES) to the normalized keys (e.g., `"nombre"` → `first_name`).
* Validates with `PersonRecord`. If valid → **done**.

### 2) Heuristics path

* **Read pages** (`fitz`): native text; optionally OCR pages where text is scarce (`< ocr_min_chars`) or `--ocr-force-all`.
* **Candidate harvest** (`harvest_candidates`):

  * **Label match** per line: `"First name:"`, `"Apellidos:"`, `"Fecha de nacimiento:"`, etc. (EN/ES regex).
  * **Regex catchers**: email, phone, date, postal code.
  * **Address-ish** heuristic: number + street tokens (`calle`, `avenida`, `street`, `rd`, `blvd`, …).
  * Scores and keeps page/evidence metadata.
* **Selection**:

  * If `allow_llm=True`: LLM receives capped candidate lists + short page snippets and returns a **single JSON** with all required keys (missing → `null`). Output is validated; on failure, falls back to heuristics.
  * If `allow_llm=False`: choose **best-scoring** candidate per field.

---

## Schema (`schemas.py`)

```python
class PersonRecord(BaseModel):
    first_name: Optional[str]
    last_name: Optional[str]
    date_of_birth: Optional[date]     # normalized
    id_number: Optional[str]
    email: Optional[EmailStr]
    phone: Optional[str]
    address_line: Optional[str]
    city: Optional[str]
    state_region: Optional[str]
    postal_code: Optional[str]
    country: Optional[str]
```

> `SCHEMAS["person"]` is included as a description stub; extendable later for multi-schema dispatch.

---

## OCR notes

* **Requires**: `pytesseract` (Python pkg) **and** the **Tesseract** binary on your system (`tesseract --version`).
* Rasterization uses `fitz` at `ocr_dpi` (default 300). Higher DPI → better OCR but slower.
* Use `--ocr-lang "eng+spa"` for bilingual forms.

---

## Dependencies

Required:

* `pymupdf` (`fitz`) — PDF parsing/text & rasterization
* `pydantic` — schema/validation
* `python-dateutil` — flexible date parsing
* `langchain-openai` — LLM adjudication (optional if `--no-llm`)
* `pypdf` — AcroForm fallback

Optional (if OCR):

* `pytesseract` + system **Tesseract** binary
* `Pillow` (implicitly used for raster → PIL)

Install example:

```bash
pip install pymupdf pydantic python-dateutil langchain-openai pypdf pytesseract pillow
```

---

## Heuristics at a glance

* **Labels → keys** (EN/ES):

  * `first_name`, `last_name`, `date_of_birth`, `id_number`, `email`, `phone`,
    `address_line`, `city`, `state_region`, `postal_code`, `country`.
* **Regex**:

  * Email: RFC-lite
  * Phone: `+?\d[ \d().-]{6,}\d`
  * Date: `DD/MM/YYYY`, `YYYY-MM-DD`, `"May 12, 1990"`, …
  * Postal code: 4–6 digits, optional `-NNN`
* **Address-ish**: simple numeric street heuristic + common tokens (ES/EN).

> Heuristics are intentionally conservative to reduce junk; tune patterns for your corpus.

---

## Testing (suggested)

Create `tests/structured/`:

```python
# tests/structured/test_schemas.py
from src.structured.schemas import PersonRecord
def test_date_parsing():
    p = PersonRecord(date_of_birth="12/05/1990")
    assert str(p.date_of_birth) == "1990-05-12"

# tests/structured/test_multiform_heuristics.py
from src.structured.multiform_extractor import harvest_candidates
def test_email_and_phone_regex():
    pages = [(1, "Email: a@b.com\nTeléfono: +34 600 123 456")]
    c = harvest_candidates(pages)
    assert c["email"] and c["phone"]

# tests/structured/test_runner.py
# Provide a tiny fixture PDF (vector text) in tests/fixtures/
```

Run:

```bash
pytest -q
```

---

## Troubleshooting

* **“No PDFs provided”** → pass `--pdf` (repeatable) or `--pdf-dir`.
* **Empty text** → try `--ocr` (scanned PDFs often lack text objects).
* **OCR poor quality** → raise `--ocr-dpi`, add `--ocr-lang`, or `--ocr-force-all`.
* **LLM path fails** → ensure `OPENAI_API_KEY`; use `--no-llm` to run heuristics-only.
* **Mixed languages/labels** → add new label patterns or extend `_map_acroform_to_person()` aliases.

---

## Extending

* **New schemas**: add models to `schemas.py`, extend `SCHEMAS`, and implement dispatch in `multiform_extractor.py`.
* **Better addresses**: add country-specific street regexes or use a lightweight parser.
* **Normalization**: plug a phone/ID normalizer before validation.
* **Provenance**: persist page/line evidence with final picks for auditing (partially included now).
* **Batch outputs**: push results to a DB or append to a CSV for analytics.

---

## Privacy & compliance

This module extracts **personal data**. If you store outputs, consider:

* data minimization (only needed fields),
* retention periods,
* access controls/encryption at rest,
* audit trail if used in regulated contexts.

---

## Changelog

* **v0.2** — OCR fallback, bilingual label set, stronger evidence capture, CLI polishing.
* **v0.1** — AcroForm → Heuristics → (optional) LLM adjudication for `PersonRecord`.

---
