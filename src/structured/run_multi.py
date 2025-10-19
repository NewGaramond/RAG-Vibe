# src/structured/run_multi.py
from __future__ import annotations
import os, json
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

from src.structured.multiform_extractor import MultiFormConfig, extract_person_from_pdf

@dataclass
class Args:
    pdfs: list[Path]
    out_dir: Path
    allow_llm: bool
    max_pages: int | None = None
    # OCR knobs (give DEFAULTS so construction never fails)
    ocr: bool = False
    ocr_force_all: bool = False
    ocr_lang: str = "eng"
    ocr_min_chars: int = 25
    ocr_dpi: int = 300
    ocr_psm: str = "6"

def parse_args() -> Args:
    import argparse
    load_dotenv()
    p = argparse.ArgumentParser(
        description="Cross-PDF person extractor (AcroForm → Heuristics → optional LLM/OCR)."
    )
    p.add_argument("--pdf", action="append", default=[], help="Path to a PDF (repeatable).")
    p.add_argument("--pdf-dir", type=str, default=None, help="Directory containing PDFs (glob *.pdf)")
    p.add_argument("--out-dir", type=str, default="storage/structured")
    p.add_argument("--no-llm", action="store_true", help="Disable LLM adjudication (heuristics only).")
    p.add_argument("--max-pages", type=int, default=None, help="Limit number of pages scanned per PDF.")
    # OCR flags:
    p.add_argument("--ocr", action="store_true", help="Enable OCR fallback.")
    p.add_argument("--ocr-force-all", action="store_true", help="Force OCR on every page.")
    p.add_argument("--ocr-lang", type=str, default="eng", help="Tesseract languages, e.g. 'eng' or 'eng+spa'.")
    p.add_argument("--ocr-min-chars", type=int, default=25, help="If native text has fewer chars, OCR the page.")
    p.add_argument("--ocr-dpi", type=int, default=300, help="Rasterization DPI for OCR.")
    p.add_argument("--ocr-psm", type=str, default="6", help="Tesseract PSM (e.g., '3','4','6').")

    
    a = p.parse_args()

    pdfs = [Path(x) for x in a.pdf]
    if a.pdf_dir:
        pdfs += list(Path(a.pdf_dir).glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs provided. Use --pdf multiple times or --pdf-dir.")

    return Args(
        pdfs=pdfs,
        out_dir=Path(a.out_dir),
        allow_llm=(not a.no_llm),
        max_pages=a.max_pages,
        ocr=a.ocr,
        ocr_force_all=a.ocr_force_all,
        ocr_lang=a.ocr_lang,
        ocr_min_chars=a.ocr_min_chars,
        ocr_dpi=a.ocr_dpi,
        ocr_psm=a.ocr_psm,
    )

def main():
    args = parse_args()

    key = os.getenv("OPENAI_API_KEY")
    if args.allow_llm and not key:
        raise RuntimeError("OPENAI_API_KEY missing. Set it or run with --no-llm.")

    cfg = MultiFormConfig(
        openai_api_key=key,
        allow_llm=args.allow_llm,
        max_pages=args.max_pages,
        use_acroform=True,
        # OCR:
        use_ocr=args.ocr,
        ocr_force_all=args.ocr_force_all,
        ocr_lang=args.ocr_lang,
        ocr_min_chars=args.ocr_min_chars,
        ocr_dpi=args.ocr_dpi,
        ocr_psm=args.ocr_psm,
    )

    
    
    
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for pdf in args.pdfs:
        res = extract_person_from_pdf(str(pdf), cfg)
        out_path = args.out_dir / f"{pdf.stem}.person.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"pdf": str(pdf), **res}, f, ensure_ascii=False, indent=2, default=str)
        print(f"[OK] {res['mode']} -> {out_path}")

if __name__ == "__main__":
    main()


