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
    max_pages: int | None

def parse_args() -> Args:
    import argparse
    load_dotenv()
    p = argparse.ArgumentParser(description="Cross-PDF person extractor (AcroForm → Heuristics → optional LLM).")
    p.add_argument("--pdf", action="append", default=[], help="Path to a PDF (repeatable).")
    p.add_argument("--pdf-dir", type=str, default=None, help="Directory containing PDFs (glob *.pdf)")
    p.add_argument("--out-dir", type=str, default="storage/structured")
    p.add_argument("--no-llm", action="store_true", help="Disable LLM adjudication (heuristics only).")
    p.add_argument("--max-pages", type=int, default=None, help="Limit number of pages scanned per PDF.")
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
    )

def main():
    args = parse_args()

    key = os.getenv("OPENAI_API_KEY")
    if args.allow_llm and not key:
        raise RuntimeError("OPENAI_API_KEY missing. Set it or run with --no-llm.")

    cfg = MultiFormConfig(openai_api_key=key, allow_llm=args.allow_llm, max_pages=args.max_pages)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for pdf in args.pdfs:
        res = extract_person_from_pdf(str(pdf), cfg)
        out_path = args.out_dir / f"{pdf.stem}.person.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"pdf": str(pdf), **res}, f, ensure_ascii=False, indent=2, default=str)
        print(f"[OK] {res['mode']} -> {out_path}")

if __name__ == "__main__":
    main()
