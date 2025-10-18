# src/ingest/run.py
"""
PDF → chunks → embeddings → Chroma (OpenAI-only)

Usage:
  python -m src.ingest.run
  python -m src.ingest.run --docs-dir data/raw --vectordb-dir storage/chroma --collection pdf_chunks
  python -m src.ingest.run --chunk-size 1200 --chunk-overlap 200 --top-k 5
  python -m src.ingest.run --clean
  # NEW: also index images (captions + tags)
  python -m src.ingest.run --with-images
  python -m src.ingest.run --with-images --docs-dir data/raw

Env (.env):
  OPENAI_API_KEY=sk-...
  OPENAI_MODEL_EMBED=text-embedding-3-large
  # optional: vision model for captions
  OPENAI_MODEL_VISION=gpt-4o-mini

  DOCS_DIR=data/raw
  VECTORDB_DIR=storage/chroma
  COLLECTION=pdf_chunks
  CHUNK_SIZE=1200
  CHUNK_OVERLAP=200

  # NEW (optional)
  IMG_MIN_W=150
  IMG_MIN_H=150
  IMG_MAX_PER_PDF=24
  IMAGES_DIR=storage/images
  IMAGES_THUMBS_DIR=storage/images/thumbs
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from dotenv import load_dotenv

# LangChain modern imports (v0.3+)
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# PDF text
import fitz  # PyMuPDF

# NEW: image features ingestion
from src.ingest.image_features import ImageIngestConfig, ingest_pdf_images


# -------------------------
# Configuration
# -------------------------
@dataclass
class IngestConfig:
    docs_dir: Path
    vectordb_dir: Path
    collection: str = "pdf_chunks"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    embed_model: str = "text-embedding-3-large"
    clean: bool = False  # if True, remove existing vectors for a file before re-adding
    # NEW:
    with_images: bool = False
    vision_model: str = "gpt-4o-mini"
    img_min_w: int = 150
    img_min_h: int = 150
    img_max_per_pdf: int = 24
    images_dir: Path = Path("storage/images")
    thumbs_dir: Path = Path("storage/images/thumbs")
    img_sleep_ms: int = 300
    img_retries: int = 4
    img_backoff_base: float = 0.6

    @staticmethod
    def from_env_and_args() -> "IngestConfig":
        load_dotenv()
        parser = argparse.ArgumentParser(description="Ingest PDFs (text + optional images) into Chroma with OpenAI embeddings.")
        parser.add_argument("--docs-dir", type=str, default=os.getenv("DOCS_DIR", "data/raw"))
        parser.add_argument("--vectordb-dir", type=str, default=os.getenv("VECTORDB_DIR", "storage/chroma"))
        parser.add_argument("--collection", type=str, default=os.getenv("COLLECTION", "pdf_chunks"))
        parser.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "1200")))
        parser.add_argument("--chunk-overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "200")))
        parser.add_argument("--embed-model", type=str, default=os.getenv("OPENAI_MODEL_EMBED", "text-embedding-3-large"))
        parser.add_argument("--clean", action="store_true", help="Delete existing vectors per file before re-adding.")
        # NEW flags for image ingestion
        parser.add_argument("--with-images", action="store_true", help="Extract figures, caption/tag them, and index into Chroma.")
        parser.add_argument("--vision-model", type=str, default=os.getenv("OPENAI_MODEL_VISION", os.getenv("OPENAI_MODEL_CHAT", "gpt-4o-mini")))
        parser.add_argument("--img-min-w", type=int, default=int(os.getenv("IMG_MIN_W", "150")))
        parser.add_argument("--img-min-h", type=int, default=int(os.getenv("IMG_MIN_H", "150")))
        parser.add_argument("--img-max-per-pdf", type=int, default=int(os.getenv("IMG_MAX_PER_PDF", "24")))
        parser.add_argument("--images-dir", type=str, default=os.getenv("IMAGES_DIR", "storage/images"))
        parser.add_argument("--images-thumbs-dir", type=str, default=os.getenv("IMAGES_THUMBS_DIR", "storage/images/thumbs"))
        parser.add_argument("--img-sleep-ms", type=int, default=int(os.getenv("IMG_SLEEP_MS", "300")))
        parser.add_argument("--img-retries", type=int, default=int(os.getenv("IMG_RETRIES", "4")))
        parser.add_argument("--img-backoff-base", type=float, default=float(os.getenv("IMG_BACKOFF_BASE", "0.6")))

        args = parser.parse_args()

        return IngestConfig(
            docs_dir=Path(args.docs_dir),
            vectordb_dir=Path(args.vectordb_dir),
            collection=args.collection,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embed_model=args.embed_model,
            clean=args.clean,
            # NEW:
            with_images=args.with_images,
            vision_model=args.vision_model,
            img_min_w=args.img_min_w,
            img_min_h=args.img_min_h,
            img_max_per_pdf=args.img_max_per_pdf,
            images_dir=Path(args.images_dir),
            thumbs_dir=Path(args.images_thumbs_dir),
        )


# -------------------------
# Logging
# -------------------------
def setup_logging():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# -------------------------
# PDF → pages (text)
# -------------------------
def extract_pdf_pages(pdf_path: Path) -> Iterable[Tuple[int, str]]:
    """
    Yields (page_number, text) for each page with non-empty text.
    Page numbers are 1-based to match user expectations.
    """
    try:
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                text = text.strip()
                if text:
                    yield (i, text)
    except Exception as e:
        logging.error(f"Failed to read PDF '{pdf_path}': {e}")


# -------------------------
# Chunking
# -------------------------
def chunk_pages(
    pages: Iterable[Tuple[int, str]],
    file_path: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: List[Document] = []
    for page_num, text in pages:
        chunks = splitter.split_text(text)
        for idx, chunk in enumerate(chunks):
            metadata = {
                "source": str(file_path.resolve()),
                "file_name": file_path.name,
                "page": page_num,
                "chunk": idx,
            }
            documents.append(Document(page_content=chunk, metadata=metadata))
    return documents


# -------------------------
# Stable IDs (idempotent)
# -------------------------
def make_stable_id(doc: Document) -> str:
    """
    Create a stable ID per chunk so reruns don't duplicate.
    """
    base = f"{doc.metadata.get('source','')}|p{doc.metadata.get('page','')}|c{doc.metadata.get('chunk','')}|{doc.page_content[:64]}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


# -------------------------
# Vector store (Chroma)
# -------------------------
def get_chroma_store(cfg: IngestConfig, embeddings: OpenAIEmbeddings) -> Chroma:
    return Chroma(
        collection_name=cfg.collection,
        embedding_function=embeddings,
        persist_directory=str(cfg.vectordb_dir),
    )


def delete_docs_for_file(vs: Chroma, file_path: Path):
    """
    Delete existing TEXT chunks for a given file by metadata filter.
    """
    try:
        deleted = vs.delete(where={"source": str(file_path.resolve())})
        logging.info(f"Deleted {deleted} existing text chunks for: {file_path.name}")
    except Exception as e:
        logging.warning(f"Delete skipped/failed for {file_path.name}: {e}")


# NEW: delete image records for a file (modality=image)
def delete_images_for_file(vs: Chroma, file_path: Path):
    try:
        deleted = vs.delete(where={"file_name": file_path.name, "modality": "image"})
        logging.info(f"Deleted {deleted} existing image records for: {file_path.name}")
    except Exception as e:
        logging.warning(f"Delete images skipped/failed for {file_path.name}: {e}")


# -------------------------
# Ingest one PDF (text)
# -------------------------
def ingest_pdf_text(vs: Chroma, file_path: Path, cfg: IngestConfig) -> int:
    logging.info(f"[text] Ingesting: {file_path.name}")

    if cfg.clean:
        delete_docs_for_file(vs, file_path)

    pages = list(extract_pdf_pages(file_path))
    if not pages:
        logging.warning(f"No extractable text in: {file_path.name}")
        return 0

    docs = chunk_pages(pages, file_path, cfg.chunk_size, cfg.chunk_overlap)
    if not docs:
        logging.warning(f"No chunks produced for: {file_path.name}")
        return 0

    ids = [make_stable_id(d) for d in docs]
    vs.add_documents(documents=docs, ids=ids)
    logging.info(f"[text] Added {len(docs)} chunks from {file_path.name}")
    return len(docs)


# -------------------------
# Ingest one PDF (images)
# -------------------------
def ingest_pdf_images_wrapper(file_path: Path, cfg: IngestConfig, api_key: str):
    logging.info(f"[images] Ingesting figures: {file_path.name}")

    # If cleaning, remove old image records for this file
    if cfg.clean:
        # Use a temporary embeddings object just to get a VS handle for deletion
        embeddings = OpenAIEmbeddings(model=cfg.embed_model, api_key=api_key)
        vs = get_chroma_store(cfg, embeddings)
        delete_images_for_file(vs, file_path)

    img_cfg = ImageIngestConfig(
        openai_api_key=api_key,
        embed_model=cfg.embed_model,
        vectordb_dir=str(cfg.vectordb_dir),
        collection=cfg.collection,
        caption_model=cfg.vision_model,
        min_width=cfg.img_min_w,
        min_height=cfg.img_min_h,
        max_images_per_pdf=cfg.img_max_per_pdf,
        images_dir=str(cfg.images_dir),
        thumbs_dir=str(cfg.thumbs_dir),
    )
    results = ingest_pdf_images(str(file_path), img_cfg)
    logging.info(f"[images] {file_path.name}: indexed {len(results)} figures")


# -------------------------
# Main
# -------------------------
def main():
    setup_logging()
    cfg = IngestConfig.from_env_and_args()

    # Validate API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in your .env or environment.")

    # Ensure dirs exist
    cfg.docs_dir.mkdir(parents=True, exist_ok=True)
    cfg.vectordb_dir.mkdir(parents=True, exist_ok=True)
    cfg.images_dir.mkdir(parents=True, exist_ok=True)
    cfg.thumbs_dir.mkdir(parents=True, exist_ok=True)

    # Embeddings + Vector store (for text path)
    embeddings = OpenAIEmbeddings(model=cfg.embed_model, api_key=api_key)
    vs = get_chroma_store(cfg, embeddings)

    # Find PDFs
    pdfs = sorted([p for p in cfg.docs_dir.rglob("*.pdf") if p.is_file()])
    if not pdfs:
        logging.warning(f"No PDFs found in: {cfg.docs_dir.resolve()}")
        return

    total_chunks = 0
    for pdf in pdfs:
        total_chunks += ingest_pdf_text(vs, pdf, cfg)
        if cfg.with_images:
            ingest_pdf_images_wrapper(pdf, cfg, api_key)

    # Persist
    vs.persist()
    logging.info(f"Done. Total text chunks added: {total_chunks}")
    logging.info(f"Chroma persisted at: {cfg.vectordb_dir.resolve()}")
    logging.info(f"Collection: {cfg.collection}")


if __name__ == "__main__":
    main()
