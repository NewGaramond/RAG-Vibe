# src/ingest/run.py
"""
PDF → chunks → embeddings → Chroma (OpenAI-only)

Usage:
  python -m src.ingest.run
  python -m src.ingest.run --docs-dir data/raw --vectordb-dir storage/chroma --collection pdf_chunks
  python -m src.ingest.run --chunk-size 1200 --chunk-overlap 200 --top-k 5
  python -m src.ingest.run --clean   # deletes existing vectors for matching files before re-adding

Env (.env):
  OPENAI_API_KEY=sk-...
  OPENAI_MODEL_EMBED=text-embedding-3-large
  DOCS_DIR=data/raw
  VECTORDB_DIR=storage/chroma
  CHUNK_SIZE=1200
  CHUNK_OVERLAP=200
  TOP_K=5
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

# PDF
import fitz  # PyMuPDF

import argparse, os
from dotenv import load_dotenv
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
    clean: bool = False  # if True, remove existing docs for a file before re-adding

    @staticmethod
    def from_env_and_args() -> "IngestConfig":
        load_dotenv()
        parser = argparse.ArgumentParser(description="Ingest PDFs into Chroma with OpenAI embeddings.")
        parser.add_argument("--docs-dir", type=str, default=os.getenv("DOCS_DIR", "data/raw"))
        parser.add_argument("--vectordb-dir", type=str, default=os.getenv("VECTORDB_DIR", "storage/chroma"))
        parser.add_argument("--collection", type=str, default=os.getenv("COLLECTION", "pdf_chunks"))
        parser.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", "1200")))
        parser.add_argument("--chunk-overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", "200")))
        parser.add_argument("--embed-model", type=str, default=os.getenv("OPENAI_MODEL_EMBED", "text-embedding-3-large"))
        parser.add_argument("--clean", action="store_true", help="Delete existing vectors per file before re-adding.")
        args = parser.parse_args()

        return IngestConfig(
            docs_dir=Path(args.docs_dir),
            vectordb_dir=Path(args.vectordb_dir),
            collection=args.collection,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embed_model=args.embed_model,
            clean=args.clean,
        )


# -------------------------
# Logging
# -------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


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
    Delete existing vectors for a given file by metadata filter.
    """
    try:
        deleted = vs.delete(where={"source": str(file_path.resolve())})
        logging.info(f"Deleted {deleted} existing chunks for file: {file_path.name}")
    except Exception as e:
        logging.warning(f"Delete skipped/failed for {file_path.name}: {e}")


# -------------------------
# Ingest one PDF
# -------------------------
def ingest_pdf(vs: Chroma, file_path: Path, cfg: IngestConfig) -> int:
    logging.info(f"Ingesting: {file_path.name}")

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

    # Upsert-like behavior: try add; duplicates are naturally skipped in new Chroma versions if IDs repeat.
    # If your Chroma version errors on duplicate IDs, run with --clean to remove first.
    vs.add_documents(documents=docs, ids=ids)
    logging.info(f"Added {len(docs)} chunks from {file_path.name}")
    return len(docs)


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

    # Embeddings
    embeddings = OpenAIEmbeddings(model=cfg.embed_model, api_key=api_key)

    # Vector store
    vs = get_chroma_store(cfg, embeddings)

    # Find PDFs
    pdfs = sorted([p for p in cfg.docs_dir.rglob("*.pdf") if p.is_file()])
    if not pdfs:
        logging.warning(f"No PDFs found in: {cfg.docs_dir.resolve()}")
        return

    total_chunks = 0
    for pdf in pdfs:
        total_chunks += ingest_pdf(vs, pdf, cfg)

    # Persist
    vs.persist()
    logging.info(f"Done. Total chunks added: {total_chunks}")
    logging.info(f"Chroma persisted at: {cfg.vectordb_dir.resolve()}")
    logging.info(f"Collection: {cfg.collection}")


if __name__ == "__main__":
    main()
