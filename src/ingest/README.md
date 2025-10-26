

# Ingest (PDF Text + Image Features)

This module ingests **PDFs** into a **Chroma** vector store using **OpenAI embeddings**, with an optional path to index **figures/images** extracted from PDFs (captioned via a vision model, tagged, and stored alongside text).

## What it does

* **Text path:**
  PDF → per-page text → recursive chunking → OpenAI embeddings → Chroma
* **Image path (optional):**
  PDF → extract figures (min size, dedupe via perceptual hash) → caption + tags (vision LLM) → build text blob → OpenAI embeddings → Chroma

Both paths index into the **same Chroma collection**; you can filter by `metadata["modality"]` (e.g., `image`) when querying.


For additional chunking strategies, check https://weaviate.io/blog/chunking-strategies-for-rag

---

## Quick start

```bash
# Default: text-only ingestion from data/raw → storage/chroma
python -m src.ingest.run

# Custom directories & collection
python -m src.ingest.run --docs-dir data/raw --vectordb-dir storage/chroma --collection pdf_chunks

# Tune chunking
python -m src.ingest.run --chunk-size 1200 --chunk-overlap 200

# Clean-reingest (deletes previous vectors for each file before adding)
python -m src.ingest.run --clean

# Also index images: extract figures, caption & tag, store in same collection
python -m src.ingest.run --with-images
python -m src.ingest.run --with-images --docs-dir data/raw
```

Make sure your `.env` (or environment) has:

```env
OPENAI_API_KEY=sk-...

# Embeddings & (optional) vision model
OPENAI_MODEL_EMBED=text-embedding-3-large
OPENAI_MODEL_VISION=gpt-4o-mini

# Defaults for run.py
DOCS_DIR=data/raw
VECTORDB_DIR=storage/chroma
COLLECTION=pdf_chunks
CHUNK_SIZE=1200
CHUNK_OVERLAP=200

# Optional image controls
IMG_MIN_W=150
IMG_MIN_H=150
IMG_MAX_PER_PDF=24
IMAGES_DIR=storage/images
IMAGES_THUMBS_DIR=storage/images/thumbs
```

---

## CLI flags

| Flag                  | Type | Default               | Description                                          |
| --------------------- | ---: | --------------------- | ---------------------------------------------------- |
| `--docs-dir`          |  str | `DOCS_DIR`            | Directory to scan for PDFs (recursive).              |
| `--vectordb-dir`      |  str | `VECTORDB_DIR`        | Chroma persistence directory.                        |
| `--collection`        |  str | `COLLECTION`          | Chroma collection name.                              |
| `--chunk-size`        |  int | `CHUNK_SIZE`          | Text chunk size (chars).                             |
| `--chunk-overlap`     |  int | `CHUNK_OVERLAP`       | Overlap (chars).                                     |
| `--embed-model`       |  str | `OPENAI_MODEL_EMBED`  | OpenAI embeddings model.                             |
| `--clean`             | flag | `False`               | Delete existing vectors for a file before re-adding. |
| `--with-images`       | flag | `False`               | Enable image extraction/captioning/indexing.         |
| `--vision-model`      |  str | `OPENAI_MODEL_VISION` | Vision LLM for captions.                             |
| `--img-min-w`         |  int | `IMG_MIN_W`           | Minimum image width to index.                        |
| `--img-min-h`         |  int | `IMG_MIN_H`           | Minimum image height to index.                       |
| `--img-max-per-pdf`   |  int | `IMG_MAX_PER_PDF`     | Hard cap per PDF.                                    |
| `--images-dir`        |  str | `IMAGES_DIR`          | Where originals are saved (PNG).                     |
| `--images-thumbs-dir` |  str | `IMAGES_THUMBS_DIR`   | Where 320px thumbs are saved.                        |

> Note: `run.py` creates the `docs_dir`, `vectordb_dir`, `images_dir`, and `thumbs_dir` if missing.

---

## How it works

### 1) Text ingestion (`run.py`)

* **Extraction:** `fitz` (PyMuPDF) reads each PDF page; blank pages are skipped.
* **Chunking:** `RecursiveCharacterTextSplitter` with separators `["\n\n","\n",". "," ",""]`.
* **Stable IDs:** Each chunk gets a deterministic SHA-1 based on `(source path, page, chunk index, first 64 chars)` so re-runs don’t duplicate.
* **Indexing:** `OpenAIEmbeddings` → `Chroma.add_documents(...)` → `vs.persist()`.

### 2) Image ingestion (`image_features.py`)

* **Extraction:** For each page, collect embedded images (`page.get_images(full=True)`) and convert via xref to RGB PNG (`fitz.Pixmap → PIL.Image`).
* **Filtering:** Skip images smaller than `min_width × min_height`. Limit to `max_images_per_pdf`.
* **De-dup:** Perceptual hash (`imagehash.phash`) to avoid near-duplicates across the same PDF run.
* **Storage:** Save **originals** & **thumbnails** (`320×320`) under `IMAGES_DIR` / `IMAGES_THUMBS_DIR`.
* **Caption + tags:** Send a **data URL** of the PNG to the vision model (`ChatOpenAI`).
  Prompt asks for JSON: `{"caption": "...", "tags": ["..."]}`.
  Fallback: return a safe stub `"(uncaptioned figure)"` if parsing fails.
* **Text blob:** `Caption: …\nTags: …\nOCR: …` (OCR currently unused; placeholder).
* **Indexing:** Embed the text blob and upsert a `Document` with **image metadata**.

---

## Data model (metadata)

### Text chunks

```json
{
  "source": "/abs/path/to/file.pdf",
  "file_name": "file.pdf",
  "page": 3,
  "chunk": 1
}
```

### Image records

```json
{
  "modality": "image",
  "file_name": "file.pdf",
  "doc_id": "file",               // stem
  "page": 3,
  "width": 1240,
  "height": 820,
  "image_hash": "a1b2c3...",
  "image_path": "storage/images/file_p3_i2.png",
  "thumb_path": "storage/images/thumbs/file_p3_i2.png"
}
```

**Page content** for images is a compact text blob:

```
Caption: A line chart showing quarterly revenue by region...
Tags: chart, revenue, quarters, region, line
```

---

## Querying examples

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

emb = OpenAIEmbeddings(model="text-embedding-3-large")
vs = Chroma(collection_name="pdf_chunks", persist_directory="storage/chroma", embedding_function=emb)

# 1) Generic text retrieval
hits = vs.similarity_search("Explain the training pipeline stages", k=5)

# 2) Image-only retrieval (use metadata filter)
img_hits = vs.similarity_search("bar chart of monthly sales", k=4, where={"modality": "image"})

# 3) Restrict to one file
file_hits = vs.similarity_search("security requirements", k=4, where={"file_name": "policy.pdf"})
```

---

## Rate limits, retries & throttling (images)

`image_features.ImageIngestConfig` includes:

* `sleep_ms_between_calls` (default **300ms**) — soft throttle between caption calls
* `max_retries` (default **4**) — retries on HTTP 429/temporary errors with exponential backoff
* `backoff_base` (default **0.6s**) — grows with attempts; optional jitter

> **Wiring note:** In `run.py` you already expose `--img-sleep-ms`, `--img-retries`, `--img-backoff-base`, but the constructor call to `ImageIngestConfig(...)` doesn’t pass them yet. If you want those flags to take effect, update the call like:

```python
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
    sleep_ms_between_calls=cfg.img_sleep_ms,
    max_retries=cfg.img_retries,
    backoff_base=cfg.img_backoff_base,
)
```

This is especially helpful if you’ve hit `429` rate limits.

---

## Clean re-ingest & idempotency

* `--clean` removes **existing** vectors for the file before adding new ones:

  * Text: filter by `{"source": "<abs path>"}`.
  * Images: filter by `{"file_name": "<name>", "modality": "image"}`.
* Text chunk IDs are deterministic → re-runs won’t duplicate when `--clean` is not used (but updates to content/params will create new IDs).

---

## Expected folders

```
storage/
  chroma/                 # Chroma index (persisted)
  images/
    <doc>_p<page>_i<idx>.png
    thumbs/
      <doc>_p<page>_i<idx>.png
data/
  raw/                     # Put your PDFs here
```

---

## Dependencies

* PyMuPDF (`fitz`) for PDF parsing
* Pillow (`PIL`) for image handling
* `imagehash` for perceptual hashing
* LangChain (core, community, text_splitters), `langchain_openai`
* Chroma

> Install via your project’s `requirements.txt` or:
> `pip install pymupdf pillow ImageHash langchain langchain-openai langchain-community langchain-text-splitters chromadb python-dotenv`

---

## Troubleshooting

* **`OPENAI_API_KEY` missing:** set in `.env` or environment.
* **429 / rate limits:** enable `--with-images` controls, wire throttling (see wiring note), or reduce `IMG_MAX_PER_PDF`.
* **PDF has no text:** scans/rasterized PDFs may yield empty text; consider OCR at ingest (not implemented here).
* **No images found:** PDFs that embed pages as a single raster may not expose separate figure xrefs.
* **Chroma locks/errors:** close other processes using the same `persist_directory` before re-ingesting.

---

## Extending

* **OCR for images:** Add OCR text and pass into `build_text_blob(caption, tags, extra_ocr_text=...)`.
* **Alternate vector DBs:** Swap `Chroma` with FAISS, Milvus, etc.
* **Different vision models:** Any ChatCompletions-style vision-capable model supported by `langchain_openai.ChatOpenAI`.
* **Modality routing:** Keep text and image embeddings in **one** collection (current approach) or split into separate collections and merge at query time.

---

## Changelog

* **v0.2** — Image ingestion (captions, tags, dedupe, thumbnails), clean-delete for images, same-collection design.
* **v0.1** — PDF text chunking + embeddings + Chroma, stable IDs.

---
