# Chatbot RAG — Guarded LangGraph RAG with PDF Ingestion, Image Search & Structured Extraction

A production-minded Retrieval-Augmented Generation (RAG) stack that ingests PDFs (text **and** figures), protects against prompt-injection, plans tool use, runs a **safe** Python mini-calculator, and maintains lightweight conversational memory. Includes an optional module for **structured data extraction** from forms (e.g., person records) with OCR and LLM adjudication.

## Highlights

* **End-to-end RAG**: PDF → chunks → embeddings → **Chroma** → LangGraph RAG.
* **Figures aware**: Extracts images from PDFs, captions them (vision LLM), and indexes captions + tags for image-aware retrieval.
* **Safety first**: Heuristic **guard** for prompt-injection + **safe Python** evaluator (single expression, allow-listed).
* **Planner**: LLM “decides” per turn whether to retrieve, compute, or refuse.
* **Memory**: Token-budgeted running summary + last-turns verbatim.
* **Structured extraction**: Multi-form **person** extractor (AcroForm → heuristics/regex → optional LLM) with optional **OCR**.
* **Configurable**: Sensible CLI flags + `.env` overrides; idempotent chunk IDs; optional clean re-ingest.

---

## Architecture (at a glance)

```
[Ingest PDFs]
  ├─ Text: PyMuPDF → recursive chunking → OpenAI embeddings → Chroma
  └─ Images: extract figures → caption & tags (vision LLM) → embed → Chroma

[LangGraph RAG]
  guard ─→ planner ─→ (python?) ─→ (retrieve?) ─→ generate ─→ memory

[Structured Extraction]
  AcroForm → label/regex heuristics → (optional LLM) → validated PersonRecord
```

**Citations**: Answers cite retrieved snippets as `[1]`, `[2]`, … matching the numbered context blocks.
**Images**: When image docs are retrieved, the generator succinctly **describes** figures and cites normally.

---

## Repository layout

```
src/
  guard/          # Heuristic prompt-injection filter + README
  ingest/         # PDF text+image ingestion to Chroma + README
  memory/         # Running summary with token budget + README
  rag/            # LangGraph: graph, planner, generation + README
  structured/     # Multi-form person extractor (OCR/LLM optional) + README
  python_tool.py  # Safe Python single-expression evaluator (allow-listed)

storage/
  chroma/         # Vector store (persisted)
  images/         # Extracted figures (PNG)
    thumbs/       # Thumbnails (320px)
data/
  raw/            # Put your PDFs here
```

---

## Quickstart

### 1) Install

* **Python**: 3.10–3.12 recommended
* **System tools** (optional, for OCR): [Tesseract](https://tesseract-ocr.github.io/) (`tesseract --version`)

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
cp .env.example .env   # then edit
```

### 2) Configure `.env` (minimum)

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL_CHAT=gpt-4o-mini
OPENAI_MODEL_EMBED=text-embedding-3-large
VECTORDB_DIR=storage/chroma
COLLECTION=pdf_chunks
DOCS_DIR=data/raw
```

Useful extras:

```
TOP_K=5
GUARD_THRESHOLD=2
MEMORY_TOKEN_BUDGET=4000
MEMORY_LAST_TURNS=4
OPENAI_MODEL_PLANNER=gpt-4o-mini
# Image ingest
OPENAI_MODEL_VISION=gpt-4o-mini
IMG_MIN_W=150
IMG_MIN_H=150
IMG_MAX_PER_PDF=24
IMAGES_DIR=storage/images
IMAGES_THUMBS_DIR=storage/images/thumbs
```

### 3) Ingest PDFs

```bash
# Text-only
python -m src.ingest.run

# With images (captions + tags)
python -m src.ingest.run --with-images
```

### 4) Run the chatbot

```bash
# One-shot:
python -m src.rag.graph --query "What does the document say about Kedro layers?"

# Interactive:
python -m src.rag.graph
```

### 5) (Optional) Structured extraction from forms

```bash
# Heuristics only (no LLM):
python -m src.structured.run_multi --pdf-dir data/raw/forms --no-llm

# With LLM adjudication (needs OPENAI_API_KEY):
python -m src.structured.run_multi --pdf-dir data/raw/forms
```

---

## Configuration reference (env/CLI)

**Core**

* `OPENAI_API_KEY` (required)
* `OPENAI_MODEL_CHAT`, `OPENAI_MODEL_EMBED`, `OPENAI_MODEL_PLANNER` (optional)
* `VECTORDB_DIR`, `COLLECTION`, `DOCS_DIR`, `TOP_K`

**Ingest (text)**

* `CHUNK_SIZE`, `CHUNK_OVERLAP`

**Ingest (images)**

* `OPENAI_MODEL_VISION`, `IMG_MIN_W`, `IMG_MIN_H`, `IMG_MAX_PER_PDF`
* `IMAGES_DIR`, `IMAGES_THUMBS_DIR`
* (Optional throttling knobs if wired in your branch): `IMG_SLEEP_MS`, `IMG_RETRIES`, `IMG_BACKOFF_BASE`

**Guard & Memory**

* `GUARD_THRESHOLD`
* `MEMORY_TOKEN_BUDGET`, `MEMORY_LAST_TURNS`

**Structured**

* OCR flags available via CLI (`--ocr`, `--ocr-lang`, `--ocr-dpi`, …)

Each module folder has a short README with specifics and CLI examples.

---

## Safety & governance

* **Prompt-injection guard**: Blocks common override/exfil patterns early; tune `GUARD_THRESHOLD` to adjust sensitivity.
* **Safe Python tool**: Single-expression, allow-listed evaluator (no imports, no attributes, no I/O). Treat as a **calculator**, not a REPL.
* **PII caution**: The structured extractor may output personal data. If you persist results, apply encryption, access controls, and retention limits.

---

## Troubleshooting

* **“No PDFs found”** → Place docs under `data/raw` or pass `--docs-dir`.
* **Empty answers** → Verify ingestion completed; `VECTORDB_DIR`/`COLLECTION` match; raise `TOP_K`.
* **Rate limits (429)** → Reduce `IMG_MAX_PER_PDF`, add call spacing (image path), or retry later.
* **OCR quality** → Increase `--ocr-dpi`, set `--ocr-lang "eng+spa"`, consider deskew/denoise.
* **Planner JSON hiccups** → Falls back to retrieval by design; ensure `OPENAI_MODEL_PLANNER` is set to a chat model.

---

## Roadmap

* 🔧 Optional MMR / hybrid ranking for retriever
* 🧠 Entity memory (people, decisions) atop running summary
* 🖼️ Lightweight image OCR for charts/diagrams text
* 🔍 Per-schema dispatcher (multiple structured outputs)
* 🧪 Evaluation harness for retrieval/answer quality

---

## Credits

Built with ❤️ using **LangGraph**, **LangChain**, **Chroma**, **PyMuPDF**, **Pydantic**, and **Tesseract** (optional).

> If you rename or relocate `src/python_tool.py` (e.g., to `src/tools/python_tool.py`), update the import in `src/rag/graph.py`.

---

That’s it—clean, skimmable, and complete for a repo homepage.
