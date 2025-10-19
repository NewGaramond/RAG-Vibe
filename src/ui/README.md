# UI (Gradio Chat App)

A simple, production-friendly **Gradio** UI for your guarded RAG system over **Chroma + OpenAI**. It compiles the LangGraph once at startup and provides an interactive chat with citations and (optionally) recovered figures.

---

## Run

```bash
python -m src.ui.app
```

The app launches at `http://0.0.0.0:7860` (see `main()`).

---

## Environment (.env)

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL_CHAT=gpt-4o-mini
OPENAI_MODEL_EMBED=text-embedding-3-large
VECTORDB_DIR=storage/chroma
COLLECTION=pdf_chunks
TOP_K=5

# Optional memory controls
MEMORY_TOKEN_BUDGET=4000
MEMORY_LAST_TURNS=4
```

> When launched via Gradio, CLI flags are ignored; **env vars** control the config. Update `.env` and restart the app.

---

## What it does

* **Boots the RAG graph** once:

  * `CFG = RagConfig.from_env_and_args()`
  * `APP = build_graph(CFG)`
* **Chat flow**:

  1. User sends a message.
  2. Optional small-talk short-circuit (no retrieval/citations).
  3. Otherwise, message + memory are sent to the compiled graph.
  4. UI renders the answer, numbered **Sources**, and an optional **Figures** gallery.
  5. Memory (summary + history) is updated across turns.

---

## UI features

* **Chatbot** with copy button, 500px height.
* **“Your question”** textbox with Enter-to-send.
* **Send / Clear** buttons.
* **Settings** panel (read-only) showing active model, embeddings, K, vector DB path, and collection.
* **Show recovered figures** toggle to display a gallery of image thumbnails (from `metadata.thumb_path` or `image_path`).
* **Memory state** persisted in `gr.State({"history": [], "summary": ""})`.

---

## Citations & Figures

* **Sources** are shown only when documents are returned. Each source includes:

  * **index**, **file name**, **page**, and a short **snippet**.
* **Figures** (if present and toggle enabled) are shown in a gallery (max 6), using the thumbnail path and the first line of the figure’s caption.

---

## Small-talk fast path

Short acknowledgments (e.g., *“thanks”, “ok”, “gracias”, 👍, 🙏* ) bypass retrieval and citations to keep the chat clean.

```python
SMALLTALK_RE = re.compile(
  r"^\s*(thanks|thank you|gracias|ok(?:ay)?|cool|nice|awesome|perfect|great|cheers|got it|understood|noted|👍|👌|🙏)\s*[.!?]*\s*$",
  re.IGNORECASE,
)
```

If matched (and the message isn’t a question), the UI replies with a lightweight “You’re welcome!”.

---

## Key functions

* **`is_smalltalk(msg: str) -> bool`**
  Detects acknowledgments to skip retrieval/citations.

* **`_format_sources_md(docs: List[Document]) -> str`**
  Builds the “### Sources” section (hidden when no docs).

* **`_collect_figures(docs, max_figs=6)`**
  Extracts image thumbnails + captions for the gallery.

* **`_answer_with_details(question, mem)`**
  Calls the compiled graph with current memory and returns:
  `(answer_md, sources_md, blocked, guard_report, new_mem, mem_event, docs)`

* **`respond(user_msg, chat_history, mem, show_figs)`**
  Orchestrates a single turn and returns **four** UI outputs:

  1. updated `chat_history`
  2. cleared input (via `gr.update(value="")`)
  3. updated `mem_state`
  4. optional `gallery_items` (if `show_figs`)

* **`main()`**
  Launches Gradio (`demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)`).

---

## Memory integration

* Input to graph: `{"history": mem["history"], "summary": mem["summary"]}`
* Output from graph: updated `history`, `summary`, and optional `memory_event`.
* If summarization is triggered, the UI adds a banner:

  > 🧠 *Conversation condensed (dropped N, keeping M).*

---

## Extending the UI

* **Uploads**: add a file uploader to push new PDFs into `data/raw/` and show an “Ingest now” button that calls your ingest CLI.
* **Session IDs**: store per-user session memory in a keyed dict (e.g., `gr.State({"sessions": {sid: {...}}})`).
* **Streaming**: switch the answer call to a streaming variant and update the chatbot incrementally.
* **Advanced filters**: add checkboxes to restrict retrieval to `{"modality": "image"}` or to specific `file_name`s.

---

## Troubleshooting

* **Blank answers / no sources**: ensure ingestion ran and `VECTORDB_DIR` / `COLLECTION` match the graph config.
* **Guard blocks benign inputs**: lower `GUARD_THRESHOLD` in env, or refine patterns in `src/guard/filters.py`.
* **Figures not shown**: confirm you ingested with `--with-images`, and that `thumb_path` or `image_path` exists.
* **Config not changing**: remember the UI reads **env vars at startup**; restart after editing `.env`.

---

## Folder expectations

```
data/
  raw/                 # PDFs here
storage/
  chroma/             # Chroma index
  images/             # Saved figures
    thumbs/           # Thumbnails
```

---

## Example session

1. Start the UI: `python -m src.ui.app`
2. Ask: “What are the Kedro data layers?” → Answer + `[1]`, `[2]` citations.
3. Ask: “thanks” → Short “You’re welcome!” with no sources.
4. Toggle **Show recovered figures** to see captioned thumbnails when relevant.
