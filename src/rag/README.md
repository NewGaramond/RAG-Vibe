# RAG Graph (LangGraph + Guard + Planner + Safe Python)

Production-oriented RAG pipeline built with **LangGraph**, using:

* a **guard** (regex heuristics) to block prompt-injection,
* a **planner** LLM to decide: *use python? retrieve? refuse?*,
* **retrieval** from **Chroma** (OpenAI embeddings),
* a **generator** LLM with image-aware context formatting,
* lightweight **memory** (running summary with token budget),
* a **safe Python** tool for tiny deterministic calculations.

```
guard → planner → (python?) → (retrieve?) → generate → mem_update
```

---

## Files

* `graph.py` — builds/compiles the LangGraph, CLI entrypoint.
* `planner.py` — typed planner (Pydantic) + JSON parser chain.
* `python_guard.py` — **safe** Python one-liner evaluator (name may differ in your repo; see note below).

> **Name note:** `graph.py` imports `run_safe_python` from `src/tools/python_tool.py`.
> If your project uses `src/rag/python_guard.py` instead, export the same function:
>
> ```python
> def run_safe_python(expr: str) -> dict:  # {"ok": bool, "result"|"error": str}
>     ...
> ```
>
> and update the import in `graph.py` accordingly.

---

## Quick start

```bash
# One-shot (prints answer)
python -m src.rag.graph --query "What does the document say about Kedro layers?"

# Interactive REPL
python -m src.rag.graph

# Common overrides
python -m src.rag.graph \
  --vectordb-dir storage/chroma \
  --collection pdf_chunks \
  --top-k 5 \
  --chat-model gpt-4o-mini \
  --embed-model text-embedding-3-large \
  --planner-model gpt-4o-mini
```

### Required environment (.env)

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL_CHAT=gpt-4o-mini
OPENAI_MODEL_EMBED=text-embedding-3-large
# Optional:
OPENAI_MODEL_PLANNER=gpt-4o-mini
VECTORDB_DIR=storage/chroma
COLLECTION=pdf_chunks
TOP_K=5
GUARD_THRESHOLD=2
MEMORY_TOKEN_BUDGET=4000
MEMORY_LAST_TURNS=4
```

---

## CLI flags (`graph.py`)

| Flag                  |  Type | Default (env)                                 | Purpose                                  |
| --------------------- | ----: | --------------------------------------------- | ---------------------------------------- |
| `--query`             |   str | `None`                                        | One-shot question; otherwise REPL mode.  |
| `--vectordb-dir`      |   str | `VECTORDB_DIR`                                | Chroma persist dir.                      |
| `--collection`        |   str | `COLLECTION`                                  | Chroma collection name.                  |
| `--chat-model`        |   str | `OPENAI_MODEL_CHAT`                           | LLM for answer generation.               |
| `--embed-model`       |   str | `OPENAI_MODEL_EMBED`                          | Embedding model for retriever.           |
| `--top-k`             |   int | `TOP_K`                                       | Retriever K.                             |
| `--temperature`       | float | `0.2`                                         | Answer creativity.                       |
| `--max-tokens`        |   int | `600`                                         | Answer length cap.                       |
| `--memory-budget`     |   int | `MEMORY_TOKEN_BUDGET`                         | Target token budget for summary+history. |
| `--memory-last-turns` |   int | `MEMORY_LAST_TURNS`                           | Tail of recent turns to keep verbatim.   |
| `--planner-model`     |   str | `OPENAI_MODEL_PLANNER` or `OPENAI_MODEL_CHAT` | LLM used by planner.                     |

> **Fix for earlier error** (`Namespace` missing `planner_model`): this README assumes `graph.py` includes `parser.add_argument("--planner-model", ...)` as in your latest version.

---

## State model

```python
class RAGState(TypedDict, total=False):
    question: str
    docs: List[Document]
    answer: str
    blocked: bool
    guard_report: Dict[str, Any]
    history: List[Dict[str, str]]   # [{"role":"user","content":...}, ...]
    summary: str                    # running conversation summary
    memory_event: Dict[str, Any]
    plan: Dict[str, Any]            # planner decision JSON
    python_tool_result: Optional[str]
```

---

## Node flow (LangGraph)

```text
[guard] --blocked--> END
    |
    v
[planner] --use_python--> [python] --need_retrieval--> [retrieve] --> [generate] --> [mem_update] --> END
          \--need_retrieval--> [retrieve] --/
           \------------------> [generate] --/
```

### Guard

* Uses `src/guard/filters.py` to score the prompt (regexs for jailbreak, exfiltration, etc.).
* If suspicious (score ≥ `GUARD_THRESHOLD`, default 2 if env parse fails), returns a refusal message.

### Planner

* LLM decides per turn:

  * `use_python`: small, deterministic math only.
  * `need_retrieval`: pull from vector store (text + image captions).
  * `refusal`: unsafe/out-of-scope.
  * Optional `retrieval_query` to refine the search.
* `planner.py` provides a typed `PlannerDecision` and a `JsonOutputParser` chain.

### Python (safe)

* If `use_python=true`, runs a **single** pure expression via `run_safe_python(expr)`.
* Result is injected as **context only** (not a citation source).

### Retrieve

* `Chroma.as_retriever(k=TOP_K)` over your `pdf_chunks` collection.
* Image items are included if you ingested images; they carry `metadata["modality"] == "image"`.

### Generate

* Builds a **system** + **user** prompt with:

  * conversation **summary**,
  * **recent turns**,
  * **python tool result** (if any),
  * **numbered context blocks** + **sources** list.
* If image docs are present, prepends an extra system nudge to *describe the figure succinctly* and **cite**.

### Memory

* Calls `src/memory/summary.maybe_update_summary(...)` with a token budget.
* Keeps a rolling summary + the last `N` turns verbatim (`MEMORY_LAST_TURNS`).

---

## Prompts & formatting

* **System** (generation): *“Use ONLY the provided sources… If images are present, describe them succinctly, cite normally… If unknown, say you don’t know. Cite with [1], [2] …”*
* **User template** injects:

  * **CONVERSATION SUMMARY**
  * **RECENT TURNS**
  * **PYTHON TOOL RESULT**
  * **QUESTION**
  * **CONTEXT** (numbered snippets)
  * **SOURCES** (numbered list)

### Image context blocks

When `metadata.modality == "image"`, blocks look like:

```
[3] 📷 FIGURE (report.pdf p.7) [thumb: storage/images/thumbs/report_p7_i1.png]
Caption: ...
Tags: ...
```

---

## Planner (`planner.py`)

```python
class PlannerDecision(BaseModel):
    refusal: bool = False
    unsafe_reason: Optional[str] = None
    use_python: bool = False
    python_expression: Optional[str] = None
    need_retrieval: bool = True
    retrieval_query: Optional[str] = None
    answer_requires_citations: bool = True
    confidence: float = 0.0
```

* **Constraints enforced in prompts**:

  * Single pure expression only; **no imports, no attrs, no assignments, no dunders**.
  * Allowed funcs: `len, sum, min, max, sorted, round, abs`.
* `plan_next_action(model, api_key, user_msg) -> PlannerDecision`.

**Fallbacks:** In `graph.py`, if JSON parsing fails, plan defaults to **retrieval=true**.

---

## Safe Python execution (`python_guard.py` / `python_tool.py`)

* Input: **one** expression string.
* Return: `{"ok": True, "result": "<stringified>"}` or `{"ok": False, "error": "reason"}`.
* `graph.py` adds extra eligibility checks:

  * Blocks substrings: `__`, `import`, `open(`, `exec(`, `eval(`, `os.`, `sys.`, `subprocess`, `class`, `lambda`, `" for "`, `" while "`.
  * Requires operators/delimiters or allowed function tokens, and typically some digits.

> Keep it boring on purpose: this is a calculator, not a REPL.

---

## Retrieval (Chroma)

`get_retriever(cfg)` wires:

* `OpenAIEmbeddings(model=cfg.embed_model)`
* `Chroma(collection_name=cfg.collection, persist_directory=cfg.vectordb_dir)`
* `as_retriever(search_kwargs={"k": cfg.top_k})`

**Filtering tip:** If you want only figures:

```python
retriever.search_kwargs["filter"] = {"modality": "image"}
```

---

## Citations

* The answer LLM is instructed to cite using `[1]`, `[2]`… corresponding to the **numbered context blocks**.
* The **SOURCES** list mirrors those indices: `"[1] file.pdf p.3"` etc.

---

## Examples

### Math only

> “Convert 72 F to C.”

Planner sets `use_python=true`, `need_retrieval=false`, `python_expression="(72-32)*5/9"`.
Generate returns `Resultado del cálculo: 22.222…`.

### RAG with images

> “Explain the line chart about quarterly revenue.”

Planner: `need_retrieval=true`. Retriever returns an image doc (caption+tags).
Generator briefly **describes** the figure, cites `[1]`.

---

## Testing

**Planner parsing**

```python
from src.rag.planner import plan_next_action
d = plan_next_action(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"),
                     user_msg="Convert 5 km to miles")
assert d.use_python and not d.need_retrieval
```

**Graph smoke test**

```bash
OPENAI_API_KEY=... \
python -m src.rag.graph --query "What are the Kedro data layers?" \
  --vectordb-dir storage/chroma --collection pdf_chunks
```

---

## Troubleshooting

* **`OPENAI_API_KEY` missing** → set it in `.env`.
* **No answers / “couldn’t find passages”** → ensure you ingested PDFs (`src/ingest/run.py`) and the collection/paths match.
* **Planner JSON errors** → falls back to retrieval; check model and prompt length.
* **Python tool declined** → expression failed eligibility or tool rejected; answer proceeds without math result.
* **Images not described** → ensure your ingest used `--with-images` and that image docs have `metadata.modality="image"`.

---

## Extending

* **Router edges**: add more tools (web, code exec sandbox) with extra planner flags and nodes.
* **Ranking**: swap Chroma retriever config (MMR, metadata filters).
* **Answer styles**: add system presets (strict, tutorial, terse).
* **Memory**: persist summary/history per user/session; add entity memory.

---

## Changelog

* **v0.3** — Planner + safe-python branch + image-aware generation.
* **v0.2** — Guarded RAG graph with memory and citations.
* **v0.1** — Basic retriever → generator path.

---
