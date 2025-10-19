# src/ui/app.py
"""
Gradio UI for RAG over Chroma + OpenAI

Run:
  python -m src.ui.app

Env (.env):
  OPENAI_API_KEY=sk-...
  OPENAI_MODEL_CHAT=gpt-4o-mini
  OPENAI_MODEL_EMBED=text-embedding-3-large
  VECTORDB_DIR=storage/chroma
  COLLECTION=pdf_chunks
  TOP_K=5

Optional memory controls:
  MEMORY_TOKEN_BUDGET=4000
  MEMORY_LAST_TURNS=4
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple, Dict, Any

import gradio as gr
from dotenv import load_dotenv
from langchain_core.documents import Document
from pathlib import Path

# Import your compiled LangGraph RAG app builder
from src.rag.graph import RagConfig, build_graph  # noqa: E402

load_dotenv()

# --- Build/boot the RAG pipeline once ---
CFG = RagConfig.from_env_and_args()  # uses env; CLI args ignored when launched via Gradio
APP = build_graph(CFG)               # compiled LangGraph

# --- Small-talk detector (skip retrieval + citations for "thanks", etc.) ---
SMALLTALK_RE = re.compile(
    r"^\s*(thanks|thank you|gracias|ok(?:ay)?|cool|nice|awesome|perfect|great|cheers|got it|understood|noted|👍|👌|🙏)\s*[.!?]*\s*$",
    re.IGNORECASE,
)

def is_smalltalk(msg: str) -> bool:
    msg = (msg or "").strip()
    if not msg:
        return False
    if "?" in msg:   # treat questions as normal queries
        return False
    return bool(SMALLTALK_RE.match(msg))


def _format_sources_md(docs: List[Document]) -> str:
    # Hide the section entirely when there are no docs
    if not docs:
        return ""
    lines = ["### Sources"]
    for i, d in enumerate(docs, start=1):
        file_name = d.metadata.get("file_name") or os.path.basename(d.metadata.get("source", "unknown"))
        page = d.metadata.get("page", "?")
        snippet = (d.page_content or "").strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300].rstrip() + "…"
        lines.append(f"**[{i}] {file_name} · p.{page}**\n> {snippet}")
    return "\n\n".join(lines)

def _collect_figures(docs: List[Document], max_figs: int = 6):
    items = []
    for d in docs or []:
        meta = d.metadata or {}
        if meta.get("modality") != "image":
            continue
        path = meta.get("thumb_path") or meta.get("image_path")
        if not path:
            continue
        path = Path(path).as_posix()  # normalize for Gradio on Windows
        caption = (d.page_content or "").split("\n", 1)[0]
        if len(caption) > 140:
            caption = caption[:140] + "…"
        items.append([path, caption])
        if len(items) >= max_figs:
            break
    return items

def _answer_with_details(question: str, mem: Dict[str, Any]):
    """
    Returns (answer_md, sources_md, blocked, guard_report, new_mem, mem_event)
    - Sends prior memory (history + summary) into the graph
    - Receives updated memory back after generation
    """
    out = APP.invoke({
        "question": question,
        "history": mem.get("history", []),
        "summary": mem.get("summary", ""),
    })
    answer = (out.get("answer") or "").strip()
    docs = out.get("docs", [])
    blocked = bool(out.get("blocked", False))
    guard = out.get("guard_report") or {}
    mem_event = out.get("memory_event") or {}  # may be {} if graph not patched yet

    sources_md = _format_sources_md(docs)

    # Pull back updated memory for next turn
    new_mem = {
        "history": out.get("history", mem.get("history", [])),
        "summary": out.get("summary", mem.get("summary", "")),
    }
    # at the end of _answer_with_details(...)
    return answer, sources_md, blocked, guard, new_mem, mem_event, docs



def respond(user_msg: str, chat_history: List[Tuple[str, str]], mem: Dict[str, Any], show_figs: bool):
    """
    MUST return 3 outputs to match the Gradio wiring:
      - updated chat history
      - cleared input (Textbox)
      - updated mem_state
    """
    if not user_msg or not user_msg.strip():
        return chat_history, gr.update(value=""), mem

    user_msg = user_msg.strip()

    # Small-talk path → no retrieval, no sources
    if is_smalltalk(user_msg):
        reply = "You're welcome! 😊"
        chat_history = chat_history + [(user_msg, reply)]
        return chat_history, gr.update(value=""), mem

    try:
        answer, sources, blocked, guard, new_mem, mem_event, docs = _answer_with_details(user_msg, mem)

        if blocked:
            # Keep it concise; show top patterns that triggered the guard.
            matches = guard.get("matched") or []
            show = ", ".join(matches[:3]) if matches else "—"
            details = f"\n\n_Details: score={guard.get('score', 0)}; matches: {show}_"
            answer = f"Request blocked by guardrail.\n\n{answer}{details}"

        # Memory banner when summarizer fires
        banner = ""
        if mem_event.get("did_summarize"):
            dropped = mem_event.get("dropped_messages", 0)
            kept = mem_event.get("kept_messages", 0)
            banner = f"\n\n> 🧠 _Conversation condensed (dropped {dropped}, keeping {kept})._"

        # Only add the Sources block if there are docs
        section = f"\n\n---\n{sources}" if sources else ""
        full_answer = f"{answer}{banner}{section}"

        chat_history = chat_history + [(user_msg, full_answer)]
        gallery_items = _collect_figures(docs) if show_figs else []
        return chat_history, gr.update(value=""), new_mem, gallery_items
    
    except Exception as e:
        err = f"⚠️ Error: {e}"
        chat_history = chat_history + [(user_msg, err)]
        return chat_history, gr.update(value=""), mem


with gr.Blocks(fill_height=True) as demo:
    gr.Markdown(
        "# 📚 RAG Chat (Chroma + OpenAI)\n"
        "Ask questions about your indexed PDFs. Answers include numbered citations."
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                height=500,
                show_copy_button=True,
                bubble_full_width=False,
            )
            user_in = gr.Textbox(
                label="Your question",
                placeholder="Ask something about your documents…",
                autofocus=True,
            )
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")

        with gr.Column(scale=1):
            gr.Markdown("### Settings")
            gr.Markdown(
                f"- **Model:** `{CFG.chat_model}`  \n"
                f"- **Embeddings:** `{CFG.embed_model}`  \n"
                f"- **Top-K:** `{CFG.top_k}`  \n"
                f"- **Vector DB:** `{CFG.vectordb_dir}`  \n"
                f"- **Collection:** `{CFG.collection}`"
            )
            gr.Markdown(
                "> To change these, edit your `.env` and restart.\n"
                ">\n"
                "> Add more PDFs to `data/raw/` and re-run ingestion."
            )
            show_figs = gr.Checkbox(value=True, label="Show recovered figures (if any)")
            gallery = gr.Gallery(label="Figures", columns=3, rows=2, height=300, preview=True)

    # Memory state (history + summary) persisted across turns
    mem_state = gr.State({"history": [], "summary": ""})

    send_btn.click(
        fn=respond,
        inputs=[user_in, chatbot, mem_state, show_figs],
        outputs=[chatbot, user_in, mem_state, gallery],
    )
    user_in.submit(
        fn=respond,
        inputs=[user_in, chatbot, mem_state, show_figs],
        outputs=[chatbot, user_in, mem_state, gallery],
    )
    clear_btn.click(
        lambda: ([], "", {"history": [], "summary": ""}, []),
        None, [chatbot, user_in, mem_state, gallery]
    )

def main():
    # Helpful when launched via `python -m src.ui.app`
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
