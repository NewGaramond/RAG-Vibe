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
"""

from __future__ import annotations

import os
from typing import List, Tuple

import gradio as gr
from dotenv import load_dotenv
from langchain_core.documents import Document

# Import your compiled LangGraph RAG app builder
from src.rag.graph import RagConfig, build_graph  # noqa: E402


load_dotenv()


# --- Build/boot the RAG pipeline once ---
CFG = RagConfig.from_env_and_args()  # uses env; CLI args ignored when launched via Gradio
APP = build_graph(CFG)               # compiled LangGraph


def _format_sources_md(docs: List[Document]) -> str:
    if not docs:
        return "_No sources returned._"
    lines = ["### Sources"]
    for i, d in enumerate(docs, start=1):
        file_name = d.metadata.get("file_name") or os.path.basename(d.metadata.get("source", "unknown"))
        page = d.metadata.get("page", "?")
        snippet = (d.page_content or "").strip().replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300].rstrip() + "…"
        lines.append(f"**[{i}] {file_name} · p.{page}**\n> {snippet}")
    return "\n\n".join(lines)


def _answer_with_details(question: str) -> Tuple[str, str]:
    """
    Returns (answer_md, sources_md, blocked, guard_report)
    """
    out = APP.invoke({"question": question})
    answer = (out.get("answer") or "").strip()
    docs = out.get("docs", [])
    blocked = bool(out.get("blocked", False))
    guard = out.get("guard_report") or {}

    sources_md = _format_sources_md(docs)
    return answer, sources_md, blocked, guard

def respond(user_msg: str, chat_history: List[Tuple[str, str]]):
    if not user_msg or not user_msg.strip():
        return chat_history, gr.update(value="")
    try:
        answer, sources, blocked, guard = _answer_with_details(user_msg.strip())

        if blocked:
            # Keep it concise; show top patterns that triggered the guard.
            matches = guard.get("matched") or []
            show = ", ".join(matches[:3]) if matches else "—"
            details = f"\n\n_Details: score={guard.get('score', 0)}; matches: {show}_"
            answer = f"Request blocked by guardrail.\n\n{answer}{details}"

        full_answer = f"{answer}\n\n---\n{sources}"
        chat_history = chat_history + [(user_msg.strip(), full_answer)]
        return chat_history, gr.update(value="")
    except Exception as e:
        err = f"⚠️ Error: {e}"
        chat_history = chat_history + [(user_msg.strip(), err)]
        return chat_history, gr.update(value="")



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

    # Wire events
    send_btn.click(
        fn=respond,
        inputs=[user_in, chatbot],
        outputs=[chatbot, user_in],
    )
    user_in.submit(
        fn=respond,
        inputs=[user_in, chatbot],
        outputs=[chatbot, user_in],
    )
    clear_btn.click(lambda: ([], ""), None, [chatbot, user_in])


def main():
    # Helpful when launched via `python -m src.ui.app`
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
