# src/rag/graph.py
"""
LangGraph RAG: Chroma retriever -> OpenAI answer with citations

Usage:
  python -m src.rag.graph --query "What does the document say about X?"
  python -m src.rag.graph --top-k 6
  # Or interactive:
  python -m src.rag.graph

Env (.env):
  OPENAI_API_KEY=sk-...
  OPENAI_MODEL_CHAT=gpt-4o-mini
  OPENAI_MODEL_EMBED=text-embedding-3-large
  VECTORDB_DIR=storage/chroma
  COLLECTION=pdf_chunks
  TOP_K=5
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, TypedDict

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

# LangGraph
from langgraph.graph import StateGraph, END


# -------------------------
# Config & State
# -------------------------
@dataclass
class RagConfig:
    vectordb_dir: str
    collection: str
    openai_api_key: str
    chat_model: str
    embed_model: str
    top_k: int = 5
    temperature: float = 0.2
    max_tokens: int = 600  # keep answers concise

    @staticmethod
    def from_env_and_args() -> "RagConfig":
        load_dotenv()
        parser = argparse.ArgumentParser(description="RAG with Chroma + OpenAI via LangGraph.")
        parser.add_argument("--query", type=str, default=None)
        parser.add_argument("--vectordb-dir", type=str, default=os.getenv("VECTORDB_DIR", "storage/chroma"))
        parser.add_argument("--collection", type=str, default=os.getenv("COLLECTION", "pdf_chunks"))
        parser.add_argument("--chat-model", type=str, default=os.getenv("OPENAI_MODEL_CHAT", "gpt-4o-mini"))
        parser.add_argument("--embed-model", type=str, default=os.getenv("OPENAI_MODEL_EMBED", "text-embedding-3-large"))
        parser.add_argument("--top-k", type=int, default=int(os.getenv("TOP_K", "5")))
        parser.add_argument("--temperature", type=float, default=0.2)
        parser.add_argument("--max-tokens", type=int, default=600)
        args = parser.parse_args()

        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return RagConfig(
            vectordb_dir=args.vectordb_dir,
            collection=args.collection,
            openai_api_key=key,
            chat_model=args.chat_model,
            embed_model=args.embed_model,
            top_k=args.top_k,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )


class RAGState(TypedDict):
    question: str
    docs: List[Document]
    answer: str


# -------------------------
# Vector store / retriever
# -------------------------
def get_retriever(cfg: RagConfig):
    embeddings = OpenAIEmbeddings(model=cfg.embed_model, api_key=cfg.openai_api_key)
    vs = Chroma(
        collection_name=cfg.collection,
        persist_directory=cfg.vectordb_dir,
        embedding_function=embeddings,
    )
    return vs.as_retriever(search_kwargs={"k": cfg.top_k})


# -------------------------
# Prompt
# -------------------------
SYSTEM = (
    "You are a precise RAG assistant. Use ONLY the provided sources to answer. "
    "If the answer is not in the sources, say you don't know. "
    "Cite using bracketed numbers [1], [2], etc. corresponding to the SOURCES list. "
    "Be concise and factual."
)

USER_TEMPLATE = """\
QUESTION:
{question}

CONTEXT (numbered excerpts):
{context_blocks}

SOURCES:
{sources_list}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("user", USER_TEMPLATE),
    ]
)


def build_context_blocks(docs: List[Document]) -> tuple[str, str]:
    """
    Number docs 1..N and build a readable context and sources list.
    """
    blocks = []
    sources = []
    for i, d in enumerate(docs, start=1):
        file_name = d.metadata.get("file_name") or d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        snippet = d.page_content.strip()
        blocks.append(f"[{i}] ({file_name} p.{page})\n{snippet}")
        sources.append(f"[{i}] {file_name} p.{page}")
    return "\n\n".join(blocks), "\n".join(sources)


# -------------------------
# Nodes
# -------------------------
def make_retrieve_node(retriever):
    def retrieve(state: RAGState) -> RAGState:
        docs = retriever.get_relevant_documents(state["question"])
        return {"docs": docs}
    return retrieve


def make_generate_node(cfg: RagConfig):
    llm = ChatOpenAI(
        model=cfg.chat_model,
        api_key=cfg.openai_api_key,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )

    def generate(state: RAGState) -> RAGState:
        docs = state.get("docs", [])
        if not docs:
            return {"answer": "I couldn't find any relevant passages in the indexed documents to answer that."}

        context_blocks, sources_list = build_context_blocks(docs)
        messages = prompt.format_messages(
            question=state["question"],
            context_blocks=context_blocks,
            sources_list=sources_list,
        )
        result = llm.invoke(messages)
        return {"answer": result.content}
    return generate


# -------------------------
# Graph
# -------------------------
def build_graph(cfg: RagConfig):
    retriever = get_retriever(cfg)

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", make_retrieve_node(retriever))
    graph.add_node("generate", make_generate_node(cfg))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# -------------------------
# CLI
# -------------------------
def _interactive_ask(loop):
    print("RAG ready. Type your question (or 'exit'): ")
    for line in sys.stdin:
        q = line.strip()
        if not q:
            print("Ask me something (or 'exit'):")
            continue
        if q.lower() in {"exit", "quit"}:
            break
        out = loop.invoke({"question": q})
        print("\n" + out["answer"].strip() + "\n")
        print("— Ask another (or 'exit') —")


def main():
    cfg = RagConfig.from_env_and_args()
    app = build_graph(cfg)

    # One-shot or interactive
    query = None
    for i, a in enumerate(sys.argv):
        if a == "--query" and i + 1 < len(sys.argv):
            query = sys.argv[i + 1]
            break

    if query:
        out = app.invoke({"question": query})
        print(out["answer"])
    else:
        _interactive_ask(app)


if __name__ == "__main__":
    main()
