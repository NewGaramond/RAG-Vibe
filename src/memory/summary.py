# src/memory/summary.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import tiktoken
from langchain_openai import ChatOpenAI

# We use cl100k_base which fits OpenAI 4.x/4o families well.
ENC = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(ENC.encode(text))

@dataclass
class MemoryConfig:
    chat_model: str
    api_key: str
    token_budget: int = 4000
    last_turns: int = 4   # keep this many (user+assistant) messages verbatim

def _render_history(history: List[Dict[str, str]]) -> str:
    # history is list of {"role": "user"|"assistant", "content": "..."}
    lines = []
    for m in history:
        role = m.get("role", "user")
        lines.append(f"{role.upper()}: {m.get('content','').strip()}")
    return "\n".join(lines)

SUMMARIZE_SYSTEM = (
    "You are a conversation summarizer. Create a concise, neutral, factual running summary "
    "of the chat so far. Preserve important facts, decisions, tasks, names, and references. "
    "Do not invent. Keep it brief but complete enough to recover context later."
)

def maybe_update_summary(
    *,
    summary: str,
    history: List[Dict[str, str]],
    new_user: str,
    new_answer: str,
    cfg: MemoryConfig,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Return (new_summary, new_history). If budget exceeded, compress older history into summary
    and keep only last N turns verbatim.
    """
    # Build text we would carry forward
    candidate_history = history + [
        {"role": "user", "content": new_user},
        {"role": "assistant", "content": new_answer},
    ]
    # Token estimate for (summary + last N turns + this new pair)
    rendered = f"SUMMARY SO FAR:\n{summary}\n\nHISTORY:\n{_render_history(candidate_history)}"
    total = count_tokens(rendered)

    if total <= cfg.token_budget:
        # within budget: just append, no summarization
        return (summary, candidate_history)

    # Exceeded: summarize (summary + history) into a new shorter summary, then keep last N turns
    llm = ChatOpenAI(model=cfg.chat_model, api_key=cfg.api_key, temperature=0)
    # Keep a little extra context for the summarizer
    last_n = max(0, cfg.last_turns * 2)  # user+assistant count
    keep_tail = candidate_history[-last_n:] if last_n else []
    head = candidate_history[:-last_n] if last_n else candidate_history

    prompt_user = (
        f"CURRENT SUMMARY (may be empty):\n{summary}\n\n"
        f"CONVERSATION CHUNKS TO FOLD INTO SUMMARY:\n{_render_history(head)}\n\n"
        f"Now produce an UPDATED RUNNING SUMMARY."
    )

    msgs = [
        {"role": "system", "content": SUMMARIZE_SYSTEM},
        {"role": "user", "content": prompt_user},
    ]
    new_sum = llm.invoke(msgs).content.strip()

    # Return the compressed state
    return (new_sum, keep_tail)
