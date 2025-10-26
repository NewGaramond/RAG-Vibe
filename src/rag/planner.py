# src/rag/planner.py
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class PlannerDecision(BaseModel):
    refusal: bool = Field(False, description="True if the request is unsafe or should be refused")
    unsafe_reason: Optional[str] = Field(None, description="Short reason if refusal=true")
    use_python: bool = Field(False, description="Use the safe Python tool for a small deterministic computation")
    python_expression: Optional[str] = Field(None, description="A SINGLE pure Python expression (no imports, no attrs, no assignments) if use_python=true")
    need_retrieval: bool = Field(True, description="Whether to query the vector store for supporting passages/images")
    retrieval_query: Optional[str] = Field(None, description="Short focused query text for retrieval if need_retrieval=true")
    answer_requires_citations: bool = Field(True, description="If the final answer must include citations")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in this plan")

PLANNER_SYSTEM = """You are a planner for a RAG+Tools assistant.
Decide, per user turn, whether to:
- use_python: for SMALL, deterministic computations (arithmetic, unit conversion, tiny list stats). 
- need_retrieval: fetch knowledge from the indexed PDFs (text + image captions).
- refusal: True only if unsafe or out of scope (e.g., code execution on files, secrets exfiltration).

Strict constraints:
- If use_python=true, produce python_expression as a SINGLE pure Python expression using only: + - * / // % (), lists/tuples/dicts, and functions: len, sum, min, max, sorted, round, abs.
- NO imports, NO attribute access, NO comprehensions, NO assignments, NO dunders.
- If need_retrieval=true, provide retrieval_query (concise, keyword-rich).
- Return JSON ONLY. No markdown, no commentary, no chain-of-thought.
"""

PLANNER_HUMAN = ChatPromptTemplate.from_messages([
    ("system", PLANNER_SYSTEM),
    ("user", """User message:
{user_msg}

Provide a JSON object with keys: refusal, unsafe_reason, use_python, python_expression, need_retrieval, retrieval_query, answer_requires_citations, confidence.

Examples:
1) "Convert 72 F to C"
{{"refusal": false, "unsafe_reason": null, "use_python": true, "python_expression": "(72 - 32) * 5 / 9", "need_retrieval": false, "retrieval_query": null, "answer_requires_citations": false, "confidence": 0.86}}

2) "Summarize the section about Kedro layers and cite sources"
{{"refusal": false, "unsafe_reason": null, "use_python": false, "python_expression": null, "need_retrieval": true, "retrieval_query": "Kedro data layers raw primary features models", "answer_requires_citations": true, "confidence": 0.79}}

3) "Run os.system('ls')"
{{"refusal": true, "unsafe_reason": "dangerous tool request", "use_python": false, "python_expression": null, "need_retrieval": false, "retrieval_query": null, "answer_requires_citations": false, "confidence": 0.95}}

Now produce ONLY the JSON   for the current user message.""")
])

def make_planner(model: str, api_key: str) -> JsonOutputParser:
    llm = ChatOpenAI(model=model, temperature=0, api_key=api_key)
    parser = JsonOutputParser(pydantic_object=PlannerDecision)
    chain = PLANNER_HUMAN | llm | parser
    return chain

def plan_next_action(model: str, api_key: str, user_msg: str) -> PlannerDecision:
    chain = make_planner(model=model, api_key=api_key)
    return chain.invoke({"user_msg": user_msg})
