# src/rag/python_guard.py
from __future__ import annotations
from typing import Optional

BLOCK_SUBSTRINGS = ["__", "import", "open(", "exec(", "eval(", "os.", "sys.", "subprocess", "class", "lambda", "for ", "while "]
ALLOWED_FUNC_PREFIXES = ("len(", "sum(", "min(", "max(", "sorted(", "round(", "abs(")

def python_expression_is_eligible(expr: Optional[str]) -> bool:
    if not expr: 
        return False
    s = expr.strip().lower()
    if any(b in s for b in BLOCK_SUBSTRINGS):
        return False
    # Must look like a simple expression (heuristic MVP)
    has_paren_or_ops = any(ch in s for ch in "()*/+-%,[]{}")
    has_allowed_fn = s.startswith(ALLOWED_FUNC_PREFIXES) or any(f in s for f in ALLOWED_FUNC_PREFIXES)
    has_digit = any(ch.isdigit() for ch in s)
    # Allow pure list ops (e.g., len([...])) or arithmetic
    return has_paren_or_ops or has_allowed_fn or has_digit
