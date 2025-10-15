# src/guard/filters.py
from __future__ import annotations
import os
import re
from dataclasses import dataclass

# Common prompt-injection patterns (expand as needed)
SUSPECT_PATTERNS = [
    r"\bignore (?:all|previous|above) (?:instructions|directions)\b",
    r"\bdisregard\b",
    r"\boverride\b",
    r"\bact as\b",
    r"\bpretend to be\b",
    r"\bsystem\s+prompt\b",
    r"\bdeveloper\s+message\b",
    r"\bshow (?:the )?(?:instructions|system|prompt)\b",
    r"\bdisclose\b.*\b(prompt|instructions)\b",
    r"\bprint\s+env\b",
    r"\bAPI[_-]?KEY\b",
    r"\b(secret|token|key)s?\b",
    r"\bexfiltrat(?:e|ion)\b",
    r"\bread (?:file|filesystem|fs|disk)\b",
    r"\b(?:cat|type)\s+[/\\\w\.\-\*]+\b",
    r"\bls\s+[/\\\w\.\-\*]+\b",
    r"\bdisable\b.*\b(guard|safety|filter|moderation)\b",
    r"\bjailbreak\b",
    r"\bprompt(?:-| )?injection\b",
]

@dataclass
class GuardReport:
    is_suspicious: bool
    score: int
    matched: list[str]

def heuristic_injection_score(text: str) -> GuardReport:
    matched = []
    score = 0
    for pat in SUSPECT_PATTERNS:
        if re.search(pat, text, flags=re.I):
            score += 1
            matched.append(pat)
    # Long meta-instruction blocks sometimes correlate with attacks (very light weight)
    if len(text) > 4000:
        score += 1
    return GuardReport(is_suspicious=score >= get_threshold(), score=score, matched=matched)

def get_threshold() -> int:
    try:
        return max(1, int(os.getenv("GUARD_THRESHOLD", "1")))
    except Exception:
        return 2

def check_user_prompt_injection(prompt: str) -> GuardReport:
    return heuristic_injection_score(prompt)
