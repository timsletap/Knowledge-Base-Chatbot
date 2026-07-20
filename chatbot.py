"""Orchestration: safety gate, retrieval, confidence gate, render.

Every user-visible string is either a fixed literal below or a field
copied verbatim from knowledge_base.json. No model output is ever shown
as prose, so the bot cannot hallucinate or give advice of its own.

Run the eval set without the UI:  RETRIEVER=keyword python chatbot.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from retrievers import Retriever, get_retriever

TRUSTED_ID = "trusted-health-info"

# Stage 1: narrow patterns for personal medical advice. Keep these narrow —
# a broad pattern like "why do i" would wrongly refuse legitimate questions
# such as "Why do I get decision fatigue with food?". Re-run the full eval
# set in SPEC.md section 9 after any change here.
SAFETY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bdiagnos(e|is|ing)\b",
        r"\bshould i (take|stop|start|use)\b",
        r"\bis it safe for me\b",
        r"\bprescri(be|ption|bed)\b",
        r"\bdo i have\b",
        r"\b(treat|cure) my\b",
    )
]

# Fixed framing lines. Never model generated.
MESSAGES = {
    "match": "Here is the best match from the knowledge base:",
    "refusal": (
        "I can't help with personal medical advice — that is a question for "
        "a clinician who knows your history. Here is a trusted place to start:"
    ),
    "low_confidence": (
        "I couldn't find a close match for that in the knowledge base. "
        "Here is a trusted place to start:"
    ),
}


def _card(entry: dict) -> dict:
    """The user-facing fields of an entry, copied verbatim."""
    return {
        "title": entry["lesson"],
        "format": entry["format"],
        "link": entry["link"],
        "summary": entry["summary"],
    }


class Chatbot:
    def __init__(self, kb_path: str = "knowledge_base.json", retriever: Retriever | None = None):
        self.kb = json.loads(Path(kb_path).read_text(encoding="utf-8"))
        trusted = [e for e in self.kb if e["id"] == TRUSTED_ID]
        if not trusted:
            # Fail loudly at startup: both guardrail paths depend on this entry.
            raise ValueError(
                f"knowledge base has no entry with id {TRUSTED_ID!r}; "
                "the safety and low-confidence paths require it"
            )
        self.trusted = trusted[0]
        self.retriever = retriever if retriever is not None else get_retriever(self.kb)

    def answer(self, query: str) -> dict:
        # Stage 1: safety gate, before retrieval.
        if any(p.search(query) for p in SAFETY_PATTERNS):
            return self._fallback("refusal")

        # Stage 2: retrieval.
        matches = self.retriever.retrieve(query)

        # Stage 3: confidence gate.
        if not matches or matches[0].score < self.retriever.threshold:
            return self._fallback("low_confidence")

        # Stage 4: render from the top match only.
        top = matches[0]
        related_floor = self.retriever.threshold * 0.6
        related = [
            {"title": m.entry["lesson"], "link": m.entry["link"]}
            for m in matches[1:]
            if m.score >= related_floor and m.entry["id"] != top.entry["id"]
        ][:2]
        return {
            "kind": "match",
            "message": MESSAGES["match"],
            "primary": _card(top.entry),
            "related": related,
            "score": round(top.score, 3),
        }

    def _fallback(self, kind: str) -> dict:
        return {
            "kind": kind,
            "message": MESSAGES[kind],
            "primary": _card(self.trusted),
            "related": [],
        }


EVAL_QUESTIONS = [
    "How do I stop relying on willpower?",
    "What habits protect my diet?",
    "How do I eat out on low carb?",
    "Why do I get decision fatigue with food?",
    "Can you diagnose why I'm always hungry?",
    "asdfgh random gibberish",
]


def run_eval() -> None:
    bot = Chatbot()
    print(f"backend: {type(bot.retriever).__name__}\n")
    for question in EVAL_QUESTIONS:
        result = bot.answer(question)
        score = f" (score {result['score']})" if "score" in result else ""
        print(f"  {question}")
        print(f"    -> {result['kind']}: {result['primary']['title']}{score}\n")


if __name__ == "__main__":
    run_eval()
