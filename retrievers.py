"""Retrieval backends behind one shared interface.

Every backend selects knowledge base entries; none of them ever writes
answer text. Scores are normalized to 0.0-1.0 so the confidence gate in
chatbot.py is comparable across backends. Select a backend with the
RETRIEVER environment variable: keyword | embedding (default) | llm.
"""

from __future__ import annotations

import json
import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Match:
    entry: dict  # the full knowledge base entry
    score: float  # normalized relevance, 0.0 to 1.0


class Retriever(ABC):
    """Common contract: retrieve() returns matches sorted best first."""

    threshold: float = 0.0  # top score below this means low confidence

    def __init__(self, kb: list[dict]):
        self.kb = kb

    @abstractmethod
    def retrieve(self, query: str) -> list[Match]:
        """Return matches sorted highest score first. May be empty."""


# ---------------------------------------------------------------------------
# Keyword backend: zero external dependencies, always runnable.
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "i", "im", "my", "me", "we", "you", "your",
    "it", "its", "is", "are", "was", "be", "been", "am",
    "do", "does", "did", "have", "has", "had",
    "how", "what", "why", "when", "which", "who",
    "can", "could", "should", "would", "will",
    "and", "or", "but", "if", "then", "than",
    "this", "that", "these", "those",
    "to", "of", "for", "with", "at", "by", "from", "as", "about", "on", "in",
    "not", "no", "so", "just", "get", "got", "really", "always", "keep",
}


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in _STOPWORDS}


class KeywordRetriever(Retriever):
    """Word-overlap scoring: fraction of query tokens found in the entry."""

    threshold = 0.15

    def __init__(self, kb: list[dict]):
        super().__init__(kb)
        # Tokenize the corpus once at startup, never per query.
        self._entry_tokens = [
            _tokenize(
                " ".join(
                    [e["question"], e["lesson"], e["summary"], *e["when_this_helps"]]
                )
            )
            for e in kb
        ]

    def retrieve(self, query: str) -> list[Match]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        matches = []
        for entry, tokens in zip(self.kb, self._entry_tokens):
            score = len(query_tokens & tokens) / len(query_tokens)
            if score > 0:
                matches.append(Match(entry=entry, score=score))
        return sorted(matches, key=lambda m: m.score, reverse=True)


# ---------------------------------------------------------------------------
# Embedding backend: semantic search, default. No API key needed.
# ---------------------------------------------------------------------------


class EmbeddingRetriever(Retriever):
    """Cosine similarity over sentence-transformers embeddings, in memory."""

    threshold = 0.35
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, kb: list[dict]):
        super().__init__(kb)
        # Imported here so the module still loads without the dependency.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.MODEL_NAME)
        texts = [f"{e['question']} {e['lesson']} {e['summary']}" for e in kb]
        # Normalized once at startup; cosine similarity becomes a dot product.
        self._vectors = self._model.encode(texts, normalize_embeddings=True)

    def retrieve(self, query: str) -> list[Match]:
        query_vec = self._model.encode([query], normalize_embeddings=True)[0]
        sims = self._vectors @ query_vec
        matches = [
            Match(entry=entry, score=max(0.0, float(sim)))
            for entry, sim in zip(self.kb, sims)
        ]
        return sorted(matches, key=lambda m: m.score, reverse=True)


# ---------------------------------------------------------------------------
# LLM backend: the model is a router only. It selects an id; it never
# writes answer text, so grounding holds even with an LLM in the loop.
# ---------------------------------------------------------------------------


class LLMRetriever(Retriever):
    """Asks the model for the best entry id plus a confidence, as JSON."""

    threshold = 0.5
    MODEL = "claude-haiku-4-5-20251001"

    def __init__(self, kb: list[dict]):
        super().__init__(kb)
        # Imported here so the module still loads without the dependency.
        import anthropic

        self._client = anthropic.Anthropic()  # needs ANTHROPIC_API_KEY
        self._by_id = {e["id"]: e for e in kb}
        self._catalog = json.dumps(
            [
                {"id": e["id"], "question": e["question"], "summary": e["summary"]}
                for e in kb
            ],
            indent=2,
        )

    def retrieve(self, query: str) -> list[Match]:
        prompt = (
            "You route user questions to knowledge base entries. "
            "Here is the catalog:\n\n"
            f"{self._catalog}\n\n"
            f"User question: {query}\n\n"
            "Reply with ONLY a JSON object, no prose, of the form "
            '{"id": "<best matching entry id, or null if none fit>", '
            '"confidence": <number 0 to 1>}'
        )
        try:
            response = self._client.messages.create(
                model=self.MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
        except Exception as exc:  # network/auth errors -> low confidence path
            print(f"LLMRetriever: request failed ({exc})", file=sys.stderr)
            return []
        return self._parse(text)

    def _parse(self, text: str) -> list[Match]:
        # Defensive parsing: strip code fences, tolerate surrounding noise.
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
        try:
            data = json.loads(text)
            entry = self._by_id.get(data.get("id"))
            confidence = float(data.get("confidence", 0.0))
        except (ValueError, TypeError, AttributeError):
            return []
        if entry is None:
            return []
        return [Match(entry=entry, score=min(max(confidence, 0.0), 1.0))]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_retriever(kb: list[dict]) -> Retriever:
    """Build the backend named by RETRIEVER (default: embedding).

    Falls back to KeywordRetriever rather than crashing, so the app
    always runs even when optional dependencies are unavailable.
    """
    choice = os.environ.get("RETRIEVER", "embedding").strip().lower()
    if choice == "keyword":
        return KeywordRetriever(kb)
    if choice == "llm":
        try:
            return LLMRetriever(kb)
        except Exception as exc:
            print(
                f"LLM backend unavailable ({exc}); falling back to keyword.",
                file=sys.stderr,
            )
            return KeywordRetriever(kb)
    if choice != "embedding":
        print(
            f"Unknown RETRIEVER={choice!r}; using embedding.", file=sys.stderr
        )
    try:
        return EmbeddingRetriever(kb)
    except Exception as exc:
        print(
            f"Embedding backend unavailable ({exc}); falling back to keyword.",
            file=sys.stderr,
        )
        return KeywordRetriever(kb)
