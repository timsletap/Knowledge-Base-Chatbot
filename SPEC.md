# Build Specification: Knowledge-Base Chatbot

This document is the authoritative build spec. It defines file responsibilities, data contracts, interfaces, and acceptance criteria. Build to these interfaces exactly. Where this document and the README disagree, this document wins.

## 1. Goal

Build a chatbot that answers a user's question by pointing to the single best matching entry in a fixed knowledge base. The bot returns the entry's title, format, link, and summary, plus up to two related links. It must never generate health content of its own.

Non-goals: user accounts, persistence across sessions, analytics, multi-turn context, editing the knowledge base from the UI, and any live integration with third-party platforms.

## 2. Hard constraints

These are requirements, not preferences. A build that violates any of them is incorrect.

1. **Grounding.** Answer text is assembled only from fields that exist in `knowledge_base.json`. No model output is ever shown to the user as prose. This must hold for every retrieval backend.
2. **No medical advice.** Requests for personal diagnosis or treatment are declined before retrieval runs.
3. **No guessing.** If retrieval confidence is below threshold, the bot says so and redirects rather than returning a weak match.
4. **Swappable retrieval.** Changing the retrieval method must not require edits to the UI or orchestration layers.
5. **Runs with no API key.** The default configuration must work offline after dependencies are installed.

## 3. Stack

- Python 3.10 or later
- Streamlit for the UI
- `sentence-transformers` (model `all-MiniLM-L6-v2`) and `numpy` for the default retrieval backend
- `anthropic` optional, only for the LLM backend
- Deployment target: Streamlit Community Cloud

No database, no vector store, no web framework. The knowledge base is small enough that all retrieval happens in memory.

## 4. File layout and responsibilities

```
kb-chatbot/
  app.py                Streamlit UI. Renders only. No business logic.
  chatbot.py            Orchestration. Owns the four-stage pipeline.
  retrievers.py         Retrieval backends behind one shared interface.
  knowledge_base.json   Source of truth. Data only.
  requirements.txt
  README.md
  SPEC.md
```

Layering rule: `app.py` imports `chatbot.py`, and `chatbot.py` imports `retrievers.py`. Never the reverse. `app.py` must never import `retrievers.py` directly or contain scoring, safety, or knowledge base logic.

## 5. Data contract: knowledge base

`knowledge_base.json` is a JSON array of objects. Every object has these fields, all required:

| Field | Type | Purpose |
|---|---|---|
| `id` | string | Stable slug, unique. Used for lookups and by the LLM backend. |
| `question` | string | The canonical question this entry answers. |
| `lesson` | string | Display title. |
| `format` | string | Resource type, for example "Article". |
| `link` | string | URL to the resource. |
| `summary` | string | Short description shown to the user. |
| `when_this_helps` | array of strings | Context lines, used to enrich the search text. |

One entry must have the id `trusted-health-info`. It is the fallback target for both the safety refusal and the low-confidence path. If it is missing, the app should fail loudly at startup rather than silently.

## 6. Interface: retrieval backends

Every backend subclasses a common base and satisfies this contract:

```python
@dataclass
class Match:
    entry: dict     # the full knowledge base entry
    score: float    # normalized relevance, 0.0 to 1.0

class Retriever(ABC):
    threshold: float              # top score below this means low confidence
    def __init__(self, kb: list[dict]): ...
    def retrieve(self, query: str) -> list[Match]: ...
        # returns matches sorted best first; may return an empty list
```

Rules for all implementations:

- Scores must be normalized to the 0.0 to 1.0 range so the confidence gate is comparable across backends.
- Results must be sorted highest score first.
- An empty list is a valid return and must be handled by the caller as low confidence.
- Any expensive setup, such as loading a model or embedding the corpus, happens once in `__init__`, never per query.
- A backend selects entries. It never writes answer text.

### Required implementations

**`KeywordRetriever`** (threshold 0.15). Zero external dependencies. Tokenizes the query and each entry's combined searchable text (`question`, `lesson`, `summary`, `when_this_helps`), removes stopwords, and scores by the fraction of query tokens found in the entry. Exists so the app always runs and so the eval is reproducible without network access.

**`EmbeddingRetriever`** (threshold 0.35). Default backend. Encodes each entry's `question`, `lesson`, and `summary` into normalized embedding vectors once at startup. At query time, encodes the query and ranks entries by cosine similarity. Because vectors are normalized, cosine similarity is a dot product.

**`LLMRetriever`** (threshold 0.5). The model acts strictly as a router. It receives a catalog of entry ids, questions, and summaries, and must reply with only a JSON object of the form `{"id": "<entry id or null>", "confidence": <0..1>}`. Parse the response defensively: strip code fences, and on any parse failure or unknown id return an empty list. The model's prose is never surfaced to the user. This is what preserves grounding when an LLM is in the loop.

### Backend selection

A factory function reads the `RETRIEVER` environment variable: `keyword`, `embedding`, or `llm`. Default is `embedding`. If the embedding backend cannot initialize, for example because the dependency is missing or the model download fails, fall back to `KeywordRetriever` rather than crashing.

Import third-party dependencies inside the class that needs them, not at module top level, so the module still loads when optional dependencies are absent.

## 7. Pipeline: `chatbot.py`

`Chatbot.answer(query: str) -> dict` runs four stages in this order.

**Stage 1, safety gate.** Runs before retrieval. Matches the query against a narrow list of regex patterns indicating a request for personal medical advice, for example `diagnos(e|is|ing)`, `should i (take|stop|start|use)`, `is it safe for me`, `prescri(be|ption)`, `do i have`, `(treat|cure) my`. On a match, return the refusal response immediately.

The patterns must stay narrow. A broad pattern such as `why do i` would wrongly refuse the legitimate question "Why do I get decision fatigue with food?", which is a required passing case. Any change to these patterns must be re-tested against the full eval set in section 9.

**Stage 2, retrieval.** Call `self.retriever.retrieve(query)`.

**Stage 3, confidence gate.** If the result list is empty, or the top score is below `self.retriever.threshold`, return the low-confidence response.

**Stage 4, render.** Build the response from the top match. Include up to two related entries drawn from the next-best matches, filtered to those scoring at least 60 percent of the threshold, so weak or irrelevant related links are not shown.

### Response shape

`answer()` always returns a dict with this shape, regardless of path:

```python
{
    "kind": "match" | "refusal" | "low_confidence",
    "message": str,          # one line of framing text, from a fixed set of strings
    "primary": {             # always present; fields copied from a KB entry
        "title": str,        # entry["lesson"]
        "format": str,
        "link": str,
        "summary": str,
    },
    "related": [             # possibly empty
        {"title": str, "link": str},
    ],
    "score": float,          # present on "match" only
}
```

For `refusal` and `low_confidence`, `primary` is the `trusted-health-info` entry and `related` is empty. The `message` strings are fixed literals in the code, never model generated.

Constructor signature: `Chatbot(kb_path="knowledge_base.json", retriever=None)`. When `retriever` is None, use the factory. Accepting an injected retriever keeps the class testable and makes the swap explicit.

## 8. UI: `app.py`

A Streamlit chat interface and nothing more.

- Cache the `Chatbot` instance with `@st.cache_resource` so the knowledge base and any model load once per session, not on every rerun.
- Keep the message history in `st.session_state`. Store user turns as strings and assistant turns as the response dict returned by `answer()`.
- Streamlit reruns the whole script on each interaction, so replay the stored history on every run before handling new input. Store the response dict rather than pre-rendered markup so replay stays consistent.
- Render `primary` as title, format, link, and summary, then related links if present.
- Show a caption stating that the bot does not give personal medical advice and only points to knowledge base resources.

## 9. Acceptance criteria

The build is complete when all of the following pass.

**Functional eval.** Running the six questions below produces the stated outcome. Use `RETRIEVER=keyword` for a reproducible offline check, then confirm the same primary matches with the default embedding backend.

| Question | Expected `kind` | Expected primary entry |
|---|---|---|
| How do I stop relying on willpower? | match | Appetite vs Craving |
| What habits protect my diet? | match | Habit Formation Basics |
| How do I eat out on low carb? | match | Eating Out, Smarter Choices |
| Why do I get decision fatigue with food? | match | Decision Fatigue |
| Can you diagnose why I'm always hungry? | refusal | Trusted Health Information |
| asdfgh random gibberish | low_confidence | Trusted Health Information |

The final two rows are the ones that prove the guardrails work. They are not optional.

**Structural checks.**

- Setting `RETRIEVER` to each of the three values changes behavior with no edits to `app.py` or `chatbot.py`.
- `app.py` contains no scoring, pattern matching, or knowledge base access.
- Every string shown to the user is either a fixed literal or a field copied from `knowledge_base.json`.
- `python chatbot.py` runs the eval set from the command line without Streamlit.
- The app starts and answers correctly with no `ANTHROPIC_API_KEY` set.

## 10. Deliverables

1. A public GitHub repository containing the files in section 4.
2. A working deployment on Streamlit Community Cloud, linked from the README.
3. A README covering what the app does, how it works, how to run it locally, the eval outputs from section 9 pasted verbatim, and an architecture-only integration note for Disciple Media and GoHighLevel.

## 11. Implementation notes

Build and verify in this order: knowledge base file, then `KeywordRetriever` and the pipeline tested from the command line, then the safety and confidence gates against the eval set, then the Streamlit UI, then the embedding backend, then the LLM backend last since it is optional.

Tune thresholds against the full eval set, not against a single question. Raising a threshold to sharpen the gibberish rejection can push a legitimate question into the low-confidence path, so re-run all six after any change.

Prefer clarity over cleverness throughout. The knowledge base is roughly nine entries and the entire retrieval step is a ranked comparison over a small list. Any solution that introduces a vector database, a caching layer, or an async job queue is over-engineered for this scope.
