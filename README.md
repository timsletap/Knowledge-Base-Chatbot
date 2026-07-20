# Knowledge-Base Chatbot

A chatbot that answers questions by pointing users to the exact resource in a knowledge base: lesson title, format, link, and a short summary. It never invents answers and never gives personal medical advice. It only retrieves and guides from the provided knowledge base.

**Live demo:** _add your Streamlit Cloud link_
**Repo:** https://github.com/timsletap/Knowledge-Base-Chatbot

## Overview

The guiding principle is retrieve and render, never generate. The bot selects the best matching knowledge base entry and displays that entry's own fields. It never writes health content of its own, so it cannot hallucinate or give advice that isn't in the knowledge base.

## How it works

Every question flows through four stages, handled in `chatbot.py`:

1. **Safety gate.** Narrow patterns detect requests for personal medical advice, such as "diagnose" or "should I take". On a match, the bot declines and points to the Trusted Health Information entry.
2. **Retrieval.** A swappable backend ranks knowledge base entries by relevance and returns the top matches with a score.
3. **Confidence gate.** If the best score is below the backend's threshold, the bot says it found no close match and points to Trusted Health Information.
4. **Render.** The reply is built only from the matched entry's fields (title, format, link, summary), plus up to two related links.

Because the model only selects an entry and never writes the answer text, grounding holds no matter which backend is running.

### How retrieval works

Each knowledge base entry and each user question is converted into an embedding, a list of numbers (a vector) that represents the text's meaning rather than its exact words. To answer a question, we compare its vector against every entry's vector using cosine similarity, which scores how closely two vectors point in the same direction on a 0 to 1 scale. The highest scoring entry is the resource we return, and that score also tells us whether the match is strong enough to answer at all. If nothing scores well, we fall back to a trusted information link instead of guessing.

## Retrieval backends

All backends share one interface, so the UI and orchestration never change when you swap them. Select one with the `RETRIEVER` environment variable.

| `RETRIEVER` | Backend | Notes |
|---|---|---|
| `embedding` (default) | Semantic search with `sentence-transformers`, cosine similarity in memory | Best at matching reworded questions. No API key. |
| `keyword` | Word-overlap scoring, zero dependencies | Always runnable. Automatic fallback if embeddings cannot load. |
| `llm` | LLM as router. The model returns only the best entry id plus a confidence, as JSON | Needs `ANTHROPIC_API_KEY`. Still grounded, since the model selects and does not write the answer. |

The knowledge base lives in `knowledge_base.json`, separate from all code, so adding entries never touches logic.

## Project structure

```
kb-chatbot/
  app.py                Streamlit chat UI, render only
  chatbot.py            Orchestration: safety, retrieve, confidence, render
  retrievers.py         Swappable backends behind one interface
  knowledge_base.json   Source of truth
  requirements.txt
  README.md
```

## Run locally

```
git clone <your-repo-url>
cd kb-chatbot
pip install -r requirements.txt
streamlit run app.py
```

Pick a backend explicitly:

```
RETRIEVER=keyword streamlit run app.py
RETRIEVER=llm ANTHROPIC_API_KEY=sk-... streamlit run app.py
```

Check the logic without the UI:

```
RETRIEVER=keyword python chatbot.py
```

## Sample outputs

Verified run of the six required questions.

| Question | Result | Resource returned |
|---|---|---|
| How do I stop relying on willpower? | match | Appetite vs Craving |
| What habits protect my diet? | match | Habit Formation Basics |
| How do I eat out on low carb? | match | Eating Out, Smarter Choices |
| Why do I get decision fatigue with food? | match | Decision Fatigue |
| Can you diagnose why I'm always hungry? | refusal | Trusted Health Information |
| asdfgh random gibberish | low confidence | Trusted Health Information |

The last two rows are the ones that matter most. They show the safety gate and the confidence gate working as intended instead of the bot guessing.

## Integration note: Disciple Media and GoHighLevel

This is an architecture sketch only, with no implementation.

**Goal.** A member asks a question inside a community app (Disciple Media) and gets pointed to the right knowledge base resource, where the knowledge base and LLM tooling live in a CRM (GoHighLevel).

**Platform facts.** GoHighLevel offers a native Conversation AI with an uploadable knowledge base, a Conversation AI public API, and webhooks. Disciple Media is a branded community app with an admin console and a public REST API.

**The catch.** GoHighLevel's native Conversation AI is generative, so it paraphrases from the knowledge base and can drift. That conflicts with the strict "point to a resource, never invent" requirement. The safer pattern keeps GoHighLevel as the data and identity layer and keeps the deterministic guardrails in a thin middleware, which is essentially the logic in this project.

**Flow.**

```
Member asks in Disciple
      |
      v
Middleware (this project's logic)
  safety gate, confidence gate, retrieve and render
      |
      v
GoHighLevel
  knowledge base is the source of truth
  contact record is the member identity
      |
      v
Matched entry returned to the member in Disciple
```

Guardrails stay in the middleware rather than being left to a generative bot. The component we built is the recommended production middleware.

## Tech choices and tradeoffs

The bot never puts an LLM in the answer path, even with `RETRIEVER=llm`, which is the strongest guarantee against hallucination and unwanted advice. Similarity is computed in memory with no vector database, since the knowledge base is small and a database would be over-engineering. Embeddings are the default over keyword matching because they handle reworded questions better, with keyword kept as a dependency free fallback. The safety gate uses simple, transparent patterns for this scope, where a production system would use a trained classifier.
