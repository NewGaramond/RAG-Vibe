# Memory (Running Summary)

Lightweight conversation memory that keeps a concise running summary of older turns while preserving the most recent exchanges verbatim. Designed to give the generator compact context without ballooning prompt size.

## What this module does

* **Tracks a running summary** of the conversation so far, written in a neutral, factual style.
* **Respects a token budget**: if the combined summary + history would exceed the budget, it compresses older turns and keeps only the latest ones verbatim.
* **Preserves recency** by retaining the last N user/assistant pairs exactly as written.
* **Reports telemetry** about when summarization happens and how much content is kept vs. dropped.

## Components

* **Token counter**: estimates tokens for the current summary and rendered history using the same tokenizer family as modern OpenAI models. Used solely to decide whether to summarize.
* **MemoryConfig**: central configuration (chat model to use for summarization, API key, token budget, and how many recent turns to keep verbatim).
* **History renderer**: converts the stored message list into a compact, readable format for the summarizer.
* **Summarizer prompt**: a stable system instruction that asks for a concise, neutral, factual running summary that preserves important facts, decisions, names, and references.
* **Update function**: orchestrates the decision:

  * If within budget: append the new user and assistant turns to history without calling a model.
  * If over budget: summarize the older portion, keep the recent N pairs verbatim, and return the updated summary + trimmed history.
* **Telemetry**: returns a small report indicating whether summarization occurred, an estimate of the token size, how many messages were kept/dropped, and the size of the new summary.

## How it decides to summarize

1. Forms a candidate view consisting of the current summary plus the entire history including the new user+assistant turns.
2. Estimates tokens for that candidate view.
3. If under the configured budget, no summarization is performed and all turns are kept.
4. If over budget, only the most recent N pairs are preserved verbatim; the older portion is condensed into an updated running summary by the chat model.

## Key behaviors and guarantees

* **Deterministic intent**: summarization uses temperature 0 to reduce variance.
* **No tool leakage**: the summary is context metadata, not user-visible output.
* **Graceful degradation**: if you wrap this module in your graph (recommended), failures in summarization can be treated as non-blocking, keeping prior memory intact.

## Configuration knobs

* **Model**: which chat model performs summarization.
* **Token budget**: the maximum combined size for the summary + rendered history used by the summarizer trigger.
* **Recent turns**: how many of the latest user/assistant pairs remain verbatim after summarization.

## Outputs (telemetry)

* Whether summarization happened during this update.
* An approximate token count of the candidate view.
* The size (in tokens) of the new summary.
* Counts of messages kept and dropped from the verbatim tail.

## Integration notes

* Call the update after each assistant reply to keep memory fresh.
* Inject the returned summary and the recent verbatim turns into your next RAG prompt as auxiliary context (e.g., “Conversation summary” + “Recent turns” sections).
* Choose a token budget that leaves headroom for your retrieval context and question.

## Privacy & compliance

* Summaries may contain sensitive information present in the conversation. If persisting memory, consider encryption at rest, retention limits, access controls, and user-initiated resets.

## Limitations

* Token estimation is approximate; set conservative budgets.
* Important nuance or phrasing in older turns may be compressed; increase the number of preserved recent pairs or the budget if this matters.
* The module maintains a single running summary; it does not implement entity-level memory or per-topic threads (these can be layered on later).
