# pi-refine-compact

A [pi](https://github.com/earendil-works/pi) package that replaces the stock summarization (`/compact` and auto-compaction) with **chunked refine summarization** performed by a cheaper model of your choice.

Instead of one giant summarization call on the session model, the history is cut into chunks at turn boundaries and merged sequentially into a structured checkpoint using the *refine* summarizer ([arXiv 2308.15022](https://arxiv.org/abs/2308.15022)) — so the expensive summarization moves off the main model, and a small local model or a cheap cloud model can do the job.

**Note:** This extension does not affect the summarization used by `/tree`. Branch summarization solves a different problem and calls for a different approach.

## Install

```sh
pi install npm:pi-refine-compact
```

Alternative — install straight from the git source:

```sh
pi install git:github.com/Glashkoff/pi-refine-compact
```

## Use

1. Run `/compact-model` in pi and pick the summarization model (the menu is the same style as `/model`, with context sizes shown). It offers `(default) native: pi's stock compaction`, `session model (refinement): <current model>`, and every available `provider/id`.
2. Work until it's time to compact — manually via `/compact`, or automatically at the context threshold.

That's it. The native entry (the default) and a disabled extension leave pi's stock behavior untouched; `session` runs the refine pipeline on whatever model the session is using at compaction time.

## Notes

- The model's own context window, when known, drives chunking; unknown windows fall back to 32k.
- If the selected model disappears from the registry, a warning is shown and stock compaction proceeds for that session. `session` mode degrades the same way when the session has no model, and also when it is a virtual (router-only) entry, which cannot complete a request — the extension does no virtual routing.
- Cached chunks are keyed by the session, the resolved summarizer, and the derived budgets, so switching any of them recomputes instead of reusing stale text.
- Extensions run with full system access — review the source before installing (it's one file: `extensions/pi-refine-compact.ts`; the only network calls it makes are the summarizer LLM requests you configure).

## Settings

File: `$PI_CODING_AGENT_DIR/pi-refine-compact-settings.json` (i.e. `~/.pi/agent/pi-refine-compact-settings.json` by default).

Every field is optional. An omitted field means **auto** — a formula derived from the chosen summarizer's context window and pi's `compaction.reserveTokens` setting. An invalid value is reported in the UI and ignored (the auto formula applies); nothing fails silently.

| Key | Type | Auto (default) | Meaning |
|---|---|---|---|
| `model` | `string` \| `null` | `null` — pi's stock compaction | Summarizer as `provider/id`, `"session"` for the session's current model, or `null` for native compaction; also set by `/compact-model` |
| `maxPromptTokens` | int ≥ 0 | `min(32000, 0.5 × summarizer ctx)` | Hard cap on one summarizer request; a larger chunk is split preemptively |
| `maxOutputTokens` | int ≥ 0 | `min(max(2048, 0.8 × reserveTokens), sumCtx − maxPromptTokens − ~1000)` | `maxTokens` for summarizer calls; clamped to the model's free headroom (the clamp applies to the auto value; an explicit setting is used as-is, with a warning if it exceeds the headroom) |
| `summaryCeilingTokens` | int ≥ 0 | `min(0.35 × summarizer ctx, 0.5 × maxPromptTokens)` | When the accumulated summary exceeds this, the checkpoint itself is compressed |
| `charsPerToken` | number ≥ 1 | `3` | Chars-per-token estimate for your typical conversation language (see below) |

The auto prompt cap is `min(32000, 0.5 × summarizer ctx)` — e.g. 16000 tokens on the 32k fallback window, 32000 for any summarizer with ≥ 64k context. It scales down for smaller summarizers so the cap always leaves room for instructions, the previous summary, and the reply. If you want large summarizers to receive larger chunks, set `maxPromptTokens` explicitly.

### `charsPerToken`

Chunk sizes are estimated from character counts. The default `3` is tuned for Cyrillic prose mixed with JSON/tool-call wrappers (~2.4 real chars/token). Rough guidance:

| Typical conversation | `charsPerToken` |
|---|---|
| Cyrillic + code | `3` (default) |
| Mostly English prose + code | `4` |
| CJK text | `1.5`–`2` |

### Example

```json
{
  "model": "llamacpp/Ling-3.0-tiny",
  "maxPromptTokens": 16000,
  "charsPerToken": 4
}
```

## Details

### How it works

**Chunked refine** — the compacted prefix is cut into chunks at turn boundaries and merged sequentially by a cheaper summarizer into a structured checkpoint.
**Cheap and long** — the compression runs on a small local or cloud model, so the main model doesn't pay for it and long histories fit on a modest context window.
**Resilient** — chunks re-split on output limits, computed chunks are cached per session/model/budgets, and a failing intermediate compression degrades instead of aborting.

See [DESIGN.md](docs/DESIGN.md) for the full algorithm — budget derivation, chunking, the refine loop, resilience, and the char-based sizing rationale.

### Trade-offs and limitations

This approach trades off latency, precision, and token efficiency for cheaper main-model usage and long-history compression on small/cheap summarizers. It works best when the summarizer is significantly less expensive than the main model.

**Strengths:**

- **Cheaper main-model usage** — the compression step runs on the chosen summarizer (often a small, cheap or local model) instead of the main model, so you don't burn expensive context on a pure housekeeping task.
- **Very long histories on a small window** — chunking plus the recursive
- **Structured, downstream-friendly checkpoint** — the fixed format is easy for another LLM to consume and preserves file paths, commands, and decisions verbatim.
- **Resilient by design** — preemptive and length-cap splits, one transient retry, in-memory reuse of computed chunks, and degraded mode for checkpoint compression mean compaction almost never aborts outright.
- **Multi-stage aware** — it seeds from `previousSummary`, so repeated compactions build on the prior checkpoint instead of starting over.

**Weaknesses:**

- **Many calls, more latency** — one summarize call per chunk (plus splits and occasional checkpoint compression) instead of a single stock compaction call. On a cheap model the per-token cost is lower, but wall-clock latency and the number of round-trips are higher.
- **Redundant re-processing** — the accumulated summary is re-sent with every chunk, so the summarizer reprocesses the same content repeatedly. That extra token cost is the price of fitting a large history into a small window.
- **Cumulative information loss** — each refine step compresses already-compressed memory (lossy compression of lossy data). Over many chunks and repeated compactions the checkpoint slowly drifts; fine-grained details are lost first (the recursive method's well-known weakness, not a bug in this implementation).
- **Bounded by the summarizer's quality** — the checkpoint is only as good as the summarizer running it; a weak summarizer yields a weak checkpoint regardless of how good the main model is.
- **Imprecise size estimation** — the char-based `charsPerToken` estimate (`approxTokensOfChars = ceil(chars / charsPerToken)`) can be off, causing occasional over- or under-sizing of chunks; the splitters handle this at extra cost (see `charsPerToken`).
- **Interdependent budget constants** — the whole budget chain is tuned as a unit (see DESIGN.md); tuning one knob in isolation tends to break it.
- **Additional model dependency** — the extension needs a second model configured; if it is slow, unavailable, or misconfigured, compaction degrades or falls back to stock (which loses the benefit).

## Compatibility

Built and type-checked against `@earendil-works/pi-coding-agent` **0.87.x**. pi is 0.x — the `session_before_compact` hook and `modelRegistry` API may change in minor releases; pin the install ref if stability matters.

## License

[MIT](LICENSE)
