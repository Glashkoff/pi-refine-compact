# pi-refine-compact

A [pi](https://github.com/earendil-works/pi) package that replaces the stock summarization (`/compact` and auto-compaction) with **chunked refine summarization** performed by a cheaper model of your choice.

Instead of one giant summarization call on the session model, the history is cut into chunks at turn boundaries and merged sequentially into a structured checkpoint (Refine, [arXiv 2308.15022](https://arxiv.org/abs/2308.15022)) — so a small local or cheap cloud model can do the job the main model no longer needs to pay for.

## How it works

- **Chunking** — the compacted prefix is cut into at least `ceil(main_ctx / summarizer_ctx)` chunks, only at turn boundaries (a tool result is never separated from its tool call), with a safety budget per chunk.
- **Merging** — each chunk updates a growing summary in a strict format (Goal / Constraints / Progress / Key Decisions / Next Steps / Critical Context), following the recursive summarization method of [arXiv:2308.15022v4](https://arxiv.org/abs/2308.15022) (Wang et al., *Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models*, Neurocomputing 2025): the first chunk is memorized on its own, every following chunk updates the previous memory with its context, exactly the paper's `update_memory` recursion.
- **Resilience** — a chunk that hits the summarizer's output limit is re-split at turn boundaries (the length-stop path retries up to 2 levels of splits; a separate preemptive splitter cuts chunks whose serialized prompt exceeds `maxPromptTokens` before calling the model); transient errors retry once; computed chunks are cached in memory, so a retry never redoes finished work. If the intermediate checkpoint compression fails, compaction continues with the uncompressed summary (degraded mode) instead of aborting.
- **Summary ceiling** — when the accumulated summary outgrows its ceiling, the checkpoint itself is compressed by an intermediate call, so chunk + summary always fit into the summarizer's window.
- **English summaries** — regardless of the conversation language (small models are measurably better with EN; code identifiers and paths stay verbatim).
- **Freshness marker** — the final summary is stamped with the checkpoint time and the read/modified file lists, like pi's stock compaction.

## Install

```sh
pi install git:github.com/Glashkoff/pi-refine-compact
# pinned:
pi install git:github.com/Glashkoff/pi-refine-compact@v0.1.0
```

## Use

1. Run `/compact-model` in pi and pick the summarization model (the menu is the same style as `/model`, with context sizes shown).
2. Work until compaction triggers (manually via `/compact`, or automatically at the context threshold).

That's it. Without a selected model — or when the extension is disabled — pi's stock behavior is used.

## Settings

File: `$PI_CODING_AGENT_DIR/pi-refine-compact-settings.json` (i.e. `~/.pi/agent/pi-refine-compact-settings.json` by default).

Every field is optional. An omitted field means **auto** — a formula derived from the chosen summarizer's context window and pi's `compaction.reserveTokens` setting. An invalid value is reported in the UI and ignored (the auto formula applies); nothing fails silently.

| Key | Type | Auto (default) | Meaning |
|---|---|---|---|
| `model` | `string` \| `null` | `null` — session model (stock behavior) | Summarizer as `provider/id`; also set by `/compact-model` |
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

## Notes

- The model's own context window, when known, drives chunking; unknown windows fall back to 32k.
- If the selected model disappears from the registry, a warning is shown and stock compaction proceeds for that session.
- Extensions run with full system access — review the source before installing (it's one file: `extensions/pi-refine-compact.ts`; the only network calls it makes are the summarizer LLM requests you configure).

## Compatibility

Built and type-checked against `@earendil-works/pi-coding-agent` **0.87.x**. pi is 0.x — the `session_before_compact` hook and `modelRegistry` API may change in minor releases; pin the install ref if stability matters.

## License

[MIT](LICENSE)
