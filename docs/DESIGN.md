# pi-refine-compact — design notes

Algorithm basis: recursive summarization from arXiv:2308.15022v4 (Wang et al.,
"Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language
Models", Neurocomputing 2025; official code github.com/qingyue2014/Rsum). The
paper's core loop — memorize a small dialogue context, then recursively produce
new memory from previous memory + next context (`update_memory`), finally
respond from the latest memory — maps onto compaction as: chunk 1 → initial
summary (INITIAL_PROMPT), every chunk k>1 → update previous summary with the
chunk (UPDATE_PROMPT), final summary → the new session prefix. Deviations from
the paper are engineering adaptations for pi, listed in "What is deliberately
NOT configurable" below.

Why the constants are what they are, and which of them are configurable.

## Configurable (settings file) vs internal (code)

Principles followed (mirroring pi's own settings conventions):

- tokens are non-negative safe integers, names in camelCase with the unit in the name;
- omitted field = auto formula, explicit value wins;
- an invalid value is reported on read and ignored — never a silent fallback to a
  "best guess" that differs from the documented auto behavior;
- fractions and one-off heuristics stay internal: they are implementation
  details of the algorithm, not knobs.

## The knobs

### maxPromptTokens (auto: min(32000, 0.5 × sumCtx))

The cap on a single serialized summarizer request. Two failure modes exist:

- the provider replies with `exceed_context_size_error` (`stop: "error"`) — the
  extension's retry treats that as a transient error and fails the compaction;
- the reply hits `maxTokens` (`stop: "length"`) — that path is handled by
  re-splitting the chunk (LengthCapError).

Hence the *preemptive* split: the cap must fire before the call, based on the
char-based estimate. The estimate undershoots real tokenizer counts (see
charsPerToken), so the auto cap is deliberately bounded by half the summarizer's
context — the safety margin for prompt instructions + previous summary + reply.

The 32000 constant reproduces the original hardcoded cap; auto = the more
conservative of it and 50% of the context window.

### maxOutputTokens (auto: clamped 80% of reserveTokens, floor 2048)

pi's `compaction.reserveTokens` (16384 by default) is calibrated for the main
model's response budget. It is reused here as the summarizer's reply budget —
but a small summarizer's API may reject a too-large `maxTokens`, and worse, a
`maxTokens` that together with the chunk exceeds the window produces a
deterministic `exceed_context_size_error`. Therefore the auto value is clamped
to the model's free headroom: `hardCeiling = max(512, sumCtx − maxPromptTokens −
1000)`; auto = `hardCeiling < 2048 ? hardCeiling : min(hardCeiling, max(2048,
0.8 × reserveTokens))`. The clamp applies to the auto value only; an explicit
`maxOutputTokens` setting is used as-is (with a warning when it exceeds the
headroom). The reply hitting the cap is not a fatal error: the chunk is
re-split and re-summarized (LengthCapError → splitInto2).

### When checkpoint compression fails (degraded mode)

The intermediate compression of an oversized accumulated summary can itself
fail deterministically (e.g. its own length cap). Rather than aborting the
whole compaction (which would repeat on every retry), the extension warns and
continues with the uncompressed summary. The next refine call may then fail
with a context-exceeded error — the same outcome as before compression
existed — but transient/one-off compression failures no longer kill a run.

### summaryCeilingTokens (auto: min(0.35 × sumCtx, 0.5 × maxPromptTokens))

The refine algorithm accumulates a summary across chunks; without a ceiling it
grows monotonically and eventually competes with the chunk for context space.
When the ceiling is crossed, the checkpoint itself is compressed to ~60% of its
estimated size (never below 2000 tokens). The 0.35 fraction keeps chunk +
summary + instructions inside the window with margin; the 0.5 × maxPromptTokens
bound keeps the compression call itself within the prompt cap.

### charsPerToken (auto: 3)

All size estimates are char-based (no tokenizer dependency). The divisor was
raised from 4 to 3 after real sessions: Cyrillic prose plus JSON wrappers of
tool calls run ~2.4 chars/token, so /4 estimated chunks far below their true
token count and produced overflows. 3 keeps a margin. English-only
conversations can raise it to 4 (larger chunks, fewer calls); CJK-heavy ones
lower it toward 1.5–2.

### What is deliberately NOT configurable

- `ratio = ceil(mainCtx / sumCtx)` — the essence of the method, not a knob.
- The 0.75 factors (safetyBudget and chunkBudget) — sanity margins of the
  budget chain; individually tuning them is a way to break the algorithm.
- Floors 8000/4000/2500/2048/2000 — same reasoning.
- Split depth limits (preemptive 3, length-cap 2) and the single retry —
  resilience bounds; raising them risks runaway loops on a broken model.
- In-memory cache size (400 entries) — memory hygiene, invisible to behavior.
- English summary language — a feature, not a constant; the prompts hardcode
  the EN format. A `language` setting would require prompt surgery and
  re-validation; not worth it until asked for.
- Heuristics +200/+500 chars for assistant/toolResult messages — mirrors the
  stock serializer's truncation (~2000 chars per tool result).

## Behavior contracts

- No summarizer selected (`model: null`) → stock pi compaction, byte-for-byte.
- Model not found in registry → UI warning, stock compaction for this session.
- Every stage surfaces what it does via `ctx.ui.notify` (chunk count, splits,
  ceiling compressions) — compaction should never look like a hang.
- `details.budgets` in the CompactionEntry records the resolved budgets of that
  run, so a post-mortem does not require reproducing the auto formulas.
