# pi-refine-compact — design notes

## Overview

The extension is a worked-out *refine* summarizer built on the recursive-summarization method of [arXiv:2308.15022v4](https://arxiv.org/abs/2308.15022) (Wang et al., *Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models*, Neurocomputing 2025; reference implementation [qingyue2014/Rsum](https://github.com/qingyue2014/Rsum)), re-targeted at a different model than the main conversation model.

The paper's core loop is `update_memory`:

```
update_memory(memory, conversation):
    if memory == "":
        memory = summarize(conversation)               # first chunk: memorize
    else:
        memory = summarize_with_previous(memory, conversation)  # later chunks: refine
    return memory
```

Compaction maps onto this loop exactly:

- **chunk 1** → `INITIAL_PROMPT` — memorize the conversation into a structured checkpoint from scratch;
- **chunk k > 1** → `UPDATE_PROMPT` — update the running summary with the new chunk, preserving everything already recorded;
- **the final running summary** → the new session prefix.

Every value below — budgets, floors, split depths — is an engineering adaptation of the paper to pi. Which of them are user-configurable and which are fixed is explained in [Non-configurable constants](#non-configurable-constants) and [Settings](#settings); the runtime guarantees are in [Behavior contracts](#behavior-contracts).

## The algorithm

Every compaction run goes through the following stages.

### 0. Preconditions

The extension hooks pi's `session_before_compact`. As soon as it fires, it reads the settings file and — if no summarizer model is configured — returns immediately, so stock compaction runs byte-for-byte. If a model *is* configured but has vanished from `ctx.modelRegistry`, a warning is shown and stock compaction is used for the session (never a silent switch). Once a model is resolved, `allMessages = messagesToSummarize + turnPrefixMessages` is the full body to compress.

### 1. Budget derivation

Before any chunk is made, the extension derives a chain of token budgets from the chosen summarizer's context window (`sumCtx`; a 32k conservative fallback when the window is unknown) and pi's `compaction.reserveTokens` (the main model's reply margin, reused as the summarizer's reply budget):

```
ratio             = ceil(mainCtx / sumCtx)          # ≥ number of chunks
reserveTokens     = preparation.settings.reserveTokens (default 16384)
safetyBudget      = clamp(sumCtx − reserveTokens, ·0.75)  into [8000, 50000]
maxPromptTokens   = min(32000, 0.5 × sumCtx)        # cap on ONE request
hardCeiling       = max(512, sumCtx − maxPromptTokens − 1000)
maxOutputTokens   = hardCeiling < 2048 ? hardCeiling : min(hardCeiling, max(2048, 0.8 × reserveTokens))
maxPromptTokens   = min(0.5 × sumCtx, 32k; grows with the window for large KNOWN sumCtx)
accCeiling        = min(0.35 × sumCtx, 0.5 × maxPromptTokens, 0.6 × maxTokens)
chunkBudget       ≤ maxPromptTokens − headroom (chunking respects the prompt cap)
charsPerToken     = 3 (Cyrillic+code) / 4 (English) / 1.5–2 (CJK)
```

Each explicit setting overrides its auto formula; an invalid value is reported in the UI and ignored (the formula re-applies) — nothing fails silently. The reasoning behind each value:

- **`maxPromptTokens`** caps a single serialized request. A provider can fail two ways: `exceed_context_size_error` (`stop: "error"`) — treated by the retry as transient and fatal — or `stop: "length"` (the reply hit `maxTokens`) — handled by re-splitting the chunk. Because the first failure is unrecoverable, the cap has to fire *proactively* on the char-based estimate, before the call. That estimate undershoots real tokenizer counts (see [Char-based sizing](#why-english-and-char-based-sizing)), so the auto cap is bounded by half the summarizer's context — the margin for prompt instructions plus previous summary plus reply. The legacy `32000` cap only keeps for unknown/small windows; for large KNOWN windows the auto value grows with the budget (`max(32000, safetyBudget + 8192)`), still bounded by half the window, because a 50k chunk in a 1M context otherwise gets split into two ~25k sequential GPU calls (slow; with reasoning models each call pays extra thinking latency). Chunking itself is bounded by `maxPromptTokens − headroom`, so well-formed chunks never trip an immediate preemptive split.
- **`maxOutputTokens`** is the summarizer's reply budget. `reserveTokens` is calibrated for the main model, so it is clamped to the summarizer's free headroom (`hardCeiling = max(512, sumCtx − maxPromptTokens − 1000)`); the auto value is `hardCeiling < 2048 ? hardCeiling : min(hardCeiling, max(2048, 0.8 × reserveTokens))`. The clamp applies to the auto value only; an explicit `maxOutputTokens` is used as-is (with a warning when it exceeds headroom). A reply that hits the cap is not fatal — the chunk is re-split and re-summarized (LengthCapError → splitInto2).
- **`accCeiling`** bounds the accumulated summary (see [Summary ceiling](#5-summary-ceiling)). An update reply must PRESERVE the whole running summary, so `accCeiling` is additionally bounded by `0.6 × maxTokens`: a running summary near the ceiling must leave room for the new chunk's additions inside the same reply — otherwise `stop: "length"` is guaranteed and unfixable by splitting (the summary does not shrink).
- **`charsPerToken`** is set in [Char-based sizing](#why-english-and-char-based-sizing).

### 2. Chunking at turn boundaries

The history is serialized into chunks under three rules:

- **At least `ratio` chunks** — the formula above guarantees the whole body fits in the summarizer's window even in the worst case;
- **Never above `chunkBudget` per chunk** — `chunkBudget = clamp(safetyBudget, ⌊(sumCtx − (2500 + prevSummaryTokens)) × 0.75⌋, min 4000)`;
- **Only at turn boundaries** — a *turn* is a user message plus the assistant/tool messages that follow it, and a tool result is never separated from the tool call that produced it (`isSafeCutBefore` rejects any cut immediately before a `toolResult`).

Turns are accumulated greedily until the next turn would exceed `budget = min(⌈totalTokens / ratio⌉, chunkBudget)`, then a new chunk starts. If that produces fewer than `ratio` chunks, `rebalanceToChunks` either regroup neighboring turns into size-balanced groups or, when there are simply too few turns, cuts the flat message list at the nearest safe cut.

### 3. The refine loop

```
acc = previousSummary ?? ""   # seed from an existing checkpoint (multi-stage compaction)
for chunk in chunks:
    acc = compressAccIfNeeded(acc)   # see Summary ceiling
    acc = refineChunk(chunk, acc, depth=0)
```

`refineChunk` builds the prompt `<conversation> … </conversation>` plus, when there is a running summary, `<previous-summary> … </previous-summary>`, then appends `INITIAL_PROMPT` (first chunk) or `UPDATE_PROMPT` (rest). It calls the summarizer once (`callSummarizer`) and caches the result in an in-memory FNV-1a-keyed map (≤ 400 entries), so a retry reuses already-computed work instead of re-calling the GPU.

`callSummarizer` issues one request with the shared `SYSTEM_PROMPT` (which forces the English, verbatim-paths format) and applies a pi-style retry policy for transient failures — up to 3 attempts with exponential backoff (base 1 s, cap 60 s), matching pi's default retry settings, because `modelRegistry.complete` bypasses the session-level retry wrapper: `stop: "error"` (e.g. an empty reply) is retried; deterministic provider limits (quota/billing/usage — mirrors of pi's non-retryable family) fail fast; `stop: "length"` means the reply hit `maxTokens` and is turned into a `LengthCapError` to trigger a re-split; `stop: "aborted"` cancels the whole compaction — but only when the compaction signal really is aborted; a provider-side ghost abort (connection dropped mid-stream with the signal still live) is treated as a transient error and retried like `stop: "error"`.

### 4. Resilience: two independent splitters

Two mechanisms keep a chunk inside `maxPromptTokens`:

- **Preemptive split** — *before* the call, if the serialized prompt's char-based estimate exceeds `maxPromptTokens` and `depth < 3`, the chunk is split in two at the size-balanced turn boundary and each half is refined recursively (depth → 3). This must be proactive: a real context overflow comes back as `exceed_context_size_error` (`stop: "error"`), which the retry path treats as transient and aborts — so the cap has to fire on the estimate.
- **Length-cap split** — *after* a reply hits `maxTokens` (`stop: "length"`), the chunk is re-split at turn boundaries and re-summarized (depth → 2).
- **Deep-level summary shrink** — at depth ≥ 2 the split depth is exhausted, and at that point re-splitting would not even help: the *reply* (running summary + the new chunk's distillation) outgrew `maxTokens`, while the running summary itself does not shrink from a split. The accumulated summary is compressed to `0.6 × maxTokens` (`compressAccToBudget`) and the same chunk is retried once with the smaller running summary; a failed shrink re-raises the original error.

Both recombine the halves' summaries left-to-right through the same running `acc`. If a split is impossible (a single giant turn with no safe cut), the chunk is passed through unchanged and the underlying error surfaces.

### 5. Summary ceiling

The running summary grows monotonically across chunks, so it eventually competes with the next chunk for window space. `compressAccIfNeeded` (the `C2` guard) checks the accumulated summary against `accCeiling`; if it overflows, a dedicated `COMPRESS_PROMPT` call compresses the checkpoint to `~60%` of its size (never below 2000 tokens). The `0.35` fraction keeps chunk + summary + instructions inside the window with margin; the `0.5 × maxPromptTokens` bound keeps the compression call itself within the prompt cap; the `0.6 × maxTokens` bound guarantees the following update reply (which must reproduce the whole summary) fits into the reply budget.

The `0.6 × maxTokens` fraction is a deliberate trade-off: it caps how much detail the checkpoint may carry in exchange for the guarantee that an update reply (which must reproduce the whole summary *plus* distill a new chunk) fits into the reply budget. Chunks are still ~up to `chunkBudget` tokens, so the effective distillation ratio can reach 10:1 on large chunks — small models may occasionally still hit the reply cap on the *first* lines of defense; the deep-level shrink then kicks in. If the summarizer model is strong enough to compress further, give it more detail explicitly via `summaryCeilingTokens` (it wins over the formula); the reply budget itself is `maxOutputTokens`.

If *that* compression fails deterministically (e.g. its own length cap), compaction does not abort — it warns and continues with the uncompressed summary (degraded mode), because a deterministic failure would otherwise repeat on every retry. The next refine call may then fail with a context-exceeded error — the same outcome as before compression existed — but transient or one-off compression failures no longer kill a run.

### 6. Output assembly

The checkpoint follows a **fixed six-field structure** — Goal / Constraints / Progress / Key Decisions / Next Steps / Critical Context — so it is easy for another LLM to consume and preserves file paths, commands, and decisions verbatim.

The final summary is wrapped in a **freshness marker** (checkpoint timestamp in `YYYY-MM-DD HH:MM UTC` form) stating it covers only the compacted prefix, then appended with `<read-files>` and `<modified-files>` sections (mirroring pi's stock compaction). The returned `CompactionEntry` also carries a `details.budgets` snapshot of every resolved budget, plus `chunks`, `retrySplits`, and `cacheHits` for a post-mortem.

### Why English and char-based sizing

The `SYSTEM_PROMPT` forces summaries into English regardless of the conversation language (small/cheap models summarize measurably better in EN; code identifiers, file paths, commands, and error messages stay verbatim).

All size estimates are char-based — `approxTokensOfChars = ceil(chars / charsPerToken)` — to stay tokenizer-independent. The default `charsPerToken = 3` was raised from 4 after real sessions: Cyrillic prose plus JSON tool-call wrappers run ~2.4 real chars/token, so `/4` systematically underestimated chunk sizes and caused overflows. English-only conversations can raise it to 4 (larger chunks, fewer calls); CJK-heavy ones lower it toward 1.5–2.

### Non-configurable constants

The following are internal algorithm constants, not settings: `ratio` (the essence of the method), the `0.75` safety factors, the floors (8000/4000/2500/2048/2000), the split-depth limits (preemptive 3, length-cap 2), the deep-level summary shrink (one retry), and the single retry, the 400-entry cache, the English-language rule, and the `+200/+500` char heuristics for assistant/toolResult messages (mirroring the stock serializer's ~2000-char tool-result truncation). Each is explained in the stages above; changing any of them breaks the budget chain.

## Settings

### Configurable (settings file) vs internal (code)

Principles followed (mirroring pi's own settings conventions):

- tokens are non-negative safe integers, names in camelCase with the unit in the name;
- an omitted field means the auto formula applies, and an explicit value wins;
- an invalid value is reported on read and ignored — never a silent fallback to a "best guess" that would differ from the documented auto behavior;
- fractions and one-off heuristics stay internal: they are implementation details of the algorithm, not knobs.

The configurable settings are all derived in [Budget derivation](#1-budget-derivation): `maxPromptTokens`, `maxOutputTokens`, `summaryCeilingTokens` (used as `accCeiling`), and `charsPerToken`. `reserveTokens` is not set by the extension — it is pi's own `compaction.reserveTokens`. The constants that are deliberately left unconfigurable are listed in [Non-configurable constants](#non-configurable-constants).

## Behavior contracts

- **Stock fallback** — if no summarizer is selected (`model: null`) or a configured model is missing from `ctx.modelRegistry`, the extension defers to stock pi compaction. See [Stage 0](#0-preconditions).
- **Observability** — every stage reports what it does via `ctx.ui.notify` (chunk count, splits, ceiling compressions), so compaction never looks like a hang.
- **Post-mortem** — the returned `CompactionEntry` records `details.budgets`, `chunks`, `retrySplits`, `shrinkCount`, and `cacheHits`. See [Stage 6](#6-output-assembly).
