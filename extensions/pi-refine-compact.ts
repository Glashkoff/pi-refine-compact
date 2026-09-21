/**
 * pi-refine-compact — replaces pi's stock summarization (/compact and auto-compaction)
 * with a summarizer model of your choice, using chunked refine summarization.
 *
 * Usage:
 *   /compact-model — pick the summarization model (menu like /model). The
 *   choice is persisted in $PI_CODING_AGENT_DIR/pi-refine-compact-settings.json.
 *   Empty selection = the session's default model (pi's stock behavior).
 *
 * How it works:
 *   - Chunking: ratio = ceil(main_ctx / summarizer_ctx); the history is cut
 *     into at least `ratio` chunks at turn boundaries (a tool result is never
 *     split away from its tool call), with a safety budget cap per chunk.
 *   - Merging: sequential recursive summarization (arXiv:2308.15022v4, Wang et
 *     al., "Recursively Summarizing Enables Long-Term Dialogue Memory"): each
 *     chunk updates a growing summary in a strict format (Goal / Constraints /
 *     Progress / Key Decisions / Next Steps / Critical Context) at every step.
 *     Matches the paper's update_memory recursion: new memory = f(old memory,
 *     next dialogue context), started from a first-chunk memorization call.
 *   - Resilience: a length-stop on a chunk re-splits that chunk at turn
 *     boundaries (up to 2 levels); transient errors retry the call once.
 *   - Summary ceiling: when the accumulated summary exceeds its ceiling
 *     (~35% of the summarizer's context), it is compacted via an intermediate
 *     compression call so chunk + summary always fit into the context window.
 *   - Summary language is always English (small models do better with EN,
 *     fewer tokens); a checkpoint-time header is added; in-memory chunk
 *     caching speeds up retries.
 *
 * Settings (all optional, all with sensible auto values):
 *   model               "provider/id" — summarizer model (also set via /compact-model)
 *   maxPromptTokens     hard cap on one summarizer request, chars-per-token based
 *   maxOutputTokens     maxTokens for summarizer calls
 *   summaryCeilingTokens  when the accumulated summary exceeds this, it is compressed
 *   charsPerToken       chars-per-token estimate for your typical conversation language
 *
 * If the summarization model is unavailable — fails with an error as usual.
 * If the extension is disabled — pi's stock behavior.
 */

import { convertToLlm, serializeConversation } from "@earendil-works/pi-coding-agent";
import { uuidv7 } from "@earendil-works/pi-ai";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// ============================================================================
// Settings (persistent file in the agent dir)
// ============================================================================

function agentDir(): string {
	return process.env.PI_CODING_AGENT_DIR || join(homedir(), ".pi", "agent");
}

function settingsPath(): string {
	return join(agentDir(), "pi-refine-compact-settings.json");
}

interface CompactorSettings {
	model: string | null; // "provider/id" | null = the session's stock model
	// All optional: omitted field = auto (formula), see DEFAULTS / auto*() below.
	maxPromptTokens?: number;
	maxOutputTokens?: number;
	summaryCeilingTokens?: number;
	charsPerToken?: number;
}

/** Hard caps of the auto formulas: an explicitly set value may exceed these,
 * the derived defaults never will. */
const DEFAULTS = {
	/** Auto: min(this, 50% of the summarizer's context window). */
	maxPromptTokens: 32000,
	/** Auto: max(this, 80% of compaction.reserveTokens). */
	maxOutputTokensFloor: 2048,
	/** Auto: 35% of the summarizer's context window. */
	summaryCeilingFraction: 0.35,
	/** Auto chars-per-token (see approxTokensOfChars). */
	charsPerToken: 3,
};

/** Non-negative safe integer, or undefined. */
function optNonNegInt(v: unknown): number | undefined {
	if (v === undefined || v === null) return undefined;
	if (typeof v !== "number" || !Number.isSafeInteger(v) || v < 0) return undefined;
	return v;
}

/**
 * Parse and validate one settings file. Invalid values are reported per field
 * (pi principle: invalid settings are an error on read, not a silent rollback)
 * and fall back to their auto formula.
 */
function parseSettings(raw: unknown, warn: (msg: string) => void): CompactorSettings {
	const out: CompactorSettings = { model: null };
	if (typeof raw !== "object" || raw === null) return out;
	const r = raw as Record<string, unknown>;

	if (typeof r.model === "string" && r.model) out.model = r.model;
	const modelInvalid = r.model !== undefined && r.model !== null && !out.model;
	if (modelInvalid) warn(`model: expected non-empty string — ignoring (using the session's model)`);

	for (const key of ["maxPromptTokens", "maxOutputTokens", "summaryCeilingTokens"] as const) {
		const parsed = optNonNegInt(r[key]);
		if (r[key] !== undefined && parsed === undefined) warn(`${key}: expected a non-negative integer — ignoring (auto)`);
		if (parsed !== undefined) out[key] = parsed;
	}
	const cpt = r.charsPerToken;
	if (cpt !== undefined) {
		if (typeof cpt === "number" && Number.isFinite(cpt) && cpt >= 1) out.charsPerToken = cpt;
		else warn(`charsPerToken: expected a number >= 1 — ignoring (auto: ${DEFAULTS.charsPerToken})`);
	}
	return out;
}

function readSettings(warn: (msg: string) => void): CompactorSettings {
	const p = settingsPath();
	if (!existsSync(p)) return { model: null };
	try {
		return parseSettings(JSON.parse(readFileSync(p, "utf-8")), warn);
	} catch (e) {
		warn(`settings file is unreadable (${e instanceof Error ? e.message : String(e)}) — using defaults`);
		return { model: null };
	}
}

function writeSettings(s: CompactorSettings): void {
	const dir = agentDir();
	if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
	// Persist only meaningful values: model null stays as a marker (session model),
	// auto fields stay absent — the file contains only explicit overrides.
	const out: Record<string, unknown> = { model: s.model };
	for (const key of ["maxPromptTokens", "maxOutputTokens", "summaryCeilingTokens"] as const) {
		if (s[key] !== undefined) out[key] = s[key];
	}
	if (s.charsPerToken !== undefined) out.charsPerToken = s.charsPerToken;
	writeFileSync(settingsPath(), `${JSON.stringify(out, null, 2)}\n`, "utf-8");
}

function parseModelRef(ref: string): { provider: string; id: string } | undefined {
	const slash = ref.indexOf("/");
	if (slash <= 0 || slash >= ref.length - 1) return undefined;
	return { provider: ref.slice(0, slash), id: ref.slice(slash + 1) };
}

// ============================================================================
// Retry infrastructure: in-memory chunk cache (C4), hash, length-cap error (C1)
// ============================================================================

/** In-memory cache of summarization results (lives within the pi process):
 * a compaction retry reuses already-computed chunks instead of re-calling the GPU. */
const chunkCache = new Map<string, string>();
const CHUNK_CACHE_MAX = 400;

/** FNV-1a 32-bit, base36 — compact cache key. */
function fnv1a(s: string): string {
	let h = 0x811c9dc5;
	for (let i = 0; i < s.length; i++) {
		h ^= s.charCodeAt(i);
		h = Math.imul(h, 0x01000193);
	}
	return (h >>> 0).toString(36);
}

function cacheSet(key: string, text: string): void {
	if (chunkCache.size >= CHUNK_CACHE_MAX) chunkCache.clear();
	chunkCache.set(key, text);
}

/** The summarizer's reply hit maxTokens — signal to split the chunk (C1). */
class LengthCapError extends Error {
	constructor(label: string) {
		super(`length cap hit at ${label}`);
		this.name = "LengthCapError";
	}
}

// ============================================================================
// /compact-model command — choose the summarization model
// ============================================================================

export default function piRefineCompactExtension(pi: ExtensionAPI) {
	pi.registerCommand("compact-model", {
		description: "Choose the model used for compaction (/compact summarization)",
		handler: async (_args, ctx) => {
			const current = readSettings((m) => ctx.ui.notify(`pi-refine-compact: ${m}`, "warning"));
			const available = ctx.modelRegistry.getAvailable();
			const sessionLabel = ctx.model ? `${ctx.model.provider}/${ctx.model.id}` : "(no model)";
			const items: string[] = [`(default) session model: ${sessionLabel}`];
			const refs: (string | null)[] = [null];
			for (const m of available) {
				const ref = `${m.provider}/${m.id}`;
				const ctxK = m.contextWindow > 0 ? `${Math.round(m.contextWindow / 1000)}k ctx` : "ctx ?";
				const marker = current.model === ref ? "  ← current summarizer" : "";
				items.push(`${ref}   [${ctxK}]${marker}`);
				refs.push(ref);
			}
			const sel = await ctx.ui.select("Model for compaction summarization", items, {});
			if (sel === undefined) return; // cancelled
			const chosen = refs[items.indexOf(sel)];
			if (chosen === undefined) return;
			writeSettings({ ...current, model: chosen });
			ctx.ui.notify(
				chosen
					? `Summarization will be performed by ${chosen}`
					: "Summarization will use the current session model (stock pi behavior)",
				"info",
			);
		},
	});

	// ==========================================================================
	// Compaction hook: replace the summary generator
	// ==========================================================================

	pi.on("session_before_compact", async (event, ctx) => {
		const settings = readSettings((m) => ctx.ui.notify(`pi-refine-compact: ${m}`, "warning"));
		if (!settings.model) return; // default — pi's stock behavior

		const modelRef = parseModelRef(settings.model);
		const found = modelRef ? ctx.modelRegistry.find(modelRef.provider, modelRef.id) : undefined;
		if (!found) {
			// The model vanished from the registry — notify and fall back to stock (not silently).
			ctx.ui.notify(`pi-refine-compact: model ${settings.model} not found — using stock compaction`, "warning");
			return;
		}
		const model = found;

		const { preparation, customInstructions, signal } = event;
		const { messagesToSummarize, turnPrefixMessages, tokensBefore, firstKeptEntryId, previousSummary } = preparation;

		const allMessages: AgentMessage[] = [...messagesToSummarize, ...turnPrefixMessages];
		if (allMessages.length === 0) return; // nothing to summarize — let stock decide

		// Unknown context window: conservative 32k fallback (32 000 tokens, not 32 KiB)
		// so an undersized local model never gets an oversized chunk.
		const sumCtx = model.contextWindow > 0 ? model.contextWindow : 32000;
		const mainCtx = ctx.model && ctx.model.contextWindow > 0 ? ctx.model.contextWindow : sumCtx;
		// Formula: ratio = ceil(mainCtx / sumCtx); at least `ratio` chunks.
		const ratio = Math.max(1, Math.ceil(mainCtx / sumCtx));
		// Safety headroom for instruction prompts + previous summary + the reply.
		// pi's own compaction.reserveTokens (with modelOverrides applied) comes in
		// through preparation.settings; it describes the summarizer's reply margin here.
		const reserveTokens = preparation.settings?.reserveTokens ?? 16384;
		// The chunk budget is capped too, so a chunk never overflows the summarizer
		// by itself (see maxPromptTokens).
		const safetyBudget = Math.max(8000, Math.min(50000, Math.floor((sumCtx - reserveTokens) * 0.75)));

		// --- Configurable budgets (auto formulas when not set) -----------------
		// Hard cap on a single summarizer request; a chunk whose serialized prompt
		// exceeds it is split preemptively (before the provider replies with
		// exceed_context_size_error). Auto: never above 32k and never above half
		// the summarizer's context, because char-based estimates undershoot real
		// tokenizer counts.
		const maxPromptTokens = settings.maxPromptTokens ?? Math.min(
			DEFAULTS.maxPromptTokens,
			Math.floor(sumCtx * 0.5),
		);
		// maxTokens for summarizer calls. reserveTokens is calibrated for the MAIN
		// model; for a small summarizer the API can reject too-large limits, hence
		// the explicit setting. Auto: 80% of reserveTokens with a 2048 floor —
		// clamped to the headroom the chunk leaves in the summarizer's window
		// (hard ceiling wins when even the floor does not fit).
		const maxTokensHardCeiling = Math.max(512, sumCtx - maxPromptTokens - 1000);
		const maxTokens =
			(settings.maxOutputTokens ??
				(maxTokensHardCeiling < DEFAULTS.maxOutputTokensFloor
					? maxTokensHardCeiling
					: Math.min(maxTokensHardCeiling, Math.max(DEFAULTS.maxOutputTokensFloor, Math.floor(reserveTokens * 0.8)))));
		if (settings.maxOutputTokens !== undefined && settings.maxOutputTokens > maxTokensHardCeiling) {
			ctx.ui.notify(
				`pi-refine-compact: maxOutputTokens (${settings.maxOutputTokens}) exceeds the model's free headroom (~${maxTokensHardCeiling}) — the provider may reject the request`,
				"warning",
			);
		}
		// C2: when the accumulated summary outgrows this ceiling, the checkpoint
		// itself is compressed. Auto: 35% of the summarizer's context (bounded by
		// the prompt cap so chunk + summary always fit).
		const accCeiling = settings.summaryCeilingTokens ?? Math.min(
			Math.floor(sumCtx * DEFAULTS.summaryCeilingFraction),
			Math.floor(maxPromptTokens * 0.5),
		);
		// Chars-per-token estimate for this conversation's language (see
		// approxTokensOfChars). Default 3: tuned for Cyrillic prose + JSON wrappers.
		const charsPerToken = settings.charsPerToken ?? DEFAULTS.charsPerToken;

		const chunks = chunkByTurns(allMessages, ratio, sumCtx, safetyBudget, previousSummary, charsPerToken);
		const totalChars = chunks.reduce((acc, c) => acc + approxChars(c), 0);
		ctx.ui.notify(
			`pi-refine-compact: ${chunks.length} chunk(s), ~${Math.round(totalChars / charsPerToken / 1000)}k tok., model ${settings.model}`,
			"info",
		);

				let acc = previousSummary ?? "";
		const usageParts: UsageLike[] = [];
		let retrySplits = 0;
		let cacheHits = 0;

		const extractText = (response: unknown): string => {
			const content = (response as { content?: Array<{ type: string; text?: string }> }).content ?? [];
			return content
				.filter((c) => c.type === "text" && typeof c.text === "string")
				.map((c) => c.text as string)
				.join("\n")
				.trim();
		};

		/** One LLM call: transient error/empty → 1 retry; length → LengthCapError. */
		async function callSummarizer(promptText: string, label: string): Promise<string> {
			let lastErr = "unknown";
			for (let attempt = 0; attempt < 2; attempt++) {
				if (signal?.aborted) throw new Error("Compaction cancelled");
				const response = await ctx.modelRegistry.complete(
					model,
					{
						systemPrompt: SYSTEM_PROMPT,
						messages: [{ role: "user", content: [{ type: "text", text: promptText }], timestamp: Date.now() }],
					},
					{ maxTokens, signal, cacheRetention: "none", sessionId: uuidv7() },
				);
				const stop = (response as { stopReason?: string }).stopReason;
				if (stop === "aborted") throw new Error("Compaction cancelled");
				if (stop === "error") {
					lastErr = (response as { errorMessage?: string }).errorMessage ?? "LLM error";
					continue;
				}
				if (stop === "length") throw new LengthCapError(label);
				const text = extractText(response);
				if (!text) {
					lastErr = "empty summary";
					continue;
				}
				const u = (response as { usage?: UsageLike }).usage;
				if (u) usageParts.push(u);
				return text;
			}
			throw new Error(`pi-refine-compact: ${label}: ${lastErr}`);
		}

		/** Refine one chunk; on length-cap — split the chunk at turn boundaries (C1). */
		async function refineChunk(chunk: AgentMessage[], rawAcc: string | undefined, depth: number): Promise<string> {
			const accText = rawAcc ?? "";
			const convText = serializeConversation(convertToLlm(chunk));
			const key = `R|${fnv1a(convText)}|${fnv1a(accText)}|${fnv1a(customInstructions ?? "")}`;
			const cached = chunkCache.get(key);
			if (cached !== undefined) {
				cacheHits++;
				return cached;
			}
			let promptText = `<conversation>\n${convText}\n</conversation>\n\n`;
			if (accText) promptText += `<previous-summary>\n${accText}\n</previous-summary>\n\n`;
			promptText += accText ? UPDATE_PROMPT : INITIAL_PROMPT;
			if (customInstructions) promptText += `\n\nAdditional focus: ${customInstructions}`;
			// Preemptive split BEFORE the call: if the serialized prompt is already near
			// the cap — cut the chunk at turn boundaries and refine each half.
			// (LengthCapError = stop:"length", while exceed_context_size_error =
			// stop:"error" — a different path; so the cap must fire here.)
			if (approxTokensOfChars(promptText.length, charsPerToken) > maxPromptTokens && depth < 3) {
				const halves = splitInto2(chunk, charsPerToken);
				if (halves) {
					retrySplits++;
					ctx.ui.notify(`pi-refine-compact: prompt > ${maxPromptTokens} tok. — preemptive split (level ${depth + 1})`, "info");
					let local = accText;
					for (const half of halves) local = await refineChunk(half, local, depth + 1);
					cacheSet(key, local);
					return local;
				}
			}
			try {
				const text = await callSummarizer(promptText, depth === 0 ? "chunk" : `chunk-split-l${depth}`);
				cacheSet(key, text);
				return text;
			} catch (e) {
				if (e instanceof LengthCapError && depth < 2) {
					const halves = splitInto2(chunk, charsPerToken);
					if (halves) {
						retrySplits++;
						ctx.ui.notify(`pi-refine-compact: chunk hit the length cap — split at turn boundaries (level ${depth + 1})`, "info");
						let local = accText;
						for (const half of halves) local = await refineChunk(half, local, depth + 1);
						cacheSet(key, local);
						return local;
					}
				}
				throw e;
			}
		}

		/** C2: the accumulated summary outgrew its ceiling — compact the checkpoint itself. */
		async function compressAccIfNeeded(accText: string): Promise<string> {
			if (!accText) return accText;
			const accTokens = approxTokensOfChars(accText.length, charsPerToken);
			if (accTokens <= accCeiling) return accText;
			const key = `C|${fnv1a(accText)}`;
			const cached = chunkCache.get(key);
			if (cached !== undefined) {
				cacheHits++;
				return cached;
			}
			ctx.ui.notify(`pi-refine-compact: summary exceeded its ceiling (${accTokens} > ${accCeiling} tok.) — intermediate compression`, "info");
			const targetTokens = Math.max(2000, Math.ceil(accTokens * 0.6));
			const promptText = `<checkpoint>\n${accText}\n</checkpoint>\n\n${COMPRESS_PROMPT.replace("__TARGET__", String(targetTokens))}`;
			try {
				const text = await callSummarizer(promptText, "checkpoint-compress");
				cacheSet(key, text);
				return text;
			} catch (e) {
				// Degraded mode: a failed checkpoint compression must not kill the whole
				// compaction — continue with the oversized summary (deterministic failures
				// like a length cap would otherwise repeat on every retry).
				ctx.ui.notify(
					`pi-refine-compact: checkpoint compression failed (${e instanceof Error ? e.message : String(e)}) — continuing with the uncompressed summary`,
					"warning",
				);
				return accText;
			}
		}

		for (const chunk of chunks) {
			if (signal?.aborted) throw new Error("Compaction cancelled");
			acc = await compressAccIfNeeded(acc);
			acc = await refineChunk(chunk, acc, 0);
		}

		if (!acc) throw new Error("pi-refine-compact: no summary produced");

		// --- File ops (same as stock) -----------------------------------------
		const fileOps = preparation.fileOps;
		const modified = new Set([...fileOps.edited, ...fileOps.written]);
		const readFiles = [...fileOps.read].filter((f) => !modified.has(f)).sort();
		const modifiedFiles = [...modified].sort();
		// C5: freshness marker — the summary describes only the compacted part of the
		// history; newer events remain in the kept messages.
		const nowIso = new Date().toISOString();
		const stamp = `${nowIso.slice(0, 10)} ${nowIso.slice(11, 16)} UTC`;
		let summary = `> Checkpoint ${stamp} — summary of the compacted prefix. Newer state may live in the recent kept messages.\n\n${acc}`;
		const sections: string[] = [];
		if (readFiles.length > 0) sections.push(`<read-files>\n${readFiles.join("\n")}\n</read-files>`);
		if (modifiedFiles.length > 0) sections.push(`<modified-files>\n${modifiedFiles.join("\n")}\n</modified-files>`);
		if (sections.length > 0) summary += `\n\n${sections.join("\n\n")}`;

		return {
			compaction: {
				summary,
				firstKeptEntryId,
				tokensBefore,
				usage: combineUsage(usageParts),
				details: {
					readFiles,
					modifiedFiles,
					refineCompactModel: settings.model,
					chunks: chunks.length,
					retrySplits,
					cacheHits,
					budgets: { maxPromptTokens, maxOutputTokens: maxTokens, summaryCeilingTokens: accCeiling, charsPerToken },
				},
			},
		};
	});
}

// ============================================================================
// Chunking: turn boundaries, at least `ratio` chunks, safety budget per chunk
// ============================================================================

/** Rough token estimate from character count: ceil(chars / charsPerToken).
 * charsPerToken defaults to 3: chars/4 undershoots real tokenizer counts on
 * this workload (Cyrillic prose plus JSON wrappers run ~2.4 chars/token), so
 * chunks come out far larger than estimated. For mostly-English prose 4 is
 * closer to the truth; for CJK 1.5–2. Configurable via settings. */
function approxTokensOfChars(chars: number, charsPerToken: number): number {
	return Math.ceil(chars / charsPerToken);
}

function approxChars(messages: AgentMessage[]): number {
	let chars = 0;
	for (const m of messages) {
		const content = (m as { content?: unknown }).content;
		if (typeof content === "string") {
			chars += content.length;
		} else if (Array.isArray(content)) {
			for (const block of content) {
				if (block && typeof block === "object" && "text" in block) chars += String((block as { text?: string }).text ?? "").length;
			}
		}
		if (m.role === "assistant") chars += 200; // thinking/tool calls — heuristic
		if (m.role === "toolResult") chars += 500; // serialize truncates to ~2000 chars
	}
	return chars;
}

/** Split into turns (a turn = user + the following assistant/tool messages). */
function splitTurns(messages: AgentMessage[]): AgentMessage[][] {
	const turns: AgentMessage[][] = [];
	let current: AgentMessage[] = [];
	for (const m of messages) {
		if (m.role === "user" && current.length > 0) {
			turns.push(current);
			current = [];
		}
		current.push(m);
	}
	if (current.length > 0) turns.push(current);
	return turns;
}

/**
 * Safe-cut predicate: a cut BEFORE message m is allowed iff m is not a tool
 * result — a toolResult stays glued to the messages that precede it (its tool
 * call lives in the preceding assistant message). Turn boundaries (before a
 * user message) are always safe. In pi a provider tool result is never a user
 * message, so this covers the flat fallback paths too.
 */
function isSafeCutBefore(messages: AgentMessage[], index: number): boolean {
	const m = messages[index];
	if (!m) return false;
	return m.role !== "toolResult";
}

/**
 * Find the safe cut index closest to `target` (exclusive upper bound of the
 * left part). Scans outward from the target; returns undefined when nothing
 * qualifies (e.g. a single giant turn with toolResults everywhere).
 */
function nearestSafeCut(messages: AgentMessage[], target: number): number | undefined {
	const lo = 1;
	const hi = messages.length - 1;
	if (target < lo || target > hi) return undefined;
	for (let d = 0; d < messages.length; d++) {
		const up = target - d;
		if (up >= lo && isSafeCutBefore(messages, up)) return up;
		const down = target + d;
		if (down !== up && down <= hi && isSafeCutBefore(messages, down)) return down;
	}
	return undefined;
}

/**
 * C1: split a chunk exactly in two at the turn boundary closest to the middle.
 * When there are no turn boundaries (a single giant turn), fall back to the
 * nearest safe cut (never between a tool call and its tool result).
 */
function splitInto2(chunk: AgentMessage[], charsPerToken: number): [AgentMessage[], AgentMessage[]] | undefined {
	const turns = splitTurns(chunk);
	if (turns.length >= 2) {
		// Balance by size: accumulate turns up to half the total size.
		const sizes = turns.map((t) => approxTokensOfChars(approxChars(t), charsPerToken));
		const total = sizes.reduce((a, b) => a + b, 0);
		let accSize = 0;
		let cut = 1;
		for (let i = 0; i < turns.length - 1; i++) {
			accSize += sizes[i];
			if (accSize >= total / 2) {
				cut = i + 1;
				break;
			}
			cut = i + 1; // fallback: last safe boundary
		}
		if (cut <= 0 || cut >= turns.length) return undefined;
		const left = turns.slice(0, cut).flat();
		const right = turns.slice(cut).flat();
		if (left.length === 0 || right.length === 0) return undefined;
		return [left, right];
	}
	// Single-turn chunk: fall back to the nearest safe message-level cut.
	const flat = chunk;
	const cut = nearestSafeCut(flat, Math.floor(flat.length / 2));
	if (cut === undefined) return undefined;
	return [flat.slice(0, cut), flat.slice(cut)];
}

/**
 * Cut the history into chunks:
 * - at least `ratio` chunks (the formula ceil(mainCtx/sumCtx));
 * - never exceed safetyBudget tokens per chunk (the summarizer must fit it);
 * - cut only at turn boundaries (a tool result is never separated from its tool call).
 */
function chunkByTurns(
	messages: AgentMessage[],
	ratio: number,
	sumCtx: number,
	safetyBudget: number,
	previousSummary: string | undefined,
	charsPerToken: number,
): AgentMessage[][] {
	const turns = splitTurns(messages);
	if (turns.length === 0) return [messages];
	const prevTokens = previousSummary ? approxTokensOfChars(previousSummary.length, charsPerToken) : 0;

	// Serialization budget per chunk: sumCtx − instructions − prev-summary, with a safety cap.
	const instrReserve = 2500 + prevTokens;
	const chunkBudget = Math.max(4000, Math.min(safetyBudget, Math.floor((sumCtx - instrReserve) * 0.75)));

	// Estimate turn sizes.
	const turnSizes = turns.map((t) => approxTokensOfChars(approxChars(t), charsPerToken));
	const totalTokens = turnSizes.reduce((a, b) => a + b, 0);

	// Target: spread evenly over `ratio` chunks, but never above chunkBudget.
	const targetChunk = Math.max(Math.ceil(totalTokens / ratio), 1);
	const budget = Math.min(targetChunk, chunkBudget);

	const chunks: AgentMessage[][] = [];
	let current: AgentMessage[] = [];
	let currentTokens = 0;
	for (let i = 0; i < turns.length; i++) {
		const size = turnSizes[i];
		const willExceed = current.length > 0 && currentTokens + size > budget;
		// Avoid too many tiny chunks at the end: if `ratio` chunks are already formed
		// and the remainder is small — keep filling the current chunk.
		if (willExceed) {
			chunks.push(current);
			current = [];
			currentTokens = 0;
		}
		current.push(...turns[i]);
		currentTokens += size;
	}
	if (current.length > 0) chunks.push(current);

	// Guarantee: at least `ratio` chunks. If there are enough turns — regroup them
	// (balance sizes); otherwise — split the largest chunks at user boundaries.
	if (chunks.length < ratio) {
		const balanced = rebalanceToChunks(turns, turnSizes, ratio);
		if (balanced.length > chunks.length) return balanced;
	}
	return chunks;
}

/**
 * Regroup turns into exactly `target` chunks of roughly equal size, preserving
 * order and keeping neighboring turns together (contiguous groups).
 */
function rebalanceToChunks(turns: AgentMessage[][], turnSizes: number[], target: number): AgentMessage[][] {
	if (turns.length < target) {
		// Fewer turns than the required number of chunks — cut within the flat
		// message list, but only at safe cuts (a toolResult is never separated
		// from its tool call).
		const flat = turns.flat();
		if (flat.length < target) return [flat];
		const sizes = flat.map((m) => approxTokensOfChars(approxChars([m]), 3));
		const total = sizes.reduce((a, b) => a + b, 0);
		const chunks: AgentMessage[][] = [];
		let start = 0;
		for (let g = 1; g < target; g++) {
			// Target cut: fraction g/target of the total size, measured from start.
			let seen = 0;
			let goal = start;
			while (goal < flat.length && seen < (total * g) / target) {
				seen += sizes[goal];
				goal++;
			}
			const cut = nearestSafeCut(flat, goal);
			if (cut === undefined || cut <= start || cut >= flat.length) continue; // keep this group merged
			chunks.push(flat.slice(start, cut));
			start = cut;
		}
		chunks.push(flat.slice(start));
		return chunks;
	}
	// Contiguous partition of turns into `target` size-balanced groups.
	const totalSize = turnSizes.reduce((a, b) => a + b, 0);
	const avg = totalSize / target;
	const result: AgentMessage[][] = [];
	let current: AgentMessage[] = [];
	let currentSize = 0;
	for (let i = 0; i < turns.length; i++) {
		const remainingGroups = target - result.length;
		const remainingTurns = turns.length - i;
		// If the remaining turns exactly match the empty groups — one turn per group.
		if (remainingTurns <= remainingGroups) {
			if (current.length > 0) {
				result.push(current);
				current = [];
				currentSize = 0;
			}
			result.push(turns[i]);
			continue;
		}
		current.push(...turns[i]);
		currentSize += turnSizes[i];
		if (result.length < target - 1 && currentSize >= avg) {
			result.push(current);
			current = [];
			currentSize = 0;
		}
	}
	if (current.length > 0) result.push(current);
	return result;
}

// ============================================================================
// Prompts (copies of pi's stock prompts + one system prompt)
// ============================================================================

const SYSTEM_PROMPT = `You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.

IMPORTANT LANGUAGE RULE: write the summary in ENGLISH regardless of the conversation language. Keep code identifiers, file paths, commands, and error messages verbatim. Translate all prose (descriptions, rationale, progress notes) into English.`;

const INITIAL_PROMPT = `The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages.

Write the summary in ENGLISH regardless of the conversation language.`;

const UPDATE_PROMPT = `Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages.

Write the updated summary in ENGLISH regardless of the conversation language.`;

// C2: intermediate compaction of the checkpoint itself (when the accumulated
// summary outgrows its ceiling).
const COMPRESS_PROMPT = `The checkpoint above has grown too large. Compress it into a smaller checkpoint of roughly __TARGET__ tokens.

RULES:
- MERGE duplicate and overlapping items; drop stale items that no longer matter
- KEEP: current goal(s), active constraints, unresolved blockers, exact file paths, function names, error messages, key decisions
- DROP: completed-and-verified details, obsolete attempts, verbose rationale (keep one-line conclusions)
- Do NOT invent new information

Use this EXACT format (same as the original):

## Goal
## Constraints & Preferences
## Progress (### Done / ### In Progress / ### Blocked)
## Key Decisions
## Next Steps
## Critical Context

Write the compressed checkpoint in ENGLISH.`;

// ============================================================================
// Usage aggregation
// ============================================================================

type UsageLike = { input: number; output: number; cacheRead: number; cacheWrite: number; totalTokens: number; cost: { input: number; output: number; cacheRead: number; cacheWrite: number; total: number } };

function combineUsage(parts: UsageLike[]): UsageLike | undefined {
	if (parts.length === 0) return undefined;
	const acc: UsageLike = {
		input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
	for (const p of parts) {
		acc.input += p.input ?? 0;
		acc.output += p.output ?? 0;
		acc.cacheRead += p.cacheRead ?? 0;
		acc.cacheWrite += p.cacheWrite ?? 0;
		acc.totalTokens += p.totalTokens ?? 0;
		acc.cost.input += p.cost?.input ?? 0;
		acc.cost.output += p.cost?.output ?? 0;
		acc.cost.cacheRead += p.cost?.cacheRead ?? 0;
		acc.cost.cacheWrite += p.cost?.cacheWrite ?? 0;
		acc.cost.total += p.cost?.total ?? 0;
	}
	return acc;
}
