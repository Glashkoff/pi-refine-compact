/**
 * Regression tests for the `session` summarizer mode, the virtual-model guard,
 * and the compaction cache scope.
 *
 * The addon is imported dynamically after `mock.module` stubs the two pi packages it
 * imports at runtime, so the suite runs on a bare Bun install with no dependencies.
 */
import { afterAll, expect, mock, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const AGENT_DIR = mkdtempSync(join(tmpdir(), "pi-refine-compact-test-"));
process.env.PI_CODING_AGENT_DIR = AGENT_DIR;

// ---------------------------------------------------------------------------
// Stubs for the pi packages the addon imports at runtime (type-only imports are
// erased by the transpiler and never resolve).
// ---------------------------------------------------------------------------

let uuidCounter = 0;
mock.module("@earendil-works/pi-coding-agent", () => ({
	convertToLlm: (v: unknown) => v,
	serializeConversation: (v: unknown) => JSON.stringify(v),
}));
mock.module("@earendil-works/pi-tui", () => ({
	Container: class {},
	Text: class {},
	Spacer: class {},
	Input: class {},
	SelectList: class {},
	fuzzyFilter: () => [],
	getKeybindings: () => ({}),
}));
mock.module("@earendil-works/pi-ai", () => ({
	uuidv7: () => `test-uuid-${++uuidCounter}`,
	// Mirrors pi-ai's clampThinkingLevel (same file: models.ts).
	clampThinkingLevel: (model: Any, level: string) => {
		const order = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];
		const supported = !model.reasoning
			? ["off"]
			: order.filter((l) => {
					const mapped = model.thinkingLevelMap?.[l];
					if (mapped === null) return false;
					if (l === "xhigh" || l === "max") return mapped !== undefined;
					return true;
				});
		if (supported.includes(level)) return level;
		const i = order.indexOf(level);
		if (i === -1) return supported[0] ?? "off";
		for (let j = i; j < order.length; j++) if (supported.includes(order[j])) return order[j];
		for (let j = i - 1; j >= 0; j--) if (supported.includes(order[j])) return order[j];
		return supported[0] ?? "off";
	},
}));

type Any = any;

const extension = (await import("../extensions/pi-refine-compact.ts")).default;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

const commands: Record<string, Any> = {};
const hooks: Record<string, Any> = {};
extension({
	registerEntryRenderer: () => {},
	registerCommand: (name: string, def: Any) => {
		commands[name] = def;
	},
	on: (event: string, handler: Any) => {
		hooks[event] = handler;
		return () => {};
	},
});

const command = commands["compact-model"];
const compactHandler = hooks["session_before_compact"];

function makeModel(provider: string, id: string, contextWindow = 8000, api = "test-api", reasoning = true) {
	return { provider, id, api, contextWindow, maxTokens: 4096, reasoning };
}

function okResponse(text = "## Goal\nRefined checkpoint") {
	return { content: [{ type: "text", text }], stopReason: "stop" };
}

interface CtxOptions {
	sessionId: string;
	model?: Any;
	available?: Any[];
}

function makeCtx(opts: CtxOptions) {
	const notifications: { message: string; level: string }[] = [];
	const completeCalls: { modelRef: string; options?: Any }[] = [];
	const ctx: Any = {
		ui: {
			notify: (message: string, level: string) => {
				notifications.push({ message, level });
			},
			select: async () => undefined,
		},
		modelRegistry: {
			getAvailable: () => opts.available ?? [],
			find: (p: string, id: string) =>
				(opts.available ?? []).find((m: Any) => m.provider === p && m.id === id),
			complete: async (model: Any, _context: Any, options: Any) => {
				completeCalls.push({ modelRef: `${model.provider}/${model.id}`, options });
				return okResponse();
			},
		},
		model: opts.model,
		sessionManager: { getSessionId: () => opts.sessionId },
	};
	return { ctx, notifications, completeCalls };
}

function turns(count: number, seed: string): Any[] {
	const out: Any[] = [];
	for (let i = 0; i < count; i++) {
		out.push({ role: "user", content: `${seed} q${i} ${"x".repeat(60)}` });
		out.push({ role: "assistant", content: `${seed} a${i} ${"y".repeat(60)}` });
	}
	return out;
}

function makeEvent(messages: Any[], previousSummary?: string) {
	return {
		preparation: {
			messagesToSummarize: messages,
			turnPrefixMessages: [],
			tokensBefore: 1234,
			firstKeptEntryId: "keep-1",
			previousSummary,
			fileOps: { read: [], edited: [], written: [] },
			settings: { reserveTokens: 1024 },
		},
		customInstructions: undefined,
		signal: undefined,
		reason: "manual",
		willRetry: false,
	};
}

const SETTINGS_FILE = join(AGENT_DIR, "pi-refine-compact-settings.json");

function writeRawSettings(value: unknown): void {
	writeFileSync(SETTINGS_FILE, `${JSON.stringify(value, null, 2)}\n`, "utf-8");
}

function readRawSettings(): Any | undefined {
	return existsSync(SETTINGS_FILE) ? JSON.parse(readFileSync(SETTINGS_FILE, "utf-8")) : undefined;
}

afterAll(() => rmSync(AGENT_DIR, { recursive: true, force: true }));

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

test("registers the compact-model command and the session_before_compact hook", () => {
	expect(typeof command?.handler).toBe("function");
	expect(typeof compactHandler).toBe("function");
});

// ---------------------------------------------------------------------------
// Backwards compatibility
// ---------------------------------------------------------------------------

test("settings.model null defers to stock; an explicit provider/id refines with that model", async () => {
	writeRawSettings({ model: null });
	const native = makeCtx({ sessionId: "s-native", model: makeModel("main", "model") });
	const nativeResult = await compactHandler(makeEvent(turns(3, "native")), native.ctx);
	expect(nativeResult).toBeUndefined();
	expect(native.completeCalls.length).toBe(0);
	expect(native.notifications.some((n) => n.message.includes("chunks ×"))).toBe(false);

	writeRawSettings({ model: "prov/summarizer" });
	const fixed = makeCtx({
		sessionId: "s-fixed",
		model: makeModel("prov", "session-model", 16000),
		available: [makeModel("prov", "summarizer", 8000)],
	});
	const fixedResult = await compactHandler(makeEvent(turns(4, "fixed")), fixed.ctx);
	expect(fixed.completeCalls.length).toBeGreaterThan(0);
	expect(fixed.completeCalls.every((c) => c.modelRef === "prov/summarizer")).toBe(true);
	expect(fixedResult.compaction.details.refineCompactModel).toBe("prov/summarizer");
});

// ---------------------------------------------------------------------------
// Session mode
// ---------------------------------------------------------------------------

test("settings.model 'session' refines with the captured session model", async () => {
	const sessionModel = makeModel("anthropic", "claude-haiku");
	writeRawSettings({ model: "session" });
	const { ctx, completeCalls } = makeCtx({ sessionId: "s-session-a", model: sessionModel });

	const result = await compactHandler(makeEvent(turns(4, "sessA")), ctx);

	expect(completeCalls.length).toBeGreaterThan(0);
	expect(completeCalls.every((c) => c.modelRef === "anthropic/claude-haiku")).toBe(true);
	expect(result.compaction.details.refineCompactModel).toBe("anthropic/claude-haiku");
	expect(result.compaction.summary.startsWith("> Checkpoint ")).toBe(true);
	// The marker is a static setting: it never rewrites the active model or the file.
	expect(ctx.model).toBe(sessionModel);
	expect(readRawSettings()).toEqual({ model: "session" });
});

test("session mode with no chat model falls back to native with one info line", async () => {
	writeRawSettings({ model: "session" });
	const { ctx, notifications, completeCalls } = makeCtx({ sessionId: "s-session-empty" });

	const result = await compactHandler(makeEvent(turns(3, "empty")), ctx);

	expect(result).toBeUndefined();
	expect(completeCalls.length).toBe(0);
	const info = notifications.filter((n) => n.level === "info" && n.message.includes("no model"));
	expect(info.length).toBe(1);
	// No nag: the fallback is the mode's natural empty case, not a warning.
	expect(notifications.some((n) => n.level === "warning")).toBe(false);
});

test("a virtual session model warns and defers to stock compaction", async () => {
	const virtual = makeModel("pi", "router", 200000, "pi-virtual");
	writeRawSettings({ model: "session" });
	const { ctx, notifications, completeCalls } = makeCtx({
		sessionId: "s-session-virtual",
		model: virtual,
	});

	const result = await compactHandler(makeEvent(turns(3, "virtual")), ctx);

	expect(result).toBeUndefined();
	expect(completeCalls.length).toBe(0);
	expect(notifications.some((n) => n.level === "warning" && n.message.includes("pi-virtual") === false && n.message.includes("virtual (router) model"))).toBe(true);
});

test("a virtual model saved as an explicit ref warns and defers to stock too", async () => {
	writeRawSettings({ model: "pi/router" });
	const { ctx, notifications, completeCalls } = makeCtx({
		sessionId: "s-ref-virtual",
		model: makeModel("prov", "main", 16000),
		available: [makeModel("pi", "router", 200000, "pi-virtual")],
	});

	const result = await compactHandler(makeEvent(turns(3, "refvirtual")), ctx);

	expect(result).toBeUndefined();
	expect(completeCalls.length).toBe(0);
	expect(notifications.some((n) => n.level === "warning" && n.message.includes("virtual (router) model"))).toBe(true);
});

// ---------------------------------------------------------------------------
// Cache scope
// ---------------------------------------------------------------------------

test("the chunk cache is isolated by session, resolved model, and budgets", async () => {
	const model = makeModel("prov", "cache-model", 16000);
	const messages = () => turns(6, "cache");

	writeRawSettings({ model: "session" });

	// Same session + model + budgets: the second run reuses the first run's chunks.
	const first = makeCtx({ sessionId: "cache-same", model });
	await compactHandler(makeEvent(messages()), first.ctx);
	expect(first.completeCalls.length).toBeGreaterThan(0);

	const second = makeCtx({ sessionId: "cache-same", model });
	const secondResult = await compactHandler(makeEvent(messages()), second.ctx);
	expect(second.completeCalls.length).toBe(0);
	expect(secondResult.compaction.details.cacheHits).toBeGreaterThan(0);

	// Same content, different session id: recomputed.
	const otherSession = makeCtx({ sessionId: "cache-other", model });
	await compactHandler(makeEvent(messages()), otherSession.ctx);
	expect(otherSession.completeCalls.length).toBeGreaterThan(0);

	// Same session, different resolved summarizer: recomputed.
	const otherModel = makeCtx({ sessionId: "cache-same", model: makeModel("prov", "cache-model-2", 16000) });
	await compactHandler(makeEvent(messages()), otherModel.ctx);
	expect(otherModel.completeCalls.length).toBeGreaterThan(0);

	// Same session and model, different derived budgets: recomputed.
	writeRawSettings({ model: "session", maxPromptTokens: 4096 });
	const otherBudget = makeCtx({ sessionId: "cache-same", model });
	await compactHandler(makeEvent(messages()), otherBudget.ctx);
	expect(otherBudget.completeCalls.length).toBeGreaterThan(0);
});

// ---------------------------------------------------------------------------
// Thinking effort: explicit, validated, budgeted
// ---------------------------------------------------------------------------

test("thinking effort is explicit, uniform per model metadata; unsupported warns", async () => {
	// Muse-spark shape: thinking model whose map excludes "off" (null).
	const model = { ...makeModel("prov", "think-model", 16000), thinkingLevelMap: { off: null } };

	writeRawSettings({ model: "session" });
	const absent = makeCtx({ sessionId: "think-absent", model });
	await compactHandler(makeEvent(turns(4, "absent")), absent.ctx);
	expect(absent.completeCalls.length).toBeGreaterThan(0);
	expect(absent.completeCalls.every((c: Any) => c.options?.reasoning === "low")).toBe(true);
	expect(absent.notifications.some((n) => n.level === "warning" && n.message.includes("reasoning"))).toBe(false);

	writeRawSettings({ model: "session", reasoning: "Medium" });
	const medium = makeCtx({ sessionId: "think-medium", model });
	await compactHandler(makeEvent(turns(4, "medium")), medium.ctx);
	expect(medium.completeCalls.every((c: Any) => c.options?.reasoning === "medium")).toBe(true);

	// "off" is unsupported here: nearest supported is "minimal".
	writeRawSettings({ model: "session", reasoning: "off" });
	const off = makeCtx({ sessionId: "think-off", model });
	await compactHandler(makeEvent(turns(4, "off")), off.ctx);
	expect(off.notifications.some((n) => n.level === "warning" && n.message.includes('using "minimal"'))).toBe(true);
	expect(off.completeCalls.every((c: Any) => c.options?.reasoning === "minimal")).toBe(true);

	// Beyond this model's range: nearest supported downward is "high".
	writeRawSettings({ model: "session", reasoning: "xhigh" });
	const xhigh = makeCtx({ sessionId: "think-xhigh", model });
	await compactHandler(makeEvent(turns(4, "xhigh")), xhigh.ctx);
	expect(xhigh.completeCalls.every((c: Any) => c.options?.reasoning === "high")).toBe(true);

	// Non-thinking model: nearest supported is "off", so the option is omitted.
	const plain = { ...makeModel("prov", "plain-model", 16000), reasoning: false };
	writeRawSettings({ model: "session" });
	const plainCtx = makeCtx({ sessionId: "think-plain", model: plain });
	await compactHandler(makeEvent(turns(4, "plain")), plainCtx.ctx);
	expect(plainCtx.completeCalls.every((c: Any) => !("reasoning" in (c.options ?? {})))).toBe(true);

	writeRawSettings({ model: "session", reasoning: "ultra" });
	const junk = makeCtx({ sessionId: "think-junk", model });
	await compactHandler(makeEvent(turns(4, "junk")), junk.ctx);
	expect(junk.notifications.some((n) => n.level === "warning" && n.message.includes("reasoning"))).toBe(true);
	expect(junk.completeCalls.every((c: Any) => c.options?.reasoning === "low")).toBe(true);
});

test("requested maxTokens covers the visible budget plus thinking headroom", async () => {
	// Small model: 0.8 x reserve(1024) = 819 -> floor 2048 visible;
	// hard ceiling max(512, 8000 - 4000 - 1000) = 3000 wins over 2048 + 8192.
	const small = makeModel("prov", "small-model", 8000);
	small.maxTokens = 4096;
	writeRawSettings({ model: "session" });
	const smallCtx = makeCtx({ sessionId: "think-cap-small", model: small });
	await compactHandler(makeEvent(turns(4, "capsmall")), smallCtx.ctx);
	expect(smallCtx.completeCalls.length).toBeGreaterThan(0);
	expect(smallCtx.completeCalls.every((c: Any) => c.options?.maxTokens === 3000)).toBe(true);

	// Large model: visible 0.8 x 16384 = 13107 plus headroom 8192, under headroom and model caps.
	const big = { provider: "prov", id: "big-model", api: "test-api", contextWindow: 1048576, maxTokens: 131072 };
	const bigEvent = makeEvent(turns(4, "capbig"));
	bigEvent.preparation.settings = { reserveTokens: 16384 };
	writeRawSettings({ model: "session" });
	const bigCtx = makeCtx({ sessionId: "think-cap-big", model: big });
	await compactHandler(bigEvent, bigCtx.ctx);
	expect(bigCtx.completeCalls.length).toBeGreaterThan(0);
	expect(bigCtx.completeCalls.every((c: Any) => c.options?.maxTokens === 13107 + 8192)).toBe(true);
});
