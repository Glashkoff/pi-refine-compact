/**
 * Regression tests for the `session` summarizer mode and the compaction cache scope.
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
mock.module("@earendil-works/pi-ai", () => ({ uuidv7: () => `test-uuid-${++uuidCounter}` }));

type Any = any;

const extension = (await import("../extensions/pi-refine-compact.ts")).default;

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

const commands: Record<string, Any> = {};
const hooks: Record<string, Any> = {};
extension({
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

function makeModel(provider: string, id: string, contextWindow = 8000, api = "test-api") {
	return { provider, id, api, contextWindow, maxTokens: 4096 };
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
	const completeCalls: { modelRef: string }[] = [];
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
			complete: async (model: Any) => {
				completeCalls.push({ modelRef: `${model.provider}/${model.id}` });
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
	expect(native.notifications.some((n) => n.message.includes("chunk(s)"))).toBe(false);

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


