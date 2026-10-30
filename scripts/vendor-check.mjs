#!/usr/bin/env node
/**
 * Fails when the vendored Pi helpers drift from the installed `@earendil-works/pi-ai`.
 *
 * The copies under `vendor/pi-ai` are byte-identical on purpose: the adapter reuses Pi's
 * contract logic without importing a module the extension loader does not map. When this
 * check fails, read the upstream change, decide whether it matters here, then copy the new
 * bytes over (see `vendor/pi-ai/README.md`).
 */

import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const upstreamPackage = join(root, "node_modules", "@earendil-works", "pi-ai");

/** Vendored file → path inside the installed package. */
const FILES = [
	["api/transform-messages.js", "dist/api/transform-messages.js"],
	["api/transform-messages.js.map", "dist/api/transform-messages.js.map"],
	["api/transform-messages.d.ts", "dist/api/transform-messages.d.ts"],
	["utils/provider-retry.js", "dist/utils/provider-retry.js"],
	["utils/provider-retry.js.map", "dist/utils/provider-retry.js.map"],
	["utils/provider-retry.d.ts", "dist/utils/provider-retry.d.ts"],
];

const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");

async function version() {
	try {
		const raw = await readFile(join(upstreamPackage, "package.json"), "utf8");
		return JSON.parse(raw).version ?? "unknown";
	} catch {
		return "missing";
	}
}

const upstreamVersion = await version();
if (upstreamVersion === "missing") {
	console.error(`vendor:check — @earendil-works/pi-ai is not installed under ${upstreamPackage}`);
	process.exit(1);
}

const drifted = [];
for (const [vendored, installed] of FILES) {
	const local = join(root, "vendor", "pi-ai", vendored);
	const remote = join(upstreamPackage, installed);
	let localBytes;
	let remoteBytes;
	try {
		localBytes = await readFile(local);
	} catch {
		drifted.push(`${vendored}: missing local copy`);
		continue;
	}
	try {
		remoteBytes = await readFile(remote);
	} catch {
		drifted.push(`${vendored}: no longer shipped by pi-ai@${upstreamVersion} (${installed})`);
		continue;
	}
	if (digest(localBytes) !== digest(remoteBytes)) {
		drifted.push(`${vendored}: differs from pi-ai@${upstreamVersion} ${installed}`);
	}
}

if (drifted.length > 0) {
	console.error(`vendor:check — vendored Pi helpers drifted from pi-ai@${upstreamVersion}:`);
	for (const line of drifted) console.error(`  - ${line}`);
	console.error("\nRead the upstream change, then re-copy the bytes (vendor/pi-ai/README.md).");
	process.exit(1);
}

console.log(`vendor:check — ${FILES.length} files match pi-ai@${upstreamVersion}`);
