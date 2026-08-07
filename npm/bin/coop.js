#!/usr/bin/env node
// Thin launcher: the real CLI is the Python package `coop-ai`, run via uv.
// Runs under node and bun alike — bun's spawnSync inherits the tty, so the
// interactive first-run setup works through `bunx`.
const { spawnSync } = require("node:child_process");

const uvx = process.platform === "win32" ? "uvx.exe" : "uvx";
const probe = spawnSync(uvx, ["--version"], { stdio: "ignore" });
if (probe.error || probe.status !== 0) {
  console.error(
    "coop needs uv (a fast Python runner). install it first:\n" +
      "  macOS/Linux:  curl -LsSf https://astral.sh/uv/install.sh | sh\n" +
      '  Windows:      powershell -c "irm https://astral.sh/uv/install.ps1 | iex"\n' +
      "then run this command again."
  );
  process.exit(1);
}
// TODO: flip back to "coop-ai" once the PyPI project is approved and published
const source = "git+https://github.com/commonsense-ai/coop";
const run = spawnSync(uvx, ["--from", source, "coop", ...process.argv.slice(2)], {
  stdio: "inherit",
});
process.exit(run.status ?? 1);
