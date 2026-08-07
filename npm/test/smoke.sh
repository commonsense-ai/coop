#!/usr/bin/env sh
# Smoke-tests the published launcher under one runtime: node (npx) or bun (bunx).
# Packs the real tarball so `files` and `bin` are exercised the way a volunteer
# gets them, then checks: both bin names resolve, the argv handed to uvx is
# unchanged, the child's exit status propagates, and a broken uv prints the
# install hint instead of a stack trace.
set -eu

runtime="${1:-}"
case "$runtime" in
  node) install="npm install --no-audit --no-fund --loglevel=error"; run="npx" ;;
  bun) install="bun add"; run="bunx" ;;
  *) echo "usage: smoke.sh node|bun" >&2; exit 2 ;;
esac

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# stub uv: echoes the argv the launcher built, exits 7 so a propagated status is
# distinguishable from anything the runtime would return on its own
mkdir -p "$work/uv-ok" "$work/uv-broken"
cat > "$work/uv-ok/uvx" <<'STUB'
#!/bin/sh
[ "$1" = "--version" ] && { echo "uvx 0.0.0-stub"; exit 0; }
echo "ARGS: $*"
exit 7
STUB
# a uvx that fails its probe, rather than an unset PATH: the runner image may or
# may not ship uv, and this pins the failure branch either way
cat > "$work/uv-broken/uvx" <<'STUB'
#!/bin/sh
exit 1
STUB
chmod +x "$work/uv-ok/uvx" "$work/uv-broken/uvx"

tarball=$(cd "$root" && npm pack --silent --pack-destination "$work")

cd "$work"
echo '{"name":"coop-smoke","private":true}' > package.json
$install "$work/$tarball" >/dev/null

fail() { echo "FAIL $*" >&2; exit 1; }

# both names must be real bin entries: bunx/npx would otherwise fall back to the
# single bin and hide a dropped alias
for name in coop coop-ai; do
  [ -e "node_modules/.bin/$name" ] || fail "$runtime: install did not link node_modules/.bin/$name"
  echo "ok  $runtime      bin $name linked"
done

want="ARGS: --from git+https://github.com/commonsense-ai/coop coop start --rounds 3"
for name in coop coop-ai; do
  got=$(PATH="$work/uv-ok:$PATH" $run "$name" start --rounds 3) && status=0 || status=$?
  [ "$status" = 7 ] || fail "$run $name: exit $status, want 7 (child status not propagated)"
  [ "$got" = "$want" ] || fail "$run $name: uvx got [$got], want [$want]"
  echo "ok  $run $name  argv passthrough + exit status"
done

got=$(PATH="$work/uv-broken:$PATH" $run coop start 2>&1) && status=0 || status=$?
[ "$status" = 1 ] || fail "$run: broken uv exited $status, want 1"
case "$got" in
  *"coop needs uv"*) ;;
  *) fail "$run: broken uv printed [$got], want the uv install hint" ;;
esac
echo "ok  $run       broken uv -> install hint, exit 1"
