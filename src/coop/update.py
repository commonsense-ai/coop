"""Staying current: release checks and self-update.

The release workflow publishes `release.json` to the `ledger` branch on every tag,
and everyone polls that one small file. It rides `ledger` because workflows cannot
push `main`, and it is a file rather than a registry because volunteers install
straight from git — the tag lands there before PyPI or npm see it.

Applying an update depends on how a copy got onto the machine: a uvx environment is
a cache entry only the resolver may rewrite, a `uv tool` install upgrades in place,
and a git checkout is the volunteer's own working copy — never ours to move.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from coop import __version__, settings

log = logging.getLogger(__name__)

PACKAGE = "coop-ai"
MANIFEST = "release.json"
CHANNEL = "ledger"
RAW = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"
CACHE = "update.json"
TTL = 6 * 3600

GIT, UVX, UV_TOOL, PIP = "git", "uvx", "uv-tool", "pip"


def parse_version(v) -> tuple[int, ...]:
    """ "0.3.1" -> (0, 3, 1). Anything unparseable sorts oldest, so a corrupt manifest
    can never talk a volunteer into an update."""
    out = []
    for chunk in str(v).split(".")[:3]:
        digits = ""
        for c in chunk:
            if not c.isdigit():
                break
            digits += c
        out.append(int(digits) if digits else 0)
    return tuple(out + [0] * (3 - len(out)))


def is_newer(latest, current: str | None = None) -> bool:
    return parse_version(latest) > parse_version(current or __version__)


def cache_path(home) -> Path:
    return Path(home) / CACHE


def read_cache(home) -> dict:
    try:
        return json.loads(cache_path(home).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write_cache(home, **fields) -> dict:
    """Best-effort: a full disk or a read-only home costs a cached check, never a
    round. It is the least important file coop writes."""
    data = read_cache(home) | fields
    try:
        Path(home).mkdir(parents=True, exist_ok=True)
        cache_path(home).write_text(json.dumps(data, indent=2))
    except OSError as e:
        log.debug("could not save the update cache: %s", e)
    return data


def fetch_manifest(repo: str, timeout: float = 5) -> dict:
    url = RAW.format(repo=repo, ref=CHANNEL, path=MANIFEST)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def latest_release(repo: str, home, ttl: float = TTL, force: bool = False) -> dict:
    """Cached view of the release channel. A failed check is not news: a volunteer on
    hotel wifi keeps training on the version they have."""
    cached = read_cache(home)
    if not force and cached.get("manifest") and time.time() - cached.get("checked_at", 0) < ttl:
        return cached["manifest"]
    try:
        manifest = fetch_manifest(repo)
    except Exception as e:
        log.debug("release check failed: %s", e)
        return cached.get("manifest", {})
    write_cache(home, manifest=manifest, checked_at=time.time())
    return manifest


def available(repo: str, home, ttl: float = TTL, force: bool = False) -> dict | None:
    """The release manifest when it names a version newer than this one, else None."""
    manifest = latest_release(repo, home, ttl=ttl, force=force)
    return manifest if is_newer(manifest.get("version", "")) else None


def notice(manifest: dict, auto: bool = False) -> str:
    tail = "auto-update will take it between rounds" if auto else "run `coop update` to get it"
    return f"coop {manifest.get('version', '?')} is out (you're on {__version__}) — {tail}"


def install_kind() -> str:
    """How this copy of coop got here — each way updates differently."""
    root = Path(__file__).resolve().parents[2]  # src/coop/update.py -> checkout root
    # exists(), not is_dir(): in a git worktree `.git` is a file pointing at the real
    # one, and mistaking a checkout for a pip install would pip-upgrade over it
    if (root / ".git").exists() and (root / "pyproject.toml").is_file():
        return GIT
    prefix = Path(sys.prefix).resolve()
    for var, kind in (("UV_TOOL_DIR", UV_TOOL), ("UV_CACHE_DIR", UVX)):
        d = os.environ.get(var)
        if d and prefix.is_relative_to(Path(d).expanduser().resolve()):
            return kind
    if "uv" in prefix.parts:
        # ~/.local/share/uv/tools/coop-ai vs ~/.cache/uv/archive-v0/<hash>
        return UV_TOOL if "tools" in prefix.parts else UVX
    return PIP


def exe(name: str) -> str:
    return shutil.which(name) or name


def source(repo: str) -> str:
    return f"git+https://github.com/{repo}"


def update_argv(repo: str, kind: str | None = None) -> list[str] | None:
    """The command that pulls the new version in, or None when we must not."""
    kind = kind or install_kind()
    src = source(repo)
    if kind == UVX:
        # a uvx env is a cache entry keyed by its requirement; refreshing the package
        # re-resolves git HEAD and rebuilds the entry that every later launch reuses
        return [exe("uvx"), "--refresh-package", PACKAGE, "--from", src, "coop", "--version"]
    if kind == UV_TOOL:
        return [exe("uv"), "tool", "install", "--force", "--refresh-package", PACKAGE, src]
    if kind == PIP:
        return [sys.executable, "-m", "pip", "install", "--upgrade", src]
    return None


def apply(repo: str, kind: str | None = None, timeout: int = 1800) -> tuple[bool, str]:
    """Run the update. Returns (ok, output) and never raises — a volunteer must not
    lose a working install to a failed upgrade."""
    argv = update_argv(repo, kind)
    if argv is None:
        return False, "this is a git checkout of coop — `git pull` updates it"
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"could not run {argv[0]}: {e}"
    return p.returncode == 0, (p.stdout + p.stderr).strip()


def worker_argv(repo: str, args: list[str], kind: str | None = None) -> list[str]:
    """How a running worker relaunches itself into the new version. A uvx worker sits
    in a cache entry it cannot rewrite from the inside, so it re-enters through the
    resolver — which by then serves the entry update_argv() just rebuilt."""
    if (kind or install_kind()) == UVX:
        return [exe("uvx"), "--from", source(repo), "coop-join", *args]
    return [sys.executable, "-m", "coop.join", *args]


def restart_into_update(repo: str, args: list[str], kind: str | None = None) -> str:
    """Update in place, then re-exec the worker into the new code. On success this
    never returns; on failure it returns why, and the caller keeps training.

    The update always lands *before* the exec, never as part of it: execvp is the
    point of no return, and a volunteer whose download dies halfway would otherwise
    be left with no worker at all. For uvx that first command also runs the new
    `coop --version`, so a release too broken to start is caught while we can still
    decline to restart."""
    kind = kind or install_kind()
    if kind == GIT:
        return "this is a git checkout — `git pull` to update it"
    ok, out = apply(repo, kind)
    if not ok:
        return out.splitlines()[-1] if out else "the update command failed"
    argv = worker_argv(repo, args, kind)
    try:
        os.execvp(argv[0], argv)
    except OSError as e:
        return f"could not restart into the new version: {e}"
    return "the new version did not start"  # execvp replaced us if it worked


def auto_enabled(home) -> bool:
    return bool(settings.read(home).get("auto_update"))


def attempted(home, version: str) -> bool:
    """One shot per version: an update that lands but leaves __version__ unchanged
    would otherwise restart the worker forever."""
    return bool(version) and read_cache(home).get("attempted") == version


def mark_attempted(home, version: str) -> None:
    write_cache(home, attempted=version)
