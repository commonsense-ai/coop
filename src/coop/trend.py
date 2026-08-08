"""Val-loss history and the statistics that answer "is the loss going down?".

One val loss answers nothing: an outer step moves it up as often as down. What a run
has to show is the slope, so the series is kept append-only on the `ledger` branch and
every progress question is read off it.

Tokens are the x axis, not outer steps: a step is however much work happened to show up
that tick, so a per-step curve stretches the quiet ticks and compresses the busy ones.
"""

import json
import math
from pathlib import Path

HISTORY = "history.jsonl"
WINDOW = 20  # trailing points the slope is fitted over
MIN_POINTS = 4  # under this, a slope is noise wearing a trend's clothes
SIGMA = 2.0  # a slope must clear 2 standard errors before it is called a direction
BLOCKS = "▁▂▃▄▅▆▇█"

# Read through these rather than inlining defaults at the eval call site: the fingerprint
# below has to name the measurement that actually ran, or a series splices two metrics.
DEFAULTS = {"batches": 8, "batch_size": 4, "seed": 0}


def fmt_tokens(n: int) -> str:
    """Bars and captions are read at a glance; 37.2M lands and 37,224,448 does not."""
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= scale:
            return f"{n / scale:.1f}{suffix}"
    return str(int(n))


def eval_params(cfg: dict) -> dict:
    e = cfg.get("eval") or {}
    return {
        "val_file": e.get("val_file"),
        "batches": int(e.get("batches", DEFAULTS["batches"])),
        "batch_size": int(e.get("batch_size", DEFAULTS["batch_size"])),
        "seed": int(e.get("seed", DEFAULTS["seed"])),
    }


def spec(cfg: dict) -> str:
    """Fingerprint of what the eval measures. Widening the eval or pointing it at another
    val file redefines the number, and a series that splices two definitions shows a step
    change no amount of training caused — so points carry the spec and only matching ones
    are compared."""
    p = eval_params(cfg)
    return f"{p['val_file']}:{p['batches']}x{p['batch_size']}@{p['seed']}"


def load(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a torn line from a killed tick costs its own point, not the series
    return out


def append(path, entry: dict) -> list[dict]:
    """One line per outer step, append-only: the aggregator is stateless and must never
    rewrite history it did not measure."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # A tick killed mid-write leaves a line with no newline. Close it first, or this
    # point lands inside the torn one and the crash costs two measurements, not one.
    existing = p.read_text() if p.exists() else ""
    lead = "\n" if existing and not existing.endswith("\n") else ""
    with p.open("a") as f:
        f.write(lead + json.dumps(entry, sort_keys=True) + "\n")
    return load(p)


def comparable(history: list[dict], current: str | None = None) -> list[dict]:
    """The trailing run of points measuring the same thing, oldest first."""
    want = current if current is not None else (history[-1].get("spec") if history else None)
    out: list[dict] = []
    for e in reversed(history):
        if e.get("spec") != want:
            break
        out.append(e)
    return list(reversed(out))


def ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares slope and its standard error. Three points minimum: a two-point fit
    is exact, and an exact fit reports zero error for a trend it cannot see."""
    n = len(xs)
    if n < 3:
        return 0.0, math.inf
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return 0.0, math.inf
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    sse = sum((y - slope * x - intercept) ** 2 for x, y in zip(xs, ys))
    return slope, math.sqrt(sse / (n - 2) / sxx)


def step_signs(ys: list[float]) -> tuple[int, int]:
    """(improved, worsened) adjacent-step counts. One spike cannot swing this the way it
    swings a slope, so it is the cross-check on the fit."""
    return (
        sum(1 for a, b in zip(ys, ys[1:]) if b < a),
        sum(1 for a, b in zip(ys, ys[1:]) if b > a),
    )


def ema(ys: list[float], alpha: float = 0.3) -> float:
    v = ys[0]
    for y in ys[1:]:
        v += alpha * (y - v)
    return v


def sparkline(series: list[dict], width: int = 32) -> str:
    """Loss against tokens. Points are binned by token count, so a stretch where little
    work landed reads as a flat stretch of line instead of vanishing into one cell."""
    pts = [(float(e.get("tokens") or 0), float(e["val_loss"])) for e in series]
    lo_x, span = pts[0][0], pts[-1][0] - pts[0][0]
    bins: list[list[float]] = [[] for _ in range(width)]
    for x, y in pts:
        i = width - 1 if span <= 0 else min(width - 1, int((x - lo_x) / span * width))
        bins[i].append(y)
    vals, level = [], None
    for b in bins:
        if b:
            level = sum(b) / len(b)
        if level is not None:  # an empty bin holds the level: no work landed, invent none
            vals.append(level)
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return BLOCKS[0] * len(vals)
    return "".join(BLOCKS[min(7, int((v - lo) / (hi - lo) * 8))] for v in vals)


def summarize(history: list[dict], current: str | None = None, window: int = WINDOW) -> dict | None:
    seg = comparable(history, current)
    if not seg:
        return None
    ys = [float(e["val_loss"]) for e in seg]
    best = min(seg, key=lambda e: float(e["val_loss"]))
    tail = seg[-window:]
    ys_t = [float(e["val_loss"]) for e in tail]
    improved, worsened = step_signs(ys_t)
    out = {
        "n": len(seg),
        "step": seg[-1]["step"],
        "latest": ys[-1],
        "smoothed": round(ema(ys), 4),
        "best": float(best["val_loss"]),
        "best_step": best["step"],
        "since_best": seg[-1]["step"] - best["step"],
        "tokens": seg[-1].get("tokens"),
        "tokens_from": tail[0].get("tokens"),
        "window": len(tail),
        "improved": improved,
        "worsened": worsened,
        "verdict": "early",
    }
    xs = [float(e.get("tokens") or 0) / 1e6 for e in tail]
    out["per"] = "1M tokens"
    if len(set(xs)) < 2:  # no token growth recorded: fall back to the step axis
        xs = [float(e["step"]) for e in tail]
        out["per"] = "step"
    if len(tail) >= MIN_POINTS:
        slope, stderr = ols(xs, ys_t)
        out |= {"slope": slope, "stderr": stderr}
        if math.isfinite(stderr):
            if slope + SIGMA * stderr < 0:
                out["verdict"] = "down"
            elif slope - SIGMA * stderr > 0:
                out["verdict"] = "up"
            else:
                out["verdict"] = "flat"
    # Drawn over the same window the slope is fitted on: the first tick starts at chance
    # loss, and a cliff that stays in frame forever flattens every later step into one row.
    if len(tail) >= 5:
        out["spark"] = sparkline(tail)
    return out


def describe(s: dict | None) -> str:
    """The statistical sentence, shared by the leaderboard and `coop status` so a reader
    of one has read the other."""
    if not s:
        return ""
    rate = f"{s.get('slope', 0):+.3g} per {s['per']} (±{s.get('stderr', 0):.3g})"
    head = {
        "down": f"going down, {rate}",
        "up": f"RISING, {rate} — training is not converging",
        "flat": f"flat, {rate} — too noisy to call yet",
        "early": f"only {s['n']} measurement{'s' if s['n'] != 1 else ''} so far",
    }[s["verdict"]]
    bits = [head]
    moves = s["improved"] + s["worsened"]
    if moves:
        bits.append(f"{s['improved']} of {moves} steps improved it")
    ago = s["since_best"]
    when = "this step" if ago == 0 else f"{ago} step{'s' if ago != 1 else ''} ago"
    bits.append(f"best {s['best']:.4f} at step {s['best_step']} ({when})")
    return " · ".join(bits)


def spark_caption(s: dict) -> str:
    lo, hi = s.get("tokens_from") or 0, s.get("tokens") or 0
    return f"loss against tokens, {fmt_tokens(lo)} → {fmt_tokens(hi)}"


def headline(s: dict | None) -> str:
    """The plain-English form for anyone who did not ask for a slope."""
    if not s:
        return ""
    word = {
        "down": "and falling — the model is learning",
        "up": "and rising — something is wrong, check with the coordinators",
        "flat": "and holding flat for now",
        "early": "— too early to say which way it is heading",
    }[s["verdict"]]
    return f"loss {s['latest']:.3f} {word}"


def chance_loss(vocab: int) -> float:
    """The loss of guessing uniformly over the vocabulary — where every run starts."""
    return math.log(vocab)
