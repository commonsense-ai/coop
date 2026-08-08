"""Submission client: package a pseudo-gradient and open a PR on the inbox repo."""

import json
import logging
import uuid
from pathlib import Path

import torch
from huggingface_hub import CommitOperationAdd
from safetensors.torch import save as st_save

from coop import hubio

log = logging.getLogger(__name__)

SCALE_SUFFIX = "::scale"
NUMEL_SUFFIX = "::numel"

PENDING = "pending"
PENDING_MAX = 8  # ~75MB each at int4; a long outage must not fill a volunteer's disk


def quantize_delta(delta: dict) -> dict:
    """Symmetric per-tensor int8; cuts the upload ~4x. Scales ride along as f32 scalars."""
    out = {}
    for k, v in delta.items():
        scale = max(float(v.abs().max()) / 127.0, 1e-12)
        out[k] = torch.round(v / scale).clamp(-127, 127).to(torch.int8)
        out[k + SCALE_SUFFIX] = torch.tensor(scale, dtype=torch.float32)
    return out


def dequantize_delta(q: dict) -> dict:
    return {
        k: v.float() * q[k + SCALE_SUFFIX].item()
        for k, v in q.items()
        if not k.endswith(SCALE_SUFFIX)
    }


def quantize_delta_int4(delta: dict) -> dict:
    """Symmetric per-tensor int4, two values per byte: ~8x smaller than f32 uploads.
    Shapes are not preserved (values come back 1-D); the aggregator only ever flattens."""
    out = {}
    for k, v in delta.items():
        scale = max(float(v.abs().max()) / 7.0, 1e-12)
        q = (torch.round(v.reshape(-1) / scale).clamp(-7, 7) + 7).to(torch.uint8)  # 0..14
        if q.numel() % 2:
            q = torch.cat([q, torch.full((1,), 7, dtype=torch.uint8)])  # pad encodes 0.0
        out[k] = (q[0::2] << 4) | q[1::2]
        out[k + SCALE_SUFFIX] = torch.tensor(scale, dtype=torch.float32)
        out[k + NUMEL_SUFFIX] = torch.tensor(v.numel(), dtype=torch.int64)
    return out


def dequantize_delta_int4(q: dict) -> dict:
    out = {}
    for k, v in q.items():
        if k.endswith((SCALE_SUFFIX, NUMEL_SUFFIX)):
            continue
        n = int(q[k + NUMEL_SUFFIX].item())
        nibbles = torch.stack([(v >> 4), (v & 0x0F)], dim=1).reshape(-1)[:n]
        out[k] = (nibbles.to(torch.float32) - 7.0) * q[k + SCALE_SUFFIX].item()
    return out


def submission_paths(meta: dict) -> tuple[str, str]:
    base = f"submissions/step_{meta['start_step']}/{meta['username']}_{uuid.uuid4().hex[:8]}"
    return f"{base}.safetensors", f"{base}.json"


class StepAccumulator:
    """Token-weighted running average of a worker's rounds at one outer step, so the
    inbox holds a single growing PR per user instead of one PR per round."""

    def __init__(self):
        self.step = None
        self.delta: dict | None = None
        self.tokens_total = 0
        self.largest = 0
        self.rounds = 0
        self.pr: int | None = None
        self.paths: tuple[str, str] | None = None
        self.pending: Path | None = None

    def start_fresh(self, delta: dict, meta: dict) -> tuple[dict, dict]:
        self.step = meta["start_step"]
        self.delta = {k: v.clone() for k, v in delta.items()}
        self.tokens_total = self.largest = int(meta["tokens"])
        self.rounds = 1
        self.pr = None
        self.paths = None
        # only drop the reference: a parked payload holds rounds this state no longer
        # carries, so it stays on disk for drain() instead of being superseded
        self.pending = None
        return self.delta, self.merged_meta(meta)

    def add(self, delta: dict, meta: dict) -> tuple[dict, dict]:
        if meta["start_step"] != self.step:
            return self.start_fresh(delta, meta)
        t_new, t_old = int(meta["tokens"]), self.tokens_total
        tot = t_old + t_new
        for k in self.delta:
            self.delta[k] = (t_old * self.delta[k] + t_new * delta[k]) / tot
        self.tokens_total = tot
        self.largest = max(self.largest, t_new)
        self.rounds += 1
        return self.delta, self.merged_meta(meta)

    def merged_meta(self, meta: dict) -> dict:
        # credit stays largest-single-round (current server policy); the true totals
        # ride along so a future credit-policy change is a server-side switch
        return {
            **meta,
            "tokens": self.largest,
            "tokens_total": self.tokens_total,
            "rounds": self.rounds,
        }


def _payload(cfg: dict, delta: dict, meta: dict) -> tuple[bytes, bytes]:
    """The exact bytes an upload would send. Parked payloads keep these verbatim so a
    retry never re-quantizes what is already quantized."""
    quant = cfg["inner"].get("quantize")
    if quant == "int8":
        delta, meta["quant"] = quantize_delta(delta), "int8"
    elif quant == "int4":
        delta, meta["quant"] = quantize_delta_int4(delta), "int4"
    else:
        meta["quant"] = "none"
    return st_save(delta), json.dumps(meta, indent=2).encode()


def _ops(paths: tuple[str, str], payload: tuple[bytes, bytes]) -> list:
    return [CommitOperationAdd(p, b) for p, b in zip(paths, payload)]


def _parts(base: Path) -> tuple[Path, Path]:
    return base.with_suffix(".safetensors"), base.with_suffix(".json")


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def discard(base: Path) -> None:
    for p in _parts(base):
        p.unlink(missing_ok=True)


def _trim(d: Path) -> None:
    for js in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime)[:-PENDING_MAX]:
        log.warning("pending queue full — dropping the oldest parked round %s", js.stem)
        discard(js.with_suffix(""))


def pending_count(out_dir) -> int:
    d = Path(out_dir) / PENDING
    return len(list(d.glob("*.json"))) if d.is_dir() else 0


def current_step(cfg: dict) -> int | None:
    """None when the hub is unreachable: nothing gets discarded on a guess."""
    try:
        return hubio.get_step(cfg["repos"]["model"])
    except Exception:
        return None


def expired(meta: dict, step: int | None, tau_max: int) -> bool:
    """A pseudo-gradient is pinned to the step it was trained from, and past tau_max the
    aggregator rejects it on sight. Sending one anyway spends a volunteer's bandwidth and
    a slot in the tick's sweep to be told no."""
    if step is None or not tau_max:
        return False
    return step - int(meta["start_step"]) >= tau_max


def stash(
    out_dir, payload: tuple[bytes, bytes], meta: dict, supersedes: Path | None = None
) -> Path:
    """Park a payload the hub wouldn't take. The round is trained work and only the
    upload failed, so it waits on disk instead of dying with the process."""
    d = Path(out_dir) / PENDING
    d.mkdir(parents=True, exist_ok=True)
    base = d / f"step{meta['start_step']}_{uuid.uuid4().hex[:8]}"
    blob, js = _parts(base)
    _write_atomic(blob, payload[0])
    _write_atomic(js, payload[1])  # the .json lands last: drain() treats it as the receipt
    if supersedes is not None:
        discard(supersedes)
    _trim(d)
    log.warning("upload failed — parked this round at %s; it retries later", blob)
    return base


def drain(cfg: dict, out_dir, skip: Path | None = None) -> int:
    """Re-upload rounds parked by an earlier failure. Each never reached the inbox, so
    each gets its own PR. `skip` is the payload a live accumulator will resend itself."""
    d = Path(out_dir) / PENDING
    if not d.is_dir() or not (queued := sorted(d.glob("*.json"))):
        return 0
    tau = cfg.get("staleness", {}).get("tau_max", 0)
    step = current_step(cfg) if tau else None
    sent = 0
    for js in queued:
        base = js.with_suffix("")
        if skip is not None and base.name == skip.name:
            continue
        blob = _parts(base)[0]
        try:
            payload = (blob.read_bytes(), js.read_bytes())
            meta = json.loads(payload[1])
            paths = submission_paths(meta)  # here, so a malformed meta drops rather than loops
        except (OSError, ValueError, KeyError):
            # unreadable on disk is unrecoverable; drop it rather than wedge the queue
            log.warning("parked round %s can't be read — dropping it", base.name)
            discard(base)
            continue
        if expired(meta, step, tau):
            log.warning(
                "parked round %s trained at step %s, the run is at %d — too stale to be "
                "accepted, dropping it",
                base.name,
                meta["start_step"],
                step,
            )
            discard(base)
            continue
        try:
            info = hubio.open_pr(
                cfg["repos"]["dataset"],
                _ops(paths, payload),
                f"pseudo-gradient from {meta['username']} @ step {meta['start_step']}",
            )
        except Exception as e:
            # still offline: stop here so the queue keeps its order and its disk budget
            log.warning("parked round %s still won't upload (%s); leaving it queued", base.name, e)
            break
        log.info("resent parked round %s: %s", base.name, getattr(info, "pr_url", info))
        discard(base)
        sent += 1
    return sent


def _pr_num(info) -> int | None:
    url = getattr(info, "pr_url", None)
    try:
        return int(str(url).rstrip("/").rsplit("/", 1)[-1])
    except (TypeError, ValueError):
        return None


def submit_accumulated(cfg: dict, acc: StepAccumulator, delta: dict, meta: dict, out_dir=None):
    """Fold this round into the step's running average and create-or-refresh its PR."""
    merged_delta, merged_meta = acc.add(delta, meta)
    repo = cfg["repos"]["dataset"]
    if acc.pr is not None:
        try:
            hubio.update_pr(
                repo,
                acc.pr,
                _ops(acc.paths, _payload(cfg, merged_delta, dict(merged_meta))),
                f"round {merged_meta['rounds']}: {merged_meta['tokens_total']} tokens "
                f"@ step {merged_meta['start_step']}",
            )
            log.info(
                "refreshed PR #%d: %d rounds, %d tokens @ step %d",
                acc.pr,
                merged_meta["rounds"],
                merged_meta["tokens_total"],
                merged_meta["start_step"],
            )
            return None
        except Exception as e:
            # the tick consumed our PR mid-step: earlier rounds were already
            # judged, so restart the accumulator from this round alone
            log.warning("PR #%d gone (%s); resubmitting the newest round", acc.pr, e)
            merged_delta, merged_meta = acc.start_fresh(delta, meta)
    acc.paths = submission_paths(merged_meta)
    payload = _payload(cfg, merged_delta, dict(merged_meta))
    try:
        info = hubio.open_pr(
            repo,
            _ops(acc.paths, payload),
            f"pseudo-gradient from {merged_meta['username']} @ step {merged_meta['start_step']}",
        )
    except Exception:
        if out_dir is not None:
            acc.pending = stash(out_dir, payload, merged_meta, supersedes=acc.pending)
        raise
    acc.pr = _pr_num(info)
    if acc.pending is not None:
        discard(acc.pending)  # this PR carries every round the parked copy held
        acc.pending = None
    log.info("opened %s", getattr(info, "pr_url", info))
    return info


def submit(cfg: dict, delta_path: str, meta: dict, dry_run: bool = False, out_dir=None):
    st_path, js_path = submission_paths(meta)
    repo = cfg["repos"]["dataset"]
    if dry_run:
        log.info("dry-run: would open a PR on %s adding %s and %s", repo, st_path, js_path)
        return None
    js = json.dumps(meta, indent=2).encode()
    ops = [CommitOperationAdd(st_path, str(delta_path)), CommitOperationAdd(js_path, js)]
    try:
        info = hubio.open_pr(
            repo, ops, f"pseudo-gradient from {meta['username']} @ step {meta['start_step']}"
        )
    except Exception:
        # the file on disk is already quantized, so park its bytes as-is
        if out_dir is not None:
            stash(out_dir, (Path(delta_path).read_bytes(), js), meta)
        raise
    log.info("opened %s", getattr(info, "pr_url", info))
    return info
