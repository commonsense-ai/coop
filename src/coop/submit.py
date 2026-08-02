"""Submission client: package a pseudo-gradient and open a PR on the inbox repo."""

import json
import logging
import uuid

import torch
from huggingface_hub import CommitOperationAdd
from safetensors.torch import save as st_save

from coop import hubio

log = logging.getLogger(__name__)

SCALE_SUFFIX = "::scale"
NUMEL_SUFFIX = "::numel"


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

    def start_fresh(self, delta: dict, meta: dict) -> tuple[dict, dict]:
        self.step = meta["start_step"]
        self.delta = {k: v.clone() for k, v in delta.items()}
        self.tokens_total = self.largest = int(meta["tokens"])
        self.rounds = 1
        self.pr = None
        self.paths = None
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


def _payload_ops(cfg: dict, delta: dict, meta: dict, paths: tuple[str, str]) -> list:
    quant = cfg["inner"].get("quantize")
    if quant == "int8":
        delta, meta["quant"] = quantize_delta(delta), "int8"
    elif quant == "int4":
        delta, meta["quant"] = quantize_delta_int4(delta), "int4"
    else:
        meta["quant"] = "none"
    return [
        CommitOperationAdd(paths[0], st_save(delta)),
        CommitOperationAdd(paths[1], json.dumps(meta, indent=2).encode()),
    ]


def _pr_num(info) -> int | None:
    url = getattr(info, "pr_url", None)
    try:
        return int(str(url).rstrip("/").rsplit("/", 1)[-1])
    except (TypeError, ValueError):
        return None


def submit_accumulated(cfg: dict, acc: StepAccumulator, delta: dict, meta: dict):
    """Fold this round into the step's running average and create-or-refresh its PR."""
    merged_delta, merged_meta = acc.add(delta, meta)
    repo = cfg["repos"]["dataset"]
    if acc.pr is not None:
        try:
            hubio.update_pr(
                repo,
                acc.pr,
                _payload_ops(cfg, merged_delta, dict(merged_meta), acc.paths),
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
    info = hubio.open_pr(
        repo,
        _payload_ops(cfg, merged_delta, dict(merged_meta), acc.paths),
        f"pseudo-gradient from {merged_meta['username']} @ step {merged_meta['start_step']}",
    )
    acc.pr = _pr_num(info)
    log.info("opened %s", getattr(info, "pr_url", info))
    return info


def submit(cfg: dict, delta_path: str, meta: dict, dry_run: bool = False):
    st_path, js_path = submission_paths(meta)
    repo = cfg["repos"]["dataset"]
    if dry_run:
        log.info("dry-run: would open a PR on %s adding %s and %s", repo, st_path, js_path)
        return None
    ops = [
        CommitOperationAdd(st_path, str(delta_path)),
        CommitOperationAdd(js_path, json.dumps(meta, indent=2).encode()),
    ]
    info = hubio.open_pr(
        repo, ops, f"pseudo-gradient from {meta['username']} @ step {meta['start_step']}"
    )
    log.info("opened %s", getattr(info, "pr_url", info))
    return info
