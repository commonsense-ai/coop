"""Submission client: package a pseudo-gradient and open a PR on the inbox repo."""

import json
import logging
import uuid

import torch
from huggingface_hub import CommitOperationAdd

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
