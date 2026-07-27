"""Submission client: package a pseudo-gradient and open a PR on the inbox repo."""

import json
import logging
import uuid

import torch
from huggingface_hub import CommitOperationAdd

from coop import hubio

log = logging.getLogger(__name__)

SCALE_SUFFIX = "::scale"


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
