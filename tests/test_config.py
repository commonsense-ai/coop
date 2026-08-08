from pathlib import Path

import torch

from coop import load_config
from coop.model import GPT, GPTConfig

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(str(ROOT / "config" / "run.yaml"))


def _param_count(cfg: dict) -> int:
    # meta device: shapes only, so counting a 146M-param model costs nothing
    with torch.device("meta"):
        model = GPT.from_config(GPTConfig(**cfg["model"]))
    return sum(p.numel() for p in model.parameters())


def test_outer_clip_stays_above_an_honest_round():
    """max_norm is a Byzantine outlier guard, so it has to sit ABOVE what an honest
    worker submits — below it, it silently becomes the step-size knob instead.

    It was 10.0 from the 14.8M-param bootstrap and was copied onto the 146M stage-2
    model unchanged. An absolute L2 over the flattened delta scales with sqrt(params),
    so the same number got 3.1x tighter: every submission arrived clipped ~7x, work
    stopped buying influence (a 40-step round and a 500-step round both landed at
    norm 10), and val loss flattened at 6.07 while contributed tokens went 58M -> 331M.
    """
    n = _param_count(CFG)
    inner = CFG["inner"]
    # AdamW moves ~lr per step and the steps partly cancel, so a full round drifts
    # ~lr*sqrt(h) per parameter; over n parameters that is sqrt(n)*lr*sqrt(h) in L2.
    # Measured against real submissions: 81 predicted vs 63-141 observed at h_max=500.
    honest = n**0.5 * inner["lr"] * inner["h_max"] ** 0.5
    assert CFG["outer"]["max_norm"] >= honest, (
        f"max_norm={CFG['outer']['max_norm']} clips an honest {inner['h_max']}-step "
        f"round (expected L2 ~{honest:.0f} over {n:,} params)"
    )
