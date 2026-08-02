"""Stateless aggregator: one outer step per tick, all state lives in the HF repos."""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import load_file

from coop import hubio, ledger, load_config, robust
from coop.data import load_tokenizer
from coop.eval import eval_val_loss, sample
from coop.model import GPT, GPTConfig, canonical_state, load_canonical_state
from coop.staleness import staleness_weight
from coop.submit import dequantize_delta

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _flatten(delta: dict, keys: list[str]) -> torch.Tensor:
    return torch.cat([delta[k].reshape(-1).float() for k in keys])


def _unflatten(vec: torch.Tensor, keys: list[str], ref: dict) -> dict:
    out, i = {}, 0
    for k in keys:
        n = ref[k].numel()
        out[k] = vec[i : i + n].view(ref[k].shape)
        i += n
    return out


def _load_submission(files: dict[str, str]) -> tuple[dict, dict]:
    st = next((p for f, p in files.items() if f.endswith(".safetensors")), None)
    js = next((p for f, p in files.items() if f.endswith(".json")), None)
    if st is None or js is None:
        raise ValueError("expected one .safetensors and one .json under submissions/")
    meta = json.loads(Path(js).read_text())
    for req in ("username", "start_step", "tokens"):
        if req not in meta:
            raise ValueError(f"meta missing '{req}'")
    delta = load_file(st)
    if meta.get("quant") == "int8":
        delta = dequantize_delta(delta)
    return delta, meta


def _eval_checkpoint(cfg: dict, state: dict, hub) -> dict | None:
    """Val loss + a sample for the new checkpoint. Never allowed to kill the tick."""
    ecfg = cfg.get("eval")
    if not ecfg:
        return None
    try:
        val_path = hub.download_file(cfg["repos"]["model"], ecfg["val_file"])
        model = GPT.from_config(GPTConfig(**cfg["model"]))
        load_canonical_state(model, state)
        loss = eval_val_loss(
            model, val_path, batch=ecfg.get("batch_size", 4), iters=ecfg.get("batches", 8)
        )
        tok = load_tokenizer(cfg["data"]["tokenizer"])
        text = sample(
            model,
            tok,
            ecfg.get("sample_prompt", "Once upon a time"),
            ecfg.get("sample_tokens", 120),
        )
        return {"val_loss": round(loss, 4), "sample": text}
    except Exception as e:
        log.warning("eval skipped: %s", e)
        return None


def run_tick(cfg: dict, hub=hubio, repo_root: str = ".") -> dict | None:
    t0 = time.time()
    model_repo, dataset_repo = cfg["repos"]["model"], cfg["repos"]["dataset"]
    out_cfg = cfg["outer"]

    rev = hub.resolve_revision(model_repo)  # one sha for all reads: no mixed-step state
    state, meta = hub.download_checkpoint(model_repo, revision=rev)
    step = meta["step"]
    keys = list(state.keys())
    momentum = hub.download_optimizer(model_repo, revision=rev)
    if momentum is None:
        momentum = {k: torch.zeros_like(v) for k, v in state.items()}

    prs = sorted(hub.list_open_prs(dataset_repo), key=lambda p: p.num)
    log.info("step %d: %d open PRs", step, len(prs))
    if not prs:
        log.info("nothing to do (%.1fs)", time.time() - t0)
        return None

    base_files = set(hub.list_repo_files(dataset_repo))
    accepted: list[tuple] = []  # (pr, vec, weight, meta)
    rejected: list[tuple] = []  # (pr, meta | None, reason)
    for pr in prs:
        try:
            delta, sub_meta = _load_submission(
                hub.download_pr_files(dataset_repo, pr.num, base_files)
            )
            if set(delta.keys()) != set(keys):
                raise ValueError("delta keys do not match checkpoint")
        except Exception as e:
            rejected.append((pr, None, f"malformed: {e}"))
            continue
        # Identity is the PR's authenticated author, never the submission body:
        # meta.username is self-reported, so it can be flaked ("anonymous") or spoofed.
        author = getattr(pr, "author", None)
        if author and sub_meta["username"] != author:
            log.info(
                "PR #%d: stamping username %r -> author %r", pr.num, sub_meta["username"], author
            )
            sub_meta["username"] = author
        w = staleness_weight(sub_meta["start_step"], step, cfg["staleness"]["tau_max"])
        if w <= 0.0:
            rejected.append(
                (pr, sub_meta, f"stale: from step {sub_meta['start_step']}, now {step}")
            )
            continue
        vec = robust.clip_norm(_flatten(delta, keys), out_cfg["max_norm"])
        accepted.append((pr, vec, w, sub_meta))

    # Same user, same start step: token-weighted merge into one entry. All the
    # signal is used, but one contributor is one vote in the robust layer (no
    # electorate stuffing) and credit counts only the largest single round (no
    # token farming). Clipped vectors stay within max_norm under a convex mix.
    groups: dict[tuple[str, int], list] = {}
    for entry in accepted:
        groups.setdefault((entry[3]["username"], entry[3]["start_step"]), []).append(entry)
    accepted, absorbed = [], []
    for group in groups.values():
        if len(group) == 1:
            accepted.append(group[0])
            continue
        toks = [max(float(e[3].get("tokens", 0)), 1.0) for e in group]
        vec = sum(t * e[1] for t, e in zip(toks, group)) / sum(toks)
        best = max(group, key=lambda e: e[3].get("tokens", 0))
        accepted.append((best[0], vec, best[2], best[3]))
        absorbed += [(e[0], e[3]) for e in group if e[0] is not best[0]]

    if accepted:
        vecs = [v for _, v, _, _ in accepted]
        ws = [w for _, _, w, _ in accepted]
        # Clipping above bounds any single submission's pull on this reference, which is
        # what makes a plain weighted mean safe to gate against.
        ref = sum(w * v for v, w in zip(vecs, ws)) / sum(ws)
        keep = robust.cosine_gate(vecs, ref, out_cfg["min_cos"])
        gated = [entry for entry, k in zip(accepted, keep) if not k]
        accepted = [entry for entry, k in zip(accepted, keep) if k]
        rejected += [(pr, m, "cosine gate: anti-correlated with cohort") for pr, _, _, m in gated]

    advanced = bool(accepted)
    evals, tripped = None, False
    if advanced:
        # Staleness weight scales each survivor: stale work still helps, just less.
        scaled = [v * w for _, v, w, _ in accepted]
        if out_cfg["method"] == "geometric_median":
            agg = robust.geometric_median(scaled)
        else:
            agg = robust.trimmed_mean(scaled, out_cfg["trim_frac"])
        d_agg = _unflatten(agg, keys, state)

        mu, lr = out_cfg["momentum"], out_cfg["lr"]
        for k in keys:
            momentum[k] = mu * momentum[k] + d_agg[k]
            # Nesterov look-ahead. delta = theta_outer - theta_local, so subtracting
            # moves the outer weights toward where the workers went.
            state[k] = state[k] - lr * (mu * momentum[k] + d_agg[k])

        evals = _eval_checkpoint(cfg, state, hub)
        # Circuit breaker: a cohort whose step damages val loss beyond the threshold
        # is discarded wholesale — no upload, no credit, no reputation damage (blame
        # is not attributable within a cohort). Eval failure never blocks (evals None).
        prev_val = (meta.get("eval") or {}).get("val_loss")
        max_reg = out_cfg.get("max_val_regression")
        if None not in (max_reg, prev_val) and evals:
            if evals["val_loss"] > prev_val + max_reg:
                log.warning(
                    "circuit breaker: val loss %.4f vs %.4f (+%.2f allowed) — step discarded",
                    evals["val_loss"],
                    prev_val,
                    max_reg,
                )
                advanced, tripped = False, True
    if advanced:
        new_meta = {
            **meta,
            "step": step + 1,
            "updated": _now(),
            "contributors": sorted({m["username"] for _, _, _, m in accepted}),
        }
        if evals:
            new_meta["eval"] = evals
            log.info("eval @ step %d: val loss %.4f", step + 1, evals["val_loss"])
        else:
            new_meta.pop("eval", None)
        hub.upload_checkpoint(model_repo, state, new_meta, opt_state=momentum)

    led_path = Path(repo_root) / "ledger" / "ledger.json"
    led = ledger.load_ledger(led_path)
    if evals and advanced:
        led["eval"] = {"step": step + 1, **evals}
    ledger.update_ledger(
        led,
        [] if tripped else [m for _, _, _, m in accepted],
        step + 1 if advanced else step,
        rejected=[m for _, m, _ in rejected if m],
    )
    ledger.save_ledger(led, led_path)
    (Path(repo_root) / "LEADERBOARD.md").write_text(ledger.render_leaderboard(led))

    discard_note = (
        "Discarded: this cohort's outer step regressed validation loss past the safety "
        "threshold, so the whole step was skipped. No credit and no penalty — resubmit "
        "against the current checkpoint."
    )
    for pr, _, _, m in accepted:
        comment = (
            discard_note
            if tripped
            else f"Accepted into outer step {step + 1}: +{m['tokens']} tokens for "
            f"{m['username']}. Closed without merging to keep the inbox small."
        )
        hub.merge_or_close_pr(dataset_repo, pr.num, merge=False, comment=comment)
    for pr, _, reason in rejected:
        hub.merge_or_close_pr(dataset_repo, pr.num, merge=False, comment=f"Rejected ({reason}).")
    for pr, m in absorbed:
        comment = (
            discard_note
            if tripped
            else f"Merged: averaged into {m['username']}'s step {m['start_step']} submission "
            "(one vote per contributor per step); credit counts the largest single round."
        )
        hub.merge_or_close_pr(dataset_repo, pr.num, merge=False, comment=comment)

    wall = time.time() - t0
    log.info(
        "tick done in %.1fs: step %d -> %d, %d accepted, %d rejected, %d merged%s",
        wall,
        step,
        step + 1 if advanced else step,
        0 if tripped else len(accepted),
        len(rejected),
        len(absorbed),
        f", {len(accepted)} discarded (circuit breaker)" if tripped else "",
    )
    summary = {
        "step": step + 1 if advanced else step,
        "accepted": 0 if tripped else len(accepted),
        "rejected": len(rejected),
        "merged": len(absorbed),
        "wall_secs": round(wall, 1),
    }
    if tripped:
        summary["discarded"] = len(accepted)
    return summary


def init_checkpoint(cfg: dict, hub=hubio, seed: int = 0) -> None:
    """Genesis: random-init model at step 0, zero outer momentum."""
    torch.manual_seed(seed)
    model = GPT.from_config(GPTConfig(**cfg["model"]))
    state = canonical_state(model)
    hub.ensure_repos(cfg["repos"]["model"], cfg["repos"]["dataset"])
    hub.upload_checkpoint(
        cfg["repos"]["model"],
        state,
        {"step": 0, "created": _now(), "config": cfg["model"]},
        opt_state={k: torch.zeros_like(v) for k, v in state.items()},
    )
    log.info("genesis checkpoint uploaded to %s", cfg["repos"]["model"])


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="run one aggregator tick")
    ap.add_argument("--config", default="config/run.yaml")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--init", action="store_true", help="upload a genesis checkpoint and exit")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.init:
        init_checkpoint(cfg)
    else:
        run_tick(cfg, repo_root=a.repo_root)


if __name__ == "__main__":
    main()
