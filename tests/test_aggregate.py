import json
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import save_file

from coop.aggregate import _flatten, run_tick
from coop.data import train_tokenizer
from coop.model import GPT, GPTConfig, canonical_state
from coop.robust import clip_norm
from coop.submit import dequantize_delta, quantize_delta

CFG = {
    "repos": {"model": "x/model", "dataset": "x/inbox"},
    "model": {
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 16,
        "block_size": 16,
        "vocab_size": 64,
        "dropout": 0.0,
        "bias": True,
    },
    "outer": {
        "lr": 0.7,
        "momentum": 0.9,
        "method": "trimmed_mean",
        "trim_frac": 0.2,
        "min_cos": 0.0,
        "max_norm": 1.0,
    },
    "staleness": {"tau_max": 4},
}


class FakeHub:
    def __init__(self, state, meta, pr_files, authors=None):
        self.state, self.meta, self.pr_files = state, meta, pr_files
        self.authors = authors or {}
        self.uploaded = None
        self.closed = {}
        self.files = {}

    def download_file(self, repo, filename, **kw):
        return self.files[filename]

    def resolve_revision(self, repo, revision="main", **kw):
        return "main"

    def download_checkpoint(self, repo, revision="main"):
        return {k: v.clone() for k, v in self.state.items()}, dict(self.meta)

    def download_optimizer(self, repo, revision="main"):
        return None

    def list_open_prs(self, repo):
        return [SimpleNamespace(num=n, author=self.authors.get(n)) for n in sorted(self.pr_files)]

    def list_repo_files(self, repo, **kw):
        return []

    def download_pr_files(self, repo, num, base_files=None):
        return self.pr_files[num]

    def upload_checkpoint(self, repo, state, meta, opt_state=None):
        self.uploaded = (state, meta, opt_state)

    def merge_or_close_pr(self, repo, num, merge=False, comment=None):
        self.closed[num] = comment


def write_submission(tmp_path, name, delta, meta):
    st, js = tmp_path / f"{name}.safetensors", tmp_path / f"{name}.json"
    save_file(delta, str(st))
    js.write_text(json.dumps(meta))
    prefix = f"submissions/step_{meta['start_step']}/{name}"
    return {f"{prefix}.safetensors": str(st), f"{prefix}.json": str(js)}


def sub_meta(user, start_step, tier="gpu", quant="none"):
    return {
        "username": user,
        "start_step": start_step,
        "tokens": 1000,
        "tier": tier,
        "quant": quant,
    }


def test_username_stamped_from_pr_author(tmp_path):
    torch.manual_seed(0)
    state = canonical_state(GPT.from_config(GPTConfig(**CFG["model"])))
    direction = {k: 0.01 * torch.randn_like(v) for k, v in state.items()}
    near = {k: v + 0.0005 * torch.randn_like(v) for k, v in direction.items()}
    pr_files = {
        1: write_submission(tmp_path, "spoof_s", direction, sub_meta("alice", 5)),  # claims alice
        2: write_submission(tmp_path, "anon_a", near, sub_meta("anonymous", 5)),  # whoami flaked
    }
    hub = FakeHub(state, {"step": 5}, pr_files, authors={1: "mallory", 2: "miacx"})

    run_tick(CFG, hub=hub, repo_root=str(tmp_path))

    led = json.loads((tmp_path / "ledger" / "ledger.json").read_text())
    # credit follows the authenticated PR author, not the self-reported field
    assert "alice" not in led["contributors"]
    assert "anonymous" not in led["contributors"]
    assert led["contributors"]["mallory"]["tokens"] == 1000
    assert led["contributors"]["miacx"]["tokens"] == 1000
    assert sorted(hub.uploaded[1]["contributors"]) == ["mallory", "miacx"]


def test_tick_end_to_end(tmp_path):
    torch.manual_seed(0)
    state = canonical_state(GPT.from_config(GPTConfig(**CFG["model"])))
    keys = list(state.keys())

    # honest norm ~0.75 each: their weighted sum must dominate the outlier once it is
    # clipped to max_norm=1.0, otherwise the gate reference is meaningless
    direction = {k: 0.01 * torch.randn_like(v) for k, v in state.items()}
    honest = [{k: v + 0.0005 * torch.randn_like(v) for k, v in direction.items()} for _ in range(3)]
    outlier = {k: -50.0 * v for k, v in direction.items()}  # huge and anti-correlated

    pr_files = {
        1: write_submission(tmp_path, "alice_a", honest[0], sub_meta("alice", 5)),
        2: write_submission(
            tmp_path, "bob_b", quantize_delta(honest[1]), sub_meta("bob", 5, "cpu", "int8")
        ),
        3: write_submission(tmp_path, "carol_c", honest[2], sub_meta("carol", 4)),  # tau=1
        4: write_submission(tmp_path, "mallory_m", outlier, sub_meta("mallory", 5)),
        5: write_submission(tmp_path, "sleepy_s", direction, sub_meta("sleepy", 0)),  # tau=5>4
    }
    hub = FakeHub(state, {"step": 5}, pr_files)

    summary = run_tick(CFG, hub=hub, repo_root=str(tmp_path))

    assert summary == {
        "step": 6,
        "accepted": 3,
        "rejected": 2,
        "merged": 0,
        "wall_secs": summary["wall_secs"],
    }

    # outer Nesterov step: m = d_agg (zero momentum), theta -= lr * (mu*m + d_agg)
    clip = CFG["outer"]["max_norm"]
    vecs = [
        clip_norm(_flatten(honest[0], keys), clip),
        clip_norm(_flatten(dequantize_delta(quantize_delta(honest[1])), keys), clip),
        0.75 * clip_norm(_flatten(honest[2], keys), clip),  # carol's staleness weight: 1 - 1/4
    ]
    d_agg = torch.stack(vecs).mean(0)
    new_state, new_meta, new_m = hub.uploaded
    expected = _flatten(state, keys) - 0.7 * 1.9 * d_agg
    assert torch.allclose(_flatten(new_state, keys), expected, atol=1e-5)
    assert torch.allclose(_flatten(new_m, keys), d_agg, atol=1e-6)
    assert new_meta["step"] == 6
    assert new_meta["contributors"] == ["alice", "bob", "carol"]

    led = json.loads((tmp_path / "ledger" / "ledger.json").read_text())
    assert set(led["contributors"]) == {"alice", "bob", "carol", "mallory", "sleepy"}
    assert led["contributors"]["alice"]["tokens"] == 1000
    assert led["contributors"]["mallory"]["reputation"] < 1.0
    assert led["contributors"]["mallory"]["tokens"] == 0
    assert (tmp_path / "LEADERBOARD.md").read_text().startswith("# Leaderboard")

    assert set(hub.closed) == {1, 2, 3, 4, 5}  # inbox fully pruned
    assert "Accepted" in hub.closed[1]
    assert "cosine" in hub.closed[4]
    assert "stale" in hub.closed[5]


def test_tick_runs_eval_when_configured(tmp_path):
    torch.manual_seed(0)
    cfg = {
        **CFG,
        "model": {**CFG["model"], "vocab_size": 512},  # covers all tokenizer ids
        "data": {"tokenizer": str(tmp_path / "tok.json")},
        "eval": {
            "val_file": "val.bin",
            "batches": 2,
            "batch_size": 2,
            "sample_prompt": "once upon",
            "sample_tokens": 4,
        },
    }
    corpus = tmp_path / "c.txt"
    corpus.write_text("once upon a time there was a robot. " * 50)
    train_tokenizer([str(corpus)], 512, str(tmp_path / "tok.json"))
    val = tmp_path / "val.bin"
    np.random.default_rng(0).integers(0, 512, size=500).astype(np.uint16).tofile(val)

    state = canonical_state(GPT.from_config(GPTConfig(**cfg["model"])))
    delta = {k: 0.005 * torch.randn_like(v) for k, v in state.items()}
    pr_files = {1: write_submission(tmp_path, "alice_a", delta, sub_meta("alice", 5))}
    hub = FakeHub(state, {"step": 5}, pr_files)
    hub.files["val.bin"] = str(val)

    run_tick(cfg, hub=hub, repo_root=str(tmp_path))

    _, meta, _ = hub.uploaded
    assert meta["step"] == 6
    assert meta["eval"]["val_loss"] > 0
    assert isinstance(meta["eval"]["sample"], str) and meta["eval"]["sample"]
    board = (tmp_path / "LEADERBOARD.md").read_text()
    assert f"Val loss at step 6: **{meta['eval']['val_loss']}**" in board


def test_probation_downweights_unproven_identities(tmp_path):
    torch.manual_seed(0)
    cfg = {**CFG, "outer": {**CFG["outer"], "probation_weight": 0.0}}
    state = canonical_state(GPT.from_config(GPTConfig(**CFG["model"])))
    keys = list(state.keys())
    d1 = {k: 0.005 * torch.randn_like(v) for k, v in state.items()}
    d2 = {k: v + 0.0005 * torch.randn_like(v) for k, v in d1.items()}  # correlated: passes gate

    (tmp_path / "ledger").mkdir()
    (tmp_path / "ledger" / "ledger.json").write_text(
        json.dumps(
            {
                "step": 5,
                "updated": None,
                "contributors": {
                    "alice": {
                        "first_seen": 1,
                        "submissions": 3,
                        "tokens": 5000,
                        "tier": "gpu",
                        "reputation": 1.0,
                    }
                },
            }
        )
    )
    pr_files = {
        1: write_submission(tmp_path, "alice_a", d1, sub_meta("alice", 5)),
        2: write_submission(tmp_path, "newbie_n", d2, sub_meta("newbie", 5)),
    }
    hub = FakeHub(state, {"step": 5}, pr_files)

    summary = run_tick(cfg, hub=hub, repo_root=str(tmp_path))
    assert summary["accepted"] == 2

    # newbie's vote is scaled to zero influence; alice's carries the step alone
    v1 = clip_norm(_flatten(d1, keys), CFG["outer"]["max_norm"])
    d_agg = torch.stack([v1, torch.zeros_like(v1)]).mean(0)
    new_state, _, _ = hub.uploaded
    expected = _flatten(state, keys) - 0.7 * 1.9 * d_agg
    assert torch.allclose(_flatten(new_state, keys), expected, atol=1e-5)

    # probation scales influence, never credit
    led = json.loads((tmp_path / "ledger" / "ledger.json").read_text())
    assert led["contributors"]["newbie"]["tokens"] == 1000
    assert led["contributors"]["newbie"]["reputation"] > 1.0 - 0.11


def test_circuit_breaker_discards_regressing_step(tmp_path):
    torch.manual_seed(0)
    cfg = {
        **CFG,
        "model": {**CFG["model"], "vocab_size": 512},
        "outer": {**CFG["outer"], "max_val_regression": 0.5},
        "data": {"tokenizer": str(tmp_path / "tok.json")},
        "eval": {"val_file": "val.bin", "batches": 2, "batch_size": 2, "sample_tokens": 4},
    }
    corpus = tmp_path / "c.txt"
    corpus.write_text("once upon a time there was a robot. " * 50)
    train_tokenizer([str(corpus)], 512, str(tmp_path / "tok.json"))
    val = tmp_path / "val.bin"
    np.random.default_rng(0).integers(0, 512, size=500).astype(np.uint16).tofile(val)

    state = canonical_state(GPT.from_config(GPTConfig(**cfg["model"])))
    delta = {k: 0.005 * torch.randn_like(v) for k, v in state.items()}
    pr_files = {1: write_submission(tmp_path, "alice_a", delta, sub_meta("alice", 5))}
    # previous eval is impossibly good, so any real step regresses past the threshold
    hub = FakeHub(state, {"step": 5, "eval": {"val_loss": 0.001}}, pr_files)
    hub.files["val.bin"] = str(val)

    summary = run_tick(cfg, hub=hub, repo_root=str(tmp_path))

    assert summary["step"] == 5 and summary["accepted"] == 0 and summary["discarded"] == 1
    assert hub.uploaded is None  # checkpoint untouched
    assert "Discarded" in hub.closed[1]
    led = json.loads((tmp_path / "ledger" / "ledger.json").read_text())
    assert "alice" not in led["contributors"]  # no credit, no reputation damage
    assert "eval" not in led


def test_same_user_same_step_submissions_merge(tmp_path):
    torch.manual_seed(0)
    state = canonical_state(GPT.from_config(GPTConfig(**CFG["model"])))
    keys = list(state.keys())
    d1 = {k: 0.005 * torch.randn_like(v) for k, v in state.items()}
    d2 = {k: v + 0.0005 * torch.randn_like(v) for k, v in d1.items()}

    m1 = {**sub_meta("alice", 5), "tokens": 1000}
    m2 = {**sub_meta("alice", 5), "tokens": 3000}  # deeper re-run before the tick
    pr_files = {
        1: write_submission(tmp_path, "alice_a", d1, m1),
        2: write_submission(tmp_path, "alice_b", d2, m2),
    }
    hub = FakeHub(state, {"step": 5}, pr_files)
    summary = run_tick(CFG, hub=hub, repo_root=str(tmp_path))

    assert summary["accepted"] == 1 and summary["merged"] == 1 and summary["step"] == 6

    # signal: token-weighted average of both rounds; one vote in the robust layer
    v1, v2 = _flatten(d1, keys), _flatten(d2, keys)
    merged = (1000 * v1 + 3000 * v2) / 4000
    new_state, new_meta, _ = hub.uploaded
    expected = _flatten(state, keys) - 0.7 * 1.9 * merged
    assert torch.allclose(_flatten(new_state, keys), expected, atol=1e-5)

    # credit: the largest single round only, no farming
    assert "Accepted" in hub.closed[2] and "Merged" in hub.closed[1]
    led = json.loads((tmp_path / "ledger" / "ledger.json").read_text())
    assert led["contributors"]["alice"]["submissions"] == 1
    assert led["contributors"]["alice"]["tokens"] == 3000
    assert led["contributors"]["alice"]["reputation"] == 1.0


def test_tick_no_prs_is_noop(tmp_path):
    state = canonical_state(GPT.from_config(GPTConfig(**CFG["model"])))
    hub = FakeHub(state, {"step": 5}, {})
    assert run_tick(CFG, hub=hub, repo_root=str(tmp_path)) is None
    assert hub.uploaded is None


def test_malformed_pr_rejected_without_outer_step(tmp_path):
    state = canonical_state(GPT.from_config(GPTConfig(**CFG["model"])))
    js = tmp_path / "bad.json"
    js.write_text(json.dumps({"username": "eve"}))  # no delta file, missing fields
    hub = FakeHub(state, {"step": 5}, {1: {"submissions/step_5/bad.json": str(js)}})
    summary = run_tick(CFG, hub=hub, repo_root=str(tmp_path))
    assert summary["step"] == 5 and summary["accepted"] == 0 and summary["rejected"] == 1
    assert hub.uploaded is None
    assert "malformed" in hub.closed[1]
