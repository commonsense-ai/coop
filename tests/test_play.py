from types import SimpleNamespace

import pytest
import torch

import coop.cli as cli
from coop import play
from coop.data import EOT

RUNS = [
    {"name": "fineweb-150m", "config": "config/run.yaml", "status": "live"},
    {"name": "tinystories-15m", "config": "config/stage1.yaml", "status": "complete"},
]

VOCAB = {0: "hello", 1: " world", 2: " and", 3: "…", 4: EOT}
EOT_ID = 4


class Tok:
    """Stands in for a tokenizers.Tokenizer over the toy vocab above."""

    def encode(self, text):
        return SimpleNamespace(ids=[0])

    def decode(self, ids):
        return "".join(VOCAB[i] for i in ids)

    def token_to_id(self, token):
        return EOT_ID if token == EOT else None


class Scripted:
    """Emits a fixed token sequence, so streaming is deterministic under sampling."""

    def __init__(self, script, block_size=8):
        self.cfg = SimpleNamespace(block_size=block_size)
        self.script = list(script)
        self.widths = []

    def __call__(self, idx):
        self.widths.append(idx.shape[1])
        logits = torch.full((1, idx.shape[1], len(VOCAB)), -30.0)
        logits[0, -1, self.script.pop(0)] = 30.0
        return logits, None


def test_latest_is_the_live_run():
    assert play.pick_run(RUNS, "latest")["name"] == "fineweb-150m"
    assert play.pick_run(RUNS, None)["name"] == "fineweb-150m"


def test_complete_runs_are_playable_unlike_trainable():
    # coop start refuses these; coop run must not — playing is read-only
    assert play.pick_run(RUNS, "tinystories")["name"] == "tinystories-15m"
    with pytest.raises(SystemExit):
        cli.choose_run(RUNS, "tinystories", None, False)


def test_latest_falls_back_when_every_run_is_complete():
    done = [dict(r, status="complete") for r in RUNS]
    assert play.pick_run(done, "latest")["name"] == "fineweb-150m"


def test_unknown_model_names_the_alternatives():
    with pytest.raises(SystemExit, match="fineweb-150m"):
        play.pick_run(RUNS, "gpt5")


def test_run_words_drop_filler():
    assert cli.run_model_from_words([]) is None
    assert cli.run_model_from_words(["the", "latest", "model"]) == "latest"
    with pytest.raises(SystemExit):
        cli.run_model_from_words(["tinystories", "fineweb"])


def test_playing_never_overwrites_the_workers_config_cache(tmp_path, monkeypatch):
    """A worker training another run owns ~/.coop/run.yaml; fetch_raw writes in place."""
    dests = []

    def fake_fetch(repo, path, dest, ref="main"):
        dests.append(dest.name)
        dest.write_text("repos:\n  model: x/y\n")
        return dest

    monkeypatch.setattr(cli, "fetch_raw", fake_fetch)
    monkeypatch.setattr(cli, "HOME", tmp_path)
    cli.load_run_config("commonsense-ai/coop", "config/stage1.yaml", dest="run.play.yaml")
    assert dests == ["run.play.yaml"]


def test_stream_yields_text_as_it_goes():
    model = Scripted([1, 2, 1])
    assert list(play.stream(model, Tok(), "hello", n_tokens=3, top_k=None)) == [
        " world",
        " and",
        " world",
    ]


def test_stream_stops_at_end_of_text():
    model = Scripted([1, EOT_ID, 2])
    assert list(play.stream(model, Tok(), "hello", n_tokens=3, top_k=None)) == [" world"]


def test_stream_never_exceeds_the_context_window():
    model = Scripted([1] * 12, block_size=4)
    list(play.stream(model, Tok(), "hello", n_tokens=12, top_k=None))
    assert max(model.widths) <= 4


def test_stream_holds_back_half_decoded_characters():
    """A char split across tokens must not surface as a replacement byte."""

    class Split(Tok):
        def decode(self, ids):
            return "é" if len(ids) > 1 else ""  # first token decodes to nothing yet

    pieces = list(play.stream(Scripted([1, 2]), Split(), "hello", n_tokens=2, top_k=None))
    assert pieces == ["é"]
