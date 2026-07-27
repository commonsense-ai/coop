import torch

from coop.model import GPT, GPTConfig, canonical_state, count_params, load_canonical_state


def test_stage1_param_count():
    model = GPT.from_config(GPTConfig())
    assert 14_000_000 <= count_params(model) <= 16_000_000


def test_forward_shapes():
    cfg = GPTConfig()
    model = GPT.from_config(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 512))
    logits, loss = model(x)
    assert logits.shape == (2, 512, cfg.vocab_size)
    assert loss is None
    _, loss = model(x, x)
    assert loss is not None and torch.isfinite(loss)


def test_weight_tying_and_canonical_state():
    model = GPT.from_config(GPTConfig(n_layer=1, n_head=2, n_embd=16, vocab_size=64, block_size=8))
    assert model.transformer.wte.weight is model.lm_head.weight
    state = canonical_state(model)
    assert "lm_head.weight" not in state  # deduped by named_parameters
    state["transformer.wte.weight"] += 1.0
    load_canonical_state(model, state)
    assert torch.equal(model.lm_head.weight, state["transformer.wte.weight"])
