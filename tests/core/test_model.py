import pytest
import torch

from lorm.model import ModelConfig, Transformer, build_model, initialize

SETTINGS = dict(vocab_size=31, dim=16, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=32, max_seq_len=12,
                attention_backend="sdpa")


def test_config_round_trips_and_rejects_bad_shapes():
    config = ModelConfig(**SETTINGS)
    assert ModelConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError):
        ModelConfig(**{**SETTINGS, "n_heads": 3})
    with pytest.raises(ValueError, match="instance.name"):
        ModelConfig.from_dict({**SETTINGS, "instance": {"name": "unknown"}})


def test_logits_are_causal():
    torch.manual_seed(0)
    model = Transformer(ModelConfig(**SETTINGS))
    tokens = torch.randint(0, 31, (2, 12))
    changed = tokens.clone()
    changed[:, 7:] = (changed[:, 7:] + 1) % 31
    logits, other = model(tokens), model(changed)
    assert logits.shape == (2, 12, 31)
    assert torch.allclose(logits[:, :7], other[:, :7]) and not torch.allclose(logits[:, 7:], other[:, 7:])


def test_initialize_sets_gains_to_one_and_weights_to_small_normals():
    torch.manual_seed(0)
    model = build_model(ModelConfig(**SETTINGS))
    initialize(model)
    assert all(torch.equal(p, torch.ones_like(p)) for p in model.parameters() if p.ndim == 1)
    assert 0.015 < model.embedding.weight.std().item() < 0.025


def test_tied_embeddings_share_one_matrix():
    model = Transformer(ModelConfig(**{**SETTINGS, "tie_embeddings": True}))
    assert model.head.weight is model.embedding.weight
