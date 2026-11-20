"""The tiny experiment of tests/conftest.py as a Feedback Shortcut model."""

import pytest
import torch

from lorm.model import ModelConfig, build_model, initialize

import works.feedback_shortcut  # noqa: F401  (registers its models and algorithms)


@pytest.fixture
def feedback(config):
    """The tiny experiment as a feedback Transformer with two MTP heads, trained with sampled passes."""
    config["model"]["instance"] = {"name": "mtp_feedback", "heads": 2}
    config["algo"] = {"name": "mtp_sampled_passes", "pass_mix": [0.5, 0.5]}
    config["eval"]["variants"] = {"passes": [1, 3]}
    return config


SETTINGS = dict(vocab_size=23, dim=16, n_layers=2, n_heads=2, n_kv_heads=2, hidden_dim=32, max_seq_len=64,
                attention_backend="sdpa", instance={"name": "mtp_feedback", "heads": 2})


def tiny(**instance):
    torch.manual_seed(0)
    model = build_model(ModelConfig.from_dict({**SETTINGS, "instance": {**SETTINGS["instance"], **instance}}))
    initialize(model, std=0.2)  # large weights make the feedback matter
    return model.eval()


@pytest.fixture
def net():
    """A tiny feedback Transformer with two MTP heads and state alignment's map."""
    return tiny(stand_in="state_alignment")


@pytest.fixture
def transformer():
    return tiny(name="mtp")
