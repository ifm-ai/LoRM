import os

import pytest
import torch

from lorm.model import ModelConfig, build_model, initialize
from lorm.train import run
import works.attnres_feedback.model  # noqa: F401

# A tiny norm_eps keeps RMSNorm scale-invariant: the zeros fed to the first position take softmax weight, which the
# norms that read each mix undo only up to norm_eps.
SETTINGS = dict(vocab_size=17, dim=16, n_layers=4, n_heads=2, n_kv_heads=1, hidden_dim=32, max_seq_len=8,
                attention_backend="sdpa", norm_eps=1e-12)


def attnres(**instance):
    """A random model, nonzero queries included."""
    torch.manual_seed(0)
    model = build_model(ModelConfig.from_dict({**SETTINGS, "instance": {"name": "attnres", **instance}})).double()
    initialize(model, std=0.2)
    for name, p in model.named_parameters():
        if name.endswith("query"):
            torch.nn.init.normal_(p, std=1.0)
    return model.eval()


def test_settings_are_checked():
    for bad in ({"feedback": "all"}, {"feedback": "top", "feedback_map": "identity"}, {"passes": 2},
                {"feedback": "blocks", "shift": 2}, {"block_layers": 3}):
        with pytest.raises(ValueError):
            attnres(**bad)


def test_zero_queries_start_every_mix_as_the_mean_of_its_sources():
    model = build_model(ModelConfig.from_dict({**SETTINGS, "instance": {"name": "attnres"}}))
    initialize(model)
    assert all(p.abs().sum() == 0 for n, p in model.named_parameters() if n.endswith("query"))
    values = torch.randn(3, 2, 5, 16)
    torch.testing.assert_close(model.backbone.output_residual(values), values.mean(0))


@pytest.mark.parametrize("instance", [{"feedback": "top"}, {"feedback": "blocks"},
                                      {"feedback": "blocks", "feedback_map": "identity"}])
def test_one_pass_is_the_plain_model_and_passes_converge_to_token_by_token_decoding(instance):
    model, plain = attnres(passes=2, **instance), attnres()
    plain.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("fusion.")})
    tokens = torch.randint(0, 17, (2, 8), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        torch.testing.assert_close(model(tokens, passes=1), plain(tokens), rtol=0, atol=0)
        assert (model(tokens) - plain(tokens)).abs().max() > 1e-3
        exact = torch.cat([model(tokens[:, :t + 1], passes=t + 1)[:, -1:] for t in range(8)], 1)
        torch.testing.assert_close(model(tokens, passes=8), exact, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("shift", [1, 0])
def test_feedback_is_causal(shift):
    model = attnres(feedback="blocks", passes=3, shift=shift)
    tokens = torch.randint(0, 17, (2, 8), generator=torch.Generator().manual_seed(1))
    changed = tokens.clone()
    changed[:, 5] = (changed[:, 5] + 1) % 17
    with torch.no_grad():
        torch.testing.assert_close(model(changed)[:, :5], model(tokens)[:, :5], rtol=0, atol=0)


@pytest.mark.parametrize("full_gradient", [False, True])
def test_earlier_passes_train_only_with_full_gradient(full_gradient):
    model = attnres(feedback="blocks", feedback_map="identity", shift=0, passes=2, full_gradient=full_gradient)
    tokens = torch.randint(0, 17, (2, 8), generator=torch.Generator().manual_seed(1))
    reference = attnres(feedback="blocks", feedback_map="identity", shift=0, passes=2, full_gradient=not full_gradient)
    model(tokens).square().mean().backward()
    reference(tokens).square().mean().backward()
    # The same forward; full gradient adds the earlier pass's contribution.
    assert not torch.allclose(model.embedding.weight.grad, reference.embedding.weight.grad)


def test_flops_count_each_pass_and_each_fed_back_state():
    shape, plain = (1, 8), attnres()
    stack = plain.backbone.forward_macs((1, 8, 16))
    for instance, states in (({"feedback": "top"}, 1), ({"feedback": "blocks"}, 2),
                             ({"feedback": "blocks", "feedback_map": "identity"}, 0)):
        model = attnres(passes=2, **instance)
        assert model.forward_macs(shape, passes=3) == plain.forward_macs(shape) + 2 * (stack + 8 * 256 * states)
        assert model.no_grad_macs(shape, passes=2) == stack + 8 * 256 * states // 2
    loop = attnres(feedback="blocks", feedback_map="identity", shift=0, passes=2, full_gradient=True)
    assert loop.no_grad_macs(shape) == 0 and loop.forward_macs(shape) == plain.forward_macs(shape) + stack


@pytest.mark.parametrize("instance, algo", [({"feedback": "blocks", "passes": 2}, "sampled_passes"),
                                            ({"feedback": "blocks", "feedback_map": "identity", "shift": 0,
                                              "passes": 2, "full_gradient": True}, "standard")])
def test_training_runs(config, instance, algo):
    config["model"]["n_layers"] = 2
    config["model"]["instance"] = {"name": "attnres", **instance}
    config["algo"] = {"name": algo, "pass_mix": [0.5, 0.5]} if algo == "sampled_passes" else {"name": algo}
    run(config)
    assert "eval/nll" in open(os.path.join(config["output_dir"], "metrics.jsonl")).read()
