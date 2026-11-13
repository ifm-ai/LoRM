import os

import torch

from lorm.model import ModelConfig, Transformer, build_model, initialize
from lorm.train import run

SETTINGS = dict(vocab_size=17, dim=16, n_layers=2, n_heads=2, n_kv_heads=1, hidden_dim=32, max_seq_len=8,
                attention_backend="sdpa")


def models():
    torch.manual_seed(0)
    feedback = build_model(ModelConfig.from_dict({**SETTINGS, "instance": {"name": "feedback"}}))
    initialize(feedback, std=0.2)
    plain = Transformer(ModelConfig.from_dict(SETTINGS))
    plain.load_state_dict({k: v for k, v in feedback.state_dict().items() if not k.startswith("fusion.")})
    return feedback.eval(), plain.eval()


def test_one_pass_is_the_transformer_and_more_passes_read_feedback():
    feedback, plain = models()
    tokens = torch.randint(0, 17, (2, 8))
    assert torch.equal(feedback(tokens, passes=1), plain(tokens))
    two = feedback(tokens, passes=2)
    # The first position reads no feedback.
    assert torch.equal(two[:, 0], plain(tokens)[:, 0]) and not torch.allclose(two[:, 1:], plain(tokens)[:, 1:])


def test_passes_converge_to_token_by_token_decoding():
    feedback, _ = models()
    tokens = torch.randint(0, 17, (1, 8))
    with torch.no_grad():
        exact = torch.cat([feedback(tokens[:, :t + 1], passes=t + 1)[:, -1:] for t in range(8)], 1)
        # After p passes the first p positions are exact.
        assert torch.allclose(feedback(tokens, passes=8), exact, atol=1e-5)


def test_each_pass_after_the_first_adds_the_blocks_and_the_fusion():
    feedback, plain = models()
    shape = (1, 8)
    blocks = feedback.backbone.forward_macs((1, 8, 16))
    assert feedback.forward_macs(shape, passes=1) == plain.forward_macs(shape)
    assert feedback.forward_macs(shape, passes=3) == plain.forward_macs(shape) + 2 * (blocks + 8 * 16 * 16)


def test_sampled_passes_train_the_feedback_model(config):
    config["model"]["instance"] = {"name": "feedback"}
    config["algo"] = {"name": "sampled_passes", "pass_mix": [0.5, 0.5]}
    config["eval"]["variants"] = {"passes": [1, 4]}
    run(config)
    metrics = open(os.path.join(config["output_dir"], "metrics.jsonl")).read()
    assert "train/passes_mean" in metrics and "eval/nll@P1" in metrics and "eval/nll@P4" in metrics


def test_eval_scripts_import_first():
    import subprocess
    import sys
    from pathlib import Path

    subprocess.run([sys.executable, "-c", "import lorm.eval.shards"], check=True, cwd=Path(__file__).parents[2])
