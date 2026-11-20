import copy
import os

import pytest
import torch

from lorm.model import ModelConfig, build_model, initialize
from lorm.train import run
from lorm.train.loss import causal_loss
from lorm.train.algo import PassMix, SampledPasses
from works.feedback_shortcut.model import window_offsets

SETTINGS = dict(vocab_size=29, dim=16, n_layers=2, n_heads=2, n_kv_heads=2, hidden_dim=32, max_seq_len=16,
                attention_backend="sdpa")


def model(**instance):
    torch.manual_seed(0)
    net = build_model(ModelConfig.from_dict({**SETTINGS, "instance": {"name": "mtp_feedback", "heads": 3, **instance}}))
    initialize(net)
    return net


def test_window_offsets_count_up_within_windows_of_at_most_heads():
    torch.manual_seed(0)
    offsets = window_offsets(64, 40, 3, 0.75, "cpu")
    assert (offsets[:, 0] == 0).all() and (offsets[:, 1] == 0).all()
    steps = offsets[:, 2:] - offsets[:, 1:-1]
    assert ((steps == 1) | (offsets[:, 2:] == 0)).all() and offsets.max() == 3
    # Windows end with probability 0.25 at each stand-in, so about 0.75 of positions after a window start read one.
    assert 0.55 < (offsets[:, 2:] > 0).float().mean() < 0.75


def test_mtp_heads_predict_further_tokens_causally():
    net = model()
    tokens = torch.randint(0, 29, (2, 16))
    logits, output = net(tokens, passes=2, mtp=True)
    assert [h.shape[1] for h in output.logits] == [15, 14, 13]
    changed = tokens.clone()
    changed[:, 10:] = (changed[:, 10:] + 1) % 29
    _, other = net(changed, passes=2, mtp=True)
    # Head k at t reads tokens up to t + k.
    assert all(torch.allclose(a[:, :10 - k], b[:, :10 - k]) for k, (a, b) in enumerate(zip(output.logits, other.logits), 1))


def test_in_place_stand_ins_change_only_training_calls_of_two_or_more_passes():
    plain, in_place = model(), model(stand_in="in_place")
    in_place.load_state_dict(plain.state_dict())
    tokens = torch.randint(0, 29, (4, 16))
    with torch.no_grad():
        assert torch.equal(plain(tokens, passes=2), in_place(tokens, passes=2))
    torch.manual_seed(1)
    a, _ = plain(tokens, passes=2, mtp=True)
    torch.manual_seed(1)
    b, _ = in_place(tokens, passes=2, mtp=True)
    assert not torch.allclose(a, b)
    torch.manual_seed(1)
    c, _ = in_place(tokens, passes=1, mtp=True)
    assert torch.equal(c, plain(tokens, passes=1, mtp=True)[0])


def test_in_place_gradient_reaches_the_heads_through_the_next_token_loss():
    net = model(stand_in="in_place")
    tokens = torch.randint(0, 29, (4, 16))
    logits, _ = net(tokens, passes=2, mtp=True)
    causal_loss(logits, tokens).backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in net.mtp_later[-1].projection.parameters())


def test_state_alignment_trains_only_the_map():
    tokens = torch.randint(0, 29, (4, 16))
    aligned = model(stand_in="state_alignment")
    assert torch.equal(aligned.mtp.feedback.linear.weight, torch.eye(16))
    _, output = aligned(tokens, passes=2, mtp=True)
    output.stand_in.backward()
    # Only the map learns from the alignment.
    assert aligned.mtp.feedback.linear.weight.grad is not None and aligned.mtp.projection.weight.grad is None


def test_share_and_window_are_validated():
    with pytest.raises(ValueError, match="share"):
        model(share=0.5)
    with pytest.raises(ValueError, match="stand_in"):
        model(stand_in="zeros")


def test_sampled_passes_draw_the_same_passes_for_a_step():
    a, b = SampledPasses(PassMix([0.31, 0.69]), 5), SampledPasses(PassMix([0.31, 0.69]), 5)
    draws = []
    for step in range(50):
        a.begin_update(step), b.begin_update(step)
        draws.append(a.passes)
        assert a.passes == b.passes
    assert set(draws) == {1, 2}


def test_feedback_training_with_normuon_logs_pass_variants(feedback):
    feedback["optimizer"] |= {"muon": {"momentum": 0.95, "beta2": 0.95},
                              "matrix_optimizer": "lorm.train.optim.normuon:NorMuon"}
    run(feedback)
    metrics = open(os.path.join(feedback["output_dir"], "metrics.jsonl")).read()
    assert "eval/nll@P1" in metrics and "eval/nll@P3" in metrics and "train/passes_mean" in metrics


def test_a_decay_branch_may_turn_on_in_place_substitution(feedback):
    feedback["train"]["branch_points"] = [4]
    run(feedback)
    branch = copy.deepcopy(feedback)
    branch["output_dir"] += "-decay"
    branch["branch_from"] = os.path.join(feedback["output_dir"], "checkpoint-step4.pt")
    branch["model"]["instance"]["stand_in"] = "in_place"
    branch["train"].pop("branch_points")
    assert run(branch)["step"] == 6
    changed = copy.deepcopy(branch)
    changed["output_dir"] += "-lr"
    changed["optimizer"]["lr"] = 0.01
    with pytest.raises(ValueError, match="same experiment settings"):
        run(changed)