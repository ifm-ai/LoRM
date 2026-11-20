import numpy as np
import pytest
import torch

from works.feedback_shortcut.measure import flops, quality


def documents(count=5, seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([[1, *rng.integers(3, 23, rng.integers(4, 30)), 2] for _ in range(count)]).astype(np.int64)


def test_utf8_bytes_leave_out_each_document_s_prepended_space():
    lengths = np.array([0, 0, 0, 2, 3])
    tokens = np.array([1, 3, 4, 2, 1, 4])
    assert quality.utf8_bytes(lengths, tokens[1:], tokens[:-1]) == (2 + 3 + 3) - 2


def test_document_bpb_does_not_depend_on_the_chunk_length(net, monkeypatch):
    tokens, lengths = documents(), torch.ones(23, dtype=torch.long)
    whole = quality.document_bpb(net, tokens, lengths, passes=40)
    # Chunks overlap by one token, so every token after BOS is still scored once.
    monkeypatch.setattr(quality, "CHUNK", 7)
    chunked = quality.document_bpb(net, tokens, lengths, passes=40)
    assert whole["bytes"] == chunked["bytes"] == len(tokens) - 5 * 2


def test_corrected_context_costs_nothing_where_a_real_state_is_read(net, monkeypatch):
    monkeypatch.setattr(quality, "BURN_IN", 4)
    ids = torch.from_numpy(quality.windows(documents(40), 4, 48))
    nll, _, _, reads, tops = quality.trajectory(net, ids, None, torch.Generator().manual_seed(0))
    corrected = quality.trajectory(net, ids, "mtp_state", torch.Generator().manual_seed(0), (reads, tops))
    real = corrected[1] == 0
    assert torch.allclose(corrected[0][real], nll[real], atol=1e-5) and (~real).any()


def test_stand_in_costs_report_both_contexts_and_each_head(net, monkeypatch):
    monkeypatch.setattr(quality, "BURN_IN", 4)
    ids = torch.from_numpy(quality.windows(documents(40), 4, 48))
    out = quality.stand_in_costs(net, ids, ["zeros", "state_alignment"])
    assert len(out["acceptance"]) == 2 and all(0 <= a <= 1 for a in out["acceptance"])
    assert set(out["zeros"]) == {"kept", "corrected"}


@pytest.mark.parametrize("size, extra", [("135m", {"state-alignment": 0.024, "in-place": 0.080, "in-place-decay": 0.016}),
                                         ("780m", {"state-alignment": 0.020, "in-place": 0.061, "in-place-decay": 0.012})])
def test_extra_training_flops_are_the_paper_s(size, extra):
    from lorm.train.config import load_config
    from lorm.train.main import resolve_config
    value = {s: flops.per_token(resolve_config(load_config(flops.CONFIGS / size / f"{s}.yaml")))
             for s in ("feedback", *extra)}
    for setting, share in extra.items():
        assert round(value[setting] / value["feedback"] - 1, 3) == share, setting