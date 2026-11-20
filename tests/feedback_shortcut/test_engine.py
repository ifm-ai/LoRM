import pytest
import torch

from works.feedback_shortcut.measure.engine import Engine, kl, speculative

DEPTHS = (None, 0, 1, 2)


def prompt():
    return torch.randint(0, 23, (12,), generator=torch.Generator().manual_seed(1)).tolist()


@torch.no_grad()
def exact(model, prompt, new):
    """Exact decoding by its definition (as many passes as positions reach token-by-token decoding's fixed point):
    the greedy continuation and the log-probability rows that score it."""
    tokens, rows = list(prompt), []
    for _ in range(new):
        logits = model(torch.tensor([tokens]), passes=len(tokens)) if hasattr(model, "fusion") else model(torch.tensor([tokens]))
        rows.append(logits[0, -1].float().log_softmax(-1))
        tokens.append(int(rows[-1].argmax()))
    return tokens[len(prompt):], torch.stack(rows)


@pytest.mark.parametrize("passes", [None, 12])
def test_exact_engine_decodes_token_by_token(net, passes):
    # 12 parallel prefill passes over a 12-token prompt reach the fixed point.
    out = Engine(net, "zeros", None, 48, prefill_passes=passes).generate(prompt(), 10)
    assert out.tokens == exact(net, prompt(), 10)[0] and out.calls == 9 and out.commits == [1] * 9


@pytest.mark.parametrize("stand_in", ["zeros", "mtp_state", "state_alignment"])
def test_pipelined_correction_commits_several_tokens_per_call(net, stand_in):
    for depth in (0, 2):
        out = Engine(net, stand_in, depth, 64).generate(prompt(), 16)
        assert len(out.tokens) == 16 and sum(out.commits) >= 15 and out.calls == len(out.commits) <= 15
        assert all(w <= 3 * (depth + 1) for w in out.widths)


def test_transformer_speculative_decoding_is_lossless(transformer):
    reference = exact(transformer, prompt(), 16)[0]
    for depth in DEPTHS:
        out = Engine(transformer, "mtp_state", depth, 64, prefill_passes=4).generate(prompt(), 16)
        assert out.tokens == reference and out.calls <= 15


def test_decoding_stops_before_the_end_token(net):
    full = Engine(net, "mtp_state", 1, 64).generate(prompt(), 16).tokens
    end = full[5]
    out = Engine(net, "mtp_state", 1, 64).generate(prompt(), 16, end=end)
    assert out.tokens == full[:full.index(end)]


def test_teacher_forced_scores_are_exact_without_stand_ins(net, transformer):
    for model in (net, transformer):
        target, rows = exact(model, prompt(), 14)
        reference = -rows[torch.arange(14), torch.tensor(target)]
        for depth in DEPTHS:
            out = Engine(model, "mtp_state", depth, 64, score=True).generate(prompt(), 0, target=target)
            assert out.tokens == target and len(out.nll) == 14
            served = torch.tensor(out.nll)
            if depth is None or model is transformer:
                assert torch.allclose(served, reference, atol=1e-4), (depth, served - reference)


def test_speculative_sampling_follows_the_verifier():
    torch.manual_seed(0)
    p, r = torch.tensor([0.5, 0.3, 0.2, 0.0]), torch.tensor([0.1, 0.1, 0.4, 0.4])
    drawn = torch.multinomial(r, 20000, replacement=True).tolist()
    dists = torch.stack([p, torch.full((4,), 0.25)])
    successors = [speculative(dists, [x], [r])[0] for x in drawn]
    assert torch.allclose(torch.bincount(torch.tensor(successors), minlength=4) / len(drawn), p, atol=0.015)


@pytest.mark.parametrize("depth", [None, 1])
def test_sampling_records_each_generated_token_distribution(net, depth):
    torch.manual_seed(0)
    record = []
    out = Engine(net, "state_alignment", depth, 64, temperature=1.0).generate(prompt(), 16, record=record)
    assert len(out.tokens) == 16 and len(record) >= 16
    reference = Engine(net, "zeros", None, 64, temperature=1.0).exact_distributions(prompt(), out.tokens)
    divergence = kl(torch.stack(record[:16]), reference)
    assert reference.shape == (16, 23) and (divergence < 1e-6 if depth is None else divergence >= 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
@pytest.mark.parametrize("stand_in, depth, passes", [("zeros", None, None), ("mtp_state", 0, 4), ("state_alignment", 2, 4)])
def test_graphs_match_eager(net, stand_in, depth, passes):
    model = net.cuda()
    eager = Engine(model, stand_in, depth, 64, prefill_passes=passes, graphs=False).generate(prompt(), 16)
    graphed = Engine(model, stand_in, depth, 64, prefill_passes=passes, graphs=True)
    graphed.generate(prompt(), 16)  # captures
    again = graphed.generate(prompt(), 16)
    assert (again.tokens, again.widths, again.commits) == (eager.tokens, eager.widths, eager.commits)