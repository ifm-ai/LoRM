import pytest
import torch

from works.feedback_shortcut.measure.batched import BatchedEngine, tier
from works.feedback_shortcut.measure.engine import Engine


def prompts(count=6):
    g = torch.Generator().manual_seed(2)
    return [torch.randint(3, 23, (int(n),), generator=g).tolist() for n in torch.randint(4, 14, (count,), generator=g)]


@pytest.mark.parametrize("stand_in", ["zeros", "state_alignment"])
def test_batched_requests_decode_as_they_do_alone(net, stand_in):
    model = net.double()  # float64, so that packed and single-request arithmetic agree
    requests, depths = prompts(), ["exact", 0, 1, 2, "exact", 1]
    engine = BatchedEngine(model, 3, stand_in, length=96, attention="sdpa", passes=4, new=16, end=-1)
    results, steps = engine.serve(requests, depths, batch=3)
    for prompt, depth, result in zip(requests, depths, results):
        alone = Engine(model, stand_in, None if depth == "exact" else depth, 96, prefill_passes=4).generate(prompt, 16)
        assert result["output"] == alone.tokens, depth
        assert [c[2] for c in result["calls"]] == alone.commits and [c[1] for c in result["calls"]] == alone.widths
    assert max(s["rows"] for s in steps) == 3


def test_token_tiers_round_up():
    assert [tier(x) for x in (1, 16, 17, 33, 100)] == [1, 16, 24, 40, 104]
