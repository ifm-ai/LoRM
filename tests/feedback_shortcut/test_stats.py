import numpy as np
import pytest

stats = pytest.importorskip("works.feedback_shortcut.measure.stats")


def test_suite_score_averages_groups_before_tasks():
    tasks = {"mmlu_a": dict(suite="qa", items=[1.0]), "mmlu_b": dict(suite="qa", items=[3.0]),
             "piqa": dict(suite="qa", items=[4.0, 6.0]), "gsm8k": dict(suite="math", items=[9.0])}
    assert stats.suite_score(tasks, "qa") == (2.0 + 5.0) / 2 and stats.suite_score(tasks, "math") == 9.0


def test_difference_of_a_setting_with_itself_is_zero():
    rng = np.random.default_rng(0)
    runs = [{"t": dict(suite="qa", items=list(rng.normal(1, 0.1, 50) + s))} for s in (0.0, 0.01, 0.02)]
    out = stats.difference(runs, runs, "qa", np.random.default_rng(1))
    assert out["diff"] == 0 and out["lo"] < 0 < out["hi"] and out["item_se"] == 0
