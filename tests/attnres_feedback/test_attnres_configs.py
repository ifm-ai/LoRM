from lorm.train.main import resolve_config
import works.attnres_feedback.model  # noqa: F401
from works.attnres_feedback.make_config import SEQUENCES, make_config, runs


def test_every_run_resolves():
    table = runs()
    assert len(table) == 136
    for row in table.values():
        config = resolve_config(make_config(row))
        assert config["train"]["batch_size"] * config["train"]["gradient_accumulation"] * 8 == SEQUENCES
        instance = config["model"].get("instance") or {}
        assert (instance.get("name") == "attnres") == ("attnres" in row["arm"])
        assert instance.get("scaled_logits", True) == (not row["arm"].endswith("-unscaled"))
