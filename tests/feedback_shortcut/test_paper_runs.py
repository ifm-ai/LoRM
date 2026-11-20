from pathlib import Path

import yaml

from lorm.train.config import load_config
from lorm.train.main import resolve_config
from works.feedback_shortcut.make_config import make_config, runs

CONFIGS = Path(__file__).resolve().parents[2] / "works/feedback_shortcut/configs"


def test_every_setting_resolves():
    paths = sorted(CONFIGS.glob("*/*.yaml"))
    assert len(paths) == 12
    for path in paths:
        assert resolve_config(load_config(path))["model"]["instance"]["heads"] == 3, path


def test_every_paper_run_resolves_with_its_seed(tmp_path):
    table = runs()
    assert len(table) == 36
    for name, row in table.items():
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(make_config(row, tmp_path)))
        config = resolve_config(load_config(path))
        assert config["seed"] == int(row["seed"]) and config["output_dir"] == f"runs/{name}"
        if row["setting"] == "in-place-decay":
            trunk = name.replace("in-place-decay", "feedback")
            assert config["branch_from"].startswith(f"runs/{trunk}/checkpoint-step"), config["branch_from"]
