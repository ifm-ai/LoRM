import copy
import os
import signal

import pytest
import torch
import yaml

from lorm.train import run
from lorm.train.config import load_config, merge
from lorm.train.logger import Logger


def parameters(output_dir):
    return torch.load(os.path.join(output_dir, "checkpoint.pt"), weights_only=True)["model"]


def stop_at(monkeypatch, step):
    """Send SIGTERM once `step` is logged, as a preemption would."""
    original = Logger.log

    def log(self, logged, metrics):
        original(self, logged, metrics)
        if logged == step and "train/loss" in metrics:
            os.kill(os.getpid(), signal.SIGTERM)
    monkeypatch.setattr(Logger, "log", log)


def test_training_logs_and_writes_a_resolved_config(config):
    progress = run(config)
    assert progress["step"] == 6
    written = yaml.safe_load(open(os.path.join(config["output_dir"], "config.yaml")))
    assert written["algo"] == {"name": "standard"} and written["train"]["max_steps"] == 6
    lines = open(os.path.join(config["output_dir"], "metrics.jsonl")).read().splitlines()
    assert any('"eval/nll"' in line for line in lines)


def test_a_run_stopped_by_sigterm_resumes_exactly(config, monkeypatch):
    whole = copy.deepcopy(config)
    whole["output_dir"] += "-whole"
    run(whole)
    original = Logger.log
    stop_at(monkeypatch, 3)
    with pytest.raises(SystemExit):
        run(config)
    monkeypatch.setattr(Logger, "log", original)
    run(config, resume="auto")
    expected, resumed = parameters(whole["output_dir"]), parameters(config["output_dir"])
    assert all(torch.equal(expected[k], resumed[k]) for k in expected)


def test_a_finished_run_is_not_overwritten(config):
    run(config)
    with pytest.raises(FileExistsError):
        run(config)


def test_yaml_bases_merge_relative_to_the_file(tmp_path):
    (tmp_path / "base.yaml").write_text("a: {b: 1, c: 2}\nd: 3\n")
    (tmp_path / "run.yaml").write_text("base: base.yaml\na: {c: 4}\n")
    assert load_config(tmp_path / "run.yaml") == {"a": {"b": 1, "c": 4}, "d": 3}
    assert merge({"a": {"b": 1}}, {"a": {"_replace": True, "c": 2}}) == {"a": {"c": 2}}
