"""Downstream suite scores and their uncertainty (Tables 3, 4 and 9), from lorm.eval.task_bpb's outputs and served.py's
downstream outputs, by the release run names of configs/runs.csv.

A suite score is OLMES's macro mean: the mean over the suite's groups of their tasks' mean BPB, where MMLU, ARC, Basic
Skills and MT-MBPP's languages each form one group. A difference between two settings has two sources of noise: which
seeds were trained, from Welch's test on each setting's three seed scores, and which items each benchmark happens to
contain, the standard error over 2,000 paired resamples of every task's items (the same items for both settings, every
seed kept). The two add in variance, with Welch-Satterthwaite degrees of freedom. A served setting's error is its
served BPB minus its own exact decoding after the same prefill, on the same items: paired, so the seed term is Student t
on the seeds' differences.

    python -m works.feedback_shortcut.measure.stats --exact results/task_bpb --served results/served-downstream \
        --out results/downstream.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from ..make_config import runs

GROUPS = [("mmlu", r"^mmlu_"), ("arc", r"^arc_"), ("basic_skills", r"^basic_skills_"), ("mt_mbpp", r"^mt_mbpp_v2fix_")]
SUITES = ("math", "code", "qa")
REPLICATES = 2000
DEPTHS = ("0", "1", "2", "3")
COMPARISONS = [("feedback", "transformer-mtp"), ("state-alignment", "feedback"), ("in-place", "feedback"),
               ("state-alignment", "in-place")]


def suite_score(tasks: dict, suite: str, pick: dict | None = None) -> float:
    groups = defaultdict(list)
    for name, task in tasks.items():
        if task["suite"] == suite:
            items = np.asarray(task["items"])
            groups[next((g for g, pattern in GROUPS if re.match(pattern, name)), name)].append(
                (items if pick is None else items[pick[name]]).mean())
    return float(np.mean([np.mean(v) for v in groups.values()]))


def seeds(table: dict, size: str, setting: str) -> list[str]:
    return sorted((r["name"] for r in table.values() if r["size"] == size and r["setting"] == setting),
                  key=lambda name: table[name]["seed"])


def difference(a: list[dict], b: list[dict], suite: str, rng) -> dict:
    """First setting minus second: the difference, its 95% interval, and the seed and item standard errors."""
    names = [n for n, t in a[0].items() if t["suite"] == suite]
    scores = [np.array([suite_score(run, suite) for run in setting]) for setting in (a, b)]
    seed_var = [s.var(ddof=1) / len(s) for s in scores]
    resampled = []
    for _ in range(REPLICATES):
        pick = {n: rng.integers(0, len(a[0][n]["items"]), len(a[0][n]["items"])) for n in names}
        resampled.append(np.mean([suite_score(r, suite, pick) for r in a]) - np.mean([suite_score(r, suite, pick) for r in b]))
    var = sum(seed_var) + np.var(resampled, ddof=1)
    df = var ** 2 / sum(v ** 2 / (len(s) - 1) for v, s in zip(seed_var, scores))
    d, half = scores[0].mean() - scores[1].mean(), stats.t.ppf(0.975, df) * np.sqrt(var)
    return dict(diff=float(d), lo=float(d - half), hi=float(d + half), seed_se=float(np.sqrt(sum(seed_var))),
                item_se=float(np.std(resampled, ddof=1)))


def paired(a: list[dict], b: list[dict], suite: str, rng) -> dict:
    """Mean of a - b over paired runs, with its 95% interval over seeds and items."""
    names = [n for n, t in a[0].items() if t["suite"] == suite]
    diff = np.array([suite_score(x, suite) - suite_score(y, suite) for x, y in zip(a, b)])
    resampled = []
    for _ in range(REPLICATES):
        pick = {n: rng.integers(0, len(a[0][n]["items"]), len(a[0][n]["items"])) for n in names}
        resampled.append(np.mean([suite_score(x, suite, pick) - suite_score(y, suite, pick) for x, y in zip(a, b)]))
    seed_var = diff.var(ddof=1) / len(diff)
    var = seed_var + np.var(resampled, ddof=1)
    df = var ** 2 / (seed_var ** 2 / (len(diff) - 1)) if seed_var > 0 else np.inf
    half = stats.t.ppf(0.975, df) * np.sqrt(var)
    return dict(diff=float(diff.mean()), lo=float(diff.mean() - half), hi=float(diff.mean() + half))


def served_tasks(directory: Path, run: str) -> dict:
    """{depth: tasks} of a run's served downstream shards, each task's items back in request order."""
    merged = defaultdict(lambda: defaultdict(lambda: dict(index=[], items=[])))
    for path in sorted(directory.glob(f"{run}*.json")):
        result = json.loads(path.read_text())
        if result["run"] != run:
            continue
        for name, task in result["parts"].items():
            for depth, items in task["items"].items():
                entry = merged[depth][name]
                entry["suite"] = task["suite"]
                entry["index"] += task["index"]
                entry["items"] += items
    return {depth: {name: dict(suite=t["suite"], items=[x for _, x in sorted(zip(t["index"], t["items"]))])
                    for name, t in tasks.items()} for depth, tasks in merged.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--exact", type=Path, required=True, help="lorm.eval.task_bpb outputs, OUT/<run>.json")
    parser.add_argument("--served", type=Path, help="served.py --source downstream outputs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    table, rng, out = runs(), np.random.default_rng(0), {"absolute": {}, "differences": {}, "served": {}}
    served_rng = np.random.default_rng(0)
    load = lambda name: json.loads((args.exact / f"{name}.json").read_text())["tasks"]  # noqa: E731
    for size in ("135m", "780m"):
        settings = {s: [load(n) for n in seeds(table, size, s)] for s in
                    ("transformer-mtp", "feedback", "state-alignment", "in-place", "in-place-half", "in-place-decay")}
        for setting, runs_ in settings.items():
            values = np.array([[suite_score(r, s) for s in SUITES] for r in runs_])
            out["absolute"][f"{size} {setting}"] = dict(mean=dict(zip(SUITES, values.mean(0).tolist())),
                                                        sd=dict(zip(SUITES, values.std(0, ddof=1).tolist())),
                                                        per_seed=[dict(zip(SUITES, v)) for v in values.tolist()])
        for a, b in COMPARISONS:
            for suite in SUITES:
                out["differences"][f"{size} {a} - {b} {suite}"] = difference(settings[a], settings[b], suite, rng)
        if args.served:
            ceiling = {s: np.mean([suite_score(r, s) for r in settings["feedback"]]) for s in SUITES}
            for setting in ("state-alignment", "in-place"):
                served = [served_tasks(args.served, n) for n in seeds(table, size, setting)]
                row = out["served"][f"{size} {setting}"] = {"drop": {
                    s: float(np.mean([suite_score(r, s) for r in settings[setting]]) - ceiling[s]) for s in SUITES}}
                for d in DEPTHS:
                    row[d] = {}
                    for suite in SUITES:
                        point = [suite_score(r[d], suite) for r in served]
                        row[d][suite] = dict(mean=float(np.mean(point)), sd=float(np.std(point, ddof=1)),
                                             **paired([r[d] for r in served], [r["exact"] for r in served], suite,
                                                      served_rng))
    args.out.write_text(json.dumps(out, indent=1))
    for key, d in out["differences"].items():
        print(f"{key}: {d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]")


if __name__ == "__main__":
    main()
