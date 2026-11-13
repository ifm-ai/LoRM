"""Write the Olmo 3 Base Easy BPB requests (suites olmo3:base_easy:{math,code,qa}_bpb) exactly as OLMES builds them:
OLMES's task classes download the data and build the few-shot contexts and continuations, and this script writes one
JSONL per task with the context, continuation, gold label and ids. No model is loaded. It runs in OLMES's own
environment (github.com/allenai/olmes; the paper used commit 5a51f502d463b8cdc4a2dcad7d7096c41ff1197e), whose pins
conflict with this repository's:

    python works/latent_feedback/measure/requests.py --olmes OLMES_CHECKOUT --out data/task-bpb [TASK ...]

The dataset repository IFM/latent-feedback-data holds the request files the paper scored, under task-bpb/.
"""

import argparse
import json
import sys
from pathlib import Path

SUITES = ["olmo3:base_easy:math_bpb", "olmo3:base_easy:code_bpb", "olmo3:base_easy:qa_bpb"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tasks", nargs="*")
    parser.add_argument("--olmes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.olmes))
    from oe_eval.configs.task_suites import TASK_SUITE_CONFIGS
    from oe_eval.configs.tasks import TASK_CONFIGS
    from oe_eval.tasks.oe_eval_tasks import TASK_REGISTRY
    from oe_eval.utils import parse_args_string

    def expand(suite):  # oe_eval/launch.py's suite resolution, without its CLI imports
        return [t for task in TASK_SUITE_CONFIGS[suite]["tasks"] for t in expand(task)] if suite in TASK_SUITE_CONFIGS \
            else [suite]

    args.out.mkdir(parents=True, exist_ok=True)
    index = []
    for suite in SUITES:
        for spec in expand(suite):
            if args.tasks and spec not in args.tasks:
                continue
            settings = dict(TASK_CONFIGS[spec]) if spec in TASK_CONFIGS else parse_args_string(spec, "task_name")
            settings.setdefault("metadata", {})["alias"] = spec
            name = settings.pop("task_name")
            task = TASK_REGISTRY[name](task_name=name, task_config=settings)
            task.download()
            task.build_all_requests()
            with (args.out / f"{spec.replace(':', '_')}.jsonl").open("w") as f:
                for instance in task._instances:
                    d = instance.to_dict()
                    f.write(json.dumps({"suite": suite.split(":")[-1], "task": spec, "doc_id": d["doc_id"],
                                        "idx": d.get("idx"), "label": d.get("label"), "native_id": d.get("native_id"),
                                        "context": d["request"].get("context"),
                                        "continuation": d["request"].get("continuation")}) + "\n")
            index.append({"suite": suite, "task": spec, "requests": len(task._instances), "config": task.task_config})
            print(f"{spec:45s} {len(task._instances):7d} requests", flush=True)
    (args.out / "index.json").write_text(json.dumps(index, indent=1, default=str))


if __name__ == "__main__":
    main()
