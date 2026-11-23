"""Heatmaps of where each depth mix looks, one PDF per model, from results/patterns (measure/patterns.py): the seed-mean
softmax weight of every read on a fixed source axis. Rows are reads (a loop's first visit, then its second); columns
are b0, the fed-back blocks f1..f6 or the first visit's blocks, the pass's own blocks B1..B6 and the running sum.

    python -m works.attnres_feedback.figures [--out figures]
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
LAYERS, BLOCKS = 12, 6
TITLES = {"transformer-attnres": "Transformer + AttnRes", "feedback-attnres-blocks": "Latent feedback + AttnRes blocks (pass 4)",
          "loop2-attnres": "Loop × 2 + AttnRes"}


def place(kind, read, call, weights):
    """One read's weights on the fixed axis; NaN where the source does not exist yet."""
    fed = BLOCKS if kind == "feedback-attnres-blocks" else 0
    row = np.full(1 + fed + BLOCKS * (2 if kind == "loop2-attnres" else 1) + 1, np.nan)
    if read == "output":
        row[:len(weights)] = weights
        return row
    sub, layer = read.split()
    layer = int(layer)
    sources = 1 + fed + (layer - 1) // 2 + (BLOCKS if kind == "loop2-attnres" and call == 2 else 0)
    row[:sources] = weights[:sources]
    if not (sub == "attn" and layer % 2):
        row[-1] = weights[-1]
    return row


def matrix(kind, results):
    calls = {"transformer-attnres": [1], "feedback-attnres-blocks": [4], "loop2-attnres": [1, 2]}[kind]
    reads = [f"{s} {k}" for k in range(1, LAYERS + 1) for s in ("attn", "mlp")]
    rows, labels = [], []
    for call in calls:
        for read in reads + (["output"] if call == calls[-1] else []):
            key = f"{read}#{1 if read == 'output' and kind != 'feedback-attnres-blocks' else call}"
            rows.append(np.mean([place(kind, read, call, r["reads"][key]) for r in results], 0))
            labels.append(read if len(calls) == 1 or read == "output" else f"v{call} {read}")
    columns = ["b0"] + ([f"f{j}" for j in range(1, BLOCKS + 1)] if kind == "feedback-attnres-blocks" else [])
    columns += ([f"v1 B{j}" for j in range(1, BLOCKS + 1)] + [f"v2 B{j}" for j in range(1, BLOCKS + 1)]
                if kind == "loop2-attnres" else [f"B{j}" for j in range(1, BLOCKS + 1)])
    return np.array(rows), labels, columns + ["sum"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    results = defaultdict(list)
    for path in sorted((HERE / "results" / "patterns").glob("*.json")):
        result = json.loads(path.read_text())
        results[result["run"].split("-134m-")[0]].append(result)
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Palatino Linotype", "Palatino", "DejaVu Serif"]})
    cmap = plt.get_cmap("Greys").copy()
    cmap.set_bad("white")
    for kind, title in TITLES.items():
        m, labels, columns = matrix(kind, results[kind])
        fig, ax = plt.subplots(figsize=(1.6 + 0.42 * m.shape[1], 1.4 + 0.2 * m.shape[0]))
        image = ax.imshow(np.ma.masked_invalid(m), cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(columns)), columns, rotation=90, fontsize=10)
        ax.set_yticks(range(len(labels)), labels, fontsize=9)
        ax.set_title(title, fontsize=13)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if kind != "transformer-attnres":
            ax.axvline(BLOCKS + 0.5, color="#1f5fa8", lw=1.2)
        if kind == "loop2-attnres":
            ax.axhline(2 * LAYERS - 0.5, color="#1f5fa8", lw=1.2)
        bar = fig.colorbar(image, ax=ax, shrink=0.6 if kind == "loop2-attnres" else 0.8, pad=0.03)
        bar.set_label("softmax weight")
        bar.outline.set_visible(False)
        fig.savefig(args.out / f"patterns-{kind}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(args.out / f"patterns-{kind}.pdf")


if __name__ == "__main__":
    main()
