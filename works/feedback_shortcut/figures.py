"""The paper's data figures as vector PDFs, from results/: served quality against speed at each size, speedup against
batch size, and downstream BPB under exact decoding. Means over the three seeds; error bars are standard deviations over
seeds. Needs --extra analysis.

    python -m works.feedback_shortcut.figures --out figures/
"""

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .tables import RESULTS, SIZES, SUITES, exact_bpb, load, seeds  # noqa: E402

TRANSFORMER, PURPLE, PURPLE_LIGHT, GRID = "#404040", "#5E3C99", "#AAA8D0", "#E5E6E5"
STAND_INS = {"state-alignment": ("state_alignment", PURPLE), "in-place": ("mtp_state", PURPLE_LIGHT)}
DMARK, DSIZE = {"0": "o", "1": "^", "2": "s", "3": "D"}, {"0": 3.8, "1": 3.8, "2": 3.2, "3": 3.0}
CEILING = (0, (1, 1.5))


def style():
    plt.rcParams.update({"font.family": ["Palatino Linotype", "Palatino", "serif"], "font.size": 8, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "legend.frameon": False, "hatch.linewidth": 0.6})


def grid(ax):
    ax.grid(True, axis="y", color=GRID, lw=0.6, ls="-")
    ax.set_axisbelow(True)


def ytitle(ax, text, x=0.0, y=1.03):
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=8)


def speedups(size: str, setting: str, stand_in: str, batch: int, depth: str) -> list[float]:
    out = []
    for name in seeds(size, setting):
        b = load("batched", f"{name}-{stand_in}")["batches"][str(batch)]
        out.append(b[depth]["throughput"] / b["exact"]["throughput"])
    return out


def served_bpb(size: str, setting: str, suite: str, depth: str | None, down: dict) -> float:
    """Mean BPB over seeds served at `depth`; depth None is exact decoding."""
    if suite == "paloma":
        if depth is None:
            return st.mean(exact_bpb(n) for n in seeds(size, setting))
        stand_in = STAND_INS[setting][0]
        return st.mean(exact_bpb(n) + load("served_paloma", f"{n}-{stand_in}")["served_minus_exact"][depth]
                       for n in seeds(size, setting))
    if depth is None:
        return down["absolute"][f"{size} {setting}"]["mean"][suite]
    return down["served"][f"{size} {setting}"][depth][suite]["mean"]


def quality_speed(size: str, down: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(6.9, 3.7))
    fig.subplots_adjust(left=0.1, right=0.985, bottom=0.17, top=0.9, wspace=0.42, hspace=0.55)
    for r, batch in enumerate((1, 256)):
        for c, (suite, title) in enumerate((("paloma", "Paloma"), ("math", "Math"), ("code", "Code"), ("qa", "QA"))):
            ax = axes[r, c]
            ax.axhline(served_bpb(size, "feedback", suite, None, down), color=TRANSFORMER, lw=0.6, ls=CEILING, zorder=2)
            for setting in ("in-place", "state-alignment"):
                stand_in, colour = STAND_INS[setting]
                points = [(d, st.mean(speedups(size, setting, stand_in, batch, d)), served_bpb(size, setting, suite, d, down))
                          for d in "3210"]
                ax.plot([p[1] for p in points], [p[2] for p in points], color=colour, lw=1.0, zorder=3)
                for d, x, y in points:
                    ax.plot([x], [y], DMARK[d], ms=DSIZE[d], mfc="white" if d == "0" else colour, mec=colour, mew=0.9, zorder=4)
            x = st.mean(speedups(size, "transformer-mtp", "mtp_state", batch, "0"))
            y = served_bpb(size, "transformer-mtp", suite, None, down)
            ax.plot([x], [y], "o", ms=5.2, mfc="white", mec=TRANSFORMER, mew=0.9, zorder=4)
            ax.plot([x], [y], "o", ms=1.6, mfc=TRANSFORMER, mec=TRANSFORMER, zorder=5)
            if c == 0:
                ax.text(-0.42, 0.5, f"$B={batch}$", transform=ax.transAxes, rotation=90, ha="center", va="center",
                        fontsize=8.5, fontweight="bold")
            if r == 0:
                ax.set_title(title, fontsize=8, fontweight="bold", pad=3)
            grid(ax)
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo - 0.12 * (hi - lo), hi + 0.05 * (hi - lo))
            ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(r"%.1f$\times$"))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
            ax.tick_params(labelsize=6.5)
            if r == 1:
                ax.set_xlabel("Speedup over exact decoding", fontsize=7, labelpad=1)
    ytitle(axes[0, 0], "BPB", x=-0.05, y=1.12)
    handles = [Line2D([], [], color=PURPLE, lw=1.4), Line2D([], [], color=PURPLE_LIGHT, lw=1.4),
               (Line2D([], [], color=TRANSFORMER, marker="o", ms=5.2, ls="none", mfc="white", mew=0.9),
                Line2D([], [], color=TRANSFORMER, marker="o", ms=1.6, ls="none")),
               Line2D([], [], color=TRANSFORMER, lw=0.6, ls=CEILING)]
    handles += [Line2D([], [], color=TRANSFORMER, marker=DMARK[d], ms=DSIZE[d], ls="none",
                       mfc="white" if d == "0" else TRANSFORMER, mec=TRANSFORMER) for d in "0123"]
    labels = ["state alignment", "in-place", "Transformer", "feedback ceiling"] + [f"$D={d}$" for d in "0123"]
    fig.legend(handles=handles, labels=labels, handler_map={tuple: HandlerTuple(ndivide=1)}, loc="lower center", ncol=8,
               bbox_to_anchor=(0.5, 0.0), fontsize=7, handlelength=1.6, columnspacing=1.0, handletextpad=0.4)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def batched(path: Path) -> None:
    lines = [("transformer-mtp", "mtp_state", "0", "Transformer", "o", "white", TRANSFORMER, "--"),
             ("state-alignment", "state_alignment", "0", "state alignment, $D{=}0$", "o", "white", PURPLE, "-"),
             ("state-alignment", "state_alignment", "1", "state alignment, $D{=}1$", "^", PURPLE, PURPLE, "-"),
             ("state-alignment", "state_alignment", "3", "state alignment, $D{=}3$", "D", PURPLE, PURPLE, "-"),
             ("in-place", "mtp_state", "1", "in-place, $D{=}1$", "^", PURPLE_LIGHT, PURPLE_LIGHT, "-"),
             ("in-place", "mtp_state", "3", "in-place, $D{=}3$", "D", PURPLE_LIGHT, PURPLE_LIGHT, "-")]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.3, top=0.86, wspace=0.08)
    handles = []
    for ax, size in zip(axes, SIZES):
        ax.axhline(1, color=TRANSFORMER, lw=0.6, ls=":", zorder=1)
        for setting, stand_in, d, label, marker, face, edge, ls in lines:
            batches = sorted(int(b) for b in load("batched", f"{seeds(size, setting)[0]}-{stand_in}")["batches"])
            values = [speedups(size, setting, stand_in, b, d) for b in batches]
            h = ax.errorbar(batches, [st.mean(v) for v in values], yerr=[st.stdev(v) for v in values], color=edge, ls=ls,
                            lw=0.9, marker=marker, ms=4.4 if setting == "transformer-mtp" else 3.6, mfc=face, mec=edge,
                            mew=0.8, capsize=1.5, elinewidth=0.6, label=label, zorder=3)
            if setting == "transformer-mtp":
                ax.plot(batches, [st.mean(v) for v in values], "o", ms=1.4, color=edge, zorder=4)
            if size == "135m":
                handles.append(h)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
        ax.minorticks_off()
        ax.set_xlabel("Batch size $B$")
        ax.text(0.97, 0.95, SIZES[size], transform=ax.transAxes, ha="right", va="top", fontsize=8, fontweight="bold")
        grid(ax)
    axes[0].set_ylim(0.9, 3.05)
    ytitle(axes[0], "Speedup over exact decoding")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles), fontsize=7.5,
               handlelength=2.6, columnspacing=1.4)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def downstream(down: dict, path: Path) -> None:
    rows = [("transformer-mtp", "Transformer", "white", TRANSFORMER, ""), ("feedback", "feedback", "white", PURPLE, ""),
            ("state-alignment", "+ state alignment", PURPLE, PURPLE, ""), ("in-place", "+ in-place", PURPLE_LIGHT, PURPLE_LIGHT, ""),
            ("in-place-half", "+ in-place, half", "white", PURPLE_LIGHT, "////"),
            ("in-place-decay", "+ in-place, decay only", "white", PURPLE_LIGHT, "....")]
    xs = [0, 1, 2.5, 3.5, 4.5, 5.5]  # a gap after the two baselines
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 3.4))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.9, wspace=0.34, hspace=0.55)
    handles = []
    for r, size in enumerate(SIZES):
        for c, suite in enumerate(SUITES):
            ax, lo, hi = axes[r, c], [], []
            for x, (setting, label, face, edge, hatch) in zip(xs, rows):
                entry = down["absolute"][f"{size} {setting}"]
                m, sd = entry["mean"][suite], entry["sd"][suite]
                lo.append(m - sd)
                hi.append(m + sd)
                h = ax.bar(x, m, width=0.78, color=face, edgecolor=edge, hatch=hatch, lw=0.8, yerr=sd,
                           error_kw=dict(ecolor=TRANSFORMER, elinewidth=0.6, capsize=1.5), label=label, zorder=3)
                if r == 0 and c == 0:
                    handles.append(h)
            width = max(hi) - min(lo)
            ax.set_ylim(min(lo) - 0.25 * width, max(hi) + 0.12 * width)
            ax.set_xticks([])
            ax.set_xlim(-0.6, 6.1)
            ax.set_title(f"{('Math', 'Code', 'QA')[c]}, {SIZES[size]}", fontsize=8, fontweight="bold", pad=4)
            grid(ax)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
            # The bars reach below the axis; draw the spines above their fill.
            for side in ("bottom", "left"):
                ax.spines[side].set_zorder(6)
    ytitle(axes[0, 0], "BPB", y=1.17)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=len(handles), fontsize=7.5,
               handlelength=1.4, columnspacing=1.1)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    down = json.loads((RESULTS / "downstream.json").read_text())
    quality_speed("780m", down, args.out / "fig_results_quality_speed.pdf")
    quality_speed("135m", down, args.out / "fig_results_quality_speed_135m.pdf")
    batched(args.out / "fig_results_batched.pdf")
    downstream(down, args.out / "fig_results_downstream_bars.pdf")
    print("figures in", args.out)


if __name__ == "__main__":
    main()
