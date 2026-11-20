"""The paper's tables as LaTeX rows, from results/ (see README.md for the measurements that write it). Entries are means
over the three seeds with their standard deviations (ddof 1); the 780M MTP state, whose two seeds fail as stand-ins,
shows its range over seeds.

    python -m works.feedback_shortcut.tables [--out tables.tex] [--check main.tex]

--check compares every generated row with the paper's source and lists the rows that differ.
"""

import argparse
import json
import math
import statistics as st
from pathlib import Path

from .make_config import runs

RESULTS = Path(__file__).resolve().parent / "results"
SIZES = {"135m": "135M", "780m": "780M"}
SUITES = ("math", "code", "qa")
# Setting rows: (label, release setting, stand-in).
ROWS = [("Transformer", "transformer-mtp", "mtp_state"), ("zeros", "feedback", "zeros"),
        ("MTP state", "feedback", "mtp_state"), ("state alignment", "state-alignment", "state_alignment"),
        ("in-place", "in-place", "mtp_state"), ("\\quad in half the sequences", "in-place-half", "mtp_state"),
        ("\\quad only during decay", "in-place-decay", "mtp_state")]
RANGE = ("780m", "feedback", "mtp_state")


def load(kind: str, name: str) -> dict:
    return json.loads((RESULTS / kind / f"{name}.json").read_text())


def seeds(size: str, setting: str) -> list[str]:
    table = runs()
    return sorted((r["name"] for r in table.values() if r["size"] == size and r["setting"] == setting),
                  key=lambda n: table[n]["seed"])


def pm(mean: float, sd: float, digits: int, unit: str = "") -> str:
    sd = f"{sd:.{digits}f}" if round(sd, digits) else f"<{10 ** -digits:.{digits}f}"
    return f"${mean:.{digits}f}{{\\scriptstyle\\pm{sd}}}{unit}$"


def cell(values: list[float], digits: int, unit: str = "") -> str:
    assert len(values) == 3, values
    return pm(st.mean(values), st.stdev(values), digits, unit)


def span(values: list[float], digits: int, unit: str = "") -> str:
    return f"${min(values):.{digits}f}$--${max(values):.{digits}f}{unit}$"


def exact_bpb(name: str) -> float:
    return load("quality", name)["mean"]["exact"]["bpb"]


def bold_best(rows: list[list[str]], column: int) -> None:
    """Bold the lowest mean in a column, in every row that ties with it as printed."""
    def mean(c):
        try:
            return float(c.split("$")[1].split("{")[0])
        except (IndexError, ValueError):
            return None
    best = min(mean(r[column]) for r in rows if mean(r[column]) is not None)
    for r in rows:
        if mean(r[column]) == best:
            r[column] = r[column].replace("$", "$\\mathbf{", 1).replace("{\\scriptstyle", "}{\\scriptstyle", 1)


def size_row(size: str, columns: int) -> str:
    return f"\\multicolumn{{{columns}}}{{l}}{{\\textit{{{SIZES[size]}}}}} \\\\"


def speedup(name: str, stand_in: str, batch: str, depth: str) -> float:
    b = load("batched", f"{name}-{stand_in}")["batches"][batch]
    return b[depth]["throughput"] / b["exact"]["throughput"]


def main_table() -> list[str]:
    flops, down = json.loads((RESULTS / "flops.json").read_text()), json.loads((RESULTS / "downstream.json").read_text())
    lines = []
    for size in SIZES:
        if lines:
            lines.append("\\midrule")
        lines.append(size_row(size, 7))
        aligned = seeds(size, "state-alignment")
        for label, setting in (("Transformer", "transformer-mtp"), ("Latent feedback, exact", "feedback"),
                               ("Feedback Shortcut (ours)", "state-alignment")):
            if setting == "state-alignment":
                paloma = [exact_bpb(n) + load("served_paloma", f"{n}-state_alignment")["served_minus_exact"]["1"]
                          for n in aligned]
                served = down["served"][f"{size} state-alignment"]["1"]
                tasks = [pm(served[s]["mean"], served[s]["sd"], 4) for s in SUITES]
                speed = cell([speedup(n, "state_alignment", "1", "1") for n in aligned], 2, "\\times")
            else:
                paloma = [exact_bpb(n) for n in seeds(size, setting)]
                tasks = [cell([r[s] for r in down["absolute"][f"{size} {setting}"]["per_seed"]], 4) for s in SUITES]
                speed = (cell([speedup(n, "mtp_state", "1", "0") for n in seeds(size, setting)], 2, "\\times")
                         if setting == "transformer-mtp" else "$1.00\\times$")
            over = flops[f"{size} {setting}"]["over_transformer"]
            lines.append(" & ".join([label, f"${over:.2f}\\times$", cell(paloma, 4), *tasks, speed]) + " \\\\")
    return lines


def quality_table() -> list[str]:
    flops = json.loads((RESULTS / "flops.json").read_text())
    rows = []
    for label, setting, stand_in in [ROWS[0], ROWS[2], ROWS[1], *ROWS[3:]]:
        cells = []
        for size in SIZES:
            names = seeds(size, setting)
            if setting == "transformer-mtp":
                cells += ["---", "---"]
            else:
                extra = flops[f"{size} {setting}"]["extra"]
                errors = [1e3 * load("quality", n)["mean"][stand_in]["corrected"] for n in names]
                cells += ["0\\%" if extra == 0 else f"{100 * extra:.1f}\\%",
                          f"${min(errors):.0f}$ to ${max(errors):.0f}$" if (size, setting, stand_in) == RANGE
                          else cell(errors, 1)]
            cells.append(cell([exact_bpb(n) for n in names], 4))
        rows.append([label, *cells])
    for column in (2, 3, 5, 6):
        bold_best(rows[1:], column)
    return [" & ".join(rows[0]) + " \\\\", "\\midrule", *(" & ".join(r) + " \\\\" for r in rows[1:])]


def speed_table() -> list[str]:
    def results(name, stand_in, temperature):
        return load("specbench", f"{name}-{stand_in}" + ("" if temperature == 0 else f"-t{temperature}"))["depths"]

    def per_seed(size, setting, stand_in, temperature, depth, key):
        values = []
        for n in seeds(size, setting):
            r = results(n, stand_in, temperature)[depth]
            values.append(r["speedup"]["overall"] if key == "speedup" else r[key])
        return values

    lines = []
    for block, temperature in (("Speedup, greedy $\\uparrow$", 0), ("Speedup, sampling at temperature 1 $\\uparrow$", 1)):
        if not all((RESULTS / "specbench" / f"{seeds('135m', s)[0]}-{i}{'' if temperature == 0 else '-t1'}.json").exists()
                   for _, s, i in ROWS):
            continue
        lines += ["\\midrule", f"\\multicolumn{{9}}{{l}}{{\\textit{{{block}}}}} \\\\"]
        for label, setting, stand_in in ROWS:
            cells = []
            for size in SIZES:
                if setting == "transformer-mtp":
                    cells.append(f"\\multicolumn{{4}}{{c}}{{{cell(per_seed(size, setting, stand_in, temperature, '0', 'speedup'), 2, '\\times')}}}")
                    continue
                for d in "0123":
                    v = per_seed(size, setting, stand_in, temperature, d, "speedup")
                    cells.append(span(v, 2, "\\times") if (size, setting, stand_in) == RANGE else cell(v, 2, "\\times"))
            lines.append(" & ".join([label, *cells]) + " \\\\")
            if setting == "transformer-mtp":
                lines.append("\\cmidrule{1-9}")
    for block, key in (("Tokens per call $\\uparrow$", "tokens_per_call"), ("Positions verified per committed token $\\rho$ $\\downarrow$", "rho")):
        lines += ["\\midrule", f"\\multicolumn{{9}}{{l}}{{\\textit{{{block}}}}} \\\\"]
        for label, setting, stand_in in ROWS[1:]:
            cells = []
            for size in SIZES:
                for d in "0123":
                    v = per_seed(size, setting, stand_in, 0, d, key)
                    cells.append(span(v, 2) if (size, setting, stand_in) == RANGE else
                                 cell(v, 2) if key == "tokens_per_call" else f"${st.mean(v):.2f}$")
            lines.append(" & ".join([label, *cells]) + " \\\\")
    return lines


def sci(values: list[float]) -> str:
    """Mean and standard deviation over seeds at the mean's power of ten."""
    m = st.mean(values)
    e = math.floor(math.log10(m))
    e += round(m / 10 ** e, 1) >= 10
    sd = st.stdev(values) / 10 ** e
    sd = f"{sd:.1f}" if round(sd, 1) else "<0.1"
    return f"${m / 10 ** e:.1f}{{\\scriptstyle\\pm{sd}}}\\times10^{{{e}}}$"


LONG = [("zeros", "feedback", "zeros"), ("state alignment", "state-alignment", "state_alignment"),
        ("in-place", "in-place", "mtp_state")]


def long_table() -> list[str]:
    lines = []
    for (label, setting, stand_in), d in ((LONG[0], "1"), (LONG[1], "1"), (LONG[1], "2"), (LONG[2], "1"), (LONG[2], "2")):
        cells = [sci([load("long", f"{n}-{stand_in}")["kl"][d] for n in seeds(size, setting)]) for size in SIZES]
        lines.append(" & ".join([label, d, *cells]) + " \\\\")
    return lines


def tie_cell(values: list[float], digits: int) -> str:
    m, s = st.mean(values), st.stdev(values)
    return (f"${m:.{digits}f}{{\\scriptstyle\\pm{s:.{digits}f}}}$" if s >= 0.5 * 10 ** -digits
            else f"${m:.{digits}f}{{\\scriptstyle\\pm<{10 ** -digits:g}}}$")


def near_tie_table() -> list[str]:
    lines = []
    for size in SIZES:
        lines += ["\\midrule", size_row(size, 9)]
        for label, setting, stand_in in LONG:
            for d in "012":
                gs = [load("long", f"{n}-{stand_in}")["greedy"][d] for n in seeds(size, setting)]
                columns = [([g["departed"] for g in gs], 0), ([100 * g["near_at_risk"] / g["at_risk"] for g in gs], 2),
                           ([100 * g["at_near"] / max(g["departed"], 1) for g in gs], 0),
                           ([1000 * g["at_near"] / g["near_at_risk"] for g in gs], 0),
                           ([1000 * (g["departed"] - g["at_near"]) / (g["at_risk"] - g["near_at_risk"]) for g in gs], 2),
                           ([g["served"] for g in gs], 3), ([g["exact"] for g in gs], 3)]
                lines.append(" & ".join([label, d, *(tie_cell(v, k) for v, k in columns)]) + " \\\\")
    return lines


def acceptance_table() -> list[str]:
    def seed_cells(name, stand_in):
        q = load("quality", name)["mean"]
        if "acceptance" not in q:
            return ["---"] * 5
        error = q[stand_in]["corrected"]
        return ([f"{a:.3f}" for a in q["acceptance"]]
                + [f"{error:.2f}\\phantom{{00}}" if error >= 0.1 else f"{error:.4f}", f"{q['exact']['bpb']:.4f}"])

    lines = []
    for label, setting, stand_in in (("MTP state", "feedback", "mtp_state"), ("state alignment", "state-alignment", "state_alignment"),
                                     ("in-place", "in-place", "mtp_state"), ("in-place, half", "in-place-half", "mtp_state")):
        lines.append("\\cmidrule{1-12}" if lines else "\\midrule")
        for seed in range(3):
            cells = [label if seed == 0 else "", str(seed)]
            for size in SIZES:
                cells += seed_cells(seeds(size, setting)[seed], stand_in)
            lines.append(" & ".join(cells) + " \\\\")
    return lines


COMPARISONS = [("feedback $-$ Transformer", "feedback - transformer-mtp"), ("state alignment $-$ feedback", "state-alignment - feedback"),
               ("in-place $-$ feedback", "in-place - feedback"), ("state alignment $-$ in-place", "state-alignment - in-place")]


def differences_table() -> list[str]:
    down = json.loads((RESULTS / "downstream.json").read_text())["differences"]
    lines = []
    for size in SIZES:
        lines += ["\\cmidrule{1-4}" if lines else "\\midrule", size_row(size, 4)]
        for label, key in COMPARISONS:
            cells = [f"${d['diff']:+.4f}$ {{\\scriptsize$[{d['lo']:+.4f}, {d['hi']:+.4f}]$}}"
                     for d in (down[f"{size} {key} {s}"] for s in SUITES)]
            lines.append(" & ".join([label, *cells]) + " \\\\")
    return lines


def served_table() -> list[str]:
    served = json.loads((RESULTS / "downstream.json").read_text())["served"]
    num = lambda x: f"${x:.4f}$".replace("$-0.0000$", "$0.0000$")  # noqa: E731
    lines = []
    for size in SIZES:
        lines += ["\\cmidrule{1-10}" if lines else "\\midrule", size_row(size, 10)]
        for label, setting in (("state alignment", "state-alignment"), ("in-place", "in-place")):
            row = served[f"{size} {setting}"]
            cells = [c for s in SUITES for c in (num(row["drop"][s]), num(row["0"][s]["diff"]), num(row["1"][s]["diff"]))]
            lines.append(" & ".join([label, *cells]) + " \\\\")
    return lines


TABLES = {"tab:main": main_table, "tab:quality": quality_table, "tab:speed": speed_table, "tab:long": long_table,
          "tab:neartie": near_tie_table, "tab:acceptance": acceptance_table, "tab:downdiff": differences_table,
          "tab:downserved": served_table}


def paper_rows(tex: str, label: str) -> list[str]:
    body = tex.split(f"\\label{{{label}}}", 1)[1].split("\\bottomrule", 1)[0]
    return [line.strip() for line in body.splitlines() if "&" in line or "\\multicolumn{" in line and "{l}" in line]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", type=Path, help="the paper's LaTeX source")
    args = parser.parse_args()
    text = []
    for label, build in TABLES.items():
        rows = build()
        text += [f"% {label}", *rows, ""]
        if args.check:
            theirs = set(paper_rows(args.check.read_text(), label))
            ours = [r.strip() for r in rows if "&" in r or "{l}" in r]
            missing = [r for r in ours if r not in theirs]
            print(f"{label}: {len(ours) - len(missing)} of {len(ours)} rows as in the paper")
            for row in missing:
                print(f"  not in the paper: {row}")
    if args.out:
        args.out.write_text("\n".join(text))
    elif not args.check:
        print("\n".join(text))


if __name__ == "__main__":
    main()
