#!/usr/bin/env python3
"""Summarise a results folder: tables with 95% confidence intervals and charts.

    python analyze.py results/mock
    python analyze.py results/haiku results/llama      # compare several runs

Writes summary.md, summary_by_defense.csv, by_attack.csv and two PNG charts
into each results folder.
"""
import json
import math
import os
import sys

import pandas as pd

DEF_ORDER = ["none", "D1", "D2", "D3", "D1+D2", "D1+D3", "D2+D3", "D1+D2+D3"]
ATTACK_ORDER = ["role_impersonation", "instruction_laundering", "goal_hijacking",
                "split_payload", "tool_result_injection"]
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(series):
    k, n = int(series.sum()), int(series.count())
    lo, hi = wilson(k, n)
    return k, n, (k / n if n else float("nan")), lo, hi


def fmt(k, n, p, lo, hi):
    return "n/a" if n == 0 else f"{p:.0%} [{lo:.0%}-{hi:.0%}]"


def load(folder):
    path = os.path.join(folder, "results.jsonl")
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path) as f:
        return pd.DataFrame([json.loads(l) for l in f if l.strip()])


def summarise(df):
    att, ben = df[df.attack != "benign"], df[df.attack == "benign"]
    rows = []
    for d in [x for x in DEF_ORDER if x in set(df.defenses)]:
        a = att[att.defenses == d]
        dev, held = rate(a[a.split == "dev"].attack_success), rate(a[a.split == "heldout"].attack_success)
        ben_r = rate(ben[ben.defenses == d].benign_task_ok)
        rows.append({
            "defenses": d,
            "dev_asr": dev[2], "dev_lo": dev[3], "dev_hi": dev[4], "dev_n": dev[1],
            "heldout_asr": held[2], "heldout_lo": held[3], "heldout_hi": held[4], "heldout_n": held[1],
            "generalization_gap": held[2] - dev[2],
            "agents_compromised_mean": a.agents_compromised.mean(),
            "benign_ok": ben_r[2], "benign_n": ben_r[1],
            "parse_failures": int(df[df.defenses == d].parse_failures.sum()),
            "_dev": fmt(*dev), "_held": fmt(*held), "_ben": fmt(*ben_r),
        })
    return pd.DataFrame(rows)


def by_attack(df):
    att = df[df.attack != "benign"]
    t = att.pivot_table(index="attack", columns="defenses", values="attack_success", aggfunc="mean")
    return t.reindex(index=[a for a in ATTACK_ORDER if a in t.index],
                     columns=[d for d in DEF_ORDER if d in t.columns])


def plots(summary, heat, folder, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})

    # 1. Attack success, known vs held-out, per defense setup
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x, w = np.arange(len(summary)), 0.38
    for off, col, key, lab in [(-w / 2, BLUE, "dev", "Known (dev) attacks"),
                               (w / 2, ORANGE, "heldout", "Held-out attacks")]:
        p = summary[f"{key}_asr"].values * 100
        err = np.clip([p - summary[f"{key}_lo"].values * 100, summary[f"{key}_hi"].values * 100 - p], 0, None)
        ax.bar(x + off, p, w - 0.03, color=col, label=lab, zorder=2)
        ax.errorbar(x + off, p, yerr=err, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2, zorder=3)
    ax.set_xticks(x, summary.defenses)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Attack success rate (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(f"Attack success by defense setup ({title}); bars show 95% CI", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "asr_by_defense.png"), facecolor=SURFACE)
    plt.close(fig)

    # 2. Heatmap: attack x defense (single-hue sequential)
    fig, ax = plt.subplots(figsize=(8, 2.8), dpi=160, facecolor=SURFACE)
    im = ax.imshow(heat.values * 100, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(heat.shape[1]), heat.columns)
    ax.set_yticks(range(heat.shape[0]), [a.replace("_", " ") for a in heat.index])
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.values[i, j] * 100
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if v > 55 else INK)
    ax.set_title(f"Attack success per attack ({title})", loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01).outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "asr_heatmap.png"), facecolor=SURFACE)
    plt.close(fig)


def report(folder):
    df = load(folder)
    if df.empty:
        print(f"{folder}: no finished episodes yet. Fix the error shown by run_experiment.py "
              "and run it again, then re-run analyze.py.")
        return
    s, heat = summarise(df), by_attack(df)
    title = df.backend.iloc[0] if "backend" in df else folder
    s.drop(columns=[c for c in s if c.startswith("_")]).to_csv(os.path.join(folder, "summary_by_defense.csv"), index=False)
    heat.to_csv(os.path.join(folder, "by_attack.csv"))
    plots(s, heat, folder, title)

    lines = [f"# Results: {title}", "",
             f"{len(df)} episodes ({(df.attack != 'benign').sum()} attacked, {(df.attack == 'benign').sum()} benign), "
             f"layouts: {', '.join(sorted(df.topology.unique()))}.", "",
             "| Defenses | Known-attack success [95% CI] | Held-out success [95% CI] | Gap | Agents compromised (of 4) | Benign task OK | Parse failures |",
             "|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        lines.append(f"| {r.defenses} | {r._dev} | {r._held} | {r.generalization_gap:+.0%} | "
                     f"{r.agents_compromised_mean:.2f} | {r._ben} | {r.parse_failures} |")
    lines += ["", "## Success rate per attack", "", "| Attack | " + " | ".join(heat.columns) + " |",
              "|---|" + "---|" * len(heat.columns)]
    for a, row in heat.iterrows():
        lines.append(f"| {a} | " + " | ".join(f"{v:.0%}" for v in row.values) + " |")
    topo = df[df.attack != "benign"].groupby(["topology", "defenses"]).agents_compromised.mean().unstack()
    topo = topo[[d for d in DEF_ORDER if d in topo.columns]]
    lines += ["", "## Average agents compromised, by layout", "", "| Layout | " + " | ".join(topo.columns) + " |",
              "|---|" + "---|" * len(topo.columns)]
    for t, row in topo.iterrows():
        lines.append(f"| {t} | " + " | ".join(f"{v:.2f}" for v in row.values) + " |")
    lines += ["", "Charts: asr_by_defense.png, asr_heatmap.png"]
    with open(os.path.join(folder, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    for folder in sys.argv[1:] or ["results/mock"]:
        report(folder)
        print()
