#!/usr/bin/env python3
"""Figures for the J1+J2 paper, read from the locked verdict and the pilot fits.

  fig2_calibration.png  J1: accuracy vs mean confidence per arm (F1, F3)
  fig3_hold.png         J2: P(no action) by rung — stakes (a), the name on the door (b)
  fig4_threshold.png    J2: certainty needed to act, ideal vs Jev, by stakes
  fig1_pilot.png        titration pilot: accuracy by difficulty and layout (F1, F2, F3)

Palette: dataviz reference instance (validated: categorical slots 1–3 all-pairs,
blue ordinal ramp 250/450/650); every series is direct-labeled.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FLIGHT = REPO / "jev/results/flight_20260925_2352/verdict.json"
FITS = REPO / "jev/results/pilot_20260925_2349/titration_fits.json"
OUT = HERE / "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = {1: "#86b6ef", 3: "#2a78d6", 9: "#104281"}
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def fig1(v):
    s1 = v["J1"]["secondary"]["S1_auroc2"]
    arms = [("A0", "o", "clean"), ("A1-same", "s", "messy, same evidence"),
            ("A1-matched", "^", "messy, stronger evidence"), ("A2", "D", "evidence removed")]
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=MUTED, zorder=1)
    ax.text(0.655, 0.635, "perfectly calibrated", color=MUTED, fontsize=8.5, rotation=40, ha="center")
    ax.axvline(0.25, ls=":", lw=1, color=MUTED, zorder=1)
    ax.text(0.255, 0.405, "chance", color=MUTED, fontsize=8.5)
    for fam, col, name in (("F1", BLUE, "F1 machines"), ("F3", ORANGE, "F3 lanes")):
        pts = {a: (s1[f"{fam}/{a}"]["acc"], s1[f"{fam}/{a}"]["mean_c"]) for a, _, _ in arms}
        path = [pts[a] for a in ("A0", "A1-same", "A1-matched")]
        ax.plot(*zip(*path), color=col, lw=1.6, zorder=2)
        for a, mk, _ in arms:
            ax.plot(*pts[a], marker=mk, ms=8, color=col, mec=SURF, mew=1.5, ls="none", zorder=3)
        ax.plot([], [], color=col, lw=1.6, label=name)
    lab = {("F1", "A0"): (8, 6), ("F1", "A1-same"): (8, -4), ("F1", "A1-matched"): (8, -12),
           ("F1", "A2"): (8, 2)}
    for (fam, a), (dx, dy) in lab.items():
        x, y = s1[f"{fam}/{a}"]["acc"], s1[f"{fam}/{a}"]["mean_c"]
        text = {k: t for k, _, t in arms}[a]
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color=INK2)
    for a, mk, text in arms:
        ax.plot([], [], marker=mk, ms=7, color=INK2, ls="none", label=text)
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(0.4, 0.92)
    ax.set_xlabel("accuracy (share of choices correct)")
    ax.set_ylabel("mean confidence (probability on the choice)")
    ax.set_title("J1 · Confidence follows the page as well as the answer")
    ax.legend(loc="lower right", fontsize=8.5, ncol=2, handlelength=1.6)
    save(fig, "fig2_calibration.png")


def fig2(v):
    cur = v["J2"]["secondary"]["S6_curves"]
    rungs = list(range(9))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for c in (1, 3, 9):
        ys = [cur[f"M-walked|c{c}"][str(k)]["phold"] for k in rungs]
        a1.plot(rungs, ys, color=RAMP[c], lw=2, marker="o", ms=4.5, label=f"wrong call costs {c}×")
    a1.set_title("(a) Passing rises as the evidence runs out,\n     and rises with the stakes")
    a1.set_xlabel("sources switched off (of 8)")
    a1.set_ylabel("probability on “no action”")
    a1.legend(loc="upper left", fontsize=8.5)
    neu = [cur["M-walked|c3"][str(k)]["phold"] for k in rungs]
    nam = [cur["M-named|c3"][str(k)]["phold"] for k in rungs]
    a2.fill_between(rungs, neu, nam, color=ORANGE, alpha=0.12, lw=0)
    a2.plot(rungs, neu, color=BLUE, lw=2, marker="o", ms=4.5, label="“Take no action this round”")
    a2.plot(rungs, nam, color=ORANGE, lw=2, marker="o", ms=4.5, label="“witness: observe without acting”")
    gap = max(range(9), key=lambda k: nam[k] - neu[k])
    a2.annotate(f"+{nam[gap] - neu[gap]:.2f}", (gap, (nam[gap] + neu[gap]) / 2), xytext=(7, -4),
                textcoords="offset points", fontsize=9, color=INK, fontweight="bold")
    a2.set_title("(b) Same situation and payoffs;\n     only the option's name differs (c = 3)")
    a2.set_xlabel("sources switched off (of 8)")
    a2.legend(loc="upper left", fontsize=8.5)
    for ax in (a1, a2):
        ax.set_xticks(rungs)
        ax.set_ylim(-0.02, 1.02)
    save(fig, "fig3_hold.png")


def fig3(v):
    s7 = v["J2"]["secondary"]["S7_boundary"]
    cs = [1, 3, 9]
    ideal = [s7[str(c)]["ideal_threshold"] for c in cs]
    jev = [s7[str(c)]["crossing_pstar"] for c in cs]
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    x = range(3)
    ax.plot(x, ideal, color=BLUE, lw=2, marker="o", ms=8, mec=SURF, mew=1.5, label="ideal: act only above c/(1+c)")
    ax.plot(x, jev, color=ORANGE, lw=2, marker="s", ms=8, mec=SURF, mew=1.5, label="Jev: where it switches to acting")
    for i in x:
        ax.annotate(f"{ideal[i]:.2f}", (i, ideal[i]), xytext=(0, 9), textcoords="offset points", fontsize=8.5,
                    color=INK2, ha="center")
        ax.annotate(f"{jev[i]:.2f}", (i, jev[i]), xytext=(0, -16), textcoords="offset points", fontsize=8.5,
                    color=INK2, ha="center")
    ax.set_xticks(list(x), [f"{c}×" for c in cs])
    ax.set_xlim(-0.3, 2.5)
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("cost of a wrong call (right call earns 1)")
    ax.set_ylabel("certainty needed before acting")
    ax.set_title("J2 · The bar for acting barely rises with the stakes")
    ax.legend(loc="upper left", fontsize=8.5)
    save(fig, "fig4_threshold.png")


def fig4():
    fits = json.loads(FITS.read_text())
    fams = [("F1", "F1 machines", False, "effect size d  (easier →)", [0.5, 1, 2, 4]),
            ("F2", "F2 dictionary codes", True, "noise s  (easier →)", [2, 1, 0.5, 0.25]),
            ("F3", "F3 lanes", False, "mean sensor reliability  (easier →)", [0.5, 0.6, 0.7, 0.8, 0.9])]
    pres = [("A0", BLUE, "clean"), ("A1-standard", ORANGE, "messy (standard)"), ("A1-strong", AQUA, "messy (strong)")]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5), sharey=True)
    for ax, (fam, title, invert, xlabel, ticks) in zip(axes, fams):
        for p, col, name in pres:
            tab = sorted((float(k), v[1]) for k, v in fits[fam][p]["table"].items())
            xs, ys = zip(*tab)
            ax.plot(xs, ys, color=col, lw=2, marker="o", ms=4.5, label=name)
        ax.axhline(0.25, ls=":", lw=1, color=MUTED)
        ax.axhline(0.75, ls="--", lw=0.8, color=MUTED)
        if fam in ("F1", "F2"):
            ax.set_xscale("log")
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        if invert:
            ax.invert_xaxis()
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylim(0, 1.03)
    axes[0].set_ylabel("accuracy (40 stimuli per point)")
    axes[0].text(axes[0].get_xlim()[0], 0.765, " 75%", color=MUTED, fontsize=8)
    axes[0].text(axes[0].get_xlim()[0], 0.265, " chance", color=MUTED, fontsize=8)
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Titration pilot · which layouts slow Jev down", x=0.01, ha="left", fontsize=10.5,
                 fontweight="bold", color=INK)
    save(fig, "fig1_pilot.png")


if __name__ == "__main__":
    v = json.loads(FLIGHT.read_text())
    fig1(v)
    fig2(v)
    fig3(v)
    fig4()
