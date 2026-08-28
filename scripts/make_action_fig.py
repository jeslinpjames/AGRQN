#!/usr/bin/env python
"""Action-analysis figure: overall action distribution and coarse-to-fine action share over
the course of an episode.

Uses triggered episodes only for the temporal-share panel, since a genuinely triggered
success stops early (the trigger step is the point of interest), unlike an exhausted
episode which always runs the full budget.

Usage:
    python scripts/make_action_fig.py --results ./results/results_final \
        --tag desc_dinov2 --out ./paper_assets
"""
import argparse
import gzip
import json
import logging
from collections import Counter

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({"font.size": 10, "font.family": "serif", "savefig.dpi": 300,
                      "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3})
ANAMES = ["right", "left", "up", "down", "big", "small", "fat", "tall", "TRIG"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="./results/results_final")
    ap.add_argument("--tag", default="desc_dinov2")
    ap.add_argument("--out", default="./paper_assets")
    a = ap.parse_args()
    R = json.load(gzip.open(f"{a.results}/{a.tag}_records.json.gz", "rt"))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))

    # (a) overall action distribution
    allacts = [x for r in R for x in r["actions"]]
    cnt = Counter(allacts); tot = sum(cnt.values())
    freq = [cnt.get(i + 1, 0) / tot for i in range(9)]
    cols = ["#2b6cb0"] * 8 + ["#c0392b"]
    ax[0].bar(ANAMES, freq, color=cols, edgecolor="black", linewidth=0.5)
    for i, f in enumerate(freq):
        ax[0].annotate(f"{f:.2f}", (i, f), ha="center", va="bottom", fontsize=7)
    ax[0].set_ylabel("frequency"); ax[0].tick_params(axis="x", labelsize=7, rotation=45)
    ax[0].set_title("(a) Action distribution")

    # (b) action share vs. normalized step index, triggered episodes only
    trg = [r for r in R if r["triggered"] and r["n_steps"] >= 3]
    B = 10; share = np.zeros((B, 9))
    for r in trg:
        n = len(r["actions"])
        for i, act in enumerate(r["actions"]):
            share[min(int(i / n * B), B - 1), act - 1] += 1
    share = share / share.sum(1, keepdims=True)
    bottom = np.zeros(B); x = np.linspace(0, 1, B)
    C = cm.tab10(np.linspace(0, 1, 9))
    for k in range(9):
        ax[1].fill_between(x, bottom, bottom + share[:, k], label=ANAMES[k], color=C[k], alpha=.85)
        bottom += share[:, k]
    ax[1].set_xlabel("normalized step index"); ax[1].set_ylabel("action share")
    ax[1].set_ylim(0, 1); ax[1].set_xlim(0, 1)
    ax[1].legend(fontsize=6, ncol=2, loc="upper center", frameon=False)
    ax[1].set_title("(b) Coarse-to-fine: trigger concentrates at the end")

    fig.tight_layout()
    fig.savefig(f"{a.out}/fig_actions.pdf"); fig.savefig(f"{a.out}/fig_actions.png")
    logger.info("wrote fig_actions")
    logger.info("trigger share in final decile: %.3f", share[-1, 8])
    logger.info("contraction (small) share overall: %.3f | down: %.3f", freq[5], freq[3])


if __name__ == "__main__":
    main()
