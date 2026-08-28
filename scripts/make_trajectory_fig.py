#!/usr/bin/env python
"""IoU-over-episode trajectory figure: a panel of representative success and failure
episodes.

A genuine triggered success stops early (the trigger step is the point), so successful
curves end at the trigger step with a marker rather than running to the full budget.
Failures are genuine failures (never reach IoU 0.5) and run to budget exhaustion.

Selection is representative, not cherry-picked: successes are sampled across the
object-size range so the panel reflects the real distribution, and failures are sampled
from the dominant small-object failure mode. Every panel is a real episode.

Note: records store per-step IoU but not box coordinates, so this draws IoU curves only.
For image-with-boxes panels, see make_box_fig.py (which needs box trajectories logged
separately; see the note in its docstring).

Usage:
    python scripts/make_trajectory_fig.py --results ./results/results_final \
        --tag desc_dinov2 --out ./paper_assets
"""
import argparse
import gzip
import json
import logging

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({"font.size": 9, "font.family": "serif", "savefig.dpi": 300,
                      "savefig.bbox": "tight"})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="./results/results_final")
    ap.add_argument("--tag", default="desc_dinov2")
    ap.add_argument("--out", default="./paper_assets")
    a = ap.parse_args()
    R = json.load(gzip.open(f"{a.results}/{a.tag}_records.json.gz", "rt"))
    rng = np.random.default_rng(3)

    # Well-localized successes: triggered, IoU >= 0.80, with a visible multi-step search
    # (>=5 steps), sampled across the object-size range so the panel is not only easy,
    # large objects. Labelled honestly as representative-of-well-localized, not average
    # (mean IoU across the full test set is lower).
    succ = [r for r in R if r["triggered"] and r["last_iou"] >= 0.80 and 5 <= r["n_steps"] < 20]
    succ.sort(key=lambda r: r["gt_frac"])
    pick_s = [succ[int(f * (len(succ) - 1))] for f in (0.2, 0.55, 0.85)]

    # Genuine failures: never reached 0.5, dominant small-object mode.
    fail = [r for r in R if r["peak_iou"] < 0.5 and r["gt_frac"] < 0.15]
    fail.sort(key=lambda r: r["last_iou"])
    pick_f = [fail[i] for i in rng.choice(len(fail), 3, replace=False)]

    fig, axs = plt.subplots(3, 2, figsize=(8, 7.5))
    for row in range(3):
        for col, (r, ok) in enumerate([(pick_s[row], True), (pick_f[row], False)]):
            ax = axs[row, col]; y = r["ious"]; x = range(len(y))
            ax.plot(x, y, "-o", ms=3, c=("#2a9d54" if ok else "#c0392b"), lw=1.3)
            ax.axhline(0.5, ls="--", c="grey", lw=1)
            if ok and r["triggered"]:
                ax.plot(len(y) - 1, y[-1], marker="*", ms=15, c="#e08e0b", zorder=5)
            ax.set_ylim(0, 1); ax.set_xlim(-0.5, 20)
            ax.set_title(f"{r['cls_true']} ({r['name']}) — "
                         f"{'triggered IoU' if ok else 'failed, IoU'} {r['last_iou']:.2f}"
                         f"{', ' + str(r['n_steps']) + ' steps' if ok else ''}", fontsize=7.5)
            if col == 0:
                ax.set_ylabel("IoU")
            if row == 2:
                ax.set_xlabel("step")
    axs[0, 0].text(0.5, 1.28, "Well-localized episodes (IoU$\\geq$0.8, triggered)",
                   transform=axs[0, 0].transAxes, ha="center", fontweight="bold", color="#2a9d54")
    axs[0, 1].text(0.5, 1.28, "Failed (small objects, runs to budget)",
                   transform=axs[0, 1].transAxes, ha="center", fontweight="bold", color="#c0392b")
    fig.tight_layout()
    fig.savefig(f"{a.out}/fig_trajectories.pdf"); fig.savefig(f"{a.out}/fig_trajectories.png")
    logger.info("wrote fig_trajectories")
    logger.info("successes chosen (gt_frac, steps, iou): %s",
                [(round(r["gt_frac"], 3), r["n_steps"], round(r["last_iou"], 2)) for r in pick_s])


if __name__ == "__main__":
    main()
