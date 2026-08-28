#!/usr/bin/env python
"""Qualitative box-overlay figure from a set of logged episode trajectories.

Per panel, on the real VOC image:
  * ground-truth box            -- solid green
  * intermediate search boxes   -- faint grey, fading in over steps
  * final box                   -- bold, green if a triggered success, red if a failure

Successes (left column) and failures (right column) are whichever episodes are recorded in
--results/qual_boxes.json.

NOTE: this script expects a `qual_boxes.json` file with per-episode box trajectories
(gt_boxes, boxes, triggered, last_iou, cls_true, n_steps, name) already selected and
written to disk. That selection step is not included in this release; see the note in
README.md if you need to regenerate it.

Usage:
    python scripts/make_box_fig.py --voc ./data/VOC2012_train_val \
        --results ./results/results_final --out ./paper_assets
"""
import argparse
import json
import logging
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({"font.size": 9, "font.family": "serif",
                      "savefig.dpi": 300, "savefig.bbox": "tight"})
GREEN, RED, GT = "#2a9d54", "#c0392b", "#1a7a1a"


def draw(ax, img, ep):
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    ok = ep["triggered"] and ep["last_iou"] >= 0.5
    for g in ep["gt_boxes"]:
        ax.add_patch(Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1],
                     fill=False, edgecolor=GT, lw=2.2, ls="-"))
    bxs = ep["boxes"]; n = len(bxs)
    for i, b in enumerate(bxs[:-1]):
        al = 0.15 + 0.5 * (i / max(n - 1, 1))
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                     fill=False, edgecolor="0.4", lw=0.8, alpha=al))
    fb = bxs[-1]
    ax.add_patch(Rectangle((fb[0], fb[1]), fb[2] - fb[0], fb[3] - fb[1],
                 fill=False, edgecolor=(GREEN if ok else RED), lw=2.6))
    tag = "triggered" if ep["triggered"] else "exhausted"
    ax.set_title(f"{ep['cls_true']} — {tag}, IoU {ep['last_iou']:.2f}, {ep['n_steps']} steps",
                 fontsize=8, color=(GREEN if ok else RED))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voc", required=True)
    ap.add_argument("--results", default="./results/results_final")
    ap.add_argument("--out", default="./paper_assets")
    a = ap.parse_args()
    eps = json.load(open(f"{a.results}/qual_boxes.json"))
    succ = [e for e in eps if e["triggered"] and e["last_iou"] >= 0.5]
    fail = [e for e in eps if not (e["triggered"] and e["last_iou"] >= 0.5)]
    rows = max(len(succ), len(fail))
    fig, axs = plt.subplots(rows, 2, figsize=(7.2, 3.4 * rows))
    if rows == 1:
        axs = axs.reshape(1, 2)
    for r in range(rows):
        for c, pool in ((0, succ), (1, fail)):
            ax = axs[r, c]
            if r < len(pool):
                e = pool[r]
                img = Image.open(os.path.join(a.voc, "JPEGImages", e["name"] + ".jpg")).convert("RGB")
                draw(ax, np.array(img), e)
            else:
                ax.axis("off")
    axs[0, 0].text(0.5, 1.22, "Well-localized (triggered)", transform=axs[0, 0].transAxes,
                   ha="center", fontweight="bold", color=GREEN)
    axs[0, 1].text(0.5, 1.22, "Failed (small objects)", transform=axs[0, 1].transAxes,
                   ha="center", fontweight="bold", color=RED)
    fig.tight_layout()
    fig.savefig(f"{a.out}/fig_qualitative_boxes.pdf")
    fig.savefig(f"{a.out}/fig_qualitative_boxes.png")
    logger.info("wrote %s/fig_qualitative_boxes.png (green=GT, bold=final, faint grey=search path)",
                a.out)


if __name__ == "__main__":
    main()
