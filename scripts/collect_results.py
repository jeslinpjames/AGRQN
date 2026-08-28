#!/usr/bin/env python
"""Summarize every finished training run into tables and figures.

Produces, under --report:
    summary.csv          one row per run
    summary.md            the same, readable in a terminal
    summary.tex            LaTeX table
    fig_exhaustion.png    exhaustion rate vs. failure rate, across all runs
    fig_tradeoff.png      exhaustion vs. trigger precision
    fig_curves.png        per-epoch training curves for every run
    fig_accuracy.png      mIoU and CorLoc per run

Usage:
    python scripts/collect_results.py --out ./results --report ./report
"""
import argparse
import glob
import json
import logging
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

TRACK_COLOR = {"accuracy": "tab:blue", "incentive": "tab:red"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="./results")
    ap.add_argument("--report", default="./report")
    a = ap.parse_args()
    os.makedirs(a.report, exist_ok=True)

    rows, hists = [], {}
    for p in sorted(glob.glob(os.path.join(a.out, "*_test.json"))):
        r = json.load(open(p))
        nm = r["config"]
        hists[nm] = r["history"]
        for tag in ("best", "final"):
            if f"test_{tag}" not in r:
                continue
            b = r[f"test_{tag}"]["last_box"]
            c = r[f"test_{tag}"]["conf_box"]
            rows.append(dict(
                config=nm, ckpt=tag, track=r["cfg"].get("track", "?"),
                mIoU=b["miou"], CorLoc=b["acc05"],
                mIoU_confbox=c["miou"], CorLoc_confbox=c["acc05"],
                exhaust=b["exhaust_rate"], fail_rate=1 - b["acc05"],
                gap=b["exh_minus_failrate"],
                trigger_ppv=b["trigger_ppv"], exhaust_ppv=b["exhaust_ppv"],
                steps=b["mean_steps"], epochs=len(r["history"]),
                params_M=r["params_M"], minutes=r["train_minutes"]))

    if not rows:
        raise SystemExit(f"no *_test.json found in {a.out}/ -- has anything finished?")

    df = pd.DataFrame(rows).sort_values(["track", "config", "ckpt"])
    df.to_csv(os.path.join(a.report, "summary.csv"), index=False)

    show = df[df.ckpt == "best"].copy()

    with open(os.path.join(a.report, "summary.md"), "w") as f:
        f.write("# AG-RQN sweep -- results\n\n")
        f.write("Trained on trainval minus the held-out test set; evaluated once on the "
                "200-image/class test set (see the leak-free split in `agrqn/core.py`).\n\n")
        f.write("## Best-checkpoint results\n\n")
        f.write(show.drop(columns=["ckpt"]).to_markdown(index=False, floatfmt=".3f"))
        f.write("\n\n## The `gap` column\n\n")
        f.write("`gap` = exhaust - (1 - CorLoc). If `gap` is near 0, the agent triggers "
                "exactly when it succeeds, and the exhaustion rate is the failure rate -- a "
                "calibrated rejection signal, not a stopping bug. A large negative `gap` "
                "means the agent was forced to commit to boxes it knew were bad: check "
                "whether `trigger_ppv` also collapsed in that row.\n\n")
        f.write("## All checkpoints\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n")

    with open(os.path.join(a.report, "summary.tex"), "w") as f:
        f.write(show[["config", "mIoU", "CorLoc", "exhaust", "trigger_ppv", "steps"]]
                .to_latex(index=False, float_format="%.3f"))

    # -------------------------------------------------- fig 1: exhaustion vs. failure rate
    fig, ax = plt.subplots(figsize=(8, 5.5))
    lim = [0, max(show.exhaust.max(), show.fail_rate.max()) * 1.15 + 0.02]
    ax.plot(lim, lim, "k--", lw=1, label="exhaustion = failure rate")
    for _, r in show.iterrows():
        ax.scatter(r.fail_rate, r.exhaust, s=90,
                   c=TRACK_COLOR.get(r.track, "grey"), zorder=3)
        ax.annotate(r.config, (r.fail_rate, r.exhaust), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("failure rate  (1 - CorLoc)")
    ax.set_ylabel("budget-exhaustion rate")
    ax.set_title("Exhaustion tracks the failure rate.\n"
                 "Points on the line: the agent triggers iff it succeeds (blue = accuracy track).\n"
                 "Points below the line: forced to commit to boxes it knew were bad (red).")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.report, "fig_exhaustion.png"), dpi=200)

    # -------------------------------------------------- fig 2: exhaustion vs. precision
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for _, r in show.iterrows():
        ax.scatter(r.exhaust, r.trigger_ppv, s=90,
                   c=TRACK_COLOR.get(r.track, "grey"), zorder=3)
        ax.annotate(r.config, (r.exhaust, r.trigger_ppv), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("budget-exhaustion rate (lower is better)")
    ax.set_ylabel("trigger precision (PPV)")
    ax.set_title("Cutting exhaustion by force costs precision.\n"
                 "Only the accuracy track moves left without moving down.")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.report, "fig_tradeoff.png"), dpi=200)

    # -------------------------------------------------- fig 3: training curves
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    for nm, h in sorted(hists.items()):
        ep = [x["epoch"] for x in h]
        axs[0, 0].plot(ep, [x["miou"] for x in h], label=nm)
        axs[0, 1].plot(ep, [x["exhaust_rate"] for x in h], label=nm)
        axs[1, 0].plot(ep, [x["trigger_ppv"] for x in h], label=nm)
        axs[1, 1].plot(ep, [x["exh_minus_failrate"] for x in h], label=nm)
    for ax, t in zip(axs.ravel(), ["validation mIoU", "exhaustion rate",
                                   "trigger PPV", "exhaust  -  failure rate"]):
        ax.set_xlabel("epoch"); ax.set_title(t); ax.grid(alpha=0.3)
    axs[1, 1].axhline(0, c="k", ls="--", lw=1)
    axs[0, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(a.report, "fig_curves.png"), dpi=200)

    # -------------------------------------------------- fig 4: accuracy by configuration
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(show)); w = 0.38
    ax.bar(x - w / 2, show.mIoU, w, label="mIoU")
    ax.bar(x + w / 2, show.CorLoc, w, label="CorLoc (Acc@0.5)")
    ax.set_xticks(x); ax.set_xticklabels(show.config, rotation=30, ha="right")
    ax.legend(); ax.grid(alpha=0.3, axis="y"); ax.set_title("Accuracy by configuration")
    fig.tight_layout(); fig.savefig(os.path.join(a.report, "fig_accuracy.png"), dpi=200)

    logger.info("\n%s", show[["config", "track", "mIoU", "CorLoc", "exhaust", "fail_rate",
                              "gap", "trigger_ppv", "steps"]].to_string(index=False,
                                                                         float_format="%.3f"))
    logger.info("written to %s/  (summary.csv, summary.md, summary.tex, 4 figures)", a.report)


if __name__ == "__main__":
    main()
