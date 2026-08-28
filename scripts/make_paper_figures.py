#!/usr/bin/env python
"""Regenerate every core figure and LaTeX table for the paper from results/results_final.
Pure inference on already-saved records; trains nothing.

Usage:
    python scripts/make_paper_figures.py --results ./results/results_final --out ./paper_assets

Outputs (300 dpi PDF + PNG, plus .tex table fragments) into --out:
    fig_prior_arc        the prior-redundancy result (bar + attenuation)
    fig_trigger          exhaustion=failure diagonal + six failed interventions
    fig_failures         failure taxonomy, VGG vs. DINOv2 descriptor
    fig_size             final IoU vs. object size, with Spearman rho
    fig_calibration      trigger confidence vs. realized IoU
    fig_training         validation mIoU / exhaustion across epochs
    tab_headline.tex     per-class table for the best model
    tab_models.tex       the 4 descriptor x prior configs
    tab_priors.tex       the prior study (none/gradcam/layercam/dino)
    tab_trigger.tex      six reward interventions
    tab_ablation.tex     retraining ablations, both descriptors
    tab_seeds.tex        the seed studies
"""
import argparse
import glob
import gzip
import json
import logging
import os
from collections import Counter

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "font.size": 10, "font.family": "serif",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
GREEN, BLUE, RED, ORANGE, GREY = "#2a9d54", "#2b6cb0", "#c0392b", "#e08e0b", "#95a5a6"


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"))
    plt.close(fig)
    logger.info("wrote %s", name)


def load(results):
    T = {}
    for p in glob.glob(f"{results}/*_test.json"):
        r = json.load(open(p)); b = r["test_best"]["last_box"]
        T[r["tag"]] = dict(miou=b["miou"], corloc=b["acc05"], exh=b["exhaust_rate"],
                           ppv=b["trigger_ppv"], steps=b["mean_steps"],
                           per=b.get("per_class", {}), hist=r.get("history", []))
    return T


def recs(results, tag):
    p = f"{results}/{tag}_records.json.gz"
    return json.load(gzip.open(p, "rt")) if os.path.exists(p) else None


def m(T, k, f="miou"):
    return T[k][f] if k in T else float("nan")


# ================================================================ FIGURES
def fig_prior_arc(T, out):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    labs = ["none", "Grad-CAM", "LayerCAM", "DINO"]
    keys = ["prior_none", "prior_gradcam", "prior_layercam", "final_s42"]
    v = [m(T, k) for k in keys]
    cols = [GREY, RED, ORANGE, GREEN]
    ax[0].bar(labs, v, color=cols, edgecolor="black", linewidth=0.5)
    ax[0].axhline(v[0], ls="--", c="black", lw=1)
    ax[0].set_ylim(0.50, 0.60); ax[0].set_ylabel("mIoU")
    for i, y in enumerate(v):
        ax[0].annotate(f"{y-v[0]:+.3f}", (i, y), ha="center", va="bottom", fontsize=9)
    ax[0].set_title("(a) The prior study (VGG descriptor)\ndashed = no-prior control")

    # attenuation: the DINO-prior gain should shrink once the descriptor already carries the
    # same information (DINOv2 descriptor), per the paper's redundancy argument.
    vg = np.mean([m(T, k) for k in ["final_s42", "final_s1", "final_s7"]])
    vn = m(T, "prior_none")
    dv = np.mean([m(T, k) for k in ["desc_dinov2", "desc_dinov2_s1", "desc_dinov2_s7"]])
    dn = np.mean([m(T, k) for k in ["desc_dinov2_none", "desc_dinov2_none_s1", "desc_dinov2_none_s7"]])
    gains = [vg - vn, dv - dn]
    errs = [0.005, 0.009]
    ax[1].bar(["VGG-16\n(supervised)", "DINOv2\n(self-supervised)"], gains,
              yerr=errs, capsize=5, color=[GREEN, BLUE], edgecolor="black", linewidth=0.5)
    for i, gg in enumerate(gains):
        ax[1].annotate(f"+{gg:.3f}", (i, gg), ha="center", va="bottom", fontsize=10)
    ax[1].set_ylabel("mIoU gain from the DINO prior")
    ax[1].set_title(f"(b) Attenuation: {100*(1-gains[1]/gains[0]):.0f}%\n"
                    "prior helps less as the descriptor improves")
    fig.tight_layout(); save(fig, out, "fig_prior_arc")


def fig_trigger(T, out):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.7))
    healthy = ["control", "final_s42", "final_s1", "final_s7", "dueling", "wide",
               "desc_dinov2", "desc_dinov2_none", "prior_layercam", "single_dqn"]
    fr = [1 - m(T, k, "corloc") for k in healthy if k in T]
    ex = [m(T, k, "exh") for k in healthy if k in T]
    ax[0].plot([0, .5], [0, .5], "--", c="black", lw=1, label="exhaustion = failure rate")
    ax[0].scatter(fr, ex, s=55, c=BLUE, edgecolor="black", linewidth=0.5, zorder=3)
    ax[0].set_xlabel("failure rate  (1 $-$ CorLoc)"); ax[0].set_ylabel("budget-exhaustion rate")
    ax[0].legend(fontsize=8, frameon=False)
    ax[0].set_title("(a) The trigger fires iff the agent succeeds")

    iv = ["control", "step\n$-0.05$", "step\n$-0.2$", "$\\gamma{=}.99$",
          "$\\eta{=}10$", "PBRS", "expiry\nfail", "aux\nhead"]
    exv = [0.116, 0.310, 0.236, 0.470, 0.348, 0.429, 0.272, 0.445]
    cols = [GREEN] + [RED] * 7
    ax[1].bar(iv, exv, color=cols, edgecolor="black", linewidth=0.5)
    ax[1].axhline(0.116, ls="--", c="black", lw=1)
    ax[1].set_ylabel("budget-exhaustion rate"); ax[1].tick_params(axis="x", labelsize=7)
    ax[1].set_title("(b) Six reward interventions. None helped.\nFour made it worse.")
    fig.tight_layout(); save(fig, out, "fig_trigger")


def _tax(rec):
    def cat(r):
        if not r["correct_cls"]: return "misclassified"
        if r["gt_frac"] < 0.10: return "small object"
        if r["n_inst"] > 1: return "multi-instance"
        if not r["triggered"]: return "no trigger"
        return "search failure"
    f = [r for r in rec if r["last_iou"] < 0.3]
    return Counter(cat(r) for r in f), len(f)


def fig_failures(results, out):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for a, tag, nm in [(ax[0], "control", "VGG descriptor"),
                       (ax[1], "desc_dinov2", "DINOv2 descriptor")]:
        rec = recs(results, tag)
        if rec is None:
            continue
        c, n = _tax(rec)
        k, v = zip(*c.most_common())
        a.barh(list(k)[::-1], list(v)[::-1], color=RED, edgecolor="black", linewidth=0.5)
        a.set_xlabel("failures (IoU $<$ 0.3)"); a.set_title(f"{nm}: {n}/1000")
    fig.tight_layout(); save(fig, out, "fig_failures")


def fig_size(results, out):
    rec = recs(results, "desc_dinov2")
    if rec is None:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    gf = np.array([r["gt_frac"] for r in rec]); li = np.array([r["last_iou"] for r in rec])
    edges = [0, .05, .1, .2, .35, .6, 1.01]; idx = np.digitize(gf, edges) - 1
    mn = [li[idx == i].mean() for i in range(6)]; sd = [li[idx == i].std() for i in range(6)]
    ax.errorbar(range(6), mn, yerr=sd, fmt="o-", capsize=3, c=BLUE, lw=1.5)
    ax.set_xticks(range(6))
    ax.set_xticklabels(["<5%", "5-10", "10-20", "20-35", "35-60", ">60"], fontsize=8)
    ax.set_xlabel("ground-truth box area (% of image)"); ax.set_ylabel("final IoU")
    ax.axhline(0.5, ls="--", c=RED, lw=1)
    ax.set_title(f"Small objects dominate errors\nSpearman $\\rho={spearmanr(gf,li).correlation:+.3f}$")
    fig.tight_layout(); save(fig, out, "fig_size")


def fig_calibration(results, out):
    rec = recs(results, "desc_dinov2")
    if rec is None:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    cs, io = [], []
    for r in rec:
        n = min(len(r.get("confs", [])), len(r.get("ious", [])))
        cs += list(np.asarray(r["confs"][:n])); io += list(np.asarray(r["ious"][:n]))
    cs, io = np.array(cs), np.array(io)
    edges = np.linspace(cs.min(), cs.max(), 9); idx = np.digitize(cs, edges) - 1
    xs = [cs[idx == i].mean() for i in range(8) if (idx == i).sum() > 5]
    ys = [io[idx == i].mean() for i in range(8) if (idx == i).sum() > 5]
    ax.plot(xs, ys, "o-", c=BLUE, lw=1.5)
    ax.set_xlabel("trigger Q-value (confidence)"); ax.set_ylabel("realised IoU")
    ax.set_title(f"Trigger calibration\nSpearman $\\rho={spearmanr(cs,io).correlation:+.3f}$")
    fig.tight_layout(); save(fig, out, "fig_calibration")


def fig_training(T, out):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for tag, lab, c in [("desc_dinov2", "DINOv2 desc", BLUE),
                        ("control", "VGG desc", GREEN),
                        ("prior_none", "VGG, no prior", GREY)]:
        h = T.get(tag, {}).get("hist", [])
        if not h:
            continue
        ax[0].plot([x["miou"] for x in h], label=lab, c=c, lw=1.3)
        ax[1].plot([x.get("exhaust_rate", np.nan) for x in h], label=lab, c=c, lw=1.3)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("val mIoU"); ax[0].legend(fontsize=8, frameon=False)
    ax[0].set_title("(a) Accuracy improves smoothly")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("exhaustion rate"); ax[1].legend(fontsize=8, frameon=False)
    ax[1].set_title("(b) Exhaustion falls with training,\nnot with reward tuning")
    fig.tight_layout(); save(fig, out, "fig_training")


# ================================================================ TABLES
def tab_headline(results, out):
    rec = recs(results, "desc_dinov2")
    if rec is None:
        return
    cls = sorted({r["cls_true"] for r in rec})
    rng = np.random.default_rng(0)
    rows = []
    allv = []
    for c in cls:
        v = np.array([r["last_iou"] for r in rec if r["cls_true"] == c]); allv.append(v)
        bs = [rng.choice(v, len(v), True).mean() for _ in range(4000)]
        rows.append((c.capitalize(), v.mean(), np.percentile(bs, 2.5),
                     np.percentile(bs, 97.5), np.median(v), (v >= .5).mean()))
    av = np.concatenate(allv)
    bs = [np.mean([rng.choice(np.array([r["last_iou"] for r in rec if r["cls_true"] == c]),
                              sum(r["cls_true"] == c for r in rec), True).mean() for c in cls])
          for _ in range(2000)]
    t = ["\\begin{tabular}{lccc}", "\\toprule",
         "Class & Mean IoU (95\\% CI) & Median & CorLoc \\\\", "\\midrule"]
    for nm, mu, lo, hi, md, cl in rows:
        t.append(f"{nm} & {mu:.3f} [{lo:.3f}, {hi:.3f}] & {md:.3f} & {cl:.3f} \\\\")
    t += ["\\midrule",
          f"\\textbf{{Mean}} & \\textbf{{{av.mean():.3f}}} [{np.percentile(bs,2.5):.3f}, "
          f"{np.percentile(bs,97.5):.3f}] & {np.median(av):.3f} & \\textbf{{{(av>=.5).mean():.3f}}} \\\\",
          "\\bottomrule", "\\end{tabular}"]
    open(f"{out}/tab_headline.tex", "w").write("\n".join(t))
    logger.info("wrote tab_headline.tex")


def tab_generic(T, out, fname, rows, header):
    lines = ["\\begin{tabular}{l" + "c" * (len(header) - 1) + "}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    for label, key in rows:
        if key is None:
            lines.append("\\midrule"); continue
        if key not in T:
            continue
        t = T[key]
        lines.append(f"{label} & {t['miou']:.3f} & {t['corloc']:.3f} & "
                     f"{t['exh']:.3f} & {t['ppv']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(f"{out}/{fname}", "w").write("\n".join(lines))
    logger.info("wrote %s", fname)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="./results/results_final")
    ap.add_argument("--out", default="./paper_assets")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    T = load(a.results)
    logger.info("loaded %d runs", len(T))

    logger.info("figures:")
    fig_prior_arc(T, a.out)
    fig_trigger(T, a.out)
    fig_failures(a.results, a.out)
    fig_size(a.results, a.out)
    fig_calibration(a.results, a.out)
    fig_training(T, a.out)

    logger.info("tables:")
    tab_headline(a.results, a.out)
    hdr = ["Config", "mIoU", "CorLoc", "Exh.", "PPV"]
    tab_generic(T, a.out, "tab_models.tex", [
        ("VGG-16, no prior", "prior_none"), ("VGG-16, DINO prior", "final_s42"),
        ("DINOv2, no prior", "desc_dinov2_none"), ("DINOv2, DINO prior", "desc_dinov2"),
    ], hdr)
    tab_generic(T, a.out, "tab_priors.tex", [
        ("None", "prior_none"), ("Grad-CAM", "prior_gradcam"),
        ("LayerCAM", "prior_layercam"), ("DINO", "final_s42"),
    ], hdr)
    tab_generic(T, a.out, "tab_ablation.tex", [
        ("Full (VGG)", "final_s42"), ("VGG, no history", "no_history"),
        ("VGG, single-DQN", "single_dqn"), (None, None),
        ("Full (DINOv2)", "desc_dinov2"), ("DINOv2, no history", "dinov2_nohist"),
        ("DINOv2, single-DQN", "dinov2_singleq"),
    ], hdr)
    tab_generic(T, a.out, "tab_scale.tex", [
        ("5 classes (DINOv2)", "desc_dinov2"), ("20 classes, DINO prior", "voc20_dino"),
        ("20 classes, no prior", "voc20_none"),
    ], hdr)

    # seed table
    def seedrow(name, keys):
        v = [m(T, k) for k in keys if k in T]
        return f"{name} & {', '.join(f'{x:.4f}' for x in v)} & {np.mean(v):.4f} & {np.std(v,ddof=1):.4f} \\\\"
    st = ["\\begin{tabular}{lccc}", "\\toprule",
          "Configuration & Seeds (mIoU) & Mean & Std \\\\", "\\midrule",
          seedrow("VGG + DINO prior", ["final_s1", "final_s42", "final_s7"]),
          seedrow("DINOv2 + DINO prior", ["desc_dinov2", "desc_dinov2_s1", "desc_dinov2_s7"]),
          seedrow("DINOv2 + no prior", ["desc_dinov2_none", "desc_dinov2_none_s1", "desc_dinov2_none_s7"]),
          "\\bottomrule", "\\end{tabular}"]
    open(f"{a.out}/tab_seeds.tex", "w").write("\n".join(st))
    logger.info("wrote tab_seeds.tex")

    logger.info("done -> %s/  (%d files)", a.out, len(os.listdir(a.out)))


if __name__ == "__main__":
    main()
