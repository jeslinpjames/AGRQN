#!/usr/bin/env python
"""Statistical tests for the "exhaustion rate == failure rate" calibration claim.

Runs entirely on CPU from an existing *_records.json.gz (written by train_agrqn.py or
reevaluate_checkpoints.py) -- no retraining, no GPU. Each record needs `cls_true`,
`triggered`, `final_iou`, and the per-step trajectories `ious` and `confs`:

  Test A (permutation test on the PPV gap)      -- triggered, final_iou
  Test B (Brier resolution/reliability)         -- triggered, final_iou
  Test C (per-class deviation + bootstrap)      -- cls_true, triggered, final_iou
  Test D (fixed-schedule stopping control)      -- per-step `ious`
  Test E (risk-coverage / AURC)                 -- per-step `confs`

Only reads the given records file and writes to --out; does not touch training outputs or
checkpoints.

Usage:
    python scripts/analyze_calibration.py \
        --records ./results/results_final/duel_wide_records.json.gz \
        --out ./results/results_calib/duel_wide
    python scripts/analyze_calibration.py \
        --records ./results/results_20class/voc20_none_20cls_records.json.gz \
        --iou-key final_iou_oracle --out ./results/results_calib/voc20_none_oracle
"""
import argparse
import gzip
import json
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

RNG = np.random.default_rng(0)
TAU = 0.5
B_PERM = 10000
B_BOOT = 10000


# --------------------------------------------------------------------------- I/O
def load_records(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


# ----------------------------------------------------------------- shared quantities
def get_success(recs, iou_key):
    return np.array([r[iou_key] >= TAU for r in recs])


def get_trig(recs):
    return np.array([bool(r["triggered"]) for r in recs])


def ppv_gap(recs, iou_key):
    s, t = get_success(recs, iou_key), get_trig(recs)
    ppv_trig = s[t].mean() if t.any() else float("nan")
    ppv_exh = s[~t].mean() if (~t).any() else float("nan")
    return ppv_trig - ppv_exh, ppv_trig, ppv_exh


# --------------------------------------------------------------- Test A: permutation
def test_A(recs, iou_key, B=B_PERM):
    """Permutation test: is the observed PPV gap larger than would arise if trigger/exhaust
    labels were assigned at random (holding the trigger rate fixed)?"""
    s, t = get_success(recs, iou_key), get_trig(recs)
    obs = s[t].mean() - s[~t].mean()
    n_trig = int(t.sum())
    n = t.size
    null = np.empty(B)
    for b in range(B):
        perm = RNG.permutation(n)
        tt = np.zeros(n, bool)
        tt[perm[:n_trig]] = True
        null[b] = s[tt].mean() - s[~tt].mean()
    p = (np.sum(null >= obs) + 1) / (B + 1)
    return dict(observed_gap=float(obs), p_value=float(p), null=null)


# ------------------------------------------------------------- Test B: Brier decomposition
def test_B(recs, iou_key, B=B_BOOT):
    s, t = get_success(recs, iou_key), get_trig(recs)

    def decomp(ss, tt):
        obar = ss.mean()
        UNC = obar * (1 - obar)
        REL = RES = 0.0
        for grp in (tt, ~tt):
            if grp.any():
                n_frac = grp.mean()
                o_k = ss[grp].mean()
                f_k = o_k  # forecast = group rate (two-bin decomposition)
                RES += n_frac * (o_k - obar) ** 2
                REL += n_frac * (f_k - o_k) ** 2
        return dict(REL=REL, RES=RES, UNC=UNC,
                     skill=(RES / UNC if UNC > 0 else float("nan")),
                     BS=REL - RES + UNC)

    point = decomp(s, t)
    n = s.size
    boot_res = np.empty(B)
    for b in range(B):
        idx = RNG.integers(0, n, n)
        boot_res[b] = decomp(s[idx], t[idx])["RES"]
    ci = np.percentile(boot_res, [2.5, 97.5])
    point["RES_ci"] = ci.tolist()
    point["RES_excludes_zero"] = bool(ci[0] > 0)
    return point


# ----------------------------------------------------- Test C: per-class vs random null
def test_C(recs, iou_key, B=B_BOOT):
    classes = sorted({r["cls_true"] for r in recs})
    out = {}
    for c in classes:
        ce = [r for r in recs if r["cls_true"] == c]
        s = get_success(ce, iou_key)
        t = get_trig(ce)
        n = len(ce)
        if n < 5:
            continue
        e_rate = (~t).mean()
        fail = 1 - s.mean()
        dev = abs(e_rate - fail)
        boot = np.empty(B)
        rand = np.empty(B)
        k = int(round(e_rate * n))
        for b in range(B):
            idx = RNG.integers(0, n, n)
            e_b = (~t[idx]).mean()
            f_b = 1 - s[idx].mean()
            boot[b] = abs(e_b - f_b)
            # Random-rejector null: the same resampled outcomes (idx), but the abstention
            # mask is assigned randomly rather than by the model's actual termination
            # decision. Coverage is held fixed at k/n, so this is a genuine
            # coverage-matched comparator rather than a constant baseline.
            mask = np.zeros(n, bool)
            mask[RNG.permutation(n)[:k]] = True
            rand[b] = abs(mask.mean() - (1 - s[idx].mean()))
        out[c] = dict(
            n=n, exhaustion=float(e_rate), fail_rate=float(fail), dev=float(dev),
            dev_ci=np.percentile(boot, [2.5, 97.5]).tolist(),
            random_band=np.percentile(rand, [2.5, 97.5]).tolist(),
            below_random_band=bool(dev < np.percentile(rand, [2.5, 97.5])[0]),
        )
    return out


# --------------------------------------------- Test D: fixed-schedule stopping control
def test_D(recs, max_k=20):
    out = {}
    n_total = len(recs)
    for k in range(1, max_k + 1):
        sub = [r for r in recs if len(r["ious"]) > k]
        if not sub:
            continue
        corloc_k = float(np.mean([r["ious"][k] >= TAU for r in sub]))
        out[k] = dict(corloc=corloc_k, n=len(sub), coverage=len(sub) / n_total)
    return out


# ----------------------------------------------- Test E: risk-coverage / AURC
def test_E(recs, iou_key):
    s = get_success(recs, iou_key).astype(float)
    score = np.array([r["confs"][-1] if r.get("confs") else float("nan") for r in recs])
    valid = ~np.isnan(score)
    s, score = s[valid], score[valid]
    n = s.size
    if n == 0:
        return None

    def rc_curve(order):
        s_sorted = s[order]
        cum_correct = np.cumsum(s_sorted)
        coverage = np.arange(1, n + 1) / n
        risk = 1 - cum_correct / np.arange(1, n + 1)
        return coverage, risk

    # np.trapz was removed in numpy>=2.0 in favor of np.trapezoid; support both.
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

    order_score = np.argsort(-score)          # highest confidence first
    cov_m, risk_m = rc_curve(order_score)
    aurc_model = float(_trapz(risk_m, cov_m))

    order_oracle = np.argsort(-s)              # perfect ranking (successes first)
    cov_o, risk_o = rc_curve(order_oracle)
    aurc_oracle = float(_trapz(risk_o, cov_o))

    shuffled = RNG.permutation(n)
    cov_r, risk_r = rc_curve(shuffled)
    aurc_random = float(_trapz(risk_r, cov_r))

    return dict(aurc_model=aurc_model, aurc_oracle=aurc_oracle, aurc_random=aurc_random,
                e_aurc=aurc_model - aurc_oracle,
                curves=dict(model=(cov_m.tolist(), risk_m.tolist()),
                            oracle=(cov_o.tolist(), risk_o.tolist()),
                            random=(cov_r.tolist(), risk_r.tolist())))


# --------------------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--iou-key", default="final_iou",
                     help="use 'final_iou_oracle' for the 20-class oracle-scored records")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    recs = load_records(args.records)
    if args.iou_key not in recs[0]:
        raise SystemExit(f"'{args.iou_key}' not in records. Available: {list(recs[0].keys())}")

    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log(f"=== Calibration analysis: {args.records}  (iou_key={args.iou_key}) ===")
    log(f"n episodes: {len(recs)}\n")

    gap, ppv_t, ppv_e = ppv_gap(recs, args.iou_key)
    log(f"PPV(triggered)={ppv_t:.4f}  PPV(exhausted)={ppv_e:.4f}  gap={gap:.4f}\n")

    log("--- Test A: coverage-matched random-rejector permutation test ---")
    a = test_A(recs, args.iou_key)
    log(f"observed PPV gap = {a['observed_gap']:.4f}")
    log(f"permutation p-value (B={B_PERM}) = {a['p_value']:.5f}")
    log(f"{'supports' if a['p_value'] < 0.001 else 'does not clearly support'} "
        f"the calibrated-rejection claim\n")

    log("--- Test B: Brier reliability/resolution decomposition ---")
    b = test_B(recs, args.iou_key)
    log(f"REL={b['REL']:.4f}  RES={b['RES']:.4f}  UNC={b['UNC']:.4f}  skill=RES/UNC={b['skill']:.3f}")
    log(f"RES 95% bootstrap CI = [{b['RES_ci'][0]:.4f}, {b['RES_ci'][1]:.4f}]")
    log(f"{'supports' if b['RES_excludes_zero'] else 'does not clearly support'} "
        f"(CI excludes 0: {b['RES_excludes_zero']})\n")

    log("--- Test C: per-class deviation vs random-rejector null ---")
    c = test_C(recs, args.iou_key)
    n_pass = sum(1 for v in c.values() if v["below_random_band"])
    for cls, v in sorted(c.items()):
        flag = "ok " if v["below_random_band"] else "!! "
        log(f"  {flag}{cls:14s} n={v['n']:4d}  dev={v['dev']:.3f}  "
            f"CI=[{v['dev_ci'][0]:.3f},{v['dev_ci'][1]:.3f}]  "
            f"random_band=[{v['random_band'][0]:.3f},{v['random_band'][1]:.3f}]")
    log(f"{n_pass}/{len(c)} classes fall below the random-rejector band\n")

    log("--- Test D: fixed-schedule stopping control ---")
    d = test_D(recs)
    for k in sorted(d):
        log(f"  k={k:2d}  CorLoc={d[k]['corloc']:.4f}  n={d[k]['n']:4d}  coverage={d[k]['coverage']:.3f}")
    log("(compare against the learned model's actual CorLoc, printed above)\n")

    log("--- Test E: risk-coverage / AURC ---")
    e = test_E(recs, args.iou_key)
    if e is None:
        log("skipped -- no 'confs' field found in records\n")
    else:
        log(f"AURC(model)  = {e['aurc_model']:.4f}")
        log(f"AURC(oracle) = {e['aurc_oracle']:.4f}  (lower bound)")
        log(f"AURC(random) = {e['aurc_random']:.4f}  (upper bound / chance)")
        log(f"E-AURC (model - oracle) = {e['e_aurc']:.4f}")
        log(f"{'supports' if e['aurc_model'] < e['aurc_random'] else 'does not support'} "
            f"(model AURC {'<' if e['aurc_model'] < e['aurc_random'] else '>='} random AURC)\n")

    with open(os.path.join(args.out, "report.txt"), "w") as f:
        f.write("\n".join(lines))

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(dict(ppv_gap=gap, ppv_trig=ppv_t, ppv_exh=ppv_e,
                        test_A={k: v for k, v in a.items() if k != "null"},
                        test_B={k: v for k, v in b.items()},
                        test_C=c, test_D=d,
                        test_E=({k: v for k, v in e.items() if k != "curves"} if e else None)),
                  f, indent=2, default=float)

    if HAVE_MPL:
        make_plots(args.out, a, b, c, d, e)

    log(f"\ndone. report + JSON + plots written to {args.out}/")


def make_plots(out, a, b, c, d, e):
    # Test A: null histogram
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(a["null"], bins=50, color="#a6a6a6")
    ax.axvline(a["observed_gap"], color="red", linewidth=2,
               label=f"observed ({a['observed_gap']:.3f})")
    ax.set_xlabel("PPV(trig) - PPV(exh) under random rejector")
    ax.set_ylabel("count")
    ax.set_title(f"Test A: permutation null (p={a['p_value']:.4f})")
    ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out, "testA_permutation.png"), dpi=150); plt.close()

    # Test C: per-class deviation with error bars + random band
    fig, ax = plt.subplots(figsize=(7, 4))
    classes = sorted(c.keys())
    devs = [c[k]["dev"] for k in classes]
    ci_lo = [c[k]["dev"] - c[k]["dev_ci"][0] for k in classes]
    ci_hi = [c[k]["dev_ci"][1] - c[k]["dev"] for k in classes]
    band_lo = np.mean([c[k]["random_band"][0] for k in classes])
    band_hi = np.mean([c[k]["random_band"][1] for k in classes])
    ax.axhspan(band_lo, band_hi, color="#f4cccc", alpha=0.5, label="random-rejector band (avg)")
    ax.errorbar(range(len(classes)), devs, yerr=[ci_lo, ci_hi], fmt="o", color="#1f618d")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("|exhaustion - failure rate|")
    ax.set_title("Test C: per-class deviation vs random-rejector null")
    ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out, "testC_perclass.png"), dpi=150); plt.close()

    # Test D: CorLoc vs fixed stop-step k
    fig, ax = plt.subplots(figsize=(5, 4))
    ks = sorted(d.keys())
    ax.plot(ks, [d[k]["corloc"] for k in ks], "o-", color="#1e8449")
    ax.set_xlabel("fixed stop step k"); ax.set_ylabel("CorLoc")
    ax.set_title("Test D: fixed-schedule stopping control")
    plt.tight_layout(); plt.savefig(os.path.join(out, "testD_fixedschedule.png"), dpi=150); plt.close()

    # Test E: risk-coverage curve
    if e is not None:
        fig, ax = plt.subplots(figsize=(5, 4))
        for name, color in [("model", "#1f618d"), ("oracle", "#1e8449"), ("random", "#a6a6a6")]:
            cov, risk = e["curves"][name]
            ax.plot(cov, risk, label=name, color=color)
        ax.set_xlabel("coverage"); ax.set_ylabel("selective risk")
        ax.set_title("Test E: risk-coverage curve")
        ax.legend()
        plt.tight_layout(); plt.savefig(os.path.join(out, "testE_riskcoverage.png"), dpi=150); plt.close()


if __name__ == "__main__":
    main()
