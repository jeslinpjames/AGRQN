#!/usr/bin/env python
"""
calib_analysis.py -- calibration of the trigger Q-value (paper: Appendix E).
CPU only, numpy only, ~1-3 min.

Reads <dir>/<tag>.npz written by calib_qvalues.py.

Forecast:   p = (Q(s,trigger)/eta + 1)/2, clipped to [0,1]   (exact for +/-eta terminal reward)
Outcome:    y = IoU of the box at state s >= 0.5  (predicted-class scoring, as in the paper)

Two evaluation sets per policy:
  DECISION  one pair per episode: the state where the episode ended (trigger step, or the
            last state before the budget ran out). This is the stop decision itself.
  ALLSTATES every visited state. Q(s,trigger) predicts "would stopping HERE succeed?" at
            every step, so every state is a valid (forecast, outcome) pair.

Metrics (95% CIs from a bootstrap over IMAGES, so within-episode correlation is respected):
  AUROC     discrimination of Q(s,trigger) (mapping-free)
  ECE       10 equal-width bins, and adaptive ECE (10 equal-mass bins)
  Brier, and its Murphy decomposition REL - RES + UNC on the same 10 bins
  Platt     logistic recalibration of Q(s,trigger), 2-fold cross-fitted by image, then ECE
  gap       mean(p) - mean(y)  (> 0 = overconfident in success)

Outputs: <dir>/summary.txt, summary.json, bins.json
"""
import argparse, glob, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="./results/results_qcalib")
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--bins", type=int, default=10)
a = ap.parse_args()
R = np.random.default_rng(0)
NB = a.bins


# ------------------------------------------------------------------ metric helpers
def rankdata(x):
    x = np.asarray(x, float); s = np.argsort(x, kind="mergesort")
    inv = np.empty_like(s); inv[s] = np.arange(len(x))
    xs = x[s]; obs = np.r_[True, xs[1:] != xs[:-1]]; dense = obs.cumsum()[inv]
    cnt = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (cnt[dense] + cnt[dense - 1] + 1)


def auroc(s, y):
    y = y.astype(bool); n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return (rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def binstats(p, y, adaptive=False):
    if adaptive:
        edges = np.quantile(p, np.linspace(0, 1, NB + 1)); edges[0], edges[-1] = -1e-9, 1 + 1e-9
        b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, NB - 1)
    else:
        b = np.minimum((p * NB).astype(int), NB - 1)
    n = np.bincount(b, minlength=NB).astype(float)
    ps = np.bincount(b, p, NB); ys = np.bincount(b, y, NB)
    with np.errstate(invalid="ignore", divide="ignore"):
        pb, yb = ps / n, ys / n
    return n, pb, yb


def calib(p, y):
    y = y.astype(float); N = len(y); ybar = y.mean()
    n, pb, yb = binstats(p, y)
    m = n > 0
    ece = float(np.sum(n[m] / N * np.abs(pb[m] - yb[m])))
    rel = float(np.sum(n[m] / N * (pb[m] - yb[m]) ** 2))
    res = float(np.sum(n[m] / N * (yb[m] - ybar) ** 2))
    na, pa, ya = binstats(p, y, adaptive=True); ma = na > 0
    aece = float(np.sum(na[ma] / N * np.abs(pa[ma] - ya[ma])))
    return dict(ece=ece, aece=aece, rel=rel, res=res, unc=float(ybar * (1 - ybar)),
                brier=float(np.mean((p - y) ** 2)), gap=float(p.mean() - ybar))


def platt_fit(x, y, iters=50, ridge=1e-4):
    mu, sd = x.mean(), x.std() + 1e-9; z = (x - mu) / sd
    w = np.zeros(2); X = np.c_[z, np.ones_like(z)]
    for _ in range(iters):
        pr = 1 / (1 + np.exp(-X @ w)); g = X.T @ (pr - y) + ridge * w
        H = (X * (pr * (1 - pr))[:, None]).T @ X + ridge * np.eye(2)
        w -= np.linalg.solve(H, g)
    return lambda xx: 1 / (1 + np.exp(-(np.c_[(xx - mu) / sd, np.ones_like(xx)] @ w)))


def platt_crossfit(x, y, img):
    ids = np.unique(img); perm = R.permutation(ids); fold = np.isin(img, perm[: len(ids) // 2])
    out = np.empty_like(x, dtype=float)
    for f in (fold, ~fold):
        if y[~f].min() == y[~f].max():
            out[f] = y[~f].mean(); continue
        out[f] = platt_fit(x[~f], y[~f].astype(float))(x[f])
    return out


def evaluate_set(x, p, y, img, groups):
    """x = raw Q_trigger, p = mapped prob (or None), y = outcome, img = image id per row."""
    pc = platt_crossfit(x, y, img)
    point = dict(auroc=auroc(x, y), n=int(len(y)), base=float(y.mean()))
    if p is not None:
        point["oor"] = float(np.mean((p < 0) | (p > 1)))       # out-of-range before clipping
        point.update({k: v for k, v in calib(np.clip(p, 0, 1), y).items()})
    point.update({"platt_" + k: v for k, v in calib(pc, y).items() if k in ("ece", "rel", "brier")})
    # bootstrap over images
    keys = ["auroc", "platt_ece"] + (["ece", "aece", "rel", "res", "brier", "gap"] if p is not None else [])
    bs = {k: [] for k in keys}
    gl = list(groups.values())
    for _ in range(a.boot):
        idx = np.concatenate([gl[j] for j in R.integers(0, len(gl), len(gl))])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        bs["auroc"].append(auroc(x[idx], yy))
        bs["platt_ece"].append(calib(pc[idx], yy)["ece"])
        if p is not None:
            c = calib(np.clip(p[idx], 0, 1), yy)
            for k in ("ece", "aece", "rel", "res", "brier", "gap"):
                bs[k].append(c[k])
    point["ci"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                   for k, v in bs.items() if v}
    if p is not None:
        n, pb, yb = binstats(np.clip(p, 0, 1), y)
        point["bins"] = dict(n=n.tolist(), p=np.nan_to_num(pb, nan=-1).tolist(),
                             y=np.nan_to_num(yb, nan=-1).tolist())
    nP, pP, yP = binstats(pc, y)
    point["bins_platt"] = dict(n=nP.tolist(), p=np.nan_to_num(pP, nan=-1).tolist(),
                               y=np.nan_to_num(yP, nan=-1).tolist())
    return point


# ------------------------------------------------------------------ main loop
os.makedirs(a.dir, exist_ok=True)
rows, lines = [], []
for f in sorted(glob.glob(f"{a.dir}/*.npz")):
    tag = os.path.basename(f)[:-4]
    meta = json.load(open(f"{a.dir}/{tag}.json"))
    D = np.load(f)
    q, img, t, iou = D["q"], D["img"], D["t"], D["iou"]
    ep_trig, ep_last_act, ep_k = D["ep_trig"], D["ep_last_act"], D["ep_k"]

    # decision state = last captured state of each episode
    last_idx = np.r_[np.nonzero(np.diff(img))[0], len(img) - 1]
    last = np.zeros(len(img), bool); last[last_idx] = True
    trig_row = ep_trig[img]

    # action numbering vs Q columns: recorded actions may be offset from Q columns (e.g. 1..9
    # vs 0..8, since agrqn.core has TRIGGER = 9 with 9 outputs). Infer the offset from the
    # last step of every episode (recorded last action vs argmax of the captured Q there).
    eps = img[last_idx]
    diff = ep_last_act[eps].astype(int) - q[last_idx].argmax(1)
    offset = int(np.bincount(diff - diff.min()).argmax() + diff.min())
    greedy_last = float(np.mean(diff == offset))
    la = ep_last_act[ep_trig].astype(int)
    trig = (int(np.bincount(la).argmax()) if len(la) else 8 + offset) - offset
    trig_consistent = float(np.mean(la - offset == trig)) if len(la) else float("nan")
    assert 0 <= trig < q.shape[1], f"{tag}: trigger column {trig} out of range (offset {offset})"

    x_all = q[:, trig].astype(float)
    y_all = (iou >= 0.5)
    eta = float(meta["eta"])
    mappable = meta["shaping"] == "sign"
    p_all = (x_all / eta + 1) / 2 if mappable else None

    # sanity: does the saved per-step `confs` equal Q_trigger, or a margin?
    conf = D["conf"]; ok = ~np.isnan(conf)
    other = np.delete(q, trig, axis=1).max(1)
    conf_vs = dict(
        corr_qtrig=float(np.corrcoef(conf[ok], x_all[ok])[0, 1]) if ok.sum() > 2 else None,
        corr_margin=float(np.corrcoef(conf[ok], (x_all - other)[ok])[0, 1]) if ok.sum() > 2 else None,
        maxabs_qtrig=float(np.max(np.abs(conf[ok] - x_all[ok]))) if ok.any() else None,
        best_col=(int(np.nanargmax([abs(np.corrcoef(conf[ok], q[ok, j])[0, 1]) for j in range(q.shape[1])]))
                  if ok.sum() > 2 else None))

    res = dict(meta=meta, trig_col=trig, trig_consistent=trig_consistent, conf_check=conf_vs,
               action_offset=offset, greedy_last=greedy_last,
               qtrig_range=[float(x_all.min()), float(x_all.max())])
    for name, m in (("decision", last), ("allstates", np.ones(len(img), bool)),
                    ("decision_triggered", last & trig_row), ("decision_exhausted", last & ~trig_row)):
        if m.sum() < 20 or y_all[m].min() == y_all[m].max():
            res[name] = None; continue
        ii = img[m]; groups = {}
        for j, g in enumerate(ii):
            groups.setdefault(g, []).append(j)
        groups = {k: np.array(v) for k, v in groups.items()}
        res[name] = evaluate_set(x_all[m], None if p_all is None else p_all[m], y_all[m], ii, groups)
    rows.append(res)

    def fmt(s, k, d=3):
        if s is None or k not in s:
            return "   n/a        "
        lo, hi = s.get("ci", {}).get(k, [np.nan, np.nan])
        return f"{s[k]:.{d}f} [{lo:.{d}f},{hi:.{d}f}]"

    hdr = (f"\n=== {tag}  (config {meta['config']}, eta {eta:g}, shaping {meta['shaping']}, "
           f"selfcheck {meta['selfcheck']}, k==n {meta['align_k_eq_n']:.3f}, action offset {offset:+d}, "
           f"greedy(last step, offset-corrected) {greedy_last:.3f}, trig col {trig} ({trig_consistent:.3f}))")
    lines.append(hdr)
    lines.append(f"    Q_trig range [{x_all.min():.2f}, {x_all.max():.2f}] | confs~Q_trig corr "
                 f"{conf_vs['corr_qtrig']} | confs~margin corr {conf_vs['corr_margin']} | confs best-matching Q col {conf_vs['best_col']}")
    for name in ("decision", "allstates", "decision_triggered", "decision_exhausted"):
        s = res[name]
        if s is None:
            lines.append(f"    {name:19s} (too few / single-class)"); continue
        lines.append(
            f"    {name:19s} n={s['n']:6d} base={s['base']:.3f} | AUROC {fmt(s,'auroc')}"
            + (f" | ECE {fmt(s,'ece')} aECE {s['aece']:.3f} | REL {fmt(s,'rel',4)} RES {s['res']:.4f}"
               f" UNC {s['unc']:.4f} | Brier {s['brier']:.4f} | gap {s['gap']:+.3f} | oor {s['oor']:.3f}"
               if "ece" in s else " | (no closed-form mapping: PBRS)")
            + f" | Platt-ECE {fmt(s,'platt_ece')}")
    print("\n".join(lines[-6:]), flush=True)

# ------------------------------------------------------------------ compact table
tab = ["\n\nCOMPACT TABLE  (decision set = one stop decision per episode; 95% bootstrap CIs over images)",
       f"{'tag':28s} {'AUROC':>6s} {'ECE':>18s} {'REL':>8s} {'RES':>8s} {'gap':>7s} {'oor':>6s} {'PlattECE':>9s}"
       f" | {'allst ECE':>9s} {'allst AUROC':>11s}"]
for r in rows:
    d, s = r["decision"], r["allstates"]
    if d is None:
        tab.append(f"{r['meta']['tag']:28s}  n/a"); continue
    if "ece" in d:
        ece = f"{d['ece']:.3f} [{d['ci']['ece'][0]:.3f},{d['ci']['ece'][1]:.3f}]"
        tab.append(f"{r['meta']['tag']:28s} {d['auroc']:6.3f} {ece:>18s} {d['rel']:8.4f} {d['res']:8.4f}"
                   f" {d['gap']:+7.3f} {d['oor']:6.3f} {d['platt_ece']:9.3f} | "
                   f"{(s or {}).get('ece', float('nan')):9.3f} {(s or {}).get('auroc', float('nan')):11.3f}")
    else:
        tab.append(f"{r['meta']['tag']:28s} {d['auroc']:6.3f} {'(PBRS: no mapping)':>18s} {'':8s} {'':8s}"
                   f" {'':7s} {'':6s} {d['platt_ece']:9.3f} | {'':9s} {(s or {}).get('auroc', float('nan')):11.3f}")
print("\n".join(tab))

open(f"{a.dir}/summary.txt", "w").write("\n".join(lines + tab) + "\n")
json.dump(rows, open(f"{a.dir}/summary.json", "w"), indent=1, default=float)
print(f"\nwrote {a.dir}/summary.txt and summary.json", flush=True)
