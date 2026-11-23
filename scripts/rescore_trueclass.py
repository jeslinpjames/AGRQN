#!/usr/bin/env python
"""
rescore_trueclass.py -- predicted-class vs TRUE-class scoring (paper: Table 3).

Inference only, no training. For every 5-class run in --results it reloads the saved best
checkpoint, replays the test episodes with agrqn.core.evaluate() (which computes both
scorings), and reports both.

SELF-CHECK: the predicted-class numbers it recomputes must match the run's own *_test.json.
If a row says MISMATCH, the cache or config used here differs from the one the run was
trained/evaluated with -- do not use that row until it is fixed.

Usage (minutes on one GPU), from the repository root:
    python scripts/rescore_trueclass.py --voc <VOC> --gpu 0 \
        | tee ./results/results_trueclass/trueclass.txt
"""
import argparse, glob, json, os
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--voc", required=True)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--results", default="./results/results_final")
ap.add_argument("--vgg-cache", default="./checkpoints/feat_cache")
ap.add_argument("--dinov2-cache", default="./checkpoints/feat_cache_dinov2")
ap.add_argument("--weights", default="./checkpoints/vgg16_5class.pth")
ap.add_argument("--out", default="./results/results_trueclass")
ap.add_argument("--only", nargs="*", default=None, help="restrict to these tags")
a = ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
from agrqn import core as L          # noqa: E402  (after CUDA_VISIBLE_DEVICES)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L.set_classes(5)
frozen = L.Frozen(a.weights, dev)
_, _, _, (te_n, te_l) = L.build_splits(a.voc)

rams, rows = {}, []
for f in sorted(glob.glob(f"{a.results}/*_test.json")):
    r = json.load(open(f))
    cname = r.get("config", "")
    tag = r.get("tag") or os.path.basename(f)[: -len("_test.json")]
    if cname.startswith("voc20") or cname not in L.CONFIGS:
        continue
    if a.only and tag not in a.only:
        continue
    ck = f"{a.results}/{tag}_best.pth"
    if not os.path.exists(ck):
        print(f"skip {tag}: no {ck}", flush=True)
        continue
    cfg = L.get_config(cname); cfg["_voc"] = a.voc
    cache = a.dinov2_cache if cfg["desc"] == "dinov2" else a.vgg_cache
    key = (cache, cfg["feat"], cfg["prior"])
    if key not in rams:
        rams[key] = L.CacheRAM(sorted(te_n), cache, cfg["feat"], dev, prior=cfg["prior"])
    net = L.AGRQN(cfg).to(dev)
    net.load_state_dict(torch.load(ck, map_location=dev)); net.eval()
    recs = L.evaluate(net, te_n, te_l, cfg, frozen, rams[key], dev)

    p = L.summarize(recs, "last_iou")
    t = L.summarize(recs, "last_iou_oracle")
    rep = r.get("test_best", {}).get("last_box", {})
    ok = (abs(p["miou"] - rep.get("miou", -9)) < 1e-3 and
          abs(p["acc05"] - rep.get("acc05", -9)) < 1e-3)
    clf_acc = float(np.mean([x["correct_cls"] for x in recs]))
    n_mis = int(sum(not x["correct_cls"] for x in recs))
    row = dict(tag=tag, config=cname, cache=cache,
               pred_miou=p["miou"], pred_corloc=p["acc05"],
               true_miou=t["miou"], true_corloc=t["acc05"],
               reported_miou=rep.get("miou"), reported_corloc=rep.get("acc05"),
               clf_acc=clf_acc, n_misclassified=n_mis, selfcheck="OK" if ok else "MISMATCH")
    rows.append(row)
    print(f"{tag:26s} pred mIoU {p['miou']:.4f} CorLoc {p['acc05']:.3f} | "
          f"TRUE mIoU {t['miou']:.4f} CorLoc {t['acc05']:.3f} | clf acc {clf_acc:.3f} "
          f"({n_mis} wrong) | reported {rep.get('miou', float('nan')):.4f} -> {row['selfcheck']}",
          flush=True)

os.makedirs(a.out, exist_ok=True)
json.dump(rows, open(f"{a.out}/trueclass.json", "w"), indent=2)
print(f"\nwrote {a.out}/trueclass.json")
