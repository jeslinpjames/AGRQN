#!/usr/bin/env python
"""Regenerate *_test.json / *_records.json.gz for every saved checkpoint in a results
directory, by pure inference on the saved *_best.pth / *_final.pth weights. No retraining.

Useful for re-scoring existing checkpoints under a different evaluation protocol (e.g. after
changing --test-per-class or adding a new summary statistic) without rerunning training.

Usage:
    python scripts/reevaluate_checkpoints.py --voc ./data/VOC2012_train_val \
        --out ./results/results_final --gpu 0
"""
import argparse
import glob
import gzip
import json
import logging
import os

import torch

from agrqn import core as L

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# The config each tag was trained with. Tags that differ from the config name are listed
# explicitly; everything else maps to itself.
TAG2CFG = {
    "final_s42": "duel_wide", "final_s1": "duel_wide", "final_s7": "duel_wide",
    "voc20_dino": "voc20_dino", "voc20_none": "voc20_none",
    "voc20_dino_20cls": "voc20_dino", "voc20_none_20cls": "voc20_none",
}
CLASSES20 = {"voc20_dino", "voc20_none", "voc20_dino_20cls", "voc20_none_20cls"}
CACHE_FOR = {
    "desc_dinov2": "feat_cache_dinov2", "desc_dinov2_none": "feat_cache_dinov2",
    "voc20_dino": "feat_cache_20", "voc20_none": "feat_cache_20",
    "voc20_dino_20cls": "feat_cache_20", "voc20_none_20cls": "feat_cache_20",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voc", required=True)
    ap.add_argument("--out", default="./results/results_final")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--clf-weights", default="./checkpoints/clf_dinov2_20.pth")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tags = sorted(os.path.basename(p)[:-len("_best.pth")]
                  for p in glob.glob(os.path.join(a.out, "*_best.pth")))
    logger.info("found %d checkpoints: %s", len(tags), tags)

    for tag in tags:
        cfg_name = TAG2CFG.get(tag, tag)
        n_cls = 20 if tag in CLASSES20 else 5
        cache = CACHE_FOR.get(tag, "./checkpoints/feat_cache")
        weights = ("./checkpoints/vgg16_20class.pth" if n_cls == 20
                   else "./checkpoints/vgg16_5class.pth")

        L.set_classes(n_cls)
        if n_cls == 20:
            L.N_TEST_PER_CLASS = 100
        try:
            cfg = L.get_config(cfg_name)
        except SystemExit:
            logger.warning("skip %s: config '%s' not in agrqn.core", tag, cfg_name)
            continue
        cfg["_voc"] = a.voc

        logger.info("=== %s (config %s, %d classes, cache %s)", tag, cfg_name, n_cls, cache)
        frozen = L.Frozen(weights, device)
        if n_cls == 20 and os.path.exists(a.clf_weights):
            frozen.load_dinov2_clf(a.clf_weights)

        _, train_all, (ck_n, ck_l), (te_n, te_l) = L.build_splits(a.voc)
        RAM = L.CacheRAM(sorted(set(train_all) | set(ck_n) | set(te_n)),
                          cache, cfg["feat"], device, prior=cfg["prior"])

        hist_p = os.path.join(a.out, f"{tag}_history.json")
        history = json.load(open(hist_p)) if os.path.exists(hist_p) else []

        res = {"config": cfg_name, "tag": tag,
               "cfg": {k: v for k, v in cfg.items() if not k.startswith("_")},
               "seed": 42, "history": history, "params_M": 0.0, "train_minutes": 0.0}

        for ck in ("best", "final"):
            p = os.path.join(a.out, f"{tag}_{ck}.pth")
            if not os.path.exists(p):
                continue
            net = L.AGRQN(cfg).to(device)
            net.load_state_dict(torch.load(p, map_location=device))
            res["params_M"] = sum(x.numel() for x in net.parameters()) / 1e6
            recs = L.evaluate(net, te_n, te_l, cfg, frozen, RAM, device)
            res[f"test_{ck}"] = {"last_box": L.summarize(recs, "last_iou"),
                                  "conf_box": L.summarize(recs, "final_iou")}
            res[f"test_{ck}"]["last_box_oracle"] = L.summarize(recs, "last_iou_oracle")
            res[f"test_{ck}"]["conf_box_oracle"] = L.summarize(recs, "final_iou_oracle")
            res[f"test_{ck}"]["last_box_area"] = L.summarize(recs, "last_iou_area")
            res[f"test_{ck}"]["conf_box_area"] = L.summarize(recs, "final_iou_area")
            if ck == "best":
                keep = ("name", "cls_true", "cls_pred", "area_cls", "correct_cls", "n_inst",
                        "gt_frac", "triggered", "n_steps", "last_iou", "final_iou", "peak_iou",
                        "last_iou_oracle", "final_iou_oracle", "peak_iou_oracle",
                        "last_iou_area", "final_iou_area", "peak_iou_area",
                        "actions", "ious", "confs")
                with gzip.open(os.path.join(a.out, f"{tag}_records.json.gz"), "wt") as f:
                    json.dump([{k: r[k] for k in keep} for r in recs], f)
                b = res["test_best"]["last_box"]
                bo = res["test_best"]["last_box_oracle"]
                ba = res["test_best"]["last_box_area"]
                logger.info(
                    "  mIoU %.4f (pred) / %.4f (oracle) / %.4f (area) | "
                    "CorLoc %.4f (pred) / %.4f (oracle) / %.4f (area) | "
                    "exh %.4f | ppv %.4f",
                    b["miou"], bo["miou"], ba["miou"],
                    b["acc05"], bo["acc05"], ba["acc05"],
                    b["exhaust_rate"], b["trigger_ppv"])

        with open(os.path.join(a.out, f"{tag}_test.json"), "w") as f:
            json.dump(res, f, indent=2, default=float)

    logger.info("done. %s now has one *_test.json and one *_records.json.gz per run.", a.out)


if __name__ == "__main__":
    main()
