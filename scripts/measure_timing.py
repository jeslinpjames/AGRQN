#!/usr/bin/env python
"""Measure per-episode and per-component inference timing for the paper's timing table.

Compares the cached-ROI descriptor path against a naive per-crop-recompute baseline, and
reports the per-image cost of the DINO and Grad-CAM priors.

Usage:
    python scripts/measure_timing.py --voc ./data/VOC2012_train_val \
        --cache ./checkpoints/feat_cache_dinov2 --tag desc_dinov2 --gpu 0
"""
import argparse
import logging
import os
import time

import numpy as np
import torch
from PIL import Image

from agrqn import core as L

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def gts_for(RAM, frozen, voc, n):
    C = RAM.get(n, frozen.pool)
    return C, L.gt_boxes(L.annotation(n, voc), L.VOC_CLASS_IDS[L.LABEL_TO_CLASS_NAME[C.pred]])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voc", required=True)
    ap.add_argument("--cache", default="./checkpoints/feat_cache_dinov2")
    ap.add_argument("--tag", default="desc_dinov2")
    ap.add_argument("--weights", default="./checkpoints/vgg16_5class.pth")
    ap.add_argument("--checkpoint", default=None,
                     help="defaults to ./results/results_final/<tag>_best.pth")
    ap.add_argument("--out", default="./timing_full.txt")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    checkpoint = a.checkpoint or f"./results/results_final/{a.tag}_best.pth"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)

    cfg = L.get_config(a.tag); cfg["_voc"] = a.voc
    frozen = L.Frozen(a.weights, dev)
    _, _, _, (te_n, te_l) = L.build_splits(a.voc)
    names = te_n[:a.n]
    RAM = L.CacheRAM(sorted(names), a.cache, cfg["feat"], dev, prior=cfg["prior"])
    net = L.AGRQN(cfg).to(dev)
    net.load_state_dict(torch.load(checkpoint, map_location=dev)); net.eval()

    # warm-up
    for n in names[:5]:
        C, g = gts_for(RAM, frozen, a.voc, n); L.rollout(net, C, g, cfg, dev)
    sync()

    t0 = time.time(); steps_tot = 0
    for n in names:
        C, g = gts_for(RAM, frozen, a.voc, n)
        ep = L.rollout(net, C, g, cfg, dev); steps_tot += ep["n_steps"]
    sync(); roi_ms = (time.time() - t0) / len(names) * 1000
    per_step = roi_ms / (steps_tot / len(names))

    # naive baseline: recompute the descriptor from a re-cropped image at every step
    t0 = time.time()
    for n in names:
        img = np.array(Image.open(os.path.join(a.voc, "JPEGImages", n + ".jpg")).convert("RGB"))
        C, g = gts_for(RAM, frozen, a.voc, n)
        box = [0, 0, C.W, C.H]
        for t in range(8):
            crop = img[max(0, box[1]):box[3], max(0, box[0]):box[2]]
            if crop.size == 0:
                crop = img
            _ = frozen.feat_dinov2(crop)
            box = [int(box[0] + 10), int(box[1] + 10), int(box[2] - 10), int(box[3] - 10)]
            if box[2] - box[0] < 20 or box[3] - box[1] < 20:
                break
    sync(); crop_ms = (time.time() - t0) / len(names) * 1000

    t0 = time.time()
    for n in names:
        img = np.array(Image.open(os.path.join(a.voc, "JPEGImages", n + ".jpg")).convert("RGB"))
        _ = frozen.dino_map(img)
    sync(); dino_ms = (time.time() - t0) / len(names) * 1000

    gc = L.GradCAMGenerator(frozen.vgg, dev)
    t0 = time.time()
    for n in names:
        img = np.array(Image.open(os.path.join(a.voc, "JPEGImages", n + ".jpg")).convert("RGB"))
        _ = gc.generate(img, 0)
    sync(); gc_ms = (time.time() - t0) / len(names) * 1000

    out = f"""timing (n={len(names)}, single GPU)
=================================
descriptor path (per episode):
  ROI-cached : {roi_ms:6.1f} ms/image   ({per_step:.1f} ms/step, mean {steps_tot/len(names):.1f} steps)
  per-crop   : {crop_ms:6.1f} ms/image
  speedup    : {crop_ms/roi_ms:.1f}x

prior cost (per image):
  DINO ViT-S/16 (forward only)      : {dino_ms:6.1f} ms
  Grad-CAM VGG (forward+backward)   : {gc_ms:6.1f} ms
"""
    with open(a.out, "w") as f:
        f.write(out)
    logger.info("%s", out)


if __name__ == "__main__":
    main()
