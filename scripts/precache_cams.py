#!/usr/bin/env python
"""Add Grad-CAM and LayerCAM maps to an existing feature cache.

Only needed for the prior-study configs (prior_gradcam / prior_layercam); the DINO configs
already have everything they need from precache.py.

Grad-CAM and LayerCAM are class-conditional -- they backprop from a chosen class logit -- so
we store one map per (image, class): with 5 classes and 2 methods that is 10 maps of
224x224 fp16 per image, written to <name>_cam.npz alongside the existing <name>.npz (roughly
1 MB/image). DINO needs none of this: it is class-agnostic, one map per image, which is
precisely the independence property the paper argues for.

Usage:
    python scripts/precache_cams.py --voc ./data/VOC2012_train_val \
        --weights vgg16_5class.pth --cache ./checkpoints/feat_cache --gpu 0
"""
import argparse
import logging
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from agrqn import core as L

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voc", required=True)
    ap.add_argument("--weights", default="vgg16_5class.pth")
    ap.add_argument("--cache", default="./checkpoints/feat_cache")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s (physical GPU %d)", device, a.gpu)

    frozen = L.Frozen(a.weights, device)
    gc = L.GradCAMGenerator(frozen.vgg, device)
    lc = L.LayerCAMGenerator(frozen.vgg, device)

    pool, train_all, (ckpt_n, _), (test_n, _) = L.build_splits(a.voc)
    names = sorted(set(train_all) | set(ckpt_n) | set(test_n))
    todo = [n for n in names if not os.path.exists(os.path.join(a.cache, n + "_cam.npz"))]
    logger.info("%d images | %d to compute", len(names), len(todo))

    nC = len(L.SELECTED_CLASSES)
    for n in tqdm(todo, desc="caching CAMs"):
        img = np.array(Image.open(
            os.path.join(a.voc, "JPEGImages", f"{n}.jpg")).convert("RGB"))
        blob = {}
        for c in range(nC):
            blob[f"gradcam_{c}"] = gc.generate(img, c).astype(np.float16)
            blob[f"layercam_{c}"] = lc.generate(img, c).astype(np.float16)
        np.savez_compressed(os.path.join(a.cache, n + "_cam.npz"), **blob)

    sz = sum(os.path.getsize(os.path.join(a.cache, f))
             for f in os.listdir(a.cache) if f.endswith("_cam.npz"))
    logger.info("done. CAM cache: %.2f GB", sz / 1e9)


if __name__ == "__main__":
    main()
