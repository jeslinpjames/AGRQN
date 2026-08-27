#!/usr/bin/env python
"""Compute frozen backbone features once, ahead of training.

VGG-16 and DINO are both frozen, so re-running them every episode is wasted work. For each
image we store:
    feat7   512x7x7    VGG conv map from a 224px input   (the standard descriptor grid)
    feat14  512x14x14  VGG conv map from a 448px input   (finer grid, for the 'hires' config,
                        which targets small objects)
    hm      224x224    DINO ViT-S/16 [CLS] self-attention (the class-agnostic prior)
    pred    int         the classifier's predicted class
    wh      int, int    original image size

After this, training performs zero backbone forward passes -- only the ~5M-parameter head
runs. Disk footprint is roughly 350 KB/image (~1.5 GB for the full VOC2012 pool).

Usage:
    python scripts/precache.py --voc ./data/VOC2012_train_val --weights vgg16_5class.pth \
        --cache ./checkpoints/feat_cache --gpu 0
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
    ap.add_argument("--voc", required=True, help="path to VOC2012_train_val")
    ap.add_argument("--weights", default="vgg16_5class.pth")
    ap.add_argument("--cache", default="./checkpoints/feat_cache")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--classes", type=int, default=5, choices=[5, 20])
    ap.add_argument("--desc", default="vgg", choices=["vgg", "dino", "dinov2"])
    ap.add_argument("--classifier", default="vgg", choices=["vgg", "dinov2"])
    ap.add_argument("--clf-weights", default=None,
                     help="linear-probe checkpoint, e.g. clf_dinov2_20.pth")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s (physical GPU %d)", device, a.gpu)

    L.set_classes(a.classes)
    os.makedirs(a.cache, exist_ok=True)
    frozen = L.Frozen(a.weights, device)
    if a.classifier == "dinov2":
        if not a.clf_weights:
            raise SystemExit("--classifier dinov2 needs --clf-weights")
        frozen.load_dinov2_clf(a.clf_weights)

    pool, train_all, (ckpt_n, _), (test_n, _) = L.build_splits(a.voc)
    logger.info("split sizes:")
    for c in L.SELECTED_CLASSES:
        logger.info("  %-10s train %5d", c, len(pool[c]))
    logger.info("  train (unique) %d", len(train_all))
    logger.info("  ckpt           %d  (early stopping only)", len(ckpt_n))
    logger.info("  test           %d  (reported once; never trained on)", len(test_n))

    names = sorted(set(train_all) | set(ckpt_n) | set(test_n))
    todo = [n for n in names if not os.path.exists(os.path.join(a.cache, n + ".npz"))]
    logger.info("%d images | %d to compute | %d already cached",
                len(names), len(todo), len(names) - len(todo))

    for n in tqdm(todo, desc="caching"):
        img = np.array(Image.open(
            os.path.join(a.voc, "JPEGImages", f"{n}.jpg")).convert("RGB"))
        np.savez_compressed(
            os.path.join(a.cache, n + ".npz"),
            feat7=(frozen.feat_dinov2(img) if a.desc == "dinov2"
                   else frozen.feat(img, 224))[0].cpu().numpy().astype(np.float16),
            feat14=frozen.feat(img, 448)[0].cpu().numpy().astype(np.float16),
            hm=frozen.dino_map(img).astype(np.float16),
            pred=np.int16(frozen.predict_class(img)),
            wh=np.int32([img.shape[1], img.shape[0]]))

    sz = sum(os.path.getsize(os.path.join(a.cache, f)) for f in os.listdir(a.cache))
    logger.info("done. cache: %d files, %.2f GB", len(os.listdir(a.cache)), sz / 1e9)


if __name__ == "__main__":
    main()
