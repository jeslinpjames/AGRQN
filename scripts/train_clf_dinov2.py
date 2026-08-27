#!/usr/bin/env python
"""Fit a linear probe on frozen DINOv2 for N-way VOC classification.

For the 20-class experiment, a detection is scored against the ground truth of the
classifier's predicted class, so every misclassification costs the localizer an automatic
IoU of 0. The VGG-16 20-way classifier (see train_classifier.py) tops out well below what a
strong classifier can reach, which would cap CorLoc before the agent takes a single step and
turn the 20-class experiment into a test of the classifier rather than the localizer. A
linear probe on frozen DINOv2 features reaches noticeably higher accuracy on VOC while
keeping the same frozen-backbone philosophy as the rest of the pipeline.

Label assignment follows train_classifier.py: each image gets exactly one label, the class
of its largest ground-truth box, to avoid feeding the same pixels contradictory labels.

Two stages, both cheap because the backbone never trains:
  1. extract the frozen DINOv2 [CLS] embedding for every image (once)
  2. fit a linear head on those embeddings

Usage:
    python scripts/train_clf_dinov2.py --voc ./data/VOC2012_train_val \
        --classes 20 --gpu 0 --out ./checkpoints/clf_dinov2_20.pth
"""
import argparse
import logging
import os
import random
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

VOC20 = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
         "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
         "train", "tvmonitor"]
VOC5 = ["bicycle", "aeroplane", "horse", "cat", "dog"]

TF = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_ids(name, voc):
    with open(os.path.join(voc, "ImageSets", "Main", name + ".txt")) as f:
        out = []
        for line in f:
            p = line.split()
            if len(p) == 1 or "-1" not in p[1]:
                out.append(p[0])
    return out


def dominant_label(name, voc, cls2idx):
    """The class of the largest ground-truth box. One label per image, no contradictions."""
    try:
        root = ET.parse(os.path.join(voc, "Annotations", name + ".xml")).getroot()
    except Exception:
        return None
    best, best_a = None, -1.0
    for obj in root.findall("object"):
        c = obj.find("name").text
        if c not in cls2idx:
            continue
        bb = obj.find("bndbox")
        a = ((float(bb.find("xmax").text) - float(bb.find("xmin").text))
             * (float(bb.find("ymax").text) - float(bb.find("ymin").text)))
        if a > best_a:
            best, best_a = cls2idx[c], a
    return best


@torch.inference_mode()
def embed_split(model, voc, classes, split, device):
    cls2idx = {c: i for i, c in enumerate(classes)}
    names = sorted({n for c in classes for n in load_ids(f"{c}_{split}", voc)})
    X, Y = [], []
    for n in tqdm(names, desc=f"embedding {split}"):
        y = dominant_label(n, voc, cls2idx)
        if y is None:
            continue
        img = Image.open(os.path.join(voc, "JPEGImages", f"{n}.jpg")).convert("RGB")
        t = TF(img).unsqueeze(0).to(device)
        z = model(t)                                  # DINOv2 returns the [CLS] embedding
        X.append(z[0].float().cpu().numpy()); Y.append(y)
    return np.stack(X), np.array(Y)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voc", required=True)
    ap.add_argument("--classes", type=int, default=20, choices=[5, 20])
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = VOC20 if a.classes == 20 else VOC5
    out = a.out or f"./checkpoints/clf_dinov2_{a.classes}.pth"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.manual_seed(42); np.random.seed(42); random.seed(42)

    logger.info("loading frozen DINOv2 ViT-S/14 ...")
    bb = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").eval().to(dev)
    for p in bb.parameters():
        p.requires_grad = False

    # Train on `train`, validate on `val`, for the same leak-avoidance reason as
    # train_classifier.py.
    Xtr, Ytr = embed_split(bb, a.voc, classes, "train", dev)
    Xva, Yva = embed_split(bb, a.voc, classes, "val", dev)
    logger.info("%d classes | train %d | val %d | dim %d", a.classes, len(Ytr), len(Yva), Xtr.shape[1])

    Xtr_t = torch.tensor(Xtr, device=dev); Ytr_t = torch.tensor(Ytr, device=dev)
    Xva_t = torch.tensor(Xva, device=dev); Yva_t = torch.tensor(Yva, device=dev)

    head = nn.Linear(Xtr.shape[1], len(classes)).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    best = 0.0

    for ep in range(1, a.epochs + 1):
        head.train()
        perm = torch.randperm(len(Ytr_t), device=dev)
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            opt.zero_grad(set_to_none=True)
            lossf(head(Xtr_t[idx]), Ytr_t[idx]).backward()
            opt.step()
        sched.step()
        head.eval()
        with torch.inference_mode():
            acc = (head(Xva_t).argmax(1) == Yva_t).float().mean().item()
        if acc > best:
            best = acc
            torch.save({"state_dict": head.state_dict(), "classes": classes,
                        "dim": Xtr.shape[1]}, out)
        if ep % 10 == 0 or ep == 1:
            logger.info("ep %3d: val top-1 %.4f (best %.4f)", ep, acc, best)

    logger.info("done. best val top-1 = %.4f -> %s", best, out)


if __name__ == "__main__":
    main()
