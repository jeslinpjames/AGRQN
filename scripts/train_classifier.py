#!/usr/bin/env python
"""Fine-tune a VGG-16 classifier for N-way VOC classification.

Produces the frozen classifier used by the localization agent: the agent's detection is
scored against the ground truth of the classifier's *predicted* class, so classifier
accuracy sets an upper bound on achievable CorLoc.

Label assignment: VOC images are multi-label (39.9% of train images carry more than one
class label; 73.9% of those involve 'person'). Emitting one dataset entry per
(image, positive-class) pair would feed the same pixels into training multiple times under
different "correct" labels, capping achievable accuracy regardless of training time. Instead
each image is resolved to exactly one label: the class of its largest ground-truth bounding
box, a standard heuristic for picking the "dominant" object in an image.

Usage:
    python scripts/train_classifier.py --voc ./data/VOC2012_train_val \
        --classes 20 --out ./checkpoints/vgg16_20class.pth --gpu 0 --epochs 20
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
from torch.utils.data import DataLoader, Dataset
from torchvision.models import VGG16_Weights, vgg16
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

VOC20 = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
         "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
         "train", "tvmonitor"]
VOC5 = ["bicycle", "aeroplane", "horse", "cat", "dog"]

TRAIN_TF = T.Compose([
    T.Resize((256, 256)), T.RandomCrop(224), T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
EVAL_TF = T.Compose([
    T.Resize((224, 224)), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_ids(name, voc):
    with open(os.path.join(voc, "ImageSets", "Main", name + ".txt")) as f:
        out = []
        for line in f:
            p = line.split()
            if len(p) == 1 or "-1" not in p[1]:
                out.append(p[0])
    return out


def dominant_label(voc, image_id, classes):
    """Resolve one label per image: the class of the largest bounding box, restricted to
    `classes`. Avoids putting contradictory labels on the same pixels when a VOC image
    contains multiple object classes."""
    tree = ET.parse(os.path.join(voc, "Annotations", f"{image_id}.xml"))
    best_cls, best_area = None, -1.0
    for obj in tree.findall("object"):
        name = obj.find("name").text
        if name not in classes:
            continue
        bb = obj.find("bndbox")
        w = float(bb.find("xmax").text) - float(bb.find("xmin").text)
        h = float(bb.find("ymax").text) - float(bb.find("ymin").text)
        area = w * h
        if area > best_area:
            best_area, best_cls = area, name
    return best_cls


class VOCCls(Dataset):
    """Single label per image: the class with the largest bounding-box area."""

    def __init__(self, voc, classes, split, tf):
        self.voc, self.tf, self.classes = voc, tf, classes
        ids = set()
        for c in classes:
            ids.update(load_ids(f"{c}_{split}", voc))
        self.items = []
        skipped = 0
        for n in sorted(ids):
            cls = dominant_label(voc, n, classes)
            if cls is not None:
                self.items.append((n, classes.index(cls)))
            else:
                skipped += 1
        random.Random(0).shuffle(self.items)
        logger.info("[%s] %d single-labeled images (%d skipped, no matching object found)",
                    split, len(self.items), skipped)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        n, y = self.items[i]
        img = Image.open(os.path.join(self.voc, "JPEGImages", f"{n}.jpg")).convert("RGB")
        return self.tf(img), y


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voc", required=True)
    ap.add_argument("--classes", type=int, default=20, choices=[5, 20])
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = VOC20 if a.classes == 20 else VOC5
    out = a.out or f"./checkpoints/vgg16_{a.classes}class.pth"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.manual_seed(42); random.seed(42); np.random.seed(42)

    # Train on `train`, validate on `val`: the localizer's test set is drawn from `val`, so
    # a classifier trained on `val` would reintroduce the train/test leak the split avoids.
    tr = VOCCls(a.voc, classes, "train", TRAIN_TF)
    va = VOCCls(a.voc, classes, "val", EVAL_TF)
    logger.info("%d classes | train %d samples | val %d", a.classes, len(tr), len(va))

    dl_tr = DataLoader(tr, batch_size=a.bs, shuffle=True, num_workers=8, pin_memory=True)
    dl_va = DataLoader(va, batch_size=a.bs, shuffle=False, num_workers=8, pin_memory=True)

    m = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    m.classifier[6] = nn.Linear(4096, len(classes))
    m = m.to(dev)
    for p in m.features[:20].parameters():      # freeze the early conv stack
        p.requires_grad = False

    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    best = 0.0

    for ep in range(1, a.epochs + 1):
        m.train()
        for x, y in tqdm(dl_tr, desc=f"ep {ep}/{a.epochs}", leave=False):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            lossf(m(x), y).backward()
            opt.step()
        sched.step()

        m.eval(); ok = n = 0
        with torch.inference_mode():
            for x, y in dl_va:
                x, y = x.to(dev), y.to(dev)
                ok += (m(x).argmax(1) == y).sum().item(); n += y.numel()
        acc = ok / n
        logger.info("ep %2d: val top-1 %.4f", ep, acc)
        if acc > best:
            best = acc
            torch.save(m.state_dict(), out)
            logger.info("  -> saved %s", out)

    logger.info("done. best val top-1 = %.4f -> %s", best, out)


if __name__ == "__main__":
    main()
