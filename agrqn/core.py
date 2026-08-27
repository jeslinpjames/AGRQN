"""Shared components for AG-RQN: VOC data loading, the leak-free train/ckpt/test split,
box transforms, frozen backbones (VGG-16, DINO, DINOv2), the feature cache, the Q-network,
and the evaluation/rollout loop.

Used by both ``scripts/precache.py`` and ``scripts/train_agrqn.py``.

Design note on CONFIGS
-----------------------
Across every architecture variant we trained, the budget-exhaustion rate tracks the
localization failure rate almost exactly:

    exhaustion_rate  ~=  1 - Acc@0.5   (to within ~0.01, for a converged policy)

In other words, the agent triggers if and only if it succeeds: exhaustion is a calibrated
abstention signal, not a stopping defect. It can only be reduced by making the agent more
accurate (the "accuracy" track below), or by forcing it to commit to boxes it already
suspects are bad, which trades trigger precision for coverage (the "incentive" track,
included as a controlled negative result). See the paper for the full argument and the
`analyze_calibration.py` script for the statistical tests that support it.
"""
import logging
import math
import os
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.models import vgg16

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- constants
VOC20 = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
         "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
         "train", "tvmonitor"]
VOC5 = ["bicycle", "aeroplane", "horse", "cat", "dog"]
_ALL_IDS = {n: i + 1 for i, n in enumerate(VOC20)}      # VOC ids are 1-indexed, in this order
_ID_TO_NAME = {v: k for k, v in _ALL_IDS.items()}       # reverse lookup, all 20 classes always

SELECTED_CLASSES = VOC5                                  # switched by set_classes()
VOC_CLASS_IDS = {c: _ALL_IDS[c] for c in SELECTED_CLASSES}
CLASS_NAME_TO_LABEL = {c: i for i, c in enumerate(SELECTED_CLASSES)}
LABEL_TO_CLASS_NAME = {i: c for c, i in CLASS_NAME_TO_LABEL.items()}


def set_classes(n):
    """Switch the whole pipeline between the 5-class and 20-class settings."""
    global SELECTED_CLASSES, VOC_CLASS_IDS, CLASS_NAME_TO_LABEL, LABEL_TO_CLASS_NAME
    SELECTED_CLASSES = VOC20 if n == 20 else VOC5
    VOC_CLASS_IDS = {c: _ALL_IDS[c] for c in SELECTED_CLASSES}
    CLASS_NAME_TO_LABEL = {c: i for i, c in enumerate(SELECTED_CLASSES)}
    LABEL_TO_CLASS_NAME = {i: c for c, i in CLASS_NAME_TO_LABEL.items()}
    return SELECTED_CLASSES


NUM_ACTIONS, TRIGGER = 9, 9
ALPHA, MIN_BOX = 0.2, 24
ACTION_NAMES = ["right", "left", "up", "down", "bigger", "smaller", "fatter", "taller", "trigger"]

HISTORY_LEN = 9
HIST_DIM = HISTORY_LEN * NUM_ACTIONS
HM_SIZE = 64
MAX_STEPS = 20
IOU_THRESH = 0.5

# The paper's evaluation protocol.
N_TEST_PER_CLASS = 200      # 20-class runs override this (see train_agrqn.py --classes)
N_CKPT_PER_CLASS = 40
EVAL_SEED, SPLIT_SEED = 123, 2024

transform = T.Compose([
    T.Resize((224, 224)), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform448 = T.Compose([
    T.Resize((448, 448)), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ============================================================================ CONFIGS
# Two tracks (see the module docstring):
#   accuracy  -- the only track that can genuinely reduce exhaustion, since exhaustion is
#                the failure rate. Each config attacks a specific, known failure source.
#   incentive -- forces the agent to commit. Expected to cut exhaustion while destroying
#                trigger precision; included as a controlled negative result.
#
# keys:
#   eta          terminal trigger reward magnitude
#   gamma        discount
#   shaping      'sign' (simple +/-1 shaping) | 'pbrs' (potential-based, Ng et al. 1999)
#   expiry       'pass' (running out of budget with IoU>=0.5 still pays +eta even without
#                triggering) | 'fail' (always -eta) | 'zero' (no terminal reward)
#   step_penalty per-move cost
#   feat         '7x7' (224px backbone) | '14x14' (448px backbone -> finer ROI grid)
#   hidden       fusion-layer width
#   dueling      dueling Q-head (decouples V(s) from A(s,a); reduces the max-operator bias
#                that systematically over-values the 8 movement actions vs. the 1 trigger)
#   aux_iou      auxiliary IoU-regression head
#   stop_mode    'argmax' | 'aux' (trigger when predicted IoU >= aux_tau)
CONFIGS = {
    # ---- accuracy track ----
    "control": dict(track="accuracy"),
    "dueling": dict(track="accuracy", dueling=True),
    "wide": dict(track="accuracy", hidden=1024),
    "hires": dict(track="accuracy", feat="14x14"),
    "combo_acc": dict(track="accuracy", dueling=True, hidden=1024, feat="14x14"),

    # ---- incentive track (negative-result controls) ----
    "expiry_fail": dict(track="incentive", expiry="fail"),
    "steppen02": dict(track="incentive", step_penalty=0.2),
    "eta10": dict(track="incentive", eta=10.0),
    "pbrs": dict(track="incentive", shaping="pbrs"),

    # ---- decoupled stopping ----
    # Train an IoU head and stop on it instead of on argmax(Q). This lets the exhaustion
    # rate be dialed at inference with a single threshold, without retraining -- an explicit
    # coverage/precision trade rather than one hidden inside the reward function.
    "aux_stop": dict(track="accuracy", aux_iou=True, stop_mode="aux", aux_tau=0.5),

    # ---- best accuracy-track architecture ----
    # `dueling` and `wide` each help for independent reasons: dueling cancels the
    # max-operator bias that over-values the 8 movement actions against the 1 trigger;
    # `wide` removes the 512-unit bottleneck that made Q(s, TRIGGER) -- a terminal,
    # essentially binary "is my box good enough?" value -- hard to represent. `combo_acc`
    # combines both with `hires`, the weakest of the three; `duel_wide` is the clean
    # combination of just the two that help.
    "duel_wide": dict(track="accuracy", dueling=True, hidden=1024),

    # ---- prior study on the clean split (paper: prior redundancy result) ----
    # `prior` selects the attention stream fed to the agent alongside the descriptor:
    #   dino     : class-agnostic ViT self-attention (the paper's model)
    #   gradcam  : from the same VGG that supplies the descriptor -- redundant and coarse
    #   layercam : same classifier, but spatially sharper -- isolates the coarseness half
    #   none     : attention stream zeroed at training time (retraining ablation)
    "prior_dino": dict(track="prior", dueling=True, hidden=1024, prior="dino"),
    "prior_gradcam": dict(track="prior", dueling=True, hidden=1024, prior="gradcam"),
    "prior_layercam": dict(track="prior", dueling=True, hidden=1024, prior="layercam"),
    "prior_none": dict(track="prior", dueling=True, hidden=1024, prior="none"),

    # ---- retraining ablations on the final architecture (paper: ablation table) ----
    # These are trained from scratch: they remove a component and ask whether the converged
    # policy needed it, as opposed to the inference-time zeroing ablations above (which
    # measure reliance of an already-trained policy and need no retraining).
    "no_history": dict(track="ablation", dueling=True, hidden=1024, history=False),
    "single_dqn": dict(track="ablation", dueling=True, hidden=1024, double_q=False),

    # ---- redundancy prediction: swap the descriptor family ----
    # If the DINO prior is redundant because it shares a family with the descriptor, then
    # switching the descriptor to DINOv2 should make the prior stop helping, mirroring the
    # Grad-CAM result above. This is the paper's own diagnosis stated as a falsifiable
    # prediction.
    "desc_dinov2": dict(track="prediction", dueling=True, hidden=1024,
                         desc="dinov2", prior="dino"),
    "desc_dinov2_none": dict(track="prediction", dueling=True, hidden=1024,
                              desc="dinov2", prior="none"),

    # ---- 20-class scaling runs ----
    # DINOv2 descriptor + DINOv2 linear-probe classifier, the strongest architecture found
    # on the 5-class study. Two runs test whether the prior stays redundant when the task
    # scales to all 20 VOC classes.
    "voc20_dino": dict(track="scale", dueling=True, hidden=1024,
                        desc="dinov2", prior="dino"),
    "voc20_none": dict(track="scale", dueling=True, hidden=1024,
                        desc="dinov2", prior="none"),

    # ---- ablations on the shipped (DINOv2-descriptor) model ----
    "dinov2_nohist": dict(track="ablation", dueling=True, hidden=1024,
                           desc="dinov2", prior="dino", history=False),
    "dinov2_singleq": dict(track="ablation", dueling=True, hidden=1024,
                            desc="dinov2", prior="dino", double_q=False),

    # ---- redundancy prediction, other direction ----
    # Grad-CAM is redundant with a VGG descriptor (near-tied). With a DINOv2 descriptor it
    # is a different network family entirely, so it should no longer be redundant and
    # should help. Confirming this in both directions supports independence, not just
    # architecture identity, as the operative variable.
    "dinov2_gradcam": dict(track="prediction", dueling=True, hidden=1024,
                            desc="dinov2", prior="gradcam"),
    # Control for the above: same architecture, VGG descriptor, no prior.
    "desc_vgg_none": dict(track="prediction", dueling=True, hidden=1024,
                           desc="vgg", prior="none"),
}

DEFAULTS = dict(eta=3.0, gamma=0.90, shaping="sign", expiry="pass", step_penalty=0.0,
                feat="7x7", hidden=512, dueling=False, aux_iou=False,
                stop_mode="argmax", aux_tau=0.5, track="accuracy", prior="dino",
                history=True, double_q=True, desc="vgg")


def get_config(name):
    if name not in CONFIGS:
        raise SystemExit(f"unknown config '{name}'. choices: {list(CONFIGS)}")
    return {**DEFAULTS, **CONFIGS[name]}


# ============================================================================ VOC
def load_ids(data_set_name, path_voc):
    with open(os.path.join(path_voc, "ImageSets", "Main", data_set_name + ".txt")) as f:
        out = []
        for line in f:
            p = line.split()
            if len(p) == 1 or "-1" not in p[1]:
                out.append(p[0])
    return out


def voc_class_id(name):
    tbl = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5, "bus": 6, "car": 7,
           "cat": 8, "chair": 9, "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
           "motorbike": 14, "person": 15, "pottedplant": 16, "sheep": 17, "sofa": 18,
           "train": 19, "tvmonitor": 20}
    return tbl.get(name, -1)


def annotation(name, voc):
    tree = ET.parse(os.path.join(voc, "Annotations", name + ".xml"))
    rows = []
    for obj in tree.getroot().findall("object"):
        bb = obj.find("bndbox")
        rows.append([voc_class_id(obj.find("name").text),
                     float(bb.find("xmin").text), float(bb.find("xmax").text),
                     float(bb.find("ymin").text), float(bb.find("ymax").text)])
    return np.array(rows, np.float32) if rows else np.zeros((0, 5), np.float32)


def gt_boxes(ann, cid):
    return [[int(r[1]), int(r[3]), int(r[2]), int(r[4])] for r in ann if int(r[0]) == cid]


def largest_object_class(ann):
    """Class id of the single largest-area GT instance in this image, any class -- ignores
    the classifier and the image's nominal label entirely. This is the convention
    class-agnostic localizers (LOST, TokenCut) implicitly assume: score against the most
    prominent object, not a specific label."""
    if len(ann) == 0:
        return None
    areas = (ann[:, 2] - ann[:, 1]) * (ann[:, 4] - ann[:, 3])
    return int(ann[int(np.argmax(areas)), 0])


def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    ua = (max(0, a[2] - a[0]) * max(0, a[3] - a[1])
          + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def max_iou(box, gts):
    return max([iou(box, g) for g in gts], default=0.0)


def transform_box(box, a, W, H, alpha=ALPHA):
    x0, y0, x1, y1 = box
    aw, ah = alpha * (x1 - x0), alpha * (y1 - y0)
    if a == 1: x0 += aw; x1 += aw
    elif a == 2: x0 -= aw; x1 -= aw
    elif a == 3: y0 -= ah; y1 -= ah
    elif a == 4: y0 += ah; y1 += ah
    elif a == 5: x0 -= aw; x1 += aw; y0 -= ah; y1 += ah
    elif a == 6: x0 += aw; x1 -= aw; y0 += ah; y1 -= ah
    elif a == 7: y0 += ah; y1 -= ah
    elif a == 8: x0 += aw; x1 -= aw
    x0 = max(0.0, min(x0, W - MIN_BOX)); y0 = max(0.0, min(y0, H - MIN_BOX))
    x1 = min(float(W), max(x0 + MIN_BOX, x1)); y1 = min(float(H), max(y0 + MIN_BOX, y1))
    return [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]


def crop_hm(hm, box, W, H, size=HM_SIZE):
    hh, hw = hm.shape
    x0 = max(0, int(box[0] / W * hw)); y0 = max(0, int(box[1] / H * hh))
    x1 = min(hw, max(x0 + 1, int(box[2] / W * hw)))
    y1 = min(hh, max(y0 + 1, int(box[3] / H * hh)))
    sub = hm[y0:y1, x0:x1]
    if sub.size == 0:
        sub = hm
    return cv2.resize(sub, (size, size)).astype(np.float32)


def push_hist(hist, a):
    oh = np.zeros(NUM_ACTIONS, np.float32); oh[a - 1] = 1.0
    h = hist.reshape(HISTORY_LEN, NUM_ACTIONS)
    h = np.roll(h, -1, axis=0); h[-1] = oh
    return h.flatten()


# ============================================================================ SPLITS
def build_splits(path_voc):
    """The leak-free train/checkpoint/test protocol used throughout the paper.

    TEST : 200/class from val, seed 123  -- the paper's reported test set, unchanged.
    CKPT :  40/class from val, disjoint  -- early-stopping / model-selection only.
    TRAIN: trainval minus both.          -- everything else (~2,870 images).
    """
    rng = random.Random(EVAL_SEED)
    test_n, test_l = [], []
    for c in SELECTED_CLASSES:
        pool = load_ids(f"{c}_val", path_voc)
        for n in rng.sample(pool, min(N_TEST_PER_CLASS, len(pool))):
            test_n.append(n); test_l.append(CLASS_NAME_TO_LABEL[c])

    rng2 = random.Random(SPLIT_SEED)
    ckpt_n, ckpt_l = [], []
    for c in SELECTED_CLASSES:
        pool = [n for n in load_ids(f"{c}_val", path_voc) if n not in set(test_n)]
        for n in rng2.sample(pool, min(N_CKPT_PER_CLASS, len(pool))):
            if n not in ckpt_n:
                ckpt_n.append(n); ckpt_l.append(CLASS_NAME_TO_LABEL[c])

    excl = set(test_n) | set(ckpt_n)
    pool = {c: [n for n in load_ids(f"{c}_trainval", path_voc) if n not in excl]
            for c in SELECTED_CLASSES}

    all_train = sorted({n for c in SELECTED_CLASSES for n in pool[c]})
    assert not (set(all_train) & set(test_n)), "train/test leak"
    assert not (set(ckpt_n) & set(test_n)), "ckpt/test leak"
    assert not (set(all_train) & set(ckpt_n)), "train/ckpt leak"
    return pool, all_train, (ckpt_n, ckpt_l), (test_n, test_l)


# ============================================================================ BACKBONES
class Frozen:
    """The frozen perception stack: VGG-16 (descriptor + classifier) and DINO ViT-S/16."""

    def __init__(self, weights, device):
        self.device = device
        m = vgg16(weights=None)
        m.classifier[6] = nn.Linear(4096, len(SELECTED_CLASSES))
        m.load_state_dict(torch.load(weights, map_location=device))
        self.vgg = m.to(device).eval()
        self.backbone = self.vgg.features
        self.pool = nn.AdaptiveAvgPool2d((4, 4)).to(device)
        self._dino = None

    @property
    def dinov2(self):
        """DINOv2 ViT-S/14 patch tokens, used as an alternative region descriptor.
        Reshaped to a 16x16x384 grid so roi_descriptor() can pool it exactly like VGG's."""
        if getattr(self, "_dv2", None) is None:
            self._dv2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            self._dv2.eval().to(self.device)
        return self._dv2

    @torch.inference_mode()
    def feat_dinov2(self, img):
        t = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(
            Image.fromarray(img)).unsqueeze(0).to(self.device)
        f = self.dinov2.forward_features(t)["x_norm_patchtokens"]   # (1, 256, 384)
        s = int(round(f.shape[1] ** 0.5))
        return f.reshape(1, s, s, -1).permute(0, 3, 1, 2)           # (1, 384, 16, 16)

    @property
    def dino(self):
        if self._dino is None:
            self._dino = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
            self._dino.eval().to(self.device)
        return self._dino

    def load_dinov2_clf(self, path):
        """A linear probe on the frozen DINOv2 [CLS] embedding. Replaces the VGG classifier
        for the 20-class runs, where VGG's ~60-75% accuracy would dominate the result (every
        misclassification is an automatic IoU of 0)."""
        ck = torch.load(path, map_location=self.device)
        self._clf = nn.Linear(ck["dim"], len(ck["classes"])).to(self.device)
        self._clf.load_state_dict(ck["state_dict"])
        self._clf.eval()
        logger.info("loaded DINOv2 linear probe: %d classes", len(ck["classes"]))
        return self._clf

    @torch.inference_mode()
    def predict_class(self, img):
        t = transform(Image.fromarray(img)).unsqueeze(0).to(self.device)
        if getattr(self, "_clf", None) is not None:      # DINOv2 linear probe
            return int(self._clf(self.dinov2(t)).argmax(1).item())
        return int(self.vgg(t).argmax(1).item())         # VGG-16 head

    @torch.inference_mode()
    def feat(self, img, size=224):
        tf = transform if size == 224 else transform448
        t = tf(Image.fromarray(img)).unsqueeze(0).to(self.device)
        return self.backbone(t)                      # 512x7x7  or  512x14x14

    @torch.inference_mode()
    def dino_map(self, img):
        t = transform(Image.fromarray(img)).unsqueeze(0).to(self.device)
        att = self.dino.get_last_selfattention(t)
        side = int(round(math.sqrt(att.shape[-1] - 1)))
        cam = att[0, :, 0, 1:].reshape(-1, side, side).mean(0).cpu().numpy()
        cam = cam - cam.min(); m = cam.max()
        return cv2.resize(cam / m if m > 0 else cam, (224, 224)).astype(np.float32)


# ============================================================================ CAM PRIORS
class GradCAMGenerator:
    """Grad-CAM from the last conv block: 7x7 -> blobby. It is also produced by the very
    classifier whose pooled features form the region descriptor, so the two state streams
    carry overlapping information -- redundant and coarse."""

    def __init__(self, vgg, device):
        self.model, self.device = vgg, device
        for m in self.model.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
        self.act = self.grad = None
        self.model.features[28].register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        self.act = out.detach()
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "grad", g.detach()))

    def generate(self, img, cls):
        t = transform(Image.fromarray(img)).unsqueeze(0).to(self.device)
        self.act = self.grad = None
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            self.model(t)[0, cls].backward()
        w = self.grad.mean(dim=[0, 2, 3])
        cam = torch.relu((self.act[0] * w[:, None, None]).sum(0)).cpu().numpy()
        m = cam.max()
        return cv2.resize(cam / m if m > 0 else cam, (224, 224)).astype(np.float32)


class LayerCAMGenerator:
    """LayerCAM fuses several conv stages -> spatially sharper than Grad-CAM, but still from
    the same classifier. It fixes the coarseness half of the diagnosis while leaving the
    redundancy half untouched, which is why it recovers only part of the gap that DINO
    recovers in full."""

    def __init__(self, vgg, device, layers=(15, 22, 28)):
        self.model, self.device, self.layers = vgg, device, layers
        for m in self.model.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False
        self.acts, self.grads = {}, {}
        for lid in layers:
            self.model.features[lid].register_forward_hook(self._mk(lid))

    def _mk(self, lid):
        def hook(mod, inp, out):
            self.acts[lid] = out.detach()
            if out.requires_grad:
                out.register_hook(lambda g, l=lid: self.grads.__setitem__(l, g.detach()))
        return hook

    def generate(self, img, cls):
        t = transform(Image.fromarray(img)).unsqueeze(0).to(self.device)
        self.acts.clear(); self.grads.clear()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            self.model(t)[0, cls].backward()
        fused = None
        for lid in self.layers:
            a, g = self.acts[lid][0], self.grads[lid][0]
            cam = torch.relu((torch.relu(g) * a).sum(0))
            cam = cam / cam.max() if cam.max() > 0 else cam
            cam = F.interpolate(cam[None, None], size=(224, 224),
                                 mode="bilinear", align_corners=False)[0, 0]
            fused = cam if fused is None else torch.maximum(fused, cam)
        cam = fused.cpu().numpy(); m = cam.max()
        return (cam / m if m > 0 else cam).astype(np.float32)


def roi_descriptor(feat, box, W, H, pool):
    _, _, fh, fw = feat.shape
    x0 = max(0, min(fw - 1, int(box[0] / W * fw)))
    y0 = max(0, min(fh - 1, int(box[1] / H * fh)))
    x1 = max(x0 + 1, min(fw, int(round(box[2] / W * fw))))
    y1 = max(y0 + 1, min(fh, int(round(box[3] / H * fh))))
    return pool(feat[:, :, y0:y1, x0:x1]).flatten().cpu().numpy().astype(np.float16)


class CacheRAM:
    """The whole feature cache, resident in memory. Built once; then the training loop never
    touches the filesystem again.

    feat -> GPU (small: 512x7x7 fp16 = 50 KB/img).  hm -> RAM (cv2 crops it on CPU).
    CAM priors stay on disk: they are 10x larger and only the prior-study runs need them.
    """

    def __init__(self, names, cache_dir, feat_key, device, prior="dino"):
        from tqdm import tqdm as _tqdm
        self.feat, self.hm, self.pred, self.wh = {}, {}, {}, {}
        self.cam = {}
        self.device, self.prior, self.cache_dir = device, prior, cache_dir
        k = "feat7" if feat_key == "7x7" else "feat14"
        for n in _tqdm(names, desc="preloading cache into RAM", leave=False):
            z = np.load(os.path.join(cache_dir, n + ".npz"))
            self.feat[n] = torch.from_numpy(z[k].astype(np.float32)).unsqueeze(0).to(device)
            self.pred[n] = int(z["pred"])
            self.wh[n] = (int(z["wh"][0]), int(z["wh"][1]))
            if prior == "dino":
                self.hm[n] = z["hm"].astype(np.float32)
            elif prior == "none":
                self.hm[n] = np.zeros((224, 224), np.float32)
            else:                                  # gradcam / layercam: class-conditional
                zc = np.load(os.path.join(cache_dir, n + "_cam.npz"))
                self.cam[n] = {c: zc[f"{prior}_{c}"].astype(np.float32)
                               for c in range(len(SELECTED_CLASSES))}
        gb = sum(t.numel() * 4 for t in self.feat.values()) / 1e9
        logger.info("cache resident: %d images, %.2f GB on GPU, zero disk I/O ahead",
                     len(names), gb)

    def get(self, name, pool, cls=None):
        return _RamView(self, name, pool, cls)


class _RamView:
    """Same interface as Cached, but every field is already in memory."""
    __slots__ = ("feat", "hm", "pred", "W", "H", "pool")

    def __init__(self, C, name, pool, cls):
        self.feat = C.feat[name]
        self.pred = C.pred[name]
        self.W, self.H = C.wh[name]
        self.pool = pool
        if C.prior in ("dino", "none"):
            self.hm = C.hm[name]
        else:
            self.hm = C.cam[name][self.pred if cls is None else cls]

    def desc(self, box): return roi_descriptor(self.feat, box, self.W, self.H, self.pool)
    def hmcrop(self, box): return crop_hm(self.hm, box, self.W, self.H)


class Cached:
    """One pre-computed image, loaded from disk. Kept for compatibility; the training loop
    now uses CacheRAM instead.

    `prior` picks the attention stream:
      dino     -> the class-agnostic map in the main cache (one per image)
      gradcam  -> class-conditional map from <name>_cam.npz, keyed on `cls`
      layercam -> ditto
      none     -> a zero map (the retraining ablation: train without the prior)

    At training time `cls` is the episode's true class; at eval time it is the classifier's
    prediction, so classifier errors corrupt the CAM priors but cannot touch DINO. That
    asymmetry is part of the paper's argument for an independent prior.
    """

    def __init__(self, name, cache_dir, feat_key, device, pool, prior="dino", cls=None):
        z = np.load(os.path.join(cache_dir, name + ".npz"))
        self.feat = torch.from_numpy(
            z["feat7" if feat_key == "7x7" else "feat14"].astype(np.float32)
        ).unsqueeze(0).to(device)
        self.pred = int(z["pred"])
        self.W, self.H = [int(v) for v in z["wh"]]
        self.pool = pool
        if prior == "dino":
            self.hm = z["hm"].astype(np.float32)
        elif prior == "none":
            self.hm = np.zeros((224, 224), np.float32)
        else:
            c = self.pred if cls is None else cls
            zc = np.load(os.path.join(cache_dir, name + "_cam.npz"))
            self.hm = zc[f"{prior}_{c}"].astype(np.float32)

    def desc(self, box): return roi_descriptor(self.feat, box, self.W, self.H, self.pool)
    def hmcrop(self, box): return crop_hm(self.hm, box, self.W, self.H)


# ============================================================================ MODEL
class HistAttn(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.s = nn.Linear(h, 1)

    def forward(self, seq):
        return (seq * torch.softmax(self.s(seq), 1)).sum(1)


class AGRQN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        desc_dim = (384 if cfg.get("desc") == "dinov2" else 512) * 4 * 4
        hid = cfg["hidden"]
        self.desc_fc = nn.Linear(desc_dim, 512)
        self.hm_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)))
        self.hm_fc = nn.Linear(32 * 4 * 4, 128)
        self.gru = nn.GRU(NUM_ACTIONS, 64, batch_first=True)
        self.attn = HistAttn(64)
        self.fc1 = nn.Linear(512 + 128 + 64, hid)
        self.drop = nn.Dropout(0.2)
        if cfg["dueling"]:
            self.val = nn.Linear(hid, 1)
            self.adv = nn.Linear(hid, NUM_ACTIONS)
        else:
            self.out = nn.Linear(hid, NUM_ACTIONS)
        if cfg["aux_iou"]:
            self.iou_out = nn.Linear(hid, 1)

    def forward(self, desc, hist, hm):
        d = F.relu(self.desc_fc(desc))
        h = F.relu(self.hm_fc(self.hm_cnn(hm).flatten(1)))
        if not self.cfg["history"]:
            hist = torch.zeros_like(hist)      # ablation: no temporal context at all
        g, _ = self.gru(hist.view(hist.size(0), HISTORY_LEN, NUM_ACTIONS))
        x = self.drop(F.relu(self.fc1(torch.cat([d, h, self.attn(g)], 1))))
        if self.cfg["dueling"]:
            v, a = self.val(x), self.adv(x)
            q = v + a - a.mean(1, keepdim=True)
        else:
            q = self.out(x)
        aux = torch.sigmoid(self.iou_out(x)).squeeze(1) if self.cfg["aux_iou"] else None
        return q, aux


# ============================================================================ REWARD
def movement_reward(cfg, io, inw):
    """'sign' : a simple +/-1 shaping term, not potential-based, so it may distort the
                optimum.
       'pbrs' : gamma*Phi(s') - Phi(s), Phi = 5*IoU, Phi(terminal)=0. Policy-invariant
                (Ng, Harada & Russell 1999); telescopes to a constant, so it cannot be
                farmed by looping actions."""
    if cfg["shaping"] == "sign":
        return (1.0 if inw > io else -1.0) - cfg["step_penalty"]
    return cfg["gamma"] * 5.0 * inw - 5.0 * io - cfg["step_penalty"]


def terminal_reward(cfg, final_iou, triggered):
    """`expiry` controls what happens when the step budget runs out without a trigger.
    'pass' (the default): running out of budget with a good box still pays +eta, so the
    agent can collect the full terminal reward without ever committing. 'fail'/'zero' close
    that path, at the cost of forcing commitment on boxes the agent already knows are bad."""
    good = final_iou >= IOU_THRESH
    if triggered:
        r = cfg["eta"] if good else -cfg["eta"]
    else:
        r = {"pass": cfg["eta"] if good else -cfg["eta"],
             "fail": -cfg["eta"],
             "zero": 0.0}[cfg["expiry"]]
    if cfg["shaping"] == "pbrs":
        r += 0.0 - 5.0 * final_iou               # Phi(terminal) = 0
    return r


# ============================================================================ EVAL
@torch.inference_mode()
def rollout(net, C, gts, cfg, device):
    box = [0, 0, C.W, C.H]
    hist = np.zeros(HIST_DIM, np.float32)
    boxes, ious, confs, acts = [box.copy()], [max_iou(box, gts)], [], []
    triggered, stop_t = False, None
    for t in range(MAX_STEPS):
        q, aux = net(
            torch.as_tensor(C.desc(box).astype(np.float32)).unsqueeze(0).to(device),
            torch.as_tensor(hist).unsqueeze(0).to(device),
            torch.as_tensor(C.hmcrop(box)).unsqueeze(0).unsqueeze(0).to(device))
        qn = q[0].cpu().numpy()
        aux_v = float(aux[0].item()) if aux is not None else None
        confs.append(aux_v if aux_v is not None else float(qn[TRIGGER - 1] - qn[:TRIGGER - 1].max()))

        if cfg["stop_mode"] == "aux" and aux_v is not None and aux_v >= cfg["aux_tau"]:
            a = TRIGGER
        else:
            a = int(np.argmax(qn)) + 1
        acts.append(a)
        if a == TRIGGER:
            triggered, stop_t = True, t
            break
        box = transform_box(box, a, C.W, C.H)
        hist = push_hist(hist, a)
        boxes.append(box.copy()); ious.append(max_iou(box, gts))
    if stop_t is None:
        stop_t = len(boxes) - 1
    last_t = min(stop_t, len(boxes) - 1)
    bt = min(int(np.argmax(confs[:min(stop_t, len(confs) - 1) + 1])) if confs else 0, len(boxes) - 1)
    return dict(boxes=boxes, ious=ious, confs=confs, actions=acts,
                triggered=triggered, n_steps=len(acts), best_t=bt, last_t=last_t,
                last_iou=ious[last_t], final_iou=ious[bt],
                peak_iou=float(max(ious[:last_t + 1])))


@torch.inference_mode()
def evaluate(net, names, labels, cfg, frozen, cache_dir, device):
    net.eval()
    recs = []
    for name, lab in zip(names, labels):
        C = (cache_dir.get(name, frozen.pool)            # CacheRAM -> eval keys CAMs on pred
             if isinstance(cache_dir, CacheRAM)
             else Cached(name, cache_dir, cfg["feat"], device, frozen.pool,
                         prior=cfg["prior"]))
        ann = annotation(name, cfg["_voc"])
        gts = gt_boxes(ann, VOC_CLASS_IDS[LABEL_TO_CLASS_NAME[C.pred]])
        gts_t = gt_boxes(ann, VOC_CLASS_IDS[LABEL_TO_CLASS_NAME[lab]])
        ep = rollout(net, C, gts, cfg, device)
        ar = [(g[2] - g[0]) * (g[3] - g[1]) for g in gts_t]

        # Oracle-class scoring: the same trajectory and box-selection indices, re-scored
        # against the true-class GT boxes instead of the classifier's predicted-class GT
        # boxes. Isolates localization quality from classifier error.
        ious_oracle = [max_iou(b, gts_t) for b in ep["boxes"]]
        last_t_o = min(ep["last_t"], len(ious_oracle) - 1)

        # Area-oracle scoring: score against whichever object is largest by area,
        # class-agnostic.
        largest_cid = largest_object_class(ann)
        gts_area = gt_boxes(ann, largest_cid) if largest_cid is not None else []
        ious_area = ([max_iou(b, gts_area) for b in ep["boxes"]]
                     if gts_area else [0.0] * len(ep["boxes"]))
        last_t_a = min(ep["last_t"], len(ious_area) - 1)

        recs.append(dict(name=name, cls_true=LABEL_TO_CLASS_NAME[lab],
                          cls_pred=LABEL_TO_CLASS_NAME[C.pred],
                          area_cls=(_ID_TO_NAME.get(largest_cid) if largest_cid else None),
                          correct_cls=(C.pred == lab), n_inst=len(gts_t),
                          gt_frac=(max(ar) / (C.W * C.H) if ar else 0.0),
                          final_iou_oracle=ious_oracle[ep["best_t"]],
                          last_iou_oracle=ious_oracle[last_t_o],
                          peak_iou_oracle=float(max(ious_oracle[:last_t_o + 1])),
                          final_iou_area=ious_area[ep["best_t"]],
                          last_iou_area=ious_area[last_t_a],
                          peak_iou_area=float(max(ious_area[:last_t_a + 1])),
                          **ep))
    return recs


def summarize(recs, key="last_iou"):
    per = {c: np.array([r[key] for r in recs if r["cls_true"] == c]) for c in SELECTED_CLASSES}
    v = np.array([r[key] for r in recs])
    st = np.array([r["n_steps"] for r in recs])
    tg = np.array([r["triggered"] for r in recs])
    out = dict(
        miou=float(np.mean([per[c].mean() for c in SELECTED_CLASSES if len(per[c])])),
        acc05=float((v >= IOU_THRESH).mean()),          # == CorLoc
        mean_steps=float(st.mean()), median_steps=float(np.median(st)),
        trigger_rate=float(tg.mean()), exhaust_rate=float((~tg).mean()),
        trigger_ppv=float((v[tg] >= IOU_THRESH).mean()) if tg.any() else float("nan"),
        exhaust_ppv=float((v[~tg] >= IOU_THRESH).mean()) if (~tg).any() else float("nan"),
        per_class={c: float(per[c].mean()) for c in SELECTED_CLASSES if len(per[c])},
    )
    # The core relationship the paper examines: exhaustion should equal the failure rate.
    # If |gap| is small, the agent triggers iff it succeeds and exhaustion is a symptom of
    # imperfect accuracy, not a stopping defect.
    out["exh_minus_failrate"] = out["exhaust_rate"] - (1.0 - out["acc05"])
    return out
