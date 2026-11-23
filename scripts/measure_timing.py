#!/usr/bin/env python
"""
measure_timing.py -- per-episode timing for the paper (Appendix D).

Compares ROI caching (one backbone forward per episode, region descriptors ROI-pooled from
the cached map) with per-crop extraction (one backbone forward per step), on test images
spaced evenly over all five classes. Reports
  (a) end-to-end wall clock, as deployed (includes CPU preprocessing, the classifier and the
      DINO prior), repeated --rounds times, and
  (b) GPU compute only for every frozen network, timed with CUDA events on pre-transformed
      tensors.
Batch size 1. Run on an idle GPU with low CPU load; the script prints the load average.

    python scripts/measure_timing.py --voc <VOC> --gpu 0 --rounds 3
Writes <out>/timing.txt
"""
import argparse, os, time, platform
import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--voc", required=True)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--weights", default="./checkpoints/vgg16_5class.pth")
ap.add_argument("--checkpoint", default="./results/results_final/control_best.pth",
                help="policy weights to time (the plain-head control)")
ap.add_argument("--out", default="./results/results_timing")
a = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
from agrqn import core as L     # noqa: E402

dev = torch.device("cuda"); sync = torch.cuda.synchronize
def loadavg():
    try:
        return os.getloadavg()
    except (AttributeError, OSError):
        try:
            import psutil; c = psutil.cpu_percent(interval=1.0); return (c, c, c)
        except ImportError:
            return (float("nan"),) * 3
load0 = loadavg()
L.set_classes(5)
frozen = L.Frozen(a.weights, dev)
_ = frozen.dino; _ = frozen.dinov2
_, _, _, (te_n, te_l) = L.build_splits(a.voc)
step = max(1, len(te_n) // a.n)
idx = list(range(0, len(te_n), step))[:a.n]
names = [te_n[i] for i in idx]
classes = sorted({L.LABEL_TO_CLASS_NAME[te_l[i]] for i in idx})
imgs = {n: np.array(Image.open(os.path.join(a.voc, "JPEGImages", n + ".jpg")).convert("RGB"))
        for n in names}


class LiveView:
    def __init__(self, img, mode):
        self.img, self.mode = img, mode
        self.H, self.W = img.shape[0], img.shape[1]
        self.pool = frozen.pool
        self.pred = frozen.predict_class(img)
        self.hm = frozen.dino_map(img)
        self.feat = frozen.feat(img, 224) if mode == "roi" else None
        self.n_backbone = 1 if mode == "roi" else 0
    def desc(self, box):
        if self.mode == "roi":
            return L.roi_descriptor(self.feat, box, self.W, self.H, self.pool)
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        crop = self.img[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
        f = frozen.feat(crop if crop.size else self.img, 224); self.n_backbone += 1
        return self.pool(f).flatten().cpu().numpy().astype(np.float16)
    def hmcrop(self, box):
        return L.crop_hm(self.hm, box, self.W, self.H)


cfg = L.get_config("control"); cfg["_voc"] = a.voc
net = L.AGRQN(cfg).to(dev)
net.load_state_dict(torch.load(a.checkpoint, map_location=dev)); net.eval()
gts = lambda n, p: L.gt_boxes(L.annotation(n, a.voc), L.VOC_CLASS_IDS[L.LABEL_TO_CLASS_NAME[p]])


@torch.inference_mode()
def episodes(mode):
    for n in names[:5]:
        C = LiveView(imgs[n], mode); L.rollout(net, C, gts(n, C.pred), cfg, dev)
    sync(); T, S, F, I = [], [], [], []
    for n in names:
        sync(); t0 = time.perf_counter()
        C = LiveView(imgs[n], mode); ep = L.rollout(net, C, gts(n, C.pred), cfg, dev)
        sync(); T.append((time.perf_counter() - t0) * 1e3)
        S.append(ep["n_steps"]); F.append(C.n_backbone); I.append(ep["last_iou"])
    return map(np.array, (T, S, F, I))

ROUNDS = []
for _r in range(a.rounds):
    roi_t, roi_s, roi_f, roi_i = episodes("roi")
    crop_t, crop_s, crop_f, crop_i = episodes("crop")
    ROUNDS.append((roi_t.mean(), crop_t.mean()))

# ---- GPU compute only: CUDA events on pre-transformed tensors -------------------------
x224 = torch.stack([L.transform(Image.fromarray(imgs[n])) for n in names[:8]]).to(dev)
def gpu_ms(fn, x):
    with torch.inference_mode():
        for i in range(5): fn(x[i:i+1])
    ts = []
    for r in range(a.reps):
        x1 = x[r % len(x)][None]
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            s.record(); fn(x1); e.record()
        sync(); ts.append(s.elapsed_time(e))
    return np.array(ts)
g_vgg = gpu_ms(lambda t: frozen.backbone(t), x224)
g_clf = gpu_ms(lambda t: frozen.vgg(t), x224)
g_dino = gpu_ms(lambda t: frozen.dino.get_last_selfattention(t), x224)
g_dv2 = gpu_ms(lambda t: frozen.dinov2.forward_features(t), x224)
def gc_fn(t):
    with torch.enable_grad():
        t = t.clone().requires_grad_(False)
        frozen.vgg.zero_grad(set_to_none=True)
        frozen.vgg(t)[0, 0].backward()
for m in frozen.vgg.modules():
    if isinstance(m, torch.nn.ReLU): m.inplace = False
ts = []
for r in range(5 + a.reps):
    x1 = x224[r % len(x224)][None]
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); gc_fn(x1); e.record(); sync()
    if r >= 5: ts.append(s.elapsed_time(e))
g_gc = np.array(ts)
# one policy step on GPU: ROI pool + head forward
feat = frozen.backbone(x224[:1]); box = [0, 0, 224, 224]
hist = torch.zeros(1, L.HIST_DIM, device=dev); hm = torch.zeros(1, 1, L.HM_SIZE, L.HM_SIZE, device=dev)
def step_fn(_):
    d = frozen.pool(feat).flatten()[None]
    net(d, hist, hm)
g_step = gpu_ms(step_fn, x224)

load1 = loadavg()
ms = lambda v: f"{v.mean():7.2f} +/- {v.std(ddof=1):5.2f} ms"
out = f"""TIMING -- {torch.cuda.get_device_name(0)} | torch {torch.__version__} CUDA {torch.version.cuda} | {platform.platform()} | CPU {platform.processor() or platform.machine()}
rounds (ROI ms, per-crop ms, speedup): {"; ".join(f"{r:.1f}, {c:.1f}, {c/r:.2f}x" for r, c in ROUNDS)}
batch size 1 | n={len(names)} test images over classes {classes}
CPU load average (1/5/15 min) at start {load0[0]:.1f}/{load0[1]:.1f}/{load0[2]:.1f}, end {load1[0]:.1f}/{load1[1]:.1f}/{load1[2]:.1f}  | CPUs: {os.cpu_count()}
GPU utilisation by other processes must be ~0. If CPU load is high, trust section (b) over (a).

(a) END-TO-END wall clock, VGG-16 descriptor, plain head (control), incl. CPU preprocessing,
    classifier and DINO prior:
  ROI-cached episode : {ms(roi_t)}  steps {roi_s.mean():.1f}  backbone fwd/episode {roi_f.mean():.1f}  mIoU(subset) {roi_i.mean():.4f}
  per-crop   episode : {ms(crop_t)}  steps {crop_s.mean():.1f}  backbone fwd/episode {crop_f.mean():.1f}  mIoU(subset) {crop_i.mean():.4f}
  speedup            : {crop_t.mean()/roi_t.mean():.2f}x

(b) GPU COMPUTE ONLY (CUDA events, pre-transformed 224x224 tensors, {a.reps} reps):
  VGG-16 conv backbone (descriptor)  : {ms(g_vgg)}
  VGG-16 full (5-way classifier)     : {ms(g_clf)}
  DINO ViT-S/16 attention (prior)    : {ms(g_dino)}
  DINOv2 ViT-S/14 tokens (descriptor): {ms(g_dv2)}
  Grad-CAM (VGG fwd+bwd)             : {ms(g_gc)}
  one policy step (ROI pool + head)  : {ms(g_step)}
"""
os.makedirs(a.out, exist_ok=True)
open(f"{a.out}/timing.txt", "w").write(out); print(out)
