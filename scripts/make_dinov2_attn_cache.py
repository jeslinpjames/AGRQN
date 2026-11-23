#!/usr/bin/env python
"""
make_dinov2_attn_cache.py -- the feature cache for the same-backbone prior
(config desc_dinov2_selfattn; paper: Section 4.5, Table 6).

The question: if the attention prior comes from the SAME DINOv2 forward pass that produces
the region descriptor (instead of a separate DINO ViT-S/16), does it still help?

This copies every image in the DINOv2 cache into a new cache, keeping the DINOv2 descriptor
(feat7), feat14, pred and wh byte-for-byte, and replacing ONLY the prior map `hm`:
    old hm : DINO ViT-S/16 [CLS] self-attention, last block, head-mean   (independent backbone)
    new hm : DINOv2 ViT-S/14 [CLS] self-attention, last block, head-mean (same backbone/pass)
Post-processing is identical to Frozen.dino_map (min-max normalise, resize to 224x224).

Self-checks printed at the end:
  * desc max|diff|: the DINOv2 patch tokens from THIS pass vs the cached feat7. Should be at
    fp16 rounding level (~1e-2 or less). If large, the cache was built differently -> stop.
  * map correlation old vs new: informative only (DINOv2 maps are known to be noisier).
Writes report/attn_compare/*.png (old vs new maps for 8 test images) for a visual check.

Usage (~10-20 min on one GPU), from the repository root:
    python scripts/make_dinov2_attn_cache.py --voc <VOC> --gpu 0 \
        --src ./checkpoints/feat_cache_dinov2 --dst ./checkpoints/feat_cache_dinov2_selfattn
"""
import argparse, math, os
import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--voc", required=True)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--src", default="./checkpoints/feat_cache_dinov2")
ap.add_argument("--dst", default="./checkpoints/feat_cache_dinov2_selfattn")
ap.add_argument("--weights", default="./checkpoints/vgg16_5class.pth")
a = ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
import torchvision.transforms as T   # noqa: E402
from agrqn import core as L                # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L.set_classes(5)
frozen = L.Frozen(a.weights, dev)
m = frozen.dinov2
blk = m.blocks[-1]
if isinstance(blk, torch.nn.ModuleList):        # chunked-block builds: take the real last block
    blk = [b for b in blk if not isinstance(b, torch.nn.Identity)][-1]
n_reg = int(getattr(m, "num_register_tokens", 0) or 0)

cap = {}
blk.attn.register_forward_hook(lambda mod, inp, out: cap.__setitem__("x", inp[0]))
tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])   # == feat_dinov2

@torch.inference_mode()
def desc_and_map(img):
    t = tf(Image.fromarray(img)).unsqueeze(0).to(dev)
    f = m.forward_features(t)["x_norm_patchtokens"]                   # (1, 256, 384)
    s = int(round(f.shape[1] ** 0.5))
    desc = f.reshape(1, s, s, -1).permute(0, 3, 1, 2)[0]              # (384, 16, 16)
    x = cap["x"]                                                      # input to last attn
    A = blk.attn
    B, N, C = x.shape
    H = A.num_heads
    qkv = A.qkv(x).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
    q, k = qkv[0] * A.scale, qkv[1]
    att = (q @ k.transpose(-2, -1)).softmax(-1)                       # (1, H, N, N)
    side = int(round(math.sqrt(N - 1 - n_reg)))
    cam = att[0, :, 0, 1 + n_reg:].reshape(H, side, side).mean(0).float().cpu().numpy()
    cam = cam - cam.min(); mx = cam.max()
    cam = cv2.resize(cam / mx if mx > 0 else cam, (224, 224)).astype(np.float32)
    return desc.float().cpu().numpy(), cam

os.makedirs(a.dst, exist_ok=True)
names = sorted(fn[:-4] for fn in os.listdir(a.src)
               if fn.endswith(".npz") and not fn.endswith("_cam.npz"))
print(f"{len(names)} images  {a.src} -> {a.dst}  (registers={n_reg})", flush=True)

diffs, corrs = [], []
for i, n in enumerate(tqdm(names, desc="dinov2 self-attention")):
    out_p = os.path.join(a.dst, n + ".npz")
    z = dict(np.load(os.path.join(a.src, n + ".npz")))
    if os.path.exists(out_p) and i >= 50:
        continue
    img = np.array(Image.open(os.path.join(a.voc, "JPEGImages", n + ".jpg")).convert("RGB"))
    d, hm = desc_and_map(img)
    if i < 50:                                   # self-checks on the first 50 images
        diffs.append(float(np.abs(d - z["feat7"].astype(np.float32)).max()))
        corrs.append(float(np.corrcoef(hm.ravel(), z["hm"].astype(np.float32).ravel())[0, 1]))
    z["hm"] = hm.astype(np.float16)
    np.savez_compressed(out_p, **z)

print(f"\nSELF-CHECK desc max|diff| over 50 images: median {np.median(diffs):.4f}, "
      f"max {max(diffs):.4f}   (expect fp16 rounding level; if >0.1 STOP and report)")
print(f"map correlation DINO-v1 vs DINOv2 (median over 50): {np.median(corrs):.3f}  (informative)")

# visual check
_, _, _, (te_n, _) = L.build_splits(a.voc)
os.makedirs("report/attn_compare", exist_ok=True)
for n in te_n[:8]:
    img = cv2.resize(np.array(Image.open(os.path.join(a.voc, "JPEGImages", n + ".jpg"))
                              .convert("RGB")), (224, 224))
    old = np.load(os.path.join(a.src, n + ".npz"))["hm"].astype(np.float32)
    new = np.load(os.path.join(a.dst, n + ".npz"))["hm"].astype(np.float32)
    col = lambda h: cv2.applyColorMap((np.clip(h, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    row = np.concatenate([img[:, :, ::-1], col(old), col(new)], axis=1)
    cv2.imwrite(f"report/attn_compare/{n}.png", row)
print("wrote report/attn_compare/*.png  (image | DINO ViT-S/16 | DINOv2 same-pass)")
print(f"DONE. new cache: {a.dst}")
