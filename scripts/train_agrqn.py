#!/usr/bin/env python
"""Train one AG-RQN config on one GPU. Launch several in parallel with run_all.sh.

Training runs for a fixed number of epochs with no early stopping by default (see
--early-stop): on a 200-image validation set, mIoU noise is roughly +/-0.02, which is
large enough that early stopping can trigger on noise rather than genuine plateau, making
runs incomparable. Checkpoint selection instead uses a 3-epoch moving average of
validation mIoU.

Outputs (all under --out):
    <tag>_history.json    per-epoch metrics
    <tag>_best.pth         best 3-epoch-average validation mIoU weights
    <tag>_final.pth        last-epoch weights
    <tag>_test.json        full test-set metrics for both checkpoints
    <tag>_records.json.gz  per-episode test records (for downstream plots/analysis)
"""
import argparse
import json
import logging
import os
import random
import sys
import time
from gzip import open as gzip_open

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from agrqn import core as L

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# --------------------------------------------------------------- training constants
EPOCHS = 60
MIN_EPOCHS = 25      # earliest epoch at which --early-stop is allowed to fire
PATIENCE = 15
REPLAY_CAP = 30000
BATCH = 64
WARMUP = 1000
LR = 1e-4
TARGET_TAU = 0.005
GRAD_CLIP = 5.0
AUX_W = 0.5
EPS0, EPS1, EPS_DECAY = 1.0, 0.1, 60000
IMAGES_PER_CLASS = {"bicycle": 350, "aeroplane": 300, "horse": 300, "cat": 300, "dog": 350}


class Replay:
    def __init__(self, cap):
        self.cap, self.buf, self.i = cap, [], 0

    def push(self, x):
        if len(self.buf) < self.cap:
            self.buf.append(x)
        else:
            self.buf[self.i] = x; self.i = (self.i + 1) % self.cap

    def sample(self, n):
        return random.sample(self.buf, n)

    def __len__(self):
        return len(self.buf)


def batchify(items, device):
    tt = lambda k: torch.as_tensor(np.stack([b[k] for b in items]).astype(np.float32)).to(device)
    return (tt(0), tt(1), tt(2).unsqueeze(1),
            torch.as_tensor(np.array([b[3] for b in items]) - 1).long().to(device),
            torch.as_tensor(np.array([b[4] for b in items], np.float32)).to(device),
            tt(5), tt(6), tt(7).unsqueeze(1),
            torch.as_tensor(np.array([b[8] for b in items], np.float32)).to(device),
            torch.as_tensor(np.array([b[9] for b in items], np.float32)).to(device))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, choices=list(L.CONFIGS))
    ap.add_argument("--voc", required=True)
    ap.add_argument("--weights", default="vgg16_5class.pth")
    ap.add_argument("--cache", default="./checkpoints/feat_cache")
    ap.add_argument("--out", default="./results")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--classes", type=int, default=5, choices=[5, 20],
                     help="20 needs vgg16_20class.pth and a 20-class feature cache")
    ap.add_argument("--test-per-class", type=int, default=None)
    ap.add_argument("--clf-weights", default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--tag", default=None,
                     help="output basename, defaults to --config. Set this explicitly when "
                          "running the same config with different seeds so the runs do not "
                          "overwrite each other's outputs.")
    ap.add_argument("--train-freq", type=int, default=1,
                     help="gradient steps per environment step. Values other than 1 change "
                          "the training dynamics; do not mix results across --train-freq "
                          "settings in one table.")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--early-stop", action="store_true",
                     help="opt-in early stopping (off by default; see module docstring)")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    L.set_classes(a.classes)
    if a.test_per_class:
        L.N_TEST_PER_CLASS = a.test_per_class
    cfg = L.get_config(a.config)
    cfg["_voc"] = a.voc
    tag = a.tag or a.config          # every output file is named by tag, not config
    logger.info("config '%s' tag='%s' [%s] gpu=%d seed=%d",
                a.config, tag, cfg["track"], a.gpu, a.seed)
    logger.info("%s", {k: v for k, v in cfg.items() if not k.startswith("_")})

    frozen = L.Frozen(a.weights, device)
    if a.clf_weights:
        frozen.load_dinov2_clf(a.clf_weights)
    pool, train_all, (ckpt_n, ckpt_l), (test_n, test_l) = L.build_splits(a.voc)
    logger.info("train %d | ckpt %d | test %d", len(train_all), len(ckpt_n), len(test_n))

    RAM = L.CacheRAM(sorted(set(train_all) | set(ckpt_n) | set(test_n)),
                      a.cache, cfg["feat"], device, prior=cfg["prior"])

    online = L.AGRQN(cfg).to(device)
    target = L.AGRQN(cfg).to(device)
    target.load_state_dict(online.state_dict()); target.eval()
    nparam = sum(p.numel() for p in online.parameters())
    logger.info("head parameters: %.2fM", nparam / 1e6)

    opt = torch.optim.Adam(online.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
    D = Replay(REPLAY_CAP)

    gstep, best, bad, hist_log = 0, -1.0, 0, []
    t0 = time.time()
    best_p = os.path.join(a.out, f"{tag}_best.pth")
    final_p = os.path.join(a.out, f"{tag}_final.pth")

    for epoch in range(1, a.epochs + 1):
        online.train()
        eps_list = []
        for c in L.SELECTED_CLASSES:
            # IMAGES_PER_CLASS only names the 5 original classes; every other class gets the
            # same 300-episode budget per epoch.
            k = min(IMAGES_PER_CLASS.get(c, 300), len(pool[c]))
            eps_list += [(n, c) for n in random.sample(pool[c], k)]
        random.shuffle(eps_list)

        for name, c in tqdm(eps_list, desc=f"[{a.config}] ep {epoch}/{a.epochs}",
                             leave=False, disable=not sys.stdout.isatty()):
            C = RAM.get(name, frozen.pool, cls=L.CLASS_NAME_TO_LABEL[c])  # true class
            gts = L.gt_boxes(L.annotation(name, a.voc), L.VOC_CLASS_IDS[c])
            if not gts:
                continue

            box = [0, 0, C.W, C.H]
            hist = np.zeros(L.HIST_DIM, np.float32)
            desc, hm = C.desc(box), C.hmcrop(box)
            prev = L.max_iou(box, gts)
            eps = max(EPS1, EPS0 - (EPS0 - EPS1) * gstep / EPS_DECAY)

            for step in range(L.MAX_STEPS):
                if random.random() < eps:
                    act = random.randint(1, L.NUM_ACTIONS)
                else:
                    with torch.inference_mode():
                        q, _ = online(
                            torch.as_tensor(desc.astype(np.float32)).unsqueeze(0).to(device),
                            torch.as_tensor(hist).unsqueeze(0).to(device),
                            torch.as_tensor(hm).unsqueeze(0).unsqueeze(0).to(device))
                    act = int(q.argmax(1).item()) + 1

                forced = (step == L.MAX_STEPS - 1) and (act != L.TRIGGER)
                done = (act == L.TRIGGER) or forced

                if act == L.TRIGGER:
                    nbox, ndesc, nhm, new = box, desc, hm, prev
                    nhist = L.push_hist(hist, act)
                    r = L.terminal_reward(cfg, new, triggered=True)
                else:
                    nbox = L.transform_box(box, act, C.W, C.H)
                    nhist = L.push_hist(hist, act)
                    ndesc, nhm = C.desc(nbox), C.hmcrop(nbox)
                    new = L.max_iou(nbox, gts)
                    r = (L.terminal_reward(cfg, new, triggered=False) if forced
                         else L.movement_reward(cfg, prev, new))

                D.push((desc, hist.astype(np.float16), hm.astype(np.float16), act, r,
                        ndesc, nhist.astype(np.float16), nhm.astype(np.float16),
                        float(done), float(prev)))
                gstep += 1

                if gstep % a.train_freq == 0 and len(D) >= max(WARMUP, a.batch):
                    d, h, m, aa, rr, d2, h2, m2, dn, iu = batchify(D.sample(a.batch), device)
                    with torch.no_grad():
                        q_tg, _ = target(d2, h2, m2)
                        if cfg["double_q"]:
                            q_on, _ = online(d2, h2, m2)
                            nq = q_tg.gather(1, q_on.argmax(1, keepdim=True)).squeeze(1)
                        else:                      # ablation: plain DQN max, overestimates
                            nq = q_tg.max(1).values
                        y = rr + cfg["gamma"] * nq * (1.0 - dn)
                    q, aux = online(d, h, m)
                    loss = F.smooth_l1_loss(q.gather(1, aa.view(-1, 1)).squeeze(1), y)
                    if cfg["aux_iou"]:
                        loss = loss + AUX_W * F.mse_loss(aux, iu)
                    opt.zero_grad(set_to_none=True); loss.backward()
                    nn.utils.clip_grad_norm_(online.parameters(), GRAD_CLIP)
                    opt.step()
                    with torch.no_grad():
                        for tp, op in zip(target.parameters(), online.parameters()):
                            tp.mul_(1 - TARGET_TAU).add_(TARGET_TAU * op)

                box, hist, desc, hm, prev = nbox, nhist, ndesc, nhm, new
                if done:
                    break

        # ------------------------------------------------------------ validate
        m = L.summarize(L.evaluate(online, ckpt_n, ckpt_l, cfg, frozen, RAM, device))
        m["epoch"] = epoch; m["eps"] = eps
        hist_log.append(m)
        logger.info(
            "ep %2d | mIoU %.4f | acc %.3f | steps %5.2f | exhaust %.3f | "
            "ppv %.3f | exh-failrate %+.3f | eps %.2f",
            epoch, m["miou"], m["acc05"], m["mean_steps"], m["exhaust_rate"],
            m["trigger_ppv"], m["exh_minus_failrate"], eps)
        sched.step(m["miou"])

        torch.save(online.state_dict(), final_p)          # always keep the last epoch

        # Checkpoint selection on a 3-epoch moving average: a single epoch on a 200-image
        # validation set carries roughly +/-0.02 of mIoU noise, so one lucky epoch is not
        # sufficient evidence of improvement.
        sm = float(np.mean([x["miou"] for x in hist_log[-3:]]))
        if sm > best:
            best, bad = sm, 0
            torch.save(online.state_dict(), best_p)
            logger.info("  -> new best (3-epoch avg %.4f), saved", sm)
        else:
            bad += 1
            if a.early_stop and epoch >= MIN_EPOCHS and bad >= PATIENCE:
                logger.info("early stop at epoch %d (best %.4f)", epoch, best)
                break

        with open(os.path.join(a.out, f"{tag}_history.json"), "w") as f:
            json.dump(hist_log, f, indent=2)

    # ------------------------------------------------------------ final test evaluation
    logger.info("training done in %.1f min. evaluating on test set...", (time.time() - t0) / 60)
    results = {"config": a.config, "tag": tag,
               "cfg": {k: v for k, v in cfg.items() if not k.startswith("_")},
               "seed": a.seed, "params_M": nparam / 1e6,
               "train_minutes": (time.time() - t0) / 60, "history": hist_log}

    for ckpt_tag, path in (("best", best_p), ("final", final_p)):
        net = L.AGRQN(cfg).to(device)
        net.load_state_dict(torch.load(path, map_location=device))
        recs = L.evaluate(net, test_n, test_l, cfg, frozen, RAM, device)
        results[f"test_{ckpt_tag}"] = {
            "last_box": L.summarize(recs, "last_iou"),      # as deployed
            "conf_box": L.summarize(recs, "final_iou"),     # + confidence-based box selection
        }
        if ckpt_tag == "best":
            slim = [{k: r[k] for k in ("name", "cls_true", "cls_pred", "correct_cls", "n_inst",
                                        "gt_frac", "triggered", "n_steps", "last_iou", "final_iou",
                                        "peak_iou", "actions", "ious", "confs")} for r in recs]
            with gzip_open(os.path.join(a.out, f"{tag}_records.json.gz"), "wt") as f:
                json.dump(slim, f)

    with open(os.path.join(a.out, f"{tag}_test.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    b = results["test_best"]["last_box"]
    logger.info("test result [%s]", tag)
    logger.info("  mIoU            %.4f", b["miou"])
    logger.info("  CorLoc/Acc@0.5  %.4f", b["acc05"])
    logger.info("  exhaust rate    %.4f", b["exhaust_rate"])
    logger.info("  1 - Acc@0.5     %.4f", 1 - b["acc05"])
    logger.info("  gap (exh - failrate) %+.4f  (near 0: exhaustion tracks the failure rate)",
                b["exh_minus_failrate"])
    logger.info("  trigger PPV     %.4f", b["trigger_ppv"])
    logger.info("  mean steps      %.2f", b["mean_steps"])
    logger.info("  mIoU (conf box) %.4f", results["test_best"]["conf_box"]["miou"])


if __name__ == "__main__":
    main()
