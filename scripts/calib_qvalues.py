#!/usr/bin/env python
"""
calib_qvalues.py -- is the trigger Q-value a calibrated probability? (paper: Appendix E)

Inference only, no training. For every 5-class run in --results it reloads the saved best
checkpoint, replays the
test set with the unchanged L.evaluate(), and -- through a forward hook on the network plus
a thin wrapper around L.rollout -- records the FULL 9-way Q-vector (and the aux-IoU head,
if present) at every step of every episode. Writes one <out>/<tag>.npz per run.

Why this is a valid probability test: triggering ends the episode, so Q(s, trigger) has no
bootstrap term. terminal_reward() pays +eta if IoU >= 0.5 else -eta (the step penalty is
not added on the trigger step), hence  Q(s,trigger) = eta*(2p-1)  ->  p = (Q/eta + 1)/2,
where p is the agent's implied probability that stopping NOW succeeds. (PBRS adds
-5*IoU to the terminal reward, so the mapping is not identifiable there; the analysis
reports discrimination + a fitted mapping only for that run.)

SELF-CHECKS (printed per run):
  * mIoU / CorLoc recomputed here must equal the run's own *_test.json  (else MISMATCH)
  * number of captured Q-vectors per episode must equal n_steps          (alignment)
  * argmax of each captured Q-vector must equal the action actually taken (greedy policy)

Usage (~20-60 min on one GPU; ~3-5 GB), from the repository root:
    python -u scripts/calib_qvalues.py --voc <VOC> --gpu 0 \
        | tee ./results/results_qcalib/capture.log
then:
    python scripts/calib_analysis.py --dir ./results/results_qcalib
"""
import argparse, glob, json, os
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--voc", required=True)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--results", default="./results/results_final")
ap.add_argument("--vgg-cache", default="./checkpoints/feat_cache")
ap.add_argument("--dinov2-cache", default="./checkpoints/feat_cache_dinov2")
ap.add_argument("--selfattn-cache", default="./checkpoints/feat_cache_dinov2_selfattn")
ap.add_argument("--weights", default="./checkpoints/vgg16_5class.pth")
ap.add_argument("--out", default="./results/results_qcalib")
ap.add_argument("--only", nargs="*", default=None, help="restrict to these tags")
a = ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
from agrqn import core as L          # noqa: E402  (after CUDA_VISIBLE_DEVICES)

os.makedirs(a.out, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {dev} (physical GPU {a.gpu})", flush=True)
L.set_classes(5)
frozen = L.Frozen(a.weights, dev)
_, _, _, (te_n, te_l) = L.build_splits(a.voc)

# ---------------------------------------------------------------- capture machinery
BUF = []
_orig_rollout = L.rollout


def _rollout_capture(net, C, gts, cfg, device):
    BUF.clear()
    ep = _orig_rollout(net, C, gts, cfg, device)
    ep = dict(ep)
    ep["_q"] = [b[0] for b in BUF]
    ep["_aux"] = [b[1] for b in BUF]
    return ep


L.rollout = _rollout_capture   # evaluate() looks `rollout` up in the module at call time


def _hook(_m, _inp, out):
    q, aux = out
    BUF.append((q.detach().float().cpu().numpy(),
                None if aux is None else aux.detach().float().cpu().numpy().reshape(-1)))


def cache_for(tag, cfg):
    if tag.startswith("desc_dinov2_selfattn"):
        return a.selfattn_cache
    return a.dinov2_cache if cfg["desc"] == "dinov2" else a.vgg_cache


rams = {}
for f in sorted(glob.glob(f"{a.results}/*_test.json")):
    r = json.load(open(f))
    cname = r.get("config", "")
    tag = r.get("tag") or os.path.basename(f)[: -len("_test.json")]
    if cname.startswith("voc20") or cname not in L.CONFIGS:
        continue
    if a.only and tag not in a.only:
        continue
    ck = f"{a.results}/{tag}_best.pth"
    if not os.path.exists(ck):
        print(f"skip {tag}: no {ck}", flush=True)
        continue
    cfg = L.get_config(cname); cfg["_voc"] = a.voc
    cache = cache_for(tag, cfg)
    key = (cache, cfg["feat"], cfg["prior"])
    if key not in rams:
        rams[key] = L.CacheRAM(sorted(te_n), cache, cfg["feat"], dev, prior=cfg["prior"])
    net = L.AGRQN(cfg).to(dev)
    net.load_state_dict(torch.load(ck, map_location=dev)); net.eval()
    h = net.register_forward_hook(_hook)
    with torch.no_grad():
        recs = L.evaluate(net, te_n, te_l, cfg, frozen, rams[key], dev)
    h.remove()

    # ---- self-check against the run's own reported test numbers
    p = L.summarize(recs, "last_iou")
    rep = r.get("test_best", {}).get("last_box", {})
    ok = (abs(p["miou"] - rep.get("miou", -9)) < 1e-3 and
          abs(p["acc05"] - rep.get("acc05", -9)) < 1e-3)

    # ---- flatten to per-state arrays
    Q, AUX, IMG, T, IOU, CONF = [], [], [], [], [], []
    ep_n, ep_k, ep_trig, ep_last_iou, ep_last_act, ep_rows = [], [], [], [], [], []
    n_argmax_ok = n_argmax = 0
    for i, rec in enumerate(recs):
        qs = rec["_q"]
        rows = [x.shape[0] for x in qs]
        ep_rows.append(max(rows) if rows else 0)
        qmat = np.concatenate([x[:1] for x in qs], 0) if qs else np.zeros((0, L.NUM_ACTIONS))
        aux = (np.concatenate([x[:1] for x in rec["_aux"]]) if qs and rec["_aux"][0] is not None
               else np.full(len(qs), np.nan))
        n, k = int(rec["n_steps"]), len(qs)
        ious = rec["ious"]; acts = rec["actions"]; confs = rec.get("confs", [])
        ep_n.append(n); ep_k.append(k); ep_trig.append(bool(rec["triggered"]))
        ep_last_iou.append(float(rec["last_iou"])); ep_last_act.append(int(acts[-1]) if acts else -1)
        for t in range(min(k, len(ious))):
            Q.append(qmat[t]); AUX.append(aux[t]); IMG.append(i); T.append(t)
            IOU.append(float(ious[t])); CONF.append(float(confs[t]) if t < len(confs) else np.nan)
            if t < len(acts):
                n_argmax += 1; n_argmax_ok += int(np.argmax(qmat[t]) == acts[t])

    ep_n, ep_k = np.array(ep_n), np.array(ep_k)
    align = float((ep_k == ep_n).mean())
    align_p1 = float((ep_k == ep_n + 1).mean())
    greedy = n_argmax_ok / max(n_argmax, 1)
    np.savez_compressed(
        f"{a.out}/{tag}.npz",
        q=np.array(Q, np.float32), aux=np.array(AUX, np.float32), img=np.array(IMG, np.int32),
        t=np.array(T, np.int16), iou=np.array(IOU, np.float32), conf=np.array(CONF, np.float32),
        ep_n=ep_n, ep_k=ep_k, ep_trig=np.array(ep_trig), ep_last_iou=np.array(ep_last_iou),
        ep_last_act=np.array(ep_last_act), ep_rows=np.array(ep_rows))
    meta = dict(tag=tag, config=cname, cache=cache,
                eta=cfg["eta"], gamma=cfg["gamma"], shaping=cfg["shaping"], expiry=cfg["expiry"],
                step_penalty=cfg["step_penalty"], dueling=cfg["dueling"], hidden=cfg["hidden"],
                desc=cfg["desc"], prior=cfg["prior"], aux_iou=cfg["aux_iou"],
                stop_mode=cfg["stop_mode"],
                miou=p["miou"], corloc=p["acc05"], exhaust=p["exhaust_rate"], ppv=p["trigger_ppv"],
                reported_miou=rep.get("miou"), selfcheck="OK" if ok else "MISMATCH",
                align_k_eq_n=align, align_k_eq_n_plus_1=align_p1, greedy_agree=greedy,
                max_batch_rows=int(max(ep_rows) if ep_rows else 0),
                n_states=len(Q), n_episodes=len(recs))
    json.dump(meta, open(f"{a.out}/{tag}.json", "w"), indent=1)
    print(f"{tag:28s} mIoU {p['miou']:.4f} (rep {rep.get('miou', float('nan')):.4f}) {meta['selfcheck']:8s}"
          f" | states {len(Q):6d} | k==n {align:.3f} k==n+1 {align_p1:.3f} | greedy {greedy:.3f}"
          f" | rows/call {meta['max_batch_rows']}", flush=True)

print(f"\nwrote {a.out}/*.npz  -> now run:  python scripts/calib_analysis.py --dir {a.out}", flush=True)
