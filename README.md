# AG-RQN: Saliency Priors, Descriptor Strength, and Selective Stopping in Active Object Localization

Code, configurations and evaluation records for the paper

> Jeslin P. James and Shailesh S.
> **Saliency Priors, Descriptor Strength, and Selective Stopping in Active Object Localization.**
> *Transactions on Machine Learning Research*, 2026.
> [OpenReview](https://openreview.net/forum?id=k78lV4wIvR)

AG-RQN is a single class-agnostic Double-DQN policy for active object localization on
PASCAL VOC 2012, paired with a separate 5-way image classifier. Starting from a box over the
whole image, the agent applies one of nine actions per step (four translations, two scalings,
two aspect-ratio changes, and a terminal *trigger*) until it stops or a 20-step budget runs
out. The policy fuses a region descriptor (VGG-16 or DINOv2), a class-agnostic attention
prior, and a GRU over recent actions.

The paper tests two design choices that are usually adopted without ablation:

1. **The saliency prior.** A Grad-CAM prior gives no measurable gain over a retrained no-prior
   control; LayerCAM adds +0.034 mIoU and a DINO self-attention prior +0.043. With a DINOv2
   descriptor the DINO prior's benefit falls by 56% (+0.043 to +0.019, three seeds per
   configuration). A prior taken from the descriptor's own DINOv2 forward pass recovers almost
   the same gain, so independence from the descriptor is not what makes a prior useful.
   The best configuration reaches 0.634 ± 0.004 mIoU / 0.803 ± 0.016 CorLoc.
2. **The stop decision.** Stopping is discriminative of success: 90% of voluntarily stopped
   episodes reach IoU ≥ 0.5 against 25% of budget-exhausted ones, so budget exhaustion acts as
   a learned rejection signal. Read as a probability, the trigger Q-value is overconfident
   (ECE 0.09–0.19): the signal is discriminative, not calibrated. Seven reward-level
   interventions, each retrained with three seeds, trace a coverage–precision trade-off; a
   per-step cost of −0.2 lowers exhaustion from 0.153 to 0.087 without losing precision.

## Repository layout

```
agrqn/            importable package: data loading, the train/val/test split, frozen
                  backbones, the feature cache, the Q-network, rewards, and the
                  evaluation/rollout loop (agrqn/core.py)
scripts/          command-line entry points: caching, training, evaluation, analysis,
                  figures and tables, and the .sh runners for the experiment sweeps
results/          result JSONs and per-episode evaluation records behind the paper's
                  tables and figures (data only, no code)
```

Experiment configurations are named presets in `agrqn/core.py: CONFIGS`, selected with
`--config`. Each preset lists only what differs from `DEFAULTS` in the same file.

## Installation

```bash
git clone https://github.com/jeslinpjames/AGRQN.git
cd AGRQN
python -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # or your CUDA version
pip install -r requirements.txt
pip install -e .        # makes `agrqn` importable from scripts/
```

`scripts/setup_env.sh` automates these steps and pre-downloads the frozen DINO ViT-S/16
backbone. DINO (`facebookresearch/dino`) and DINOv2 (`facebookresearch/dinov2`) are loaded
through `torch.hub`, which needs network access the first time.

## Data

Download PASCAL VOC 2012 from <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/> and extract it
so that `<VOC>/JPEGImages`, `<VOC>/Annotations` and `<VOC>/ImageSets/Main` exist. Every script
takes this path with `--voc`; we recommend `./data/VOC2012_train_val` (ignored by git).

Trained weights (`*.pth`) are not included because of their size; the commands below
reproduce them.

## Reproducing the experiments

Run from the repository root. `<VOC>` is your VOC 2012 path.

```bash
# 1. Train the frozen 5-way classifier.
python scripts/train_classifier.py --voc <VOC> --classes 5 \
    --out ./checkpoints/vgg16_5class.pth --gpu 0 --epochs 20

# 2. Cache frozen features once per descriptor.
#    VGG-16 descriptor + DINO prior:
python scripts/precache.py --voc <VOC> --weights ./checkpoints/vgg16_5class.pth \
    --cache ./checkpoints/feat_cache --gpu 0
#    DINOv2 descriptor + DINO prior:
python scripts/precache.py --voc <VOC> --weights ./checkpoints/vgg16_5class.pth \
    --desc dinov2 --cache ./checkpoints/feat_cache_dinov2 --gpu 0
#    Grad-CAM / LayerCAM priors (prior study, and Grad-CAM with the DINOv2 descriptor):
for C in feat_cache feat_cache_dinov2; do
  python scripts/precache_cams.py --voc <VOC> --weights ./checkpoints/vgg16_5class.pth \
      --cache ./checkpoints/$C --gpu 0
done
#    Same-backbone prior (attention from the descriptor's own DINOv2 pass):
python scripts/make_dinov2_attn_cache.py --voc <VOC> --gpu 0 \
    --src ./checkpoints/feat_cache_dinov2 --dst ./checkpoints/feat_cache_dinov2_selfattn

# 3. Train a policy (one config, one seed). Every run is 120 epochs, no early stopping.
python scripts/train_agrqn.py --config desc_dinov2 --voc <VOC> \
    --weights ./checkpoints/vgg16_5class.pth --cache ./checkpoints/feat_cache_dinov2 \
    --out ./results/results_final --gpu 0 --seed 42

# 4. The three-seed runs of the paper (main configs, same-backbone prior, reward study):
VOC=<VOC> GPUS="0 1 2 3" bash scripts/run_seeds.sh            # or: ... run_seeds.sh main
#    Single-seed runs (prior study and ablations), same command as step 3 with the
#    matching cache:
for c in prior_gradcam prior_layercam no_history single_dqn; do
  python scripts/train_agrqn.py --config $c --voc <VOC> --weights ./checkpoints/vgg16_5class.pth \
      --cache ./checkpoints/feat_cache --out ./results/results_final --gpu 0
done
for c in dinov2_nohist dinov2_singleq dinov2_gradcam; do
  python scripts/train_agrqn.py --config $c --voc <VOC> --weights ./checkpoints/vgg16_5class.pth \
      --cache ./checkpoints/feat_cache_dinov2 --out ./results/results_final --gpu 0
done
#    The 20-class scale check: 20-class classifiers and cache first, then the two runs.
python scripts/train_classifier.py --voc <VOC> --classes 20 \
    --out ./checkpoints/vgg16_20class.pth --gpu 0 --epochs 20
python scripts/train_clf_dinov2.py --voc <VOC> --classes 20 \
    --gpu 0 --out ./checkpoints/clf_dinov2_20.pth
python scripts/precache.py --voc <VOC> --desc dinov2 --classifier dinov2 \
    --clf-weights ./checkpoints/clf_dinov2_20.pth --classes 20 \
    --cache ./checkpoints/feat_cache_20 --gpu 0
VOC=<VOC> GPU=0 bash scripts/run_dino.sh
VOC=<VOC> GPU=0 bash scripts/run_none.sh

# 5. Analysis (inference or CPU only; no retraining):
python scripts/rescore_trueclass.py --voc <VOC> --gpu 0          # true- vs predicted-class
python scripts/analyze_calibration.py \
    --records ./results/results_final/desc_dinov2_records.json.gz \
    --iou-key last_iou --out ./results/results_calib/headline_lastbox
python scripts/calib_qvalues.py --voc <VOC> --gpu 0              # trigger Q-values
python scripts/calib_analysis.py --dir ./results/results_qcalib
python scripts/measure_timing.py --voc <VOC> --gpu 0 --rounds 3  # Appendix D

# 6. Figures and LaTeX tables from the saved records:
python scripts/make_paper_figures.py --results ./results/results_final --out ./paper_assets
```

## Paper items and how they are produced

| Paper item | Configs / scripts |
|---|---|
| Table 1 (per-class results) | `desc_dinov2`; `make_paper_figures.py` (`tab_headline`) |
| Table 2 (four main configurations, 3 seeds) | `desc_vgg_none`, `duel_wide` (tags `final_s*`), `desc_dinov2_none`, `desc_dinov2`; `run_seeds.sh main` |
| Table 3 (true- vs predicted-class scoring) | `rescore_trueclass.py` |
| Table 4 (retraining ablations) | `no_history`, `single_dqn`, `dinov2_nohist`, `dinov2_singleq`, `dinov2_gradcam` |
| Table 5, Figure 3 (prior study, attenuation) | `prior_gradcam`, `prior_layercam` + the main configs; `make_paper_figures.py` (`fig_prior_arc`) |
| Table 6 (same-backbone prior) | `make_dinov2_attn_cache.py`, `desc_dinov2_selfattn`; `run_seeds.sh selfatt` |
| Table 7, Figure 4 (reward interventions) | `control`, `steppen005`, `steppen02`, `gamma99`, `eta10`, `pbrs`, `expiry_fail`, `aux_head`; `run_seeds.sh reward`; `make_paper_figures.py` (`fig_trigger`) |
| Table 9 (discrimination tests of the stop decision) | `analyze_calibration.py` → `results/results_calib/` |
| Appendix B (actions, trajectories, boxes) | `make_action_fig.py`, `make_trajectory_fig.py`, `make_box_fig.py` |
| Appendix C (inference-time step budget) | evaluate a saved checkpoint with a different `MAX_STEPS` in `agrqn/core.py` |
| Appendix D (computational cost) | `measure_timing.py` → `results/results_timing/` |
| Appendix E (calibration of the trigger Q-value) | `calib_qvalues.py`, `calib_analysis.py` → `results/results_qcalib/` |
| Appendix F (error analysis) | `make_paper_figures.py` (`fig_failures`, `fig_size`) |
| Appendix G (twenty-class scale check) | `run_dino.sh`, `run_none.sh` → `results/results_20class/` |

Result tags in `results/results_final/` follow `<config>` for seed 42 and `<config>_s1`,
`<config>_s7` for seeds 1 and 7 (the reward interventions use `iv_<config>`). A trailing `r`
(for example `iv_pbrs_s1r`) marks a full rerun of a seed whose first run was interrupted
before finishing; only the completed rerun is included.

Other result folders: `results_trueclass/` (Table 3), `results_calib/` (Table 9),
`results_qcalib/` (Appendix E: one JSON per policy plus `summary.txt`), `results_timing/`
(Appendix D), and `results_20class/` (Appendix G).

## Compute notes

- Every run used one NVIDIA GPU on a shared cluster; several runs shared a GPU concurrently.
- Each 5-class configuration trains for a fixed 120 epochs with no early stopping
  (`--early-stop` is available but off by default). Checkpoints are selected on a 3-epoch
  moving average of validation mIoU.
- The two 20-class runs took 26.8 h and 47.7 h of wall-clock time.
- Timing in the paper (Appendix D) was measured on an idle NVIDIA A100-SXM4-80GB, batch size 1.

## Citation

```bibtex
@article{james2026saliency,
  title   = {Saliency Priors, Descriptor Strength, and Selective Stopping in Active Object Localization},
  author  = {James, Jeslin P. and S., Shailesh},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=k78lV4wIvR}
}
```

## License

MIT. See [LICENSE](LICENSE).
