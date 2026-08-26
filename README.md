# AG-RQN: Redundant Saliency Priors and Calibrated Stopping in Active Object Localization

Code and results for the paper *"Redundant Saliency Priors and Calibrated Stopping in
Active Object Localization"* (submitted to TMLR).

AG-RQN is a Double-DQN agent for active object localization on PASCAL VOC2012. It starts
with a bounding box covering the whole image and, at each step, chooses one of 9 actions
(4 translations, 2 scalings, 2 aspect-ratio changes, and a terminal "trigger" that stops the
search) to deform the box toward the target object. The paper makes two main findings:
(1) once the region descriptor is strong (e.g. a self-supervised ViT), an auxiliary
class-agnostic saliency prior becomes largely redundant, and (2) the agent's failure to
trigger before its step budget runs out is not a stopping bug but a calibrated abstention
signal — exhaustion rate closely tracks the localization failure rate across every
architecture we tried.

## Repository layout

```
agrqn/            importable package: shared data loading, splits, backbones, the
                  Q-network, reward functions, and the evaluation/rollout loop
scripts/          entry-point scripts (training, evaluation, figure/table generation)
                  and .sh runners for the paper's experiment sweeps
results/          result JSONs, records, and calibration reports/figures backing the
                  paper's tables (data only, no code)
```

We use a single Python package plus a `scripts/` folder rather than a deeper module tree
because the codebase is small (~2,000 lines) and every script is a standalone CLI entry
point over the shared library in `agrqn/core.py`. We did not introduce a separate
`configs/` directory: the paper's experiment configurations are named presets in a Python
dictionary (`agrqn/core.py: CONFIGS`, selected via `--config`) rather than external config
files, and extracting them into YAML would touch experiment-defining code that we were
asked to leave unmodified for this cleanup.

## Installation

```bash
git clone <this repository>
cd agrqn
python -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # or your CUDA version
pip install -r requirements.txt
pip install -e .        # makes `agrqn` importable from scripts/
```

`scripts/setup_env.sh` automates the steps above and also pre-downloads the frozen DINO
ViT-S/16 backbone used by every config.

DINO (`facebookresearch/dino`) and DINOv2 (`facebookresearch/dinov2`) backbones are pulled
via `torch.hub` on first use and require outbound network access the first time each script
runs.

## Data

Download PASCAL VOC2012 from the official mirror:
<http://host.robots.ox.ac.uk/pascal/VOC/voc2012/>

Extract it so that `<VOC_ROOT>/JPEGImages`, `<VOC_ROOT>/Annotations`, and
`<VOC_ROOT>/ImageSets/Main` exist. Every script below takes this path via `--voc`; we
recommend placing it at `./data/VOC2012_train_val` (already excluded via `.gitignore`) and
either passing `--voc ./data/VOC2012_train_val` explicitly or exporting `VOC=...` for the
`.sh` runners.

Trained model weights (`*.pth`) are not included in this repository due to size; every
command below reproduces them from scratch.

## Reproducing the paper's experiments

Run these in order from the repository root. `<VOC>` is your VOC2012 path.

```bash
# 1. Cache frozen backbone features (VGG-16 conv maps + DINO attention), once.
python scripts/precache.py --voc <VOC> --weights vgg16_5class.pth \
    --cache ./checkpoints/feat_cache --gpu 0

# 1b. Optional: only needed for the prior_gradcam / prior_layercam configs.
python scripts/precache_cams.py --voc <VOC> --weights vgg16_5class.pth \
    --cache ./checkpoints/feat_cache --gpu 0

# 2. Train the frozen classifier(s) the agent scores detections against.
python scripts/train_classifier.py --voc <VOC> --classes 5 \
    --out ./checkpoints/vgg16_5class.pth --gpu 0 --epochs 20
# For the 20-class scaling experiment (see Compute note below):
python scripts/train_classifier.py --voc <VOC> --classes 20 \
    --out ./checkpoints/vgg16_20class.pth --gpu 0 --epochs 20
python scripts/train_clf_dinov2.py --voc <VOC> --classes 20 \
    --gpu 0 --out ./checkpoints/clf_dinov2_20.pth
python scripts/precache.py --voc <VOC> --desc dinov2 --classifier dinov2 \
    --clf-weights ./checkpoints/clf_dinov2_20.pth --classes 20 \
    --cache ./checkpoints/feat_cache_20 --gpu 0

# 3. Train the localization policy. A single config:
python scripts/train_agrqn.py --config control --voc <VOC> \
    --weights ./checkpoints/vgg16_5class.pth --cache ./checkpoints/feat_cache \
    --out ./results/results_final --gpu 0
# The full 5-class sweep (accuracy track, incentive track, prior study, ablations):
bash scripts/run_all.sh
# The two 20-class scaling runs:
VOC=<VOC> GPU=0 bash scripts/run_dino.sh
VOC=<VOC> GPU=0 bash scripts/run_none.sh

# 4. Evaluate / re-evaluate saved checkpoints (pure inference, no retraining):
python scripts/reevaluate_checkpoints.py --voc <VOC> --out ./results/results_final --gpu 0

# 5. Calibration analysis (CPU only, no GPU needed): the statistical tests behind the
#    "exhaustion rate == failure rate" claim. Scored on last_iou (the terminal-box
#    convention used throughout the paper).
python scripts/analyze_calibration.py \
    --records ./results/results_final/desc_dinov2_records.json.gz \
    --iou-key last_iou \
    --out ./results/results_calib/headline_lastbox
```

`scripts/collect_results.py` builds a cross-run summary table and sweep-level figures from
everything under a results directory; `scripts/make_paper_figures.py`,
`make_action_fig.py`, and `make_trajectory_fig.py` regenerate the paper's individual
figures and LaTeX table fragments from the same saved records.

## Compute and reproducibility notes

- All experiments were run on a single NVIDIA GPU on a shared cluster (one GPU per job;
  several accuracy/incentive-track jobs were run concurrently on different GPUs via
  `run_all.sh`).
- Each 5-class configuration is trained for a fixed 120 epochs with no early stopping
  (`--early-stop` is opt-in and off by default; see `train_agrqn.py`).
- The two 20-class runs (`voc20_dino`, `voc20_none`) took 26.8 h and 47.7 h wall-clock,
  respectively.
- Seed variation, measured over three seeds on the main 5-class configuration: mIoU
  0.5776 / 0.5720 / 0.5716.
- All other reported configurations are single-seed.

## Mapping paper items to scripts

| Paper item | How it's produced |
|---|---|
| Main results (accuracy-track and incentive-track configs) | `scripts/run_all.sh` → `scripts/train_agrqn.py`, summarized by `scripts/collect_results.py` |
| Prior-redundancy study (none / Grad-CAM / LayerCAM / DINO) | `--config prior_none / prior_gradcam / prior_layercam / prior_dino` in `train_agrqn.py`; figure via `scripts/make_paper_figures.py:fig_prior_arc` |
| Descriptor-swap redundancy prediction (VGG vs. DINOv2 descriptor) | `--config desc_dinov2 / desc_dinov2_none / dinov2_gradcam / desc_vgg_none` |
| Retraining ablations (no history, single-DQN) | `--config no_history / single_dqn / dinov2_nohist / dinov2_singleq` |
| Reward/incentive interventions (negative-result controls) | `--config expiry_fail / steppen02 / eta10 / pbrs`; figure via `fig_trigger` in `scripts/make_paper_figures.py` |
| 20-class scaling experiment | `scripts/run_dino.sh`, `scripts/run_none.sh` → `results/results_20class/` |
| Calibration defense of "exhaustion == failure rate" (Tests A-E) | `scripts/analyze_calibration.py` → `results/results_calib/` |
| Action-distribution / coarse-to-fine figure | `scripts/make_action_fig.py` |
| Episode trajectory figure | `scripts/make_trajectory_fig.py` |
| Per-episode timing table | `scripts/measure_timing.py` |
| Qualitative box-overlay figure | `scripts/make_box_fig.py` (reads `results/results_final/qual_boxes.json`) |
