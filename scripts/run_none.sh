#!/bin/bash
# Reproduce the 20-class run with no attention prior. Run from the repository root.
# Edit VOC/GPU below (or export them) for your machine.
set -u
VOC="${VOC:-./data/VOC2012_train_val}"
GPU="${GPU:-0}"
mkdir -p logs
source venv/bin/activate
python scripts/train_agrqn.py \
    --config voc20_none --voc "$VOC" \
    --weights ./checkpoints/vgg16_20class.pth --cache ./checkpoints/feat_cache_20 \
    --clf-weights ./checkpoints/clf_dinov2_20.pth --classes 20 \
    --out ./results/results_20class --gpu "$GPU" --epochs 120 --tag voc20_none_20cls \
    2>&1 | tee logs/voc20_none_20cls.log
