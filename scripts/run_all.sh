#!/bin/bash
# ==============================================================================
# run_all.sh -- launch the accuracy/incentive-track sweep across several GPUs,
# in parallel. Run from the repository root, inside tmux/screen if launching many
# long jobs: each job is started with nohup so it survives a detached session.
#
#   bash scripts/run_all.sh                 # run everything
#   bash scripts/run_all.sh control eta10   # run only these configs
# ==============================================================================
set -u

VOC="./data/VOC2012_train_val"
WEIGHTS="./checkpoints/vgg16_5class.pth"
CACHE="./checkpoints/feat_cache"
OUT="./results/results_final"

# GPUs to use for this sweep. Edit for your machine -- check `nvidia-smi` first if the
# cluster is shared.
GPUS=(0 1 2 3)

# Ordered by importance: the first four configs answer the paper's main question about
# accuracy-track interventions; the rest are supporting evidence.
ALL_CONFIGS=(
  control        # 1. the baseline
  hires          # 2. finer ROI grid, targets small objects
  dueling        # 3. decouples V(s) from A(s,a); reduces the max-bias against the trigger
  combo_acc      # 4. hires + dueling + wide, combined
  aux_stop       # 5. stop on a predicted-IoU head; lets exhaustion be dialed at inference
  expiry_fail    # 6. negative control: always punish budget expiry
  eta10          # 7. larger terminal reward magnitude
  steppen02      # 8. stronger per-step cost
  pbrs           # 9. potential-based shaping
  wide           # 10. wider head alone
)

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then CONFIGS=("${ALL_CONFIGS[@]}"); fi

mkdir -p "$OUT" logs

echo "=================================================================="
echo " AG-RQN sweep"
echo "=================================================================="
echo " VOC     : $VOC"
echo " configs : ${CONFIGS[*]}"
echo " GPUs    : ${GPUS[*]}"
echo "=================================================================="

# ---- sanity checks ----------------------------------------------------------
[ -d "$VOC/JPEGImages" ] || { echo "FATAL: no JPEGImages under $VOC"; exit 1; }
[ -f "$WEIGHTS" ]        || { echo "FATAL: $WEIGHTS not found. Run train_classifier.py first."; exit 1; }
if [ ! -d "$CACHE" ] || [ -z "$(ls -A $CACHE 2>/dev/null)" ]; then
  echo "FATAL: feature cache '$CACHE' is empty. Run precache.py first:"
  echo "  python scripts/precache.py --voc $VOC --weights $WEIGHTS --cache $CACHE --gpu ${GPUS[0]}"
  exit 1
fi

# ---- dispatch: config i -> GPU (i mod nGPU). Jobs on the same GPU still run
#      concurrently; each AG-RQN job uses only ~3-5 GB, so several fit on one GPU.
n=${#GPUS[@]}
i=0
for cfg in "${CONFIGS[@]}"; do
  gpu=${GPUS[$((i % n))]}
  log="logs/${cfg}.log"
  echo "  -> $cfg  on GPU $gpu   (log: $log)"
  nohup python -u scripts/train_agrqn.py \
      --config "$cfg" --voc "$VOC" --weights "$WEIGHTS" \
      --cache "$CACHE" --out "$OUT" --gpu "$gpu" \
      > "$log" 2>&1 &
  echo "$!" >> logs/pids.txt
  i=$((i + 1))
  sleep 5           # stagger, so they don't all hammer the filesystem at once
done

echo ""
echo "Launched ${#CONFIGS[@]} jobs. Useful commands:"
echo "  tail -f logs/control.log             # watch one run"
echo "  grep -H 'exhaust' logs/*.log | tail -30  # watch them all"
echo "  nvidia-smi                           # check GPU load"
echo "  python scripts/collect_results.py --out $OUT  # build the summary table + plots"
echo ""
echo "Waiting for all jobs..."
wait
echo "ALL DONE. Now run:  python scripts/collect_results.py --out $OUT"
