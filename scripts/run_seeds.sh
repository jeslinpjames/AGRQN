#!/bin/bash
# ==============================================================================
# run_seeds.sh -- the three-seed runs reported in the paper (seeds 42, 1, 7).
# All runs: 120 epochs, no early stopping. Run from the repository root.
#
#   bash scripts/run_seeds.sh --dry                # show the plan
#   bash scripts/run_seeds.sh                      # run every group
#   bash scripts/run_seeds.sh main reward          # run only these groups
#   GPUS="0 1 2 3" bash scripts/run_seeds.sh       # choose GPUs (default: 0)
#
# Groups:
#   main     the four main configurations (Table 2): VGG-16 / DINOv2 descriptor x
#            no prior / DINO prior. Seed 42 of each is run by run_all.sh or by hand.
#   selfatt  the same-backbone prior, desc_dinov2_selfattn (Table 6). Needs the cache built
#            by scripts/make_dinov2_attn_cache.py.
#   reward   the plain-head control and the seven reward interventions (Table 7).
#
# Tags match the result files in results/results_final/.
# ==============================================================================
set -u

VOC="${VOC:-./data/VOC2012_train_val}"
WEIGHTS="./checkpoints/vgg16_5class.pth"
CACHE_VGG="./checkpoints/feat_cache"
CACHE_DV2="./checkpoints/feat_cache_dinov2"
CACHE_SELF="./checkpoints/feat_cache_dinov2_selfattn"
OUT="./results/results_final"
EPOCHS=120
GPUS=(${GPUS:-0})

DRY=0
GROUPS_ARG=()
for x in "$@"; do
  if [ "$x" = "--dry" ]; then DRY=1; else GROUPS_ARG+=("$x"); fi
done
[ ${#GROUPS_ARG[@]} -eq 0 ] && GROUPS_ARG=(main selfatt reward)

# tag|config|seed|cache
JOBS=()
for g in "${GROUPS_ARG[@]}"; do
  case "$g" in
    main)
      JOBS+=(
        "desc_vgg_none_s1|desc_vgg_none|1|$CACHE_VGG"
        "desc_vgg_none_s7|desc_vgg_none|7|$CACHE_VGG"
        "final_s42|duel_wide|42|$CACHE_VGG"
        "final_s1|duel_wide|1|$CACHE_VGG"
        "final_s7|duel_wide|7|$CACHE_VGG"
        "desc_dinov2_none_s1|desc_dinov2_none|1|$CACHE_DV2"
        "desc_dinov2_none_s7|desc_dinov2_none|7|$CACHE_DV2"
        "desc_dinov2_s1|desc_dinov2|1|$CACHE_DV2"
        "desc_dinov2_s7|desc_dinov2|7|$CACHE_DV2"
      ) ;;
    selfatt)
      for s in 42 1 7; do
        JOBS+=("desc_dinov2_selfattn_s$s|desc_dinov2_selfattn|$s|$CACHE_SELF")
      done ;;
    reward)
      JOBS+=("control_s1|control|1|$CACHE_VGG" "control_s7|control|7|$CACHE_VGG")
      for c in steppen005 steppen02 gamma99 eta10 pbrs expiry_fail aux_head; do
        JOBS+=("iv_$c|$c|42|$CACHE_VGG" "iv_${c}_s1|$c|1|$CACHE_VGG" "iv_${c}_s7|$c|7|$CACHE_VGG")
      done ;;
    *) echo "unknown group '$g' (main | selfatt | reward)"; exit 1 ;;
  esac
done

if [ "$DRY" -eq 0 ]; then
  [ -d "$VOC/JPEGImages" ] || { echo "FATAL: no JPEGImages under $VOC (set VOC=...)"; exit 1; }
  [ -f "$WEIGHTS" ] || { echo "FATAL: $WEIGHTS not found. Run train_classifier.py first."; exit 1; }
  mkdir -p "$OUT" logs
fi

NG=${#GPUS[@]}
echo "GPUs: ${GPUS[*]}   jobs: ${#JOBS[@]}"
i=0
for j in "${JOBS[@]}"; do
  IFS='|' read -r tag cfg seed cache <<< "$j"
  gpu=${GPUS[$((i % NG))]}
  if [ -e "$OUT/${tag}_test.json" ]; then
    echo "  $tag already done, skipping"; i=$((i + 1)); continue
  fi
  printf "  %-28s cfg=%-22s seed=%-3s gpu=%s cache=%s\n" "$tag" "$cfg" "$seed" "$gpu" "$cache"
  if [ "$DRY" -eq 0 ]; then
    [ -d "$cache" ] || { echo "FATAL: cache $cache missing (see README)"; exit 1; }
    nohup python -u scripts/train_agrqn.py --config "$cfg" --tag "$tag" --seed "$seed" \
        --voc "$VOC" --weights "$WEIGHTS" --cache "$cache" --out "$OUT" \
        --gpu "$gpu" --epochs "$EPOCHS" > "logs/${tag}.log" 2>&1 &
    sleep 4
  fi
  i=$((i + 1))
done
[ "$DRY" -eq 1 ] && { echo "(dry run)"; exit 0; }
echo "Launched. Logs in logs/. Waiting for all jobs..."
wait
echo "DONE"
