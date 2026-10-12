#!/bin/bash
# Instance / style protocol (Table 3 of the paper) for one backbone.
#   bash examples/table3_protocol.sh sd14      # also: sd15 sd21 sdxl
# 80 CLIP templates x 3 images, seed 42, six erase settings, four evaluated
# concepts per side; the retain set is the side's non-erased concepts.
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD}"

MODEL=${1:-sd14}
case $MODEL in
  sd14|sd15) BETA=2.0; BS=3 ;;
  sd21)      BETA=3.0; BS=3 ;;
  sdxl)      BETA=3.0; BS=1 ;;   # 1024px: one image per batch
  *) echo "unknown backbone $MODEL"; exit 1 ;;
esac

VEC=vectors/$MODEL
OUT=results/table3_$MODEL
GEN="python scripts/run_with_steering.py --model_name $MODEL --template_path data/templates/clip_templates.json \
     --num_images_per_prompt 3 --batch_size $BS --seed 42"
STEER="--steering_strength $BETA --intermediate_clipping"

declare -A NAME=( [snoopy]="Snoopy" [mickey]="Mickey Mouse" [spongebob]="Spongebob" [pikachu]="Pikachu"
                  [vangogh]="Van Gogh" [picasso]="Picasso" [rembrandt]="Rembrandt" [hokusai]="Hokusai" )
declare -A MODE=( [snoopy]=concrete [mickey]=concrete [spongebob]=concrete [pikachu]=concrete
                  [vangogh]=style [picasso]=style [rembrandt]=style [hokusai]=style )
INSTANCE="snoopy mickey spongebob pikachu"
STYLE="vangogh picasso rembrandt hokusai"

# ---- 1) steering vectors, one per concept ----
for c in $INSTANCE $STYLE; do
  python scripts/estimate_steering_vectors.py --model_name $MODEL --concept "${NAME[$c]}" \
    --mode ${MODE[$c]} --num_prompts 50 --save_name $c --output_dir $VEC
done

# ---- 2) unedited reference generations ----
for c in $INSTANCE $STYLE; do
  $GEN --generate_concept "${NAME[$c]}" --output_dir $OUT/orig/$c
done

# ---- 3) the six erase settings ----
run_setting () {   # $1 = setting name, $2 = erased concepts, $3 = side concepts
  local setting=$1 erased=$2 side=$3
  local fvec="" rvec=""
  for c in $erased; do fvec="$fvec,$VEC/$c.pt"; done
  for c in $side; do case " $erased " in *" $c "*) ;; *) rvec="$rvec,$VEC/$c.pt";; esac; done
  fvec=${fvec#,}; rvec=${rvec#,}
  for c in $side; do
    # RASteer: per-step vectors, retain-orthogonal steering, overlap-adaptive calibration
    $GEN --generate_concept "${NAME[$c]}" --output_dir $OUT/ours/$setting/$c \
      $STEER --use_all_diffusion_steps erase --concept_path $fvec \
      --retain_concept_path $rvec --retain_mode auto_layer
    # CASteer baseline: first-step vectors, raw erasure direction
    $GEN --generate_concept "${NAME[$c]}" --output_dir $OUT/casteer/$setting/$c \
      $STEER erase --concept_path $fvec
  done
}
run_setting e_snoopy                  "snoopy"                  "$INSTANCE"
run_setting e_snoopy_mickey           "snoopy mickey"           "$INSTANCE"
run_setting e_snoopy_mickey_spongebob "snoopy mickey spongebob" "$INSTANCE"
run_setting e_vangogh                 "vangogh"                 "$STYLE"
run_setting e_picasso                 "picasso"                 "$STYLE"
run_setting e_rembrandt               "rembrandt"               "$STYLE"

# ---- 4) metrics: CLIP score, CLIP accuracy, FID per (setting, method, concept) ----
T3_ROOT=$OUT python eval/table3_eval.py casteer ours
echo "wrote $OUT/metrics.json"
