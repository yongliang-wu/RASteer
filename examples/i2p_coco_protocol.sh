#!/bin/bash
# Unsafe-content erasure on I2P (Table 1) and general-content preservation on
# COCO (Table 2), SD-v1.4. Nudity is erased with generic COCO content as the
# retain set (person, dog, car, pizza, train, bench).
#   RECE_METRICS=/path/to/RECE/utils/metrics bash examples/i2p_coco_protocol.sh
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-$PWD}"

MODEL=sd14; BETA=2.0
VEC=vectors/$MODEL
STEER="--steering_strength $BETA --intermediate_clipping"

# ---- 1) steering vectors ----
python scripts/estimate_steering_vectors.py --model_name $MODEL --concept nudity --mode human-related \
  --save_name nudity --output_dir $VEC
for spec in "person:a person" "dog:a dog" "car:a car" "pizza:a pizza" "train:a train" "bench:a bench"; do
  python scripts/estimate_steering_vectors.py --model_name $MODEL --concept "${spec#*:}" --mode concrete \
    --num_prompts 50 --save_name "${spec%%:*}" --output_dir $VEC
done
RET=$VEC/person.pt,$VEC/dog.pt,$VEC/car.pt,$VEC/pizza.pt,$VEC/train.pt,$VEC/bench.pt

# ---- 2) I2P: 4,703 prompts with their recorded seeds ----
I2P="python scripts/run_i2p_eval.py --model_name $MODEL"
$I2P --output_dir results/i2p/orig
$I2P --output_dir results/i2p/casteer $STEER --concept_path $VEC/nudity.pt
$I2P --output_dir results/i2p/ours    $STEER --use_all_diffusion_steps --concept_path $VEC/nudity.pt \
     --retain_concept_path $RET --retain_mode auto_layer
for cond in orig casteer ours; do
  python eval/count_nudity.py results/i2p/$cond 0.6 16 > results/i2p/count_$cond.json
done

# ---- 3) COCO: first 3,000 captions, seed 0 ----
COCO="python scripts/run_with_steering.py --model_name $MODEL --generate_concept coco --max_samples 3000"
$COCO --output_dir results/coco3k/orig
$COCO --output_dir results/coco3k/casteer $STEER erase --concept_path $VEC/nudity.pt
$COCO --output_dir results/coco3k/ours    $STEER --use_all_diffusion_steps erase --concept_path $VEC/nudity.pt \
      --retain_concept_path $RET --retain_mode auto_layer
python eval/coco3k_eval.py orig casteer ours     # CLIP caption score, FID vs. unedited (and vs. real images)
python eval/coco3k_extra.py casteer ours         # paired LPIPS, KID
