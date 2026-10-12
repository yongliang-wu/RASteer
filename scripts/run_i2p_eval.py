"""
Generate images for the I2P (Inappropriate Image Prompts) benchmark.

Downloads the I2P dataset (4,703 prompts) from HuggingFace and generates
images with optional erasure applied. Exposed body parts are then counted
with eval/count_nudity.py.

Usage:
    # Generate without steering (baseline)
    python scripts/run_i2p_eval.py \
        --model_name sd14 --output_dir results/i2p/orig

    # Generate with nudity erasure
    python scripts/run_i2p_eval.py \
        --model_name sd14 --output_dir results/i2p/casteer \
        --steering_strength 2.0 --intermediate_clipping \
        --concept_path vectors/sd14/nudity.pt
"""
import argparse
import os

from datasets import load_dataset

from core.controller import CrossAttentionOutputSteering
from core.diffusion_steering import diffusion_register_vector_controls_with_hooks
from core.utils import SUPPORTED_DIFFUSION_MODELS, get_device, init_pipeline_for_image_model, load_vectors, run_image_model


def main(args):
    print("Loading I2P dataset...")
    ds = load_dataset("AIML-TUDA/i2p", split="train")
    print(f"Loaded {len(ds)} I2P prompts")

    pipeline = init_pipeline_for_image_model(model=args.model_name)
    pipeline.set_progress_bar_config(disable=True)
    device = get_device()

    vector_control = None
    if args.concept_path is not None:
        concept = load_vectors(args.concept_path)
        retain_concepts = None
        if args.retain_concept_path:
            retain_concepts = [load_vectors(p) for p in args.retain_concept_path.split(',') if p]
        vector_control = CrossAttentionOutputSteering(
            source_concepts=[concept],
            strength=args.steering_strength,
            device=device,
            intermediate_clipping=args.intermediate_clipping,
            use_first_diffusion_step=not args.use_all_diffusion_steps,
            retain_concepts=retain_concepts,
            retain_mode=args.retain_mode,
        )
        diffusion_register_vector_controls_with_hooks(pipeline.unet, vector_control)

    os.makedirs(args.output_dir, exist_ok=True)
    skipped = generated = 0

    for i, row in enumerate(ds):
        # Shard split: result-neutral, each image keyed by global index i with row['sd_seed']
        if args.num_shards > 1 and (i % args.num_shards) != args.shard_id:
            continue
        output_path = os.path.join(args.output_dir, f"{i:05d}.png")
        if os.path.exists(output_path):
            skipped += 1
            continue

        images = run_image_model(
            model_type=args.model_name,
            pipe=pipeline,
            prompt=row['prompt'],
            seed=row['sd_seed'],
            device=device,
        )
        if vector_control is not None:
            vector_control.reset()

        images[0].save(output_path)
        generated += 1

        if (i + 1) % 100 == 0:
            print(f"Progress: {i + 1}/{len(ds)}")

    print(f"Done. Generated {generated}, skipped {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate I2P images with optional concept erasure")
    parser.add_argument('--model_name', type=str, choices=SUPPORTED_DIFFUSION_MODELS, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--concept_path', type=str, default=None, help='Path to steering vector for erasure')
    parser.add_argument('--steering_strength', type=float, default=2.0)
    parser.add_argument('--intermediate_clipping', action='store_true')
    parser.add_argument('--num_shards', type=int, default=1, help='Total number of shards for multi-GPU split')
    parser.add_argument('--shard_id', type=int, default=0, help='This process shard index in [0, num_shards)')
    parser.add_argument('--use_all_diffusion_steps', action='store_true', help='Use per-step steering vectors instead of first-step only')
    parser.add_argument('--retain_concept_path', type=str, default=None, help='Comma-separated retain-concept vectors (retain-orthogonal steering)')
    parser.add_argument('--retain_mode', type=str, default='fixed', choices=['fixed', 'auto_layer'], help='Retain projection mode')

    args = parser.parse_args()
    main(args)
