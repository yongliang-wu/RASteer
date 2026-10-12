"""
Compute steering vectors from paired positive/negative prompts.

For each prompt pair differing only by the target concept, we collect
cross-attention outputs, average across pairs per layer, and compute:

    steering_vector = normalize(mean(CA_pos) - mean(CA_neg))

Three prompt modes are supported:
  - concrete: "{ImageNet class} with {concept}" vs "{ImageNet class}"
  - style:    "{ImageNet class}, {concept} style" vs "{ImageNet class}"
  - human-related: "{person} {scene}, {concept}" vs "{person} {scene}"
"""
import argparse
import os

import torch
import tqdm

from core.construct_prompts import get_prompts_concrete, get_prompts_style, get_prompts_human_related
from core.diffusion_steering import diffusion_register_vector_controls_with_hooks
from core.utils import SUPPORTED_DIFFUSION_MODELS, init_pipeline_for_image_model, run_image_model, get_device
from core.vector_dump import CrossAttentionOutputStatsCollector


def collect_means(pipeline, model_name, prompts, device):
    """Run model on a list of prompts and return mean CA outputs per layer."""
    stats = CrossAttentionOutputStatsCollector()

    hook_manager = diffusion_register_vector_controls_with_hooks(pipeline.unet, stats)

    for prompt in tqdm.tqdm(prompts, desc="Collecting activations"):
        _ = run_image_model(
            model_type=model_name,
            pipe=pipeline,
            prompt=prompt,
            seed=0,
            device=device,
        )
        stats.reset()

    means = stats.means
    hook_manager.remove_hooks()
    return means


def compute_steering_vectors(pos_means, neg_means):
    """
    Compute normalized steering vectors: normalize(pos_mean - neg_mean) per layer.
    """
    steering_vectors = {}

    for step in pos_means:
        steering_vectors[step] = {}
        for place in pos_means[step]:
            steering_vectors[step][place] = []
            for layer_idx in range(len(pos_means[step][place])):
                pos_vec = pos_means[step][place][layer_idx]
                neg_vec = neg_means[step][place][layer_idx]

                sv = pos_vec - neg_vec
                norm = sv.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                sv = sv / norm

                steering_vectors[step][place].append(sv)

    return steering_vectors


def main(args):
    save_name = args.save_name if args.save_name else args.concept
    output_path = os.path.join(args.output_dir, f"{save_name}.pt")
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate paired prompts
    if args.mode == 'concrete':
        prompts_pos, prompts_neg = get_prompts_concrete(args.concept, num=args.num_prompts)
    elif args.mode == 'style':
        prompts_pos, prompts_neg = get_prompts_style(args.concept, num=args.num_prompts)
    elif args.mode == 'human-related':
        prompts_pos, prompts_neg = get_prompts_human_related(args.concept)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print(f"Generated {len(prompts_pos)} prompt pairs for concept '{args.concept}' (mode: {args.mode})")

    pipeline = init_pipeline_for_image_model(model=args.model_name)
    pipeline.set_progress_bar_config(disable=True)
    device = get_device()

    # Collect mean CA outputs for positive and negative prompts
    print("Collecting positive prompt activations...")
    pos_means = collect_means(pipeline, args.model_name, prompts_pos, device)

    print("Collecting negative prompt activations...")
    neg_means = collect_means(pipeline, args.model_name, prompts_neg, device)

    # Compute and save steering vectors
    steering_vectors = compute_steering_vectors(pos_means, neg_means)

    torch.save(steering_vectors, output_path)
    print(f"Saved steering vectors to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute steering vectors from paired prompts")
    parser.add_argument('--model_name', type=str, choices=SUPPORTED_DIFFUSION_MODELS, required=True,
                        help='Diffusion model to use for collecting activations')
    parser.add_argument('--concept', type=str, required=True,
                        help='Target concept to compute steering vector for')
    parser.add_argument('--mode', type=str, choices=['concrete', 'style', 'human-related'], default='concrete',
                        help='Prompt template mode')
    parser.add_argument('--num_prompts', type=int, default=50,
                        help='Number of prompt pairs (for concrete/style modes)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for steering vectors')
    parser.add_argument('--save_name', type=str, default=None,
                        help='Filename stem for the saved vector (defaults to --concept); decouples prompt text from filename')

    args = parser.parse_args()
    main(args)
