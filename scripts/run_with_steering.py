import argparse
import json
import os
import math
import typing as tp

import pandas as pd
from diffusers import DiffusionPipeline

from core.controller import CrossAttentionOutputSteering, VectorControl
from core.diffusion_steering import diffusion_register_vector_controls_with_hooks
from core.utils import SUPPORTED_DIFFUSION_MODELS, get_device, init_pipeline_for_image_model, load_vectors, run_image_model

# SD-1.x/2.x cross-attention indices in hook traversal order. Down blocks are
# ordered high to low resolution; up blocks are ordered low to high.
SD_LAYER_SCOPES = {
    'all': None,
    'mid': {('mid', 0)},
    'high': {
        ('down', 0), ('down', 1),
        ('up', 6), ('up', 7), ('up', 8),
    },
    'low': {
        ('down', 4), ('down', 5), ('mid', 0),
        ('up', 0), ('up', 1), ('up', 2),
    },
    'low_mid': {
        ('down', 2), ('down', 3), ('down', 4), ('down', 5), ('mid', 0),
        ('up', 0), ('up', 1), ('up', 2), ('up', 3), ('up', 4), ('up', 5),
    },
}


def load_prompts(args):
    """Load evaluation prompts: either from CLIP templates or COCO captions."""
    if args.generate_concept != 'coco':
        with open(args.template_path) as f:
            templates = json.load(f)
        return [t.format(args.generate_concept) for t in templates], args.num_images_per_prompt
    else:
        df = pd.read_csv('data/coco/coco_30k.csv')
        prompts = [p for p in df['prompt'] if 'horse' not in p.lower()]
        if args.max_samples:
            prompts = prompts[:args.max_samples]
        return prompts, 1


def hook_model(pipeline: DiffusionPipeline, device: tp.Any, args: argparse.Namespace) -> VectorControl:
    if args.command is None:
        return None

    # multi-concept: comma-separated concept paths -> erase all simultaneously
    source_concepts = [load_vectors(p) for p in args.concept_path.split(',') if p]
    retain_concepts = None
    if args.retain_concept_path:
        retain_concepts = [load_vectors(p) for p in args.retain_concept_path.split(',') if p]

    if args.layer_scope != 'all' and args.model_name not in {'sd14', 'sd15', 'sd21'}:
        raise ValueError('--layer_scope is defined only for SD-1.x/2.x U-Nets')

    vector_control = CrossAttentionOutputSteering(
        source_concepts=source_concepts,
        strength=args.steering_strength,
        device=device,
        intermediate_clipping=args.intermediate_clipping,
        use_first_diffusion_step=not args.use_all_diffusion_steps,
        retain_concepts=retain_concepts,
        retain_mode=args.retain_mode,
        layer_scope=SD_LAYER_SCOPES[args.layer_scope],
    )

    diffusion_register_vector_controls_with_hooks(pipeline.unet, vector_control)
    return vector_control


def main(args: argparse.Namespace):
    if args.steering_strength is None and args.command is not None:
        raise ValueError('--steering_strength (float) must be specified for steering')

    if args.command is None and args.steering_strength is not None:
        raise ValueError('--steering_strength is provided but no erase command specified')

    pipeline = init_pipeline_for_image_model(model=args.model_name)
    pipeline.set_progress_bar_config(disable=True)
    device = get_device()

    vector_control = hook_model(pipeline, device, args)

    prompts, num_images_per_prompt = load_prompts(args)
    skipped = generated = 0

    print(
        f'Generating images for concept {args.generate_concept} '
        f'with strength {args.steering_strength}, layer scope '
        f'{getattr(args, "layer_scope", "all")}'
    )
    for p_idx, prompt in enumerate(prompts):
        # Shard split (result-neutral: per-prompt deterministic seeds)
        if args.num_shards > 1 and (p_idx % args.num_shards) != args.shard_id:
            continue
        num_batches = math.ceil(num_images_per_prompt / args.batch_size)
        for batch_id in range(0, num_batches):
            seed = args.seed + batch_id
            num_images = min(args.batch_size, num_images_per_prompt - batch_id * args.batch_size)

            output_paths = [f'{args.output_dir}/{prompt}/{seed}-{idx}.png' for idx in range(num_images)]
            if all(os.path.exists(path) for path in output_paths):
                skipped += num_images
                continue
            generated += num_images
            images = run_image_model(
                model_type=args.model_name,
                pipe=pipeline,
                prompt=prompt,
                seed=seed,
                device=device,
                num_images=num_images,
            )
            if vector_control is not None:
                vector_control.reset()
            os.makedirs(os.path.dirname(output_paths[0]), exist_ok=True)
            for path, image in zip(output_paths, images):
                image.save(path, format='PNG')

    print(f'Skipped {skipped} images, generated {generated} images')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    main_parser = parser.add_argument_group('Common arguments')

    # Generation params
    main_parser.add_argument('--model_name', type=str, choices=SUPPORTED_DIFFUSION_MODELS, required=True,
                             help='Diffusion model name used for generation')
    main_parser.add_argument('--generate_concept', type=str, required=True, help='Concept for which to generate images, or "coco" for COCO captions')
    main_parser.add_argument('--output_dir', type=str, required=True, help='Directory where generated images should be written')
    main_parser.add_argument('--num_images_per_prompt', type=int, default=10, help='Number of images to generate for each prompt')
    main_parser.add_argument('--batch_size', type=int, default=1, help='Batch size used for image generation')
    main_parser.add_argument('--seed', type=int, default=0, help='Starting seed for each prompt')
    main_parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use from the dataset')
    main_parser.add_argument('--template_path', type=str, default='data/templates/clip_templates.json', help='Path to template JSON for evaluation prompts')
    main_parser.add_argument('--num_shards', type=int, default=1, help='Total shards for multi-GPU split')
    main_parser.add_argument('--shard_id', type=int, default=0, help='This process shard index in [0, num_shards)')

    # Steering params
    main_parser.add_argument('--steering_strength', type=float, default=None, help='Steering strength beta (2.0 on SD-1.x, 3.0 on SD-2.1/SDXL)')
    main_parser.add_argument('--intermediate_clipping', action='store_true', help='Gate steering by max(<d, c>, 0)')
    main_parser.add_argument('--use_all_diffusion_steps', action='store_true', help='Use per-step steering vectors instead of first-step only')

    subparsers = parser.add_subparsers(dest='command')

    # Params for concept erasure
    erase_parser = subparsers.add_parser('erase')
    erase_parser.add_argument('--concept_path', type=str, required=True,
                              help='Comma-separated paths to concept vectors to erase')
    erase_parser.add_argument('--retain_concept_path', type=str, default=None,
                              help='Comma-separated paths to retain-concept vectors (retain-orthogonal steering)')
    erase_parser.add_argument('--retain_mode', type=str, default='fixed',
                              choices=['fixed', 'auto_layer'],
                              help='Retain projection strength: fixed (hard projection) or auto_layer (overlap-adaptive)')
    erase_parser.add_argument('--layer_scope', type=str, default='all',
                              choices=list(SD_LAYER_SCOPES),
                              help='Ablation: steer only this U-Net resolution scope')

    args = parser.parse_args()

    main(args)
