import torch

from diffusers import StableDiffusionPipeline, DiffusionPipeline


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


SUPPORTED_DIFFUSION_MODELS = ['sd14', 'sd15', 'sd21', 'sdxl']


def init_pipeline_for_image_model(model: str) -> DiffusionPipeline:
    if model == 'sd14':
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
            cache_dir='./cache',
            device_map='balanced',
            safety_checker=None,
        )
    elif model == 'sd15':
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            cache_dir='./cache',
            device_map='balanced',
            safety_checker=None,
        )
    elif model == 'sd21':
        # 512px SD-2.1 base, matching the evaluation protocol of the paper.
        pipe = StableDiffusionPipeline.from_pretrained(
            "sd2-community/stable-diffusion-2-1-base",
            torch_dtype=torch.float16,
            cache_dir='./cache',
            device_map='balanced',
        )
    elif model == 'sdxl':
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            cache_dir='./cache',
            device_map='balanced',
            safety_checker=None,
        )
    else:
        raise ValueError(f'Unknown model: {model}')
    return pipe


def get_num_denoising_steps(model: str) -> int:
    if model in ('sd14', 'sd15', 'sd21'):
        return 50
    elif model in ('sdxl',):
        return 30
    else:
        raise ValueError(f'Unknown model type: {model}')


def run_image_model(model_type: str, pipe, prompt: str, seed: int, device: torch.device, num_images: int = 1):
    return pipe(
        prompt=prompt,
        num_inference_steps=get_num_denoising_steps(model_type),
        generator=torch.Generator(device=device).manual_seed(seed),
        num_images_per_prompt=num_images,
    ).images


def load_vectors(path: str):
    """Load steering vectors saved by scripts/estimate_steering_vectors.py."""
    return torch.load(path, weights_only=False)
