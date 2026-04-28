###############################################################################
# Adaptation of NVIDIA StyleGAN2-ada Code to generate images in a 5Gc context
# Copyright (c) 2021, NVIDIA CORPORATION.  
###############################################################################
# Librairies
import argparse
import json
import re
from pathlib import Path
from threading import Lock
from typing import List, Optional, Sequence, Tuple, Union
import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy


# Fixed runtime settings (deployment defaults).
DEFAULT_NETWORK_PKL = 'vanilla_weights/CelebA_128x128.pkl'
DEFAULT_OUTDIR = 'outputs'
DEFAULT_TRUNCATION_PSI = 1.0
DEFAULT_NOISE_MODE = 'none'  # const|random|none


_MODEL_LOCK = Lock()
_CACHED_DEVICE_TYPE: Optional[str] = None
_CACHED_DEVICE_INSTANCE: Optional[torch.device] = None
_CACHED_FORCE_FP32: Optional[bool] = None
_CACHED_G: Optional[torch.nn.Module] = None


def num_range(s: str) -> List[int]:
    """Accept either 'a,b,c' or a range 'a-c' and return a list of ints."""
    m = re.match(r'^(\d+)-(\d+)$', s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    return [int(x) for x in s.split(',')]


def load_prompt_json(prompt_json_path: Union[str, Path]) -> Tuple[int, List[int], str]:
    """Load the user prompt JSON (assumed to be well-formed).

    Expected format:
        {
            "prompt": [number_of_images, seeds],
            "device_type": "cpu"|"cuda"|"cuda:0"|...
        }
    """
    data = json.loads(Path(prompt_json_path).read_text(encoding='utf-8'))
    num_images = int(data['prompt'][0])
    seeds_raw = data['prompt'][1]
    seeds = num_range(seeds_raw) if isinstance(seeds_raw, str) else list(seeds_raw)
    device_type = str(data['device_type'])
    return num_images, seeds, device_type


def load_prompt_data(data: dict) -> Tuple[int, List[int], str]:
    """Load prompt from an in-memory dict (assumed well-formed).

    Expected format:
        {
            "prompt": [number_of_images, seeds],
            "device_type": "cpu"|"cuda"|"cuda:0"|...
        }
    """
    num_images = int(data['prompt'][0])
    seeds_raw = data['prompt'][1]
    seeds = num_range(seeds_raw) if isinstance(seeds_raw, str) else list(seeds_raw)
    device_type = str(data['device_type'])
    return num_images, seeds, device_type


def load_network(device: str, network_pkl: str):
    
    # Init the device for the futur loading
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit('CUDA device requested but CUDA is not available. Use --device=cpu or run with GPU support.')
    device_instance = torch.device(device)

    # Loading the SG2-ada model 
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device_instance)  

    # Switch network to evaluation mode for optimization inference time
    G.eval()
    torch.set_grad_enabled(False)

    # CPU robustness:
    # StyleGAN2-ADA can internally switch to FP16 (`use_fp16`) unless we force FP32.
    # On CPU, some conv kernels are not implemented for float16, so we must avoid FP16.
    force_fp32 = (device_instance.type == 'cpu')

    # This deployment assumes an unconditional network.
    if G.c_dim != 0:
        raise SystemExit('This script expects an unconditional network (c_dim=0).')
    
    return device_instance, force_fp32, G


def get_or_load_network(device_type: str) -> Tuple[torch.device, bool, torch.nn.Module]:
    """Return a cached network, reloading it only if device_type changes."""
    global _CACHED_DEVICE_TYPE, _CACHED_DEVICE_INSTANCE, _CACHED_FORCE_FP32, _CACHED_G

    normalized = str(device_type)
    with _MODEL_LOCK:
        if _CACHED_G is not None and _CACHED_DEVICE_TYPE == normalized:
            print(f"[model-cache] hit device_type={normalized}")
            return _CACHED_DEVICE_INSTANCE, _CACHED_FORCE_FP32, _CACHED_G

        previous = _CACHED_DEVICE_TYPE
        if previous is None:
            print(f"[model-cache] load device_type={normalized}")
        else:
            print(f"[model-cache] reload device_type={previous} -> {normalized}")

        # If switching away from CUDA, free cache when possible.
        if _CACHED_DEVICE_TYPE is not None and str(_CACHED_DEVICE_TYPE).startswith('cuda') and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        device_instance, force_fp32, G = load_network(normalized, network_pkl=DEFAULT_NETWORK_PKL)
        _CACHED_DEVICE_TYPE = normalized
        _CACHED_DEVICE_INSTANCE = device_instance
        _CACHED_FORCE_FP32 = force_fp32
        _CACHED_G = G
        return device_instance, force_fp32, G


def generate_images(
    device_instance: torch.device,
    force_fp32: bool,
    G: torch.nn.Module,
    seeds: Sequence[int],
    num_images: int,
    output_directory: Union[str, Path],
    truncation_psi: float = DEFAULT_TRUNCATION_PSI,
    noise_mode: str = DEFAULT_NOISE_MODE,
) -> List[Path]:
    outdir = Path(output_directory)
    outdir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    for seed_idx, seed in enumerate(seeds):
        z = torch.from_numpy(np.random.RandomState(seed).randn(num_images, G.z_dim)).to(device_instance)
        label = torch.zeros([z.shape[0], G.c_dim], device=device_instance)
        with torch.inference_mode():
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode, force_fp32=force_fp32)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        for i in range(img.shape[0]):
            out_path = outdir / f'seed{seed:04d}_img{i:02d}.png'
            PIL.Image.fromarray(img[i].cpu().numpy(), 'RGB').save(out_path)
            saved.append(out_path)

    return saved


def run_from_prompt_json(prompt_json_path: Union[str, Path]) -> List[Path]:
    """Entry point callable by FastAPI (via subprocess) or directly in Python."""
    num_images, seeds, device_type = load_prompt_json(prompt_json_path)
    device_instance, force_fp32, G = get_or_load_network(device_type)
    return generate_images(
        device_instance=device_instance,
        force_fp32=force_fp32,
        G=G,
        seeds=seeds,
        num_images=num_images,
        output_directory=DEFAULT_OUTDIR,
        truncation_psi=DEFAULT_TRUNCATION_PSI,
        noise_mode=DEFAULT_NOISE_MODE,
    )


def run_from_prompt_data(data: dict) -> List[Path]:
    """Entry point for FastAPI (pass JSON body as dict)."""
    num_images, seeds, device_type = load_prompt_data(data)
    device_instance, force_fp32, G = get_or_load_network(device_type)
    paths = generate_images(
        device_instance=device_instance,
        force_fp32=force_fp32,
        G=G,
        seeds=seeds,
        num_images=num_images,
        output_directory=DEFAULT_OUTDIR,
        truncation_psi=DEFAULT_TRUNCATION_PSI,
        noise_mode=DEFAULT_NOISE_MODE,
    )
    # Convert to absolute paths so other services can find the files
    return [p.resolve() for p in paths]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description='Generate images from a user JSON prompt (all other settings are fixed in this script).'
    )
    parser.add_argument('--prompt-json', dest='prompt_json', required=True, help='Path to JSON prompt file')
    args = parser.parse_args(argv)

    run_from_prompt_json(args.prompt_json)
    

if __name__ == '__main__':
    main()

#----------------------------------------------------------------------------

# python generate.py --prompt-json input/prompt.json