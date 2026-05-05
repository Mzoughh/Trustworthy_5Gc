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
from torch_utils import misc


# Fixed runtime settings (deployment defaults).
DEFAULT_NETWORK_PKL = 'vanilla_weights/CelebA_128_128_BB.pkl'
DEFAULT_OUTDIR = 'outputs'
DEFAULT_NOISE_MODE = 'const'  # const|random|none


_MODEL_LOCK = Lock()
_CACHED_DEVICE_TYPE: Optional[str] = None
_CACHED_DEVICE_INSTANCE: Optional[torch.device] = None
_CACHED_FORCE_FP32: Optional[bool] = None
_CACHED_G_SYNTHESIS: Optional[torch.nn.Module] = None
_CACHED_G_MAPPING: Optional[torch.nn.Module] = None
_CACHED_Z_DIM : Optional[int] = None
_CACHED_C_DIM : Optional[int] = None

###############################################
# UTILS FUNCTIONS # 
def num_range(s: str) -> List[int]:
    """Accept either 'a,b,c' or a range 'a-c' and return a list of ints."""
    m = re.match(r'^(\d+)-(\d+)$', s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    return [int(x) for x in s.split(',')]

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
###############################################

##################################
# LOADING NETWORK AND CACHED IT
def load_network(device: str, network_pkl_path: str):
    ###############################################
    # FUNCTION INSPIRED FROM THE EVALUATION METRICS SCRIPT OF SG2
    ###############################################

    # Init the device for the futur loading
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit('CUDA device requested but CUDA is not available. Use --device=cpu or run with GPU support.')
    device_instance = torch.device(device)

    # Loading the SG2-ada model 
    with dnnlib.util.open_url(network_pkl_path, verbose=True) as f:
        network_dict = legacy.load_network_pkl(f)
        G = network_dict['G_ema'] # subclass of torch.nn.Module

    ### DEFAULT USAGE => Inference mode
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    G = G.eval().requires_grad_(False) 
    G_mapping = G.mapping.to(device_instance)
    G_synthesis = G.synthesis.to(device_instance)

    # CPU robustness:
    # StyleGAN2-ADA can internally switch to FP16 (`use_fp16`) unless we force FP32.
    # On CPU, some conv kernels are not implemented for float16, so we must avoid FP16.
    force_fp32 = (device_instance.type == 'cpu') 

    z_dim = G.z_dim
    c_dim = G.c_dim
    
    return device_instance, force_fp32, G_mapping, G_synthesis, z_dim, c_dim


def get_or_load_network(device_type: str) -> Tuple[torch.device, bool, torch.nn.Module]:
    """Return a cached network, reloading it only if device_type changes."""
   
    global _CACHED_DEVICE_TYPE, _CACHED_DEVICE_INSTANCE, _CACHED_FORCE_FP32, _CACHED_G_MAPPING, _CACHED_G_SYNTHESIS, _CACHED_Z_DIM, _CACHED_C_DIM
    normalized = str(device_type)
    
    with _MODEL_LOCK:
        # THE NETWORK IS ALREADY LOAD ON THE CORRECT DEVICE
        if _CACHED_G_SYNTHESIS is not None and _CACHED_DEVICE_TYPE == normalized:
            print(f"[model-cache] hit device_type={normalized}")
            return _CACHED_DEVICE_INSTANCE, _CACHED_FORCE_FP32, _CACHED_G_MAPPING, _CACHED_G_SYNTHESIS, _CACHED_Z_DIM, _CACHED_C_DIM
        previous = _CACHED_DEVICE_TYPE
        if previous is None:
            print(f"[model-cache] load device_type={normalized}")
        
        # THE NETWORK IS RELOADING ON THE NEW ONE 
        else:
            print(f"[model-cache] reload device_type={previous} -> {normalized}")

        if _CACHED_DEVICE_TYPE is not None and str(_CACHED_DEVICE_TYPE).startswith('cuda') and torch.cuda.is_available(): # If switching away from CUDA, free cache when possible.
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        # RELOADING 
        device_instance, force_fp32, G_mapping, G_synthesis, z_dim, c_dim = load_network(normalized, network_pkl_path=DEFAULT_NETWORK_PKL)
        _CACHED_DEVICE_TYPE = normalized
        _CACHED_DEVICE_INSTANCE = device_instance
        _CACHED_FORCE_FP32 = force_fp32
        _CACHED_G_MAPPING = G_mapping
        _CACHED_G_SYNTHESIS = G_synthesis
        _CACHED_Z_DIM = z_dim
        _CACHED_C_DIM = c_dim

        return device_instance, force_fp32, G_mapping, G_synthesis, z_dim, c_dim


def generate_images(
    device_instance: torch.device,
    force_fp32: bool,
    G_mapping: torch.nn.Module,
    G_synthesis: torch.nn.Module,
    z_dim : int,
    c_dim : int,
    seeds: Sequence[int],
    num_images: int,
    output_directory: Union[str, Path],
    noise_mode: str = DEFAULT_NOISE_MODE,
    style_mixing_prob : int = 0
) -> List[Path]:
    outdir = Path(output_directory)
    outdir.mkdir(parents=True, exist_ok=True)


    saved: List[Path] = []
    for _ , seed in enumerate(seeds):
    
        # INIT SEED
        torch.manual_seed(seed)
        if device_instance == 'cuda' :
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # INTIT LATTENT
        latent_vector = torch.randn([num_images, z_dim], device=device_instance)
        trigger_label = torch.zeros([num_images, c_dim], device=device_instance)
        

        # DO THE GENERATION 
        with misc.ddp_sync(G_mapping, sync=True):
            ws = G_mapping(latent_vector, trigger_label)
            if style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = G_mapping(torch.randn_like(latent_vector), trigger_label, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(G_synthesis, sync=True):
            # WARNING WE WANT TO FIXE NOISE AS CST
            img = G_synthesis(ws, noise_mode=noise_mode, force_fp32=force_fp32)
                
        # SAVE THE IMAGE
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        for i in range(img.shape[0]):
            out_path = outdir / f'seed{seed:04d}_img{i:02d}.png'
            PIL.Image.fromarray(img[i].cpu().numpy(), 'RGB').save(out_path)
            saved.append(out_path)

    return saved

##################################
def run_from_prompt_data(data: dict) -> List[Path]:
    """Entry point for FastAPI (pass JSON body as dict)."""
    num_images, seeds, device_type = load_prompt_data(data)
    device_instance, force_fp32, G_mapping, G_synthesis, z_dim, c_dim = get_or_load_network(device_type)
    return generate_images(
        device_instance=device_instance,
        force_fp32=force_fp32,
        G_mapping=G_mapping,
        G_synthesis=G_synthesis,
        z_dim=z_dim,
        c_dim=c_dim,
        seeds=seeds,
        num_images=num_images,
        output_directory=DEFAULT_OUTDIR,
    )

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description='Generate images from a user JSON prompt (all other settings are fixed in this script).'
    )
    parser.add_argument('--prompt-json', dest='prompt_json', required=True, help='Path to JSON prompt file')
    args = parser.parse_args(argv)

    run_from_prompt_json(args.prompt_json)



###############################################
###############################################
# DEBUGGING FUNCTION TO USE THE SCRIPT ALONE #
def load_prompt_json(prompt_json_path: Union[str, Path]) -> Tuple[int, List[int], str]:
    """Load the user prompt JSON (assumed to be well-formed). """
    data = json.loads(Path(prompt_json_path).read_text(encoding='utf-8'))
    num_images = int(data['prompt'][0])
    seeds_raw = data['prompt'][1]
    seeds = num_range(seeds_raw) if isinstance(seeds_raw, str) else list(seeds_raw)
    device_type = str(data['device_type'])
    return num_images, seeds, device_type

def run_from_prompt_json(prompt_json_path: Union[str, Path]) -> List[Path]:
    """Entry point callable by FastAPI (via subprocess) or directly in Python."""
    num_images, seeds, device_type = load_prompt_json(prompt_json_path)
    device_instance, force_fp32, G_mapping, G_synthesis, z_dim, c_dim = get_or_load_network(device_type)
    return generate_images(
        device_instance=device_instance,
        force_fp32=force_fp32,
        G_mapping=G_mapping,
        G_synthesis=G_synthesis,
        z_dim=z_dim,
        c_dim=c_dim,
        seeds=seeds,
        num_images=num_images,
        output_directory=DEFAULT_OUTDIR,
    )
###################################
###################################



if __name__ == '__main__':
    main()

#----------------------------------------------------------------------------

# python generate.py --prompt-json input/prompt.json