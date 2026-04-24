# ======================================================================
# CLASS FUNCTION INSPIRED FROM: CARL DE SOUSA TRIAS MPAI IMPLEMENTATION
# Credit: We wrote an implementation inspired from HiDDen @Meta
# ======================================================================

# ──────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────

## TORCH
import argparse
import json
import torch
import torch.nn as nn
from threading import Lock
from typing import List, Optional, Tuple, Dict, Any 

## SPECIFIC FOR METHOD HiDDen
from hidden.models import HiddenDecoder
from torchvision import transforms
import PIL.Image
from utils_custom.normalization import minmax_normalize
from pathlib import Path

# Fixed runtime settings (deployment defaults).
# Path to the decoder checkpoint used for detection.
DEFAULT_DECODER_CKPT = "detection_A/weights/hidden_replicate_whit.pth"
# Input image size expected by this decoder.
DEFAULT_IMAGE_SIZE = 128
# Default device if prompt doesn't specify it.
DEFAULT_DEVICE_TYPE = "cpu"  # "cpu" | "cuda" | "cuda:0" ...

# For a POC, we derive the key deterministically from model_name.
# If you prefer fixed keys, replace this function to load keys from a file.
DEFAULT_NUM_BITS = 48

_MODEL_LOCK = Lock()
_CACHED_DEVICE_TYPE: Optional[str] = None
_CACHED_TOOL: Optional["TONDI_tools"] = None
# ──────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
# HIDDEN CLASS
# ──────────────────────────────────────────────────────────────
class Params():
    def __init__(self, encoder_depth:int, encoder_channels:int, decoder_depth:int, decoder_channels:int, num_bits:int,
                attenuation:str, scale_channels:bool, scaling_i:float, scaling_w:float):
        # encoder and decoder parameters
        self.encoder_depth = encoder_depth
        self.encoder_channels = encoder_channels
        self.decoder_depth = decoder_depth
        self.decoder_channels = decoder_channels
        self.num_bits = num_bits
        # attenuation parameters
        self.attenuation = attenuation
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w


# ──────────────────────────────────────────────────────────────
# METHOD CLASS
# ──────────────────────────────────────────────────────────────
class TONDI_tools():

    def __init__(self,device) -> None:
        
        self.device = device
        self.NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.UNNORMALIZE_IMAGENET = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
        self.default_transform = transforms.Compose([transforms.ToTensor(), self.NORMALIZE_IMAGENET])  

        # Load Network Architecture
        params = Params(
            encoder_depth=4, encoder_channels=64, decoder_depth=8, decoder_channels=64, num_bits=48,
            attenuation="jnd", scale_channels=False, scaling_i=1, scaling_w=1.5
        ) 
        decoder = HiddenDecoder(
            num_blocks=params.decoder_depth, 
            num_bits=params.num_bits, 
            channels=params.decoder_channels
        )

        decoder = torch.jit.load(DEFAULT_DECODER_CKPT, map_location='cpu')

        # Freezing HiDDen Decoder for Evaluation 
        self.msg_decoder = decoder.to(self.device).eval()
        for param in [*self.msg_decoder.parameters()]:
            param.requires_grad = False
        super(TONDI_tools, self).__init__()

    # ----------------------------------------------------------
    # MARK LOSS 
    # ---------------------------------------------------------
    def detection(self, gen_imgs, keys):
        
        # Convert tuple images to tensors if necessary
        if isinstance(gen_imgs, tuple):
            gen_imgs = torch.stack(gen_imgs)

        # Normalization MIN_MAX (LayerNorm-style): -> [0,1]
        epsilon = 1e-8  # numerical stabilization
        gen_imgs_shifted, _, _ = minmax_normalize(gen_imgs, epsilon=epsilon)

        # Normalize  to ImageNet stats (Do a step of UNNORMALIZE if done in the dataloader)
        gen_imgs_imnet = self.NORMALIZE_IMAGENET(gen_imgs_shifted) 
        
        # Extract watermark
        decoded = self.msg_decoder((gen_imgs_imnet))

        # Compute bit accuracy
        diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
        bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
        bit_accs_avg = torch.mean(bit_accs).item()

        return bit_accs_avg 


def load_prompt_json(prompt_json_path: str) -> Tuple[str, str, str]:
    """Load the user prompt JSON (assumed to be well-formed).

    Expected format:
        {
          "image": "path_image.png",
          "model_name": "A",
          "device_type": "cpu"|"cuda"|"cuda:0"|...   # optional
        }
    """
    data = json.loads(Path(prompt_json_path).read_text(encoding="utf-8"))
    image_path = str(data["image"])
    model_name = str(data["model_name"])
    device_type = str(data.get("device_type", DEFAULT_DEVICE_TYPE))
    return image_path, model_name, device_type


def load_image_tensor(image_path: str, device: torch.device) -> torch.Tensor:
    """Load a single image as a tensor of shape (1, 3, H, W) on device."""
    img = PIL.Image.open(image_path).convert("RGB")
    img = img.resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), resample=PIL.Image.BICUBIC)
    to_tensor = transforms.ToTensor()
    x = to_tensor(img).unsqueeze(0).to(device)
    return x


def key_from_model_name(model_name: str) -> torch.Tensor:
    """Load the key associated with the model from the file secret_database/key.json."""
    key_json_path = Path(__file__).parent.parent / "secret_database" / "key.json"
    with open(key_json_path, "r", encoding="utf-8") as f:
        key_dict = json.load(f)
    key_list = key_dict[model_name]
    key = torch.tensor([key_list], dtype=torch.float32)
    return key


def get_or_load_detector(device_type: str) -> "TONDI_tools":
    """Cache the detector and reload only if device_type changes."""
    global _CACHED_DEVICE_TYPE, _CACHED_TOOL
    normalized = str(device_type)
    with _MODEL_LOCK:
        if _CACHED_TOOL is not None and _CACHED_DEVICE_TYPE == normalized:
            print(f"[detector-cache] hit device_type={normalized}")
            return _CACHED_TOOL

        previous = _CACHED_DEVICE_TYPE
        if previous is None:
            print(f"[detector-cache] load device_type={normalized}")
        else:
            print(f"[detector-cache] reload device_type={previous} -> {normalized}")

        if previous is not None and str(previous).startswith("cuda") and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        device = torch.device(normalized)
        tool = TONDI_tools(device=device)
        _CACHED_DEVICE_TYPE = normalized
        _CACHED_TOOL = tool
        return tool
    
def load_prompt_data(data: Dict[str, Any]) -> Tuple[str, str, str]:
    """Load the user prompt from an in-memory dict.

    Expected format:
        {
          "image": "path_image.png",
          "model_name": "A",
          "device_type": "cpu"|"cuda"|"cuda:0"|...   # optional
        }
    """
    image_path = str(data["image"])
    model_name = str(data["model_name"])
    device_type = str(data.get("device_type", DEFAULT_DEVICE_TYPE))
    return image_path, model_name, device_type


def run_from_prompt_data(data: Dict[str, Any]) -> float:
    """No-disk entry point for FastAPI (accepts JSON body as dict)."""
    image_path, model_name, device_type = load_prompt_data(data)
    tool = get_or_load_detector(device_type)
    x = load_image_tensor(image_path, device=tool.device)
    keys = key_from_model_name(model_name).to(tool.device)
    keys = keys.repeat(x.shape[0], 1)
    score = tool.detection(x, keys)
    return float(score)


def run_from_prompt_json(prompt_json_path: str) -> float:
    """Entry point similar to ai_provider/generate.py."""
    image_path, model_name, device_type = load_prompt_json(prompt_json_path)
    tool = get_or_load_detector(device_type)
    x = load_image_tensor(image_path, device=tool.device)
    keys = key_from_model_name(model_name).to(tool.device)
    # Match batch size
    keys = keys.repeat(x.shape[0], 1)
    score = tool.detection(x, keys)
    return float(score)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run detection from a JSON prompt.")
    parser.add_argument("--prompt-json", dest="prompt_json", required=True, help="Path to JSON prompt file")
    args = parser.parse_args(argv)
    score = run_from_prompt_json(args.prompt_json)
    print(f"detection_score={score:.6f}")


if __name__ == "__main__":
    main()


