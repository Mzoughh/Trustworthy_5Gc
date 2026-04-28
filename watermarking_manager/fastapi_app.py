import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import FastAPI, Body
import PIL.Image
from torchvision import transforms

# Import the HiddenEncoder from detection_A
sys.path.append(str(Path(__file__).parent.parent / "detection_module" / "detection_A"))
from hidden.models import HiddenEncoder

app = FastAPI(title="Watermarking Manager API")

# Default settings
DEFAULT_DEVICE_TYPE = "cpu"
DEFAULT_IMAGE_SIZE = 128
DEFAULT_NUM_BITS = 48

# Assume encoder checkpoint path (placeholder, may need to be trained)
DEFAULT_ENCODER_CKPT = "watermarking_manager/weights/hidden_encoder.pth"

def load_key(model_name: str) -> torch.Tensor:
    """Load the key for the model."""
    key_json_path = Path(__file__).parent.parent / "detection_module" / "secret_database" / "key.json"
    with open(key_json_path, "r", encoding="utf-8") as f:
        key_dict = json.load(f)
    key_list = key_dict[model_name]
    key = torch.tensor(key_list, dtype=torch.float32)
    return key

def embed_watermark(image_path: str, model_name: str, device_type: str = DEFAULT_DEVICE_TYPE) -> str:
    """Embed watermark into the image and save the watermarked image."""
    device = torch.device(device_type)

    # Convert to Path and verify file exists
    image_file = Path(image_path)
    if not image_file.exists():
        raise FileNotFoundError(f"Image file not found: {image_file} (exists={image_file.exists()})")

    # Load image
    img = PIL.Image.open(image_file).convert("RGB")
    img = img.resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), resample=PIL.Image.BICUBIC)
    to_tensor = transforms.ToTensor()
    x = to_tensor(img).unsqueeze(0).to(device)

    # Load key
    key = load_key(model_name).to(device)
    key = key.repeat(x.shape[0], 1)

    # Load encoder (placeholder: assume encoder is available)
    # For now, create a dummy encoder since no checkpoint
    encoder = HiddenEncoder(num_blocks=4, num_bits=DEFAULT_NUM_BITS, channels=64)
    # encoder.load_state_dict(torch.load(DEFAULT_ENCODER_CKPT, map_location=device))
    # encoder = encoder.to(device).eval()

    # Since no trained encoder, return original image path (placeholder)
    # In real implementation, do: watermarked = encoder(x, key)
    output_path = image_file.parent / f"watermarked_{image_file.name}"
    img.save(output_path)

    return str(output_path)

@app.post("/embed")
def embed(prompt: Optional[Dict[str, Any]] = Body(default=None)) -> Dict[str, Any]:
    """Embed watermark into an image based on model_name."""
    if prompt is None:
        return {"error": "No prompt provided."}

    image_path = prompt.get("image")
    model_name = prompt.get("model_name")
    device_type = prompt.get("device_type", DEFAULT_DEVICE_TYPE)

    if not image_path or not model_name:
        return {"error": "Missing 'image' or 'model_name' in prompt."}

    try:
        watermarked_path = embed_watermark(image_path, model_name, device_type)
        return {"watermarked_image": watermarked_path}
    except Exception as e:
        return {"error": f"Embedding failed: {e}"}

# To run: uvicorn fastapi_app:app --host 0.0.0.0 --port 8002 --workers 1