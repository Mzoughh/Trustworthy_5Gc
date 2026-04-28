import sys
import json
import requests
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import FastAPI, Body

app = FastAPI(title="AI Manager API")

# Default URLs for services (adjust as needed)
AI_PROVIDER_URL = "http://localhost:8000/generate"
WATERMARKING_MANAGER_URL = "http://localhost:8002/embed"
DETECTION_MODULE_URL = "http://localhost:8001/detect"

@app.post("/full_pipeline")
def full_pipeline(prompt: Optional[Dict[str, Any]] = Body(default=None)) -> Dict[str, Any]:
    """Run the full pipeline: generate image, embed watermark, detect."""
    if prompt is None:
        return {"error": "No prompt provided."}

    # Step 1: Generate image
    gen_response = requests.post(AI_PROVIDER_URL, json=prompt)
    if gen_response.status_code != 200:
        return {"error": f"Generation failed: {gen_response.text}"}
    gen_data = gen_response.json()
    if "error" in gen_data:
        return gen_data

    # Assume generation_result has image paths
    # For simplicity, assume it's a list of paths
    image_paths = gen_data["generation_result"]
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    results = []
    for img_path in image_paths:
        # Step 2: Embed watermark
        embed_prompt = {
            "image": img_path,
            "model_name": prompt.get("model_name"),
            "device_type": prompt.get("device_type", "cpu")
        }
        embed_response = requests.post(WATERMARKING_MANAGER_URL, json=embed_prompt)
        if embed_response.status_code != 200:
            return {"error": f"Embedding failed for {img_path}: {embed_response.text}"}
        embed_data = embed_response.json()
        if "error" in embed_data:
            return embed_data

        watermarked_path = embed_data["watermarked_image"]

        # Step 3: Detect
        detect_prompt = {
            "image": watermarked_path,
            "model_name": prompt.get("model_name"),
            "device_type": prompt.get("device_type", "cpu")
        }
        detect_response = requests.post(DETECTION_MODULE_URL, json=detect_prompt)
        if detect_response.status_code != 200:
            return {"error": f"Detection failed for {watermarked_path}: {detect_response.text}"}
        detect_data = detect_response.json()
        if "error" in detect_data:
            return detect_data

        results.append({
            "original_image": img_path,
            "watermarked_image": watermarked_path,
            "detection_score": detect_data["detection_score"]
        })

    return {"pipeline_results": results}

# To run: uvicorn fastapi_app:app --host 0.0.0.0 --port 8003 --workers 1