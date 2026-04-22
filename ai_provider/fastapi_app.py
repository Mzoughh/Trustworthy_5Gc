from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI

from generate import run_from_prompt_data

app = FastAPI(title="Demo_GAN API")

_DEMO_PROMPT_PATH = Path("input/prompt.json")


@app.post("/generate")
def generate(prompt: Optional[Dict[str, Any]] = Body(default=None)) -> Dict[str, Any]:
    """Generate images.

    For now, if no JSON body is provided, this endpoint uses `input/prompt.json`
    to simulate an incoming user prompt.

    Expected JSON format:
      {
        "prompt": [number_of_images, seeds],
        "device_type": "cpu"|"cuda"|"cuda:0"|...
      }
    """
    if prompt is None:
        prompt = json.loads(_DEMO_PROMPT_PATH.read_text(encoding="utf-8"))

    paths = run_from_prompt_data(prompt)
    return {"images": [str(p) for p in paths]}


# uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --workers 1
# curl -X POST http://localhost:8000/generate
