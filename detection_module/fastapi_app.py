import sys
import json
import importlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Body

# Dynamically add all detection_X folders to sys.path for import compatibility
for d in Path(__file__).parent.glob("detection_*"):
    if d.is_dir():
        sys.path.append(str(d.resolve()))

app = FastAPI(title="Detection Modules API")

_DEMO_PROMPT_PATH = Path(__file__).parent / "input" / "prompt.json"
_OUTPUT_DIR = Path(__file__).parent / "output"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/detect")
def detect(prompt: Optional[Dict[str, Any]] = Body(default=None)) -> Dict[str, Any]:
    """Run detection based on the model_name in the prompt JSON."""
    if prompt is None:
        prompt = json.loads(_DEMO_PROMPT_PATH.read_text(encoding="utf-8"))

    model_name = prompt.get("model_name")
    if not model_name:
        return {"error": "Missing 'model_name' in prompt."}

    module_name = f"detection_{model_name}"  # e.g. detection_A
    try:
        detection_module = importlib.import_module(module_name)
    except Exception as e:
        return {"error": f"Could not import module {module_name}: {e}"}

    try:
        if hasattr(detection_module, "run_from_prompt_data"):
            score = float(detection_module.run_from_prompt_data(prompt))  # no-disk
        else:
            return {"error": f"{module_name} does not implement run_from_prompt_data (no-disk mode)."}
    except Exception as e:
        return {"error": f"Detection failed: {e}"}

    prompt["detection_score"] = score
    out_path = _OUTPUT_DIR / f"{model_name}_{datetime.utcnow():%S%fZ}.json"
    out_path.write_text(json.dumps(prompt, indent=2), encoding="utf-8")

    return {"detection_score": score, "output_json": str(out_path)}

# To run: uvicorn fastapi_app:app --host 0.0.0.0 --port 8001 --workers 1
# Example: curl -X POST http://localhost:8001/detect -H "Content-Type: application/json" -d '@input/prompt.json'
