import sys
from pathlib import Path
from fastapi import FastAPI, Body
from typing import Any, Dict, Optional
import json
import importlib

# Dynamically add all generate_X folders to sys.path for import compatibility
for d in Path(__file__).parent.glob("generate_*"):
    if d.is_dir():
        sys.path.append(str(d.resolve()))

app = FastAPI(title="Generation Modules API")

_DEMO_PROMPT_PATH = Path(__file__).parent / "input" / "prompt.json"


def _jsonify_paths(x: Any) -> Any:
    """Convert Path (and list/tuple of Path) to JSON-serializable types."""
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (list, tuple)):
        return [str(p) if isinstance(p, Path) else p for p in x]
    return x


@app.post("/generate")
def generate(prompt: Optional[Dict[str, Any]] = Body(default=None)) -> Dict[str, Any]:
    """
    Run generation based on the model_name in the prompt JSON.
    If no JSON body is provided, uses 'input/prompt.json' as default.
    The prompt must contain a 'model_name' field (e.g. "A" for generate_A).
    """
    if prompt is None:
        prompt = json.loads(_DEMO_PROMPT_PATH.read_text(encoding="utf-8"))

    model_name = prompt.get("model_name")
    if not model_name:
        return {"error": "Missing 'model_name' in prompt."}

    module_name = f"generate_{model_name}"  # e.g. generate_A
    module_path = f"{module_name}.generate_{model_name}"  # e.g. generate_A.generate_A

    try:
        generate_module = importlib.import_module(module_path)
    except Exception as e:
        return {"error": f"Could not import module {module_path}: {e}"}

    try:
        if hasattr(generate_module, "run_from_prompt_data"):
            result = generate_module.run_from_prompt_data(prompt)  # no-disk
        else:
            return {"error": f"{module_path} does not implement run_from_prompt_data (no-disk mode)."}
    except Exception as e:
        return {"error": f"Generation failed: {e}"}

    return {"generation_result": _jsonify_paths(result)}


# To run: uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --workers 1
# Example: curl -X POST http://localhost:8000/generate -H "Content-Type: application/json" -d '@input/prompt.json'