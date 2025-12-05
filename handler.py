"""
RunPod Serverless Handler for LTX-Video
"""
# IMPORTANT: Set all temp/cache directories to /tmp BEFORE importing anything
# This ensures everything uses /tmp (which has more space) instead of /root/.cache
import os

# HuggingFace cache
os.environ.setdefault("HF_HOME", "/tmp/huggingface_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/huggingface_cache")

# Python temp files
os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("TMP", "/tmp")
os.environ.setdefault("TEMP", "/tmp")

# Create directories if they don't exist
os.makedirs("/tmp/huggingface_cache", exist_ok=True)
os.makedirs("/tmp/outputs", exist_ok=True)

import runpod
from pathlib import Path
from ltx_video.inference import infer, InferenceConfig
from typing import Dict, Any


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serverless handler for LTX-Video inference
    
    Expected input:
    {
        "input": {
            "prompt": "Your prompt here",
            "conditioning_media_paths": ["path/to/image.jpg"],  # Optional
            "conditioning_start_frames": [0],  # Optional
            "height": 512,
            "width": 512,
            "num_frames": 121,
            "seed": 42,
            "pipeline_config": "configs/ltxv-2b-0.9.8-distilled.yaml"
        }
    }
    """
    try:
        input_data = event.get("input", {})
        
        # Validate required fields
        if "prompt" not in input_data:
            return {
                "error": "Missing required field: 'prompt'"
            }
        
        # Set default values
        prompt = input_data["prompt"]
        conditioning_media_paths = input_data.get("conditioning_media_paths")
        conditioning_start_frames = input_data.get("conditioning_start_frames", [0] if conditioning_media_paths else None)
        height = input_data.get("height", 512)
        width = input_data.get("width", 512)
        num_frames = input_data.get("num_frames", 121)
        seed = input_data.get("seed", 42)
        pipeline_config = input_data.get("pipeline_config", "configs/ltxv-2b-0.9.8-distilled.yaml")
        
        # Create output directory
        output_dir = Path("/tmp/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / "output.mp4")
        
        # Create inference config
        config = InferenceConfig(
            prompt=prompt,
            conditioning_media_paths=conditioning_media_paths,
            conditioning_start_frames=conditioning_start_frames,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            pipeline_config=pipeline_config,
            output_path=output_path
        )
        
        # Run inference
        infer(config)
        
        # Return success with output path
        return {
            "status": "success",
            "output_path": output_path,
            "message": "Video generated successfully"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


# Start RunPod serverless
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

