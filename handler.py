"""
RunPod Serverless Handler for LTX-Video
"""
# IMPORTANT: Set all temp/cache directories to /tmp BEFORE importing anything
# This ensures everything uses /tmp (which has more space) instead of /root/.cache
import os
import shutil

# HuggingFace cache
os.environ.setdefault("HF_HOME", "/tmp/huggingface_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/huggingface_cache")

# Python temp files
os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("TMP", "/tmp")
os.environ.setdefault("TEMP", "/tmp")

# ImageIO/FFmpeg temp directory (critical for video processing)
os.environ.setdefault("IMAGEIO_TEMP_DIR", "/tmp/imageio_temp")
os.environ.setdefault("IMAGEIO_FFMPEG_EXE", "")  # Let imageio find it, but use our temp dir

# Create directories if they don't exist
os.makedirs("/tmp/huggingface_cache", exist_ok=True)
os.makedirs("/tmp/outputs", exist_ok=True)
os.makedirs("/tmp/imageio_temp", exist_ok=True)

import runpod
from pathlib import Path
from ltx_video.inference import infer, InferenceConfig
from typing import Dict, Any

# Configure imageio to use /tmp for temp files (after import)
try:
    import imageio
    # Set imageio temp directory
    imageio.config.known_extensions['.mp4'] = 'ffmpeg'
    # imageio will use TMPDIR which we set above
except ImportError:
    pass


def get_disk_space(path: str) -> tuple:
    """Get free disk space in GB for a given path."""
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024 ** 3)
    total_gb = stat.total / (1024 ** 3)
    return free_gb, total_gb


def cleanup_temp_files():
    """Clean up temporary files to free space."""
    temp_dirs = [
        "/tmp/imageio_temp",
        "/tmp/outputs",
    ]
    
    freed_mb = 0
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            try:
                for item in os.listdir(temp_dir):
                    item_path = os.path.join(temp_dir, item)
                    try:
                        if os.path.isfile(item_path):
                            size = os.path.getsize(item_path) / (1024 ** 2)  # MB
                            os.remove(item_path)
                            freed_mb += size
                        elif os.path.isdir(item_path):
                            size = sum(
                                os.path.getsize(os.path.join(dirpath, filename))
                                for dirpath, dirnames, filenames in os.walk(item_path)
                                for filename in filenames
                            ) / (1024 ** 2)  # MB
                            shutil.rmtree(item_path)
                            freed_mb += size
                    except Exception:
                        continue
            except Exception:
                continue
    
    return freed_mb


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
        # Check disk space before processing
        tmp_free, tmp_total = get_disk_space("/tmp")
        root_free, root_total = get_disk_space("/")
        
        # If /tmp has less than 2GB free, clean up old files
        if tmp_free < 2.0:
            freed_mb = cleanup_temp_files()
            tmp_free, _ = get_disk_space("/tmp")
        
        # If still low on space, return error with diagnostics
        if tmp_free < 1.0:
            return {
                "status": "error",
                "error": f"Insufficient disk space. /tmp has {tmp_free:.2f}GB free (need at least 1GB). Root has {root_free:.2f}GB free.",
                "error_type": "DiskSpaceError",
                "diagnostics": {
                    "tmp_free_gb": round(tmp_free, 2),
                    "tmp_total_gb": round(tmp_total, 2),
                    "root_free_gb": round(root_free, 2),
                    "root_total_gb": round(root_total, 2)
                }
            }
        
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

