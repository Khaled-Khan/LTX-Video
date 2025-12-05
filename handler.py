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


def cleanup_huggingface_cache():
    """Aggressively clean up HuggingFace cache to free space, but preserve text encoder."""
    cache_dir = Path("/tmp/huggingface_cache")
    hub_dir = cache_dir / "hub"
    
    if not hub_dir.exists():
        return 0.0
    
    freed_mb = 0.0
    
    # Preserve text encoder (PixArt) - it's large and takes time to download
    text_encoder_pattern = "PixArt-alpha--PixArt-XL-2-1024-MS"
    
    # Remove cached models (but preserve text encoder)
    try:
        for item in hub_dir.iterdir():
            try:
                if item.is_dir():
                    # Skip text encoder - preserve it
                    if text_encoder_pattern in str(item):
                        continue
                    
                    # Calculate size
                    size = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(item)
                        for filename in filenames
                    ) / (1024 ** 2)  # MB
                    shutil.rmtree(item)
                    freed_mb += size
            except Exception:
                continue
    except Exception:
        pass
    
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
        # Use smaller defaults to avoid GPU OOM (22GB GPU can't handle 512x512x121)
        height = input_data.get("height", 256)
        width = input_data.get("width", 256)
        num_frames = input_data.get("num_frames", 25)  # Reduced from 121
        seed = input_data.get("seed", 42)
        offload_to_cpu = input_data.get("offload_to_cpu", False)
        # Default to 2B model (smaller, requires less disk space)
        pipeline_config = input_data.get("pipeline_config", "configs/ltxv-2b-0.9.8-distilled.yaml")
        
        # Check disk space before processing
        tmp_free, tmp_total = get_disk_space("/tmp")
        root_free, root_total = get_disk_space("/")
        
        # Determine required space based on model
        # Note: Text encoder (PixArt) is ~19GB, but it's cached after first download
        # So we need space for: main model + text encoder (if not cached) + processing overhead
        is_2b_model = "2b" in pipeline_config.lower()
        
        # Check if text encoder is already cached
        text_encoder_cache = Path("/tmp/huggingface_cache/hub/models--PixArt-alpha--PixArt-XL-2-1024-MS")
        text_encoder_cached = text_encoder_cache.exists() and any(text_encoder_cache.rglob("*.safetensors"))
        
        if text_encoder_cached:
            # Text encoder already cached, only need space for main model
            required_space = 3.0 if is_2b_model else 7.0
        else:
            # Need to download text encoder (~19GB) + main model
            required_space = 22.0 if is_2b_model else 26.0  # 19GB text encoder + model + overhead
        
        # Aggressively clean up if space is low
        if tmp_free < required_space:
            # Clean up temp files
            freed_mb = cleanup_temp_files()
            # Clean up HuggingFace cache (models will be re-downloaded)
            freed_mb += cleanup_huggingface_cache()
            tmp_free, _ = get_disk_space("/tmp")
        
        # If still low on space, return error with diagnostics
        if tmp_free < required_space:
            return {
                "status": "error",
                "error": f"Insufficient disk space. /tmp has {tmp_free:.2f}GB free (need at least {required_space}GB for {('2B' if is_2b_model else '13B')} model). Root has {root_free:.2f}GB free.",
                "error_type": "DiskSpaceError",
                "diagnostics": {
                    "tmp_free_gb": round(tmp_free, 2),
                    "tmp_total_gb": round(tmp_total, 2),
                    "root_free_gb": round(root_free, 2),
                    "root_total_gb": round(root_total, 2),
                    "required_space_gb": required_space,
                    "model_type": "2B" if is_2b_model else "13B",
                    "suggestion": "Use pipeline_config: 'configs/ltxv-2b-0.9.8-distilled.yaml' for a smaller model"
                }
            }
        
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
            output_path=output_path,
            offload_to_cpu=offload_to_cpu
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

