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

# PyTorch CUDA memory management - conservative settings to reduce fragmentation
# max_split_size_mb: Smaller chunks reduce fragmentation (32MB chunks)
# Note: expandable_segments can cause assertion failures in some PyTorch versions, so we disable it
# roundup_power2_divisions: Helps with memory alignment
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:32,roundup_power2_divisions:2")

# CUDA error handling - make errors synchronous for better debugging
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

# ImageIO/FFmpeg temp directory (critical for video processing)
os.environ.setdefault("IMAGEIO_TEMP_DIR", "/tmp/imageio_temp")
os.environ.setdefault("IMAGEIO_FFMPEG_EXE", "")  # Let imageio find it, but use our temp dir

# Create directories if they don't exist
os.makedirs("/tmp/huggingface_cache", exist_ok=True)
os.makedirs("/tmp/outputs", exist_ok=True)
os.makedirs("/tmp/imageio_temp", exist_ok=True)

import runpod
from runpod.serverless.utils.rp_cleanup import clean
import base64
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


def cleanup_temp_files(protected_files=None):
    """Clean up temporary files to free space.
    
    Args:
        protected_files: List of file paths to protect from deletion
    """
    if protected_files is None:
        protected_files = []
    
    temp_dirs = [
        "/tmp/imageio_temp",
        "/tmp/outputs",
        "/tmp/input_images",
    ]
    
    freed_mb = 0
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            try:
                for item in os.listdir(temp_dir):
                    item_path = os.path.join(temp_dir, item)
                    # Skip protected files
                    if item_path in protected_files or any(item_path.endswith(prot) for prot in protected_files):
                        continue
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
    
    if freed_mb > 0:
        print(f"[DEBUG] Freed {freed_mb:.2f} MB from temp files")
    
    return freed_mb


def aggressive_cleanup():
    """Aggressively clean up all possible temp files and caches."""
    freed_mb = 0.0
    
    # Clean temp files
    freed_mb += cleanup_temp_files()
    
    # Clean HuggingFace cache (except text encoder)
    freed_mb += cleanup_huggingface_cache()
    
    # Clean Python cache files
    import glob
    for cache_dir in ["/tmp/__pycache__", "/tmp/*.pyc", "/tmp/*.pyo"]:
        for item in glob.glob(cache_dir):
            try:
                if os.path.isfile(item):
                    size = os.path.getsize(item) / (1024 ** 2)
                    os.remove(item)
                    freed_mb += size
                elif os.path.isdir(item):
                    size = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(item)
                        for filename in filenames
                    ) / (1024 ** 2)
                    shutil.rmtree(item)
                    freed_mb += size
            except Exception:
                continue
    
    # Clean system temp files older than 1 hour
    try:
        import time
        current_time = time.time()
        for root, dirs, files in os.walk("/tmp"):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if os.path.getmtime(file_path) < current_time - 3600:  # 1 hour old
                        size = os.path.getsize(file_path) / (1024 ** 2)
                        os.remove(file_path)
                        freed_mb += size
                except Exception:
                    continue
    except Exception:
        pass
    
    print(f"[DEBUG] Aggressive cleanup freed {freed_mb:.2f} MB ({freed_mb/1024:.2f} GB)")
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
    
    Note: GPU memory is cleared at the start of each handler call to ensure
    clean state for each job.
    
    Expected input:
    {
        "input": {
            "prompt": "Your prompt here",
            "conditioning_media_paths": ["data:image/jpeg;base64,..."],  # Optional: base64 image or file path on RunPod
            "conditioning_start_frames": [0],  # Optional
            "height": 512,
            "width": 512,
            "num_frames": 121,
            "seed": 42,
            "pipeline_config": "configs/ltxv-2b-0.9.8-distilled.yaml"
        }
    }
    
    For images, you can use:
    - Base64: "data:image/jpeg;base64,iVBORw0KG..." (for local images)
    - File path: "/path/to/image.jpg" (on RunPod server)
    """
    # CRITICAL: Clear GPU memory at the very start of handler
    # This ensures we start with a clean slate, even if previous job didn't clean up
    try:
        import torch
        import gc
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            return {
                "status": "error",
                "error": "CUDA is not available on this worker",
                "error_type": "CUDAUnavailableError"
            }
        
        # Check if device is accessible (not busy) - with retry
        device = None
        for retry in range(3):
            try:
                device = torch.cuda.current_device()
                device_props = torch.cuda.get_device_properties(device)
                print(f"[DEBUG] Using GPU device {device}: {device_props.name}")
                break
            except RuntimeError as e:
                error_msg = str(e).lower()
                if "busy" in error_msg or "unavailable" in error_msg:
                    if retry < 2:
                        print(f"[DEBUG] GPU busy, retrying in 1 second... (attempt {retry + 1}/3)")
                        import time
                        time.sleep(1)
                        continue
                    else:
                        return {
                            "status": "error",
                            "error": f"GPU device is busy or unavailable after 3 retries. Worker may need restart. Error: {str(e)}",
                            "error_type": "CUDABusyError",
                            "suggestion": "Restart the RunPod worker or wait a few minutes and try again"
                        }
                raise
        
        # Clear GPU memory
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # Get memory info
        initial_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        print(f"[DEBUG] Handler started. Initial GPU free memory: {initial_free / 1024**3:.2f} GB")
        if initial_free < 5 * 1024**3:
            print(f"[DEBUG] WARNING: Very low GPU memory at handler start! Only {initial_free / 1024**3:.2f} GB free")
            
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "busy" in error_msg or "unavailable" in error_msg:
            return {
                "status": "error",
                "error": f"GPU device is busy or unavailable: {str(e)}",
                "error_type": "CUDABusyError",
                "suggestion": "Restart the RunPod worker or wait a few minutes and try again"
            }
        raise
    except Exception as e:
        print(f"[DEBUG] Warning: Could not clear GPU at handler start: {e}")
    
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
        
        # Handle base64 encoded images (for local images sent via API)
        # IMPORTANT: Save images FIRST before any cleanup
        saved_image_paths = []  # Track saved images to protect from cleanup
        if conditioning_media_paths:
            temp_image_dir = Path("/tmp/input_images")
            temp_image_dir.mkdir(parents=True, exist_ok=True)
            
            processed_paths = []
            for i, media in enumerate(conditioning_media_paths):
                if isinstance(media, str):
                    # Check if it's base64 encoded (starts with data:image or is long base64 string)
                    if media.startswith("data:image") or (len(media) > 100 and not os.path.exists(media)):
                        # Extract base64 data
                        if media.startswith("data:image"):
                            # Format: data:image/png;base64,iVBORw0KG...
                            header, encoded = media.split(",", 1)
                            ext = header.split("/")[1].split(";")[0]  # Extract extension
                        else:
                            # Assume it's raw base64, try to detect format
                            encoded = media
                            ext = "jpg"  # Default to jpg
                        
                        # Decode and save
                        try:
                            image_data = base64.b64decode(encoded)
                            temp_path = temp_image_dir / f"input_image_{i}.{ext}"
                            with open(temp_path, "wb") as f:
                                f.write(image_data)
                            # Verify file was saved
                            if not temp_path.exists():
                                return {
                                    "status": "error",
                                    "error": f"Failed to save image to {temp_path}",
                                    "error_type": "FileSaveError"
                                }
                            processed_paths.append(str(temp_path))
                            saved_image_paths.append(str(temp_path))
                            print(f"[DEBUG] Saved base64 image to {temp_path} ({len(image_data)} bytes)")
                        except Exception as e:
                            return {
                                "status": "error",
                                "error": f"Failed to decode base64 image: {str(e)}",
                                "error_type": "Base64DecodeError"
                            }
                    else:
                        # It's a file path on RunPod server
                        processed_paths.append(media)
                else:
                    processed_paths.append(media)
            
            conditioning_media_paths = processed_paths
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
            # Text encoder already cached, only need space for main model + processing overhead
            required_space = 5.0 if is_2b_model else 9.0  # Model + processing overhead
        else:
            # Need to download text encoder (~19GB) + main model + overhead
            required_space = 22.0 if is_2b_model else 26.0  # 19GB text encoder + model + overhead
        
        # Always clean up temp files before processing to maximize available space
        # But protect the images we just saved
        print(f"[DEBUG] Cleaning up temp files before processing (protecting {len(saved_image_paths)} input images)...")
        cleanup_temp_files(protected_files=saved_image_paths)
        tmp_free, _ = get_disk_space("/tmp")
        
        # Aggressively clean up if space is still low
        if tmp_free < required_space:
            print(f"[DEBUG] Low disk space ({tmp_free:.2f}GB free, need {required_space}GB). Running aggressive cleanup...")
            # Run aggressive cleanup
            aggressive_cleanup()
            tmp_free, _ = get_disk_space("/tmp")
            print(f"[DEBUG] After cleanup: {tmp_free:.2f}GB free")
        
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
        # NOTE: inference.py treats output_path as a DIRECTORY, not a file
        output_dir = Path("/tmp/outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir)  # Pass directory, not file path
        
        # Get additional parameters
        frame_rate = input_data.get("frame_rate", 30)
        negative_prompt = input_data.get("negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted")
        guidance_scale = input_data.get("guidance_scale")
        stg_scale = input_data.get("stg_scale")
        rescaling_scale = input_data.get("rescaling_scale")
        
        # Create inference config
        config = InferenceConfig(
            prompt=prompt,
            conditioning_media_paths=conditioning_media_paths,
            conditioning_start_frames=conditioning_start_frames,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            seed=seed,
            pipeline_config=pipeline_config,
            output_path=output_path,
            offload_to_cpu=offload_to_cpu,
            negative_prompt=negative_prompt
        )
        
        # Store additional parameters for pipeline (will be passed via pipeline_config override if needed)
        # Note: guidance_scale, stg_scale, rescaling_scale are typically set in the YAML config
        # but can be overridden if needed in the future
        
        # Clear GPU cache before inference to free up fragmented memory
        # Also verify GPU is accessible before starting inference
        try:
            import torch
            import gc
            if torch.cuda.is_available():
                # Final check that GPU is accessible before inference
                try:
                    device = torch.cuda.current_device()
                    torch.cuda.synchronize()  # This will fail if GPU is busy
                    print(f"[DEBUG] GPU device {device} is accessible")
                except RuntimeError as e:
                    error_msg = str(e).lower()
                    if "busy" in error_msg or "unavailable" in error_msg:
                        return {
                            "status": "error",
                            "error": f"GPU device became busy/unavailable right before inference. Worker needs restart. Error: {str(e)}",
                            "error_type": "CUDABusyError",
                            "suggestion": "Restart the RunPod worker via web console (Edit endpoint → Save)"
                        }
                    raise
                
                # Clear memory
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                print(f"[DEBUG] GPU cache cleared. Free memory: {free_memory / 1024**3:.2f} GB")
        except Exception as e:
            print(f"[DEBUG] Warning: Could not clear GPU cache: {e}")
        
        # Run inference with error handling
        try:
            infer(config)
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "busy" in error_msg or "unavailable" in error_msg or "cuda error" in error_msg:
                return {
                    "status": "error",
                    "error": f"GPU device error during inference: {str(e)}",
                    "error_type": "CUDABusyError",
                    "suggestion": "The GPU is stuck in a busy state. You MUST restart the RunPod worker via web console (Edit endpoint → Save) to clear the GPU state."
                }
            raise
        
        # Aggressively clear GPU cache after inference to free memory for next job
        try:
            import torch
            import gc
            if torch.cuda.is_available():
                # Force garbage collection
                gc.collect()
                # Clear all CUDA caches
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                print(f"[DEBUG] GPU cache cleared after inference. Free memory: {free_memory / 1024**3:.2f} GB")
        except Exception as e:
            print(f"[DEBUG] Warning: Could not clear GPU cache after inference: {e}")

        # Find the actual output file (inference.py creates unique filenames)
        import glob

        # Search for .mp4 and .png files in output directory
        output_files = glob.glob(str(output_dir / "*.mp4")) + glob.glob(str(output_dir / "*.png"))
        
        # Filter out directories
        output_files = [f for f in output_files if os.path.isfile(f)]
        
        if output_files:
            # Get most recent file
            actual_output = max(output_files, key=os.path.getmtime)
            file_size = os.path.getsize(actual_output)
            print(f"[DEBUG] Video saved to {actual_output} ({file_size} bytes)")
        else:
            return {
                "status": "error",
                "error": "No output file generated",
                "error_type": "FileNotFoundError"
            }

        # Read video and encode as base64
        with open(actual_output, "rb") as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode("utf-8")

        print(f"[DEBUG] Returning base64 video ({len(video_base64)} chars)")

        # Clean up temp files after successful generation to free space for next request
        print("[DEBUG] Cleaning up temp files after video generation...")
        cleanup_temp_files()
        
        # Use RunPod's clean() function for standard cleanup
        try:
            clean(folder_list=["/tmp/input_images", "/tmp/imageio_temp"])
            print("[DEBUG] RunPod cleanup completed")
        except Exception as e:
            print(f"[DEBUG] Warning: RunPod cleanup error: {e}")

        # Return success with base64 encoded video
        return {
            "status": "success",
            "video_base64": video_base64,
            "filename": os.path.basename(actual_output),
            "file_size_bytes": file_size,
            "message": "Video generated successfully"
        }

    except Exception as e:
        import traceback
        print(f"[DEBUG] Error occurred: {type(e).__name__}: {e}")
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        
        # Cleanup on error as well (per RunPod best practices)
        try:
            cleanup_temp_files()
            clean(folder_list=["/tmp/input_images", "/tmp/imageio_temp"])
            print("[DEBUG] Cleanup completed after error")
        except Exception as cleanup_error:
            print(f"[DEBUG] Warning: Cleanup error: {cleanup_error}")
        
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


# Start RunPod serverless
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

