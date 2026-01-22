"""
MediaPipe model file manager for Tasks API.

Downloads and caches MediaPipe Tasks model files required for Python 3.13+.
"""

import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

# Model configuration
HAND_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
HAND_LANDMARKER_FILENAME = "hand_landmarker.task"
HAND_LANDMARKER_SIZE_MB = 9.1  # Approximate size in MB

# Download settings
DOWNLOAD_TIMEOUT = 120  # seconds
DOWNLOAD_CHUNK_SIZE = 8192  # bytes
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def get_model_cache_dir() -> Path:
    """
    Get the model cache directory.

    Returns:
        Path to model cache directory (creates if needed).
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    else:
        base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))

    cache_dir = Path(base) / "AROverlay" / "mediapipe_models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def ensure_hand_landmarker_model() -> str:
    """
    Ensure the hand landmarker model is available.

    Downloads the model if not present in cache.

    Returns:
        Path to the model file.

    Raises:
        RuntimeError: If download fails after retries.
    """
    cache_dir = get_model_cache_dir()
    model_path = cache_dir / HAND_LANDMARKER_FILENAME

    if model_path.exists():
        print(f"[ModelManager] Using cached model: {model_path}")
        return str(model_path)

    print(f"[ModelManager] Downloading hand landmarker model (~{HAND_LANDMARKER_SIZE_MB} MB)...")
    print(f"[ModelManager] URL: {HAND_LANDMARKER_URL}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _download_model(HAND_LANDMARKER_URL, model_path)
            print(f"[ModelManager] Model downloaded successfully: {model_path}")
            return str(model_path)
        except Exception as e:
            print(f"[ModelManager] Download attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                print(f"[ModelManager] Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY * attempt)  # Exponential backoff
            else:
                raise RuntimeError(
                    f"Failed to download MediaPipe model after {MAX_RETRIES} attempts. "
                    f"Please check your internet connection and try again."
                ) from e

    # Should not reach here
    raise RuntimeError("Model download failed")


def _download_model(url: str, dest_path: Path) -> None:
    """
    Download a model file with progress reporting.

    Args:
        url: URL to download from.
        dest_path: Destination file path.
    """
    # Create temp file for download
    temp_path = dest_path.with_suffix(".tmp")

    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "AROverlay/1.0"}
        )

        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0

            with open(temp_path, "wb") as f:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break

                    f.write(chunk)
                    downloaded += len(chunk)

                    # Progress reporting
                    if total_size > 0:
                        progress = downloaded / total_size * 100
                        print(f"\r[ModelManager] Progress: {progress:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end="")

        print()  # Newline after progress

        # Move temp file to final location
        temp_path.rename(dest_path)

    except Exception:
        # Clean up temp file on failure
        if temp_path.exists():
            temp_path.unlink()
        raise


def verify_model_hash(model_path: Path, expected_hash: str) -> bool:
    """
    Verify a model file's SHA256 hash.

    Args:
        model_path: Path to the model file.
        expected_hash: Expected SHA256 hash (lowercase hex).

    Returns:
        True if hash matches, False otherwise.
    """
    if not model_path.exists():
        return False

    sha256 = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(DOWNLOAD_CHUNK_SIZE), b""):
            sha256.update(chunk)

    actual_hash = sha256.hexdigest()
    return actual_hash.lower() == expected_hash.lower()


def get_model_info() -> dict:
    """
    Get information about the cached model.

    Returns:
        Dictionary with model status information.
    """
    cache_dir = get_model_cache_dir()
    model_path = cache_dir / HAND_LANDMARKER_FILENAME

    info = {
        "cache_dir": str(cache_dir),
        "model_path": str(model_path),
        "model_exists": model_path.exists(),
        "model_size_mb": 0.0,
    }

    if model_path.exists():
        info["model_size_mb"] = model_path.stat().st_size / 1024 / 1024

    return info


if __name__ == "__main__":
    # Test model download
    print("MediaPipe Model Manager Test")
    print("=" * 40)

    info = get_model_info()
    print(f"Cache directory: {info['cache_dir']}")
    print(f"Model exists: {info['model_exists']}")

    if info['model_exists']:
        print(f"Model size: {info['model_size_mb']:.2f} MB")
    else:
        print("Model not cached, downloading...")
        try:
            model_path = ensure_hand_landmarker_model()
            print(f"Model ready: {model_path}")
        except Exception as e:
            print(f"Download failed: {e}")
