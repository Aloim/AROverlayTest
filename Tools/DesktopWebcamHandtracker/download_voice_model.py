#!/usr/bin/env python3
"""
Download Sherpa-ONNX voice recognition model.

This script downloads and extracts the English streaming zipformer model
used for voice-to-text functionality. Run this before first use or
to bundle the model with the installer.

Usage:
    python download_voice_model.py

Model info:
    - Name: sherpa-onnx-streaming-zipformer-en-2023-06-26
    - Size: ~65MB compressed, ~200MB extracted
    - License: Apache 2.0
"""

import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

# Model configuration
MODEL_DIR = Path(__file__).parent / "models" / "sherpa-onnx-streaming-zipformer-en-2023-06-26"
MODEL_ARCHIVE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2"
MODEL_FILES = [
    "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
    "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    "tokens.txt",
]


def progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Display download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        bar_len = 40
        filled = int(bar_len * percent / 100)
        bar = '=' * filled + '-' * (bar_len - filled)
        sys.stdout.write(f"\r[{bar}] {percent:.1f}% ({downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB)")
        sys.stdout.flush()


def check_model_exists() -> bool:
    """Check if all model files already exist."""
    if not MODEL_DIR.exists():
        return False
    for filename in MODEL_FILES:
        if not (MODEL_DIR / filename).exists():
            return False
    return True


def download_model() -> None:
    """Download and extract the voice recognition model."""
    models_base = MODEL_DIR.parent
    models_base.mkdir(parents=True, exist_ok=True)

    archive_path = models_base / "sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2"

    print(f"Downloading Sherpa-ONNX English streaming model...")
    print(f"URL: {MODEL_ARCHIVE_URL}")
    print()

    try:
        # Download archive
        urlretrieve(MODEL_ARCHIVE_URL, archive_path, reporthook=progress_hook)
        print()  # Newline after progress bar
        print(f"Downloaded: {archive_path.stat().st_size / (1024*1024):.1f} MB")

        # Extract archive
        print("Extracting model files...")
        with tarfile.open(archive_path, 'r:bz2') as tar:
            tar.extractall(path=models_base)
        print("Extraction complete!")

        # Clean up archive
        archive_path.unlink()
        print("Cleaned up archive file.")

        # Verify files
        print("\nVerifying model files:")
        for filename in MODEL_FILES:
            file_path = MODEL_DIR / filename
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024*1024)
                print(f"  ✓ {filename} ({size_mb:.1f} MB)")
            else:
                print(f"  ✗ {filename} - MISSING!")
                raise RuntimeError(f"Model file missing: {filename}")

        print("\n✓ Model download complete!")
        print(f"  Location: {MODEL_DIR}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        # Clean up partial downloads
        if archive_path.exists():
            archive_path.unlink()
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("Sherpa-ONNX Voice Recognition Model Downloader")
    print("=" * 60)
    print()

    if check_model_exists():
        print("✓ Model already downloaded!")
        print(f"  Location: {MODEL_DIR}")
        print("\nTo re-download, delete the model directory first:")
        print(f"  rmdir /s /q \"{MODEL_DIR}\"")
        return

    download_model()


if __name__ == "__main__":
    main()
