"""
Voice recognizer for DesktopWebcamHandtracker.

Uses Sherpa-ONNX for streaming speech recognition with PyAudio for microphone capture.
Supports wake word detection for voice-command activation mode.
"""

import os
import shutil
import tarfile
import threading
import time
from pathlib import Path
from typing import Optional, Callable
from urllib.request import urlretrieve

import numpy as np
try:
    import pyaudio
except ImportError as e:
    raise ImportError(
        "PyAudio is required. Install with: pip install pyaudio"
    ) from e

try:
    import sherpa_onnx
except ImportError as e:
    raise ImportError(
        "Sherpa-ONNX is required. Install with: pip install sherpa-onnx"
    ) from e

from logger import get_logger

logger = get_logger("VoiceRecognizer")

# Model configuration
MODEL_DIR = Path(__file__).parent / "models" / "sherpa-onnx-streaming-zipformer-en-2023-06-26"
MODEL_ARCHIVE_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2"
MODEL_FILES = {
    "encoder": "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    "decoder": "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
    "joiner": "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    "tokens": "tokens.txt",
}

# Audio configuration
SAMPLE_RATE = 16000  # Sherpa-ONNX requires 16kHz
CHUNK_SIZE = 3200  # 0.2 seconds at 16kHz
SILENCE_THRESHOLD = 500.0  # RMS threshold for silence detection
SILENCE_DURATION_SECONDS = 2.0  # Stop after 2 seconds of silence


class VoiceRecognizer:
    """
    Streaming speech recognition using Sherpa-ONNX.

    Captures audio from microphone and performs real-time transcription.
    Supports automatic silence detection and wake word activation.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device_index: int = -1,
        silence_timeout: int = 5
    ):
        """
        Initialize voice recognizer.

        Args:
            model_path: Path to Sherpa-ONNX model directory (None for auto-download).
            device_index: PyAudio device index (-1 for default).
            silence_timeout: Seconds of silence before auto-stop (default: 5).
        """
        self.device_index = device_index
        self.silence_timeout = silence_timeout

        # Audio capture
        self._audio = pyaudio.PyAudio()
        self._stream: Optional[pyaudio.Stream] = None
        self._stream_obj = None  # Sherpa-ONNX stream object (for cleanup)
        self._recognizing = False
        self._recognition_thread: Optional[threading.Thread] = None

        # Recognition results
        self._partial_result = ""
        self._final_result = ""
        self._result_lock = threading.Lock()

        # Silence detection
        self._last_speech_time = time.time()
        self._silence_threshold = SILENCE_THRESHOLD

        # Initialize model
        if model_path is None:
            model_path = str(MODEL_DIR)

        self._ensure_model_exists(model_path)
        self._recognizer = self._create_recognizer(model_path)

        logger.info(f"VoiceRecognizer initialized (device={device_index}, timeout={silence_timeout}s)")

    def _ensure_model_exists(self, model_path: str) -> None:
        """
        Ensure Sherpa-ONNX model is downloaded.

        Downloads the model archive and extracts it if necessary.

        Args:
            model_path: Path to model directory.
        """
        model_dir = Path(model_path)
        models_base = model_dir.parent  # models/ directory
        models_base.mkdir(parents=True, exist_ok=True)

        # Check if all required files exist
        missing_files = []
        for file_key, filename in MODEL_FILES.items():
            file_path = model_dir / filename
            if not file_path.exists():
                missing_files.append((file_key, filename))

        if not missing_files:
            logger.debug("Sherpa-ONNX model already downloaded")
            return

        # Download and extract model archive
        logger.info(f"Downloading Sherpa-ONNX model ({len(missing_files)} files missing)...")
        archive_path = models_base / "sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2"

        try:
            # Download archive
            logger.info(f"Downloading model archive (~65MB)...")
            logger.info(f"URL: {MODEL_ARCHIVE_URL}")

            def _progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, downloaded * 100 / total_size)
                    if block_num % 100 == 0:  # Log every 100 blocks
                        logger.info(f"Download progress: {percent:.1f}%")

            urlretrieve(MODEL_ARCHIVE_URL, archive_path, reporthook=_progress_hook)
            logger.info(f"Downloaded archive ({archive_path.stat().st_size / (1024*1024):.1f} MB)")

            # Extract archive
            logger.info("Extracting model files...")
            with tarfile.open(archive_path, 'r:bz2') as tar:
                tar.extractall(path=models_base)
            logger.info("Model extraction complete")

            # Clean up archive
            archive_path.unlink()
            logger.info("Cleaned up archive file")

        except Exception as e:
            logger.error(f"Model download/extraction failed: {e}")
            # Clean up partial downloads
            if archive_path.exists():
                archive_path.unlink()
            raise RuntimeError(f"Model download failed: {e}")

        # Verify all files exist after extraction
        for file_key, filename in MODEL_FILES.items():
            file_path = model_dir / filename
            if not file_path.exists():
                raise RuntimeError(f"Model file missing after extraction: {filename}")

        logger.info("Sherpa-ONNX model ready")

    def _create_recognizer(self, model_path: str) -> sherpa_onnx.OnlineRecognizer:
        """
        Create Sherpa-ONNX recognizer instance.

        Args:
            model_path: Path to model directory.

        Returns:
            Configured OnlineRecognizer instance.
        """
        model_dir = Path(model_path)

        # Create recognizer using the new factory method API (sherpa-onnx v1.10+)
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(model_dir / MODEL_FILES["tokens"]),
            encoder=str(model_dir / MODEL_FILES["encoder"]),
            decoder=str(model_dir / MODEL_FILES["decoder"]),
            joiner=str(model_dir / MODEL_FILES["joiner"]),
            num_threads=2,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            max_active_paths=4,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,  # Silence duration for endpoint
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=20.0
        )

        logger.debug("Sherpa-ONNX recognizer created")
        return recognizer

    def start_recognition(self) -> None:
        """
        Begin streaming speech recognition.

        Opens microphone stream and starts recognition thread.
        """
        if self._recognizing:
            logger.warning("Recognition already in progress")
            return

        self._recognizing = True
        self._partial_result = ""
        self._final_result = ""
        self._last_speech_time = time.time()

        # Open audio stream
        try:
            self._stream = self._audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=self.device_index if self.device_index >= 0 else None,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=None
            )
            logger.info("Microphone stream opened")
        except Exception as e:
            logger.error(f"Failed to open microphone: {e}")
            self._recognizing = False
            raise RuntimeError(f"Microphone error: {e}")

        # Start recognition thread
        self._recognition_thread = threading.Thread(
            target=self._recognition_loop,
            daemon=True
        )
        self._recognition_thread.start()
        logger.info("Voice recognition started")

    def stop_recognition(self) -> str:
        """
        Stop speech recognition and return final transcription.

        Returns:
            Final transcribed text.
        """
        was_recognizing = self._recognizing
        self._recognizing = False  # Signal thread to stop

        if not was_recognizing:
            logger.warning("Recognition not in progress")
            # Still try cleanup in case of partial state (LEAK FIX P3)

        # Always try to join the thread, regardless of previous state (LEAK FIX P3)
        if self._recognition_thread and self._recognition_thread.is_alive():
            self._recognition_thread.join(timeout=3.0)
            if self._recognition_thread.is_alive():
                logger.warning("Recognition thread did not stop in time")
        self._recognition_thread = None

        # Always clean up audio stream (LEAK FIX P3)
        # Note: _recognition_loop's finally block may have already cleaned up,
        # but _cleanup_audio_stream() is safe to call multiple times
        self._cleanup_audio_stream()
        logger.info("Microphone stream closed")

        # Clean up Sherpa stream reference if still present
        self._stream_obj = None

        # Get final result
        with self._result_lock:
            final_text = self._final_result if self._final_result else self._partial_result
            logger.info(f"Voice recognition stopped, final text: '{final_text}'")
            return final_text

    def get_partial_result(self) -> str:
        """
        Get current partial transcription result.

        Returns:
            Current partial transcription text.
        """
        with self._result_lock:
            return self._partial_result

    def is_listening(self) -> bool:
        """
        Check if recognition is currently active.

        Returns:
            True if actively recognizing speech, False otherwise.
        """
        return self._recognizing

    def set_silence_timeout(self, seconds: int) -> None:
        """
        Set silence timeout for automatic stopping.

        Args:
            seconds: Seconds of silence before auto-stop.
        """
        self.silence_timeout = seconds
        logger.debug(f"Silence timeout set to {seconds}s")

    def _cleanup_audio_stream(self) -> None:
        """Clean up PyAudio stream resources safely."""
        if self._stream:
            try:
                self._stream.stop_stream()
            except Exception as e:
                logger.debug(f"Error stopping stream: {e}")
            try:
                self._stream.close()
            except Exception as e:
                logger.debug(f"Error closing stream: {e}")
            self._stream = None
            logger.debug("Audio stream cleaned up")

    def _normalize_text(self, text: str) -> str:
        """
        Normalize transcription text to proper sentence case.

        Sherpa-ONNX often outputs uppercase text. This converts to natural case.

        Args:
            text: Raw transcription text.

        Returns:
            Text in proper sentence case.
        """
        if not text:
            return text

        # Strip whitespace
        text = text.strip()

        # If all uppercase or all lowercase, convert to sentence case
        if text.isupper() or text.islower():
            # Simple sentence case: capitalize first letter, lowercase rest
            # But preserve proper casing for common abbreviations
            words = text.lower().split()
            if words:
                # Capitalize first word
                words[0] = words[0].capitalize()
                # Capitalize 'I' when standalone
                words = ['I' if w == 'i' else w for w in words]
            return ' '.join(words)

        return text

    def _recognition_loop(self) -> None:
        """Main recognition loop running in separate thread."""
        # Store stream as instance variable for proper cleanup (LEAK FIX P1)
        self._stream_obj = self._recognizer.create_stream()

        try:
            while self._recognizing:
                # Read audio chunk
                if not self._stream:
                    break

                try:
                    audio_data = self._stream.read(CHUNK_SIZE, exception_on_overflow=False)
                except Exception as e:
                    logger.error(f"Audio read error: {e}")
                    break

                # Convert to numpy array
                samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Check for speech activity (simple RMS-based)
                rms = np.sqrt(np.mean(samples ** 2)) * 32768.0

                if rms > self._silence_threshold:
                    self._last_speech_time = time.time()

                # Feed samples to recognizer
                self._stream_obj.accept_waveform(SAMPLE_RATE, samples)

                # Decode if ready
                if self._recognizer.is_ready(self._stream_obj):
                    self._recognizer.decode_stream(self._stream_obj)

                # Get partial result
                result = self._recognizer.get_result(self._stream_obj)
                if result:
                    normalized = self._normalize_text(result)
                    with self._result_lock:
                        self._partial_result = normalized
                        logger.debug(f"Partial: '{normalized}'")

                # Check for endpoint (end of utterance)
                if self._recognizer.is_endpoint(self._stream_obj):
                    result = self._recognizer.get_result(self._stream_obj)
                    if result:
                        normalized = self._normalize_text(result)
                        with self._result_lock:
                            self._final_result = normalized
                            logger.info(f"Final (endpoint): '{normalized}'")
                    # Reset for next utterance
                    self._recognizer.reset(self._stream_obj)

                # Check silence timeout
                silence_duration = time.time() - self._last_speech_time
                if silence_duration >= self.silence_timeout:
                    logger.info(f"Silence timeout ({self.silence_timeout}s) reached")
                    # Get final result before stopping
                    result = self._recognizer.get_result(self._stream_obj)
                    if result:
                        normalized = self._normalize_text(result)
                        with self._result_lock:
                            self._final_result = normalized
                    break

        except Exception as e:
            logger.error(f"Recognition loop error: {e}")
        finally:
            # Clean up Sherpa stream to prevent memory leak (LEAK FIX P1)
            self._stream_obj = None
            # Clean up audio stream (LEAK FIX P2)
            self._cleanup_audio_stream()
            logger.debug("Recognition loop ended with cleanup")

    def is_recognizing(self) -> bool:
        """
        Check if recognition is currently active.

        Returns:
            True if recognizing, False otherwise.
        """
        return self._recognizing

    def list_microphones(self) -> list[tuple[int, str]]:
        """
        List available microphone devices.

        Returns:
            List of (device_index, device_name) tuples.
        """
        devices = []
        for i in range(self._audio.get_device_count()):
            info = self._audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append((i, info['name']))
        return devices

    def cleanup(self) -> None:
        """Clean up audio resources."""
        if self._recognizing:
            self.stop_recognition()

        if self._audio:
            self._audio.terminate()
            logger.debug("PyAudio terminated")


class WakeWordDetector:
    """
    Simple wake word detection using keyword spotting.

    Uses Sherpa-ONNX's keyword spotting capability for voice command activation.
    """

    def __init__(self, keyword: str = "hey computer", device_index: int = -1):
        """
        Initialize wake word detector.

        Args:
            keyword: Wake word phrase to detect.
            device_index: PyAudio device index (-1 for default).
        """
        self.keyword = keyword.lower()
        self.device_index = device_index

        # Use the main recognizer for keyword detection
        self._recognizer_instance = VoiceRecognizer(device_index=device_index)
        self._detecting = False
        self._callback: Optional[Callable[[], None]] = None

        logger.info(f"WakeWordDetector initialized (keyword='{keyword}')")

    def start_detection(self, callback: Callable[[], None]) -> None:
        """
        Start wake word detection.

        Args:
            callback: Function to call when wake word is detected.
        """
        self._callback = callback
        self._detecting = True

        # Start continuous recognition
        self._recognizer_instance.start_recognition()

        # Monitor in background thread
        detection_thread = threading.Thread(
            target=self._detection_loop,
            daemon=True
        )
        detection_thread.start()

        logger.info(f"Wake word detection started ('{self.keyword}')")

    def stop_detection(self) -> None:
        """Stop wake word detection."""
        self._detecting = False
        self._recognizer_instance.stop_recognition()
        logger.info("Wake word detection stopped")

    def _detection_loop(self) -> None:
        """Monitor recognition results for wake word."""
        while self._detecting:
            partial = self._recognizer_instance.get_partial_result().lower()

            if self.keyword in partial:
                logger.info(f"Wake word detected: '{self.keyword}'")
                if self._callback:
                    self._callback()
                # Brief pause after detection
                time.sleep(1.0)

            time.sleep(0.1)

    def cleanup(self) -> None:
        """Clean up detector resources."""
        self.stop_detection()
        self._recognizer_instance.cleanup()
