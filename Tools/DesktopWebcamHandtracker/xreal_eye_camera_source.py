"""
XREAL Eye Camera Source for DesktopWebcamHandtracker.

Provides a camera source that streams video from XREAL One Pro Eye camera
via TCP connection. Compatible with CameraManager interface for drop-in
replacement.

Protocol:
    - TCP connection to 169.254.2.1:52997
    - Packet size: 193,862 bytes (320-byte header + 193,536 image + 6-byte footer)
    - Image format: 512x378 @ 4-bit greyscale (high nibble encoding)
    - Decode: ((pixel >> 4) & 0x0F) * 17 -> 0-255 range

Phase 2 Preprocessing Pipeline:
    1. Sigmoid LUT (Phase 2.1) - Better bit-depth mapping for mid-tones
    2. CLAHE - Contrast enhancement
    3. Ordered Dithering (Phase 2.2) - Break up banding artifacts
    4. Guided Filter OR Gaussian (Phase 2.2) - Edge-preserving smoothing
    5. [Optional] FFT Denoise (Phase 2.4) - Remove periodic artifacts
"""

import socket
import time
from typing import Optional

import cv2
import numpy as np

# Optional opencv-contrib import for guided filter
try:
    import cv2.ximgproc as ximgproc
    HAS_XIMGPROC = True
except (ImportError, AttributeError):
    ximgproc = None
    HAS_XIMGPROC = False

from logger import get_logger
from config import (
    XREAL_EYE_HOST,
    XREAL_EYE_VIDEO_PORT,
    XREAL_EYE_PACKET_SIZE,
    XREAL_EYE_HEADER_OFFSET,
    XREAL_EYE_WIDTH,
    XREAL_EYE_HEIGHT,
    XREAL_EYE_CLAHE_ENABLED,
    XREAL_EYE_CLAHE_CLIP_LIMIT,
    XREAL_EYE_CLAHE_TILE_SIZE,
    XREAL_EYE_GAUSSIAN_KERNEL,
    XREAL_EYE_SIGMOID_LUT_ENABLED,
    XREAL_EYE_SIGMOID_VIBRANCE,
    XREAL_EYE_DITHER_ENABLED,
    XREAL_EYE_GUIDED_FILTER_ENABLED,
    XREAL_EYE_GUIDED_FILTER_RADIUS,
    XREAL_EYE_GUIDED_FILTER_EPS,
    XREAL_EYE_FFT_DENOISE_ENABLED,
)

logger = get_logger("XREALEyeCameraSource")

# Bayer 4x4 ordered dithering matrix (Phase 2.2)
# Normalized to 0.0-1.0 for threshold comparison
BAYER_4x4 = np.array([
    [0,  8,  2, 10],
    [12, 4, 14,  6],
    [3, 11,  1,  9],
    [15, 7, 13,  5]
], dtype=np.float32) / 16.0


class CameraError(Exception):
    """Raised when camera operations fail."""
    pass


class XREALEyeCameraSource:
    """
    Camera source for XREAL Eye glasses via TCP streaming.

    Connects to XREAL One Pro Eye camera on 169.254.2.1:52997 and decodes
    the proprietary video stream to provide OpenCV-compatible BGR frames.

    This class is API-compatible with CameraManager for duck-typed usage
    in the hand tracking pipeline.

    Phase 2 Enhancements:
        - Sigmoid LUT for improved 4-bit to 8-bit mapping
        - CLAHE contrast enhancement
        - Ordered dithering to reduce banding
        - Edge-preserving guided filter
        - Optional FFT denoising

    Attributes:
        width: Target frame width (default: 512).
        height: Target frame height (default: 378).
        fps: Target frames per second (default: 30).
        flip_horizontal: Apply horizontal flip to frames (default: True).
    """

    def __init__(
        self,
        width: int = XREAL_EYE_WIDTH,
        height: int = XREAL_EYE_HEIGHT,
        fps: int = 30,
        flip_horizontal: bool = True,
        clahe_enabled: bool = XREAL_EYE_CLAHE_ENABLED,
        clahe_clip_limit: float = XREAL_EYE_CLAHE_CLIP_LIMIT,
        clahe_tile_size: int = XREAL_EYE_CLAHE_TILE_SIZE
    ):
        """
        Initialize XREAL Eye camera source.

        Args:
            width: Desired frame width (must match XREAL Eye: 512).
            height: Desired frame height (must match XREAL Eye: 378).
            fps: Target FPS (informational, actual rate is ~30 FPS).
            flip_horizontal: Mirror frames horizontally for user-facing view.
            clahe_enabled: Apply CLAHE contrast enhancement (Phase 2).
            clahe_clip_limit: CLAHE clip limit (1.0-4.0, default 3.0).
            clahe_tile_size: CLAHE tile grid size (default 4x4).
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_horizontal = flip_horizontal

        self._sock: Optional[socket.socket] = None
        self._is_open = False
        self._frame_count = 0
        self._last_frame_time = 0.0
        self._reconnect_attempts = 0
        self._max_reconnect_delay = 4.0  # Max backoff: 4 seconds

        # Packet assembly buffer
        self._buffer = b''
        self._last_frame: Optional[np.ndarray] = None

        # Phase 2.1: Sigmoid LUT for better bit-depth mapping
        self._sigmoid_lut_enabled = XREAL_EYE_SIGMOID_LUT_ENABLED
        if self._sigmoid_lut_enabled:
            self._sigmoid_lut = self._create_sigmoid_lut(XREAL_EYE_SIGMOID_VIBRANCE)
            logger.info(f"Sigmoid LUT enabled: vibrance={XREAL_EYE_SIGMOID_VIBRANCE}")

        # CLAHE (Contrast Limited Adaptive Histogram Equalization) - Phase 2
        self._clahe_enabled = clahe_enabled
        if self._clahe_enabled:
            self._clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit,
                tileGridSize=(clahe_tile_size, clahe_tile_size)
            )
            logger.info(f"CLAHE enabled: clipLimit={clahe_clip_limit}, tiles={clahe_tile_size}x{clahe_tile_size}")
        else:
            self._clahe = None

        # Phase 2.2: Ordered dithering with pre-computed Bayer matrix
        self._dither_enabled = XREAL_EYE_DITHER_ENABLED
        if self._dither_enabled:
            # Pre-compute tiled Bayer matrix to avoid per-frame allocation
            h, w = XREAL_EYE_HEIGHT, XREAL_EYE_WIDTH
            self._bayer_tiled = np.tile(BAYER_4x4, (h // 4 + 1, w // 4 + 1))[:h, :w]
            logger.info("Ordered dithering (Bayer 4x4) enabled with pre-computed matrix")

        # Phase 2.2: Guided filter (edge-preserving)
        self._guided_filter_enabled = XREAL_EYE_GUIDED_FILTER_ENABLED
        self._guided_radius = XREAL_EYE_GUIDED_FILTER_RADIUS
        self._guided_eps = XREAL_EYE_GUIDED_FILTER_EPS
        self._ximgproc_available = HAS_XIMGPROC
        if self._guided_filter_enabled:
            if self._ximgproc_available:
                logger.info(f"Guided filter enabled: radius={self._guided_radius}, eps={self._guided_eps}")
            else:
                logger.warning("opencv-contrib-python not installed, falling back to Gaussian blur")

        # Phase 2.4: FFT denoising (optional)
        self._fft_denoise_enabled = XREAL_EYE_FFT_DENOISE_ENABLED
        self._scipy_available = False
        if self._fft_denoise_enabled:
            try:
                from scipy import fftpack
                self._scipy_available = True
                logger.info("FFT denoising enabled (experimental)")
            except ImportError:
                logger.warning("scipy not installed, FFT denoising disabled")
                self._fft_denoise_enabled = False

        # Gaussian kernel size (used as fallback or when guided filter disabled)
        self._gaussian_kernel = XREAL_EYE_GAUSSIAN_KERNEL

        # Validate dimensions - XREAL Eye has fixed resolution
        if width != XREAL_EYE_WIDTH or height != XREAL_EYE_HEIGHT:
            raise ValueError(
                f"XREAL Eye only supports {XREAL_EYE_WIDTH}x{XREAL_EYE_HEIGHT}, "
                f"requested {width}x{height} is not supported. "
                f"Use the default resolution or omit width/height parameters."
            )

    @property
    def is_open(self) -> bool:
        """Check if camera connection is currently open."""
        return self._is_open and self._sock is not None

    @property
    def actual_width(self) -> int:
        """Get actual frame width (always 512 for XREAL Eye)."""
        return XREAL_EYE_WIDTH

    @property
    def actual_height(self) -> int:
        """Get actual frame height (always 378 for XREAL Eye)."""
        return XREAL_EYE_HEIGHT

    @property
    def actual_fps(self) -> float:
        """Get actual frame rate (estimated from capture timing)."""
        # Could be calculated from frame timestamps if needed
        return float(self.fps)

    def open(self) -> None:
        """
        Open TCP connection to XREAL Eye camera.

        Establishes connection to 169.254.2.1:52997 with 5-second timeout.

        Raises:
            CameraError: If connection fails after timeout.
        """
        if self._is_open:
            logger.warning("Camera already open, closing first")
            self.close()

        logger.info(f"Connecting to XREAL Eye at {XREAL_EYE_HOST}:{XREAL_EYE_VIDEO_PORT}...")

        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(5.0)  # 5-second connect timeout
            self._sock.connect((XREAL_EYE_HOST, XREAL_EYE_VIDEO_PORT))
            self._sock.settimeout(1.0)  # 1-second read timeout during operation

            logger.info(
                f"XREAL Eye connected: {XREAL_EYE_WIDTH}x{XREAL_EYE_HEIGHT} @ ~{self.fps} FPS"
            )

            self._is_open = True
            self._frame_count = 0
            self._buffer = b''
            self._last_frame = None
            self._last_frame_time = time.perf_counter()
            self._reconnect_attempts = 0

        except socket.timeout:
            raise CameraError(
                f"Connection timeout: XREAL Eye not responding on {XREAL_EYE_HOST}:{XREAL_EYE_VIDEO_PORT}"
            )
        except OSError as e:
            raise CameraError(f"Failed to connect to XREAL Eye: {e}")

    def close(self) -> None:
        """Close TCP connection and release resources."""
        if self._sock:
            logger.info("Closing XREAL Eye connection")
            try:
                self._sock.close()
            except Exception as e:
                logger.warning(f"Error during socket close: {e}")
            self._sock = None

        self._is_open = False
        self._buffer = b''
        self._last_frame = None

    def _decode_packet(self, packet: bytes) -> Optional[np.ndarray]:
        """
        Decode XREAL Eye video packet to greyscale image.

        Args:
            packet: Raw packet data (193,862 bytes).

        Returns:
            Greyscale image as numpy array (512x378), or None if invalid.
        """
        if len(packet) < XREAL_EYE_HEADER_OFFSET + (XREAL_EYE_WIDTH * XREAL_EYE_HEIGHT):
            logger.warning(f"Packet too small: {len(packet)} bytes")
            return None

        # Extract image data (skip 320-byte header)
        image_data = packet[XREAL_EYE_HEADER_OFFSET:XREAL_EYE_HEADER_OFFSET + (XREAL_EYE_WIDTH * XREAL_EYE_HEIGHT)]

        # Decode 4-bit greyscale: high nibble = pixel value (0-15), scale to 0-255
        pixels = np.frombuffer(image_data, dtype=np.uint8)
        pixels = ((pixels >> 4) & 0x0F) * 17  # Scale 0-15 -> 0-255

        # Reshape to image dimensions
        try:
            grey = pixels.reshape((XREAL_EYE_HEIGHT, XREAL_EYE_WIDTH))
        except ValueError as e:
            logger.error(f"Reshape failed: {e}")
            return None

        return grey

    def _create_sigmoid_lut(self, vibrance: float = 1.5) -> np.ndarray:
        """
        Create 256-element LUT with sigmoid contrast curve.

        Stretches middle gray levels (where hands appear) for better detection.
        Input 4-bit values (0-15 scaled to 0-255) get remapped to emphasize
        mid-tones while compressing extreme values.

        Args:
            vibrance: S-curve steepness (1.0-2.0, higher = more contrast)

        Returns:
            256-element uint8 lookup table
        """
        x = np.arange(256, dtype=np.float32) / 255.0
        # Sigmoid centered at 0.5, scaled by vibrance
        sigmoid = 1.0 / (1.0 + np.exp(-vibrance * 5.0 * (x - 0.5)))
        return (sigmoid * 255).astype(np.uint8)

    def _apply_sigmoid_lut(self, frame: np.ndarray) -> np.ndarray:
        """Apply sigmoid LUT for enhanced mid-tone contrast."""
        return cv2.LUT(frame, self._sigmoid_lut)

    def _apply_ordered_dither(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply Bayer 4x4 ordered dithering to break up 4-bit banding.

        Adds structured noise pattern that, when blurred, creates smoother
        gradients than the original 16-level staircase.

        Uses pre-computed Bayer matrix from __init__ for efficiency.
        """
        # Add dithering offset using pre-computed matrix: (threshold - 0.5) * step_size
        # For 4-bit to 8-bit: step_size = 255/15 = 17 (not 16, since 0-15 has 16 values)
        dithered = frame.astype(np.float32) + (self._bayer_tiled - 0.5) * 17.0
        return np.clip(dithered, 0, 255).astype(np.uint8)

    def _apply_guided_filter(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply edge-preserving guided filter.

        Better than Gaussian blur for hand tracking - preserves finger edges
        while smoothing flat regions.
        """
        return ximgproc.guidedFilter(
            guide=frame,
            src=frame,
            radius=self._guided_radius,
            eps=self._guided_eps
        )

    def _apply_fft_denoise(self, frame: np.ndarray) -> np.ndarray:
        """
        Remove periodic banding artifacts via FFT notch filter.

        Identifies and suppresses frequency spikes caused by 4-bit quantization.
        Expensive (~10-20ms) - only use if other methods insufficient.
        """
        from scipy import fftpack

        # Forward FFT
        f_transform = fftpack.fft2(frame.astype(np.float32))
        f_shift = fftpack.fftshift(f_transform)

        # Create notch filter to suppress mid-frequencies (banding)
        h, w = frame.shape
        center_h, center_w = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)

        # Suppress frequencies between 10-20 (typical banding range)
        notch = np.where((dist > 10) & (dist < 20), 0.3, 1.0)

        # Apply filter and inverse FFT
        f_filtered = f_shift * notch
        f_shift_back = fftpack.ifftshift(f_filtered)
        img_denoised = fftpack.ifft2(f_shift_back).real

        return np.clip(img_denoised, 0, 255).astype(np.uint8)

    def _receive_frame_internal(self) -> Optional[np.ndarray]:
        """
        Receive and decode a single frame from TCP stream.

        Returns:
            Greyscale frame (512x378) or None if read failed.
        """
        if not self._sock:
            return None

        try:
            # Read data in chunks and assemble packets
            while len(self._buffer) < XREAL_EYE_PACKET_SIZE:
                chunk = self._sock.recv(65536)
                if not chunk:
                    logger.warning("Socket closed by remote")
                    return None
                self._buffer += chunk

            # Extract complete packet
            packet = self._buffer[:XREAL_EYE_PACKET_SIZE]
            self._buffer = self._buffer[XREAL_EYE_PACKET_SIZE:]

            # Decode frame
            frame = self._decode_packet(packet)
            return frame

        except socket.timeout:
            # Timeout is normal during low frame rates, not an error
            return None
        except OSError as e:
            logger.error(f"Socket receive error: {e}")
            self._is_open = False
            return None

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a single frame from XREAL Eye camera.

        Phase 2 Pipeline:
        1. Receive greyscale frame from TCP
        2. Apply Sigmoid LUT (Phase 2.1) - better bit-depth mapping
        3. Apply CLAHE - contrast enhancement
        4. Apply Ordered Dithering (Phase 2.2) - break up banding
        5. Apply Guided Filter OR Gaussian (Phase 2.2) - edge-preserving smooth
        6. [Optional] FFT Denoise (Phase 2.4) - remove periodic artifacts
        7. Convert to BGR
        8. Horizontal flip

        Returns:
            BGR image as numpy array (512x378x3), or None if read failed.
            Returns last valid frame on temporary read failure.

        Raises:
            CameraError: If camera is not open.
        """
        if not self._is_open or self._sock is None:
            raise CameraError("XREAL Eye camera is not open")

        # Receive greyscale frame
        grey_frame = self._receive_frame_internal()

        if grey_frame is None:
            # Return last valid frame if available (handles transient errors)
            if self._last_frame is not None:
                logger.debug("Using last valid frame (read timeout)")
                return self._last_frame.copy()
            return None

        # Phase 2.1: Apply Sigmoid LUT for better mid-tone contrast
        if self._sigmoid_lut_enabled:
            grey_frame = self._apply_sigmoid_lut(grey_frame)

        # Apply CLAHE for contrast enhancement
        if self._clahe_enabled and self._clahe is not None:
            grey_frame = self._clahe.apply(grey_frame)

        # Phase 2.2: Apply ordered dithering to break up banding
        if self._dither_enabled:
            grey_frame = self._apply_ordered_dither(grey_frame)

        # Phase 2.2: Apply edge-preserving filter
        if self._guided_filter_enabled and self._ximgproc_available:
            grey_frame = self._apply_guided_filter(grey_frame)
        else:
            # Fallback to Gaussian blur
            kernel = (self._gaussian_kernel, self._gaussian_kernel)
            grey_frame = cv2.GaussianBlur(grey_frame, kernel, 0)

        # Phase 2.4: Optional FFT denoising
        if self._fft_denoise_enabled and self._scipy_available:
            grey_frame = self._apply_fft_denoise(grey_frame)

        # Convert greyscale to BGR (3-channel)
        bgr_frame = cv2.cvtColor(grey_frame, cv2.COLOR_GRAY2BGR)

        # Apply horizontal flip if enabled
        if self.flip_horizontal:
            bgr_frame = cv2.flip(bgr_frame, 1)

        self._frame_count += 1
        self._last_frame_time = time.perf_counter()
        self._last_frame = bgr_frame

        return bgr_frame

    def read_frame_rgb(self) -> Optional[np.ndarray]:
        """
        Read a single frame and convert to RGB.

        Returns:
            RGB image as numpy array (512x378x3), or None if read failed.
        """
        frame = self.read_frame()
        if frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def reconnect_with_backoff(self) -> bool:
        """
        Attempt to reconnect with exponential backoff.

        Returns:
            True if reconnection succeeded, False otherwise.
        """
        if self._is_open:
            return True

        # Calculate backoff delay: 1s, 2s, 4s, 4s, ...
        delay = min(2 ** self._reconnect_attempts, self._max_reconnect_delay)
        logger.info(f"Reconnect attempt {self._reconnect_attempts + 1} after {delay}s delay...")
        time.sleep(delay)

        try:
            self.open()
            logger.info("Reconnection successful")
            return True
        except CameraError as e:
            logger.warning(f"Reconnection failed: {e}")
            self._reconnect_attempts += 1
            return False

    def get_frame_count(self) -> int:
        """Get total frames captured since opening."""
        return self._frame_count

    def __enter__(self) -> "XREALEyeCameraSource":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


def test_xreal_eye_capture():
    """
    Test function to verify XREAL Eye camera connection and display live feed.

    Press 'q' to quit, 's' to save frame.
    """
    print("=" * 60)
    print("  XREAL Eye Camera Source Test")
    print(f"  Connecting to {XREAL_EYE_HOST}:{XREAL_EYE_VIDEO_PORT}")
    print("=" * 60)

    try:
        with XREALEyeCameraSource(flip_horizontal=True) as camera:
            print(f"Camera opened: {camera.actual_width}x{camera.actual_height}")
            print("Press 'q' to quit, 's' to save frame")

            cv2.namedWindow("XREAL Eye Test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("XREAL Eye Test", 1024, 756)

            while True:
                frame = camera.read_frame()

                if frame is None:
                    time.sleep(0.01)
                    continue

                # Add frame counter overlay
                cv2.putText(
                    frame,
                    f"Frame: {camera.get_frame_count()}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.imshow("XREAL Eye Test", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"xreal_eye_frame_{camera.get_frame_count()}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved: {filename}")

            cv2.destroyAllWindows()
            print(f"\nTotal frames captured: {camera.get_frame_count()}")

    except CameraError as e:
        print(f"ERROR: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(test_xreal_eye_capture())
