"""
Camera manager for DesktopWebcamHandtracker.

Provides OpenCV VideoCapture wrapper with configuration and frame capture.
"""

import time
from typing import Optional

import cv2
import numpy as np

from logger import get_logger
from config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, DEFAULT_CAMERA_INDEX

logger = get_logger("CameraManager")


class CameraError(Exception):
    """Raised when camera operations fail."""
    pass


class CameraManager:
    """
    Manages webcam capture using OpenCV VideoCapture.

    Attributes:
        camera_index: Index of the camera device.
        width: Capture width in pixels.
        height: Capture height in pixels.
        fps: Target frames per second.
    """

    def __init__(
        self,
        camera_index: int = DEFAULT_CAMERA_INDEX,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        fps: int = CAMERA_FPS
    ):
        """
        Initialize camera manager.

        Args:
            camera_index: Camera device index (0 for default).
            width: Desired capture width.
            height: Desired capture height.
            fps: Desired frame rate.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps

        self._capture: Optional[cv2.VideoCapture] = None
        self._is_open = False
        self._frame_count = 0
        self._last_frame_time = 0.0

    @property
    def is_open(self) -> bool:
        """Check if camera is currently open."""
        return self._is_open and self._capture is not None

    @property
    def actual_width(self) -> int:
        """Get actual capture width."""
        if self._capture:
            return int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        return 0

    @property
    def actual_height(self) -> int:
        """Get actual capture height."""
        if self._capture:
            return int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return 0

    @property
    def actual_fps(self) -> float:
        """Get actual capture FPS."""
        if self._capture:
            return self._capture.get(cv2.CAP_PROP_FPS)
        return 0.0

    def open(self) -> None:
        """
        Open the camera for capture.

        Raises:
            CameraError: If camera cannot be opened.
        """
        if self._is_open:
            logger.warning("Camera already open, closing first")
            self.close()

        logger.info(f"Opening camera {self.camera_index}...")

        # Try DirectShow backend first on Windows for better performance
        self._capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        if not self._capture.isOpened():
            # Fallback to default backend
            logger.debug("DirectShow failed, trying default backend")
            self._capture = cv2.VideoCapture(self.camera_index)

        if not self._capture.isOpened():
            raise CameraError(f"Failed to open camera {self.camera_index}")

        # Configure capture properties
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

        # Verify settings
        actual_w = self.actual_width
        actual_h = self.actual_height
        actual_fps = self.actual_fps

        logger.info(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

        if actual_w != self.width or actual_h != self.height:
            logger.warning(
                f"Requested {self.width}x{self.height}, got {actual_w}x{actual_h}"
            )

        self._is_open = True
        self._frame_count = 0
        self._last_frame_time = time.perf_counter()

    def close(self) -> None:
        """Close the camera and release resources."""
        if self._capture:
            logger.info("Closing camera")
            self._capture.release()
            self._capture = None
        self._is_open = False

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a single frame from the camera.

        Returns:
            BGR image as numpy array, or None if read failed.

        Raises:
            CameraError: If camera is not open.
        """
        if not self._is_open or self._capture is None:
            raise CameraError("Camera is not open")

        ret, frame = self._capture.read()

        if not ret or frame is None:
            logger.warning("Failed to read frame from camera")
            return None

        self._frame_count += 1
        self._last_frame_time = time.perf_counter()

        return frame

    def read_frame_rgb(self) -> Optional[np.ndarray]:
        """
        Read a single frame and convert to RGB.

        Returns:
            RGB image as numpy array, or None if read failed.
        """
        frame = self.read_frame()
        if frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def get_frame_count(self) -> int:
        """Get total frames captured since opening."""
        return self._frame_count

    def __enter__(self) -> "CameraManager":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


def list_available_cameras(max_index: int = 10) -> list[int]:
    """
    Enumerate available camera indices.

    Args:
        max_index: Maximum index to probe.

    Returns:
        List of available camera indices.
    """
    available = []

    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(i)
            cap.release()

    logger.debug(f"Available cameras: {available}")
    return available


def select_camera(
    preferred_index: int = -1,
    preferred_name: Optional[str] = None
) -> int:
    """
    Select the best available camera.

    Args:
        preferred_index: Preferred camera index (-1 for auto).
        preferred_name: Preferred camera name (currently unused on Windows).

    Returns:
        Selected camera index.

    Raises:
        CameraError: If no camera is available.
    """
    available = list_available_cameras()

    if not available:
        raise CameraError("No cameras available")

    # If preferred index is valid, use it
    if preferred_index >= 0:
        if preferred_index in available:
            logger.info(f"Using preferred camera index: {preferred_index}")
            return preferred_index
        else:
            logger.warning(
                f"Preferred camera {preferred_index} not available, "
                f"using {available[0]}"
            )

    # Use first available camera
    selected = available[0]
    logger.info(f"Auto-selected camera index: {selected}")
    return selected
