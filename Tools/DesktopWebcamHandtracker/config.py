"""
Configuration constants for DesktopWebcamHandtracker.

This module contains all tunable parameters for camera capture,
hand detection, gesture recognition, and mouse control.
"""

from dataclasses import dataclass
from typing import Final


# Camera configuration
CAMERA_WIDTH: Final[int] = 640
CAMERA_HEIGHT: Final[int] = 480
CAMERA_FPS: Final[int] = 30
DEFAULT_CAMERA_INDEX: Final[int] = 0

# XREAL Eye Camera Constants
XREAL_EYE_HOST: Final[str] = "169.254.2.1"
XREAL_EYE_VIDEO_PORT: Final[int] = 52997
XREAL_EYE_PACKET_SIZE: Final[int] = 193862  # 320 header + 193536 image + 6 footer
XREAL_EYE_HEADER_OFFSET: Final[int] = 0x140  # 320 bytes
XREAL_EYE_WIDTH: Final[int] = 512
XREAL_EYE_HEIGHT: Final[int] = 378

# XREAL Eye IMU Constants
XREAL_EYE_IMU_PORT: Final[int] = 52998
IMU_GYRO_MAX: Final[float] = 10.0  # rad/s (validation range)
IMU_ACCEL_MIN: Final[float] = 9.0  # m/s² (gravity lower bound)
IMU_ACCEL_MAX: Final[float] = 11.0  # m/s² (gravity upper bound)
IMU_LOWPASS_ALPHA: Final[float] = 0.8  # Low-pass filter coefficient (0=max smooth, 1=no smooth)
IMU_CONNECT_TIMEOUT: Final[float] = 5.0  # seconds
IMU_READ_TIMEOUT: Final[float] = 1.0  # seconds
IMU_RECONNECT_DELAY: Final[float] = 2.0  # seconds

# XREAL Eye Camera FOV (degrees) - User estimate: 110-130°
# Used for proper IMU head compensation scaling
XREAL_EYE_FOV_HORIZONTAL_DEG: Final[float] = 120.0  # Wide FOV camera
XREAL_EYE_FOV_VERTICAL_DEG: Final[float] = 90.0     # Estimate based on aspect ratio
# Deadzone to ignore micro head movements (radians, ~0.5 degrees)
XREAL_EYE_IMU_DEADZONE_RAD: Final[float] = 0.01

# MediaPipe configuration
MEDIAPIPE_MODEL_COMPLEXITY: Final[int] = 0  # Lite model for performance
MEDIAPIPE_MAX_NUM_HANDS: Final[int] = 1
MEDIAPIPE_MIN_DETECTION_CONFIDENCE: Final[float] = 0.7
MEDIAPIPE_MIN_TRACKING_CONFIDENCE: Final[float] = 0.5

# =============================================================================
# XREAL Eye Phase 2.3: Model & Confidence Settings
# =============================================================================
# With Phase 2 preprocessing improvements, we can increase these values.
# Start conservative and tune based on testing.

# Model complexity: 0 = Lite (fast), 1 = Full (more robust)
# Testing showed Lite model works better with Phase 2 preprocessing (less jitter)
XREAL_EYE_MODEL_COMPLEXITY: Final[int] = 0  # Lite model - Full model caused jitter

# Detection confidence - 0.01 is minimum needed for stable detection on 4-bit grayscale
# Higher values cause skeleton flickering during movement
XREAL_EYE_MIN_DETECTION_CONFIDENCE: Final[float] = 0.01  # Minimum for stable skeleton
XREAL_EYE_MIN_TRACKING_CONFIDENCE: Final[float] = 0.01  # Match detection
XREAL_EYE_MIN_PRESENCE_CONFIDENCE: Final[float] = 0.01  # Match detection

# XREAL Eye requires IMAGE mode (not VIDEO mode) because VIDEO mode needs
# consistent consecutive detections to maintain tracking state, which is
# impossible with ~50% detection rate on degraded greyscale.
XREAL_EYE_USE_IMAGE_MODE: Final[bool] = True

# =============================================================================
# XREAL Eye Phase 2 Enhancements
# =============================================================================

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# Improves 4-bit greyscale contrast before Gaussian blur
XREAL_EYE_CLAHE_ENABLED: Final[bool] = True
XREAL_EYE_CLAHE_CLIP_LIMIT: Final[float] = 3.0  # More contrast for 4-bit greyscale
XREAL_EYE_CLAHE_TILE_SIZE: Final[int] = 4  # Smaller tiles for low resolution (was 8)

# Gaussian Blur kernel size (smaller = sharper edges, more noise)
XREAL_EYE_GAUSSIAN_KERNEL: Final[int] = 7  # Reduced from 13 for better edge preservation

# =============================================================================
# XREAL Eye Phase 2.1: Sigmoid LUT Bit-Depth Mapping
# =============================================================================
# Replace linear mapping (pixels * 17) with S-curve that stretches middle grays
XREAL_EYE_SIGMOID_LUT_ENABLED: Final[bool] = True
XREAL_EYE_SIGMOID_VIBRANCE: Final[float] = 1.5  # S-curve steepness (1.0-2.0)

# =============================================================================
# XREAL Eye Phase 2.2: Ordered Dithering
# =============================================================================
# Bayer matrix dithering to break up 4-bit banding artifacts before blur
XREAL_EYE_DITHER_ENABLED: Final[bool] = True

# =============================================================================
# XREAL Eye Phase 2.2: Fast Guided Filter (Edge-Preserving)
# =============================================================================
# Replaces Gaussian blur with edge-preserving guided filter
# Requires opencv-contrib-python
XREAL_EYE_GUIDED_FILTER_ENABLED: Final[bool] = True
XREAL_EYE_GUIDED_FILTER_RADIUS: Final[int] = 8  # Filter kernel radius
XREAL_EYE_GUIDED_FILTER_EPS: Final[float] = 1.0  # Regularization (0.5-2.0)

# =============================================================================
# XREAL Eye Phase 2.4: FFT Banding Removal (Optional)
# =============================================================================
# Frequency domain filtering to remove periodic banding artifacts
# Only enable if Phases 2.1-2.3 don't reach 90% detection
XREAL_EYE_FFT_DENOISE_ENABLED: Final[bool] = False  # Off by default (expensive)

# Landmark Temporal Smoothing (One Euro Filter)
# Reduces jitter in 21 hand landmarks without adding lag
XREAL_EYE_LANDMARK_SMOOTH_ENABLED: Final[bool] = True
XREAL_EYE_LANDMARK_MIN_CUTOFF: Final[float] = 0.8  # Lower = more smoothing (was 1.0)
XREAL_EYE_LANDMARK_BETA: Final[float] = 0.1  # Lower = more stable (was 0.15)
XREAL_EYE_LANDMARK_D_CUTOFF: Final[float] = 1.0  # Derivative cutoff

# =============================================================================
# XREAL Eye Phase 2.3: ROI Temporal Smoothing
# =============================================================================
# ROI (Region of Interest) Tracking with One Euro Filter on bounding box
# Re-enabled with temporal smoothing to fix Phase 1 instability issues
XREAL_EYE_ROI_ENABLED: Final[bool] = False  # Disabled to test flickering (re-enable after stabilizing)
XREAL_EYE_ROI_TEMPORAL_SMOOTH: Final[bool] = True  # Apply One Euro Filter to bbox
XREAL_EYE_ROI_EXPANSION: Final[float] = 1.2  # 20% larger than detected hand
XREAL_EYE_ROI_MIN_SIZE: Final[int] = 200  # Minimum 200x200 ROI
XREAL_EYE_ROI_MAX_MISSES: Final[int] = 3  # Quick fallback to full-frame (was 10)
XREAL_EYE_ROI_CONFIDENCE_BOOST: Final[float] = 0.1  # Boost when using ROI
# One Euro Filter settings for bbox smoothing
XREAL_EYE_ROI_BBOX_MIN_CUTOFF: Final[float] = 0.5  # Bbox position smoothing
XREAL_EYE_ROI_BBOX_BETA: Final[float] = 0.1  # Bbox responsiveness
XREAL_EYE_ROI_SIZE_MIN_CUTOFF: Final[float] = 0.3  # Bbox size smoothing (more stable)
XREAL_EYE_ROI_SIZE_BETA: Final[float] = 0.05  # Bbox size responsiveness

# Hand Presence Confirmation
# Prevents false gesture triggers when hand is not clearly visible
HAND_PRESENCE_REQUIRED_FRAMES: Final[int] = 2  # Fast entry (was 4)
HAND_LOST_RESET_FRAMES: Final[int] = 30  # Very tolerant of brief drops (was 15)
HAND_MIN_CONFIDENCE: Final[float] = 0.01  # Accept any detection (XREAL Eye has very low scores)

# Gesture recognition thresholds
PINCH_DISTANCE_THRESHOLD: Final[float] = 0.09  # Normalized distance (more sensitive, easier to trigger)
FIST_CURL_THRESHOLD: Final[float] = 0.15  # Fingers curled threshold
POINT_EXTENSION_THRESHOLD: Final[float] = 0.12  # Index finger extension
PALM_SPREAD_THRESHOLD: Final[float] = 0.08  # All fingers extended
THUMBS_UP_ANGLE_THRESHOLD: Final[float] = 45.0  # Degrees from vertical

# Swipe detection configuration
SWIPE_WINDOW_SIZE: Final[int] = 10  # Frames for sliding window (reduced for faster response)
SWIPE_VELOCITY_THRESHOLD: Final[float] = 0.3  # Normalized velocity (lowered for easier triggering)
SWIPE_MIN_DISTANCE: Final[float] = 0.06  # Minimum travel distance (6% of screen)

# Cursor smoothing (One Euro Filter parameters)
CURSOR_MIN_CUTOFF: Final[float] = 1.5  # Minimum cutoff frequency (higher = less lag, more jitter)
CURSOR_BETA: Final[float] = 0.5  # Speed coefficient (higher = more responsive to fast moves)
CURSOR_DEADZONE: Final[float] = 0.001  # Ignore small movements

# Gesture state machine
GESTURE_HOLD_FRAMES: Final[int] = 3  # Default frames to confirm gesture
GESTURE_HOLD_FRAMES_PINCH: Final[int] = 2  # Keep pinch quick (intentional action)
GESTURE_HOLD_FRAMES_FIST: Final[int] = 5  # Require ~165ms to confirm fist (reduces false triggers)
GESTURE_RELEASE_FRAMES: Final[int] = 20  # Frames to confirm release (very tolerant of drops)
GESTURE_DEBOUNCE_MS: Final[int] = 80  # Minimum time between gestures

# Finger extension detection margin (helps with noisy landmarks)
FINGER_EXTENSION_MARGIN: Final[float] = 0.03  # Tip must be this much above PIP (normalized)
FIST_EXTENSION_MARGIN: Final[float] = 0.05  # More lenient margin for fist detection (easier to trigger)

# Mouse control
MOUSE_MOVE_SENSITIVITY: Final[float] = 1.0  # Base sensitivity multiplier
SCREEN_MARGIN_PERCENT: Final[float] = 0.1  # Edge margin for cursor mapping

# =============================================================================
# Alternate Cursor Control Mode (Experimental)
# =============================================================================
# Relative cursor control where movement is based on distance from origin point,
# rather than absolute screen mapping.
ALTERNATE_CURSOR_DEADZONE: Final[float] = 0.03  # 3% deadzone - prevents jitter without being noticeable
ALTERNATE_CURSOR_LINEAR_ZONE: Final[float] = 0.30  # Linear zone to 30% - provides precision for small movements
ALTERNATE_CURSOR_MAX_SPEED: Final[float] = 50.0  # Max pixels/frame - ~1.3s to cross 1920px screen at 30fps
ALTERNATE_CURSOR_BASE_SENSITIVITY: Final[float] = 800.0  # Base speed multiplier
ALTERNATE_CURSOR_ACCEL_FACTOR: Final[float] = 2.5  # Quadratic acceleration beyond linear zone

# =============================================================================
# Dual Camera Settings (Phone + Webcam)
# =============================================================================

# Phone camera TCP connection
PHONE_LANDMARK_PORT: Final[int] = 52990
PHONE_MAX_LATENCY_MS: Final[float] = 50.0
PHONE_BUFFER_SIZE: Final[int] = 10
PHONE_CONNECT_TIMEOUT: Final[float] = 5.0
PHONE_RECONNECT_DELAY: Final[float] = 2.0

# Camera fusion settings
FUSION_MIN_CONFIDENCE: Final[float] = 0.3
FUSION_PHONE_DEPTH_WEIGHT: Final[float] = 1.5  # Phone better for depth
FUSION_MAX_TIMESTAMP_DIFF_MS: Final[float] = 50.0

# EKF (Extended Kalman Filter) settings
EKF_PROCESS_NOISE_POS: Final[float] = 0.001  # Position noise (m^2)
EKF_PROCESS_NOISE_VEL: Final[float] = 0.01   # Velocity noise ((m/s)^2)
EKF_JUMP_THRESHOLD: Final[float] = 0.2       # Position jump detection (m)

# Triangulation reference values
REFERENCE_HAND_SIZE: Final[float] = 0.08  # 8cm average hand width
REFERENCE_DEPTH: Final[float] = 0.5        # At 50cm distance

# Dual-view gesture agreement
DUAL_VIEW_AGREEMENT_BOOST: Final[float] = 1.2
DUAL_VIEW_DISAGREEMENT_PENALTY: Final[float] = 0.7
DUAL_VIEW_PARTIAL_PENALTY: Final[float] = 0.8

# Logging
LOG_FILENAME: Final[str] = "desktop_webcam_handtracker.log"
LOG_MAX_BYTES: Final[int] = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT: Final[int] = 3

# Exit codes
EXIT_SUCCESS: Final[int] = 0
EXIT_PROFILE_ERROR: Final[int] = 1
EXIT_CAMERA_ERROR: Final[int] = 2
EXIT_RUNTIME_ERROR: Final[int] = 3


@dataclass
class GestureThresholds:
    """Container for gesture detection thresholds."""

    pinch_distance: float = PINCH_DISTANCE_THRESHOLD
    fist_curl: float = FIST_CURL_THRESHOLD
    point_extension: float = POINT_EXTENSION_THRESHOLD
    palm_spread: float = PALM_SPREAD_THRESHOLD
    thumbs_up_angle: float = THUMBS_UP_ANGLE_THRESHOLD
    swipe_velocity: float = SWIPE_VELOCITY_THRESHOLD
    swipe_min_distance: float = SWIPE_MIN_DISTANCE


@dataclass
class CursorSettings:
    """Container for cursor control settings (One Euro Filter)."""

    min_cutoff: float = CURSOR_MIN_CUTOFF  # Lower = smoother when slow
    beta: float = CURSOR_BETA  # Higher = more responsive when fast
    deadzone: float = CURSOR_DEADZONE
    sensitivity: float = MOUSE_MOVE_SENSITIVITY
    screen_margin: float = SCREEN_MARGIN_PERCENT


# Gesture to action mapping defaults
DEFAULT_GESTURE_MAPPINGS: dict[str, dict] = {
    "pinch": {"action": "leftClick", "isEnabled": True},
    "fist": {"action": "rightClick", "isEnabled": True},
    "point": {"action": "moveCursor", "isEnabled": True},
    "palm": {"action": "none", "isEnabled": False},
    "thumbsUp": {"action": "none", "isEnabled": False},
    "swipeUp": {"action": "scrollUp", "isEnabled": True},
    "swipeDown": {"action": "scrollDown", "isEnabled": True},
    "doublePinch": {"action": "doubleClick", "isEnabled": True},
    "triplePinch": {"action": "none", "isEnabled": False},
}
