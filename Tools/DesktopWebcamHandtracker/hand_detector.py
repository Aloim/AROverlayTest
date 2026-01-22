"""
Hand detector using MediaPipe Hands.

Provides hand landmark detection with MediaPipe integration.
Supports both Solutions API (Python 3.9-3.12) and Tasks API (Python 3.13+).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Detect which MediaPipe API is available
USING_TASKS_API = False
_mp_hands = None
_mp_drawing = None

print("[HandDetector] Detecting MediaPipe API...")

try:
    import mediapipe as mp
    print(f"[HandDetector] MediaPipe version: {mp.__version__ if hasattr(mp, '__version__') else 'unknown'}")

    # Try legacy Solutions API first (Python 3.9-3.12)
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
        _mp_hands = mp.solutions.hands
        _mp_drawing = mp.solutions.drawing_utils
        USING_TASKS_API = False
        print("[HandDetector] Using Solutions API (mp.solutions.hands)")
    # Fallback to Tasks API (Python 3.13+)
    elif hasattr(mp, 'tasks'):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        USING_TASKS_API = True
        print("[HandDetector] Using Tasks API (mp.tasks.python.vision)")
    else:
        raise ImportError(
            "MediaPipe installation incomplete. "
            "Neither Solutions API nor Tasks API found."
        )
except ImportError as e:
    print(f"[HandDetector] ERROR: Failed to import MediaPipe: {e}")
    raise ImportError(
        "MediaPipe is required. Install with: pip install mediapipe\n"
        "Note: MediaPipe officially supports Python 3.9-3.12, "
        "Python 3.13+ uses Tasks API with automatic model download."
    ) from e

from logger import get_logger
from config import (
    MEDIAPIPE_MODEL_COMPLEXITY,
    MEDIAPIPE_MAX_NUM_HANDS,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    XREAL_EYE_ROI_ENABLED,
    XREAL_EYE_ROI_EXPANSION,
    XREAL_EYE_ROI_MIN_SIZE,
    XREAL_EYE_ROI_MAX_MISSES,
    XREAL_EYE_ROI_CONFIDENCE_BOOST,
    XREAL_EYE_ROI_TEMPORAL_SMOOTH,
    XREAL_EYE_ROI_BBOX_MIN_CUTOFF,
    XREAL_EYE_ROI_BBOX_BETA,
    XREAL_EYE_ROI_SIZE_MIN_CUTOFF,
    XREAL_EYE_ROI_SIZE_BETA,
)

# Import OneEuroFilter from dedicated module (avoids circular import)
from one_euro_filter import OneEuroFilter

logger = get_logger("HandDetector")


# MediaPipe landmark indices
class LandmarkIndex:
    """MediaPipe hand landmark indices."""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


@dataclass
class Landmark:
    """Single hand landmark with 3D coordinates and visibility."""
    x: float  # Normalized x [0, 1]
    y: float  # Normalized y [0, 1]
    z: float  # Relative depth
    visibility: float = 1.0


@dataclass
class HandLandmarks:
    """
    Complete hand landmark data.

    Attributes:
        landmarks: List of 21 hand landmarks.
        handedness: 'Left' or 'Right'.
        score: Detection confidence score.
    """
    landmarks: list[Landmark]
    handedness: str
    score: float

    def get_landmark(self, index: int) -> Optional[Landmark]:
        """Get landmark by index."""
        if 0 <= index < len(self.landmarks):
            return self.landmarks[index]
        return None

    @property
    def wrist(self) -> Landmark:
        """Get wrist landmark."""
        return self.landmarks[LandmarkIndex.WRIST]

    @property
    def thumb_tip(self) -> Landmark:
        """Get thumb tip landmark."""
        return self.landmarks[LandmarkIndex.THUMB_TIP]

    @property
    def index_tip(self) -> Landmark:
        """Get index finger tip landmark."""
        return self.landmarks[LandmarkIndex.INDEX_TIP]

    @property
    def middle_tip(self) -> Landmark:
        """Get middle finger tip landmark."""
        return self.landmarks[LandmarkIndex.MIDDLE_TIP]

    @property
    def ring_tip(self) -> Landmark:
        """Get ring finger tip landmark."""
        return self.landmarks[LandmarkIndex.RING_TIP]

    @property
    def pinky_tip(self) -> Landmark:
        """Get pinky finger tip landmark."""
        return self.landmarks[LandmarkIndex.PINKY_TIP]

    def get_fingertips(self) -> list[Landmark]:
        """Get all fingertip landmarks."""
        return [
            self.thumb_tip,
            self.index_tip,
            self.middle_tip,
            self.ring_tip,
            self.pinky_tip
        ]

    def get_palm_center(self) -> tuple[float, float]:
        """
        Calculate approximate palm center.

        Returns:
            (x, y) normalized coordinates of palm center.
        """
        # Average of wrist and MCP joints
        points = [
            self.landmarks[LandmarkIndex.WRIST],
            self.landmarks[LandmarkIndex.INDEX_MCP],
            self.landmarks[LandmarkIndex.MIDDLE_MCP],
            self.landmarks[LandmarkIndex.RING_MCP],
            self.landmarks[LandmarkIndex.PINKY_MCP],
        ]

        x = sum(p.x for p in points) / len(points)
        y = sum(p.y for p in points) / len(points)

        return (x, y)


class HandDetector:
    """
    Hand detector using MediaPipe Hands.

    Detects hand landmarks in RGB images using the MediaPipe
    Hands solution with configurable model complexity.
    Supports both Solutions API (Python 3.9-3.12) and Tasks API (Python 3.13+).
    """

    def __init__(
        self,
        model_complexity: int = MEDIAPIPE_MODEL_COMPLEXITY,
        max_num_hands: int = MEDIAPIPE_MAX_NUM_HANDS,
        min_detection_confidence: float = MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        use_image_mode: bool = False
    ):
        """
        Initialize hand detector.

        Args:
            model_complexity: Model complexity (0=Lite, 1=Full).
            max_num_hands: Maximum number of hands to detect.
            min_detection_confidence: Minimum detection confidence.
            min_tracking_confidence: Minimum tracking confidence.
            use_image_mode: Use IMAGE mode instead of VIDEO mode (Tasks API only).
                           IMAGE mode processes each frame independently.
                           VIDEO mode maintains tracking state between frames.
                           Use IMAGE mode for low-quality input (like XREAL Eye greyscale).
        """
        self.model_complexity = model_complexity
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.use_image_mode = use_image_mode

        self._hands = None  # Solutions API Hands object
        self._landmarker = None  # Tasks API HandLandmarker object
        self._is_initialized = False
        self._frame_count = 0
        self._using_tasks_api = USING_TASKS_API

        api_type = "Tasks API" if self._using_tasks_api else "Solutions API"
        mode_str = "IMAGE" if use_image_mode else "VIDEO"
        logger.info(
            f"HandDetector initialized ({api_type}, complexity={model_complexity}, "
            f"max_hands={max_num_hands}, mode={mode_str})"
        )

    def initialize(self) -> None:
        """Initialize MediaPipe Hands model."""
        if self._is_initialized:
            return

        if self._using_tasks_api:
            self._initialize_tasks_api()
        else:
            self._initialize_solutions_api()

        self._is_initialized = True

    def _initialize_solutions_api(self) -> None:
        """Initialize using Solutions API (Python 3.9-3.12)."""
        logger.debug("Initializing MediaPipe Hands (Solutions API)...")

        self._hands = _mp_hands.Hands(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

        logger.info("MediaPipe Hands initialized (Solutions API)")

    def _initialize_tasks_api(self) -> None:
        """Initialize using Tasks API (Python 3.13+)."""
        logger.info("Initializing MediaPipe Hands (Tasks API)...")
        logger.debug("Using Tasks API (Python 3.13+)")

        try:
            # Import model manager for model download
            from model_manager import ensure_hand_landmarker_model

            # Download model if needed
            logger.debug("Checking for hand landmarker model...")
            model_path = ensure_hand_landmarker_model()
            logger.debug(f"Model path: {model_path}")

            # Create HandLandmarker options
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            logger.debug("Creating HandLandmarker...")
            base_options = mp_python.BaseOptions(model_asset_path=model_path)

            # Select running mode:
            # - VIDEO mode: maintains tracking state between frames, better for consistent input
            # - IMAGE mode: processes each frame independently, better for low-quality input
            if self.use_image_mode:
                running_mode = mp_vision.RunningMode.IMAGE
                mode_str = "IMAGE"
            else:
                running_mode = mp_vision.RunningMode.VIDEO
                mode_str = "VIDEO"

            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=running_mode,
                num_hands=self.max_num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )

            self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
            self._timestamp_ms = 0  # Track timestamp for VIDEO mode
            logger.debug(f"HandLandmarker created successfully ({mode_str} mode)")

            logger.info(f"MediaPipe Hands initialized (Tasks API, {mode_str} mode)")
        except Exception as e:
            logger.error(f"Failed to initialize Tasks API: {e}")
            import traceback
            traceback.print_exc()
            raise

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._hands:
            self._hands.close()
            self._hands = None
        if self._landmarker:
            self._landmarker.close()
            self._landmarker = None
        self._is_initialized = False
        logger.debug("HandDetector closed")

    def detect(self, rgb_image: np.ndarray) -> Optional[HandLandmarks]:
        """
        Detect hand landmarks in an RGB image.

        Args:
            rgb_image: RGB image as numpy array (H, W, 3).

        Returns:
            HandLandmarks if a hand is detected, None otherwise.
        """
        if not self._is_initialized:
            self.initialize()

        self._frame_count += 1

        if self._using_tasks_api:
            return self._detect_tasks_api(rgb_image)
        else:
            return self._detect_solutions_api(rgb_image)

    def _detect_solutions_api(self, rgb_image: np.ndarray) -> Optional[HandLandmarks]:
        """Detect using Solutions API (Python 3.9-3.12)."""
        if self._hands is None:
            logger.error("HandDetector not initialized (Solutions API)")
            return None

        # Process image
        results = self._hands.process(rgb_image)

        if not results.multi_hand_landmarks:
            return None

        # Get first hand (we only track one hand)
        hand_landmarks = results.multi_hand_landmarks[0]

        # Get handedness
        handedness = "Right"
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label
            score = results.multi_handedness[0].classification[0].score
        else:
            score = 1.0

        # Convert to our Landmark format
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append(Landmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility if hasattr(lm, 'visibility') else 1.0
            ))

        return HandLandmarks(
            landmarks=landmarks,
            handedness=handedness,
            score=score
        )

    def _detect_tasks_api(self, rgb_image: np.ndarray) -> Optional[HandLandmarks]:
        """Detect using Tasks API (Python 3.13+)."""
        if self._landmarker is None:
            logger.error("HandDetector not initialized (Tasks API)")
            return None

        # Convert numpy array to MediaPipe Image
        import mediapipe as mp

        # Ensure image is contiguous and in correct format
        if not rgb_image.flags['C_CONTIGUOUS']:
            rgb_image = np.ascontiguousarray(rgb_image)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Detect hands using appropriate mode
        if self.use_image_mode:
            # IMAGE mode: process each frame independently
            result = self._landmarker.detect(mp_image)
        else:
            # VIDEO mode: maintain tracking state with timestamp
            self._timestamp_ms += 33  # ~30 FPS
            result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not result.hand_landmarks:
            return None

        # Get first hand (we only track one hand)
        hand_landmarks = result.hand_landmarks[0]

        # Get handedness
        handedness = "Right"
        score = 1.0
        if result.handedness:
            handedness_info = result.handedness[0][0]
            handedness = handedness_info.category_name
            score = handedness_info.score

        # Convert to our Landmark format
        landmarks = []
        for lm in hand_landmarks:
            landmarks.append(Landmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility if hasattr(lm, 'visibility') else 1.0
            ))

        return HandLandmarks(
            landmarks=landmarks,
            handedness=handedness,
            score=score
        )

    def draw_landmarks(
        self,
        image: np.ndarray,
        hand_landmarks: HandLandmarks,
        draw_connections: bool = True
    ) -> np.ndarray:
        """
        Draw hand landmarks on an image.

        Args:
            image: BGR image to draw on.
            hand_landmarks: Detected hand landmarks.
            draw_connections: Draw connections between landmarks.

        Returns:
            Image with landmarks drawn.
        """
        if self._using_tasks_api:
            return self._draw_landmarks_manual(image, hand_landmarks, draw_connections)
        else:
            return self._draw_landmarks_solutions(image, hand_landmarks, draw_connections)

    def _draw_landmarks_solutions(
        self,
        image: np.ndarray,
        hand_landmarks: HandLandmarks,
        draw_connections: bool = True
    ) -> np.ndarray:
        """Draw using Solutions API drawing utilities."""
        # Create a temporary landmark object
        class TempLandmark:
            def __init__(self, lm: Landmark):
                self.x = lm.x
                self.y = lm.y
                self.z = lm.z

        class TempLandmarkList:
            def __init__(self, landmarks: list[Landmark]):
                self.landmark = [TempLandmark(lm) for lm in landmarks]

        temp_landmarks = TempLandmarkList(hand_landmarks.landmarks)

        _mp_drawing.draw_landmarks(
            image,
            temp_landmarks,
            _mp_hands.HAND_CONNECTIONS if draw_connections else None,
            _mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            _mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        return image

    def _draw_landmarks_manual(
        self,
        image: np.ndarray,
        hand_landmarks: HandLandmarks,
        draw_connections: bool = True
    ) -> np.ndarray:
        """Draw landmarks manually (for Tasks API or fallback)."""
        import cv2

        h, w = image.shape[:2]

        # Hand connections (same as MediaPipe)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]

        # Draw connections
        if draw_connections:
            for start_idx, end_idx in connections:
                start = hand_landmarks.landmarks[start_idx]
                end = hand_landmarks.landmarks[end_idx]
                start_pt = (int(start.x * w), int(start.y * h))
                end_pt = (int(end.x * w), int(end.y * h))
                cv2.line(image, start_pt, end_pt, (0, 0, 255), 2)

        # Draw landmarks
        for lm in hand_landmarks.landmarks:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(image, pt, 3, (0, 255, 0), -1)

        return image

    def __enter__(self) -> "HandDetector":
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


def distance(lm1: Landmark, lm2: Landmark) -> float:
    """
    Calculate Euclidean distance between two landmarks.

    Args:
        lm1: First landmark.
        lm2: Second landmark.

    Returns:
        Euclidean distance in normalized coordinates.
    """
    return np.sqrt(
        (lm1.x - lm2.x) ** 2 +
        (lm1.y - lm2.y) ** 2 +
        (lm1.z - lm2.z) ** 2
    )


def distance_2d(lm1: Landmark, lm2: Landmark) -> float:
    """
    Calculate 2D distance between two landmarks (ignoring z).

    Args:
        lm1: First landmark.
        lm2: Second landmark.

    Returns:
        2D Euclidean distance in normalized coordinates.
    """
    return np.sqrt(
        (lm1.x - lm2.x) ** 2 +
        (lm1.y - lm2.y) ** 2
    )


# =============================================================================
# Phase 2 Enhancement 3: ROI (Region of Interest) Tracking
# =============================================================================

@dataclass
class ROIConfig:
    """Configuration for ROI-based detection."""

    enabled: bool = XREAL_EYE_ROI_ENABLED
    expansion_factor: float = XREAL_EYE_ROI_EXPANSION  # Expand ROI by this factor
    min_roi_size: int = XREAL_EYE_ROI_MIN_SIZE  # Minimum ROI dimension
    max_misses_before_reset: int = XREAL_EYE_ROI_MAX_MISSES  # Frames without detection
    confidence_boost: float = XREAL_EYE_ROI_CONFIDENCE_BOOST  # Add when using ROI

    # Phase 2.3: Temporal smoothing with One Euro Filter
    temporal_smooth_enabled: bool = XREAL_EYE_ROI_TEMPORAL_SMOOTH
    bbox_min_cutoff: float = XREAL_EYE_ROI_BBOX_MIN_CUTOFF
    bbox_beta: float = XREAL_EYE_ROI_BBOX_BETA
    size_min_cutoff: float = XREAL_EYE_ROI_SIZE_MIN_CUTOFF
    size_beta: float = XREAL_EYE_ROI_SIZE_BETA


class ROITracker:
    """
    Tracks hand region of interest between frames for focused detection.

    Phase 2 Enhancement 3: When a hand is detected, maintain a bounding box
    of the hand region. For subsequent frames, crop to this expanded region
    to reduce false positives and potentially improve detection speed.

    Phase 2.3 Upgrade: Added One Euro Filter temporal smoothing on bbox
    coordinates to fix instability issues with XREAL Eye 4-bit greyscale.

    Attributes:
        config: ROI tracking configuration.
    """

    def __init__(self, config: Optional[ROIConfig] = None):
        """
        Initialize ROI tracker.

        Args:
            config: ROI configuration. Uses defaults if None.
        """
        self.config = config or ROIConfig()

        self._last_bbox: Optional[tuple[int, int, int, int]] = None  # x, y, w, h
        self._last_roi_size: Optional[tuple[int, int]] = None  # actual (roi_w, roi_h) from cropping
        self._consecutive_misses: int = 0
        self._frame_size: Optional[tuple[int, int]] = None  # h, w
        self._roi_count: int = 0

        # Phase 2.3: One Euro Filters for temporal bbox smoothing
        self._temporal_smooth = self.config.temporal_smooth_enabled
        if self._temporal_smooth:
            # Position filters (x, y)
            self._filter_x = OneEuroFilter(
                min_cutoff=self.config.bbox_min_cutoff,
                beta=self.config.bbox_beta
            )
            self._filter_y = OneEuroFilter(
                min_cutoff=self.config.bbox_min_cutoff,
                beta=self.config.bbox_beta
            )
            # Size filters (w, h) - more stable, less responsive
            self._filter_w = OneEuroFilter(
                min_cutoff=self.config.size_min_cutoff,
                beta=self.config.size_beta
            )
            self._filter_h = OneEuroFilter(
                min_cutoff=self.config.size_min_cutoff,
                beta=self.config.size_beta
            )
            logger.debug("ROI temporal smoothing enabled (One Euro Filter)")

        if self.config.enabled:
            logger.info(
                f"ROITracker initialized: expansion={self.config.expansion_factor}, "
                f"min_size={self.config.min_roi_size}, temporal_smooth={self._temporal_smooth}"
            )

    def get_roi(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Get region of interest from frame.

        Args:
            frame: Full input frame.

        Returns:
            Tuple of (cropped_frame, offset) where offset is (x_offset, y_offset)
            for mapping landmarks back to full frame coordinates.
            If ROI is not available, returns (full_frame, (0, 0)).
        """
        if not self.config.enabled or self._last_bbox is None:
            return frame, (0, 0)

        h, w = frame.shape[:2]
        self._frame_size = (h, w)

        x, y, bw, bh = self._last_bbox

        # Calculate expansion
        expand_x = int(bw * (self.config.expansion_factor - 1) / 2)
        expand_y = int(bh * (self.config.expansion_factor - 1) / 2)

        # Calculate new bounds with expansion
        x1 = max(0, x - expand_x)
        y1 = max(0, y - expand_y)
        x2 = min(w, x + bw + expand_x)
        y2 = min(h, y + bh + expand_y)

        # Enforce minimum size
        roi_w = x2 - x1
        roi_h = y2 - y1

        if roi_w < self.config.min_roi_size:
            center_x = (x1 + x2) // 2
            x1 = max(0, center_x - self.config.min_roi_size // 2)
            x2 = min(w, x1 + self.config.min_roi_size)

        if roi_h < self.config.min_roi_size:
            center_y = (y1 + y2) // 2
            y1 = max(0, center_y - self.config.min_roi_size // 2)
            y2 = min(h, y1 + self.config.min_roi_size)

        self._roi_count += 1
        cropped = frame[y1:y2, x1:x2]

        # Store actual ROI dimensions for coordinate scaling in update()
        self._last_roi_size = (x2 - x1, y2 - y1)

        return cropped, (x1, y1)

    def update(
        self,
        landmarks: Optional[HandLandmarks],
        frame_size: tuple[int, int],
        offset: tuple[int, int] = (0, 0)
    ) -> Optional[HandLandmarks]:
        """
        Update ROI based on detection and scale landmarks to full frame.

        Args:
            landmarks: Detected landmarks (in ROI coordinates if cropped).
            frame_size: Size of original full frame (h, w).
            offset: (x_offset, y_offset) from cropping.

        Returns:
            Landmarks scaled to full frame coordinates, or None.
        """
        h, w = frame_size
        x_off, y_off = offset

        if landmarks is None:
            self._consecutive_misses += 1
            if self._consecutive_misses >= self.config.max_misses_before_reset:
                if self._last_bbox is not None:
                    logger.debug("ROI reset after consecutive misses")
                self.reset()
            return None

        self._consecutive_misses = 0

        # If we used a cropped ROI, scale landmarks back to full frame
        if offset != (0, 0) and self._last_roi_size is not None:
            # Use actual ROI dimensions stored during get_roi()
            roi_w, roi_h = self._last_roi_size

            scaled_landmarks = []
            for lm in landmarks.landmarks:
                # Convert from ROI-relative normalized to full-frame normalized
                full_x = (lm.x * roi_w + x_off) / w
                full_y = (lm.y * roi_h + y_off) / h

                scaled_landmarks.append(Landmark(
                    x=np.clip(full_x, 0.0, 1.0),
                    y=np.clip(full_y, 0.0, 1.0),
                    z=lm.z,
                    visibility=lm.visibility
                ))

            # Create new HandLandmarks with scaled positions and boosted confidence
            landmarks = HandLandmarks(
                landmarks=scaled_landmarks,
                handedness=landmarks.handedness,
                score=min(1.0, landmarks.score + self.config.confidence_boost)
            )

        # Update bounding box from current detection
        self._update_bbox(landmarks, w, h)

        return landmarks

    def _update_bbox(self, landmarks: HandLandmarks, frame_w: int, frame_h: int) -> None:
        """
        Update bounding box from detected landmarks.

        Uses One Euro Filter for temporal smoothing (Phase 2.3) instead of EMA.
        This provides adaptive smoothing that reduces jitter during slow movements
        while remaining responsive during fast movements.

        Args:
            landmarks: Detected hand landmarks.
            frame_w: Frame width in pixels.
            frame_h: Frame height in pixels.
        """
        # Extract landmark coordinates
        xs = [lm.x * frame_w for lm in landmarks.landmarks]
        ys = [lm.y * frame_h for lm in landmarks.landmarks]

        # Calculate bounding box with padding
        padding = 20
        x = max(0, int(min(xs)) - padding)
        y = max(0, int(min(ys)) - padding)
        w = min(frame_w, int(max(xs)) + padding) - x
        h = min(frame_h, int(max(ys)) + padding) - y

        if self._temporal_smooth and self._last_bbox is not None:
            # Apply One Euro Filter to each bbox component
            x = int(self._filter_x.filter(float(x)))
            y = int(self._filter_y.filter(float(y)))
            w = int(self._filter_w.filter(float(w)))
            h = int(self._filter_h.filter(float(h)))

            # Clamp to frame bounds
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            w = max(self.config.min_roi_size, min(w, frame_w - x))
            h = max(self.config.min_roi_size, min(h, frame_h - y))
        elif self._last_bbox is not None:
            # Legacy EMA smoothing (fallback if temporal disabled)
            alpha = 0.3
            old_x, old_y, old_w, old_h = self._last_bbox
            x = int(alpha * x + (1 - alpha) * old_x)
            y = int(alpha * y + (1 - alpha) * old_y)
            w = int(alpha * w + (1 - alpha) * old_w)
            h = int(alpha * h + (1 - alpha) * old_h)

        self._last_bbox = (x, y, w, h)

    def reset(self) -> None:
        """
        Reset ROI tracker state.

        Clears bbox history and reinitializes One Euro Filters to prevent
        stale state from affecting future tracking.
        """
        self._last_bbox = None
        self._last_roi_size = None
        self._consecutive_misses = 0

        # Reset One Euro Filters
        if self._temporal_smooth:
            self._filter_x = OneEuroFilter(
                min_cutoff=self.config.bbox_min_cutoff,
                beta=self.config.bbox_beta
            )
            self._filter_y = OneEuroFilter(
                min_cutoff=self.config.bbox_min_cutoff,
                beta=self.config.bbox_beta
            )
            self._filter_w = OneEuroFilter(
                min_cutoff=self.config.size_min_cutoff,
                beta=self.config.size_beta
            )
            self._filter_h = OneEuroFilter(
                min_cutoff=self.config.size_min_cutoff,
                beta=self.config.size_beta
            )

        logger.debug("ROITracker reset")

    @property
    def has_roi(self) -> bool:
        """Check if a valid ROI is being tracked."""
        return self._last_bbox is not None

    @property
    def roi_count(self) -> int:
        """Get number of frames processed with ROI cropping."""
        return self._roi_count

    @property
    def is_enabled(self) -> bool:
        """Check if ROI tracking is enabled."""
        return self.config.enabled
