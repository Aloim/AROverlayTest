"""
Landmark smoother for temporal filtering of MediaPipe hand landmarks.

Phase 2 Enhancement 2: Applies One Euro Filter to all 21 hand landmarks
to reduce jitter without introducing noticeable lag. Adapts smoothing
based on movement speed.
"""

from dataclasses import dataclass
from typing import Optional
import time

from hand_detector import HandLandmarks, Landmark
from one_euro_filter import OneEuroFilter
from logger import get_logger
from config import (
    XREAL_EYE_LANDMARK_SMOOTH_ENABLED,
    XREAL_EYE_LANDMARK_MIN_CUTOFF,
    XREAL_EYE_LANDMARK_BETA,
    XREAL_EYE_LANDMARK_D_CUTOFF,
)

logger = get_logger("LandmarkSmoother")


@dataclass
class LandmarkSmootherConfig:
    """Configuration for landmark temporal smoothing."""

    enabled: bool = XREAL_EYE_LANDMARK_SMOOTH_ENABLED
    min_cutoff: float = XREAL_EYE_LANDMARK_MIN_CUTOFF  # Lower = more smoothing
    beta: float = XREAL_EYE_LANDMARK_BETA  # Speed responsiveness
    d_cutoff: float = XREAL_EYE_LANDMARK_D_CUTOFF  # Derivative cutoff

    # Alternative: Simple EMA (faster, simpler)
    use_ema: bool = False
    ema_alpha: float = 0.4  # 0.3-0.4 recommended for hand tracking


class LandmarkSmoother:
    """
    Smooths hand landmarks temporally using One Euro Filter.

    Maintains 21 x 3 = 63 filters (one for each x, y, z coordinate).
    Automatically resets when detection is lost for extended period.

    Attributes:
        config: Smoother configuration.
    """

    NUM_LANDMARKS = 21
    RESET_TIMEOUT_SEC = 0.5  # Reset filters after 500ms of no detection

    def __init__(self, config: Optional[LandmarkSmootherConfig] = None):
        """
        Initialize landmark smoother.

        Args:
            config: Smoother configuration. Uses defaults if None.
        """
        self.config = config or LandmarkSmootherConfig()

        # One Euro Filters: 21 landmarks x 3 coordinates
        self._filters_x: list[OneEuroFilter] = []
        self._filters_y: list[OneEuroFilter] = []
        self._filters_z: list[OneEuroFilter] = []

        # EMA state (alternative to One Euro)
        self._prev_landmarks: Optional[list[Landmark]] = None

        self._last_detection_time: float = 0.0
        self._smoothed_count: int = 0

        self._initialize_filters()

        if self.config.enabled:
            mode = "EMA" if self.config.use_ema else "One Euro Filter"
            logger.info(f"LandmarkSmoother initialized ({mode})")

    def _initialize_filters(self) -> None:
        """Create One Euro Filters for all landmark coordinates."""
        self._filters_x.clear()
        self._filters_y.clear()
        self._filters_z.clear()

        for _ in range(self.NUM_LANDMARKS):
            self._filters_x.append(OneEuroFilter(
                min_cutoff=self.config.min_cutoff,
                beta=self.config.beta,
                d_cutoff=self.config.d_cutoff
            ))
            self._filters_y.append(OneEuroFilter(
                min_cutoff=self.config.min_cutoff,
                beta=self.config.beta,
                d_cutoff=self.config.d_cutoff
            ))
            self._filters_z.append(OneEuroFilter(
                min_cutoff=self.config.min_cutoff,
                beta=self.config.beta,
                d_cutoff=self.config.d_cutoff
            ))

    def smooth(self, landmarks: HandLandmarks) -> HandLandmarks:
        """
        Apply temporal smoothing to hand landmarks.

        Args:
            landmarks: Raw detected landmarks from MediaPipe.

        Returns:
            HandLandmarks with smoothed coordinates.
        """
        if not self.config.enabled:
            return landmarks

        current_time = time.time()

        # Check for timeout - reset filters if detection was lost too long
        if self._last_detection_time > 0:
            time_since_last = current_time - self._last_detection_time
            if time_since_last > self.RESET_TIMEOUT_SEC:
                logger.debug(f"Resetting filters after {time_since_last:.2f}s gap")
                self.reset()

        self._last_detection_time = current_time

        if self.config.use_ema:
            return self._smooth_ema(landmarks)
        else:
            return self._smooth_one_euro(landmarks, current_time)

    def _smooth_one_euro(self, landmarks: HandLandmarks, t: float) -> HandLandmarks:
        """Apply One Euro Filter smoothing to all 21 landmarks."""
        smoothed_landmarks = []

        for i, lm in enumerate(landmarks.landmarks):
            smooth_x = self._filters_x[i].filter(lm.x, t)
            smooth_y = self._filters_y[i].filter(lm.y, t)
            smooth_z = self._filters_z[i].filter(lm.z, t)

            smoothed_landmarks.append(Landmark(
                x=smooth_x,
                y=smooth_y,
                z=smooth_z,
                visibility=lm.visibility
            ))

        self._smoothed_count += 1

        return HandLandmarks(
            landmarks=smoothed_landmarks,
            handedness=landmarks.handedness,
            score=landmarks.score
        )

    def _smooth_ema(self, landmarks: HandLandmarks) -> HandLandmarks:
        """Apply simple EMA smoothing (alternative to One Euro Filter)."""
        alpha = self.config.ema_alpha

        if self._prev_landmarks is None:
            self._prev_landmarks = landmarks.landmarks
            return landmarks

        smoothed_landmarks = []

        for curr, prev in zip(landmarks.landmarks, self._prev_landmarks):
            smooth_x = alpha * curr.x + (1 - alpha) * prev.x
            smooth_y = alpha * curr.y + (1 - alpha) * prev.y
            smooth_z = alpha * curr.z + (1 - alpha) * prev.z

            smoothed_landmarks.append(Landmark(
                x=smooth_x,
                y=smooth_y,
                z=smooth_z,
                visibility=curr.visibility
            ))

        self._prev_landmarks = smoothed_landmarks
        self._smoothed_count += 1

        return HandLandmarks(
            landmarks=smoothed_landmarks,
            handedness=landmarks.handedness,
            score=landmarks.score
        )

    def reset(self) -> None:
        """Reset all filter states (call when tracking is lost)."""
        self._initialize_filters()
        self._prev_landmarks = None
        self._last_detection_time = 0.0
        logger.debug("LandmarkSmoother reset")

    @property
    def smoothed_count(self) -> int:
        """Get total number of frames smoothed."""
        return self._smoothed_count

    @property
    def is_enabled(self) -> bool:
        """Check if smoothing is enabled."""
        return self.config.enabled
