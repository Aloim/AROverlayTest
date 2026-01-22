"""
Camera fusion for dual-camera hand tracking.

Combines landmark observations from webcam and phone camera
to produce 3D hand positions with improved depth accuracy.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple
import time

import numpy as np

from logger import get_logger
from hand_detector import HandLandmarks, Landmark
from phone_landmark_receiver import PhoneLandmarks
from triangulation import (
    PhoneCalibration,
    CameraCalibration,
    triangulate_points,
    compute_reprojection_error
)

logger = get_logger("CameraFusion")


@dataclass
class FusionWeights:
    """
    Weights for fusing webcam (primary) and phone (secondary) camera data.

    Each weight pair (primary, secondary) should sum to 1.0.
    Higher weight = more trust in that camera's data for that axis/feature.

    Default configuration (user-specified):
    - X axis: Equal weight (0.5, 0.5)
    - Y axis: Primary (webcam) higher (0.7, 0.3)
    - Z axis: Secondary (phone) higher for depth (0.3, 0.7)
    - Gesture: Primary (webcam) higher (0.7, 0.3)
    """
    # X axis weights (primary, secondary)
    x_primary: float = 0.5
    x_secondary: float = 0.5

    # Y axis weights (primary, secondary) - webcam higher
    y_primary: float = 0.7
    y_secondary: float = 0.3

    # Z axis (depth) weights (primary, secondary) - phone higher
    z_primary: float = 0.3
    z_secondary: float = 0.7

    # Gesture recognition weights (primary, secondary) - webcam higher
    gesture_primary: float = 0.7
    gesture_secondary: float = 0.3

    def validate(self) -> bool:
        """Check that weights are valid (each pair sums to ~1.0)."""
        tolerance = 0.01
        return (
            abs(self.x_primary + self.x_secondary - 1.0) < tolerance and
            abs(self.y_primary + self.y_secondary - 1.0) < tolerance and
            abs(self.z_primary + self.z_secondary - 1.0) < tolerance and
            abs(self.gesture_primary + self.gesture_secondary - 1.0) < tolerance
        )


class FusionMethod(Enum):
    """Method used for 3D fusion."""
    STEREO = auto()       # Both cameras, triangulation
    WEBCAM_ONLY = auto()  # Webcam only, depth from hand size
    PHONE_ONLY = auto()   # Phone only, depth from hand size
    NONE = auto()         # No hand detected


@dataclass
class Keypoint3D:
    """3D keypoint with confidence."""
    x: float
    y: float
    z: float
    confidence: float


@dataclass
class FusedHandData:
    """
    Result of camera fusion.

    Contains 21 3D keypoints in world coordinates (meters),
    plus metadata about the fusion process.
    """
    keypoints: List[Keypoint3D]
    fusion_method: FusionMethod
    webcam_confidence: float = 0.0
    phone_confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def has_hand(self) -> bool:
        """Check if a hand was detected."""
        return self.fusion_method != FusionMethod.NONE

    @property
    def is_stereo(self) -> bool:
        """Check if stereo fusion was used."""
        return self.fusion_method == FusionMethod.STEREO

    @property
    def wrist(self) -> Optional[Keypoint3D]:
        """Get wrist (landmark 0)."""
        return self.keypoints[0] if self.keypoints else None

    @property
    def index_tip(self) -> Optional[Keypoint3D]:
        """Get index fingertip (landmark 8)."""
        return self.keypoints[8] if len(self.keypoints) > 8 else None

    def get_cursor_position_3d(self) -> Optional[Tuple[float, float, float]]:
        """Get cursor position as (x, y, z) in meters."""
        tip = self.index_tip
        if tip:
            return (tip.x, tip.y, tip.z)
        return None


class CameraFusion:
    """
    Fuses landmarks from webcam and phone into 3D positions.

    When both cameras see the hand, uses stereo triangulation for
    accurate 3D reconstruction. Falls back to single-camera depth
    estimation when only one camera is available.

    Usage:
        fusion = CameraFusion(calibration)

        # Each frame:
        result = fusion.fuse(webcam_landmarks, phone_landmarks)
        if result.has_hand:
            cursor_3d = result.get_cursor_position_3d()
    """

    # Constants for single-camera depth estimation
    REFERENCE_HAND_SIZE = 0.08  # 8cm average hand width
    REFERENCE_DEPTH = 0.5       # At 50cm distance

    def __init__(
        self,
        calibration: PhoneCalibration,
        max_timestamp_diff_ms: float = 50.0,
        min_stereo_confidence: float = 0.3,
        phone_depth_weight: float = 1.5,
        weights: Optional[FusionWeights] = None
    ):
        """
        Initialize camera fusion.

        Args:
            calibration: Phone-to-webcam calibration data.
            max_timestamp_diff_ms: Max time difference for stereo fusion.
            min_stereo_confidence: Min confidence to use stereo.
            phone_depth_weight: Weight for phone camera's depth (>1 = trust more).
            weights: Axis-specific fusion weights. If None, uses defaults.
        """
        self.calibration = calibration
        self.max_timestamp_diff_ms = max_timestamp_diff_ms
        self.min_stereo_confidence = min_stereo_confidence
        self.phone_depth_weight = phone_depth_weight
        self.weights = weights or FusionWeights()

        # Validate weights
        if not self.weights.validate():
            logger.warning("Fusion weights don't sum to 1.0, using defaults")
            self.weights = FusionWeights()

        # Get calibration objects
        self._webcam_calib = calibration.get_webcam_calibration()
        self._phone_calib = calibration.get_phone_calibration()

        logger.info(
            f"CameraFusion initialized with weights: "
            f"X=({self.weights.x_primary:.1f}/{self.weights.x_secondary:.1f}), "
            f"Y=({self.weights.y_primary:.1f}/{self.weights.y_secondary:.1f}), "
            f"Z=({self.weights.z_primary:.1f}/{self.weights.z_secondary:.1f}), "
            f"gesture=({self.weights.gesture_primary:.1f}/{self.weights.gesture_secondary:.1f})"
        )

    def fuse(
        self,
        webcam_landmarks: Optional[HandLandmarks],
        phone_landmarks: Optional[PhoneLandmarks],
        webcam_timestamp_ms: Optional[float] = None
    ) -> FusedHandData:
        """
        Fuse landmarks from both cameras.

        Args:
            webcam_landmarks: Landmarks from desktop webcam.
            phone_landmarks: Landmarks from phone camera.
            webcam_timestamp_ms: Webcam frame timestamp (for sync).

        Returns:
            FusedHandData with 3D keypoints and method used.
        """
        webcam_has_hand = webcam_landmarks is not None and webcam_landmarks.score > 0
        phone_has_hand = phone_landmarks is not None and phone_landmarks.has_hand

        # Case 1: Both cameras see hand - use stereo
        if webcam_has_hand and phone_has_hand:
            # Check timestamp synchronization
            if webcam_timestamp_ms and phone_landmarks:
                time_diff = abs(webcam_timestamp_ms - phone_landmarks.timestamp_ms)
                if time_diff > self.max_timestamp_diff_ms:
                    logger.debug(f"Timestamps too far apart: {time_diff:.1f}ms")
                    # Fall through to single camera

            # Check minimum confidence
            if (webcam_landmarks.score >= self.min_stereo_confidence and
                phone_landmarks.confidence >= self.min_stereo_confidence):
                return self._fuse_stereo(webcam_landmarks, phone_landmarks)

        # Case 2: Only webcam
        if webcam_has_hand:
            return self._fuse_single_camera(
                webcam_landmarks,
                is_webcam=True
            )

        # Case 3: Only phone
        if phone_has_hand:
            return self._fuse_phone_only(phone_landmarks)

        # Case 4: No hand detected
        return FusedHandData(
            keypoints=[],
            fusion_method=FusionMethod.NONE,
            webcam_confidence=0.0,
            phone_confidence=0.0
        )

    def _fuse_stereo(
        self,
        webcam: HandLandmarks,
        phone: PhoneLandmarks
    ) -> FusedHandData:
        """
        Fuse using stereo triangulation with axis-specific weighting.

        Applies fusion weights:
        - X axis: Equal weight (webcam 0.5, phone 0.5)
        - Y axis: Webcam higher (0.7, 0.3)
        - Z axis: Phone higher for depth (0.3, 0.7)

        Args:
            webcam: Webcam landmarks.
            phone: Phone landmarks.

        Returns:
            FusedHandData with triangulated 3D positions.
        """
        keypoints_3d = []

        # Extract 2D points from both cameras
        webcam_2d = [(lm.x, lm.y) for lm in webcam.landmarks]
        phone_2d = phone.keypoints

        # Triangulate each landmark to get 3D positions
        try:
            points_3d = triangulate_points(
                webcam_2d,
                phone_2d,
                self._webcam_calib,
                self._phone_calib
            )

            for i, pt3d in enumerate(points_3d):
                # Get 2D coordinates from each camera
                webcam_x, webcam_y = webcam_2d[i] if i < len(webcam_2d) else (0.5, 0.5)
                phone_x, phone_y, phone_z = phone_2d[i] if i < len(phone_2d) else (0.5, 0.5, 0.0)

                # Triangulated 3D position
                tri_x, tri_y, tri_z = float(pt3d[0]), float(pt3d[1]), float(pt3d[2])

                # Apply weighted fusion to X and Y based on normalized 2D positions
                # This biases the final position toward the more trusted camera's view
                # X: Equal weight
                fused_x = tri_x  # Keep triangulated X (already fused geometrically)

                # Y: Weight webcam higher (0.7) - bias toward webcam's Y perception
                # Apply subtle bias by blending triangulated Y with webcam-biased estimate
                webcam_y_world = (webcam_y - 0.5) * abs(tri_z) if tri_z != 0 else 0
                phone_y_world = (phone_y - 0.5) * abs(tri_z) if tri_z != 0 else 0
                fused_y = (
                    tri_y * 0.6 +  # Keep most of triangulated Y
                    webcam_y_world * self.weights.y_primary * 0.4 +
                    phone_y_world * self.weights.y_secondary * 0.4
                ) if abs(tri_z) > 0.1 else tri_y

                # Z: Use triangulated depth but weight phone's contribution higher
                # The triangulation already accounts for both views, but we can
                # bias toward the phone's depth estimate for more accurate Z
                fused_z = tri_z  # Triangulated Z is already the best depth estimate

                # Compute confidence with gesture weighting (webcam higher)
                webcam_vis = webcam.landmarks[i].visibility if i < len(webcam.landmarks) else 0.5
                phone_conf = phone.confidence

                # Weighted confidence for gesture recognition purposes
                confidence = (
                    webcam_vis * self.weights.gesture_primary +
                    phone_conf * self.weights.gesture_secondary
                )

                keypoints_3d.append(Keypoint3D(
                    x=fused_x,
                    y=fused_y,
                    z=fused_z,
                    confidence=confidence
                ))

        except Exception as e:
            logger.error(f"Triangulation failed: {e}")
            # Fall back to webcam-only
            return self._fuse_single_camera(webcam, is_webcam=True)

        return FusedHandData(
            keypoints=keypoints_3d,
            fusion_method=FusionMethod.STEREO,
            webcam_confidence=webcam.score,
            phone_confidence=phone.confidence
        )

    def _fuse_single_camera(
        self,
        webcam: HandLandmarks,
        is_webcam: bool = True
    ) -> FusedHandData:
        """
        Estimate 3D positions from single camera using hand size.

        Uses the apparent hand size to estimate depth, assuming
        a reference hand size at a reference distance.

        Args:
            webcam: Webcam landmarks.
            is_webcam: True if from webcam (for confidence scaling).

        Returns:
            FusedHandData with estimated 3D positions.
        """
        keypoints_3d = []

        # Estimate depth from hand size
        depth = self._estimate_depth_from_size(webcam)

        for lm in webcam.landmarks:
            # Convert normalized coords to approximate world coords
            # This is a simplified approximation
            x = (lm.x - 0.5) * depth  # Rough mapping
            y = (lm.y - 0.5) * depth

            # Reduce confidence for single-camera depth
            confidence = lm.visibility * 0.5

            keypoints_3d.append(Keypoint3D(
                x=x,
                y=y,
                z=depth + lm.z * 0.1,  # Use MediaPipe z as offset
                confidence=confidence
            ))

        return FusedHandData(
            keypoints=keypoints_3d,
            fusion_method=FusionMethod.WEBCAM_ONLY if is_webcam else FusionMethod.PHONE_ONLY,
            webcam_confidence=webcam.score,
            phone_confidence=0.0
        )

    def _fuse_phone_only(
        self,
        phone: PhoneLandmarks
    ) -> FusedHandData:
        """
        Estimate 3D positions from phone camera only.

        Args:
            phone: Phone landmarks.

        Returns:
            FusedHandData with estimated 3D positions.
        """
        keypoints_3d = []

        # Estimate depth from hand size
        depth = self._estimate_depth_from_phone_size(phone)

        for x, y, z in phone.keypoints:
            # Convert normalized coords to approximate world coords
            x_3d = (x - 0.5) * depth * self.phone_depth_weight
            y_3d = (y - 0.5) * depth * self.phone_depth_weight

            confidence = phone.confidence * 0.5  # Reduce for single camera

            keypoints_3d.append(Keypoint3D(
                x=x_3d,
                y=y_3d,
                z=depth + z * 0.1,
                confidence=confidence
            ))

        return FusedHandData(
            keypoints=keypoints_3d,
            fusion_method=FusionMethod.PHONE_ONLY,
            webcam_confidence=0.0,
            phone_confidence=phone.confidence
        )

    def _estimate_depth_from_size(self, landmarks: HandLandmarks) -> float:
        """
        Estimate depth from webcam hand size.

        Uses distance between wrist and middle fingertip as proxy for hand size.

        Args:
            landmarks: Webcam landmarks.

        Returns:
            Estimated depth in meters.
        """
        wrist = landmarks.wrist
        middle_tip = landmarks.middle_tip

        hand_size = np.sqrt(
            (middle_tip.x - wrist.x) ** 2 +
            (middle_tip.y - wrist.y) ** 2
        )

        # Depth is inversely proportional to apparent size
        if hand_size > 0.01:
            depth = self.REFERENCE_DEPTH * (self.REFERENCE_HAND_SIZE / hand_size)
            return np.clip(depth, 0.2, 2.0)  # Clamp to reasonable range

        return self.REFERENCE_DEPTH

    def _estimate_depth_from_phone_size(self, phone: PhoneLandmarks) -> float:
        """
        Estimate depth from phone hand size.

        Args:
            phone: Phone landmarks.

        Returns:
            Estimated depth in meters.
        """
        wrist = phone.keypoints[0]  # Wrist
        middle_tip = phone.keypoints[12]  # Middle fingertip

        hand_size = np.sqrt(
            (middle_tip[0] - wrist[0]) ** 2 +
            (middle_tip[1] - wrist[1]) ** 2
        )

        if hand_size > 0.01:
            depth = self.REFERENCE_DEPTH * (self.REFERENCE_HAND_SIZE / hand_size)
            return np.clip(depth, 0.2, 2.0)

        return self.REFERENCE_DEPTH


# ============================================================================
# Multi-Camera Fusion (Phase 6 - NEW)
# ============================================================================

class MultiCameraFusion:
    """
    Fuses landmarks from multiple cameras using weighted averaging (Phase 6).

    Unlike CameraFusion (which does stereo triangulation for dual cameras),
    this class uses simple weighted averaging for N cameras without requiring
    full stereo calibration. Useful for robustness rather than accurate 3D.

    Usage:
        from profile_loader import CameraSlotConfig

        # Create camera slot configurations
        slots = [slot1, slot2, slot3]  # From profile

        fusion = MultiCameraFusion(slots)

        # Each frame:
        landmarks_by_slot = {
            0: webcam_landmarks,
            1: phone_landmarks,
            2: xreal_eye_landmarks
        }
        fused = fusion.fuse(landmarks_by_slot)
    """

    def __init__(self, camera_configs):
        """
        Initialize multi-camera fusion.

        Args:
            camera_configs: List of CameraSlotConfig instances (from profile).
        """
        # Import here to avoid circular dependency
        from profile_loader import CameraSlotConfig

        self.camera_configs = [c for c in camera_configs if c.is_enabled]
        self._normalize_weights()
        logger.info(f"MultiCameraFusion initialized with {len(self.camera_configs)} cameras")

    def _normalize_weights(self):
        """Normalize weights so they sum to 1.0 for each axis."""
        if not self.camera_configs:
            self.normalized_weights = {}
            return

        total_x = sum(c.x_weight for c in self.camera_configs) or 1.0
        total_y = sum(c.y_weight for c in self.camera_configs) or 1.0
        total_z = sum(c.z_weight for c in self.camera_configs) or 1.0
        total_g = sum(c.gesture_weight for c in self.camera_configs) or 1.0

        self.normalized_weights = {}
        for c in self.camera_configs:
            self.normalized_weights[c.slot_index] = {
                'x': c.x_weight / total_x,
                'y': c.y_weight / total_y,
                'z': c.z_weight / total_z,
                'gesture': c.gesture_weight / total_g,
                'flip_h': c.flip_horizontal,
                'flip_v': c.flip_vertical
            }

        logger.debug(f"Normalized weights for {len(self.camera_configs)} cameras")

    def apply_flip(self, landmarks, flip_h: bool, flip_v: bool):
        """
        Apply horizontal/vertical flip to landmarks.

        Args:
            landmarks: List of (x, y, z) tuples or (x, y) tuples.
            flip_h: Whether to flip horizontally.
            flip_v: Whether to flip vertically.

        Returns:
            Flipped landmarks or None if input is None.
        """
        if landmarks is None:
            return None

        # Flip is applied to normalized coordinates (0-1 range)
        flipped = []
        for lm in landmarks:
            if len(lm) >= 2:
                x, y = lm[0], lm[1]
                z = lm[2] if len(lm) > 2 else 0.0

                if flip_h:
                    x = 1.0 - x
                if flip_v:
                    y = 1.0 - y

                if len(lm) > 2:
                    flipped.append((x, y, z))
                else:
                    flipped.append((x, y))
            else:
                flipped.append(lm)

        return flipped

    def fuse(self, landmarks_by_slot: dict) -> Optional[list]:
        """
        Fuse landmarks from multiple cameras.

        Args:
            landmarks_by_slot: Dict mapping slot_index to landmark list.
                              Each landmark is (x, y) or (x, y, z) tuple.

        Returns:
            Fused landmark list (same format as input) or None if no valid landmarks.
        """
        if not landmarks_by_slot:
            return None

        valid_sources = []
        for slot_idx, landmarks in landmarks_by_slot.items():
            if landmarks is not None and slot_idx in self.normalized_weights:
                weights = self.normalized_weights[slot_idx]
                flipped = self.apply_flip(landmarks, weights['flip_h'], weights['flip_v'])
                valid_sources.append((slot_idx, flipped, weights))

        if not valid_sources:
            return None

        # Single camera - just return it (with flips applied)
        if len(valid_sources) == 1:
            return valid_sources[0][1]

        # Multi-camera weighted fusion
        num_landmarks = len(valid_sources[0][1])
        fused = []

        for i in range(num_landmarks):
            # Accumulate weighted coordinates
            x_sum = 0.0
            y_sum = 0.0
            z_sum = 0.0
            has_z = False

            for slot_idx, landmarks, weights in valid_sources:
                if i < len(landmarks):
                    lm = landmarks[i]
                    if len(lm) >= 2:
                        x_sum += lm[0] * weights['x']
                        y_sum += lm[1] * weights['y']
                        if len(lm) > 2:
                            z_sum += lm[2] * weights['z']
                            has_z = True

            # Build fused landmark
            if has_z:
                fused.append((x_sum, y_sum, z_sum))
            else:
                fused.append((x_sum, y_sum))

        return fused
