"""
Dual-view gesture recognizer for improved gesture detection.

Combines gesture recognition from webcam and phone camera views
to improve confidence and reduce false positives.
"""

from dataclasses import dataclass
from typing import Optional

from logger import get_logger
from gesture_recognizer import GestureRecognizer, GestureType, GestureResult
from hand_detector import HandLandmarks, Landmark, LandmarkIndex
from phone_landmark_receiver import PhoneLandmarks
from camera_fusion import FusionWeights

logger = get_logger("DualViewGestureRecognizer")


@dataclass
class DualViewGestureResult:
    """
    Result of dual-view gesture recognition.

    Attributes:
        gesture: Detected gesture type.
        confidence: Confidence boosted/reduced by camera agreement.
        webcam_gesture: What webcam detected.
        phone_gesture: What phone detected.
        agreement: True if both cameras agree on gesture.
        cursor_position: (x, y) cursor position (normalized).
    """
    gesture: GestureType
    confidence: float
    webcam_gesture: GestureType
    phone_gesture: GestureType
    agreement: bool
    cursor_position: tuple[float, float]

    @property
    def is_high_confidence(self) -> bool:
        """Check if confidence is above threshold for action."""
        return self.confidence >= 0.7


class DualViewGestureRecognizer:
    """
    Fuses gestures from two camera views for improved accuracy.

    Agreement Logic:
    - Both agree: confidence boosted (max * 1.2, capped at 1.0)
    - Disagree: confidence penalized (higher * 0.7)
    - One NONE: confidence slightly reduced (detected * 0.8)

    Gesture-specific benefits:
    - PINCH: Side view disambiguates from FIST (thumb position clearer)
    - FIST: Top view confirms all fingers closed
    - POINT: Side view verifies only index extended
    - PALM: Strong confirmation from both views

    Usage:
        recognizer = DualViewGestureRecognizer()

        # Each frame:
        result = recognizer.recognize(webcam_landmarks, phone_landmarks)
        if result.gesture != GestureType.NONE:
            # Handle gesture
            if result.is_high_confidence:
                execute_action(result.gesture)
    """

    # Confidence modifiers
    AGREEMENT_BOOST = 1.2      # Multiply confidence when cameras agree
    DISAGREEMENT_PENALTY = 0.7  # Multiply when cameras disagree
    PARTIAL_PENALTY = 0.8      # Multiply when one camera sees NONE

    def __init__(self, weights: Optional[FusionWeights] = None):
        """
        Initialize dual-view recognizer with two internal recognizers.

        Args:
            weights: Fusion weights for gesture confidence.
                     Webcam (primary) has higher weight by default (0.7 vs 0.3).
        """
        self._webcam_recognizer = GestureRecognizer()
        self._phone_recognizer = PhoneGestureRecognizer()
        self._weights = weights or FusionWeights()

        logger.info(
            f"DualViewGestureRecognizer initialized "
            f"(webcam weight: {self._weights.gesture_primary:.1f}, "
            f"phone weight: {self._weights.gesture_secondary:.1f})"
        )

    def recognize(
        self,
        webcam_landmarks: Optional[HandLandmarks],
        phone_landmarks: Optional[PhoneLandmarks]
    ) -> DualViewGestureResult:
        """
        Recognize gesture from dual camera views.

        Args:
            webcam_landmarks: Landmarks from desktop webcam.
            phone_landmarks: Landmarks from phone camera.

        Returns:
            DualViewGestureResult with fused gesture and confidence.
        """
        # Get individual recognitions
        webcam_result = None
        if webcam_landmarks:
            webcam_result = self._webcam_recognizer.recognize(webcam_landmarks)

        phone_result = None
        if phone_landmarks and phone_landmarks.has_hand:
            phone_result = self._phone_recognizer.recognize(phone_landmarks)

        # Extract gesture types
        webcam_gesture = webcam_result.gesture if webcam_result else GestureType.NONE
        phone_gesture = phone_result.gesture if phone_result else GestureType.NONE
        webcam_conf = webcam_result.confidence if webcam_result else 0.0
        phone_conf = phone_result.confidence if phone_result else 0.0

        # Determine cursor position (prefer webcam)
        cursor_pos = (0.5, 0.5)
        if webcam_result and webcam_result.cursor_position:
            cursor_pos = webcam_result.cursor_position
        elif phone_result and phone_landmarks:
            cursor_pos = phone_landmarks.index_tip[:2]

        # Fuse gestures
        gesture, confidence, agreement = self._fuse_gestures(
            webcam_gesture, webcam_conf,
            phone_gesture, phone_conf
        )

        return DualViewGestureResult(
            gesture=gesture,
            confidence=confidence,
            webcam_gesture=webcam_gesture,
            phone_gesture=phone_gesture,
            agreement=agreement,
            cursor_position=cursor_pos
        )

    def _fuse_gestures(
        self,
        webcam_gesture: GestureType,
        webcam_conf: float,
        phone_gesture: GestureType,
        phone_conf: float
    ) -> tuple[GestureType, float, bool]:
        """
        Fuse gesture detections from both cameras using weighted confidence.

        Uses FusionWeights to determine how much to trust each camera's
        gesture detection. Webcam (primary) has higher weight by default.

        Args:
            webcam_gesture: Gesture detected by webcam.
            webcam_conf: Webcam gesture confidence.
            phone_gesture: Gesture detected by phone.
            phone_conf: Phone gesture confidence.

        Returns:
            Tuple of (final_gesture, adjusted_confidence, cameras_agree).
        """
        # Both NONE
        if webcam_gesture == GestureType.NONE and phone_gesture == GestureType.NONE:
            return GestureType.NONE, 0.0, True

        # Apply weights to confidence
        webcam_weight = self._weights.gesture_primary  # Default 0.7
        phone_weight = self._weights.gesture_secondary  # Default 0.3

        weighted_webcam = webcam_conf * webcam_weight
        weighted_phone = phone_conf * phone_weight

        # Agreement: both cameras detect same gesture
        if webcam_gesture == phone_gesture:
            # Use weighted average of confidence, then boost for agreement
            combined_conf = weighted_webcam + weighted_phone
            boosted_conf = min(1.0, combined_conf * self.AGREEMENT_BOOST)
            return webcam_gesture, boosted_conf, True

        # One camera sees NONE
        if webcam_gesture == GestureType.NONE:
            # Phone sees something, webcam doesn't - use phone but penalize more
            # since webcam (higher weight) doesn't see it
            reduced_conf = weighted_phone * self.PARTIAL_PENALTY
            return phone_gesture, reduced_conf, False

        if phone_gesture == GestureType.NONE:
            # Webcam sees something, phone doesn't - use webcam with mild penalty
            # since webcam (higher weight) is more trusted
            reduced_conf = weighted_webcam * self.PARTIAL_PENALTY * 1.2  # Less penalty
            return webcam_gesture, min(1.0, reduced_conf), False

        # Disagreement: cameras detect different gestures
        # Use the camera with higher WEIGHTED confidence
        if weighted_webcam >= weighted_phone:
            final_gesture = webcam_gesture
            final_conf = weighted_webcam * self.DISAGREEMENT_PENALTY
        else:
            final_gesture = phone_gesture
            final_conf = weighted_phone * self.DISAGREEMENT_PENALTY

        logger.debug(f"Gesture disagreement: webcam={webcam_gesture.name} "
                    f"(w={weighted_webcam:.2f}), phone={phone_gesture.name} "
                    f"(w={weighted_phone:.2f}), using {final_gesture.name}")

        return final_gesture, final_conf, False

    def reset(self) -> None:
        """Reset both internal recognizers."""
        self._webcam_recognizer.reset()
        self._phone_recognizer.reset()


class PhoneGestureRecognizer:
    """
    Gesture recognizer for phone camera landmarks.

    Adapts the standard gesture recognition to work with PhoneLandmarks
    data structure (list of tuples instead of Landmark objects).
    """

    def __init__(self):
        """Initialize phone gesture recognizer."""
        # Gesture thresholds (same as main recognizer)
        from config import (
            PINCH_DISTANCE_THRESHOLD,
            FINGER_EXTENSION_MARGIN,
            FIST_EXTENSION_MARGIN,
            THUMBS_UP_ANGLE_THRESHOLD
        )

        self.pinch_threshold = PINCH_DISTANCE_THRESHOLD
        self.finger_margin = FINGER_EXTENSION_MARGIN
        self.fist_margin = FIST_EXTENSION_MARGIN
        self.thumbsup_angle = THUMBS_UP_ANGLE_THRESHOLD

    def recognize(self, landmarks: PhoneLandmarks) -> GestureResult:
        """
        Recognize gesture from phone landmarks.

        Args:
            landmarks: PhoneLandmarks from phone camera.

        Returns:
            GestureResult with detected gesture.
        """
        kp = landmarks.keypoints
        handedness = "Right" if landmarks.handedness == 1 else "Left"

        # Cursor position (index tip)
        cursor_pos = kp[LandmarkIndex.INDEX_TIP][:2]

        # Calculate finger states
        thumb_up = self._is_thumb_extended(kp, handedness)
        index_up = kp[LandmarkIndex.INDEX_TIP][1] < kp[LandmarkIndex.INDEX_PIP][1] - self.finger_margin
        middle_up = kp[LandmarkIndex.MIDDLE_TIP][1] < kp[LandmarkIndex.MIDDLE_PIP][1] - self.finger_margin
        ring_up = kp[LandmarkIndex.RING_TIP][1] < kp[LandmarkIndex.RING_PIP][1] - self.finger_margin
        pinky_up = kp[LandmarkIndex.PINKY_TIP][1] < kp[LandmarkIndex.PINKY_PIP][1] - self.finger_margin

        num_extended = sum([index_up, middle_up, ring_up, pinky_up])

        # Pinch distance
        thumb_tip = kp[LandmarkIndex.THUMB_TIP]
        index_tip = kp[LandmarkIndex.INDEX_TIP]
        pinch_dist = self._distance_2d(thumb_tip, index_tip)

        # Classify gesture
        if pinch_dist < self.pinch_threshold:
            confidence = 1.0 - (pinch_dist / self.pinch_threshold)
            return GestureResult(GestureType.PINCH, min(1.0, confidence), cursor_pos)

        if num_extended >= 4:
            return GestureResult(GestureType.PALM, 0.9, cursor_pos)

        if index_up and not ring_up and not pinky_up:
            return GestureResult(GestureType.POINT, 0.9, cursor_pos)

        if thumb_up and num_extended == 0:
            thumb_angle = self._get_thumb_angle(kp)
            if thumb_angle < self.thumbsup_angle:
                return GestureResult(GestureType.THUMBS_UP, 0.85, cursor_pos)

        # Fist with lenient margin
        fist_index = kp[LandmarkIndex.INDEX_TIP][1] < kp[LandmarkIndex.INDEX_PIP][1] - self.fist_margin
        fist_middle = kp[LandmarkIndex.MIDDLE_TIP][1] < kp[LandmarkIndex.MIDDLE_PIP][1] - self.fist_margin
        fist_ring = kp[LandmarkIndex.RING_TIP][1] < kp[LandmarkIndex.RING_PIP][1] - self.fist_margin
        fist_pinky = kp[LandmarkIndex.PINKY_TIP][1] < kp[LandmarkIndex.PINKY_PIP][1] - self.fist_margin
        fist_num = sum([fist_index, fist_middle, fist_ring, fist_pinky])

        if fist_num == 0 and not thumb_up and pinch_dist >= self.pinch_threshold:
            return GestureResult(GestureType.FIST, 0.9, cursor_pos)

        return GestureResult(GestureType.NONE, 0.0, cursor_pos)

    def _is_thumb_extended(self, kp: list, handedness: str) -> bool:
        """Check if thumb is extended."""
        thumb_tip = kp[LandmarkIndex.THUMB_TIP]
        thumb_ip = kp[LandmarkIndex.THUMB_IP]

        if handedness == "Right":
            return thumb_tip[0] < thumb_ip[0]
        else:
            return thumb_tip[0] > thumb_ip[0]

    def _get_thumb_angle(self, kp: list) -> float:
        """Get thumb angle from vertical."""
        import math

        thumb_tip = kp[LandmarkIndex.THUMB_TIP]
        thumb_mcp = kp[LandmarkIndex.THUMB_MCP]

        dx = thumb_tip[0] - thumb_mcp[0]
        dy = thumb_tip[1] - thumb_mcp[1]

        return abs(math.degrees(math.atan2(abs(dx), -dy)))

    def _distance_2d(self, p1: tuple, p2: tuple) -> float:
        """Calculate 2D distance between points."""
        import math
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def reset(self) -> None:
        """Reset recognizer state."""
        pass  # Stateless
