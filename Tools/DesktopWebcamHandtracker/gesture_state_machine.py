"""
Gesture state machine for DesktopWebcamHandtracker.

Tracks gesture hold/release pairs with debouncing to prevent
false triggers and provide stable gesture detection.
Also detects multi-tap gestures (double/triple pinch).
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from logger import get_logger
from config import (
    GESTURE_HOLD_FRAMES,
    GESTURE_HOLD_FRAMES_PINCH,
    GESTURE_HOLD_FRAMES_FIST,
    GESTURE_RELEASE_FRAMES,
    GESTURE_DEBOUNCE_MS,
)
from gesture_recognizer import GestureType, GestureResult

logger = get_logger("GestureStateMachine")

# Multi-tap configuration
MULTI_TAP_WINDOW_MS: float = 500.0  # Time window for consecutive taps (ms)
MULTI_TAP_MAX_HOLD_MS: float = 300.0  # Max hold time to count as a tap (ms)


class GestureState(Enum):
    """State of a tracked gesture."""
    IDLE = auto()       # No gesture detected
    PENDING = auto()    # Gesture detected, awaiting confirmation
    ACTIVE = auto()     # Gesture confirmed and active
    RELEASING = auto()  # Gesture ended, awaiting release confirmation


@dataclass
class GestureEvent:
    """Event emitted when gesture state changes."""
    gesture: GestureType
    event_type: str  # "start", "end", "hold"
    cursor_position: Optional[tuple[float, float]] = None
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)


class TrackedGesture:
    """Tracks state of a single gesture type."""

    def __init__(
        self,
        gesture_type: GestureType,
        hold_frames: int = GESTURE_HOLD_FRAMES,
        release_frames: int = GESTURE_RELEASE_FRAMES,
        activation_delay_ms: int = 0  # NEW: 0 = use hold_frames parameter
    ):
        """
        Initialize tracked gesture.

        Args:
            gesture_type: Type of gesture to track.
            hold_frames: Frames required to confirm gesture (used if activation_delay_ms == 0).
            release_frames: Frames required to confirm release.
            activation_delay_ms: Activation delay in milliseconds (0 = use hold_frames).
        """
        self.gesture_type = gesture_type
        self.release_frames = release_frames

        # Convert activation_delay_ms to hold_frames if provided
        if activation_delay_ms > 0:
            # Assume ~30 FPS: frame_duration = 1000ms / 30 = 33.3ms
            self.hold_frames = max(1, int(activation_delay_ms / 33.3))
            logger.debug(
                f"{gesture_type.name}: activation_delay_ms={activation_delay_ms} "
                f"-> hold_frames={self.hold_frames}"
            )
        else:
            # Use provided hold_frames parameter (legacy behavior)
            self.hold_frames = hold_frames

        self.state = GestureState.IDLE
        self._pending_count = 0
        self._release_count = 0
        self._last_position: Optional[tuple[float, float]] = None
        self._activation_time: float = 0.0

    def update(
        self, detected: bool, position: Optional[tuple[float, float]] = None
    ) -> Optional[str]:
        """
        Update gesture state.

        Args:
            detected: Whether this gesture is currently detected.
            position: Current cursor position.

        Returns:
            Event type ("start", "end", "hold") or None.
        """
        self._last_position = position
        event: Optional[str] = None

        if self.state == GestureState.IDLE:
            if detected:
                self.state = GestureState.PENDING
                self._pending_count = 1

        elif self.state == GestureState.PENDING:
            if detected:
                self._pending_count += 1
                if self._pending_count >= self.hold_frames:
                    self.state = GestureState.ACTIVE
                    self._activation_time = time.time()
                    event = "start"
                    logger.debug(f"Gesture {self.gesture_type.name} started")
            else:
                # Reset if not detected
                self.state = GestureState.IDLE
                self._pending_count = 0

        elif self.state == GestureState.ACTIVE:
            if detected:
                event = "hold"
            else:
                self.state = GestureState.RELEASING
                self._release_count = 1

        elif self.state == GestureState.RELEASING:
            if detected:
                # False release, go back to active
                self.state = GestureState.ACTIVE
                self._release_count = 0
                event = "hold"
            else:
                self._release_count += 1
                if self._release_count >= self.release_frames:
                    self.state = GestureState.IDLE
                    self._release_count = 0
                    event = "end"
                    logger.debug(f"Gesture {self.gesture_type.name} ended")

        return event

    @property
    def is_active(self) -> bool:
        """Check if gesture is currently active."""
        return self.state in (GestureState.ACTIVE, GestureState.RELEASING)

    @property
    def last_position(self) -> Optional[tuple[float, float]]:
        """Get last known cursor position."""
        return self._last_position

    def reset(self) -> None:
        """Reset gesture state."""
        self.state = GestureState.IDLE
        self._pending_count = 0
        self._release_count = 0


class MultiTapDetector:
    """
    Detects multi-tap gestures (double/triple pinch).

    Tracks consecutive quick taps of a base gesture and generates
    multi-tap events when thresholds are reached.
    """

    def __init__(
        self,
        base_gesture: GestureType,
        double_gesture: GestureType,
        triple_gesture: GestureType,
        tap_window_ms: float = MULTI_TAP_WINDOW_MS,
        max_hold_ms: float = MULTI_TAP_MAX_HOLD_MS
    ):
        """
        Initialize multi-tap detector.

        Args:
            base_gesture: The gesture to track taps for (e.g., PINCH).
            double_gesture: Gesture type to emit for double tap.
            triple_gesture: Gesture type to emit for triple tap.
            tap_window_ms: Time window for consecutive taps.
            max_hold_ms: Maximum hold time to count as a tap.
        """
        self.base_gesture = base_gesture
        self.double_gesture = double_gesture
        self.triple_gesture = triple_gesture
        self.tap_window_ms = tap_window_ms
        self.max_hold_ms = max_hold_ms

        self._tap_count = 0
        self._last_tap_time: float = 0.0
        self._gesture_start_time: float = 0.0
        self._is_gesture_active = False

    def on_gesture_start(self, timestamp: float) -> None:
        """Called when base gesture starts."""
        self._gesture_start_time = timestamp
        self._is_gesture_active = True

    def on_gesture_end(self, timestamp: float) -> Optional[GestureType]:
        """
        Called when base gesture ends.

        Returns:
            Multi-tap gesture type if threshold reached, None otherwise.
        """
        if not self._is_gesture_active:
            return None

        self._is_gesture_active = False

        # Check if this was a quick tap (not a hold)
        hold_duration_ms = (timestamp - self._gesture_start_time) * 1000
        if hold_duration_ms > self.max_hold_ms:
            # Too long, reset tap count
            self._tap_count = 0
            logger.debug(f"Tap too long ({hold_duration_ms:.0f}ms), reset count")
            return None

        # Check if within tap window
        time_since_last_tap_ms = (timestamp - self._last_tap_time) * 1000
        if self._tap_count > 0 and time_since_last_tap_ms > self.tap_window_ms:
            # Too slow, reset
            self._tap_count = 0
            logger.debug(f"Tap window expired ({time_since_last_tap_ms:.0f}ms), reset count")

        # Count this tap
        self._tap_count += 1
        self._last_tap_time = timestamp
        logger.debug(f"Tap count: {self._tap_count}")

        # Check for multi-tap
        if self._tap_count >= 3:
            self._tap_count = 0
            logger.info(f"Triple tap detected!")
            return self.triple_gesture
        elif self._tap_count == 2:
            # Don't reset yet - wait to see if there's a third tap
            # Return double pinch but keep counting
            pass

        return None

    def check_timeout(self, current_time: float) -> Optional[GestureType]:
        """
        Check if tap window has expired and emit double tap if applicable.

        Should be called periodically to detect double taps when
        no third tap arrives.

        Returns:
            DOUBLE_PINCH if double tap confirmed, None otherwise.
        """
        if self._tap_count == 2:
            time_since_last_tap_ms = (current_time - self._last_tap_time) * 1000
            if time_since_last_tap_ms > self.tap_window_ms:
                # Window expired with 2 taps = double tap
                self._tap_count = 0
                logger.info(f"Double tap detected!")
                return self.double_gesture
        elif self._tap_count == 1:
            # Single tap timeout - just reset
            time_since_last_tap_ms = (current_time - self._last_tap_time) * 1000
            if time_since_last_tap_ms > self.tap_window_ms:
                self._tap_count = 0
        return None

    def reset(self) -> None:
        """Reset detector state."""
        self._tap_count = 0
        self._last_tap_time = 0.0
        self._gesture_start_time = 0.0
        self._is_gesture_active = False


class GestureStateMachine:
    """
    State machine for tracking multiple gestures.

    Handles hold/release confirmation and debouncing to provide
    stable gesture detection with start/hold/end events.
    """

    def __init__(
        self,
        debounce_ms: int = GESTURE_DEBOUNCE_MS,
        hold_frames: int = GESTURE_HOLD_FRAMES,
        release_frames: int = GESTURE_RELEASE_FRAMES,
        gesture_delays: Optional[dict[GestureType, int]] = None  # NEW: per-gesture delays in ms
    ):
        """
        Initialize gesture state machine.

        Args:
            debounce_ms: Minimum time between gesture activations.
            hold_frames: Frames required to confirm gesture (default).
            release_frames: Frames required to confirm release.
            gesture_delays: Optional dict mapping GestureType to activation_delay_ms.
                           If provided, overrides hold_frames for those gestures.
        """
        self.debounce_ms = debounce_ms
        self.hold_frames = hold_frames
        self.release_frames = release_frames

        # Track each gesture type (except generated multi-tap gestures)
        self._tracked: dict[GestureType, TrackedGesture] = {}

        # Multi-tap gestures are generated, not tracked directly
        generated_gestures = {GestureType.DOUBLE_PINCH, GestureType.TRIPLE_PINCH}

        # Per-gesture hold frame overrides (some gestures need different confirmation times)
        gesture_hold_overrides = {
            GestureType.PINCH: GESTURE_HOLD_FRAMES_PINCH,  # Keep quick for intentional pinch
            GestureType.FIST: GESTURE_HOLD_FRAMES_FIST,    # Slower to prevent false triggers
            GestureType.SWIPE_UP: 1,    # Swipes are instant - trigger immediately
            GestureType.SWIPE_DOWN: 1,  # Swipes are instant - trigger immediately
        }

        # Per-gesture release frame overrides
        gesture_release_overrides = {
            GestureType.SWIPE_UP: 1,    # Swipes release immediately
            GestureType.SWIPE_DOWN: 1,  # Swipes release immediately
        }

        # Build gesture_delays dict (profile activationDelayMs or None)
        gesture_delays = gesture_delays or {}

        # Initialize trackers for all gesture types
        for gesture_type in GestureType:
            if gesture_type != GestureType.NONE and gesture_type not in generated_gestures:
                # Priority: gesture_delays (profile) > gesture_hold_overrides (hardcoded) > default hold_frames
                activation_delay_ms = gesture_delays.get(gesture_type, 0)

                # If no profile delay, use hardcoded override or default
                if activation_delay_ms == 0:
                    gesture_hold = gesture_hold_overrides.get(gesture_type, hold_frames)
                    activation_delay_ms = 0  # Will use gesture_hold via hold_frames param
                else:
                    gesture_hold = hold_frames  # Won't be used (activation_delay_ms takes precedence)

                gesture_release = gesture_release_overrides.get(gesture_type, release_frames)

                self._tracked[gesture_type] = TrackedGesture(
                    gesture_type,
                    hold_frames=gesture_hold,
                    release_frames=gesture_release,
                    activation_delay_ms=activation_delay_ms
                )

        # Debounce state
        self._last_activation_time: dict[GestureType, float] = {}

        # Multi-tap detector for pinch gestures
        self._pinch_tap_detector = MultiTapDetector(
            base_gesture=GestureType.PINCH,
            double_gesture=GestureType.DOUBLE_PINCH,
            triple_gesture=GestureType.TRIPLE_PINCH
        )

        # Event callbacks
        self._on_start: Optional[Callable[[GestureEvent], None]] = None
        self._on_end: Optional[Callable[[GestureEvent], None]] = None
        self._on_hold: Optional[Callable[[GestureEvent], None]] = None

        logger.debug(
            f"GestureStateMachine initialized (debounce={debounce_ms}ms, "
            f"hold={hold_frames}, release={release_frames}, "
            f"custom_delays={len(gesture_delays)})"
        )

    def set_callbacks(
        self,
        on_start: Optional[Callable[[GestureEvent], None]] = None,
        on_end: Optional[Callable[[GestureEvent], None]] = None,
        on_hold: Optional[Callable[[GestureEvent], None]] = None
    ) -> None:
        """
        Set event callbacks.

        Args:
            on_start: Called when gesture starts.
            on_end: Called when gesture ends.
            on_hold: Called each frame while gesture is held.
        """
        self._on_start = on_start
        self._on_end = on_end
        self._on_hold = on_hold

    def update(self, result: GestureResult) -> list[GestureEvent]:
        """
        Update state machine with new gesture result.

        Args:
            result: Result from gesture recognizer.

        Returns:
            List of gesture events generated.
        """
        events: list[GestureEvent] = []
        current_time = time.time()

        # Track if PALM was just activated (to cancel FIST)
        palm_activated = False

        # Update each tracked gesture
        for gesture_type, tracker in self._tracked.items():
            # Check if this gesture is currently detected
            is_detected = (result.gesture == gesture_type and result.confidence > 0.5)

            # Priority: POINT > PINCH > FIST/PALM
            # POINT blocks PINCH from starting
            if gesture_type == GestureType.PINCH and is_detected and not tracker.is_active:
                point_tracker = self._tracked.get(GestureType.POINT)
                if point_tracker and point_tracker.is_active:
                    is_detected = False  # Block pinch while pointing

            # PINCH blocks FIST and PALM from starting
            if gesture_type in (GestureType.FIST, GestureType.PALM) and is_detected and not tracker.is_active:
                pinch_tracker = self._tracked.get(GestureType.PINCH)
                if pinch_tracker and pinch_tracker.is_active:
                    is_detected = False  # Block fist/palm while pinching

            # POINT also blocks FIST from starting (kept from before)
            if gesture_type == GestureType.FIST and is_detected and not tracker.is_active:
                point_tracker = self._tracked.get(GestureType.POINT)
                if point_tracker and point_tracker.is_active:
                    is_detected = False  # Block fist while pointing

            # Check debounce
            if is_detected and not tracker.is_active:
                last_activation = self._last_activation_time.get(gesture_type, 0)
                time_since = (current_time - last_activation) * 1000
                if time_since < self.debounce_ms:
                    is_detected = False  # Skip due to debounce

            # Update tracker
            event_type = tracker.update(is_detected, result.cursor_position)

            if event_type:
                event = GestureEvent(
                    gesture=gesture_type,
                    event_type=event_type,
                    cursor_position=result.cursor_position,
                    confidence=result.confidence,
                    timestamp=current_time
                )
                events.append(event)

                # Update debounce time
                if event_type == "start":
                    self._last_activation_time[gesture_type] = current_time
                    # Track PALM activation for gesture cancellation
                    if gesture_type == GestureType.PALM:
                        palm_activated = True

                # Track pinch events for multi-tap detection
                if gesture_type == GestureType.PINCH:
                    if event_type == "start":
                        self._pinch_tap_detector.on_gesture_start(current_time)
                    elif event_type == "end":
                        multi_tap = self._pinch_tap_detector.on_gesture_end(current_time)
                        if multi_tap:
                            # Generate multi-tap event
                            multi_event = GestureEvent(
                                gesture=multi_tap,
                                event_type="start",
                                cursor_position=result.cursor_position,
                                confidence=1.0,
                                timestamp=current_time
                            )
                            events.append(multi_event)
                            self._fire_callback(multi_event)
                            # Also fire "end" immediately (it's an instant gesture)
                            multi_event_end = GestureEvent(
                                gesture=multi_tap,
                                event_type="end",
                                cursor_position=result.cursor_position,
                                confidence=1.0,
                                timestamp=current_time
                            )
                            events.append(multi_event_end)
                            self._fire_callback(multi_event_end)

                # Fire callbacks
                self._fire_callback(event)

        # Check for multi-tap timeout (for double tap detection)
        timeout_multi_tap = self._pinch_tap_detector.check_timeout(current_time)
        if timeout_multi_tap:
            multi_event = GestureEvent(
                gesture=timeout_multi_tap,
                event_type="start",
                cursor_position=result.cursor_position,
                confidence=1.0,
                timestamp=current_time
            )
            events.append(multi_event)
            self._fire_callback(multi_event)
            # Also fire "end" immediately
            multi_event_end = GestureEvent(
                gesture=timeout_multi_tap,
                event_type="end",
                cursor_position=result.cursor_position,
                confidence=1.0,
                timestamp=current_time
            )
            events.append(multi_event_end)
            self._fire_callback(multi_event_end)

        # PALM cancels FIST: Opening hand should immediately end any active fist
        if palm_activated:
            fist_tracker = self._tracked.get(GestureType.FIST)
            if fist_tracker and fist_tracker.is_active:
                fist_tracker.reset()
                end_event = GestureEvent(
                    gesture=GestureType.FIST,
                    event_type="end",
                    cursor_position=result.cursor_position,
                    confidence=1.0,
                    timestamp=current_time
                )
                events.append(end_event)
                self._fire_callback(end_event)
                logger.debug("PALM activated - force-ended FIST")

        return events

    def _fire_callback(self, event: GestureEvent) -> None:
        """Fire appropriate callback for event."""
        try:
            if event.event_type == "start" and self._on_start:
                self._on_start(event)
            elif event.event_type == "end" and self._on_end:
                self._on_end(event)
            elif event.event_type == "hold" and self._on_hold:
                self._on_hold(event)
        except Exception as e:
            logger.error(f"Error in gesture callback: {e}")

    def get_active_gestures(self) -> list[GestureType]:
        """
        Get list of currently active gestures.

        Returns:
            List of active gesture types.
        """
        return [
            gesture_type
            for gesture_type, tracker in self._tracked.items()
            if tracker.is_active
        ]

    def is_gesture_active(self, gesture_type: GestureType) -> bool:
        """
        Check if a specific gesture is active.

        Args:
            gesture_type: Gesture type to check.

        Returns:
            True if gesture is currently active.
        """
        tracker = self._tracked.get(gesture_type)
        return tracker.is_active if tracker else False

    def reset(self) -> None:
        """Reset all tracked gestures."""
        for tracker in self._tracked.values():
            tracker.reset()
        self._last_activation_time.clear()
        self._pinch_tap_detector.reset()
        logger.debug("GestureStateMachine reset")
