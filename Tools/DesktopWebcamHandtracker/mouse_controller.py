"""
Mouse controller for DesktopWebcamHandtracker.

Provides pynput-based mouse control for Windows with cursor smoothing
and gesture-to-action mapping.
"""

import math
import time
import threading
from dataclasses import dataclass
from typing import Optional

try:
    from pynput.mouse import Button, Controller as MouseController
    from pynput.keyboard import Controller as KeyboardController
except ImportError as e:
    raise ImportError(
        "pynput is required. Install with: pip install pynput"
    ) from e

try:
    import win32api
    import win32con
    import win32gui
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

from logger import get_logger
from config import (
    CursorSettings, CURSOR_DEADZONE, SCREEN_MARGIN_PERCENT,
    XREAL_EYE_FOV_HORIZONTAL_DEG, XREAL_EYE_FOV_VERTICAL_DEG, XREAL_EYE_IMU_DEADZONE_RAD,
    ALTERNATE_CURSOR_DEADZONE, ALTERNATE_CURSOR_LINEAR_ZONE, ALTERNATE_CURSOR_MAX_SPEED,
    ALTERNATE_CURSOR_BASE_SENSITIVITY, ALTERNATE_CURSOR_ACCEL_FACTOR
)
from gesture_recognizer import GestureType, GESTURE_NAMES
from gesture_state_machine import GestureEvent
from profile_loader import HandtrackerProfile

# Optional IMU reader import for head motion compensation
try:
    from imu_reader import IMUReader
    HAS_IMU_READER = True
except ImportError:
    IMUReader = None
    HAS_IMU_READER = False

# Optional voice recognition imports
try:
    from voice_recognizer import VoiceRecognizer
    from text_processor import TextProcessor
    from voice_ui_bridge import VoiceUIBridge
    HAS_VOICE_RECOGNITION = True
    VOICE_RECOGNITION_ERROR = None
except ImportError as e:
    VoiceRecognizer = None
    TextProcessor = None
    VoiceUIBridge = None
    HAS_VOICE_RECOGNITION = False
    VOICE_RECOGNITION_ERROR = str(e)

logger = get_logger("MouseController")

# Window drag configuration
WINDOW_DRAG_ACTIVATION_DELAY_SEC = 1.0  # 1 second hold for experimental mode
SWP_NOSIZE = 0x0001
SWP_NOZORDER = 0x0004


class HoldIndicator:
    """Visual indicator that follows cursor when hold is active."""

    # Indicator modes with symbols and colors
    MODES = {
        "hold": {"symbol": "✋", "color": "#00CC00", "outline": "#008800"},  # Green hand for hold
        "drag": {"symbol": "✊", "color": "#0088FF", "outline": "#0066CC"},  # Blue fist for drag
        "wait": {"symbol": "✊", "color": "#888888", "outline": "#666666"},  # Grey fist for waiting
        "point": {"symbol": "👆", "color": "#00CC00", "outline": "#008800"},  # Green point for cursor
        "pinch": {"symbol": "🤏", "color": "#0088FF", "outline": "#0066CC"},  # Blue pinch
        "rightclick_wait": {"symbol": "✋", "color": "#888888", "outline": "#666666"},  # Grey palm waiting
        "rightclick": {"symbol": "✋", "color": "#0088FF", "outline": "#0066CC"},  # Blue palm active
        "voice_wait": {"symbol": "🎤", "color": "#888888", "outline": "#666666"},  # Grey microphone waiting
        "voice": {"symbol": "🎤", "color": "#FF6600", "outline": "#CC4400"},  # Orange microphone for voice
        "error": {"symbol": "❌", "color": "#FF0000", "outline": "#CC0000"},  # Red X for errors
    }

    def __init__(self):
        """Initialize the hold indicator overlay."""
        self._visible = False
        self._root = None
        self._canvas = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._tk_ready = threading.Event()
        self._text_id = None
        self._oval_id = None
        self._current_mode = "hold"

        # Start tkinter in separate thread
        self._tk_thread = threading.Thread(target=self._tk_main, daemon=True)
        self._tk_thread.start()

        # Wait for tkinter to initialize (max 2 seconds)
        self._tk_ready.wait(timeout=2.0)

    def _tk_main(self) -> None:
        """Main tkinter loop (runs in separate thread)."""
        try:
            import tkinter as tk

            self._root = tk.Tk()
            self._root.withdraw()  # Hide initially

            # Configure window
            self._root.overrideredirect(True)  # No title bar
            self._root.attributes('-topmost', True)  # Always on top
            self._root.attributes('-transparentcolor', 'black')  # Black = transparent
            self._root.geometry('32x32+0+0')

            # Create canvas with indicator
            self._canvas = tk.Canvas(
                self._root,
                width=32, height=32,
                bg='black',
                highlightthickness=0
            )
            self._canvas.pack()

            # Draw initial indicator (hold mode)
            mode = self.MODES["hold"]
            self._oval_id = self._canvas.create_oval(2, 2, 30, 30, fill=mode["color"], outline=mode["outline"], width=2)
            self._text_id = self._canvas.create_text(16, 16, text=mode["symbol"], font=('Arial', 12), fill='white')

            self._tk_ready.set()
            logger.debug("Hold indicator initialized")

            # Run tkinter main loop
            self._root.mainloop()
        except Exception as e:
            logger.warning(f"Could not create hold indicator: {e}")
            self._tk_ready.set()  # Signal ready even on failure

    def set_mode(self, mode: str) -> None:
        """Set the indicator mode (changes symbol and color)."""
        if mode not in self.MODES:
            mode = "hold"

        self._current_mode = mode

        if self._canvas is None or self._root is None:
            return

        try:
            mode_config = self.MODES[mode]
            self._root.after(0, lambda: self._update_appearance(mode_config))
        except Exception as e:
            logger.warning(f"Could not set indicator mode: {e}")

    def _update_appearance(self, mode_config: dict) -> None:
        """Update the indicator appearance (must be called from tk thread)."""
        try:
            if self._canvas and self._oval_id and self._text_id:
                self._canvas.itemconfig(self._oval_id, fill=mode_config["color"], outline=mode_config["outline"])
                self._canvas.itemconfig(self._text_id, text=mode_config["symbol"])
        except Exception:
            pass

    def show(self, mode: str = "hold") -> None:
        """Show the indicator and start following cursor."""
        if self._root is None:
            logger.warning(f"HoldIndicator.show({mode}) called but _root is None!")
            return

        # Set mode before showing
        self.set_mode(mode)

        self._visible = True
        self._running = True

        # Start position update thread
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._update_loop, daemon=True)
            self._thread.start()

        try:
            self._root.after(0, self._root.deiconify)
            logger.debug(f"Hold indicator shown (mode={mode})")
        except Exception as e:
            logger.warning(f"Could not show indicator: {e}")

    def hide(self) -> None:
        """Hide the indicator."""
        if self._root is None:
            return

        self._visible = False

        try:
            self._root.after(0, self._root.withdraw)
            logger.debug("Hold indicator hidden")
        except Exception as e:
            logger.warning(f"Could not hide indicator: {e}")

    def _update_loop(self) -> None:
        """Update indicator position to follow cursor."""
        while self._running and self._visible:
            try:
                if self._root and self._visible and HAS_WIN32:
                    # Get cursor position
                    x, y = win32api.GetCursorPos()
                    # Offset to bottom-right of cursor
                    self._root.after(0, lambda px=x, py=y: self._move_to(px, py))
            except Exception:
                pass
            time.sleep(0.033)  # ~30fps update

    def _move_to(self, x: int, y: int) -> None:
        """Move window to position (must be called from tk thread)."""
        try:
            if self._root:
                self._root.geometry(f'+{x + 20}+{y + 20}')
        except Exception:
            pass

    def destroy(self) -> None:
        """Destroy the indicator window."""
        self._running = False
        self._visible = False

        if self._root:
            try:
                self._root.after(0, self._root.quit)
            except Exception:
                pass
            self._root = None


@dataclass
class ScreenInfo:
    """Screen dimensions."""
    width: int
    height: int


class OneEuroFilter:
    """
    One Euro Filter - adaptive low-pass filter for noisy input.

    Adapts smoothing based on signal speed:
    - Slow movement = heavy smoothing (reduces jitter)
    - Fast movement = light smoothing (reduces latency)

    Reference: Casiez et al. "1€ Filter: A Simple Speed-based Low-pass
    Filter for Noisy Input in Interactive Systems" (CHI 2012)
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0
    ):
        """
        Initialize One Euro Filter.

        Args:
            min_cutoff: Minimum cutoff frequency (Hz). Lower = smoother but more lag.
                        Good starting value: 1.0
            beta: Speed coefficient. Higher = more responsive to fast movements.
                  Good starting value: 0.007
            d_cutoff: Derivative cutoff frequency for velocity smoothing.
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[float] = None

    def _smoothing_factor(self, te: float, cutoff: float) -> float:
        """Calculate smoothing factor alpha from cutoff frequency."""
        import math
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, t: float) -> float:
        """
        Apply One Euro Filter to a single value.

        Args:
            x: Input value.
            t: Timestamp in seconds.

        Returns:
            Filtered value.
        """
        if self._x_prev is None:
            self._x_prev = x
            self._t_prev = t
            return x

        te = t - self._t_prev
        if te <= 0:
            return self._x_prev

        # Estimate velocity (derivative)
        dx = (x - self._x_prev) / te

        # Smooth the derivative
        a_d = self._smoothing_factor(te, self.d_cutoff)
        dx_smooth = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Adaptive cutoff based on velocity
        cutoff = self.min_cutoff + self.beta * abs(dx_smooth)

        # Filter the signal
        a = self._smoothing_factor(te, cutoff)
        x_filtered = a * x + (1.0 - a) * self._x_prev

        # Update state
        self._x_prev = x_filtered
        self._dx_prev = dx_smooth
        self._t_prev = t

        return x_filtered

    def reset(self) -> None:
        """Reset filter state."""
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


class CursorSmoother:
    """Smooths cursor movement using One Euro Filter."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        deadzone: float = CURSOR_DEADZONE
    ):
        """
        Initialize cursor smoother.

        Args:
            min_cutoff: Minimum cutoff frequency (lower = smoother).
            beta: Speed coefficient (higher = more responsive).
            deadzone: Ignore movements smaller than this.
        """
        self.deadzone = deadzone

        # Separate filters for X and Y
        self._filter_x = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self._filter_y = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)

        # Track last output for deadzone
        self._last_x: Optional[float] = None
        self._last_y: Optional[float] = None

    def smooth(
        self, x: float, y: float
    ) -> tuple[float, float]:
        """
        Apply One Euro Filter smoothing to cursor position.

        Args:
            x: Raw normalized x position [0, 1].
            y: Raw normalized y position [0, 1].

        Returns:
            Smoothed (x, y) position.
        """
        t = time.time()

        # Apply One Euro Filter
        smooth_x = self._filter_x.filter(x, t)
        smooth_y = self._filter_y.filter(y, t)

        # Apply deadzone check
        if self._last_x is not None:
            dx = smooth_x - self._last_x
            dy = smooth_y - self._last_y
            distance = (dx ** 2 + dy ** 2) ** 0.5
            if distance < self.deadzone:
                return (self._last_x, self._last_y)

        self._last_x = smooth_x
        self._last_y = smooth_y

        return (smooth_x, smooth_y)

    def reset(self) -> None:
        """Reset smoother state."""
        self._filter_x.reset()
        self._filter_y.reset()
        self._last_x = None
        self._last_y = None


class GestureMouseController:
    """
    Controls mouse based on gesture events.

    Maps gestures to mouse actions (click, move, scroll) based on
    profile configuration. Supports optional IMU head motion compensation
    for AR glasses use cases and voice-to-text input.
    """

    def __init__(
        self,
        profile: HandtrackerProfile,
        cursor_settings: Optional[CursorSettings] = None,
        imu_reader: Optional['IMUReader'] = None,
        imu_compensation_enabled: bool = True,
        imu_compensation_scale: float = 2.0  # Radians to normalized coords (tune based on camera FOV)
    ):
        """
        Initialize mouse controller.

        Args:
            profile: Profile with gesture mappings.
            cursor_settings: Cursor movement settings.
            imu_reader: Optional IMU reader for head motion compensation.
                       When provided, head rotation is subtracted from cursor
                       movement to stabilize cursor in head reference frame.
            imu_compensation_enabled: Enable/disable IMU compensation (default: True).
            imu_compensation_scale: Conversion factor from radians to pixels
                                   (default: 2000.0 pixels/radian). Determines
                                   how much cursor moves per radian of head rotation.
        """
        self.profile = profile
        self.cursor_settings = cursor_settings or CursorSettings()

        self._mouse = MouseController()
        self._keyboard = KeyboardController()
        self._smoother = CursorSmoother(
            min_cutoff=self.cursor_settings.min_cutoff,
            beta=self.cursor_settings.beta,
            deadzone=self.cursor_settings.deadzone
        )

        # Get screen dimensions
        self._screen = self._get_screen_info()

        # Sensitivity from profile
        self._sensitivity = profile.mouse_sensitivity * self.cursor_settings.sensitivity

        # IMU head compensation settings
        self._imu_reader = imu_reader
        self._imu_compensation_enabled = imu_compensation_enabled
        self._imu_compensation_scale = imu_compensation_scale

        # Track click states
        self._left_pressed = False
        self._right_pressed = False

        # Scroll state
        self._last_scroll_time = 0.0
        self._scroll_cooldown = 0.3  # seconds

        # Visual indicator for hold state
        self._hold_indicator = HoldIndicator()

        # Track active hold gesture (for HoldPinch, HoldFist, etc.)
        self._active_hold_gesture: Optional[str] = None
        # Track pending hold gesture (waiting to see if it becomes a hold or quick tap)
        self._pending_hold_gesture: Optional[str] = None

        # Alternate cursor mode state (experimental)
        self._alternate_cursor_enabled = profile.alternate_cursor_mode_enabled
        self._cursor_origin_hand: Optional[tuple[float, float]] = None  # Normalized hand position at gesture start
        self._cursor_origin_screen: Optional[tuple[int, int]] = None  # Screen pixel position at gesture start

        # Window drag state (shared by all 3 new actions)
        self._dragged_window: Optional[int] = None  # HWND
        self._drag_start_cursor: Optional[tuple[int, int]] = None
        self._drag_start_window_rect: Optional[tuple[int, int, int, int]] = None  # (x, y, w, h)

        # Experimental action timing
        self._drag_start_time: Optional[float] = None  # time.time() at gesture start
        self._drag_experimental_activated: bool = False  # True after 1-second threshold
        self._drag_hand_origin: Optional[tuple[float, float]] = None  # Hand position at activation

        # Right-click timing (staged delay: 0-2s nothing, 2-4s warning, 4s+ click)
        self._rightclick_start_time: Optional[float] = None
        self._rightclick_activated: bool = False
        self._rightclick_warning_shown: bool = False
        self._rightclick_stop_issued: bool = False  # True after stop command issued in first 0.5s

        # Pinch scroll mode (hold pinch > 1s to scroll)
        self._pinch_start_time: Optional[float] = None
        self._pinch_scroll_mode: bool = False  # True after 1s hold
        self._pinch_scroll_origin: Optional[tuple[float, float]] = None  # Hand position when scroll mode started
        self._pinch_last_scroll_time: float = 0.0  # For scroll rate limiting

        # Voice-to-text state (NEW)
        self._voice_config = profile.voice_recognition
        self._voice_recognizer: Optional[VoiceRecognizer] = None
        self._text_processor: Optional[TextProcessor] = None
        self._voice_ui_bridge: Optional[VoiceUIBridge] = None
        self._voice_recognition_active = False
        self._voice_target_window = None  # Window handle to type into after confirmation
        self._voice_activation_start_time: Optional[float] = None  # For 1-second hold requirement
        self._voice_activation_triggered = False  # True once voice recognition actually started

        # Initialize voice recognition if enabled
        if self._voice_config.enabled:
            if HAS_VOICE_RECOGNITION:
                self._init_voice_recognition()
            else:
                logger.error(f"Voice recognition enabled but dependencies not available: {VOICE_RECOGNITION_ERROR}")
                logger.error("Install with: pip install sherpa-onnx pyaudio")

        imu_status = "enabled" if (imu_reader and imu_compensation_enabled) else "disabled"
        alt_cursor_status = "enabled" if self._alternate_cursor_enabled else "disabled"
        voice_status = "enabled" if self._voice_config.enabled else "disabled"
        if self._voice_config.enabled and not HAS_VOICE_RECOGNITION:
            voice_status = "UNAVAILABLE (missing deps)"
        elif self._voice_config.enabled and not self._voice_recognizer:
            voice_status = "FAILED (init error)"
        logger.info(
            f"MouseController initialized (screen={self._screen.width}x{self._screen.height}, "
            f"sensitivity={self._sensitivity:.2f}, IMU={imu_status}, alt_cursor={alt_cursor_status}, "
            f"voice={voice_status})"
        )

    def _init_voice_recognition(self) -> None:
        """Initialize voice recognition components."""
        try:
            # Create voice recognizer
            device_id = self._voice_config.selected_microphone_device_id
            device_index = int(device_id) if device_id and device_id.isdigit() else -1

            self._voice_recognizer = VoiceRecognizer(
                device_index=device_index,
                silence_timeout=self._voice_config.silence_timeout_seconds
            )

            # Create text processor
            self._text_processor = TextProcessor(
                enable_autocorrect=self._voice_config.enable_autocorrect,
                enable_grammar_fix=self._voice_config.enable_grammar_fix
            )

            # Create UI bridge
            self._voice_ui_bridge = VoiceUIBridge()
            self._voice_ui_bridge.start_server(command_callback=self._handle_voice_command)

            logger.info("Voice recognition initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize voice recognition: {e}")
            self._voice_recognizer = None
            self._text_processor = None
            self._voice_ui_bridge = None

    def _get_screen_info(self) -> ScreenInfo:
        """Get screen dimensions."""
        if HAS_WIN32:
            width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
            height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
        else:
            # Fallback to common resolution
            width = 1920
            height = 1080

        return ScreenInfo(width=width, height=height)

    def handle_event(self, event: GestureEvent) -> None:
        """
        Handle a gesture event.

        Args:
            event: Gesture event from state machine.
        """
        # Use GESTURE_NAMES to get the camelCase name matching JSON profile format
        gesture_name = GESTURE_NAMES.get(event.gesture, event.gesture.name.lower())

        # Get action for this gesture
        action = self.profile.get_action(gesture_name)

        # For "hold" events, check for "Hold" variant (e.g., holdPinch for pinch)
        # This allows holdPinch to trigger on sustained pinch gestures
        # Capitalize first letter to match JSON format: "pinch" -> "holdPinch"
        hold_gesture_name = f"hold{gesture_name[0].upper()}{gesture_name[1:]}" if gesture_name else ""
        hold_action = self.profile.get_action(hold_gesture_name)

        # Track if we should suppress the regular action
        suppress_regular_action = False

        # FIX: Actions that require continuous hold events must bypass the hold logic
        # These actions need the full event stream (start → hold → end) to work correctly
        # If we let hold logic intercept, it sends only start+end (quick tap fallback)
        CONTINUOUS_ACTIONS = {"windowdrag", "windowdragexperimental", "movecursor", "holdleftmouse", "rightclick", "leftclick", "voicetotextactivation"}
        action_lower = action.lower() if action else ""

        if action_lower in CONTINUOUS_ACTIONS:
            # Disable hold interception - let continuous action handle all events directly
            logger.debug(f"Bypassing hold logic for continuous action: {action}")
            hold_action = None

        if hold_action and hold_action != "none":
            # Handle the Hold variant - convert hold/end events to start/end
            if event.event_type == "start":
                # Mark that we're waiting to see if this becomes a hold
                # Track the gesture that started (for later comparison)
                self._pending_hold_gesture = hold_gesture_name
                # Suppress the regular "start" action - we'll handle it on "end" if no hold occurred
                suppress_regular_action = True
                logger.debug(f"Waiting for potential hold: {hold_gesture_name}")

            elif event.event_type == "hold":
                # First hold event triggers "start", subsequent ones are ignored
                if self._active_hold_gesture != hold_gesture_name:
                    self._active_hold_gesture = hold_gesture_name
                    self._pending_hold_gesture = None  # No longer pending, it's a hold
                    logger.debug(f"Hold gesture started: {hold_gesture_name} -> {hold_action}")
                    self._dispatch_action(hold_action, "start", event)
                # Always suppress regular action during hold
                suppress_regular_action = True

            elif event.event_type == "end":
                if self._active_hold_gesture == hold_gesture_name:
                    # End the active hold gesture
                    logger.debug(f"Hold gesture ended: {hold_gesture_name} -> {hold_action}")
                    self._dispatch_action(hold_action, "end", event)
                    self._active_hold_gesture = None
                    suppress_regular_action = True
                elif self._pending_hold_gesture == hold_gesture_name:
                    # Quick tap - no hold occurred, execute regular action as click
                    self._pending_hold_gesture = None
                    if action and action != "none":
                        logger.debug(f"Quick tap detected, executing: {gesture_name} -> {action}")
                        # Execute both start and end for a click
                        self._dispatch_action(action, "start", event)
                        self._dispatch_action(action, "end", event)
                    suppress_regular_action = True

        # Handle the regular gesture action (if not suppressed by hold logic)
        if not suppress_regular_action and action and action != "none":
            logger.debug(f"Handling {event.event_type} for {gesture_name} -> {action}")
            self._dispatch_action(action, event.event_type, event)

    def _dispatch_action(self, action: str, event_type: str, event: GestureEvent) -> None:
        """Dispatch action based on event type."""
        # Create a modified event with the specified event_type
        class ModifiedEvent:
            def __init__(self, original, new_type):
                self.gesture = original.gesture
                self.event_type = new_type
                self.cursor_position = original.cursor_position
                self.confidence = original.confidence

        modified = ModifiedEvent(event, event_type)

        # Handle different actions (case-insensitive)
        action_lower = action.lower()
        if action_lower == "movecursor":
            self._handle_move_cursor(modified)
        elif action_lower == "leftclick":
            self._handle_left_click(modified)
        elif action_lower == "rightclick":
            self._handle_right_click(modified)
        elif action_lower == "scrollup":
            self._handle_scroll_up(modified)
        elif action_lower == "scrolldown":
            self._handle_scroll_down(modified)
        elif action_lower == "doubleclick":
            self._handle_double_click(modified)
        elif action_lower == "middleclick":
            self._handle_middle_click(modified)
        elif action_lower == "drag":
            self._handle_drag(modified)
        elif action_lower == "lefthold":
            self._handle_left_hold(modified)
        elif action_lower == "leftrelease":
            self._handle_left_release(modified)
        elif action_lower == "holdleftmouse":
            self._handle_hold_left_mouse(modified)
        elif action_lower == "windowdrag":
            self._handle_window_drag(modified)
        elif action_lower == "windowdragstart":
            self._handle_window_drag_start(modified)
        elif action_lower == "windowdragend":
            self._handle_window_drag_end(modified)
        elif action_lower == "windowdragexperimental":
            self._handle_window_drag_experimental(modified)
        elif action_lower == "voicetotextactivation":
            self._handle_voice_to_text_activation(modified)

    def _handle_voice_to_text_activation(self, event: GestureEvent) -> None:
        """
        Handle voice-to-text activation action with 1-second hold requirement.

        Flow:
        - START: Begin timing, show grey waiting indicator
        - HOLD: After 1 second, start voice recognition, show orange indicator
        - END: Reset timing state, hide indicator if not yet activated
               If activated, recognition CONTINUES until user confirms/dismisses
               or silence timeout occurs. UI window controls the lifecycle.
        """
        if not self._voice_recognizer or not self._text_processor:
            # Voice recognition not available - show error feedback
            if event.event_type == "start":
                if not HAS_VOICE_RECOGNITION:
                    error_msg = f"Voice recognition dependencies not installed: {VOICE_RECOGNITION_ERROR}"
                elif not self._voice_config.enabled:
                    error_msg = "Voice recognition is disabled in profile settings"
                else:
                    error_msg = "Voice recognition failed to initialize"
                logger.warning(f"Voice recognition not available: {error_msg}")
                # Show error indicator briefly
                self._hold_indicator.show("error")
            elif event.event_type == "end":
                self._hold_indicator.hide()
                self._voice_activation_start_time = None
                self._voice_activation_triggered = False
            return

        if event.event_type == "start":
            # Diagnostic logging for debugging activation issues
            recognizer_listening = self._voice_recognizer.is_listening() if self._voice_recognizer else False
            bridge_connected = self._voice_ui_bridge.is_connected() if self._voice_ui_bridge else False
            indicator_ok = self._hold_indicator._root is not None if self._hold_indicator else False
            logger.info(f"Voice activation START - states: active={self._voice_recognition_active}, "
                       f"recognizer_listening={recognizer_listening}, bridge_connected={bridge_connected}, "
                       f"indicator_ok={indicator_ok}")

            # If recognizer is still listening from a previous session (e.g., window was closed via X button
            # without sending cancel), force stop it before starting a new session
            if recognizer_listening:
                logger.info("Recognizer still listening from previous session - forcing stop")
                self._stop_voice_recognition()
                # Update flag after stop
                recognizer_listening = False

            # Check if recognition is marked active but recognizer isn't actually running
            # This can happen when window closes via X button (no cancel sent to Python)
            if self._voice_recognition_active:
                # After the force-stop above, recognizer should not be listening
                if recognizer_listening:
                    logger.debug("Voice recognition actually active, ignoring start")
                    return
                else:
                    # Stale state - reset and allow new activation
                    logger.info("Voice recognition state was stale, resetting")
                    self._voice_recognition_active = False
                    self._hold_indicator.hide()

            # Start timing for 1-second hold requirement
            self._voice_activation_start_time = time.time()
            self._voice_activation_triggered = False
            self._hold_indicator.show("voice_wait")  # Grey microphone - waiting
            logger.debug("Voice activation: waiting for 1-second hold...")

        elif event.event_type == "hold":
            # Check if we're in the timing phase
            if self._voice_activation_start_time is not None and not self._voice_activation_triggered:
                elapsed = time.time() - self._voice_activation_start_time

                # After 1 second, trigger voice recognition
                if elapsed >= 1.0:
                    self._voice_activation_triggered = True
                    logger.info(f"Voice activation: 1-second hold completed, starting recognition")

                    # Start voice recognition
                    try:
                        # Save the current foreground window BEFORE showing our UI
                        # This is where we'll type the text later
                        self._voice_target_window = None
                        if HAS_WIN32:
                            try:
                                self._voice_target_window = win32gui.GetForegroundWindow()
                                window_title = win32gui.GetWindowText(self._voice_target_window) or "(no title)"
                                logger.info(f"Saved target window for voice typing: {self._voice_target_window} '{window_title}'")
                            except Exception as e:
                                logger.warning(f"Failed to get foreground window: {e}")

                        self._voice_recognition_active = True
                        self._voice_recognizer.start_recognition()
                        self._hold_indicator.set_mode("voice")  # Orange microphone - active
                        logger.info("Voice recognition started (gesture activation)")

                        # Notify UI - show voice input window
                        if self._voice_ui_bridge:
                            sent = self._voice_ui_bridge.send_voice_input_started()
                            if not sent:
                                logger.warning("Failed to send voice_input_started to WPF (not connected)")
                        else:
                            logger.warning("Voice UI bridge not available")

                    except Exception as e:
                        logger.error(f"Failed to start voice recognition: {e}")
                        self._voice_recognition_active = False
                        self._voice_activation_triggered = False
                        self._hold_indicator.hide()
                        if self._voice_ui_bridge:
                            self._voice_ui_bridge.send_error(f"Failed to start: {e}")

            # Send partial transcription updates if recognition is active
            elif self._voice_recognition_active and self._voice_recognizer:
                partial_text = self._voice_recognizer.get_partial_result()
                if partial_text and self._voice_ui_bridge:
                    self._voice_ui_bridge.send_partial_transcription(partial_text)

        elif event.event_type == "end":
            # Reset timing state
            was_triggered = self._voice_activation_triggered
            self._voice_activation_start_time = None
            self._voice_activation_triggered = False

            if was_triggered:
                # Gesture ended after activation - hide indicator but KEEP recognition running
                # Recognition continues until user confirms/dismisses or silence timeout
                self._hold_indicator.hide()
                logger.debug("Voice gesture ended - recognition continues until user action")
            else:
                # Gesture ended before 1-second threshold - cancel
                self._hold_indicator.hide()
                logger.debug("Voice gesture ended before 1-second threshold - cancelled")

    def _is_text_input_focused(self) -> bool:
        """
        Check if a text input control is currently focused.

        Uses Windows API to detect if the focused control is an edit box,
        rich edit, or other text input control.

        Returns:
            True if a text input appears to be focused, False otherwise.
        """
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32

            # Get the foreground window
            foreground_hwnd = user32.GetForegroundWindow()
            if not foreground_hwnd:
                return False

            # Get the window title to check if it's our own Launcher
            length = user32.GetWindowTextLengthW(foreground_hwnd) + 1
            buffer = ctypes.create_unicode_buffer(length)
            user32.GetWindowTextW(foreground_hwnd, buffer, length)
            window_title = buffer.value

            # If the Launcher window is focused, no valid text field
            if "AROverlay" in window_title or "Voice Input" in window_title:
                logger.debug(f"Launcher window is focused: '{window_title}'")
                return False

            # Get the focused control within the foreground window
            # GetFocus returns the handle of the focused control in the current thread
            # For cross-thread, we need to attach to the foreground thread
            foreground_thread_id = user32.GetWindowThreadProcessId(foreground_hwnd, None)
            current_thread_id = ctypes.windll.kernel32.GetCurrentThreadId()

            # Attach to the foreground thread to get the real focused control
            attached = False
            if foreground_thread_id != current_thread_id:
                attached = user32.AttachThreadInput(current_thread_id, foreground_thread_id, True)

            try:
                focused_hwnd = user32.GetFocus()
                if not focused_hwnd:
                    # No control focused, but a window is - might be a browser or app
                    # Be permissive and allow typing
                    logger.debug("No specific control focused, but window exists - allowing")
                    return True

                # Get the class name of the focused control
                class_buffer = ctypes.create_unicode_buffer(256)
                user32.GetClassNameW(focused_hwnd, class_buffer, 256)
                class_name = class_buffer.value.lower()

                logger.debug(f"Focused control class: '{class_name}'")

                # Common text input class names
                text_input_classes = [
                    'edit',           # Standard Windows edit control
                    'richedit',       # Rich text edit
                    'richedit20',     # Rich edit 2.0
                    'richedit50w',    # Rich edit 5.0
                    'textedit',       # Generic text edit
                    'textarea',       # HTML textarea
                    'input',          # HTML input
                    'chrome_widget',  # Chrome/Chromium input
                    'mozilla',        # Firefox
                    'scintilla',      # Code editors
                    'afx:',           # MFC edit controls
                    'windowsforme',   # .NET forms
                    'textbox',        # Various textbox controls
                ]

                for text_class in text_input_classes:
                    if text_class in class_name:
                        logger.debug(f"Text input detected: {class_name}")
                        return True

                # For browsers and modern apps, the class name might not be obvious
                # Check if it's a known application window that supports typing
                known_typing_windows = [
                    'chrome', 'firefox', 'edge', 'opera', 'brave',  # Browsers
                    'notepad', 'wordpad', 'word', 'excel',  # Office apps
                    'code', 'vscode', 'sublime', 'atom',  # Code editors
                    'discord', 'slack', 'teams', 'telegram', 'whatsapp',  # Chat apps
                    'explorer',  # Windows Explorer rename
                ]

                window_title_lower = window_title.lower()
                for known_app in known_typing_windows:
                    if known_app in window_title_lower or known_app in class_name:
                        logger.debug(f"Known typing app detected: {window_title}")
                        return True

                # If we got here and there's a focused control, be permissive
                # Modern apps use custom controls that don't match standard class names
                logger.debug(f"Unknown control type '{class_name}' in '{window_title}' - allowing")
                return True

            finally:
                if attached:
                    user32.AttachThreadInput(current_thread_id, foreground_thread_id, False)

        except Exception as e:
            logger.error(f"Error checking focused control: {e}")
            # On error, be permissive and allow typing
            return True

    def _handle_voice_command(self, message: dict) -> None:
        """
        Handle command from WPF UI bridge.

        Args:
            message: Command message dictionary.
        """
        command_type = message.get("type", "")

        if command_type == "confirm_text":
            # Type the confirmed text
            confirmed_text = message.get("text", "")
            press_enter = message.get("press_enter", False)

            if not confirmed_text:
                logger.warning("No text to type")
                self._send_typing_result(False, "No text to type")
                return

            logger.info(f"Typing confirmed text: '{confirmed_text}' (press_enter={press_enter})")
            try:
                # Stop voice recognition before typing (frees up resources)
                self._stop_voice_recognition()

                # Restore focus to the original target window before typing
                target_restored = False
                if HAS_WIN32 and self._voice_target_window:
                    try:
                        # Check if window still exists
                        if win32gui.IsWindow(self._voice_target_window):
                            win32gui.SetForegroundWindow(self._voice_target_window)
                            time.sleep(0.1)  # Brief delay for focus to settle
                            target_restored = True
                            logger.info(f"Restored focus to target window: {self._voice_target_window}")
                        else:
                            logger.warning("Target window no longer exists")
                    except Exception as e:
                        logger.warning(f"Failed to restore target window focus: {e}")

                if not target_restored:
                    # Fallback: check if any text input is focused
                    if not self._is_text_input_focused():
                        logger.warning("No text input focused and couldn't restore target window")
                        self._send_typing_result(False, "Original text field is no longer available. Please click on a text field and try again.")
                        return

                self._keyboard.type(confirmed_text)
                if press_enter:
                    # Wait 300ms then press Enter
                    time.sleep(0.3)
                    from pynput.keyboard import Key
                    self._keyboard.press(Key.enter)
                    self._keyboard.release(Key.enter)
                    logger.info("Pressed Enter after text")

                self._send_typing_result(True, "Text typed successfully")
            except Exception as e:
                logger.error(f"Failed to type text: {e}")
                self._send_typing_result(False, f"Failed to type: {str(e)}")
                self._stop_voice_recognition()

        elif command_type == "cancel_recognition":
            # Cancel ongoing recognition
            logger.info("Voice recognition cancelled by user")
            self._stop_voice_recognition()

        elif command_type == "autocorrect_request":
            # Handle autocorrect request from UI
            text = message.get("text", "")
            if text and self._text_processor:
                try:
                    corrected = self._text_processor.autocorrect(text)
                    if self._voice_ui_bridge:
                        self._voice_ui_bridge.send_autocorrect_result(corrected)
                except Exception as e:
                    logger.error(f"Autocorrect error: {e}")

    def _stop_voice_recognition(self) -> None:
        """Stop voice recognition and reset state."""
        logger.info(f"_stop_voice_recognition called - current active={self._voice_recognition_active}")
        if self._voice_recognizer:
            try:
                self._voice_recognizer.stop_recognition()
                logger.info(f"Voice recognizer stopped, is_listening now={self._voice_recognizer.is_listening()}")
            except Exception as e:
                logger.error(f"Error stopping voice recognition: {e}")
        self._voice_recognition_active = False
        self._hold_indicator.hide()
        logger.info("Voice recognition state reset complete")

    def _send_typing_result(self, success: bool, message: str) -> None:
        """
        Send typing result back to WPF UI.

        Args:
            success: Whether typing was successful.
            message: Result message to display.
        """
        if self._voice_ui_bridge and self._voice_ui_bridge.is_connected:
            try:
                self._voice_ui_bridge.send_typing_result(success, message)
            except Exception as e:
                logger.error(f"Failed to send typing result: {e}")
                self._voice_recognition_active = False
                self._hold_indicator.hide()

    def _handle_move_cursor(self, event: GestureEvent) -> None:
        """Move cursor based on hand position with optional IMU head compensation."""
        # Show/hide point indicator
        if event.event_type == "start":
            self._hold_indicator.show("point")  # Green point up
        elif event.event_type == "end":
            self._hold_indicator.hide()
            return

        # Delegate to alternate cursor mode if enabled
        if self._alternate_cursor_enabled:
            self._handle_alternate_cursor(event)
            return

        if event.cursor_position is None:
            return

        if event.event_type not in ("start", "hold"):
            return

        # Get smoothed position
        raw_x, raw_y = event.cursor_position
        smooth_x, smooth_y = self._smoother.smooth(raw_x, raw_y)

        # Apply sensitivity by scaling movement from center
        # Sensitivity > 1 = more movement, < 1 = less movement
        center_x, center_y = 0.5, 0.5
        scaled_x = center_x + (smooth_x - center_x) * self._sensitivity
        scaled_y = center_y + (smooth_y - center_y) * self._sensitivity

        # Apply IMU head compensation (if enabled and available)
        # Stabilizes cursor position when user moves head (ego-motion compensation)
        if self._imu_compensation_enabled and self._imu_reader:
            try:
                # Get accumulated head rotation since last call (in radians)
                delta_yaw, delta_pitch = self._imu_reader.get_head_rotation_delta()

                # XREAL Eye uses opposite sign convention - negate to match expected behavior
                # When head turns LEFT, we need NEGATIVE delta to shift cursor LEFT
                delta_yaw = -delta_yaw
                delta_pitch = -delta_pitch

                # Apply deadzone to ignore micro head movements (jitter reduction)
                if abs(delta_yaw) < XREAL_EYE_IMU_DEADZONE_RAD:
                    delta_yaw = 0.0
                if abs(delta_pitch) < XREAL_EYE_IMU_DEADZONE_RAD:
                    delta_pitch = 0.0

                # FOV-based compensation formula:
                # offset = delta_radians / FOV_radians
                # This converts head rotation to normalized image coordinates
                #
                # XREAL Eye FOV: ~120° horizontal, ~90° vertical
                fov_h_rad = math.radians(XREAL_EYE_FOV_HORIZONTAL_DEG)  # ~2.09 rad
                fov_v_rad = math.radians(XREAL_EYE_FOV_VERTICAL_DEG)    # ~1.57 rad

                # Calculate compensation offset in normalized coords
                offset_x = delta_yaw / fov_h_rad    # Scale ~0.48
                offset_y = delta_pitch / fov_v_rad  # Scale ~0.64

                # Apply compensation - with negated delta and += operator
                old_x, old_y = scaled_x, scaled_y
                scaled_x += offset_x
                scaled_y += offset_y

                # Log significant IMU compensation
                if abs(delta_yaw) > 0.02 or abs(delta_pitch) > 0.02:
                    logger.info(
                        f"IMU: yaw={delta_yaw:.3f} pitch={delta_pitch:.3f} "
                        f"offset=({offset_x:.3f},{offset_y:.3f}) "
                        f"pos: {old_x:.2f},{old_y:.2f} -> {scaled_x:.2f},{scaled_y:.2f}"
                    )
            except Exception as e:
                logger.warning(f"IMU compensation failed: {e}")

        # Map to screen coordinates with margin
        margin = self.cursor_settings.screen_margin

        # Invert X (mirror) and apply margin
        screen_x = int(
            (1.0 - scaled_x) * self._screen.width * (1.0 - 2 * margin) +
            self._screen.width * margin
        )
        screen_y = int(
            scaled_y * self._screen.height * (1.0 - 2 * margin) +
            self._screen.height * margin
        )

        # Clamp to screen bounds
        screen_x = max(0, min(self._screen.width - 1, screen_x))
        screen_y = max(0, min(self._screen.height - 1, screen_y))

        # Move cursor
        self._mouse.position = (screen_x, screen_y)

        # If a window is captured for dragging, update its position too
        if self._dragged_window is not None:
            self._update_dragged_window_position()

    def _handle_alternate_cursor(self, event: GestureEvent) -> None:
        """
        Handle cursor movement in alternate (relative) mode.

        In this mode, cursor position is relative to where the Point gesture started:
        - Small hand movements = proportional cursor movements
        - Large hand movements = accelerated cursor movements
        - Deadzone around origin prevents jitter

        Algorithm:
          ON "start": Capture hand position and mouse cursor as origin
          ON "hold": Calculate delta from origin, apply speed curve, move cursor
          ON "end": Clear origin state

        Args:
            event: Gesture event with cursor_position (normalized 0-1).
        """
        if event.event_type == "start":
            # Capture origin positions
            if event.cursor_position is not None:
                self._cursor_origin_hand = event.cursor_position
                # Get current mouse cursor position as screen origin
                if HAS_WIN32:
                    self._cursor_origin_screen = win32api.GetCursorPos()
                else:
                    self._cursor_origin_screen = self._mouse.position
                logger.debug(
                    f"Alternate cursor: origin captured at hand={self._cursor_origin_hand}, "
                    f"screen={self._cursor_origin_screen}"
                )
            return

        if event.event_type == "end":
            # Clear origin state
            self._cursor_origin_hand = None
            self._cursor_origin_screen = None
            logger.debug("Alternate cursor: origin cleared")
            return

        if event.event_type != "hold":
            return

        # Handle "hold" event - move cursor relative to origin
        if event.cursor_position is None or self._cursor_origin_hand is None or self._cursor_origin_screen is None:
            return

        raw_x, raw_y = event.cursor_position
        origin_x, origin_y = self._cursor_origin_hand
        screen_origin_x, screen_origin_y = self._cursor_origin_screen

        # Calculate delta from origin (in normalized coords)
        delta_x = raw_x - origin_x
        delta_y = raw_y - origin_y

        # Calculate distance from origin
        distance = math.sqrt(delta_x ** 2 + delta_y ** 2)

        # Apply deadzone - no movement if within deadzone
        if distance < ALTERNATE_CURSOR_DEADZONE:
            return

        # Calculate direction (unit vector)
        if distance > 0:
            dir_x = delta_x / distance
            dir_y = delta_y / distance
        else:
            return

        # Calculate speed based on distance zones
        linear_max_speed = ALTERNATE_CURSOR_LINEAR_ZONE * ALTERNATE_CURSOR_BASE_SENSITIVITY

        if distance < ALTERNATE_CURSOR_LINEAR_ZONE:
            # Linear zone: speed proportional to distance
            speed = distance * ALTERNATE_CURSOR_BASE_SENSITIVITY
        else:
            # Acceleration zone: quadratic scaling beyond linear zone
            excess = distance - ALTERNATE_CURSOR_LINEAR_ZONE
            speed = linear_max_speed + (excess ** 2) * ALTERNATE_CURSOR_ACCEL_FACTOR * ALTERNATE_CURSOR_BASE_SENSITIVITY

        # Clamp to max speed
        speed = min(speed, ALTERNATE_CURSOR_MAX_SPEED)

        # Apply sensitivity multiplier
        speed *= self._sensitivity

        # Calculate movement vector
        move_x = dir_x * speed
        move_y = dir_y * speed

        # Flip X direction (hand mirror effect)
        move_x = -move_x

        # Calculate new cursor position (accumulate from screen origin)
        new_x = int(screen_origin_x + move_x)
        new_y = int(screen_origin_y + move_y)

        # Clamp to screen bounds
        new_x = max(0, min(self._screen.width - 1, new_x))
        new_y = max(0, min(self._screen.height - 1, new_y))

        # Move cursor
        self._mouse.position = (new_x, new_y)

        # Update screen origin to new position for continuous motion
        self._cursor_origin_screen = (new_x, new_y)

        # If a window is captured for dragging, update its position too
        if self._dragged_window is not None:
            self._update_dragged_window_position()

    def _handle_left_click(self, event: GestureEvent) -> None:
        """
        Handle left click with scroll mode.

        - 0-1s: Normal click behavior (press on start, release on end)
        - 1s+: Scroll mode - hand movement controls scrolling
               Up/Down = vertical scroll, Left/Right = horizontal scroll
        """
        if event.event_type == "start":
            self._pinch_start_time = time.time()
            self._pinch_scroll_mode = False
            self._pinch_scroll_origin = None
            self._hold_indicator.show("pinch")  # Blue pinch
            self._mouse.press(Button.left)
            self._left_pressed = True
            logger.debug("Left click started (scroll mode activates after 1s)")

        elif event.event_type == "hold":
            if self._pinch_start_time is None:
                return

            elapsed = time.time() - self._pinch_start_time

            # After 1 second, switch to scroll mode
            if not self._pinch_scroll_mode and elapsed >= 1.0:
                self._pinch_scroll_mode = True
                # Release the click - we're now in scroll mode
                if self._left_pressed:
                    self._mouse.release(Button.left)
                    self._left_pressed = False
                # Capture hand position as scroll origin
                if event.cursor_position:
                    self._pinch_scroll_origin = event.cursor_position
                self._hold_indicator.set_mode("hold")  # Green hand for scroll mode
                logger.info("Pinch scroll mode activated")

            # In scroll mode, translate hand movement to scrolling
            if self._pinch_scroll_mode and event.cursor_position and self._pinch_scroll_origin:
                current_time = time.time()
                # Rate limit scrolling (every 50ms = 20 scroll events/sec max)
                if current_time - self._pinch_last_scroll_time >= 0.05:
                    hand_x, hand_y = event.cursor_position
                    origin_x, origin_y = self._pinch_scroll_origin

                    # Calculate delta from origin
                    delta_x = hand_x - origin_x
                    delta_y = hand_y - origin_y

                    # Threshold for scroll activation (prevent jitter)
                    scroll_threshold = 0.02  # 2% of screen movement

                    scroll_x = 0
                    scroll_y = 0

                    # Vertical scroll (Y movement) - inverted because screen Y is inverted
                    if abs(delta_y) > scroll_threshold:
                        # Scale: larger movement = faster scroll
                        scroll_amount = int(delta_y * 30)  # Adjust multiplier for sensitivity
                        scroll_y = -scroll_amount  # Negative because move down = scroll down

                    # Horizontal scroll (X movement) - inverted for mirror
                    if abs(delta_x) > scroll_threshold:
                        scroll_amount = int(delta_x * 30)
                        scroll_x = -scroll_amount  # Inverted for mirror effect

                    if scroll_x != 0 or scroll_y != 0:
                        self._mouse.scroll(scroll_x, scroll_y)
                        self._pinch_last_scroll_time = current_time

        elif event.event_type == "end":
            was_scroll_mode = self._pinch_scroll_mode
            self._hold_indicator.hide()

            # If we ended before scroll mode, release the click
            if self._left_pressed:
                self._mouse.release(Button.left)
                self._left_pressed = False
                logger.debug("Left click released (quick tap)")

            # Reset scroll mode state
            self._pinch_start_time = None
            self._pinch_scroll_mode = False
            self._pinch_scroll_origin = None

            if was_scroll_mode:
                logger.debug("Pinch scroll mode ended")

    def _handle_right_click(self, event: GestureEvent) -> None:
        """
        Handle right click action with 4-second staged delay.

        - 0-0.5s: STOP command - cancels any ongoing actions (drag, hold, etc.)
        - 0-2s: No indicator (waiting)
        - 2-4s: Grey palm indicator (warning)
        - 4s+: Right-click executes, blue palm indicator
        - END: Hide indicator, reset state
        """
        if event.event_type == "start":
            self._rightclick_start_time = time.time()
            self._rightclick_activated = False
            self._rightclick_warning_shown = False
            self._rightclick_stop_issued = False
            logger.debug("Right click: waiting (4s staged delay)")

        elif event.event_type == "hold":
            if self._rightclick_start_time is None:
                return

            elapsed = time.time() - self._rightclick_start_time

            # 0-0.5s: Issue STOP command (cancel all ongoing actions)
            if not self._rightclick_stop_issued and elapsed < 0.5:
                self._rightclick_stop_issued = True
                # Release mouse buttons
                if self._left_pressed:
                    self._mouse.release(Button.left)
                    self._left_pressed = False
                    logger.info("STOP: Released left mouse button")
                # Clear any window drag state
                if self._dragged_window is not None:
                    logger.info(f"STOP: Cancelled window drag (hwnd={self._dragged_window})")
                    self._clear_drag_state()
                # Stop voice recognition if active
                if self._voice_recognition_active and self._voice_recognizer:
                    logger.info("STOP: Cancelled voice recognition")
                    self._voice_recognizer.stop_recognition()
                    self._voice_recognition_active = False
                    self._hold_indicator.hide()
                logger.info("Palm STOP gesture issued - all actions cancelled")

            # 2-4s: Show grey palm warning
            if not getattr(self, '_rightclick_warning_shown', False) and elapsed >= 2.0:
                self._rightclick_warning_shown = True
                self._hold_indicator.show("rightclick_wait")  # Grey palm
                logger.debug("Right click: warning indicator shown")

            # 4s+: Execute right click
            if not self._rightclick_activated and elapsed >= 4.0:
                self._rightclick_activated = True
                self._hold_indicator.set_mode("rightclick")  # Blue palm
                self._mouse.click(Button.right)
                logger.info("Right click executed after 4s hold")

        elif event.event_type == "end":
            self._hold_indicator.hide()
            self._rightclick_start_time = None
            self._rightclick_activated = False
            self._rightclick_warning_shown = False
            self._rightclick_stop_issued = False
            logger.debug("Right click gesture ended")

    def _handle_double_click(self, event: GestureEvent) -> None:
        """Handle double click action."""
        if event.event_type == "start":
            self._mouse.click(Button.left, 2)
            logger.debug("Double clicked")

    def _handle_middle_click(self, event: GestureEvent) -> None:
        """Handle middle click action."""
        if event.event_type == "start":
            self._mouse.click(Button.middle)
            logger.debug("Middle clicked")

    def _handle_scroll_up(self, event: GestureEvent) -> None:
        """Handle scroll up action."""
        if event.event_type == "start":
            current_time = time.time()
            if current_time - self._last_scroll_time >= self._scroll_cooldown:
                self._mouse.scroll(0, 3)  # Scroll up
                self._last_scroll_time = current_time
                logger.debug("Scrolled up")

    def _handle_scroll_down(self, event: GestureEvent) -> None:
        """Handle scroll down action."""
        if event.event_type == "start":
            current_time = time.time()
            if current_time - self._last_scroll_time >= self._scroll_cooldown:
                self._mouse.scroll(0, -3)  # Scroll down
                self._last_scroll_time = current_time
                logger.debug("Scrolled down")

    def _handle_drag(self, event: GestureEvent) -> None:
        """Handle drag action (click and hold while moving)."""
        if event.event_type == "start":
            self._mouse.press(Button.left)
            self._left_pressed = True
            logger.debug("Drag started")
        elif event.event_type == "hold" and event.cursor_position:
            # Move while dragging
            self._handle_move_cursor(event)
        elif event.event_type == "end":
            if self._left_pressed:
                self._mouse.release(Button.left)
                self._left_pressed = False
                logger.debug("Drag ended")

    def _handle_left_hold(self, event: GestureEvent) -> None:
        """Handle left hold action (press and hold left button for drag)."""
        if event.event_type == "start":
            if not self._left_pressed:
                self._mouse.press(Button.left)
                self._left_pressed = True
                self._hold_indicator.show("drag")  # Show drag indicator (blue hand)
                logger.debug("Left hold (drag) started")

    def _handle_left_release(self, event: GestureEvent) -> None:
        """Handle left release action (release left button)."""
        if event.event_type == "start":
            if self._left_pressed:
                self._mouse.release(Button.left)
                self._left_pressed = False
                self._hold_indicator.hide()  # Hide visual indicator
                logger.debug("Left hold released (palm)")

    def _handle_hold_left_mouse(self, event: GestureEvent) -> None:
        """Handle hold left mouse action (auto-release when gesture ends)."""
        if event.event_type == "start":
            if not self._left_pressed:
                self._mouse.press(Button.left)
                self._left_pressed = True
                self._hold_indicator.show("hold")  # Show hold indicator (green fist)
                logger.debug("Hold left mouse started")
        elif event.event_type == "end":
            if self._left_pressed:
                self._mouse.release(Button.left)
                self._left_pressed = False
                self._hold_indicator.hide()  # Hide visual indicator
                logger.debug("Hold left mouse ended")

    def _handle_window_drag(self, event: GestureEvent) -> None:
        """
        Handle window drag action (grab and move any window from anywhere).

        Uses Win32 API trick: sends WM_NCLBUTTONDOWN with HTCAPTION to make
        Windows think the user clicked the title bar, enabling drag from any
        part of the window.

        State flow:
        - start: Find window, press mouse button, trigger Windows drag loop
        - hold: Continue cursor tracking (Windows handles window movement automatically)
        - end: Release mouse button (Windows detects and ends drag)

        Security Note (F-005):
        UIPI (User Interface Privilege Isolation) prevents this from affecting
        elevated/UAC-protected windows. Only user-level windows at same or
        lower integrity can be manipulated.

        Requires:
        - pywin32 (already in requirements.txt)
        - Windows OS (graceful fallback on other platforms)
        - Physical mouse button press (Windows drag loop requires it)

        Args:
            event: Gesture event with event_type ("start", "hold", "end").
        """
        if not HAS_WIN32:
            logger.warning("Window drag requires pywin32 on Windows platform")
            return

        if event.event_type == "start":
            button_pressed = False  # Track if we pressed the button for cleanup
            try:
                # Move cursor to hand position first (enables cursor control during grab)
                if event.cursor_position:
                    self._handle_move_cursor(event)

                # Get current cursor position (after move)
                cursor_pos = win32api.GetCursorPos()

                # Find window handle under cursor
                hwnd = win32gui.WindowFromPoint(cursor_pos)
                if not hwnd:
                    logger.debug("No window found under cursor for drag")
                    return

                # F-011: Filter out desktop window to prevent selection rectangle
                desktop_hwnd = win32gui.GetDesktopWindow()
                if hwnd == desktop_hwnd:
                    logger.debug("Cannot drag desktop window")
                    return

                # Get root window (not child controls like buttons/textboxes)
                root_hwnd = win32gui.GetAncestor(hwnd, win32con.GA_ROOT)
                if not root_hwnd:
                    root_hwnd = hwnd  # Fallback to original if no parent

                # F-008: Press mouse button ONLY after successful window validation
                self._mouse.press(Button.left)
                self._left_pressed = True
                button_pressed = True

                # Release any existing mouse capture
                win32api.ReleaseCapture()

                # Send non-client left button down on caption area
                # F-005: UIPI (User Interface Privilege Isolation) prevents this from
                # affecting elevated/UAC-protected windows. Only user-level windows
                # at same or lower integrity can be manipulated.
                result = win32gui.PostMessage(
                    root_hwnd,
                    win32con.WM_NCLBUTTONDOWN,
                    win32con.HTCAPTION,
                    0
                )

                # FIX-2: Validate PostMessage succeeded
                if not result:
                    logger.error(f"PostMessage failed for window {root_hwnd} - may be elevated/protected")
                    # Cleanup - release button since drag failed
                    if self._left_pressed:
                        self._mouse.release(Button.left)
                        self._left_pressed = False
                    return

                # F-013: Show visual indicator after successful drag start
                self._hold_indicator.show("drag")

                logger.info(f"Window drag started successfully (hwnd={root_hwnd})")

            except Exception as e:
                logger.warning(f"Failed to initiate window drag: {e}")
                # Clean up if we already pressed the button
                if button_pressed and self._left_pressed:
                    self._mouse.release(Button.left)
                    self._left_pressed = False
                    self._hold_indicator.hide()

        elif event.event_type == "hold":
            # F-001: Handle cursor tracking loss to avoid stuck state
            if event.cursor_position:
                # Continue tracking cursor position
                # Windows automatically moves the window based on physical mouse movements
                self._handle_move_cursor(event)
            else:
                # Cursor tracking lost - end window drag to avoid stuck state
                logger.warning("Cursor tracking lost during window drag, ending drag")
                if self._left_pressed:
                    self._mouse.release(Button.left)
                    self._left_pressed = False
                    self._hold_indicator.hide()

        elif event.event_type == "end":
            # F-014: Hide visual indicator on drag end
            if self._left_pressed:
                self._mouse.release(Button.left)
                self._left_pressed = False
                self._hold_indicator.hide()
                logger.debug("Window drag ended")

    def _clear_drag_state(self) -> None:
        """Clear all window drag state."""
        self._dragged_window = None
        self._drag_start_cursor = None
        self._drag_start_window_rect = None
        self._drag_start_time = None
        self._drag_experimental_activated = False
        self._drag_hand_origin = None
        self._hold_indicator.hide()

    def _capture_window_under_cursor(self, event: GestureEvent, move_cursor: bool = False) -> bool:
        """
        Capture the window under cursor for dragging.

        Args:
            event: Gesture event with cursor position.
            move_cursor: If True, move cursor to hand position before capturing.
                        Use False for discrete actions to avoid cursor jump.

        Returns:
            True if window captured successfully, False otherwise.
        """
        if not HAS_WIN32:
            return False

        try:
            # Optionally move cursor to hand position first (for continuous actions)
            if move_cursor and event.cursor_position:
                self._handle_move_cursor(event)

            cursor_pos = win32api.GetCursorPos()
            hwnd = win32gui.WindowFromPoint(cursor_pos)

            if not hwnd:
                logger.debug("No window found under cursor")
                return False

            # Filter desktop window
            if hwnd == win32gui.GetDesktopWindow():
                logger.debug("Cannot drag desktop window")
                return False

            # Get root window (not child controls)
            root_hwnd = win32gui.GetAncestor(hwnd, win32con.GA_ROOT) or hwnd

            # Get window rect
            rect = win32gui.GetWindowRect(root_hwnd)
            window_rect = (rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])

            # Store state
            self._dragged_window = root_hwnd
            self._drag_start_cursor = cursor_pos
            self._drag_start_window_rect = window_rect

            self._hold_indicator.show("drag")
            logger.info(f"Window captured for drag (hwnd={root_hwnd})")
            return True

        except Exception as e:
            logger.warning(f"Failed to capture window: {e}")
            return False

    def _update_dragged_window_position(self) -> None:
        """Update position of dragged window based on current cursor."""
        if not self._dragged_window or not HAS_WIN32:
            return

        try:
            if not win32gui.IsWindow(self._dragged_window):
                logger.warning("Dragged window no longer exists")
                self._clear_drag_state()
                return

            cursor_pos = win32api.GetCursorPos()
            dx = cursor_pos[0] - self._drag_start_cursor[0]
            dy = cursor_pos[1] - self._drag_start_cursor[1]

            x0, y0, w, h = self._drag_start_window_rect

            win32gui.SetWindowPos(
                self._dragged_window,
                None,
                x0 + dx,
                y0 + dy,
                0, 0,
                SWP_NOSIZE | SWP_NOZORDER
            )
        except Exception as e:
            logger.warning(f"Failed to move window: {e}")

    def _handle_window_drag_start(self, event: GestureEvent) -> None:
        """Handle discrete window drag START action."""
        if event.event_type != "start":
            return

        self._capture_window_under_cursor(event)

    def _handle_window_drag_end(self, event: GestureEvent) -> None:
        """Handle discrete window drag END action."""
        if event.event_type != "start":
            return

        if self._dragged_window:
            logger.info(f"Window drag ended (hwnd={self._dragged_window})")
        self._clear_drag_state()

    def _handle_window_drag_experimental(self, event: GestureEvent) -> None:
        """
        Handle experimental window drag with 1-second hold activation.

        - START: Record timestamp, show waiting indicator
        - HOLD: After 1 second, capture window and begin dragging
        - END: Release window
        """
        if not HAS_WIN32:
            return

        if event.event_type == "start":
            self._drag_start_time = time.time()
            self._drag_experimental_activated = False
            self._hold_indicator.show("wait")  # Grey fist during wait
            logger.debug("Experimental drag: waiting for 1s hold")

        elif event.event_type == "hold":
            if not self._drag_start_time:
                return

            # Check if 1 second has passed
            elapsed = time.time() - self._drag_start_time

            if not self._drag_experimental_activated:
                if elapsed >= WINDOW_DRAG_ACTIVATION_DELAY_SEC:
                    # Activation threshold reached - capture window at CURRENT cursor position
                    # Don't move cursor (move_cursor=False) to avoid jump
                    if self._capture_window_under_cursor(event, move_cursor=False):
                        self._drag_experimental_activated = True
                        # Store hand position at activation for relative movement
                        if event.cursor_position:
                            self._drag_hand_origin = event.cursor_position
                        logger.info("Experimental drag activated after 1s hold")
                    else:
                        # Capture failed, clear state
                        self._clear_drag_state()
            else:
                # Already activated - use RELATIVE hand movement to update window
                if event.cursor_position and self._drag_hand_origin:
                    # Calculate hand delta from activation point
                    hand_x, hand_y = event.cursor_position
                    origin_x, origin_y = self._drag_hand_origin
                    delta_x = hand_x - origin_x
                    delta_y = hand_y - origin_y

                    # Convert normalized delta to screen pixels (inverted X for mirror)
                    # Scale by screen size and sensitivity
                    pixel_delta_x = int(-delta_x * self._screen.width * self._sensitivity)
                    pixel_delta_y = int(delta_y * self._screen.height * self._sensitivity)

                    # Move window directly based on delta from capture position
                    if self._dragged_window and self._drag_start_window_rect:
                        x0, y0, w, h = self._drag_start_window_rect
                        try:
                            win32gui.SetWindowPos(
                                self._dragged_window,
                                None,
                                x0 + pixel_delta_x,
                                y0 + pixel_delta_y,
                                0, 0,
                                SWP_NOSIZE | SWP_NOZORDER
                            )
                        except Exception as e:
                            logger.warning(f"Failed to move window: {e}")
                elif not event.cursor_position:
                    # Tracking lost during drag
                    logger.warning("Tracking lost during experimental drag")
                    self._clear_drag_state()

        elif event.event_type == "end":
            if self._drag_experimental_activated:
                logger.info(f"Experimental drag ended (hwnd={self._dragged_window})")
            self._clear_drag_state()

    def release_all(self) -> None:
        """Release any pressed buttons and clear drag state."""
        if self._left_pressed:
            self._mouse.release(Button.left)
            self._left_pressed = False
            self._hold_indicator.hide()
        if self._right_pressed:
            self._mouse.release(Button.right)
            self._right_pressed = False
        self._clear_drag_state()

        # Stop voice recognition if active
        if self._voice_recognition_active and self._voice_recognizer:
            self._voice_recognizer.stop_recognition()
            self._voice_recognition_active = False

    def reset(self) -> None:
        """Reset controller state."""
        self.release_all()
        self._smoother.reset()
        self._active_hold_gesture = None
        self._pending_hold_gesture = None
        # Clear alternate cursor state
        self._cursor_origin_hand = None
        self._cursor_origin_screen = None
        # Clear rightclick state
        self._rightclick_start_time = None
        self._rightclick_activated = False
        self._rightclick_warning_shown = False
        self._rightclick_stop_issued = False

    def destroy(self) -> None:
        """Clean up resources."""
        self.release_all()
        self._hold_indicator.destroy()

        # Clean up voice recognition
        if self._voice_recognizer:
            self._voice_recognizer.cleanup()
        if self._voice_ui_bridge:
            self._voice_ui_bridge.stop_server()
