#!/usr/bin/env python3
"""
Desktop Webcam Hand Tracker

Main entry point for the DesktopWebcamHandtracker module.
Provides gesture-based mouse control using MediaPipe hand tracking.

Usage:
    python desktop_webcam_handtracker.py --profile <path> [--camera <index>] [--debug]

Exit Codes:
    0 - Success
    1 - Profile error
    2 - Camera error
    3 - Runtime error
"""

import argparse
import os
import signal
import sys
import threading
import time
from typing import Optional, Union

import cv2

from config import (
    EXIT_SUCCESS,
    EXIT_PROFILE_ERROR,
    EXIT_CAMERA_ERROR,
    EXIT_RUNTIME_ERROR,
    DEFAULT_CAMERA_INDEX,
    XREAL_EYE_MODEL_COMPLEXITY,
    XREAL_EYE_MIN_DETECTION_CONFIDENCE,
    XREAL_EYE_MIN_TRACKING_CONFIDENCE,
    XREAL_EYE_MIN_PRESENCE_CONFIDENCE,
    XREAL_EYE_USE_IMAGE_MODE,
    HAND_PRESENCE_REQUIRED_FRAMES,
    HAND_LOST_RESET_FRAMES,
    HAND_MIN_CONFIDENCE,
)
from logger import setup_logging, get_logger
from profile_loader import load_profile, ProfileLoadError, HandtrackerProfile
from camera_manager import CameraManager, CameraError, select_camera
from hand_detector import HandDetector
from gesture_recognizer import GestureRecognizer, GestureType
from gesture_state_machine import GestureStateMachine, GestureEvent
from mouse_controller import GestureMouseController

# Optional XREAL Eye imports
try:
    from xreal_eye_camera_source import XREALEyeCameraSource, CameraError as XREALCameraError
    HAS_XREAL_EYE = True
except ImportError:
    XREALEyeCameraSource = None
    XREALCameraError = CameraError
    HAS_XREAL_EYE = False

# Optional landmark smoother import (Phase 2 Enhancement)
try:
    from landmark_smoother import LandmarkSmoother, LandmarkSmootherConfig
    HAS_LANDMARK_SMOOTHER = True
except ImportError:
    LandmarkSmoother = None
    LandmarkSmootherConfig = None
    HAS_LANDMARK_SMOOTHER = False

# Optional ROI tracker import (Phase 2 Enhancement 3)
try:
    from hand_detector import ROITracker, ROIConfig
    HAS_ROI_TRACKER = True
except ImportError:
    ROITracker = None
    ROIConfig = None
    HAS_ROI_TRACKER = False

# Optional IMU reader import
try:
    from imu_reader import IMUReader
    HAS_IMU_READER = True
except ImportError:
    IMUReader = None
    HAS_IMU_READER = False

# Optional dual-camera modules (Phase 7: Dual Camera Tracker)
try:
    from phone_landmark_receiver import PhoneLandmarkReceiver, setup_adb_port_forward
    from camera_fusion import CameraFusion, FusedHandData, FusionMethod
    from triangulation import PhoneCalibration
    from hand_ekf import HandEKF
    from dual_view_gesture_recognizer import DualViewGestureRecognizer
    HAS_DUAL_CAMERA = True
except ImportError as e:
    PhoneLandmarkReceiver = None
    CameraFusion = None
    PhoneCalibration = None
    HandEKF = None
    DualViewGestureRecognizer = None
    HAS_DUAL_CAMERA = False


# Parent process heartbeat check interval (seconds)
PARENT_CHECK_INTERVAL = 2.0


class ParentProcessMonitor:
    """
    Monitors if the parent process (Launcher) is still alive.

    If the parent process terminates, triggers a callback to shut down
    this process gracefully, then forces exit if shutdown takes too long.
    This prevents orphaned hand tracker processes.
    """

    # Timeout for graceful shutdown before forcing exit
    GRACEFUL_SHUTDOWN_TIMEOUT = 3.0

    # Grace period before monitoring starts (allows for initialization)
    STARTUP_GRACE_PERIOD = 15.0

    def __init__(self, parent_pid: int, on_parent_exit: callable):
        """
        Initialize parent process monitor.

        Args:
            parent_pid: PID of the parent process to monitor.
            on_parent_exit: Callback to invoke when parent exits.
        """
        self._parent_pid = parent_pid
        self._on_parent_exit = on_parent_exit
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._logger = get_logger("ParentMonitor")
        self._startup_time: Optional[float] = None

    def start(self) -> None:
        """Start monitoring the parent process."""
        if self._running:
            return

        self._running = True
        self._startup_time = time.time()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        self._logger.info(f"Monitoring parent process PID {self._parent_pid} (grace period: {self.STARTUP_GRACE_PERIOD}s)")

    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _monitor_loop(self) -> None:
        """Background loop that checks if parent is alive."""
        # Wait for grace period before starting to monitor
        self._logger.debug(f"Waiting {self.STARTUP_GRACE_PERIOD}s grace period before monitoring...")
        grace_wait = 0.0
        while self._running and grace_wait < self.STARTUP_GRACE_PERIOD:
            time.sleep(1.0)
            grace_wait += 1.0

        if not self._running:
            return

        self._logger.info("Grace period ended, now actively monitoring parent process")

        while self._running:
            if not self._is_parent_alive():
                self._logger.warning(f"Parent process {self._parent_pid} no longer exists - shutting down")
                self._force_exit()
                break
            time.sleep(PARENT_CHECK_INTERVAL)

    def _force_exit(self) -> None:
        """Force process exit after attempting graceful shutdown."""
        # Try graceful shutdown first
        try:
            self._on_parent_exit()
        except Exception as e:
            self._logger.error(f"Error during graceful shutdown: {e}")

        # Give graceful shutdown a chance
        time.sleep(self.GRACEFUL_SHUTDOWN_TIMEOUT)

        # If still running, force exit
        self._logger.warning("Graceful shutdown timeout - forcing exit")
        os._exit(0)  # Hard exit, bypasses finally blocks

    def _is_parent_alive(self) -> bool:
        """Check if the parent process is still running (Windows-compatible)."""
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32

            # OpenProcess with PROCESS_QUERY_LIMITED_INFORMATION (0x1000)
            # Returns handle if process exists, 0 if not
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, self._parent_pid)

            if handle:
                try:
                    # Check if process is still active
                    STILL_ACTIVE = 259
                    exit_code = ctypes.c_ulong()
                    if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                        return exit_code.value == STILL_ACTIVE
                    return False
                finally:
                    kernel32.CloseHandle(handle)
            return False
        except Exception as e:
            self._logger.error(f"Error checking parent process: {e}")
            return False


class HandTrackerApp:
    """
    Main application for desktop hand tracking.

    Integrates camera capture, hand detection, gesture recognition,
    and mouse control into a real-time tracking loop.

    Supports both standard webcam and XREAL Eye camera sources with
    optional IMU-based head motion compensation.
    """

    def __init__(
        self,
        profile: HandtrackerProfile,
        camera_index: int = DEFAULT_CAMERA_INDEX,
        debug: bool = False
    ):
        """
        Initialize hand tracker application.

        Args:
            profile: Loaded profile configuration.
            camera_index: Camera device index (ignored for XREAL Eye).
            debug: Enable debug mode with visualization.
        """
        self.profile = profile
        self.camera_index = camera_index
        self.debug = debug

        self._logger = get_logger("App")
        self._running = False

        # Components (camera source is polymorphic: CameraManager or XREALEyeCameraSource)
        self._camera: Optional[Union[CameraManager, 'XREALEyeCameraSource']] = None
        self._detector: Optional[HandDetector] = None
        self._recognizer: Optional[GestureRecognizer] = None
        self._state_machine: Optional[GestureStateMachine] = None
        self._mouse_controller: Optional[GestureMouseController] = None
        self._imu_reader: Optional['IMUReader'] = None
        self._landmark_smoother: Optional['LandmarkSmoother'] = None
        self._roi_tracker: Optional['ROITracker'] = None

        # Dual-camera components (Phase 7)
        self._phone_receiver: Optional['PhoneLandmarkReceiver'] = None
        self._camera_fusion: Optional['CameraFusion'] = None
        self._hand_ekf: Optional['HandEKF'] = None
        self._dual_recognizer: Optional['DualViewGestureRecognizer'] = None

        # Stats
        self._frame_count = 0
        self._start_time = 0.0
        self._last_fps_time = 0.0
        self._fps = 0.0

        # Hand presence tracking (prevents false gestures when no hand)
        self._consecutive_detections = 0
        self._consecutive_misses = 0
        self._hand_present = False
        self._min_confidence = HAND_MIN_CONFIDENCE  # May be lowered for XREAL Eye

    def initialize(self) -> None:
        """Initialize all components."""
        self._logger.info("Initializing hand tracker...")

        # Initialize camera source based on profile
        self._initialize_camera()

        # Initialize IMU reader (if enabled and XREAL Eye)
        self._initialize_imu()

        # Initialize hand detector with appropriate settings for camera source
        if self.profile.camera_source_type == "XrealEye":
            # XREAL Eye: use Phase 2 enhanced settings for 4-bit greyscale
            self._logger.info(
                f"Using XREAL Eye settings (complexity={XREAL_EYE_MODEL_COMPLEXITY}, "
                f"confidence={XREAL_EYE_MIN_DETECTION_CONFIDENCE}, "
                f"image_mode={XREAL_EYE_USE_IMAGE_MODE})"
            )
            self._detector = HandDetector(
                model_complexity=XREAL_EYE_MODEL_COMPLEXITY,
                min_detection_confidence=XREAL_EYE_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=XREAL_EYE_MIN_TRACKING_CONFIDENCE,
                use_image_mode=XREAL_EYE_USE_IMAGE_MODE
            )
            # XREAL Eye: use Phase 2 confidence threshold (0.05 with preprocessing)
            self._min_confidence = XREAL_EYE_MIN_DETECTION_CONFIDENCE
            self._logger.info(f"Hand presence min confidence: {self._min_confidence} (XREAL Eye Phase 2)")
        else:
            self._detector = HandDetector()
            self._logger.info(f"Hand presence min confidence: {self._min_confidence} (standard mode)")
        self._detector.initialize()

        # Initialize landmark smoother (Phase 2 Enhancement 2)
        self._initialize_landmark_smoother()

        # Initialize ROI tracker (Phase 2 Enhancement 3)
        self._initialize_roi_tracker()

        # Initialize dual-camera components (Phase 7)
        self._initialize_dual_camera()

        # Initialize gesture recognizer
        self._recognizer = GestureRecognizer()

        # Initialize state machine
        self._state_machine = GestureStateMachine()

        # Initialize mouse controller (pass IMU reader if available)
        self._mouse_controller = GestureMouseController(
            profile=self.profile,
            imu_reader=self._imu_reader
        )

        # Set up state machine callbacks
        self._state_machine.set_callbacks(
            on_start=self._on_gesture_start,
            on_end=self._on_gesture_end,
            on_hold=self._on_gesture_hold
        )

        self._logger.info("Hand tracker initialized")

    def _initialize_camera(self) -> None:
        """
        Initialize camera source based on profile configuration.

        Creates either a CameraManager (for webcam) or XREALEyeCameraSource
        (for XREAL Eye glasses) based on profile.camera_source_type.

        Raises:
            CameraError: If camera initialization fails.
        """
        camera_source_type = self.profile.camera_source_type

        if camera_source_type == "XrealEye":
            # XREAL Eye camera source
            if not HAS_XREAL_EYE:
                raise CameraError(
                    "XREAL Eye camera source not available. "
                    "Missing xreal_eye_camera_source.py module."
                )

            self._logger.info("Using XREAL Eye camera source")
            self._camera = XREALEyeCameraSource(
                flip_horizontal=self.profile.flip_horizontal
            )

            try:
                self._camera.open()
                self._logger.info(
                    f"XREAL Eye connected: {self._camera.actual_width}x{self._camera.actual_height}"
                )
            except XREALCameraError as e:
                raise CameraError(
                    f"Failed to connect to XREAL Eye camera: {e}\n\n"
                    "Troubleshooting:\n"
                    "  1. Check XREAL One Pro glasses are connected via USB\n"
                    "  2. Verify TCP connection to 169.254.2.1:52997 is accessible\n"
                    "  3. Ensure no other application is using the XREAL Eye camera\n"
                    "  4. Try unplugging and reconnecting the glasses"
                )

        else:
            # Standard webcam source
            self._logger.info(f"Using webcam camera source (index={self.camera_index})")
            self._camera = CameraManager(camera_index=self.camera_index)
            self._camera.open()

    def _initialize_imu(self) -> None:
        """
        Initialize IMU reader for head motion compensation (XREAL Eye only).

        Creates and starts IMU reader if profile enables IMU compensation
        and camera source is XREAL Eye.
        """
        # IMU only applies to XREAL Eye source
        if self.profile.camera_source_type != "XrealEye":
            return

        # Check if IMU compensation is enabled
        if not self.profile.imu_head_compensation_enabled:
            self._logger.info("IMU head compensation disabled in profile")
            return

        # Check if IMU reader module is available
        if not HAS_IMU_READER:
            self._logger.warning(
                "IMU head compensation enabled but imu_reader.py module not available. "
                "Continuing without head compensation."
            )
            return

        try:
            self._logger.info("Starting IMU reader for head motion compensation...")
            self._imu_reader = IMUReader()
            self._imu_reader.start()

            # Wait briefly for connection
            time.sleep(0.5)

            if self._imu_reader.is_running:
                self._logger.info("IMU reader started successfully")
            else:
                self._logger.warning("IMU reader failed to start, continuing without compensation")
                self._imu_reader = None

        except Exception as e:
            self._logger.warning(
                f"Failed to initialize IMU reader: {e}\n"
                "Continuing without head motion compensation."
            )
            self._imu_reader = None

    def _initialize_landmark_smoother(self) -> None:
        """
        Initialize landmark smoother for temporal filtering (Phase 2 Enhancement 2).

        Creates LandmarkSmoother if the module is available.
        Uses different settings for XREAL Eye vs webcam.
        """
        if not HAS_LANDMARK_SMOOTHER:
            self._logger.debug("Landmark smoother not available")
            return

        try:
            if self.profile.camera_source_type == "XrealEye":
                # XREAL Eye: more smoothing needed due to low-quality input
                self._landmark_smoother = LandmarkSmoother(LandmarkSmootherConfig(
                    enabled=True,
                    min_cutoff=0.5,  # More smoothing for noisy 4-bit input
                    beta=0.3         # Less responsive to reduce jitter
                ))
                self._logger.info("Landmark smoother enabled (XREAL Eye mode)")
            else:
                # Webcam: less smoothing needed
                self._landmark_smoother = LandmarkSmoother(LandmarkSmootherConfig(
                    enabled=True,
                    min_cutoff=1.2,  # Less smoothing
                    beta=0.3
                ))
                self._logger.info("Landmark smoother enabled (Webcam mode)")

        except Exception as e:
            self._logger.warning(f"Failed to initialize landmark smoother: {e}")
            self._landmark_smoother = None

    def _initialize_roi_tracker(self) -> None:
        """
        Initialize ROI tracker for spatial optimization (Phase 2 Enhancement 3).

        Creates ROITracker for XREAL Eye camera source to focus detection
        on the region where the hand was last seen.
        """
        if not HAS_ROI_TRACKER:
            self._logger.debug("ROI tracker not available")
            return

        # Only enable ROI for XREAL Eye (where detection is more challenging)
        if self.profile.camera_source_type != "XrealEye":
            self._logger.debug("ROI tracker disabled (webcam mode)")
            return

        try:
            self._roi_tracker = ROITracker(ROIConfig())  # Uses config default
            if self._roi_tracker.is_enabled:
                self._logger.info("ROI tracker enabled (XREAL Eye mode)")
            else:
                self._logger.info("ROI tracker disabled in config")
        except Exception as e:
            self._logger.warning(f"Failed to initialize ROI tracker: {e}")
            self._roi_tracker = None

    def _initialize_dual_camera(self) -> None:
        """
        Initialize dual-camera components for phone + webcam tracking (Phase 7).

        Creates PhoneLandmarkReceiver, CameraFusion, HandEKF, and DualViewGestureRecognizer
        if phone camera is enabled in the profile.
        """
        if not self.profile.phone_camera_enabled:
            self._logger.debug("Dual-camera mode disabled in profile")
            return

        if not HAS_DUAL_CAMERA:
            self._logger.warning(
                "Dual-camera mode enabled but modules not available. "
                "Missing: phone_landmark_receiver, camera_fusion, triangulation, "
                "hand_ekf, or dual_view_gesture_recognizer"
            )
            return

        try:
            # Setup ADB port forwarding for USB connection
            if self.profile.phone_connection_type == "usb":
                self._logger.info("Setting up ADB port forwarding for USB connection...")
                if not setup_adb_port_forward(self.profile.phone_port):
                    self._logger.warning(
                        "ADB port forwarding failed. Make sure phone is connected "
                        "and USB debugging is enabled."
                    )
                    # Continue anyway - user might have set it up manually

            # Initialize phone landmark receiver
            host = "127.0.0.1"  # localhost for USB (ADB forwarded)
            if self.profile.phone_connection_type == "wifi":
                if self.profile.phone_wifi_ip:
                    host = self.profile.phone_wifi_ip
                else:
                    self._logger.warning("WiFi mode enabled but no phone IP configured")
                    return

            self._phone_receiver = PhoneLandmarkReceiver(
                host=host,
                port=self.profile.phone_port
            )
            self._phone_receiver.start()
            self._logger.info(f"Phone landmark receiver started: {host}:{self.profile.phone_port}")

            # Initialize camera fusion (requires calibration)
            if self.profile.phone_calibration:
                calib_data = self.profile.phone_calibration
                import numpy as np
                phone_calib = PhoneCalibration(
                    rotation=np.array(calib_data.rotation).reshape(3, 3),
                    translation=np.array(calib_data.translation),
                    phone_fx=calib_data.phone_fx,
                    phone_fy=calib_data.phone_fy,
                    phone_cx=calib_data.phone_cx,
                    phone_cy=calib_data.phone_cy,
                    reprojection_error=calib_data.reprojection_error,
                    calibration_timestamp=calib_data.calibration_date
                )
                self._camera_fusion = CameraFusion(phone_calib)
                self._logger.info(
                    f"Camera fusion initialized (calibration error: "
                    f"{calib_data.reprojection_error:.2f}px)"
                )
            else:
                self._logger.warning(
                    "Phone camera enabled but no calibration data. "
                    "Run the calibration wizard to enable stereo tracking."
                )
                # Can still run single-camera with phone fallback

            # Initialize Extended Kalman Filter for position smoothing
            self._hand_ekf = HandEKF()
            self._logger.info("Hand EKF initialized")

            # Initialize dual-view gesture recognizer
            self._dual_recognizer = DualViewGestureRecognizer()
            self._logger.info("Dual-view gesture recognizer initialized")

            self._logger.info("Dual-camera mode initialized successfully")

        except Exception as e:
            self._logger.error(f"Failed to initialize dual-camera mode: {e}")
            self._cleanup_dual_camera()

    def _cleanup_dual_camera(self) -> None:
        """Clean up dual-camera components."""
        if self._phone_receiver:
            self._phone_receiver.stop()
            self._phone_receiver = None
        self._camera_fusion = None
        self._hand_ekf = None
        self._dual_recognizer = None

    def _on_gesture_start(self, event: GestureEvent) -> None:
        """Handle gesture start event."""
        self._logger.debug(f"Gesture started: {event.gesture.name}")
        if self._mouse_controller:
            self._mouse_controller.handle_event(event)

    def _on_gesture_end(self, event: GestureEvent) -> None:
        """Handle gesture end event."""
        self._logger.debug(f"Gesture ended: {event.gesture.name}")
        if self._mouse_controller:
            self._mouse_controller.handle_event(event)

    def _on_gesture_hold(self, event: GestureEvent) -> None:
        """Handle gesture hold event."""
        if self._mouse_controller:
            self._mouse_controller.handle_event(event)

    def run(self) -> None:
        """Run the main tracking loop."""
        self._running = True
        self._start_time = time.perf_counter()
        self._last_fps_time = self._start_time

        self._logger.info("Starting tracking loop...")

        try:
            while self._running:
                self._process_frame()

                # Check for quit key in debug mode
                if self.debug:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # q or ESC
                        self._logger.info("Quit key pressed")
                        break
        except KeyboardInterrupt:
            self._logger.info("Interrupted by user")
        finally:
            self.stop()

    def _process_frame(self) -> None:
        """Process a single frame."""
        if self._camera is None or self._detector is None:
            return

        # Capture frame
        frame = self._camera.read_frame_rgb()
        if frame is None:
            self._logger.warning("Failed to capture frame")
            return

        self._frame_count += 1

        # Get ROI for focused detection (Phase 2 Enhancement 3)
        detection_frame = frame
        roi_offset = (0, 0)
        if self._roi_tracker:
            detection_frame, roi_offset = self._roi_tracker.get_roi(frame)

        # Detect hand (on cropped ROI if available)
        landmarks = self._detector.detect(detection_frame)

        # Update ROI and scale landmarks to full frame coordinates
        if self._roi_tracker:
            frame_size = (frame.shape[0], frame.shape[1])  # (h, w)
            landmarks = self._roi_tracker.update(landmarks, frame_size, roi_offset)

        # Apply landmark smoothing (Phase 2 Enhancement 2)
        if landmarks and self._landmark_smoother:
            landmarks = self._landmark_smoother.smooth(landmarks)

        # ----- Hand Presence Confirmation -----
        # Check if detection meets minimum confidence threshold (camera-specific)
        valid_detection = landmarks is not None and landmarks.score >= self._min_confidence

        if valid_detection:
            self._consecutive_detections += 1
            self._consecutive_misses = 0

            # Confirm hand presence after enough consecutive detections
            if not self._hand_present and self._consecutive_detections >= HAND_PRESENCE_REQUIRED_FRAMES:
                self._hand_present = True
                self._logger.debug(f"Hand presence confirmed after {self._consecutive_detections} frames")
        else:
            self._consecutive_misses += 1
            self._consecutive_detections = 0

            # Reset hand presence and gesture state after enough consecutive misses
            if self._hand_present and self._consecutive_misses >= HAND_LOST_RESET_FRAMES:
                self._hand_present = False
                if self._state_machine:
                    self._state_machine.reset()
                self._logger.debug(f"Hand lost after {self._consecutive_misses} misses - gesture state reset")

        # Only process gestures when hand is confirmed present
        if self._hand_present and landmarks and self._recognizer and self._state_machine:
            # ----- Phase 7: Dual-Camera Fusion -----
            phone_landmarks = None
            if self._phone_receiver and self._phone_receiver.is_connected:
                phone_landmarks = self._phone_receiver.get_latest()

            # Use dual-view gesture recognizer if available and phone connected
            if phone_landmarks and self._dual_recognizer:
                result = self._dual_recognizer.recognize(landmarks, phone_landmarks)
                self._logger.debug(
                    f"Dual-view gesture: {result.gesture.name} "
                    f"(confidence: {result.confidence:.2f})"
                )
            else:
                # Single-camera mode (original behavior)
                result = self._recognizer.recognize(landmarks)

            # Update state machine
            self._state_machine.update(result)

            # Debug visualization
            if self.debug:
                self._show_debug_frame(frame, landmarks, result.gesture)
        elif self.debug:
            # Show frame without detection (or hand not confirmed)
            self._show_debug_frame(frame, landmarks if valid_detection else None, GestureType.NONE)

        # Update FPS
        self._update_fps()

    def _show_debug_frame(
        self,
        frame,
        landmarks,
        gesture: GestureType
    ) -> None:
        """Show debug visualization window."""
        # Convert to BGR for OpenCV display
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        if landmarks and self._detector:
            display = self._detector.draw_landmarks(display, landmarks)

        # Draw gesture text
        gesture_text = f"Gesture: {gesture.name}"
        cv2.putText(
            display, gesture_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )

        # Draw FPS
        fps_text = f"FPS: {self._fps:.1f}"
        cv2.putText(
            display, fps_text, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

        # Draw camera source info
        camera_type = self.profile.camera_source_type
        camera_info = f"Camera: {camera_type}"
        if self._imu_reader:
            camera_info += " (IMU enabled)"
        cv2.putText(
            display, camera_info, (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        # Mirror the display (more intuitive for hand tracking)
        display = cv2.flip(display, 1)

        cv2.imshow("Hand Tracker Debug", display)

    def _update_fps(self) -> None:
        """Update FPS calculation."""
        current_time = time.perf_counter()
        elapsed = current_time - self._last_fps_time

        if elapsed >= 1.0:
            self._fps = self._frame_count / (current_time - self._start_time)
            self._last_fps_time = current_time

    def stop(self) -> None:
        """Stop the tracking loop and cleanup."""
        self._running = False
        self._logger.info("Stopping hand tracker...")

        # Release mouse buttons
        if self._mouse_controller:
            self._mouse_controller.release_all()

        # Stop IMU reader
        if self._imu_reader:
            self._logger.info("Stopping IMU reader...")
            self._imu_reader.stop()
            self._imu_reader = None

        # Stop dual-camera components (Phase 7)
        if self._phone_receiver:
            self._logger.info("Stopping phone landmark receiver...")
            self._phone_receiver.stop()
            self._phone_receiver = None
        self._camera_fusion = None
        self._hand_ekf = None
        self._dual_recognizer = None

        # Close components
        if self._detector:
            self._detector.close()

        if self._camera:
            self._camera.close()

        # Close debug window
        if self.debug:
            cv2.destroyAllWindows()

        # Print stats
        if self._frame_count > 0:
            elapsed = time.perf_counter() - self._start_time
            avg_fps = self._frame_count / elapsed if elapsed > 0 else 0
            self._logger.info(
                f"Tracking stopped. Processed {self._frame_count} frames "
                f"in {elapsed:.1f}s ({avg_fps:.1f} FPS average)"
            )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Desktop Webcam Hand Tracker - Gesture-based mouse control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
  0  Success
  1  Profile error (file not found, invalid JSON)
  2  Camera error (camera not available)
  3  Runtime error (unexpected error)

Examples:
  python desktop_webcam_handtracker.py --profile config.json
  python desktop_webcam_handtracker.py --profile config.json --camera 1
  python desktop_webcam_handtracker.py --profile config.json --debug
"""
    )

    parser.add_argument(
        "--profile", "-p",
        required=True,
        help="Path to JSON profile file"
    )

    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=-1,
        help="Camera index (default: auto-detect, ignored for XREAL Eye)"
    )

    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode with visualization window"
    )

    parser.add_argument(
        "--parent-pid",
        type=int,
        default=None,
        help="PID of parent process (Launcher) to monitor. If parent exits, hand tracker exits too."
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code.
    """
    args = parse_args()

    # Setup logging
    logger = setup_logging(debug=args.debug)
    logger.info("Desktop Webcam Hand Tracker starting...")

    # Load profile
    try:
        profile = load_profile(args.profile)
    except ProfileLoadError as e:
        logger.error(f"Failed to load profile: {e}")
        return EXIT_PROFILE_ERROR

    # Determine camera index (only relevant for webcam source)
    camera_index = DEFAULT_CAMERA_INDEX
    if profile.camera_source_type == "Webcam":
        try:
            if args.camera >= 0:
                camera_index = args.camera
            elif profile.selected_camera_index >= 0:
                camera_index = profile.selected_camera_index
            else:
                camera_index = select_camera()
        except CameraError as e:
            logger.error(f"Camera selection failed: {e}")
            return EXIT_CAMERA_ERROR
    else:
        logger.info("Using XREAL Eye camera source, ignoring camera index parameter")

    # Create and run application
    app: Optional[HandTrackerApp] = None

    try:
        app = HandTrackerApp(
            profile=profile,
            camera_index=camera_index,
            debug=args.debug
        )

        # Setup signal handler for graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Received shutdown signal")
            if app:
                app.stop()

        signal.signal(signal.SIGINT, signal_handler)
        # SIGTERM is not available on Windows
        if sys.platform != 'win32':
            signal.signal(signal.SIGTERM, signal_handler)

        # Setup parent process monitor (if parent PID provided)
        parent_monitor: Optional[ParentProcessMonitor] = None
        if args.parent_pid is not None:
            logger.info(f"Parent process monitoring enabled (PID: {args.parent_pid})")
            parent_monitor = ParentProcessMonitor(
                parent_pid=args.parent_pid,
                on_parent_exit=lambda: app.stop() if app else None
            )
            parent_monitor.start()

        # Initialize and run
        app.initialize()
        app.run()

        return EXIT_SUCCESS

    except CameraError as e:
        logger.error(f"Camera error: {e}")
        return EXIT_CAMERA_ERROR
    except Exception as e:
        logger.exception(f"Runtime error: {e}")
        return EXIT_RUNTIME_ERROR
    finally:
        if parent_monitor:
            parent_monitor.stop()
        if app:
            app.stop()


if __name__ == "__main__":
    sys.exit(main())
