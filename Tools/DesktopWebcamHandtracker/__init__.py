"""
DesktopWebcamHandtracker - Gesture-based mouse control using MediaPipe Hands.

This module provides an alternative to the C++ HandTracker for scenarios
where MediaPipe's robust hand tracking is preferred.
"""

__version__ = "1.0.0"
__author__ = "AROverlay Team"

from .profile_loader import HandtrackerProfile, GestureMapping, load_profile
from .camera_manager import CameraManager, CameraError
from .hand_detector import HandDetector, HandLandmarks, Landmark
from .gesture_recognizer import GestureRecognizer, GestureType, GestureResult
from .gesture_state_machine import GestureStateMachine, GestureEvent, GestureState
from .mouse_controller import GestureMouseController

__all__ = [
    "HandtrackerProfile",
    "GestureMapping",
    "load_profile",
    "CameraManager",
    "CameraError",
    "HandDetector",
    "HandLandmarks",
    "Landmark",
    "GestureRecognizer",
    "GestureType",
    "GestureResult",
    "GestureStateMachine",
    "GestureEvent",
    "GestureState",
    "GestureMouseController",
]
