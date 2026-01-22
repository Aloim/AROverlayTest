"""
Profile loader for DesktopWebcamHandtracker.

Loads and validates JSON profile files exported from the AROverlay Launcher.
Profile properties use camelCase to match the Launcher's JSON format.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from logger import get_logger
from config import DEFAULT_GESTURE_MAPPINGS

logger = get_logger("ProfileLoader")


@dataclass
class GestureMapping:
    """Represents a single gesture-to-action mapping."""

    gesture: str
    action: str
    is_enabled: bool = True
    activation_delay_ms: int = 0  # NEW: 0 = use gesture type default

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GestureMapping":
        """
        Create GestureMapping from dictionary with camelCase keys.

        Args:
            data: Dictionary with gesture, action, isEnabled, activationDelayMs keys.

        Returns:
            GestureMapping instance.
        """
        return cls(
            gesture=data.get("gesture", ""),
            action=data.get("action", "none"),
            is_enabled=data.get("isEnabled", True),
            activation_delay_ms=data.get("activationDelayMs", 0)  # NEW
        )


@dataclass
class PhoneCalibrationData:
    """
    Phone camera calibration data for dual-camera tracking.

    Stored in profile after calibration wizard completes.
    """
    rotation: list[float] = field(default_factory=lambda: [1, 0, 0, 0, 1, 0, 0, 0, 1])  # 3x3 flattened
    translation: list[float] = field(default_factory=lambda: [0, 0, 0])  # x, y, z in meters
    phone_fx: float = 500.0
    phone_fy: float = 500.0
    phone_cx: float = 320.0
    phone_cy: float = 240.0
    reprojection_error: float = 0.0
    calibration_date: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhoneCalibrationData":
        """Create from JSON dictionary."""
        intrinsics = data.get("phoneIntrinsics", {})
        return cls(
            rotation=data.get("rotation", [1, 0, 0, 0, 1, 0, 0, 0, 1]),
            translation=data.get("translation", [0, 0, 0]),
            phone_fx=intrinsics.get("fx", 500.0),
            phone_fy=intrinsics.get("fy", 500.0),
            phone_cx=intrinsics.get("cx", 320.0),
            phone_cy=intrinsics.get("cy", 240.0),
            reprojection_error=data.get("reprojectionError", 0.0),
            calibration_date=data.get("calibrationDate")
        )


@dataclass
class CameraSlotConfig:
    """Configuration for a single camera slot in multi-camera mode (Phase 6)."""
    slot_index: int
    is_enabled: bool = False
    selected_camera_index: int = -1
    selected_camera_name: Optional[str] = None
    position: str = "FacingUser"  # FacingUser, TableFacingUp, XrealEyeCamera, etc.
    flip_horizontal: bool = False
    flip_vertical: bool = False
    x_weight: float = 0.5
    y_weight: float = 0.5
    z_weight: float = 0.5
    gesture_weight: float = 0.5
    is_phone_camera: bool = False
    phone_connection_type: str = "usb"  # usb or wifi
    phone_wifi_ip: Optional[str] = None
    phone_port: int = 52990

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraSlotConfig":
        """Create from JSON dictionary."""
        weights = data.get("weights", {})
        return cls(
            slot_index=data.get("slotIndex", 0),
            is_enabled=data.get("isEnabled", False),
            selected_camera_index=data.get("selectedCameraIndex", -1),
            selected_camera_name=data.get("selectedCameraName"),
            position=data.get("position", "FacingUser"),
            flip_horizontal=data.get("flipHorizontal", False),
            flip_vertical=data.get("flipVertical", False),
            x_weight=weights.get("xWeight", 0.5),
            y_weight=weights.get("yWeight", 0.5),
            z_weight=weights.get("zWeight", 0.5),
            gesture_weight=weights.get("gestureWeight", 0.5),
            is_phone_camera=data.get("isPhoneCamera", False),
            phone_connection_type=data.get("phoneConnectionType", "usb"),
            phone_wifi_ip=data.get("phoneWifiIp"),
            phone_port=data.get("phonePort", 52990)
        )


@dataclass
class VoiceRecognitionConfig:
    """
    Voice recognition configuration for voice-to-text input.

    Enables hands-free text input via speech recognition with Sherpa-ONNX.
    """
    enabled: bool = False
    selected_microphone_device_id: Optional[str] = None
    activation_mode: str = "HandGesture"  # "HandGesture" or "VoiceCommand"
    voice_command_keyword: str = "hey computer"
    enable_autocorrect: bool = True
    enable_grammar_fix: bool = True
    silence_timeout_seconds: int = 5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoiceRecognitionConfig":
        """Create from JSON dictionary with camelCase keys."""
        return cls(
            enabled=data.get("enabled", False),
            selected_microphone_device_id=data.get("selectedMicrophoneDeviceId"),
            activation_mode=data.get("activationMode", "HandGesture"),
            voice_command_keyword=data.get("voiceCommandKeyword", "hey computer"),
            enable_autocorrect=data.get("enableAutocorrect", True),
            enable_grammar_fix=data.get("enableGrammarFix", True),
            silence_timeout_seconds=data.get("silenceTimeoutSeconds", 5)
        )


@dataclass
class HandtrackerProfile:
    """
    Profile configuration loaded from JSON.

    Attributes:
        id: Unique identifier (GUID string).
        name: Profile display name.
        profile_type: Type of profile (preset, custom).
        mouse_sensitivity: Cursor movement multiplier.
        selected_camera_index: Camera device index (-1 for auto) [LEGACY].
        selected_camera_name: Optional camera device name [LEGACY].
        camera_source_type: Camera source type (Webcam or XrealEye) [LEGACY].
        flip_horizontal: Whether to flip video horizontally (XREAL Eye) [LEGACY].
        imu_head_compensation_enabled: Whether to enable IMU-based head compensation (XREAL Eye).
        alternate_cursor_mode_enabled: Experimental relative cursor control mode.
        gesture_mappings: List of gesture-to-action mappings.
        phone_camera_enabled: Whether dual-camera phone tracking is enabled [LEGACY].
        phone_connection_type: "usb" or "wifi" for phone connection [LEGACY].
        phone_wifi_ip: IP address for WiFi connection mode [LEGACY].
        phone_port: TCP port for phone landmark streaming [LEGACY].
        phone_calibration: Calibration data for phone-webcam stereo [LEGACY].
        camera_slots: Multi-camera slot configurations (Phase 6) [NEW].
        voice_recognition: Voice-to-text configuration [NEW].
    """

    id: str
    name: str
    profile_type: str = "Preset"  # CROSS-PHASE-CONTRACT: PascalCase
    mouse_sensitivity: float = 1.0
    selected_camera_index: int = -1  # LEGACY
    selected_camera_name: Optional[str] = None  # LEGACY
    camera_source_type: str = "Webcam"  # LEGACY: "Webcam" or "XrealEye"
    flip_horizontal: bool = False  # LEGACY
    imu_head_compensation_enabled: bool = False
    alternate_cursor_mode_enabled: bool = False  # Experimental: relative cursor control
    gesture_mappings: list[GestureMapping] = field(default_factory=list)
    # Dual-camera settings (LEGACY)
    phone_camera_enabled: bool = False  # LEGACY
    phone_connection_type: str = "usb"  # LEGACY: "usb" or "wifi"
    phone_wifi_ip: Optional[str] = None  # LEGACY
    phone_port: int = 52990  # LEGACY
    phone_calibration: Optional[PhoneCalibrationData] = None  # LEGACY
    # Multi-camera settings (Phase 6 - NEW)
    camera_slots: list[CameraSlotConfig] = field(default_factory=list)
    # Voice recognition settings (Phase 3 Voice-to-Text - NEW)
    voice_recognition: VoiceRecognitionConfig = field(default_factory=VoiceRecognitionConfig)

    def get_mapping(self, gesture: str) -> Optional[GestureMapping]:
        """
        Get the mapping for a specific gesture.

        Args:
            gesture: Gesture name to look up (case-insensitive).

        Returns:
            GestureMapping if found and enabled, None otherwise.
        """
        gesture_lower = gesture.lower()
        for mapping in self.gesture_mappings:
            if mapping.gesture.lower() == gesture_lower and mapping.is_enabled:
                return mapping
        return None

    def get_action(self, gesture: str) -> Optional[str]:
        """
        Get the action for a specific gesture.

        Args:
            gesture: Gesture name to look up.

        Returns:
            Action string if found and enabled, None otherwise.
        """
        mapping = self.get_mapping(gesture)
        return mapping.action if mapping else None


class ProfileLoadError(Exception):
    """Raised when profile loading or validation fails."""
    pass


def load_profile(profile_path: str | Path) -> HandtrackerProfile:
    """
    Load and validate a profile from a JSON file.

    Args:
        profile_path: Path to the JSON profile file.

    Returns:
        Validated HandtrackerProfile instance.

    Raises:
        ProfileLoadError: If file cannot be read or validation fails.
    """
    path = Path(profile_path)
    logger.info(f"Loading profile from: {path}")

    # Check file exists
    if not path.exists():
        raise ProfileLoadError(f"Profile file not found: {path}")

    if not path.is_file():
        raise ProfileLoadError(f"Profile path is not a file: {path}")

    # Read and parse JSON
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ProfileLoadError(f"Invalid JSON in profile: {e}")
    except IOError as e:
        raise ProfileLoadError(f"Cannot read profile file: {e}")

    # Validate and create profile
    return _parse_profile(data)


def _parse_profile(data: dict[str, Any]) -> HandtrackerProfile:
    """
    Parse and validate profile data from dictionary.

    Args:
        data: Dictionary with camelCase profile properties.

    Returns:
        Validated HandtrackerProfile instance.

    Raises:
        ProfileLoadError: If required fields are missing or invalid.
    """
    # Validate required fields
    if "id" not in data:
        raise ProfileLoadError("Profile missing required field: id")

    if "name" not in data:
        raise ProfileLoadError("Profile missing required field: name")

    # Parse gesture mappings
    mappings_data = data.get("gestureMappings", [])
    gesture_mappings = []

    if isinstance(mappings_data, list):
        for mapping_dict in mappings_data:
            if isinstance(mapping_dict, dict):
                try:
                    mapping = GestureMapping.from_dict(mapping_dict)
                    gesture_mappings.append(mapping)
                except Exception as e:
                    logger.warning(f"Skipping invalid gesture mapping: {e}")

    # Apply defaults for missing gestures
    existing_gestures = {m.gesture for m in gesture_mappings}
    for gesture, defaults in DEFAULT_GESTURE_MAPPINGS.items():
        if gesture not in existing_gestures:
            gesture_mappings.append(GestureMapping(
                gesture=gesture,
                action=defaults["action"],
                is_enabled=defaults["isEnabled"]
            ))
            logger.debug(f"Applied default mapping for: {gesture}")

    # CROSS-PHASE-CONTRACT: Validate no duplicate enabled gesture mappings
    enabled_gestures = [m.gesture for m in gesture_mappings if m.is_enabled]
    duplicate_gestures = set([g for g in enabled_gestures if enabled_gestures.count(g) > 1])
    if duplicate_gestures:
        raise ProfileLoadError(f"Duplicate enabled mapping for gesture(s): {', '.join(duplicate_gestures)}")

    # CROSS-PHASE-CONTRACT: Validate LeftHold/LeftRelease pairing
    enabled_actions = {m.action for m in gesture_mappings if m.is_enabled}
    has_left_hold = "leftHold" in enabled_actions
    has_left_release = "leftRelease" in enabled_actions
    if has_left_hold and not has_left_release:
        raise ProfileLoadError("LeftHold action requires a corresponding LeftRelease gesture mapping.")
    if has_left_release and not has_left_hold:
        raise ProfileLoadError("LeftRelease action requires a corresponding LeftHold gesture mapping.")

    # CROSS-PHASE-CONTRACT: Validate profileType (accept both lowercase and PascalCase)
    profile_type_raw = str(data.get("profileType", "Custom"))
    profile_type = profile_type_raw.lower()
    if profile_type not in ("preset", "custom"):
        raise ProfileLoadError(f"Invalid profile type: {profile_type_raw} (expected 'Preset' or 'Custom')")
    # Normalize to PascalCase for internal use
    profile_type = profile_type.capitalize()

    # Validate mouse sensitivity
    sensitivity = data.get("mouseSensitivity", 1.0)
    if not isinstance(sensitivity, (int, float)):
        logger.warning(f"Invalid mouseSensitivity, using default: 1.0")
        sensitivity = 1.0
    sensitivity = max(0.5, min(10.0, float(sensitivity)))  # Clamp to valid range (CROSS-PHASE-CONTRACT: 0.5-10.0)

    # Parse camera settings (LEGACY)
    camera_index = data.get("selectedCameraIndex", -1)
    if not isinstance(camera_index, int):
        camera_index = -1

    camera_name = data.get("selectedCameraName")
    if camera_name is not None and not isinstance(camera_name, str):
        camera_name = None

    # Parse camera source type (LEGACY)
    camera_source_type = data.get("cameraSourceType", "Webcam")
    flip_horizontal = data.get("flipHorizontal", False)
    imu_compensation = data.get("imuHeadCompensationEnabled", False)

    # Backward compatibility: infer from camera index
    if "cameraSourceType" not in data:
        if camera_index == -999:  # XREAL Eye sentinel value
            camera_source_type = "XrealEye"
            flip_horizontal = True
            imu_compensation = True
            logger.info("Inferred XREAL Eye source from camera index -999")

    # Validate camera source type
    if camera_source_type not in ("Webcam", "XrealEye"):
        logger.warning(f"Invalid cameraSourceType '{camera_source_type}', defaulting to Webcam")
        camera_source_type = "Webcam"

    # Ensure boolean types
    if not isinstance(flip_horizontal, bool):
        flip_horizontal = False
    if not isinstance(imu_compensation, bool):
        imu_compensation = False

    # Parse alternate cursor mode (experimental)
    alternate_cursor_mode = data.get("alternateCursorModeEnabled", False)
    if not isinstance(alternate_cursor_mode, bool):
        alternate_cursor_mode = False

    # Parse dual-camera settings (LEGACY)
    phone_camera_enabled = data.get("phoneCameraEnabled", False)
    if not isinstance(phone_camera_enabled, bool):
        phone_camera_enabled = False

    phone_connection_type = data.get("phoneConnectionType", "usb")
    if phone_connection_type not in ("usb", "wifi"):
        phone_connection_type = "usb"

    phone_wifi_ip = data.get("phoneWifiIp")
    if phone_wifi_ip is not None and not isinstance(phone_wifi_ip, str):
        phone_wifi_ip = None

    phone_port = data.get("phonePort", 52990)
    if not isinstance(phone_port, int):
        phone_port = 52990

    phone_calibration = None
    phone_calibration_data = data.get("phoneCalibration")
    if phone_calibration_data and isinstance(phone_calibration_data, dict):
        try:
            phone_calibration = PhoneCalibrationData.from_dict(phone_calibration_data)
        except Exception as e:
            logger.warning(f"Failed to parse phone calibration: {e}")

    # Parse camera slots (NEW - Phase 6)
    camera_slots = []
    if "cameraSlots" in data:
        # New multi-camera format
        for slot_data in data.get("cameraSlots", []):
            if isinstance(slot_data, dict):
                try:
                    camera_slots.append(CameraSlotConfig.from_dict(slot_data))
                except Exception as e:
                    logger.warning(f"Skipping invalid camera slot: {e}")
        logger.info(f"Loaded {len(camera_slots)} camera slots (multi-camera format)")
    else:
        # Legacy dual-camera format - migrate to slots
        # Slot 0: Primary webcam
        slot0 = CameraSlotConfig(
            slot_index=0,
            is_enabled=True,
            selected_camera_index=camera_index,
            selected_camera_name=camera_name,
            position="XrealEyeCamera" if camera_source_type == "XrealEye" else "FacingUser",
            flip_horizontal=flip_horizontal,
            flip_vertical=False
        )
        camera_slots.append(slot0)

        # Slot 1: Phone (if enabled)
        if phone_camera_enabled:
            slot1 = CameraSlotConfig(
                slot_index=1,
                is_enabled=True,
                selected_camera_index=-1,  # Phone doesn't use local camera index
                selected_camera_name=None,
                position="TableFacingUp",
                flip_horizontal=False,
                flip_vertical=False,
                is_phone_camera=True,
                phone_connection_type=phone_connection_type,
                phone_wifi_ip=phone_wifi_ip,
                phone_port=phone_port
            )
            camera_slots.append(slot1)
        logger.info("Migrated legacy profile to camera slots format")

    # Parse voice recognition settings (NEW - Phase 3)
    voice_recognition = VoiceRecognitionConfig()
    voice_recognition_data = data.get("voiceRecognitionSettings")
    if voice_recognition_data and isinstance(voice_recognition_data, dict):
        try:
            voice_recognition = VoiceRecognitionConfig.from_dict(voice_recognition_data)
            logger.info(f"Loaded voice recognition settings (enabled={voice_recognition.enabled})")
        except Exception as e:
            logger.warning(f"Failed to parse voice recognition settings: {e}")

    profile = HandtrackerProfile(
        id=str(data["id"]),
        name=str(data["name"]),
        profile_type=profile_type,  # Already validated above
        mouse_sensitivity=sensitivity,
        selected_camera_index=camera_index,  # LEGACY
        selected_camera_name=camera_name,  # LEGACY
        camera_source_type=camera_source_type,  # LEGACY
        flip_horizontal=flip_horizontal,  # LEGACY
        imu_head_compensation_enabled=imu_compensation,
        alternate_cursor_mode_enabled=alternate_cursor_mode,
        gesture_mappings=gesture_mappings,
        phone_camera_enabled=phone_camera_enabled,  # LEGACY
        phone_connection_type=phone_connection_type,  # LEGACY
        phone_wifi_ip=phone_wifi_ip,  # LEGACY
        phone_port=phone_port,  # LEGACY
        phone_calibration=phone_calibration,  # LEGACY
        camera_slots=camera_slots,  # NEW
        voice_recognition=voice_recognition  # NEW
    )

    logger.info(f"Loaded profile: {profile.name} (id={profile.id})")
    logger.debug(f"  Sensitivity: {profile.mouse_sensitivity}")
    logger.debug(f"  Camera source: {profile.camera_source_type}")
    logger.debug(f"  Camera index: {profile.selected_camera_index}")
    logger.debug(f"  Flip horizontal: {profile.flip_horizontal}")
    logger.debug(f"  IMU compensation: {profile.imu_head_compensation_enabled}")
    logger.debug(f"  Alternate cursor mode: {profile.alternate_cursor_mode_enabled}")
    logger.debug(f"  Phone camera enabled: {profile.phone_camera_enabled}")
    if profile.phone_camera_enabled:
        logger.debug(f"  Phone connection: {profile.phone_connection_type}")
        logger.debug(f"  Phone port: {profile.phone_port}")
        if profile.phone_calibration:
            logger.debug(f"  Phone calibration: error={profile.phone_calibration.reprojection_error:.2f}px")
    logger.debug(f"  Gesture mappings: {len(profile.gesture_mappings)}")
    logger.debug(f"  Camera slots: {len(profile.camera_slots)}")
    logger.debug(f"  Voice recognition enabled: {profile.voice_recognition.enabled}")

    return profile


def create_default_profile() -> HandtrackerProfile:
    """
    Create a default profile with standard settings.

    Returns:
        HandtrackerProfile with default values.
    """
    mappings = [
        GestureMapping(gesture=g, action=d["action"], is_enabled=d["isEnabled"])
        for g, d in DEFAULT_GESTURE_MAPPINGS.items()
    ]

    # Default camera slots: single webcam
    default_slots = [
        CameraSlotConfig(
            slot_index=0,
            is_enabled=True,
            selected_camera_index=-1,
            position="FacingUser"
        )
    ]

    return HandtrackerProfile(
        id="default",
        name="Default",
        profile_type="Preset",  # CROSS-PHASE-CONTRACT: PascalCase
        mouse_sensitivity=1.0,
        selected_camera_index=-1,
        selected_camera_name=None,
        camera_source_type="Webcam",
        flip_horizontal=False,
        imu_head_compensation_enabled=False,
        gesture_mappings=mappings,
        camera_slots=default_slots,
        voice_recognition=VoiceRecognitionConfig()
    )
