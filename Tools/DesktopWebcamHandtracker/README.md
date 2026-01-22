# Desktop Webcam Hand Tracker

A Python-based hand tracking module that enables gesture-based mouse control using MediaPipe Hands. This module provides an alternative to the C++ HandTracker for scenarios where MediaPipe's robust hand tracking is preferred.

## Features

- **Real-time hand tracking** using MediaPipe Hands (Lite model for performance)
- **7 gesture types** supported:
  - `pinch` - Thumb and index finger touching (default: left click)
  - `fist` - All fingers curled (default: right click)
  - `point` - Index finger extended (default: cursor movement)
  - `palm` - All fingers extended
  - `thumbsUp` - Thumb extended upward
  - `swipeUp` - Upward hand movement (default: scroll up)
  - `swipeDown` - Downward hand movement (default: scroll down)
- **Configurable gesture-to-action mapping** via JSON profiles
- **Cursor smoothing** with exponential moving average
- **Gesture state machine** with hold/release debouncing
- **Debug visualization** window for development and troubleshooting

## Requirements

- Python 3.10+
- Windows 10/11 (for mouse control via pynput)
- Webcam

## Installation

```bash
cd Tools/DesktopWebcamHandtracker
pip install -r requirements.txt
```

### Dependencies

- `mediapipe` - Hand landmark detection
- `opencv-python` - Camera capture and image processing
- `numpy` - Numerical computations
- `pynput` - Mouse control
- `pywin32` (optional) - Screen resolution detection on Windows

## Usage

### Basic Usage

```bash
python desktop_webcam_handtracker.py --profile <path-to-profile.json>
```

### With Specific Camera

```bash
python desktop_webcam_handtracker.py --profile config.json --camera 1
```

### Debug Mode (with visualization)

```bash
python desktop_webcam_handtracker.py --profile config.json --debug
```

Press `q` or `ESC` to quit in debug mode.

### Command Line Options

| Option | Short | Description |
|--------|-------|-------------|
| `--profile` | `-p` | Path to JSON profile file (required) |
| `--camera` | `-c` | Camera index (default: auto-detect) |
| `--debug` | `-d` | Enable debug visualization window |

## Profile Format

The profile JSON file follows the AROverlay Launcher format with camelCase properties:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Standard",
  "profileType": "preset",
  "mouseSensitivity": 1.0,
  "selectedCameraIndex": -1,
  "selectedCameraName": null,
  "gestureMappings": [
    {"gesture": "pinch", "action": "leftClick", "isEnabled": true},
    {"gesture": "fist", "action": "rightClick", "isEnabled": true},
    {"gesture": "point", "action": "moveCursor", "isEnabled": true},
    {"gesture": "palm", "action": "none", "isEnabled": false},
    {"gesture": "thumbsUp", "action": "none", "isEnabled": false},
    {"gesture": "swipeUp", "action": "scrollUp", "isEnabled": true},
    {"gesture": "swipeDown", "action": "scrollDown", "isEnabled": true}
  ]
}
```

### Available Actions

| Action | Description |
|--------|-------------|
| `none` | No action |
| `moveCursor` | Move cursor to hand position |
| `leftClick` | Left mouse click (press on gesture start, release on end) |
| `rightClick` | Right mouse click |
| `doubleClick` | Double left click |
| `middleClick` | Middle mouse click |
| `scrollUp` | Scroll wheel up |
| `scrollDown` | Scroll wheel down |
| `drag` | Left click and drag |

### Mouse Sensitivity

The `mouseSensitivity` value (0.1 to 5.0) multiplies cursor movement speed:
- `0.5` - Slower, more precise
- `1.0` - Default speed
- `2.0` - Faster movement

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | Profile error (file not found, invalid JSON) |
| 2 | Camera error (camera not available) |
| 3 | Runtime error (unexpected error) |

## Configuration

### config.py

Key configuration constants:

| Constant | Default | Description |
|----------|---------|-------------|
| `CAMERA_WIDTH` | 640 | Capture width |
| `CAMERA_HEIGHT` | 480 | Capture height |
| `CAMERA_FPS` | 30 | Target frame rate |
| `MEDIAPIPE_MODEL_COMPLEXITY` | 0 | Lite model (0) or Full (1) |
| `PINCH_DISTANCE_THRESHOLD` | 0.05 | Pinch detection sensitivity |
| `SWIPE_VELOCITY_THRESHOLD` | 0.8 | Swipe detection sensitivity |
| `CURSOR_EMA_ALPHA` | 0.3 | Cursor smoothing (0=max smooth, 1=no smooth) |
| `GESTURE_HOLD_FRAMES` | 3 | Frames to confirm gesture |
| `GESTURE_DEBOUNCE_MS` | 100 | Min time between gestures |

## Logging

Logs are written to:
- Console (INFO level, or DEBUG with `--debug`)
- File: `%APPDATA%\AROverlay\logs\desktop_webcam_handtracker.log`

Log files rotate at 10MB with 3 backups.

## Architecture

```
desktop_webcam_handtracker.py  # Main entry point
    |
    +-- profile_loader.py      # Load JSON profile
    +-- camera_manager.py      # OpenCV VideoCapture
    +-- hand_detector.py       # MediaPipe Hands integration
    +-- gesture_recognizer.py  # Classify 7 gesture types
    +-- gesture_state_machine.py  # Track hold/release with debouncing
    +-- mouse_controller.py    # pynput mouse control
    +-- config.py              # Configuration constants
    +-- logger.py              # Logging setup
```

## Troubleshooting

### Camera not detected

1. Check if camera is connected and working in other apps
2. Try specifying camera index: `--camera 0` or `--camera 1`
3. Run in debug mode to see camera preview

### Gestures not recognized

1. Ensure good lighting conditions
2. Keep hand within camera frame
3. Try adjusting detection thresholds in `config.py`
4. Use debug mode to see landmark visualization

### High CPU usage

1. Reduce camera resolution in `config.py`
2. Ensure MediaPipe Lite model is being used (`MODEL_COMPLEXITY=0`)

### Cursor movement feels laggy

1. Increase `CURSOR_EMA_ALPHA` in `config.py` (less smoothing)
2. Reduce `GESTURE_HOLD_FRAMES` for faster gesture response

## License

Part of the AROverlay project.
