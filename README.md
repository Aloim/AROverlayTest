# AROverlay - Alpha Release

AROverlay is a Windows AR platform that enables hand tracking and gesture control using your PC's webcam.

## Features

- **Hand Tracking** - Real-time hand detection using MediaPipe
- **Gesture Recognition** - Control your mouse with hand gestures
- **Voice Input** - Optional voice-to-text functionality
- **Profile System** - Save and load custom gesture configurations

## Requirements

### System
- Windows 10/11 (64-bit)
- Webcam
- [.NET 6.0 Runtime](https://dotnet.microsoft.com/download/dotnet/6.0) (Download the "Run desktop apps" version)

### For Hand Tracking Module
- [Python 3.10+](https://www.python.org/downloads/)
- Required Python packages (install via pip):
  ```
  pip install mediapipe opencv-python pynput numpy
  ```

## Installation

1. Download or clone this repository
2. Install .NET 6.0 Runtime if not already installed
3. Install Python and required packages (see above)
4. Run `AROverlay.Launcher.exe`

## Usage

1. Launch `AROverlay.Launcher.exe`
2. Select your webcam from the settings
3. Enable the Desktop Webcam Handtracker module
4. Use hand gestures to control your mouse:
   - **Point** - Move cursor
   - **Pinch** - Left click
   - **Fist** - Right click
   - Additional gestures configurable in settings

## Folder Structure

```
AROverlay/
├── AROverlay.Launcher.exe    # Main application
├── Tools/
│   └── DesktopWebcamHandtracker/   # Python hand tracking module
├── profiles/                 # User profiles and settings
└── modules/                  # Optional modules (empty by default)
```

## Troubleshooting

- **App won't start**: Make sure .NET 6.0 Runtime is installed
- **Hand tracking not working**: Verify Python is installed and in PATH, and all required packages are installed
- **Webcam not detected**: Check webcam permissions in Windows Settings

## Disclaimer

### Executable File Notice
`AROverlay.Launcher.exe` is a compiled Windows executable. As with any software downloaded from the internet, please ensure you trust the source before running. This application is open-source and you can review the code in this repository.

### Privacy & Data Collection
**AROverlay does not collect, store, or transmit any personal data or private information.**

- The application runs entirely locally on your machine
- No analytics, telemetry, or tracking of any kind
- Your webcam feed is processed locally and never leaves your device

**Debug Mode:** If you explicitly enable Debug Mode in the settings, the application will generate local log files containing system and application-related diagnostic information (such as error messages, hardware configurations, and app state). This data is:
- Stored locally in the `profiles/debug-logs/` folder
- Never sent automatically
- Only transmitted if you actively choose to send logs using the Debug feedback form and clicking the Send button

You have full control over your data at all times.

## License

No license specified - All rights reserved.

## Status

This is an **Alpha Release** for demonstration purposes.
