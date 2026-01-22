"""
Diagnostic script to test Named Pipe connection between Python and WPF.
Run this while the Launcher is running (but before starting handtracker).
"""

import time
import sys

print("=" * 60)
print("Named Pipe Connection Diagnostic")
print("=" * 60)

# Test 1: Check if pywin32 is available
print("\n[1] Checking pywin32...")
try:
    import win32pipe
    import win32file
    import pywintypes
    print("    OK - pywin32 is installed")
except ImportError as e:
    print(f"    FAIL - pywin32 not installed: {e}")
    sys.exit(1)

# Test 2: Check VoiceUIBridge import
print("\n[2] Checking VoiceUIBridge import...")
try:
    from voice_ui_bridge import VoiceUIBridge, PIPE_NAME
    print(f"    OK - VoiceUIBridge imported")
    print(f"    Pipe name: {PIPE_NAME}")
except Exception as e:
    print(f"    FAIL - Import error: {e}")
    sys.exit(1)

# Test 3: Start server and wait for connection
print("\n[3] Starting Named Pipe server...")
print("    (The WPF Launcher should connect automatically when handtracker starts)")
print("    Waiting 10 seconds for a connection...")

bridge = VoiceUIBridge()
bridge.start_server()

connected = False
for i in range(100):  # 10 seconds
    time.sleep(0.1)
    if bridge.is_connected():
        print(f"\n    SUCCESS! Connected after {(i+1)*0.1:.1f} seconds")
        connected = True
        break
    if (i + 1) % 10 == 0:
        print(f"    ... waiting ({(i+1)*0.1:.0f}s)")

if not connected:
    print("\n    TIMEOUT - No connection received")
    print("\n    Possible issues:")
    print("    - Launcher's VoiceInputCoordinatorService may not be starting")
    print("    - The handtracker module needs to be running")
    print("    - Check if another process is using the pipe name")

# Test 4: If connected, try sending a message
if connected:
    print("\n[4] Testing message send...")
    result = bridge.send_voice_input_started()
    print(f"    send_voice_input_started() returned: {result}")
    if result:
        print("    SUCCESS! Message sent to WPF")
    else:
        print("    FAIL - Message could not be sent")

bridge.stop_server()
print("\n" + "=" * 60)
print("Diagnostic complete")
print("=" * 60)
