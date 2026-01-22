"""
Voice UI bridge for DesktopWebcamHandtracker.

Provides Named Pipe communication between Python voice recognition
and WPF Launcher UI for real-time transcription feedback and confirmation.

Uses overlapped (asynchronous) I/O to allow concurrent read/write operations.
"""

import json
import os
import threading
import time
from typing import Optional, Callable, Dict, Any
from queue import Queue, Empty

try:
    import win32pipe
    import win32file
    import win32event
    import pywintypes
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

from logger import get_logger

logger = get_logger("VoiceUIBridge")

# Named pipe configuration
PIPE_NAME = r"\\.\pipe\AROverlay_VoiceRecognition"
PIPE_BUFFER_SIZE = 4096
PIPE_TIMEOUT_MS = 1000


class VoiceUIBridge:
    """
    Named Pipe server for WPF communication.

    Sends transcription updates to WPF and receives confirmation/cancellation commands.
    Uses JSON message protocol for structured communication.
    Uses overlapped I/O to allow concurrent read/write from different threads.
    """

    def __init__(self, on_launcher_disconnect: Optional[Callable[[], None]] = None):
        """
        Initialize voice UI bridge.

        Args:
            on_launcher_disconnect: Optional callback invoked when WPF Launcher disconnects
                                   unexpectedly. Can be used to trigger process shutdown.
        """
        if not HAS_WIN32:
            raise RuntimeError("VoiceUIBridge requires pywin32 on Windows")

        self._pipe_handle: Optional[int] = None
        self._server_thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False
        self._was_connected = False  # Track if we ever connected

        # Thread synchronization for writes
        self._write_lock = threading.Lock()

        # Queue for outgoing messages (thread-safe)
        self._send_queue: Queue = Queue()

        # Message handlers
        self._command_callback: Optional[Callable[[Dict[str, Any]], None]] = None

        # Disconnect callback for triggering shutdown when Launcher crashes
        self._on_launcher_disconnect = on_launcher_disconnect

        # Startup grace period - don't trigger exit during initial startup
        self._startup_time = time.time()
        self._startup_grace_period = 15.0  # 15 seconds before heartbeat exit is enabled

        logger.info(f"VoiceUIBridge initialized (pipe={PIPE_NAME})")

    def start_server(self, command_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        """
        Start Named Pipe server.

        Args:
            command_callback: Callback function to handle commands from WPF.
                            Receives dict with command data.
        """
        if self._running:
            logger.warning("Server already running")
            return

        self._command_callback = command_callback
        self._running = True

        # Start server thread
        self._server_thread = threading.Thread(
            target=self._server_loop,
            daemon=True
        )
        self._server_thread.start()

        logger.info("Voice UI bridge server started")

    def stop_server(self) -> None:
        """Stop Named Pipe server."""
        if not self._running:
            return

        self._running = False

        # Close pipe if open
        if self._pipe_handle:
            try:
                win32file.CloseHandle(self._pipe_handle)
            except Exception:
                pass
            self._pipe_handle = None

        # Wait for server thread
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=2.0)

        self._connected = False
        logger.info("Voice UI bridge server stopped")

    def send_voice_input_started(self) -> bool:
        """
        Notify WPF that voice input has started.
        This triggers the VoiceInputWindow to be displayed.

        Returns:
            True if sent successfully, False otherwise.
        """
        logger.info(f"send_voice_input_started called - connected={self._connected}, pipe_handle={self._pipe_handle is not None}")
        message = {
            "type": "voice_input_started",
            "timestamp": time.time()
        }
        result = self._send_message(message)
        logger.info(f"send_voice_input_started result={result}")
        return result

    def send_partial_transcription(self, text: str) -> bool:
        """
        Send partial transcription update to WPF.

        Args:
            text: Partial transcription text.

        Returns:
            True if sent successfully, False otherwise.
        """
        message = {
            "type": "partial_transcription",
            "text": text,
            "timestamp": time.time()
        }
        return self._send_message(message)

    def send_final_transcription(self, text: str) -> bool:
        """
        Send final transcription result to WPF.

        Args:
            text: Final transcription text.

        Returns:
            True if sent successfully, False otherwise.
        """
        message = {
            "type": "final_transcription",
            "text": text,
            "timestamp": time.time()
        }
        return self._send_message(message)

    def send_recognition_ended(self) -> bool:
        """
        Notify WPF that voice recognition has ended.

        Returns:
            True if sent successfully, False otherwise.
        """
        message = {
            "type": "recognition_ended",
            "timestamp": time.time()
        }
        return self._send_message(message)

    def send_error(self, error_message: str) -> bool:
        """
        Send error notification to WPF.

        Args:
            error_message: Error description.

        Returns:
            True if sent successfully, False otherwise.
        """
        message = {
            "type": "error",
            "error": error_message,  # C# expects "error" field
            "timestamp": time.time()
        }
        return self._send_message(message)

    def send_typing_result(self, success: bool, message: str) -> bool:
        """
        Send typing result to WPF.

        Args:
            success: Whether typing was successful.
            message: Result message to display.

        Returns:
            True if sent successfully, False otherwise.
        """
        result = {
            "type": "typing_result",
            "success": success,
            "message": message,
            "timestamp": time.time()
        }
        return self._send_message(result)

    def is_connected(self) -> bool:
        """
        Check if WPF client is connected.

        Returns:
            True if connected, False otherwise.
        """
        return self._connected

    def _server_loop(self) -> None:
        """Main server loop (runs in separate thread)."""
        while self._running:
            try:
                # Create named pipe with OVERLAPPED flag for async I/O
                logger.debug("Creating named pipe with overlapped I/O...")
                self._pipe_handle = win32pipe.CreateNamedPipe(
                    PIPE_NAME,
                    win32pipe.PIPE_ACCESS_DUPLEX | win32file.FILE_FLAG_OVERLAPPED,
                    win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
                    1,  # Max instances
                    PIPE_BUFFER_SIZE,
                    PIPE_BUFFER_SIZE,
                    PIPE_TIMEOUT_MS,
                    None
                )

                if self._pipe_handle == win32file.INVALID_HANDLE_VALUE:
                    logger.error("Failed to create named pipe")
                    time.sleep(1.0)
                    continue

                logger.info("Waiting for WPF client to connect...")

                # Wait for client connection using overlapped I/O
                overlapped = pywintypes.OVERLAPPED()
                overlapped.hEvent = win32event.CreateEvent(None, True, False, None)

                try:
                    win32pipe.ConnectNamedPipe(self._pipe_handle, overlapped)
                except pywintypes.error as e:
                    if e.winerror == 997:  # ERROR_IO_PENDING - normal for async
                        # Wait for connection with timeout (allows checking _running)
                        while self._running:
                            result = win32event.WaitForSingleObject(overlapped.hEvent, 500)
                            if result == win32event.WAIT_OBJECT_0:
                                break  # Connected
                            elif result == win32event.WAIT_TIMEOUT:
                                continue  # Keep waiting
                            else:
                                logger.error(f"Wait failed: {result}")
                                break

                        if not self._running:
                            win32file.CloseHandle(overlapped.hEvent)
                            self._cleanup_pipe()
                            continue
                    elif e.winerror == 535:  # ERROR_PIPE_CONNECTED - already connected
                        pass  # Client was already waiting
                    elif e.winerror == 232:  # Pipe broken
                        logger.debug("Pipe connection closed before connect")
                        win32file.CloseHandle(overlapped.hEvent)
                        self._cleanup_pipe()
                        continue
                    else:
                        raise

                win32file.CloseHandle(overlapped.hEvent)
                self._connected = True
                # Note: _was_connected is set only after successful communication
                logger.info("WPF client connected")

                # Handle client communication
                self._handle_client_overlapped()

            except Exception as e:
                logger.error(f"Server loop error: {e}")
                self._cleanup_pipe()
                if self._running:
                    time.sleep(1.0)

        logger.debug("Server loop ended")

    def _handle_client_overlapped(self) -> None:
        """Handle communication with connected WPF client using overlapped I/O."""
        buffer = b""
        read_buffer = win32file.AllocateReadBuffer(PIPE_BUFFER_SIZE)

        # Create overlapped structure for reading
        overlapped = pywintypes.OVERLAPPED()
        overlapped.hEvent = win32event.CreateEvent(None, True, False, None)

        try:
            # Start initial async read
            pending_read = False

            while self._running and self._connected:
                # Start a new async read if none pending
                if not pending_read:
                    try:
                        win32event.ResetEvent(overlapped.hEvent)
                        hr, _ = win32file.ReadFile(
                            self._pipe_handle,
                            read_buffer,
                            overlapped
                        )
                        if hr == 0:
                            # Completed synchronously (data already available)
                            pending_read = False
                            data = bytes(read_buffer)
                            data = data.rstrip(b'\x00')  # Remove null padding
                            if data:
                                buffer += data
                        elif hr == 997:  # ERROR_IO_PENDING
                            pending_read = True
                        else:
                            logger.warning(f"ReadFile returned {hr}")
                            break
                    except pywintypes.error as e:
                        if e.winerror == 997:  # ERROR_IO_PENDING
                            pending_read = True
                        elif e.winerror == 109:  # Pipe broken
                            logger.info("WPF client disconnected")
                            break
                        elif e.winerror == 232:  # Pipe is being closed
                            logger.debug("Pipe is being closed")
                            break
                        elif e.winerror == 536:  # ERROR_PIPE_NOT_CONNECTED
                            # Connection wasn't really established - don't treat as disconnect
                            logger.debug("Pipe not actually connected (error 536)")
                            break
                        else:
                            logger.error(f"ReadFile error: {e}")
                            break

                # Check if async read completed (with short timeout to allow writes)
                if pending_read:
                    result = win32event.WaitForSingleObject(overlapped.hEvent, 50)  # 50ms timeout
                    if result == win32event.WAIT_OBJECT_0:
                        # Read completed
                        try:
                            bytes_read = win32file.GetOverlappedResult(self._pipe_handle, overlapped, False)
                            if bytes_read > 0:
                                data = bytes(read_buffer[:bytes_read])
                                buffer += data
                                logger.debug(f"Read {bytes_read} bytes from pipe")
                                # Mark as truly connected after successful communication
                                self._was_connected = True
                            pending_read = False
                        except pywintypes.error as e:
                            if e.winerror == 109:  # Pipe broken
                                logger.info("WPF client disconnected")
                                break
                            else:
                                logger.error(f"GetOverlappedResult error: {e}")
                                break
                    elif result == win32event.WAIT_TIMEOUT:
                        pass  # Continue, check for messages to send
                    else:
                        logger.error(f"Wait failed: {result}")
                        break

                # Process any complete lines in buffer
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    if line:
                        try:
                            message_str = line.decode('utf-8')
                            self._process_command(message_str)
                        except UnicodeDecodeError as e:
                            logger.error(f"UTF-8 decode error: {e}")

                # Small sleep to prevent busy-waiting
                if not pending_read:
                    time.sleep(0.01)

        except Exception as e:
            logger.error(f"Client handler error: {e}")
        finally:
            was_connected = self._connected
            self._connected = False
            try:
                win32file.CloseHandle(overlapped.hEvent)
            except Exception:
                pass
            self._cleanup_pipe()
            logger.debug("Client handler ended")

            # Only force exit if:
            # 1. We had successful communication before disconnect
            # 2. We're past the startup grace period (15 seconds)
            # This prevents exit on initial connection failures or transient errors.
            elapsed_since_startup = time.time() - self._startup_time
            if self._was_connected and elapsed_since_startup > self._startup_grace_period:
                logger.warning("WPF Launcher disconnected after successful communication - forcing process exit")
                if self._on_launcher_disconnect:
                    try:
                        self._on_launcher_disconnect()
                    except Exception as ex:
                        logger.error(f"Error in disconnect callback: {ex}")
                # Give a moment for logs to flush, then force exit
                time.sleep(0.5)
                os._exit(0)
            elif self._was_connected:
                logger.debug(f"Connection lost but still in startup grace period ({elapsed_since_startup:.1f}s < {self._startup_grace_period}s) - will retry")
                time.sleep(0.5)
            else:
                logger.debug("Connection ended before successful communication - will retry after delay")
                # Small delay to prevent CPU-spinning on repeated connection failures
                time.sleep(0.5)

    def _process_command(self, message_str: str) -> None:
        """
        Process command received from WPF.

        Args:
            message_str: JSON message string.
        """
        try:
            # Handle UTF-8 BOM if present (C# StreamWriter may include it)
            clean_str = message_str.strip()
            if clean_str.startswith('\ufeff'):
                clean_str = clean_str[1:]
            message = json.loads(clean_str)
            command_type = message.get("type", "")

            logger.debug(f"Received command: {command_type}")

            # Call registered callback
            if self._command_callback:
                self._command_callback(message)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from WPF: {e}")
        except Exception as e:
            logger.error(f"Command processing error: {e}")

    def _send_message(self, message: Dict[str, Any]) -> bool:
        """
        Send JSON message to WPF client using overlapped I/O.

        Args:
            message: Message dictionary to send.

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self._connected or not self._pipe_handle:
            logger.debug("Cannot send message - not connected")
            return False

        try:
            message_str = json.dumps(message) + '\n'
            message_bytes = message_str.encode('utf-8')

            logger.debug(f"Attempting to send {len(message_bytes)} bytes...")

            # Use overlapped I/O for writing
            overlapped = pywintypes.OVERLAPPED()
            overlapped.hEvent = win32event.CreateEvent(None, True, False, None)

            try:
                with self._write_lock:
                    hr, _ = win32file.WriteFile(
                        self._pipe_handle,
                        message_bytes,
                        overlapped
                    )

                    if hr == 0:
                        # Completed synchronously
                        bytes_written = len(message_bytes)
                    elif hr == 997:  # ERROR_IO_PENDING
                        # Wait for write to complete (with timeout)
                        result = win32event.WaitForSingleObject(overlapped.hEvent, 5000)
                        if result == win32event.WAIT_OBJECT_0:
                            bytes_written = win32file.GetOverlappedResult(self._pipe_handle, overlapped, False)
                        else:
                            logger.error(f"Write wait failed: {result}")
                            return False
                    else:
                        logger.warning(f"WriteFile returned {hr}")
                        return False

                    # Flush to ensure data is sent immediately
                    win32file.FlushFileBuffers(self._pipe_handle)

                # Mark as truly connected after successful communication
                self._was_connected = True
                logger.info(f"Sent message: {message['type']} ({bytes_written} bytes)")
                return True

            finally:
                try:
                    win32file.CloseHandle(overlapped.hEvent)
                except Exception:
                    pass

        except pywintypes.error as e:
            if e.winerror == 232:  # Pipe broken
                logger.debug("Pipe broken during write")
                self._connected = False
            elif e.winerror == 997:  # Should be handled above, but just in case
                logger.debug("Write pending (unexpected)")
            else:
                logger.error(f"WriteFile error: {e}")
            return False
        except Exception as e:
            logger.error(f"Send message error: {e}")
            return False

    def _cleanup_pipe(self) -> None:
        """Clean up pipe handle."""
        if self._pipe_handle:
            try:
                win32file.CloseHandle(self._pipe_handle)
            except Exception:
                pass
            self._pipe_handle = None


def test_voice_ui_bridge():
    """Simple test function for voice UI bridge."""
    def command_handler(message: Dict[str, Any]):
        print(f"Received command: {message}")

    bridge = VoiceUIBridge()
    bridge.start_server(command_callback=command_handler)

    print("Voice UI Bridge Test Server")
    print(f"Listening on: {PIPE_NAME}")
    print("Waiting for WPF client to connect...")
    print("Press Ctrl+C to stop")

    try:
        # Wait for connection
        while not bridge.is_connected():
            time.sleep(0.1)

        print("Client connected!")

        # Send test messages
        for i in range(5):
            time.sleep(1.0)
            bridge.send_partial_transcription(f"Test message {i}")

        bridge.send_final_transcription("Final test message")

        # Keep server running
        time.sleep(10.0)

    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        bridge.stop_server()
        print("Server stopped")


if __name__ == "__main__":
    if HAS_WIN32:
        test_voice_ui_bridge()
    else:
        print("ERROR: pywin32 is required for Named Pipe communication")
