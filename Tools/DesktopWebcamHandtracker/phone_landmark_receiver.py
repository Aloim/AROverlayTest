"""
Phone landmark receiver for dual-camera hand tracking.

Receives hand landmarks from the Android PhoneCam app via TCP.
Handles binary packet parsing, timestamp synchronization, and
connection management.

Binary packet format (284 bytes):
- Offset 0:   Magic (4 bytes) = 0x484E4431 ("HND1")
- Offset 4:   Timestamp (4 bytes, ms since session start)
- Offset 8:   21 keypoints x 12 bytes = 252 bytes (x, y, z as float32)
- Offset 260: Detection confidence (4 bytes, float32)
- Offset 264: Handedness (1 byte, 0=left, 1=right)
- Offset 265: Frame ID (4 bytes, monotonic counter)
- Offset 269: Reserved (15 bytes padding)
"""

import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple

from logger import get_logger

logger = get_logger("PhoneLandmarkReceiver")


# Packet constants
PACKET_MAGIC = 0x484E4431  # "HND1"
PACKET_SIZE = 284
NUM_LANDMARKS = 21

# Handshake protocol constants
HANDSHAKE_MAGIC = 0x50484E44  # "PHND"
HANDSHAKE_SIZE = 40
ACK_MAGIC = 0x41434B00        # "ACK\0"
ACK_SIZE = 8
HANDSHAKE_TIMEOUT = 1.0       # Timeout waiting for handshake packet


@dataclass
class HandshakeInfo:
    """
    Device information from handshake packet.

    Attributes:
        device_id: Unique device UUID (16 bytes as hex string).
        protocol_version: Protocol version (major.minor).
        capabilities: Capability bitmask.
        camera_count: Number of cameras in use.
        is_multicamera: True if using multi-camera mode.
        sensitivity: Cursor sensitivity setting.
    """
    device_id: str
    protocol_version: str
    capabilities: int
    camera_count: int
    is_multicamera: bool
    sensitivity: float


@dataclass
class PhoneLandmarks:
    """
    Hand landmarks received from phone camera.

    Attributes:
        keypoints: List of 21 (x, y, z) tuples, normalized 0-1.
        confidence: Detection confidence (0-1).
        handedness: 0=left, 1=right.
        timestamp_ms: Timestamp from phone (ms since session start).
        frame_id: Monotonic frame counter from phone.
        receive_time: Local PC time when packet was received.
    """
    keypoints: List[Tuple[float, float, float]]
    confidence: float
    handedness: int
    timestamp_ms: int
    frame_id: int
    receive_time: float

    @property
    def has_hand(self) -> bool:
        """Check if a hand was detected (confidence > 0)."""
        return self.confidence > 0.0

    @property
    def wrist(self) -> Tuple[float, float, float]:
        """Get wrist position (landmark 0)."""
        return self.keypoints[0] if self.keypoints else (0, 0, 0)

    @property
    def index_tip(self) -> Tuple[float, float, float]:
        """Get index fingertip position (landmark 8)."""
        return self.keypoints[8] if len(self.keypoints) > 8 else (0, 0, 0)


class PhoneLandmarkReceiver:
    """
    Receives hand landmarks from phone via TCP.

    Connects to the PhoneCam app's TCP server (either via ADB port
    forwarding or direct WiFi connection) and receives binary
    landmark packets.

    Usage:
        receiver = PhoneLandmarkReceiver(port=52990)
        receiver.start()

        # In main loop:
        landmarks = receiver.get_latest()
        if landmarks and landmarks.has_hand:
            # Process hand position
            pass

        receiver.stop()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 52990,
        buffer_size: int = 10,
        connect_timeout: float = 2.0,
        reconnect_delay: float = 1.0
    ):
        """
        Initialize the receiver.

        Args:
            host: Host to connect to (localhost for ADB, phone IP for WiFi).
            port: TCP port (default 52990).
            buffer_size: Number of recent packets to buffer.
            connect_timeout: Connection timeout in seconds.
            reconnect_delay: Delay between reconnection attempts.
        """
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.connect_timeout = connect_timeout
        self.reconnect_delay = reconnect_delay

        self._socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False

        # Thread-safe buffer for recent landmarks
        self._buffer: deque[PhoneLandmarks] = deque(maxlen=buffer_size)
        self._buffer_lock = threading.Lock()

        # Statistics
        self._packets_received = 0
        self._packets_dropped = 0
        self._last_receive_time = 0.0
        self._session_start_time = 0.0

        # Timestamp synchronization
        self._time_offset_ms = 0  # phone_time - local_time

        # Device info from handshake (None if legacy protocol)
        self._device_info: Optional[HandshakeInfo] = None
        self._using_new_protocol = False

        logger.info(f"PhoneLandmarkReceiver initialized: {host}:{port}")

    def start(self) -> None:
        """Start the receiver thread."""
        if self._running:
            logger.warning("Receiver already running")
            return

        self._running = True
        self._session_start_time = time.time()
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()

        logger.info("PhoneLandmarkReceiver started")

    def stop(self) -> None:
        """Stop the receiver thread and close connection."""
        if not self._running:
            return

        self._running = False

        if self._socket:
            try:
                self._socket.close()
            except Exception as e:
                logger.debug(f"Error closing socket: {e}")

        if self._thread:
            self._thread.join(timeout=2.0)

        self._socket = None
        self._thread = None
        self._connected = False

        logger.info(f"PhoneLandmarkReceiver stopped. Received: {self._packets_received}, "
                   f"Dropped: {self._packets_dropped}")

    def get_latest(self) -> Optional[PhoneLandmarks]:
        """
        Get the most recent landmarks (non-blocking).

        Returns:
            Most recent PhoneLandmarks, or None if buffer is empty.
        """
        with self._buffer_lock:
            if self._buffer:
                return self._buffer[-1]
            return None

    def get_interpolated(self, target_time: float) -> Optional[PhoneLandmarks]:
        """
        Get landmarks interpolated to a target time.

        Finds the two closest packets to target_time and linearly
        interpolates between them. Useful for synchronizing with
        webcam frames.

        Args:
            target_time: Target PC local time.

        Returns:
            Interpolated PhoneLandmarks, or None if not enough data.
        """
        with self._buffer_lock:
            if len(self._buffer) < 2:
                return self.get_latest()

            # Find bracketing packets
            before: Optional[PhoneLandmarks] = None
            after: Optional[PhoneLandmarks] = None

            for lm in self._buffer:
                if lm.receive_time <= target_time:
                    before = lm
                elif after is None:
                    after = lm
                    break

            if before is None:
                return self._buffer[0] if self._buffer else None
            if after is None:
                return self._buffer[-1]

            # Linear interpolation
            t_range = after.receive_time - before.receive_time
            if t_range < 0.001:
                return before

            alpha = (target_time - before.receive_time) / t_range
            alpha = max(0.0, min(1.0, alpha))

            interpolated_keypoints = []
            for i in range(NUM_LANDMARKS):
                bx, by, bz = before.keypoints[i]
                ax, ay, az = after.keypoints[i]
                interpolated_keypoints.append((
                    bx + alpha * (ax - bx),
                    by + alpha * (ay - by),
                    bz + alpha * (az - bz)
                ))

            return PhoneLandmarks(
                keypoints=interpolated_keypoints,
                confidence=before.confidence + alpha * (after.confidence - before.confidence),
                handedness=before.handedness,
                timestamp_ms=int(before.timestamp_ms + alpha * (after.timestamp_ms - before.timestamp_ms)),
                frame_id=after.frame_id,
                receive_time=target_time
            )

    @property
    def is_connected(self) -> bool:
        """Check if connected to phone."""
        return self._connected

    @property
    def latency_ms(self) -> float:
        """
        Estimate current latency in milliseconds.

        Based on time since last packet received.
        """
        if self._last_receive_time == 0:
            return 0.0
        return (time.time() - self._last_receive_time) * 1000

    @property
    def packet_rate(self) -> float:
        """Get packets received per second."""
        elapsed = time.time() - self._session_start_time
        if elapsed > 0:
            return self._packets_received / elapsed
        return 0.0

    @property
    def device_info(self) -> Optional[HandshakeInfo]:
        """Get device info from handshake (None if legacy protocol)."""
        return self._device_info

    @property
    def using_new_protocol(self) -> bool:
        """Check if using new protocol with handshake."""
        return self._using_new_protocol

    def verify_connection(self) -> bool:
        """
        Verify connection is properly established.

        Returns True if:
        - Connection is active
        - If using new protocol, device info was received

        Returns False if:
        - Not connected
        - Using new protocol but no device info (handshake incomplete)
        """
        if not self._connected:
            return False

        if self._using_new_protocol and self._device_info is None:
            logger.warning("New protocol detected but no device info received - handshake may be incomplete")
            return False

        return True

    def _receive_loop(self) -> None:
        """Main receive loop (runs in background thread)."""
        while self._running:
            try:
                if not self._connected:
                    self._connect()
                    continue

                # Receive packet
                data = self._receive_packet()
                if data is None:
                    continue

                # Parse and buffer
                landmarks = self._parse_packet(data)
                if landmarks:
                    with self._buffer_lock:
                        self._buffer.append(landmarks)
                    self._packets_received += 1
                    self._last_receive_time = time.time()

            except socket.timeout:
                # Normal timeout, continue
                continue
            except ConnectionResetError:
                logger.warning("Connection reset by phone")
                self._handle_disconnect()
            except Exception as e:
                logger.error(f"Receive error: {e}")
                self._handle_disconnect()
                time.sleep(self.reconnect_delay)

    def _connect(self) -> None:
        """Attempt to connect to the phone."""
        sock = None
        try:
            logger.info(f"Connecting to {self.host}:{self.port}...")

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.connect_timeout)
            sock.connect((self.host, self.port))

            # Set receive timeout
            sock.settimeout(0.1)  # 100ms timeout for responsiveness

            self._socket = sock
            self._connected = True
            self._time_offset_ms = 0  # Reset time sync
            self._device_info = None
            self._using_new_protocol = False

            logger.info("Connected to phone")

            # Try to complete handshake (non-blocking, backward compatible)
            self._try_handshake()

        except socket.timeout:
            logger.info(f"Connection timed out, retrying in {self.reconnect_delay}s... (start phone app first)")
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
            self._socket = None
            time.sleep(self.reconnect_delay)
        except ConnectionRefusedError:
            logger.info(f"Waiting for phone... (retrying in {self.reconnect_delay}s)")
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
            self._socket = None
            time.sleep(self.reconnect_delay)
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
            self._socket = None
            time.sleep(self.reconnect_delay)

    def _try_handshake(self) -> None:
        """
        Attempt to receive and process handshake packet.

        The phone sends a 40-byte handshake packet immediately after connection.
        If we receive it, we send back an ACK. If the first data looks like
        a legacy landmark packet, we process it normally.

        This is backward compatible: if phone sends landmark data without
        handshake, we just process it as legacy.
        """
        if self._socket is None:
            return

        try:
            # Set short timeout for handshake
            original_timeout = self._socket.gettimeout()
            self._socket.settimeout(HANDSHAKE_TIMEOUT)

            # Peek at first 4 bytes to check for handshake magic
            # Use MSG_PEEK to not consume the data
            peek_data = self._socket.recv(4, socket.MSG_PEEK)

            if len(peek_data) < 4:
                logger.debug("Not enough data for handshake detection")
                self._socket.settimeout(original_timeout)
                return

            # Check magic bytes
            magic = struct.unpack("<I", peek_data)[0]

            if magic == HANDSHAKE_MAGIC:
                # New protocol - read full handshake
                handshake_data = self._receive_exact(HANDSHAKE_SIZE)
                if handshake_data:
                    self._process_handshake(handshake_data)

            elif magic == PACKET_MAGIC:
                # Legacy protocol - first data is landmark packet
                logger.info("Legacy phone detected (no handshake), continuing in legacy mode")
                self._using_new_protocol = False

            else:
                logger.warning(f"Unknown magic 0x{magic:08X}, assuming legacy mode")
                self._using_new_protocol = False

            self._socket.settimeout(original_timeout)

        except socket.timeout:
            logger.debug("Handshake timeout, assuming legacy mode")
            self._using_new_protocol = False
        except Exception as e:
            logger.warning(f"Handshake detection failed: {e}, using legacy mode")
            self._using_new_protocol = False

    def _receive_exact(self, size: int) -> Optional[bytes]:
        """Receive exactly 'size' bytes from socket."""
        if self._socket is None:
            return None

        data = b""
        remaining = size
        while remaining > 0:
            chunk = self._socket.recv(remaining)
            if not chunk:
                return None
            data += chunk
            remaining -= len(chunk)
        return data

    def _process_handshake(self, data: bytes) -> None:
        """
        Parse handshake packet and send ACK.

        Handshake Format (40 bytes):
        - Offset 0:   Magic (4 bytes) = 0x50484E44 ("PHND")
        - Offset 4:   Protocol Major Version (1 byte)
        - Offset 5:   Protocol Minor Version (1 byte)
        - Offset 6:   Device ID (16 bytes, UUID)
        - Offset 22:  Capabilities bitmask (4 bytes)
        - Offset 26:  Camera count (1 byte)
        - Offset 27:  Active mode (1 byte): 0=single, 1=multi
        - Offset 28:  Sensitivity (4 bytes, float)
        - Offset 32:  Reserved (8 bytes)
        """
        if len(data) != HANDSHAKE_SIZE:
            logger.warning(f"Invalid handshake size: {len(data)}")
            return

        try:
            # Parse handshake
            magic = struct.unpack_from("<I", data, 0)[0]
            if magic != HANDSHAKE_MAGIC:
                logger.warning(f"Invalid handshake magic: 0x{magic:08X}")
                return

            version_major = data[4]
            version_minor = data[5]

            # UUID: two 64-bit values (big-endian in UUID, but stored little-endian)
            uuid_msb = struct.unpack_from("<Q", data, 6)[0]
            uuid_lsb = struct.unpack_from("<Q", data, 14)[0]
            device_id = f"{uuid_msb:016x}{uuid_lsb:016x}"

            capabilities = struct.unpack_from("<I", data, 22)[0]
            camera_count = data[26]
            is_multicamera = data[27] != 0
            sensitivity = struct.unpack_from("<f", data, 28)[0]

            # Store device info
            self._device_info = HandshakeInfo(
                device_id=device_id,
                protocol_version=f"{version_major}.{version_minor}",
                capabilities=capabilities,
                camera_count=camera_count,
                is_multicamera=is_multicamera,
                sensitivity=sensitivity
            )

            logger.info(f"Handshake received: device={device_id[:8]}..., "
                       f"protocol=v{version_major}.{version_minor}, "
                       f"cameras={camera_count}, multi={is_multicamera}, "
                       f"sensitivity={sensitivity:.1f}")

            # Send ACK
            self._send_ack(version_major)
            self._using_new_protocol = True

            # Log final handshake status
            logger.info(f"Handshake complete: protocol=new, device_info_stored=True")

        except Exception as e:
            logger.error(f"Handshake parse error: {e}")

    def _send_ack(self, accepted_version: int) -> None:
        """
        Send ACK response to phone.

        ACK Format (8 bytes):
        - Offset 0:   Magic (4 bytes) = 0x41434B00 ("ACK\0")
        - Offset 4:   Accepted version (1 byte)
        - Offset 5:   Reserved (3 bytes)
        """
        if self._socket is None:
            return

        try:
            ack_data = struct.pack("<I", ACK_MAGIC)  # Magic
            ack_data += bytes([accepted_version])    # Accepted version
            ack_data += bytes([0, 0, 0])             # Reserved

            self._socket.sendall(ack_data)
            logger.debug(f"ACK sent (version {accepted_version})")

        except Exception as e:
            logger.error(f"Failed to send ACK: {e}")

    def _handle_disconnect(self) -> None:
        """Handle disconnection from phone."""
        self._connected = False
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

    def _receive_packet(self) -> Optional[bytes]:
        """
        Receive a complete packet from socket.

        Returns:
            Packet data (284 bytes), or None on error/timeout.
        """
        if self._socket is None:
            return None

        try:
            # Read exact packet size
            data = b""
            remaining = PACKET_SIZE

            while remaining > 0:
                chunk = self._socket.recv(remaining)
                if not chunk:
                    # Connection closed
                    raise ConnectionResetError("Socket closed")
                data += chunk
                remaining -= len(chunk)

            return data

        except socket.timeout:
            return None

    def _parse_packet(self, data: bytes) -> Optional[PhoneLandmarks]:
        """
        Parse binary packet into PhoneLandmarks.

        Args:
            data: Raw packet data (284 bytes).

        Returns:
            PhoneLandmarks instance, or None if invalid.
        """
        if len(data) != PACKET_SIZE:
            logger.warning(f"Invalid packet size: {len(data)}")
            self._packets_dropped += 1
            return None

        try:
            # Parse header (little-endian)
            magic = struct.unpack_from("<I", data, 0)[0]
            if magic != PACKET_MAGIC:
                logger.warning(f"Invalid magic: 0x{magic:08X}")
                self._packets_dropped += 1
                return None

            timestamp_ms = struct.unpack_from("<I", data, 4)[0]

            # Parse 21 landmarks (x, y, z as float32)
            keypoints = []
            offset = 8
            for i in range(NUM_LANDMARKS):
                x, y, z = struct.unpack_from("<fff", data, offset)
                keypoints.append((x, y, z))
                offset += 12

            # Parse metadata
            confidence = struct.unpack_from("<f", data, 260)[0]
            handedness = data[264]
            frame_id = struct.unpack_from("<I", data, 265)[0]

            return PhoneLandmarks(
                keypoints=keypoints,
                confidence=confidence,
                handedness=handedness,
                timestamp_ms=timestamp_ms,
                frame_id=frame_id,
                receive_time=time.time()
            )

        except struct.error as e:
            logger.error(f"Packet parse error: {e}")
            self._packets_dropped += 1
            return None


# =============================================================================
# ADB Port Forward Helper
# =============================================================================

def setup_adb_port_forward(port: int = 52990) -> bool:
    """
    Setup ADB port forwarding for USB connection.

    Runs: adb forward tcp:52990 tcp:52990

    Args:
        port: Port to forward.

    Returns:
        True if successful, False otherwise.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["adb", "forward", f"tcp:{port}", f"tcp:{port}"],
            capture_output=True,
            text=True,
            timeout=5.0
        )

        if result.returncode == 0:
            logger.info(f"ADB port forward established: tcp:{port}")
            return True
        else:
            logger.error(f"ADB port forward failed: {result.stderr}")
            return False

    except FileNotFoundError:
        logger.error("ADB not found. Install Android SDK platform-tools.")
        return False
    except subprocess.TimeoutExpired:
        logger.error("ADB command timed out")
        return False
    except Exception as e:
        logger.error(f"ADB port forward error: {e}")
        return False


def check_adb_device() -> bool:
    """
    Check if an Android device is connected via ADB.

    Returns:
        True if a device is connected, False otherwise.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["adb", "devices"],
            capture_output=True,
            text=True,
            timeout=5.0
        )

        lines = result.stdout.strip().split("\n")
        # First line is "List of devices attached", check for device lines
        for line in lines[1:]:
            if "\tdevice" in line:
                return True
        return False

    except Exception as e:
        logger.error(f"ADB check error: {e}")
        return False
