"""
XREAL Eye IMU Reader for Head Movement Compensation.

This module provides TCP-based IMU data reading from XREAL One Pro glasses
for compensating cursor movement during head motion. Based on the protocol
from 6dofXrealWebcam and SamiMitwalli/One-Pro-IMU-Retriever-Demo.

Protocol Details:
    - TCP connection to 169.254.2.1:52998
    - Data: 6 floats (24 bytes) as little-endian '<6f'
    - Format: [gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z]
    - Sample rate: ~100 Hz
    - Validation: gyro ∈ [-10, +10] rad/s, accel magnitude ~9.8 m/s²

Usage:
    reader = IMUReader()
    reader.start()

    # In main loop
    imu_data = reader.get_latest_imu_data()
    if imu_data:
        print(f"Gyro: {imu_data.gyro_x:.3f} rad/s")

    # For cursor compensation
    delta_yaw, delta_pitch = reader.get_head_rotation_delta()
    compensated_x = cursor_x - delta_yaw * sensitivity

    reader.stop()
"""

import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from config import (
    XREAL_EYE_HOST,
    XREAL_EYE_IMU_PORT,
    IMU_GYRO_MAX,
    IMU_ACCEL_MIN,
    IMU_ACCEL_MAX,
    IMU_LOWPASS_ALPHA,
    IMU_CONNECT_TIMEOUT,
    IMU_READ_TIMEOUT,
    IMU_RECONNECT_DELAY,
)
from logger import get_logger

logger = get_logger("imu_reader")


@dataclass
class IMUData:
    """
    Parsed IMU sensor data from XREAL Eye glasses.

    Attributes:
        gyro_x: Angular velocity around X-axis (pitch) in rad/s
        gyro_y: Angular velocity around Y-axis (yaw) in rad/s
        gyro_z: Angular velocity around Z-axis (roll) in rad/s
        accel_x: Linear acceleration along X-axis in m/s²
        accel_y: Linear acceleration along Y-axis in m/s²
        accel_z: Linear acceleration along Z-axis in m/s²
        timestamp: Local timestamp when data was received (time.perf_counter())
    """
    gyro_x: float
    gyro_y: float
    gyro_z: float
    accel_x: float
    accel_y: float
    accel_z: float
    timestamp: float

    def __str__(self) -> str:
        return (
            f"IMU: gyro=({self.gyro_x:.3f}, {self.gyro_y:.3f}, {self.gyro_z:.3f}) "
            f"accel=({self.accel_x:.3f}, {self.accel_y:.3f}, {self.accel_z:.3f})"
        )


class IMUReader:
    """
    XREAL Eye IMU TCP client for head movement compensation.

    This class manages a background thread that continuously reads IMU data
    from the XREAL Eye glasses via TCP. It provides:
    - Latest IMU sensor readings (gyro, accel)
    - Head rotation delta for cursor compensation
    - Low-pass filtering for noise reduction
    - Automatic reconnection on connection loss
    - Thread-safe access to data

    The head rotation delta integrates gyroscope readings over time to track
    cumulative head rotation, which can be used to compensate cursor position
    when the user moves their head.
    """

    def __init__(
        self,
        host: str = XREAL_EYE_HOST,
        port: int = XREAL_EYE_IMU_PORT,
    ):
        """
        Initialize IMU reader.

        Args:
            host: XREAL Eye IP address (default: 169.254.2.1)
            port: IMU data port (default: 52998)
        """
        self.host = host
        self.port = port

        # Connection state
        self._socket: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Thread-safe data access
        self._data_lock = threading.Lock()
        self._latest_data: Optional[IMUData] = None
        self._filtered_data: Optional[IMUData] = None

        # Head rotation tracking (for cursor compensation)
        self._accumulated_yaw = 0.0
        self._accumulated_pitch = 0.0
        self._last_integration_time: Optional[float] = None

        # Statistics
        self._packets_received = 0

    @property
    def is_running(self) -> bool:
        """Check if the background thread is running."""
        return self._running

    @property
    def packets_received(self) -> int:
        """Get total number of valid IMU packets received."""
        with self._data_lock:
            return self._packets_received

    def start(self) -> None:
        """
        Start reading IMU data in background thread.

        Creates a daemon thread that continuously reads from the TCP socket,
        parses IMU packets, applies filtering, and updates the latest data.
        The thread will automatically reconnect if the connection is lost.

        Raises:
            RuntimeError: If reader is already running
        """
        if self._running:
            logger.warning("IMU reader is already running")
            return

        logger.info(f"Starting IMU reader for {self.host}:{self.port}")
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True, name="IMUReader")
        self._thread.start()

    def stop(self) -> None:
        """
        Stop reading and close connection.

        Signals the background thread to stop, closes the TCP socket, and
        waits for the thread to finish (with 1 second timeout).
        """
        if not self._running:
            return

        logger.info("Stopping IMU reader")
        self._running = False
        self._disconnect()

        if self._thread:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                logger.warning("IMU reader thread did not exit cleanly")
            self._thread = None

    def get_latest_imu_data(self) -> Optional[IMUData]:
        """
        Get the most recent filtered IMU data (thread-safe).

        Returns:
            Latest IMUData if available, None if no data received yet
        """
        with self._data_lock:
            return self._filtered_data

    def get_head_rotation_delta(self) -> Tuple[float, float]:
        """
        Get and reset accumulated head rotation since last call.

        This method returns the cumulative yaw and pitch rotation (integrated
        from gyroscope data) and resets the accumulator to zero. It's designed
        to be called once per frame to get the head motion that occurred during
        that frame.

        Returns:
            Tuple of (delta_yaw, delta_pitch) in radians
            - delta_yaw: Rotation around Y-axis (head turning left/right)
            - delta_pitch: Rotation around X-axis (head nodding up/down)
        """
        with self._data_lock:
            result = (self._accumulated_yaw, self._accumulated_pitch)
            self._accumulated_yaw = 0.0
            self._accumulated_pitch = 0.0
            return result

    def _connect(self) -> bool:
        """
        Establish TCP connection to XREAL Eye glasses.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(IMU_CONNECT_TIMEOUT)

            logger.info(f"Connecting to XREAL Eye IMU at {self.host}:{self.port}...")
            self._socket.connect((self.host, self.port))
            self._socket.settimeout(IMU_READ_TIMEOUT)

            logger.info("IMU connection established")
            return True

        except socket.timeout:
            logger.error(f"Connection timeout to {self.host}:{self.port}")
            self._disconnect()
            return False

        except ConnectionRefusedError:
            logger.error("Connection refused - are XREAL glasses connected?")
            self._disconnect()
            return False

        except OSError as e:
            logger.error(f"Connection error: {e}")
            self._disconnect()
            return False

    def _disconnect(self) -> None:
        """Close TCP connection."""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

    def _read_loop(self) -> None:
        """
        Main background thread loop.

        Continuously reads from TCP socket, parses IMU packets, applies
        filtering, and updates the latest data. Handles auto-reconnection
        on connection loss.
        """
        buffer = b''

        while self._running:
            # Connect if needed
            if self._socket is None:
                if not self._connect():
                    time.sleep(IMU_RECONNECT_DELAY)
                    continue

            # Read data
            try:
                data = self._socket.recv(4096)

                if not data:
                    logger.warning("Connection closed by XREAL Eye")
                    self._disconnect()
                    buffer = b''
                    continue

                buffer += data

                # Parse packets from buffer
                while True:
                    result = self._find_packet(buffer)
                    if result is None:
                        # Keep buffer but prevent unbounded growth
                        if len(buffer) > 10000:
                            # Keep last portion that might contain partial packet
                            buffer = buffer[-1000:]
                        break

                    imu_data, consumed = result
                    buffer = buffer[consumed:]

                    # Thread-safe update: acquire lock for filtering and state updates
                    with self._data_lock:
                        # Apply low-pass filter (needs _filtered_data which is protected)
                        filtered = self._apply_lowpass_filter(imu_data)
                        self._latest_data = imu_data
                        self._filtered_data = filtered
                        self._packets_received += 1

                        # Integrate gyro for head rotation tracking (accesses _accumulated_yaw/pitch)
                        self._integrate_rotation_locked(filtered)

            except socket.timeout:
                # Normal timeout, continue loop
                continue

            except (ConnectionResetError, BrokenPipeError, OSError) as e:
                logger.warning(f"Connection error: {e}")
                self._disconnect()
                buffer = b''

        # Cleanup on exit
        self._disconnect()
        logger.info("IMU reader stopped")

    def _find_packet(self, buffer: bytes) -> Optional[Tuple[IMUData, int]]:
        """
        Find and parse next valid IMU packet in buffer.

        Scans the buffer for a sequence of 6 floats that match the expected
        IMU data pattern: reasonable gyro values and accelerometer magnitude
        close to gravity (9.8 m/s²).

        Args:
            buffer: Byte buffer to scan

        Returns:
            (IMUData, bytes_consumed) if valid packet found, None otherwise
        """
        # Scan buffer for valid IMU data pattern (6 floats with gravity signature)
        for offset in range(0, len(buffer) - 24, 4):
            try:
                values = struct.unpack('<6f', buffer[offset:offset + 24])

                if self._is_valid_imu_data(values):
                    imu_data = IMUData(
                        gyro_x=values[0],
                        gyro_y=values[1],
                        gyro_z=values[2],
                        accel_x=values[3],
                        accel_y=values[4],
                        accel_z=values[5],
                        timestamp=time.perf_counter()
                    )
                    # Consume up to end of this IMU data
                    return (imu_data, offset + 24)

            except struct.error:
                continue

        return None

    def _is_valid_imu_data(self, values: tuple) -> bool:
        """
        Validate IMU data values.

        Checks that gyroscope values are in reasonable range and accelerometer
        magnitude is close to Earth's gravity.

        Args:
            values: Tuple of 6 floats (gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z)

        Returns:
            True if values appear to be valid IMU data
        """
        # Check gyro values are in reasonable range
        gyro_ok = all(-IMU_GYRO_MAX < v < IMU_GYRO_MAX for v in values[:3])

        # Check accelerometer magnitude is close to gravity (9.8 m/s²)
        accel_mag = (values[3]**2 + values[4]**2 + values[5]**2) ** 0.5
        accel_ok = IMU_ACCEL_MIN < accel_mag < IMU_ACCEL_MAX

        return gyro_ok and accel_ok

    def _apply_lowpass_filter(self, new_data: IMUData) -> IMUData:
        """
        Apply exponential moving average (low-pass filter) to IMU data.

        Reduces high-frequency noise in sensor readings using:
            filtered = alpha * new + (1 - alpha) * old

        Args:
            new_data: New IMU reading

        Returns:
            Filtered IMU data
        """
        # First packet, no filtering
        if self._filtered_data is None:
            return new_data

        alpha = IMU_LOWPASS_ALPHA
        prev = self._filtered_data

        return IMUData(
            gyro_x=alpha * new_data.gyro_x + (1 - alpha) * prev.gyro_x,
            gyro_y=alpha * new_data.gyro_y + (1 - alpha) * prev.gyro_y,
            gyro_z=alpha * new_data.gyro_z + (1 - alpha) * prev.gyro_z,
            accel_x=alpha * new_data.accel_x + (1 - alpha) * prev.accel_x,
            accel_y=alpha * new_data.accel_y + (1 - alpha) * prev.accel_y,
            accel_z=alpha * new_data.accel_z + (1 - alpha) * prev.accel_z,
            timestamp=new_data.timestamp
        )

    def _integrate_rotation_locked(self, imu_data: IMUData) -> None:
        """
        Integrate gyroscope data to track cumulative head rotation.

        IMPORTANT: Must be called while holding _data_lock.

        Accumulates rotation over time using:
            delta_angle = angular_velocity * dt

        Args:
            imu_data: Filtered IMU data with gyroscope readings
        """
        current_time = imu_data.timestamp

        # Initialize on first packet
        if self._last_integration_time is None:
            self._last_integration_time = current_time
            return

        # Calculate time delta
        dt = current_time - self._last_integration_time
        self._last_integration_time = current_time

        # Prevent integration of very large time gaps (e.g., after connection loss)
        if dt > 0.1:  # 100ms max dt
            return

        # Integrate gyro: delta_angle = angular_velocity * dt
        # Note: _data_lock is already held by caller
        # XREAL Eye axis mapping: gyro_z = yaw (head turning), gyro_x = pitch (nodding)
        self._accumulated_yaw += imu_data.gyro_z * dt
        self._accumulated_pitch += imu_data.gyro_x * dt


# =============================================================================
# Command-line test
# =============================================================================

def main():
    """Test IMU reader from command line."""
    import sys

    print("=" * 60)
    print("XREAL Eye IMU Reader Test")
    print("=" * 60)
    print(f"Connecting to {XREAL_EYE_HOST}:{XREAL_EYE_IMU_PORT}...")
    print("Press Ctrl+C to stop")
    print()

    reader = IMUReader()

    try:
        reader.start()

        # Wait for connection
        time.sleep(1.0)

        last_print_time = time.perf_counter()

        # Keep running until Ctrl+C
        while True:
            current_time = time.perf_counter()

            # Print status every 100ms
            if current_time - last_print_time >= 0.1:
                imu_data = reader.get_latest_imu_data()
                delta_yaw, delta_pitch = reader.get_head_rotation_delta()
                packets = reader.packets_received

                if imu_data:
                    sys.stdout.write(
                        f"\r[{packets:6d}] {imu_data} "
                        f"| Δyaw={delta_yaw:+.4f} Δpitch={delta_pitch:+.4f}    "
                    )
                    sys.stdout.flush()
                else:
                    sys.stdout.write(f"\r[{packets:6d}] Waiting for data...    ")
                    sys.stdout.flush()

                last_print_time = current_time

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        reader.stop()
        print(f"Received {reader.packets_received} packets")


if __name__ == "__main__":
    main()
