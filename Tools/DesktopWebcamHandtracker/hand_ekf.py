"""
Extended Kalman Filter for 3D hand position tracking.

Provides temporal smoothing of 3D hand positions to reduce jitter
while maintaining responsiveness to fast movements.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from logger import get_logger

logger = get_logger("HandEKF")


@dataclass
class EKFConfig:
    """Configuration for the hand EKF."""

    # Process noise - how much we expect state to change
    position_noise: float = 0.001  # meters^2 per timestep
    velocity_noise: float = 0.01   # (m/s)^2 per timestep

    # Measurement noise - how much we trust observations
    stereo_pos_noise: float = 0.02    # Stereo triangulation
    webcam_pos_noise: float = 0.03    # Webcam-only depth
    phone_pos_noise: float = 0.04     # Phone-only depth
    stereo_z_noise: float = 0.03      # Stereo depth (more uncertain)
    single_z_noise: float = 0.10      # Single-camera depth (very uncertain)

    # Jump detection threshold
    jump_threshold: float = 0.2  # 20cm position change triggers reset

    # Initialization
    initial_position_var: float = 0.1
    initial_velocity_var: float = 0.5


class HandEKF:
    """
    Extended Kalman Filter for 3D hand position.

    State vector: [x, y, z, vx, vy, vz]
    - x, y, z: Position in meters
    - vx, vy, vz: Velocity in m/s

    The filter uses a constant velocity motion model and updates
    with noisy position measurements from the camera fusion system.

    Usage:
        ekf = HandEKF()

        # Each frame:
        position = ekf.update(measured_xyz, measurement_noise)

        # On hand lost:
        ekf.reset()
    """

    def __init__(self, config: Optional[EKFConfig] = None):
        """
        Initialize the Kalman filter.

        Args:
            config: EKF configuration, or None for defaults.
        """
        self.config = config or EKFConfig()

        # State dimension
        self.n = 6  # [x, y, z, vx, vy, vz]

        # State vector
        self.x = np.zeros(self.n)

        # State covariance
        self.P = np.eye(self.n)
        self.P[:3, :3] *= self.config.initial_position_var
        self.P[3:, 3:] *= self.config.initial_velocity_var

        # Process noise covariance
        self.Q = np.diag([
            self.config.position_noise,  # x
            self.config.position_noise,  # y
            self.config.position_noise * 2,  # z (more uncertain)
            self.config.velocity_noise,  # vx
            self.config.velocity_noise,  # vy
            self.config.velocity_noise * 2  # vz
        ])

        # State transition matrix (updated with dt)
        self.F = np.eye(self.n)

        # Measurement matrix (we observe position only)
        self.H = np.zeros((3, self.n))
        self.H[:3, :3] = np.eye(3)

        # Track initialization state
        self._initialized = False
        self._last_time: Optional[float] = None

        logger.debug("HandEKF initialized")

    def predict(self, dt: float) -> np.ndarray:
        """
        Predict state forward in time.

        Args:
            dt: Time step in seconds.

        Returns:
            Predicted position [x, y, z].
        """
        if dt <= 0:
            return self.x[:3].copy()

        # Update state transition matrix
        self.F[:3, 3:] = np.eye(3) * dt

        # Predict state: x = F @ x
        self.x = self.F @ self.x

        # Predict covariance: P = F @ P @ F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q * dt

        return self.x[:3].copy()

    def update(
        self,
        measurement: np.ndarray,
        R: Optional[np.ndarray] = None,
        dt: float = 1/30
    ) -> np.ndarray:
        """
        Update state with new position measurement.

        Args:
            measurement: Observed position [x, y, z].
            R: 3x3 measurement noise covariance, or None for default.
            dt: Time since last update in seconds.

        Returns:
            Filtered position [x, y, z].
        """
        measurement = np.asarray(measurement).flatten()[:3]

        # Validate measurement - reject NaN/Inf values
        if not np.all(np.isfinite(measurement)):
            logger.warning("Invalid measurement (NaN/Inf), skipping update")
            return self.x[:3].copy() if self._initialized else np.zeros(3)

        # Initialize on first measurement
        if not self._initialized:
            self.reset(measurement)
            return measurement

        # Check for position jump
        if self._detect_jump(measurement):
            logger.warning(f"Position jump detected, resetting filter")
            self.reset(measurement)
            return measurement

        # Predict to current time
        self.predict(dt)

        # Default measurement noise
        if R is None:
            R = np.diag([
                self.config.stereo_pos_noise ** 2,
                self.config.stereo_pos_noise ** 2,
                self.config.stereo_z_noise ** 2
            ])

        # Innovation (measurement residual)
        y = measurement - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R

        # Kalman gain - use pseudo-inverse for numerical stability
        try:
            # Check if S is invertible (condition number)
            cond = np.linalg.cond(S)
            if cond > 1e10:
                logger.warning(f"Ill-conditioned innovation covariance (cond={cond:.2e}), using pseudo-inverse")
                K = self.P @ self.H.T @ np.linalg.pinv(S)
            else:
                K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError as e:
            logger.error(f"Matrix inversion failed: {e}, resetting filter")
            self.reset(measurement)
            return measurement

        # Update state
        self.x = self.x + K @ y

        # Validate state - check for NaN propagation
        if not np.all(np.isfinite(self.x)):
            logger.error("Filter state became NaN/Inf, resetting")
            self.reset(measurement)
            return measurement

        # Update covariance (Joseph form for numerical stability)
        I_KH = np.eye(self.n) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        return self.x[:3].copy()

    def reset(self, position: Optional[np.ndarray] = None) -> None:
        """
        Reset filter state.

        Args:
            position: Initial position, or None to zero.
        """
        self.x = np.zeros(self.n)

        if position is not None:
            self.x[:3] = np.asarray(position).flatten()[:3]

        self.P = np.eye(self.n)
        self.P[:3, :3] *= self.config.initial_position_var
        self.P[3:, 3:] *= self.config.initial_velocity_var

        self._initialized = position is not None
        self._last_time = None

        if position is not None:
            logger.debug(f"EKF reset to position: {position}")

    def _detect_jump(self, measurement: np.ndarray) -> bool:
        """
        Detect if measurement is a jump (sudden large change).

        Args:
            measurement: New position measurement.

        Returns:
            True if jump detected.
        """
        if not self._initialized:
            return False

        distance = np.linalg.norm(measurement - self.x[:3])
        return distance > self.config.jump_threshold

    @property
    def position(self) -> np.ndarray:
        """Get current position estimate [x, y, z]."""
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Get current velocity estimate [vx, vy, vz]."""
        return self.x[3:].copy()

    @property
    def speed(self) -> float:
        """Get current speed (magnitude of velocity)."""
        return float(np.linalg.norm(self.x[3:]))

    @property
    def is_initialized(self) -> bool:
        """Check if filter has been initialized with a measurement."""
        return self._initialized

    def get_measurement_noise(
        self,
        is_stereo: bool = True,
        is_webcam: bool = True
    ) -> np.ndarray:
        """
        Get appropriate measurement noise matrix.

        Args:
            is_stereo: True if using stereo triangulation.
            is_webcam: True if from webcam (vs phone).

        Returns:
            3x3 measurement noise covariance matrix R.
        """
        if is_stereo:
            xy_var = self.config.stereo_pos_noise ** 2
            z_var = self.config.stereo_z_noise ** 2
        elif is_webcam:
            xy_var = self.config.webcam_pos_noise ** 2
            z_var = self.config.single_z_noise ** 2
        else:
            xy_var = self.config.phone_pos_noise ** 2
            z_var = self.config.single_z_noise ** 2

        return np.diag([xy_var, xy_var, z_var])


class MultiKeypointEKF:
    """
    EKF for all 21 hand keypoints.

    Maintains separate EKF instances for each keypoint to smooth
    the entire hand skeleton.
    """

    def __init__(self, config: Optional[EKFConfig] = None):
        """
        Initialize 21 keypoint EKFs.

        Args:
            config: Shared configuration for all EKFs.
        """
        self.config = config or EKFConfig()
        self.ekfs = [HandEKF(config) for _ in range(21)]
        self._initialized = False

    def update(
        self,
        keypoints_3d: list,
        is_stereo: bool = True
    ) -> list:
        """
        Update all keypoint EKFs.

        Args:
            keypoints_3d: List of 21 (x, y, z) tuples or Keypoint3D objects.
            is_stereo: Whether stereo triangulation was used.

        Returns:
            List of 21 filtered (x, y, z) tuples.
        """
        if len(keypoints_3d) != 21:
            logger.warning(f"Expected 21 keypoints, got {len(keypoints_3d)}")
            return keypoints_3d

        filtered = []

        for i, (ekf, kp) in enumerate(zip(self.ekfs, keypoints_3d)):
            # Extract position
            if hasattr(kp, 'x'):
                pos = np.array([kp.x, kp.y, kp.z])
            else:
                pos = np.array(kp[:3])

            # Get measurement noise
            R = ekf.get_measurement_noise(is_stereo=is_stereo)

            # Update filter
            filtered_pos = ekf.update(pos, R)
            filtered.append(tuple(filtered_pos))

        self._initialized = True
        return filtered

    def reset(self) -> None:
        """Reset all keypoint EKFs."""
        for ekf in self.ekfs:
            ekf.reset()
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if EKFs are initialized."""
        return self._initialized
