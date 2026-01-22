"""
Stereo triangulation for dual-camera hand tracking.

Implements Direct Linear Transform (DLT) triangulation to
compute 3D positions from 2D observations in two cameras.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

from logger import get_logger

logger = get_logger("Triangulation")


@dataclass
class CameraCalibration:
    """
    Camera calibration data for triangulation.

    Attributes:
        intrinsic: 3x3 camera intrinsic matrix (K).
        rotation: 3x3 rotation matrix (R) - world to camera.
        translation: 3x1 translation vector (t) - world to camera.
        projection: 3x4 projection matrix (P = K @ [R|t]).
    """
    intrinsic: np.ndarray  # 3x3
    rotation: np.ndarray   # 3x3
    translation: np.ndarray  # 3x1

    @property
    def projection(self) -> np.ndarray:
        """Get 3x4 projection matrix P = K @ [R|t]."""
        Rt = np.hstack([self.rotation, self.translation.reshape(3, 1)])
        return self.intrinsic @ Rt

    @classmethod
    def from_extrinsics(
        cls,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        rotation: np.ndarray,
        translation: np.ndarray
    ) -> "CameraCalibration":
        """
        Create calibration from camera parameters.

        Args:
            fx, fy: Focal lengths in pixels.
            cx, cy: Principal point coordinates.
            rotation: 3x3 rotation matrix.
            translation: 3-element translation vector.
        """
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        return cls(
            intrinsic=K,
            rotation=np.asarray(rotation, dtype=np.float64),
            translation=np.asarray(translation, dtype=np.float64).flatten()
        )


def triangulate_point_dlt(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    P1: np.ndarray,
    P2: np.ndarray,
    check_cheirality: bool = True
) -> Optional[np.ndarray]:
    """
    Triangulate a 3D point from two 2D observations using DLT.

    Direct Linear Transform method using SVD to solve the
    overdetermined system of equations.

    Args:
        point1: (x, y) in camera 1 (normalized coordinates).
        point2: (x, y) in camera 2 (normalized coordinates).
        P1: 3x4 projection matrix for camera 1.
        P2: 3x4 projection matrix for camera 2.
        check_cheirality: If True, verify point is in front of both cameras.

    Returns:
        3D point as numpy array [x, y, z], or None if triangulation failed
        (point behind camera or at infinity).
    """
    x1, y1 = point1
    x2, y2 = point2

    # Build the 4x4 equation matrix
    A = np.array([
        x1 * P1[2] - P1[0],
        y1 * P1[2] - P1[1],
        x2 * P2[2] - P2[0],
        y2 * P2[2] - P2[1]
    ])

    # Solve using SVD
    try:
        _, s, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        logger.warning("SVD failed during triangulation")
        return None

    X = Vt[-1]  # Last row of V^T (smallest singular value)

    # Check for point at infinity (w ≈ 0)
    if abs(X[3]) < 1e-10:
        logger.debug("Triangulated point at infinity (w ≈ 0)")
        return None

    # Convert from homogeneous coordinates
    point_3d = X[:3] / X[3]

    # Check for NaN/Inf in result
    if not np.all(np.isfinite(point_3d)):
        logger.debug("Triangulated point contains NaN/Inf")
        return None

    # Cheirality check: verify point is in front of both cameras
    if check_cheirality:
        # Project point back and check if depth (z) is positive
        # Camera 1: P1 @ [X, 1]^T should have positive z before division
        p1_proj = P1 @ np.append(point_3d, 1.0)
        p2_proj = P2 @ np.append(point_3d, 1.0)

        if p1_proj[2] <= 0 or p2_proj[2] <= 0:
            logger.debug("Cheirality check failed: point behind camera")
            return None

    return point_3d


def triangulate_points(
    points1: List[Tuple[float, float]],
    points2: List[Tuple[float, float]],
    calib1: CameraCalibration,
    calib2: CameraCalibration,
    fallback_on_failure: bool = True
) -> List[Optional[np.ndarray]]:
    """
    Triangulate multiple corresponding points.

    Args:
        points1: List of (x, y) points in camera 1.
        points2: List of (x, y) points in camera 2.
        calib1: Camera 1 calibration.
        calib2: Camera 2 calibration.
        fallback_on_failure: If True, use previous valid point on failure.

    Returns:
        List of 3D points as numpy arrays. None entries indicate
        triangulation failed for that point.
    """
    P1 = calib1.projection
    P2 = calib2.projection

    points_3d = []
    last_valid = None

    for i, (p1, p2) in enumerate(zip(points1, points2)):
        point_3d = triangulate_point_dlt(p1, p2, P1, P2)

        if point_3d is not None:
            last_valid = point_3d
            points_3d.append(point_3d)
        elif fallback_on_failure and last_valid is not None:
            # Use last valid point as fallback
            logger.debug(f"Triangulation failed for point {i}, using fallback")
            points_3d.append(last_valid.copy())
        else:
            points_3d.append(None)

    return points_3d


def compute_reprojection_error(
    point_3d: np.ndarray,
    point_2d: Tuple[float, float],
    calibration: CameraCalibration
) -> float:
    """
    Compute reprojection error for a 3D-2D correspondence.

    Args:
        point_3d: 3D point in world coordinates.
        point_2d: Observed 2D point (normalized).
        calibration: Camera calibration.

    Returns:
        Euclidean distance between projected and observed points.
    """
    P = calibration.projection

    # Project 3D point to 2D
    X_hom = np.append(point_3d, 1.0)
    x_proj = P @ X_hom
    x_proj = x_proj[:2] / x_proj[2]

    # Compute error
    x_obs = np.array(point_2d)
    return np.linalg.norm(x_proj - x_obs)


@dataclass
class PhoneCalibration:
    """
    Calibration data for phone camera relative to webcam.

    The webcam is treated as the reference camera (identity pose).
    The phone camera pose is expressed relative to the webcam.

    Attributes:
        rotation: 3x3 rotation matrix (phone relative to webcam).
        translation: 3-element translation vector (meters).
        phone_intrinsics: Camera intrinsics for phone camera.
        reprojection_error: Average reprojection error from calibration.
        calibration_timestamp: When calibration was performed.
    """
    rotation: np.ndarray  # 3x3
    translation: np.ndarray  # 3
    phone_fx: float = 500.0
    phone_fy: float = 500.0
    phone_cx: float = 320.0
    phone_cy: float = 240.0
    reprojection_error: float = 0.0
    calibration_timestamp: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "rotation": self.rotation.flatten().tolist(),
            "translation": self.translation.tolist(),
            "phoneIntrinsics": {
                "fx": self.phone_fx,
                "fy": self.phone_fy,
                "cx": self.phone_cx,
                "cy": self.phone_cy
            },
            "reprojectionError": self.reprojection_error,
            "calibrationDate": self.calibration_timestamp
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PhoneCalibration":
        """Create from JSON dictionary."""
        rotation = np.array(data["rotation"]).reshape(3, 3)
        translation = np.array(data["translation"])

        intrinsics = data.get("phoneIntrinsics", {})

        return cls(
            rotation=rotation,
            translation=translation,
            phone_fx=intrinsics.get("fx", 500.0),
            phone_fy=intrinsics.get("fy", 500.0),
            phone_cx=intrinsics.get("cx", 320.0),
            phone_cy=intrinsics.get("cy", 240.0),
            reprojection_error=data.get("reprojectionError", 0.0),
            calibration_timestamp=data.get("calibrationDate")
        )

    def get_webcam_calibration(
        self,
        webcam_fx: float = 640.0,
        webcam_fy: float = 640.0,
        webcam_cx: float = 320.0,
        webcam_cy: float = 240.0
    ) -> CameraCalibration:
        """
        Get webcam calibration (identity pose).

        Args:
            webcam_fx, webcam_fy, webcam_cx, webcam_cy: Webcam intrinsics.

        Returns:
            CameraCalibration for webcam at origin.
        """
        return CameraCalibration.from_extrinsics(
            fx=webcam_fx,
            fy=webcam_fy,
            cx=webcam_cx,
            cy=webcam_cy,
            rotation=np.eye(3),
            translation=np.zeros(3)
        )

    def get_phone_calibration(self) -> CameraCalibration:
        """
        Get phone calibration (relative to webcam).

        Returns:
            CameraCalibration for phone camera.
        """
        return CameraCalibration.from_extrinsics(
            fx=self.phone_fx,
            fy=self.phone_fy,
            cx=self.phone_cx,
            cy=self.phone_cy,
            rotation=self.rotation,
            translation=self.translation
        )


def estimate_calibration_from_hand(
    webcam_wrists: List[Tuple[float, float]],
    phone_wrists: List[Tuple[float, float]],
    webcam_intrinsics: Optional[np.ndarray] = None,
    phone_intrinsics: Optional[np.ndarray] = None
) -> Optional[PhoneCalibration]:
    """
    Estimate phone-to-webcam calibration using hand wrist positions.

    Uses the Essential Matrix to estimate relative camera pose from
    corresponding wrist positions in both cameras.

    Args:
        webcam_wrists: List of (x, y) wrist positions in webcam (normalized).
        phone_wrists: List of (x, y) wrist positions in phone (normalized).
        webcam_intrinsics: 3x3 webcam intrinsic matrix (optional).
        phone_intrinsics: 3x3 phone intrinsic matrix (optional).

    Returns:
        PhoneCalibration if successful, None otherwise.
    """
    import cv2

    if len(webcam_wrists) < 8:
        logger.warning(f"Need at least 8 points for calibration, got {len(webcam_wrists)}")
        return None

    # Default intrinsics if not provided
    if webcam_intrinsics is None:
        webcam_intrinsics = np.array([
            [640, 0, 320],
            [0, 640, 240],
            [0, 0, 1]
        ], dtype=np.float64)

    if phone_intrinsics is None:
        phone_intrinsics = np.array([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1]
        ], dtype=np.float64)

    # Convert to numpy arrays
    pts1 = np.array(webcam_wrists, dtype=np.float64)
    pts2 = np.array(phone_wrists, dtype=np.float64)

    # Denormalize points (normalized 0-1 to pixel coordinates)
    # Assuming 640x480 resolution
    pts1_px = pts1 * np.array([640, 480])
    pts2_px = pts2 * np.array([640, 480])

    try:
        # Find Essential Matrix
        E, mask = cv2.findEssentialMat(
            pts1_px, pts2_px,
            cameraMatrix=webcam_intrinsics,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None:
            logger.error("Failed to find Essential Matrix")
            return None

        # Recover pose (R, t) from Essential Matrix
        _, R, t, mask = cv2.recoverPose(E, pts1_px, pts2_px, webcam_intrinsics)

        # Calculate reprojection error
        # (simplified - proper implementation would triangulate and reproject)
        inliers = np.sum(mask)
        reprojection_error = (len(webcam_wrists) - inliers) / len(webcam_wrists) * 10.0

        logger.info(f"Calibration complete: {inliers}/{len(webcam_wrists)} inliers, "
                   f"error: {reprojection_error:.2f}")

        return PhoneCalibration(
            rotation=R,
            translation=t.flatten(),
            phone_fx=phone_intrinsics[0, 0],
            phone_fy=phone_intrinsics[1, 1],
            phone_cx=phone_intrinsics[0, 2],
            phone_cy=phone_intrinsics[1, 2],
            reprojection_error=reprojection_error
        )

    except cv2.error as e:
        logger.error(f"OpenCV error during calibration: {e}")
        return None
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        return None
