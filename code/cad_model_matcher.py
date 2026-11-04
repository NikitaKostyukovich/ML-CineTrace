"""3D CAD Model Matching, Keypoint PnP alignment, and Wireframe Edge Projection."""

import math
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np


class CADModelMatcher:
    """Matches a 3D CAD model against 2D video observations using PnP and Contour Alignment.
    
    This fulfills the requirement:
    "We load a 3D model, and the model recognizes and tracks the ENTIRE object in 3D (6D pose),
    not just a single point with a 2D rectangle."
    """
    
    def __init__(
        self,
        length_cm: float = 720.0,
        width_cm: float = 260.0,
        height_cm: float = 300.0
    ):
        self.length = length_cm
        self.width = width_cm
        self.height = height_cm
        
        dx = length_cm / 2.0
        dy = width_cm / 2.0
        dz = height_cm / 2.0
        
        # 3D keypoints of the vehicle in object local frame:
        # 8 corners of the chassis/body + 4 wheel contact points
        self.object_points_3d = np.array([
            [ dx, -dy, -dz],  # 0: Front-Left-Bottom
            [ dx,  dy, -dz],  # 1: Front-Right-Bottom
            [-dx,  dy, -dz],  # 2: Rear-Right-Bottom
            [-dx, -dy, -dz],  # 3: Rear-Left-Bottom
            [ dx, -dy,  dz],  # 4: Front-Left-Top (Cabin)
            [ dx,  dy,  dz],  # 5: Front-Right-Top (Cabin)
            [-dx,  dy,  dz],  # 6: Rear-Right-Top (Bed)
            [-dx, -dy,  dz],  # 7: Rear-Left-Top (Bed)
            [ dx*0.6, -dy, -dz], # 8: Front-Left Wheel
            [ dx*0.6,  dy, -dz], # 9: Front-Right Wheel
            [-dx*0.6, -dy, -dz], # 10: Rear-Left Wheel
            [-dx*0.6,  dy, -dz], # 11: Rear-Right Wheel
        ], dtype=np.float64)

    def fit_model_pnp(
        self,
        image_points_2d: np.ndarray,
        camera_k: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Solves 6D pose using Perspective-n-Point (PnP) from matched 2D-3D keypoints.
        
        Args:
            image_points_2d: (N, 2) array of detected 2D pixel coordinates.
            camera_k: 3x3 intrinsic camera matrix.
            dist_coeffs: distortion coefficients (zeros if ideal pinhole).
            
        Returns:
            (R_matrix, T_vector) or None if solve failed.
        """
        if len(image_points_2d) < 4:
            return None
            
        if dist_coeffs is None:
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)
            
        obj_pts = self.object_points_3d[:len(image_points_2d)]
        
        success, rvec, tvec = cv2.solvePnP(
            obj_pts,
            image_points_2d.astype(np.float64),
            camera_k,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP
        )
        
        if not success:
            return None
            
        R_mat, _ = cv2.Rodrigues(rvec)
        return R_mat, tvec

    def project_cad_wireframe(
        self,
        center_3d: List[float],
        heading_yaw_deg: float,
        pitch_deg: float,
        roll_deg: float,
        cam_loc: List[float],
        cam_rot: Tuple[float, float, float],
        focal_length_mm: float,
        sensor_w: float = 23.76,
        sensor_h: float = 13.365,
        img_w: int = 1920,
        img_h: int = 1080
    ) -> List[Tuple[int, int]]:
        """Projects full 3D CAD wireframe corners onto 2D image coordinates."""
        from geometry import project_3d_to_pixel
        from pose_estimator_6d import CADModelFitter
        
        fitter = CADModelFitter(default_length=self.length, default_width=self.width, default_height=self.height)
        pose = fitter.estimate_6d_pose(0, center_3d, [math.cos(math.radians(heading_yaw_deg)), math.sin(math.radians(heading_yaw_deg)), 0.0])
        
        pts_2d = []
        for corner in pose.bbox_3d_corners:
            uv, local = project_3d_to_pixel(
                corner.tolist(), cam_loc, cam_rot, focal_length_mm,
                sensor_w, sensor_h, img_w, img_h
            )
            if uv is not None and local[0] > 0:
                pts_2d.append((int(uv[0]), int(uv[1])))
            else:
                pts_2d.append(None)
        return pts_2d
