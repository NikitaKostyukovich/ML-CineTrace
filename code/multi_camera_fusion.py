"""Arbitrary N-Camera Multi-View Fusion Engine with Robust Huber Weighting and Dynamic Camera Rig Support."""

import math
from typing import Dict, List, Tuple, Optional
import numpy as np

from geometry import pixel_to_ray, ray_plane_intersection


class RobustMultiCameraFusion:
    """Fuses observations from arbitrary N cameras (2, 5, 8, 20+ cameras) into 3D world coordinates.
    
    Features:
    - Scales seamlessly to any number of cameras without code changes.
    - Iterative Huber M-Estimator: suppresses noisy or partially-occluded camera rays.
    - Handles single-camera and multi-camera situations dynamically.
    """
    
    def __init__(self, target_ground_z: float = 300.0, huber_k: float = 50.0):
        self.ground_z = target_ground_z
        self.huber_k = huber_k  # Threshold in cm for outlier ray down-weighting

    def triangulate_frame(
        self,
        rays: List[Tuple[List[float], List[float]]],
        plane_elevation_z: Optional[float] = None
    ) -> Optional[List[float]]:
        """Triangulates a single 3D point from N rays using robust least squares.
        
        Args:
            rays: List of (origin, direction) 3D unit rays from active cameras.
            plane_elevation_z: Estimated terrain elevation if known.
            
        Returns:
            [X, Y, Z] in world space or None.
        """
        n_rays = len(rays)
        if n_rays == 0:
            return None
            
        target_z = plane_elevation_z if plane_elevation_z is not None else self.ground_z
        
        # If only 1 ray available, intersect with terrain elevation
        if n_rays == 1:
            origin, direction = rays[0]
            return ray_plane_intersection(origin, direction, target_z)
            
        # For N >= 2 cameras, run weighted least squares (Iteratively Reweighted Least Squares - IRLS)
        origins = np.array([r[0] for r in rays], dtype=np.float64)
        dirs = np.array([r[1] for r in rays], dtype=np.float64)
        
        weights = np.ones(n_rays, dtype=np.float64)
        current_pt = None
        
        # 2 iterations of robust re-weighting
        for iteration in range(2):
            M = np.zeros((3, 3), dtype=np.float64)
            B = np.zeros(3, dtype=np.float64)
            
            for i in range(n_rays):
                w = weights[i]
                d = dirs[i]
                p = origins[i]
                
                # Projection matrix onto orthogonal complement: I - d * d^T
                proj = np.eye(3) - np.outer(d, d)
                M += w * proj
                B += w * (proj @ p)
                
            # Solve M * X = B
            try:
                current_pt = np.linalg.solve(M, B)
            except np.linalg.LinAlgError:
                current_pt = np.linalg.lstsq(M, B, rcond=None)[0]
                
            # Compute orthogonal ray distances and update Huber weights
            residuals = []
            for i in range(n_rays):
                diff = current_pt - origins[i]
                dist_to_ray = np.linalg.norm(diff - np.dot(diff, dirs[i]) * dirs[i])
                residuals.append(dist_to_ray)
                # Huber loss weighting
                if dist_to_ray <= self.huber_k:
                    weights[i] = 1.0
                else:
                    weights[i] = self.huber_k / max(1.0, dist_to_ray)
                    
        return current_pt.tolist()
