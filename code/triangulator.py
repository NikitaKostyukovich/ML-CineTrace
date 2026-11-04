"""3D Triangulation and Multi-View Ray Fusion Engine."""

from typing import Dict, List, Tuple, Optional
import math

from geometry import pixel_to_ray, ray_plane_intersection, triangulate_rays_midpoint
from camera_loader import CameraDataset


class MultiViewTriangulator:
    """Combines 2D camera detections into 3D world coordinates.
    
    Supports:
    1. Multi-camera least-squares 3D ray intersection (>= 2 views).
    2. Terrain-adaptive height interpolation for single-view segments.
    """
    
    def __init__(self, camera_dataset: CameraDataset, target_plane_z: float = 300.0):
        self.camera_dataset = camera_dataset
        self.plane_z = target_plane_z

    def triangulate_sequence(
        self,
        observations_sequence: List[Dict[str, Optional[Tuple[float, float]]]]
    ) -> List[Optional[List[float]]]:
        """Triangulates an entire sequence of frames using multi-view fusion and elevation propagation.
        
        Args:
            observations_sequence: List of dicts {cam_id: (u, v) or None} for each frame.
            
        Returns:
            List of [X, Y, Z] coordinates in UE cm for each frame.
        """
        n_frames = len(observations_sequence)
        trajectory_3d: List[Optional[List[float]]] = [None] * n_frames
        
        # Pass 1: Triangulate all frames with >= 2 camera rays
        for f in range(n_frames):
            obs = observations_sequence[f]
            rays = []
            for cam_id, pt_2d in obs.items():
                if pt_2d is None:
                    continue
                u, v = pt_2d
                state = self.camera_dataset.get_camera_state(cam_id, f)
                if state is None:
                    continue
                origin, ray_dir = pixel_to_ray(
                    u, v,
                    state.loc,
                    state.rot,
                    state.focal_length_mm
                )
                rays.append((origin, ray_dir))
                
            if len(rays) >= 2:
                pt_3d = triangulate_rays_midpoint(rays)
                trajectory_3d[f] = pt_3d

        # Pass 2: For single-camera frames, interpolate elevation Z from adjacent multi-view anchors
        known_frames = [f for f in range(n_frames) if trajectory_3d[f] is not None]
        
        for f in range(n_frames):
            if trajectory_3d[f] is not None:
                continue
                
            obs = observations_sequence[f]
            single_ray = None
            for cam_id, pt_2d in obs.items():
                if pt_2d is not None:
                    u, v = pt_2d
                    state = self.camera_dataset.get_camera_state(cam_id, f)
                    if state is not None:
                        single_ray = pixel_to_ray(u, v, state.loc, state.rot, state.focal_length_mm)
                        break
                        
            if single_ray is None:
                continue
                
            origin, ray_dir = single_ray
            
            # Determine target Z
            if not known_frames:
                target_z = self.plane_z
            else:
                prev_f = max([kf for kf in known_frames if kf < f], default=known_frames[0])
                next_f = min([kf for kf in known_frames if kf > f], default=known_frames[-1])
                
                if prev_f == next_f or prev_f > f or next_f < f:
                    anchor = prev_f if prev_f != f else next_f
                    target_z = trajectory_3d[anchor][2]
                else:
                    alpha = (f - prev_f) / (next_f - prev_f)
                    target_z = (1.0 - alpha) * trajectory_3d[prev_f][2] + alpha * trajectory_3d[next_f][2]
                    
            pt_plane = ray_plane_intersection(origin, ray_dir, target_z)
            if pt_plane is not None:
                trajectory_3d[f] = pt_plane

        return trajectory_3d
