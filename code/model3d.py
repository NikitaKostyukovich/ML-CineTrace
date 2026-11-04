"""3D Object Model loader, 6D pose projection, and 3D bounding box wireframe renderer."""

import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np


class Object3DModel:
    """Manages 3D geometry of the target object (CAD model or 3D bounding box)."""
    
    def __init__(
        self,
        length_cm: float = 720.0,
        width_cm: float = 260.0,
        height_cm: float = 300.0,
        model_path: Optional[str] = None
    ):
        self.length = length_cm
        self.width = width_cm
        self.height = height_cm
        
        # 3D vertices for oriented 3D bounding box (centered at object origin)
        # In object local coordinate system (X=Forward, Y=Right, Z=Up)
        dx = length_cm / 2.0
        dy = width_cm / 2.0
        dz = height_cm / 2.0
        
        # 8 corners of 3D box
        self.local_box_vertices = np.array([
            [ dx, -dy, -dz],  # 0: Front-Left-Bottom
            [ dx,  dy, -dz],  # 1: Front-Right-Bottom
            [-dx,  dy, -dz],  # 2: Rear-Right-Bottom
            [-dx, -dy, -dz],  # 3: Rear-Left-Bottom
            [ dx, -dy,  dz],  # 4: Front-Left-Top
            [ dx,  dy,  dz],  # 5: Front-Right-Top
            [-dx,  dy,  dz],  # 6: Rear-Right-Top
            [-dx, -dy,  dz],  # 7: Rear-Left-Top
        ], dtype=np.float64)
        
        # 12 edges connecting the 8 vertices
        self.box_edges = [
            # Bottom base (0-1-2-3)
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Top roof (4-5-6-7)
            (4, 5), (5, 6), (6, 7), (7, 4),
            # Vertical pillars (pillars connecting bottom to top)
            (0, 4), (1, 5), (2, 6), (3, 7),
            # Front directional cross (0-5, 1-4)
            (0, 5), (1, 4)
        ]
        
        # Optional mesh vertices loaded from OBJ file
        self.mesh_vertices: Optional[np.ndarray] = None
        if model_path:
            self.load_obj_file(model_path)

    def load_obj_file(self, file_path: str):
        """Loads 3D vertices from an .obj CAD file."""
        verts = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("v "):
                        parts = line.strip().split()
                        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if verts:
                self.mesh_vertices = np.array(verts, dtype=np.float64)
                print(f"[Object3DModel] Loaded {len(verts)} vertices from {file_path}")
        except Exception as e:
            print(f"[Object3DModel] Failed to load {file_path}: {e}")

    def get_world_vertices(
        self,
        center_world: List[float],
        heading_yaw_deg: float,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0
    ) -> np.ndarray:
        """Transforms 3D bounding box corners into world coordinates given 6D pose."""
        # Build object rotation matrix (Yaw, Pitch, Roll)
        r_yaw = math.radians(heading_yaw_deg)
        r_pitch = math.radians(pitch_deg)
        r_roll = math.radians(roll_deg)
        
        cy, sy = math.cos(r_yaw), math.sin(r_yaw)
        cp, sp = math.cos(r_pitch), math.sin(r_pitch)
        cr, sr = math.cos(r_roll), math.sin(r_roll)
        
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        
        R = Rz @ Ry @ Rx
        
        # Transform local box corners to world coordinates
        world_corners = (R @ self.local_box_vertices.T).T + np.array(center_world)
        return world_corners

    def render_3d_box_on_frame(
        self,
        frame_bgr: np.ndarray,
        center_world: List[float],
        heading_yaw_deg: float,
        cam_loc: List[float],
        cam_rot: Tuple[float, float, float],
        focal_length_mm: float,
        box_color: Tuple[int, int, int] = (0, 255, 255),
        line_thickness: int = 2
    ):
        """Projects and renders oriented 3D wireframe bounding box onto camera video frame."""
        from geometry import project_3d_to_pixel
        
        world_corners = self.get_world_vertices(center_world, heading_yaw_deg)
        
        # Project all 8 corners onto camera image
        projected_pts = []
        for corner in world_corners:
            uv, local = project_3d_to_pixel(corner, cam_loc, cam_rot, focal_length_mm)
            if uv is not None and local[0] > 0:
                projected_pts.append((int(uv[0]), int(uv[1])))
            else:
                projected_pts.append(None)
                
        # Draw 12 3D edges
        for idx, (p1_idx, p2_idx) in enumerate(self.box_edges):
            pt1 = projected_pts[p1_idx]
            pt2 = projected_pts[p2_idx]
            if pt1 is not None and pt2 is not None:
                # Color front face differently (cyan) to show direction of vehicle
                edge_col = (0, 255, 100) if idx >= 12 else box_color
                cv2.line(frame_bgr, pt1, pt2, edge_col, line_thickness, cv2.LINE_AA)
                
        # Draw centroid marker
        uv_center, local_center = project_3d_to_pixel(center_world, cam_loc, cam_rot, focal_length_mm)
        if uv_center is not None and local_center[0] > 0:
            cx, cy = int(uv_center[0]), int(uv_center[1])
            cv2.circle(frame_bgr, (cx, cy), 4, (0, 0, 255), -1)
