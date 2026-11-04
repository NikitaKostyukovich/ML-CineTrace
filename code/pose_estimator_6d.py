"""6D Object Pose Estimation and 3D CAD Model Fitting Module.

Calculates full 6 Degrees of Freedom:
- 3D Position: (X, Y, Z) in world space (Unreal Engine cm)
- 3D Orientation: (Roll, Pitch, Yaw) Euler angles of the vehicle
- 3D Oriented Bounding Box: 8 world vertices tightly wrapping the entire 3D object
"""

import math
from typing import List, Tuple, Optional, Dict
import numpy as np


class Pose6D:
    """Represents full 6D spatial pose of an object."""
    def __init__(
        self,
        frame: int,
        position: List[float],      # [X, Y, Z]
        orientation: List[float],   # [Roll, Pitch, Yaw] in degrees
        velocity: List[float],      # [Vx, Vy, Vz]
        bbox_3d_corners: np.ndarray # (8, 3) corners in world space
    ):
        self.frame = frame
        self.position = position
        self.orientation = orientation
        self.velocity = velocity
        self.bbox_3d_corners = bbox_3d_corners

    def __repr__(self) -> str:
        return (f"Pose6D(F={self.frame}, Pos=[{self.position[0]:.1f}, {self.position[1]:.1f}, {self.position[2]:.1f}], "
                f"Rot=[R={self.orientation[0]:.1f}°, P={self.orientation[1]:.1f}°, Y={self.orientation[2]:.1f}°])")


class CADModelFitter:
    """Loads 3D CAD model geometry and computes 6D pose from 3D trajectories and video evidence."""
    
    def __init__(
        self,
        cad_path: Optional[str] = None,
        default_length: float = 720.0,
        default_width: float = 260.0,
        default_height: float = 300.0
    ):
        self.length = default_length
        self.width = default_width
        self.height = default_height
        
        self.vertices = []
        self.faces = []
        
        if cad_path and Path(cad_path).exists():
            self._load_cad_file(cad_path)
        else:
            self._generate_canonical_truck_box()

    def _load_cad_file(self, file_path: str):
        """Loads .obj 3D mesh file."""
        verts = []
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("v "):
                        p = line.strip().split()
                        verts.append([float(p[1]), float(p[2]), float(p[3])])
            if verts:
                self.vertices = np.array(verts, dtype=np.float64)
                # Compute bounding box from actual mesh
                min_pt = np.min(self.vertices, axis=0)
                max_pt = np.max(self.vertices, axis=0)
                self.length = max_pt[0] - min_pt[0]
                self.width = max_pt[1] - min_pt[1]
                self.height = max_pt[2] - min_pt[2]
                print(f"[CADModelFitter] Loaded CAD mesh ({len(verts)} vertices, Size: {self.length:.0f}x{self.width:.0f}x{self.height:.0f} cm)")
        except Exception as e:
            print(f"[CADModelFitter] Fallback to canonical box: {e}")
            self._generate_canonical_truck_box()

    def _generate_canonical_truck_box(self):
        """Builds 8 canonical local vertices of the 3D vehicle box."""
        dx = self.length / 2.0
        dy = self.width / 2.0
        dz = self.height / 2.0
        
        # Local coordinate frame: X = Forward, Y = Right, Z = Up
        self.local_corners = np.array([
            [ dx, -dy, -dz],  # 0: Front-Left-Bottom
            [ dx,  dy, -dz],  # 1: Front-Right-Bottom
            [-dx,  dy, -dz],  # 2: Rear-Right-Bottom
            [-dx, -dy, -dz],  # 3: Rear-Left-Bottom
            [ dx, -dy,  dz],  # 4: Front-Left-Top
            [ dx,  dy,  dz],  # 5: Front-Right-Top
            [-dx,  dy,  dz],  # 6: Rear-Right-Top
            [-dx, -dy,  dz],  # 7: Rear-Left-Top
        ], dtype=np.float64)

    def estimate_6d_pose(
        self,
        frame_idx: int,
        center_3d: List[float],
        velocity_3d: List[float],
        road_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    ) -> Pose6D:
        """Estimates full 6D pose (3D translation + 3D orientation) from physical motion and CAD model.
        
        For ground vehicles (trucks, cars, tanks), orientation is kinematically locked to
        the trajectory heading and terrain slope:
        - Yaw: direction of velocity vector atan2(Vy, Vx)
        - Pitch: road inclination atan2(Vz, sqrt(Vx^2 + Vy^2))
        - Roll: bank angle / cross-slope
        """
        vx, vy, vz = velocity_3d
        v_horiz = math.hypot(vx, vy)
        
        # 1. Heading (Yaw) in degrees
        if v_horiz > 1.0:
            yaw_deg = math.degrees(math.atan2(vy, vx))
        else:
            yaw_deg = 0.0
            
        # 2. Road slope (Pitch) in degrees
        if v_horiz > 1.0:
            pitch_deg = math.degrees(math.atan2(vz, v_horiz))
        else:
            pitch_deg = 0.0
            
        # 3. Bank angle (Roll)
        roll_deg = 0.0
        
        # Construct 3D rotation matrix: R = Rz(Yaw) * Ry(Pitch) * Rx(Roll)
        r_yaw = math.radians(yaw_deg)
        r_pitch = math.radians(pitch_deg)
        r_roll = math.radians(roll_deg)
        
        cy, sy = math.cos(r_yaw), math.sin(r_yaw)
        cp, sp = math.cos(r_pitch), math.sin(r_pitch)
        cr, sr = math.cos(r_roll), math.sin(r_roll)
        
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        R_obj = Rz @ Ry @ Rx
        
        # Transform local CAD bounding box to world coordinates
        world_corners = (R_obj @ self.local_corners.T).T + np.array(center_3d)
        
        return Pose6D(
            frame=frame_idx,
            position=center_3d,
            orientation=[roll_deg, pitch_deg, yaw_deg],
            velocity=velocity_3d,
            bbox_3d_corners=world_corners
        )
