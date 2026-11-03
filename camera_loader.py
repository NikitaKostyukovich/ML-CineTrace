"""Camera dataset loader and ground truth parser."""

import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from config import CAMERAS, GROUND_TRUTH_CSV


class CameraState:
    """Represents a single camera's state at a given frame."""
    def __init__(
        self,
        frame: int,
        loc: List[float],
        rot: Tuple[float, float, float],
        focal_length_mm: float
    ):
        self.frame = frame
        self.loc = loc  # [X, Y, Z] in UE cm
        self.rot = rot  # (Roll, Pitch, Yaw) in degrees
        self.focal_length_mm = focal_length_mm

    def __repr__(self) -> str:
        return f"CameraState(F={self.frame}, Loc={self.loc}, Rot={self.rot}, Focal={self.focal_length_mm})"


class CameraDataset:
    """Loads and manages per-frame parameters for all 5 cameras."""
    
    def __init__(self):
        self.cameras: Dict[str, Dict[int, CameraState]] = {}
        self.focal_lengths: Dict[str, float] = {}
        self._load_all_cameras()
        
    def _load_all_cameras(self):
        for cam_id, cam_info in CAMERAS.items():
            csv_path = cam_info["csv"]
            default_focal = cam_info["default_focal"]
            
            self.cameras[cam_id] = {}
            if not os.path.exists(csv_path):
                print(f"[Warning] Camera CSV not found: {csv_path}")
                continue
                
            with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [line.strip() for line in f if line.strip()]
                
            header = lines[0].split(";")
            focal = default_focal
            # Extract focal length from header if present
            for i, part in enumerate(header):
                if "focal" in part.lower() and i + 1 < len(header):
                    try:
                        focal = float(header[i + 1].replace(",", "."))
                    except ValueError:
                        pass
            self.focal_lengths[cam_id] = focal
            
            for line in lines[1:]:
                parts = line.split(";")
                if len(parts) < 7:
                    continue
                try:
                    frame = int(parts[0])
                    loc_x = float(parts[1].replace(",", "."))
                    loc_y = float(parts[2].replace(",", "."))
                    loc_z = float(parts[3].replace(",", "."))
                    rot_roll = float(parts[4].replace(",", "."))
                    rot_pitch = float(parts[5].replace(",", "."))
                    rot_yaw = float(parts[6].replace(",", "."))
                    
                    state = CameraState(
                        frame=frame,
                        loc=[loc_x, loc_y, loc_z],
                        rot=(rot_roll, rot_pitch, rot_yaw),
                        focal_length_mm=focal
                    )
                    self.cameras[cam_id][frame] = state
                except (ValueError, IndexError):
                    continue
                    
    def get_camera_state(self, cam_id: str, frame: int) -> Optional[CameraState]:
        """Get camera state for specific camera at frame."""
        cam_dict = self.cameras.get(cam_id)
        if not cam_dict:
            return None
        return cam_dict.get(frame)


def load_ground_truth(csv_path: Optional[Path] = None) -> Dict[int, List[float]]:
    """Loads ground truth trajectory from object_transform_SM_AA_track.csv.
    
    Returns:
        dict: {frame_idx: [X, Y, Z]}
    """
    path = csv_path or GROUND_TRUTH_CSV
    gt_trajectory: Dict[int, List[float]] = {}
    
    if not os.path.exists(path):
        print(f"[Warning] Ground truth CSV not found: {path}")
        return gt_trajectory
        
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    for line in lines[1:]:
        parts = line.split(";")
        if len(parts) < 4:
            continue
        try:
            frame = int(parts[0])
            x = float(parts[1].replace(",", "."))
            y = float(parts[2].replace(",", "."))
            z = float(parts[3].replace(",", "."))
            gt_trajectory[frame] = [x, y, z]
        except (ValueError, IndexError):
            continue
            
    return gt_trajectory
