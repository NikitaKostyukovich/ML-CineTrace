"""2D Object tracker per camera to smooth and maintain persistent detections."""

from typing import Optional, Tuple, Dict
import math


class SingleCameraTracker2D:
    """Maintains a tracked 2D point (u, v) over time for one camera view."""
    
    def __init__(self, max_missed_frames: int = 15, max_speed_px: float = 80.0):
        self.max_missed = max_missed_frames
        self.max_speed_px = max_speed_px
        
        self.current_pos: Optional[Tuple[float, float]] = None
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.missed_count: int = 0
        
    def update(self, detection: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """Updates tracker state with a new frame's detection."""
        if detection is not None:
            u, v = detection
            if self.current_pos is not None:
                # Check distance from expected position
                pred_u = self.current_pos[0] + self.velocity[0]
                pred_v = self.current_pos[1] + self.velocity[1]
                dist = math.hypot(u - pred_u, v - pred_v)
                
                # If too far, reject jump if we have confident velocity
                if dist > self.max_speed_px and self.missed_count < 3:
                    # Treat as false detection, coast forward
                    self.missed_count += 1
                    self.current_pos = (pred_u, pred_v)
                    return self.current_pos
                    
                # Valid detection: update velocity (exponential moving average)
                alpha = 0.4
                du = u - self.current_pos[0]
                dv = v - self.current_pos[1]
                self.velocity = (
                    alpha * du + (1.0 - alpha) * self.velocity[0],
                    alpha * dv + (1.0 - alpha) * self.velocity[1]
                )
            self.current_pos = (u, v)
            self.missed_count = 0
            return self.current_pos
        else:
            # Missed frame: coast forward if not too many missed
            if self.current_pos is not None and self.missed_count < self.max_missed:
                self.missed_count += 1
                self.current_pos = (
                    self.current_pos[0] + self.velocity[0],
                    self.current_pos[1] + self.velocity[1]
                )
                return self.current_pos
            else:
                self.current_pos = None
                return None


class MultiCameraTracker2D:
    """Manages 2D trackers across all camera streams."""
    
    def __init__(self, camera_ids: list):
        self.trackers = {cam_id: SingleCameraTracker2D() for cam_id in camera_ids}
        
    def update(self, detections: Dict[str, Optional[Tuple[float, float]]]) -> Dict[str, Optional[Tuple[float, float]]]:
        results = {}
        for cam_id, tracker in self.trackers.items():
            det = detections.get(cam_id)
            tracked_pt = tracker.update(det)
            results[cam_id] = tracked_pt
        return results
