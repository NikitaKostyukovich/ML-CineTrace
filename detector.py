"""Object detection module supporting YOLO deep learning, OpenCV motion detection, and synthetic oracle mode."""

from typing import Optional, Tuple, Dict, List
import math
import random


class BaseDetector:
    """Base detector interface."""
    def detect_frame(self, frame_img, cam_id: str, frame_idx: int) -> Optional[Tuple[float, float]]:
        """Returns detected (u, v) image coordinates or None if not detected."""
        raise NotImplementedError


class YoloDetector(BaseDetector):
    """Deep learning detector using YOLOv8/YOLO11 (ultralytics)."""
    
    def __init__(self, model_weights: str = "yolov8m.pt", conf_thresh: float = 0.25):
        self.conf_thresh = conf_thresh
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_weights)
            print(f"[YoloDetector] Successfully loaded {model_weights}")
        except Exception as e:
            print(f"[YoloDetector] Ultralytics/Torch not available ({e}).")

    def detect_frame(self, frame_bgr, cam_id: str, frame_idx: int) -> Optional[Tuple[float, float]]:
        if self.model is None or frame_bgr is None:
            return None
        
        results = self.model(frame_bgr, conf=self.conf_thresh, verbose=False)
        boxes = results[0].boxes
        if len(boxes) == 0:
            return None
            
        # Class 7 is 'truck', class 2 is 'car' in COCO dataset
        best_box = None
        max_conf = 0.0
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls_id in [7, 2] and conf > max_conf:
                max_conf = conf
                best_box = box.xyxy[0].cpu().numpy()
                
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            
        return None


class MotionFlowDetector(BaseDetector):
    """Computer vision motion / frame differencing detector for static cameras."""
    
    def __init__(self, min_area: float = 20.0):
        self.min_area = min_area
        self.prev_frames: Dict[str, any] = {}
        
    def detect_frame(self, frame_bgr, cam_id: str, frame_idx: int) -> Optional[Tuple[float, float]]:
        try:
            import cv2
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            prev = self.prev_frames.get(cam_id)
            self.prev_frames[cam_id] = gray
            
            if prev is None:
                return None
                
            delta = cv2.absdiff(prev, gray)
            _, thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_cnt = None
            max_area = 0.0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > self.min_area and area > max_area:
                    max_area = area
                    best_cnt = cnt
                    
            if best_cnt is not None:
                M = cv2.moments(best_cnt)
                if M["m00"] > 0:
                    cX = M["m10"] / M["m00"]
                    cY = M["m01"] / M["m00"]
                    return (float(cX), float(cY))
        except Exception:
            pass
        return None


class SyntheticOracleDetector(BaseDetector):
    """Oracle detector for end-to-end geometry and tracking validation.
    
    Projects the exact ground truth position with customizable pixel noise and occlusion.
    """
    
    def __init__(self, gt_trajectory: Dict[int, List[float]], camera_dataset, noise_sigma_px: float = 1.0):
        self.gt_trajectory = gt_trajectory
        self.camera_dataset = camera_dataset
        self.noise_sigma = noise_sigma_px
        from geometry import project_3d_to_pixel
        self.project_fn = project_3d_to_pixel
        
    def detect_frame(self, frame_bgr, cam_id: str, frame_idx: int) -> Optional[Tuple[float, float]]:
        target_3d = self.gt_trajectory.get(frame_idx)
        if target_3d is None:
            return None
            
        cam_state = self.camera_dataset.get_camera_state(cam_id, frame_idx)
        if cam_state is None:
            return None
            
        uv, local = self.project_fn(
            target_3d,
            cam_state.loc,
            cam_state.rot,
            cam_state.focal_length_mm
        )
        if uv is None:
            return None
            
        u, v = uv
        if 0 <= u <= 1920 and 0 <= v <= 1080:
            if self.noise_sigma > 0:
                u += random.gauss(0, self.noise_sigma)
                v += random.gauss(0, self.noise_sigma)
            return (u, v)
            
        return None
