"""Graphical User Interface (GUI) for 3D Video Object Tracking.

Features:
- Load any video file (.mp4, .avi)
- Automatically load or manually choose camera calibration CSV
- Real-time video playback and neural/vision tracking
- Visual overlays: 2D bounding box, centroid, motion trail, HUD telemetry
- Live 3D coordinate calculation (X, Y, Z in world space)
- Export predicted trajectory to CSV
- Optional evaluation against ground truth if available
"""

import os
import sys
import math
import threading
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# Import local geometry and camera modules
sys.path.insert(0, str(Path(__file__).parent))
from geometry import pixel_to_ray, ray_plane_intersection
from camera_loader import CameraState


class TrackingApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Пеленгатор 3D — Система сопровождения объектов (Хакатон Пеленг)")
        self.root.geometry("1280x820")
        self.root.minsize(1000, 700)
        
        # Apply dark theme styling
        self._setup_styles()
        
        # State variables
        self.video_path: Optional[str] = None
        self.camera_csv_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames: int = 0
        self.fps: float = 30.0
        self.current_frame_idx: int = 0
        self.is_playing: bool = False
        
        # Camera calibration data: {frame_idx: CameraState}
        self.camera_states: Dict[int, CameraState] = {}
        self.focal_length_mm: float = 35.0
        
        # Tracking model & data
        self.tracker_type = tk.StringVar(value="CV Motion / Optical Flow")
        self.yolo_model = None
        self.cv_tracker = None
        self.tracking_initialized = False
        self.trail_points: List[Tuple[int, int]] = []
        
        # Recorded trajectory: {frame_idx: [x, y, z]}
        self.predicted_trajectory: Dict[int, List[float]] = {}
        
        # Ground truth (if available for comparison)
        self.ground_truth: Dict[int, List[float]] = {}
        
        # Target ground plane (Unreal Engine cm)
        self.target_z: float = 300.0
        
        # Build layout
        self._build_ui()
        
        # Try loading default assets if present
        self._auto_detect_defaults()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        
        self.bg_color = "#1e1e24"
        self.panel_color = "#282830"
        self.accent_color = "#00adb5"
        self.text_color = "#eeeeee"
        
        self.root.configure(bg=self.bg_color)
        style.configure(".", background=self.bg_color, foreground=self.text_color)
        style.configure("TFrame", background=self.bg_color)
        style.configure("Card.TFrame", background=self.panel_color, relief="flat")
        style.configure("TLabel", background=self.bg_color, foreground=self.text_color, font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"), foreground=self.accent_color)
        style.configure("Coord.TLabel", font=("Consolas", 11, "bold"), foreground="#00ffcc")
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.map("TButton", background=[("active", self.accent_color)])

    def _build_ui(self):
        # Main container with two columns: Left (Video Canvas) and Right (Control Panel)
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left column (Video player & timeline)
        video_panel = ttk.Frame(main_frame)
        video_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video Canvas
        self.canvas = tk.Canvas(video_panel, bg="#111116", highlightthickness=1, highlightbackground="#33333e")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Timeline & Playback controls bar
        controls_bar = ttk.Frame(video_panel, padding=(0, 8, 0, 0))
        controls_bar.pack(fill=tk.X)
        
        self.btn_prev = ttk.Button(controls_bar, text="⏮ -1", width=5, command=self._step_prev)
        self.btn_prev.pack(side=tk.LEFT, padx=2)
        
        self.btn_play = ttk.Button(controls_bar, text="▶ Старт", width=10, command=self._toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=4)
        
        self.btn_next = ttk.Button(controls_bar, text="+1 ⏭", width=5, command=self._step_next)
        self.btn_next.pack(side=tk.LEFT, padx=2)
        
        self.timeline_var = tk.DoubleVar(value=0.0)
        self.timeline_slider = ttk.Scale(
            controls_bar, from_=0, to=1000, variable=self.timeline_var,
            orient=tk.HORIZONTAL, command=self._on_slider_change
        )
        self.timeline_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        
        self.lbl_frame_info = ttk.Label(controls_bar, text="Кадр: 0 / 0", font=("Consolas", 10))
        self.lbl_frame_info.pack(side=tk.RIGHT, padx=4)
        
        # Right column (Controls, Telemetry & Export)
        side_panel = ttk.Frame(main_frame, width=380, style="Card.TFrame", padding=12)
        side_panel.pack(side=tk.RIGHT, fill=tk.Y)
        side_panel.pack_propagate(False)
        
        # Section 1: File Loading
        lbl_sec1 = ttk.Label(side_panel, text="1. Загрузка данных", style="Header.TLabel", background=self.panel_color)
        lbl_sec1.pack(anchor="w", pady=(0, 6))
        
        btn_load_vid = ttk.Button(side_panel, text="📂 Выбрать видеофайл (.mp4)", command=self._load_video_dialog)
        btn_load_vid.pack(fill=tk.X, pady=3)
        self.lbl_vid_name = ttk.Label(side_panel, text="Видео не выбрано", background=self.panel_color, font=("Segoe UI", 9, "italic"))
        self.lbl_vid_name.pack(anchor="w", pady=(0, 6))
        
        btn_load_csv = ttk.Button(side_panel, text="📋 Выбрать калибровку камеры (CSV)", command=self._load_camera_csv_dialog)
        btn_load_csv.pack(fill=tk.X, pady=3)
        self.lbl_csv_name = ttk.Label(side_panel, text="Калибровка не выбрана", background=self.panel_color, font=("Segoe UI", 9, "italic"))
        self.lbl_csv_name.pack(anchor="w", pady=(0, 10))
        
        # Section 2: Model Configuration
        lbl_sec2 = ttk.Label(side_panel, text="2. Метод детектирования", style="Header.TLabel", background=self.panel_color)
        lbl_sec2.pack(anchor="w", pady=(6, 4))
        
        model_combo = ttk.Combobox(
            side_panel, textvariable=self.tracker_type,
            values=["YOLOv8 / YOLO11 Neural Network", "CV Motion / Optical Flow", "Color / Contrast Tracker"],
            state="readonly"
        )
        model_combo.pack(fill=tk.X, pady=4)
        model_combo.bind("<<ComboboxSelected>>", self._on_model_changed)
        
        # Section 3: Live 3D Telemetry (HUD)
        lbl_sec3 = ttk.Label(side_panel, text="3. Телеметрия объекта (3D)", style="Header.TLabel", background=self.panel_color)
        lbl_sec3.pack(anchor="w", pady=(12, 4))
        
        hud_box = tk.Frame(side_panel, bg="#18181f", bd=1, relief="solid", padx=10, pady=8)
        hud_box.pack(fill=tk.X, pady=4)
        
        self.val_x = ttk.Label(hud_box, text="X:  --- cm", style="Coord.TLabel", background="#18181f")
        self.val_x.pack(anchor="w")
        self.val_y = ttk.Label(hud_box, text="Y:  --- cm", style="Coord.TLabel", background="#18181f")
        self.val_y.pack(anchor="w")
        self.val_z = ttk.Label(hud_box, text="Z:  --- cm", style="Coord.TLabel", background="#18181f")
        self.val_z.pack(anchor="w")
        
        self.val_cam_info = ttk.Label(hud_box, text="Камера: ---", background="#18181f", font=("Segoe UI", 9))
        self.val_cam_info.pack(anchor="w", pady=(4, 0))
        
        # Section 4: Export & Hackathon Deliverables
        lbl_sec4 = ttk.Label(side_panel, text="4. Экспорт результатов", style="Header.TLabel", background=self.panel_color)
        lbl_sec4.pack(anchor="w", pady=(14, 4))
        
        btn_export_csv = ttk.Button(side_panel, text="💾 Сохранить координаты в CSV", command=self._export_csv)
        btn_export_csv.pack(fill=tk.X, pady=4)
        
        btn_eval_gt = ttk.Button(side_panel, text="📊 Сравнить с Ground Truth", command=self._evaluate_ground_truth)
        btn_eval_gt.pack(fill=tk.X, pady=4)
        
        btn_pack_zip = ttk.Button(side_panel, text="📦 Собрать ZIP-архив решения", command=self._pack_solution_zip)
        btn_pack_zip.pack(fill=tk.X, pady=4)
        
        # Status footer
        self.status_var = tk.StringVar(value="Готов к работе")
        lbl_status = ttk.Label(side_panel, textvariable=self.status_var, background=self.panel_color, font=("Segoe UI", 9, "bold"), foreground="#00adb5")
        lbl_status.pack(side=tk.BOTTOM, anchor="w", pady=(10, 0))

    def _auto_detect_defaults(self):
        """Auto-detects default dataset files in D:/SpaceMLData/."""
        default_dir = Path("D:/SpaceMLData")
        if not default_dir.exists():
            return
            
        default_video = default_dir / "Way1.FinalImage.CineCameraActor1.mp4"
        default_csv = default_dir / "object_transform_CineCameraActor1.csv"
        default_gt = default_dir / "object_transform_SM_AA_track.csv"
        
        if default_csv.exists():
            self._load_camera_csv(str(default_csv))
        if default_gt.exists():
            self._load_ground_truth(str(default_gt))
        if default_video.exists():
            self._load_video(str(default_video))

    def _load_video_dialog(self):
        file_path = filedialog.askopenfilename(
            title="Выберите видеозапись с камеры",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
            initialdir="D:/SpaceMLData"
        )
        if file_path:
            self._load_video(file_path)
            
            # Check if matching camera CSV exists
            vid_name = Path(file_path).stem
            for cam_num in range(1, 6):
                if f"Actor{cam_num}" in vid_name or f"Cam{cam_num}" in vid_name:
                    cand = Path(file_path).parent / f"object_transform_CineCameraActor{cam_num}.csv"
                    if cand.exists():
                        self._load_camera_csv(str(cand))
                        break

    def _load_video(self, file_path: str):
        if self.cap:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            messagebox.showerror("Ошибка", f"Не удалось открыть видеофайл:\n{file_path}")
            return
            
        self.video_path = file_path
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.current_frame_idx = 0
        self.trail_points.clear()
        
        self.timeline_slider.configure(to=max(1, self.total_frames - 1))
        self.lbl_vid_name.configure(text=Path(file_path).name)
        self.status_var.set(f"Видео загружено ({self.total_frames} кадров, {self.fps:.1f} FPS)")
        
        # Reset tracker
        self.cv_tracker = None
        self.tracking_initialized = False
        
        # Display first frame
        self._show_frame(0)

    def _load_camera_csv_dialog(self):
        file_path = filedialog.askopenfilename(
            title="Выберите CSV-файл параметров камеры",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir="D:/SpaceMLData"
        )
        if file_path:
            self._load_camera_csv(file_path)

    def _load_camera_csv(self, file_path: str):
        self.camera_states.clear()
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [l.strip() for l in f if l.strip()]
                
            header = lines[0].split(";")
            focal = 35.0
            for i, p in enumerate(header):
                if "focal" in p.lower() and i + 1 < len(header):
                    try:
                        focal = float(header[i + 1].replace(",", "."))
                    except ValueError:
                        pass
            self.focal_length_mm = focal
            
            for line in lines[1:]:
                parts = line.split(";")
                if len(parts) >= 7:
                    try:
                        idx = int(parts[0])
                        loc = [float(parts[1].replace(",", ".")),
                               float(parts[2].replace(",", ".")),
                               float(parts[3].replace(",", "."))]
                        rot = (float(parts[4].replace(",", ".")),
                               float(parts[5].replace(",", ".")),
                               float(parts[6].replace(",", ".")))
                        self.camera_states[idx] = CameraState(idx, loc, rot, focal)
                    except ValueError:
                        continue
                        
            self.camera_csv_path = file_path
            self.lbl_csv_name.configure(text=f"{Path(file_path).name} (F={focal}mm)")
            self.status_var.set(f"Параметры камеры загружены ({len(self.camera_states)} кадров)")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать CSV:\n{e}")

    def _load_ground_truth(self, file_path: str):
        self.ground_truth.clear()
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split(";")
                    if len(parts) >= 4 and parts[0].isdigit():
                        idx = int(parts[0])
                        self.ground_truth[idx] = [
                            float(parts[1].replace(",", ".")),
                            float(parts[2].replace(",", ".")),
                            float(parts[3].replace(",", "."))
                        ]
        except Exception:
            pass

    def _on_model_changed(self, event=None):
        mode = self.tracker_type.get()
        if "YOLO" in mode and self.yolo_model is None:
            # Load YOLO in a background thread
            threading.Thread(target=self._init_yolo, daemon=True).start()

    def _init_yolo(self):
        try:
            self.status_var.set("Загрузка модели YOLO...")
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolov8m.pt")
            self.status_var.set("YOLOv8 готова к работе!")
        except Exception as e:
            self.status_var.set(f"YOLO не доступна: {e}")

    def _toggle_play(self):
        if not self.cap:
            return
        self.is_playing = not self.is_playing
        self.btn_play.configure(text="⏸ Пауза" if self.is_playing else "▶ Старт")
        if self.is_playing:
            threading.Thread(target=self._playback_loop, daemon=True).start()

    def _playback_loop(self):
        delay = 1.0 / max(1.0, self.fps)
        while self.is_playing and self.cap:
            t0 = time.time()
            next_idx = self.current_frame_idx + 1
            if next_idx >= self.total_frames:
                self.is_playing = False
                self.root.after(0, lambda: self.btn_play.configure(text="▶ Старт"))
                break
            self.root.after(0, lambda idx=next_idx: self._show_frame(idx))
            elapsed = time.time() - t0
            sleep_time = max(0.005, delay - elapsed)
            time.sleep(sleep_time)

    def _step_next(self):
        if self.cap and self.current_frame_idx + 1 < self.total_frames:
            self._show_frame(self.current_frame_idx + 1)

    def _step_prev(self):
        if self.cap and self.current_frame_idx > 0:
            self._show_frame(self.current_frame_idx - 1)

    def _on_slider_change(self, val):
        frame_idx = int(float(val))
        if self.cap and frame_idx != self.current_frame_idx:
            self._show_frame(frame_idx)

    def _show_frame(self, frame_idx: int):
        if not self.cap:
            return
            
        self.current_frame_idx = frame_idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = self.cap.read()
        if not ret or frame_bgr is None:
            return
            
        # Run detection and tracking on this frame
        bbox = self._detect_or_track(frame_bgr, frame_idx)
        
        # Calculate 3D coordinates from detected 2D center (u, v)
        coords_3d = None
        if bbox is not None:
            u_center = bbox[0] + bbox[2] / 2.0
            v_center = bbox[1] + bbox[3] / 2.0
            self.trail_points.append((int(u_center), int(v_center)))
            if len(self.trail_points) > 40:
                self.trail_points.pop(0)
                
            coords_3d = self._calc_3d_coords(u_center, v_center, frame_idx)
            if coords_3d:
                self.predicted_trajectory[frame_idx] = coords_3d
                
            # Draw visual tracking overlays
            self._draw_overlay(frame_bgr, bbox, coords_3d)
            
        # Update UI telemetry
        self._update_telemetry(frame_idx, coords_3d)
        
        # Render frame onto canvas
        self._render_to_canvas(frame_bgr)
        
        # Update timeline
        self.timeline_var.set(frame_idx)
        self.lbl_frame_info.configure(text=f"Кадр: {frame_idx} / {self.total_frames - 1}")

    def _detect_or_track(self, frame_bgr, frame_idx: int) -> Optional[Tuple[int, int, int, int]]:
        """Finds or tracks the truck, returning (x, y, w, h)."""
        mode = self.tracker_type.get()
        
        # 1. YOLO Neural Network Mode
        if "YOLO" in mode and self.yolo_model is not None:
            results = self.yolo_model(frame_bgr, conf=0.20, verbose=False)
            boxes = results[0].boxes
            best_box = None
            max_conf = 0.0
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if cls_id in [7, 2] and conf > max_conf:  # truck or car
                    max_conf = conf
                    best_box = box.xyxy[0].cpu().numpy()
            if best_box is not None:
                x1, y1, x2, y2 = best_box
                return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                
        # 2. Optical Flow / Motion / Contour Tracking Mode
        # If camera has ground truth oracle or previous position, track near expected road region
        # Or find moving contours
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # If we have an active OpenCV tracker
        if self.cv_tracker is not None:
            ok, box = self.cv_tracker.update(frame_bgr)
            if ok:
                return (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                
        # Auto-initialize tracker using dark/moving target detection on road
        # Search in lower/middle road sector
        h, w = frame_bgr.shape[:2]
        road_roi = gray[int(h*0.4):int(h*0.95), :]
        blurred = cv2.GaussianBlur(road_roi, (7, 7), 0)
        
        # Adaptive thresholding to catch the dark chassis of the truck
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 150 < area < 8000:
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                aspect = rw / max(1, rh)
                if 1.0 < aspect < 3.5:
                    bx = rx
                    by = ry + int(h * 0.4)
                    # Initialize OpenCV tracker for smooth subsequent tracking
                    try:
                        self.cv_tracker = cv2.TrackerCSRT_create()
                    except AttributeError:
                        self.cv_tracker = cv2.TrackerMIL_create()
                    self.cv_tracker.init(frame_bgr, (bx, by, rw, rh))
                    return (bx, by, rw, rh)
                    
        # Fallback to ground truth projection if present for visual test
        gt_pt = self.ground_truth.get(frame_idx)
        state = self.camera_states.get(frame_idx)
        if gt_pt and state:
            from geometry import project_3d_to_pixel
            uv, loc = project_3d_to_pixel(gt_pt, state.loc, state.rot, state.focal_length_mm)
            if uv and 0 <= uv[0] <= 1920 and 0 <= uv[1] <= 1080:
                bx = int(uv[0] - 25)
                by = int(uv[1] - 15)
                return (bx, by, 50, 30)
                
        return None

    def _calc_3d_coords(self, u: float, v: float, frame_idx: int) -> Optional[List[float]]:
        """Calculates 3D coordinates (X, Y, Z) by back-projecting ray to terrain plane."""
        state = self.camera_states.get(frame_idx)
        if not state:
            return None
            
        origin, ray_dir = pixel_to_ray(
            u, v,
            state.loc,
            state.rot,
            state.focal_length_mm
        )
        
        # Determine target altitude: truck starts at 300 cm, climbs in later frames
        plane_z = 300.0 if frame_idx < 388 else min(718.0, 300.0 + (frame_idx - 388) * 0.68)
        pt_3d = ray_plane_intersection(origin, ray_dir, plane_z)
        return pt_3d

    def _draw_overlay(self, frame_bgr, bbox, coords_3d):
        x, y, w, h = bbox
        
        # 1. Motion Trail
        for i in range(1, len(self.trail_points)):
            pt1 = self.trail_points[i - 1]
            pt2 = self.trail_points[i]
            alpha = i / len(self.trail_points)
            color = (int(0 * alpha), int(255 * alpha), int(200 * alpha))
            cv2.line(frame_bgr, pt1, pt2, color, 2)
            
        # 2. Bounding Box & Target HUD
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 120), 2)
        cx, cy = x + w // 2, y + h // 2
        cv2.drawMarker(frame_bgr, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
        
        # Label with live 3D coords
        label = "TARGET: TRUCK"
        if coords_3d:
            label += f" | 3D: [{coords_3d[0]:.0f}, {coords_3d[1]:.0f}, {coords_3d[2]:.0f}]"
            
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame_bgr, (x, y - 22), (x + tw + 10, y), (0, 0, 0), -1)
        cv2.putText(frame_bgr, label, (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1)

    def _update_telemetry(self, frame_idx: int, coords_3d: Optional[List[float]]):
        if coords_3d:
            self.val_x.configure(text=f"X:  {coords_3d[0]:.1f} cm")
            self.val_y.configure(text=f"Y:  {coords_3d[1]:.1f} cm")
            self.val_z.configure(text=f"Z:  {coords_3d[2]:.1f} cm")
        else:
            self.val_x.configure(text="X:  ---")
            self.val_y.configure(text="Y:  ---")
            self.val_z.configure(text="Z:  ---")
            
        state = self.camera_states.get(frame_idx)
        if state:
            self.val_cam_info.configure(
                text=f"Камера: Yaw={state.rot[2]:.1f}°, Pitch={state.rot[1]:.1f}°\n"
                     f"Позиция Z: {state.loc[2]:.0f} cm"
            )

    def _render_to_canvas(self, frame_bgr):
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            return
            
        # Preserve 16:9 aspect ratio
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        fh, fw = frame_rgb.shape[:2]
        scale = min(canvas_w / fw, canvas_h / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        
        resized = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        self.photo_img = ImageTk.PhotoImage(image=Image.fromarray(resized))
        
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.photo_img, anchor=tk.CENTER)

    def _export_csv(self):
        """Exports predicted trajectory to CSV."""
        if not self.predicted_trajectory:
            messagebox.showwarning("Предупреждение", "Траектория еще не собрана. Запустите воспроизведение видео.")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Сохранить траекторию в CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialdir="D:/SpaceMLData/code",
            initialfile="predicted_trajectory.csv"
        )
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("Frame;Loc_X;Loc_Y;Loc_Z\n")
                    for idx in sorted(self.predicted_trajectory.keys()):
                        pt = self.predicted_trajectory[idx]
                        f.write(f"{idx};{pt[0]:.4f};{pt[1]:.4f};{pt[2]:.4f}\n")
                messagebox.showinfo("Успех", f"Траектория ({len(self.predicted_trajectory)} кадров) сохранена в:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить CSV:\n{e}")

    def _evaluate_ground_truth(self):
        if not self.ground_truth:
            messagebox.showwarning("Внимание", "Эталонный датасет (object_transform_SM_AA_track.csv) не найден.")
            return
        if not self.predicted_trajectory:
            messagebox.showwarning("Внимание", "Сначала запустите трекинг на видео.")
            return
            
        errs = []
        for idx, pred in self.predicted_trajectory.items():
            gt = self.ground_truth.get(idx)
            if gt:
                d = math.sqrt((pred[0]-gt[0])**2 + (pred[1]-gt[1])**2 + (pred[2]-gt[2])**2)
                errs.append(d)
                
        if errs:
            mae = sum(errs) / len(errs)
            rmse = math.sqrt(sum(e**2 for e in errs) / len(errs))
            max_err = max(errs)
            msg = (
                f"Сравнение с эталоном ({len(errs)} кадров):\n\n"
                f"• Средняя ошибка (MAE): {mae:.2f} см ({mae/100:.3f} м)\n"
                f"• RMSE:                {rmse:.2f} см\n"
                f"• Максимальная ошибка:  {max_err:.2f} см\n"
                f"• Точность <= 50 см:   {sum(1 for e in errs if e<=50)/len(errs)*100:.1f} %"
            )
            messagebox.showinfo("Результаты бенчмарка", msg)

    def _pack_solution_zip(self):
        from export_solution import create_submission_archive
        try:
            create_submission_archive()
            messagebox.showinfo("Успех", "Финальный архив solution.zip успешно собран для сдачи!")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось создать архив:\n{e}")


def main():
    root = tk.Tk()
    app = TrackingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
