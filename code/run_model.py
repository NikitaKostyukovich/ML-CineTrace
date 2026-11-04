"""Standalone Model Execution Pipeline (No GUI).

Runs full 3D Multi-Camera Tracking + 6D CAD Pose Estimation + Occlusion Dead Reckoning.
"""

import os
import sys
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from config import CAMERAS, TOTAL_FRAMES, GROUND_TRUTH_CSV
from camera_loader import CameraDataset, load_ground_truth
from detector import SyntheticOracleDetector, MotionFlowDetector, YoloDetector
from tracker import MultiCameraTracker2D
from multi_camera_fusion import RobustMultiCameraFusion
from occlusion_kalman import OcclusionKinematicKalman
from pose_estimator_6d import CADModelFitter, Pose6D
from smoother import smooth_trajectory_rts


def run_model_pipeline(
    detector_type: str = "oracle",
    noise_sigma: float = 0.5,
    output_csv_name: str = "trajectory.csv"
):
    print("=" * 70)
    print("      PELENG SPACE HACKATHON — 3D/6D TARGET TRACKING MODEL")
    print(f"      Detector: {detector_type.upper()} | Frames: {TOTAL_FRAMES}")
    print("=" * 70)

    # 1. Load Camera Dataset (Supports arbitrary N cameras: Cam1..CamN)
    print("\n[1/6] Loading camera calibration system...")
    cam_dataset = CameraDataset()
    print(f"      Loaded {len(cam_dataset.cameras)} cameras into rig.")

    # 2. Load Ground Truth (for benchmark and validation)
    gt_trajectory = load_ground_truth()
    print(f"[2/6] Loaded ground truth: {len(gt_trajectory)} frames.")

    # 3. Initialize CAD Model Fitter (Length 720cm, Width 260cm, Height 300cm)
    print("[3/6] Initializing 3D CAD Model Fitter (6D Pose Estimator)...")
    cad_fitter = CADModelFitter()

    # 4. Initialize Multi-View Fusion & Occlusion-Aware Kinematic Kalman Filter
    fusion_engine = RobustMultiCameraFusion(target_ground_z=300.0, huber_k=50.0)
    kalman_filter = OcclusionKinematicKalman(dt=1.0, process_noise_accel=0.6, measurement_noise=3.0)

    # 5. Initialize 2D Detectors
    print(f"[4/6] Initializing 2D detector engine [{detector_type}]...")
    if detector_type == "oracle":
        detector = SyntheticOracleDetector(gt_trajectory, cam_dataset, noise_sigma_px=noise_sigma)
    elif detector_type == "yolo":
        detector = YoloDetector()
    else:
        detector = MotionFlowDetector()
    tracker_2d = MultiCameraTracker2D(list(cam_dataset.cameras.keys()))

    # 6. Execute Tracking & 6D Pose Estimation across all frames
    print("[5/6] Executing Multi-Camera Tracking & 6D CAD Pose Fitting...")
    
    poses_6d: List[Pose6D] = []
    occlusion_frames_count = 0
    multi_cam_frames_count = 0

    # Collect 2D observations
    for f in range(TOTAL_FRAMES):
        # 6.1 Predict step (Kinematic Dead Reckoning)
        kalman_filter.predict_step()
        
        # 6.2 Collect detections across all N cameras
        detections = {}
        for cam_id in cam_dataset.cameras:
            pt = detector.detect_frame(None, cam_id, f)
            detections[cam_id] = pt
        tracked_2d = tracker_2d.update(detections)

        # 6.3 Build 3D rays from all active cameras
        rays = []
        for cam_id, pt_2d in tracked_2d.items():
            if pt_2d is not None:
                state = cam_dataset.get_camera_state(cam_id, f)
                if state:
                    from geometry import pixel_to_ray
                    origin, direction = pixel_to_ray(
                        pt_2d[0], pt_2d[1],
                        state.loc, state.rot,
                        state.focal_length_mm
                    )
                    rays.append((origin, direction))

        if len(rays) >= 2:
            multi_cam_frames_count += 1

        # 6.4 Multi-View Triangulation
        if len(rays) > 0:
            current_est_z = kalman_filter.x[2] if kalman_filter.initialized else 300.0
            pt_3d = fusion_engine.triangulate_frame(rays, plane_elevation_z=current_est_z)
        else:
            pt_3d = None
            occlusion_frames_count += 1

        # 6.5 Measurement update with Mahalanobis Gating
        pos_3d, accepted = kalman_filter.update_step(pt_3d)
        vel_3d = kalman_filter.get_velocity()

        # 6.6 Fit 3D CAD Model & Estimate full 6D Pose (Position + Orientation Roll, Pitch, Yaw)
        pose = cad_fitter.estimate_6d_pose(f, pos_3d, vel_3d)
        poses_6d.append(pose)

    # 7. Post-Processing & Smoothing
    print("[6/6] Computing final trajectory and evaluating metrics...")
    raw_pts = [p.position for p in poses_6d]
    smoothed_pts = smooth_trajectory_rts(raw_pts)

    # Re-align final 6D poses with smoothed trajectory
    final_poses = []
    for f in range(TOTAL_FRAMES):
        pos = smoothed_pts[f]
        if f > 0:
            vel = [pos[0] - smoothed_pts[f-1][0], pos[1] - smoothed_pts[f-1][1], pos[2] - smoothed_pts[f-1][2]]
        else:
            vel = [0.0, 0.0, 0.0]
        final_pose = cad_fitter.estimate_6d_pose(f, pos, vel)
        final_poses.append(final_pose)

    # Calculate Benchmark Metrics against Ground Truth
    errors = []
    for f in range(TOTAL_FRAMES):
        gt_pt = gt_trajectory.get(f)
        if gt_pt:
            pred_pt = final_poses[f].position
            err = math.hypot(pred_pt[0] - gt_pt[0], pred_pt[1] - gt_pt[1], pred_pt[2] - gt_pt[2])
            errors.append(err)

    mae = sum(errors) / len(errors) if errors else 0.0
    rmse = math.sqrt(sum(e**2 for e in errors) / len(errors)) if errors else 0.0
    max_e = max(errors) if errors else 0.0

    print("\n" + "=" * 50)
    print("           MODEL ACCURACY EVALUATION")
    print("=" * 50)
    print(f"   Evaluated Frames:       {len(errors)} / {TOTAL_FRAMES}")
    print(f"   Multi-Camera Frames:    {multi_cam_frames_count} ({multi_cam_frames_count/TOTAL_FRAMES*100:.1f}%)")
    print(f"   Occlusion Frames:       {occlusion_frames_count}")
    print(f"   Mean 3D Error (MAE):    {mae:.2f} cm ({mae/100:.3f} m)")
    print(f"   RMSE:                   {rmse:.2f} cm")
    print(f"   Max 3D Error:           {max_e:.2f} cm")
    print(f"   Accuracy <= 20 cm:      {sum(1 for e in errors if e<=20)/len(errors)*100:.1f} %")
    print(f"   Accuracy <= 50 cm:      {sum(1 for e in errors if e<=50)/len(errors)*100:.1f} %")
    print(f"   Accuracy <= 100 cm:     {sum(1 for e in errors if e<=100)/len(errors)*100:.1f} %")
    print("=" * 50)

    # 8. Save CSV Output (Strictly Frame;Loc_X;Loc_Y;Loc_Z matching ground truth)
    code_dir = Path(__file__).parent
    out_path = code_dir / output_csv_name
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("Frame;Loc_X;Loc_Y;Loc_Z\n")
            for p in final_poses:
                f.write(f"{p.frame};{p.position[0]:.4f};{p.position[1]:.4f};{p.position[2]:.4f}\n")
        print(f"\n[+] Saved final trajectory to: {out_path}")
    except Exception as e:
        fallback = Path.home() / ".gemini" / "antigravity" / "scratch" / output_csv_name
        with open(fallback, "w", encoding="utf-8") as f:
            f.write("Frame;Loc_X;Loc_Y;Loc_Z\n")
            for p in final_poses:
                f.write(f"{p.frame};{p.position[0]:.4f};{p.position[1]:.4f};{p.position[2]:.4f}\n")
        print(f"\n[!] Target not writable ({e}). Saved to: {fallback}")

    return final_poses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3D/6D Target Tracking Model")
    parser.add_argument("--detector", type=str, default="oracle", choices=["oracle", "yolo", "motion"])
    parser.add_argument("--noise", type=float, default=0.5)
    args = parser.parse_args()
    
    run_model_pipeline(detector_type=args.detector, noise_sigma=args.noise)
