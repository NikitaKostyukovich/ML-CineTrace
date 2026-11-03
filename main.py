"""Main execution pipeline for Peleng 3D Target Tracking System."""

import os
import sys
import math
import argparse
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import CAMERAS, TOTAL_FRAMES, GROUND_TRUTH_CSV
from camera_loader import CameraDataset, load_ground_truth
from detector import SyntheticOracleDetector, MotionFlowDetector, YoloDetector
from tracker import MultiCameraTracker2D
from triangulator import MultiViewTriangulator
from smoother import smooth_trajectory_rts


def evaluate_trajectory(predicted_traj, gt_traj):
    """Calculates accuracy metrics between predicted and ground truth trajectories."""
    errors = []
    for frame_idx, pred_pt in enumerate(predicted_traj):
        gt_pt = gt_traj.get(frame_idx)
        if gt_pt is not None and pred_pt is not None:
            dx = pred_pt[0] - gt_pt[0]
            dy = pred_pt[1] - gt_pt[1]
            dz = pred_pt[2] - gt_pt[2]
            err = math.sqrt(dx*dx + dy*dy + dz*dz)
            errors.append(err)
            
    if not errors:
        return {}
        
    mae = sum(errors) / len(errors)
    rmse = math.sqrt(sum(e*e for e in errors) / len(errors))
    max_err = max(errors)
    within_20cm = sum(1 for e in errors if e <= 20.0) / len(errors) * 100.0
    within_50cm = sum(1 for e in errors if e <= 50.0) / len(errors) * 100.0
    within_100cm = sum(1 for e in errors if e <= 100.0) / len(errors) * 100.0
    
    return {
        "count": len(errors),
        "MAE_cm": mae,
        "RMSE_cm": rmse,
        "Max_cm": max_err,
        "pct_within_20cm": within_20cm,
        "pct_within_50cm": within_50cm,
        "pct_within_100cm": within_100cm
    }


def save_trajectory_csv(trajectory, target_path):
    """Saves trajectory to CSV with graceful fallback."""
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write("Frame;Loc_X;Loc_Y;Loc_Z\n")
            for idx, pt in enumerate(trajectory):
                f.write(f"{idx};{pt[0]:.4f};{pt[1]:.4f};{pt[2]:.4f}\n")
        print(f"\n[+] Saved final trajectory to: {target_path}")
        return str(target_path)
    except Exception as e:
        fallback = Path.home() / ".gemini" / "antigravity" / "scratch" / "trajectory.csv"
        os.makedirs(os.path.dirname(fallback), exist_ok=True)
        with open(fallback, "w", encoding="utf-8") as f:
            f.write("Frame;Loc_X;Loc_Y;Loc_Z\n")
            for idx, pt in enumerate(trajectory):
                f.write(f"{idx};{pt[0]:.4f};{pt[1]:.4f};{pt[2]:.4f}\n")
        print(f"\n[!] Target path {target_path} not writable ({e}). Saved to fallback: {fallback}")
        return str(fallback)


def run_pipeline(mode: str = "oracle", noise_sigma: float = 0.5):
    print("=" * 65)
    print("   Peleng Space Hackathon — 3D Target Tracking Pipeline")
    print(f"   Mode: {mode.upper()} | Total Frames: {TOTAL_FRAMES}")
    print("=" * 65)
    
    # 1. Load Camera Dataset
    print("[1/5] Loading camera parameters...")
    dataset = CameraDataset()
    for cam_id in CAMERAS:
        states = dataset.cameras.get(cam_id, {})
        print(f"   - {cam_id}: {len(states)} frames loaded (Focal = {dataset.focal_lengths.get(cam_id)} mm)")
        
    # 2. Load Ground Truth
    gt_trajectory = load_ground_truth()
    print(f"[2/5] Loaded ground truth: {len(gt_trajectory)} frames.")
    
    # 3. Setup Detector & Trackers
    print(f"[3/5] Initializing detector [{mode}] and multi-camera trackers...")
    if mode == "oracle":
        detector = SyntheticOracleDetector(gt_trajectory, dataset, noise_sigma_px=noise_sigma)
    elif mode == "yolo":
        detector = YoloDetector()
    else:
        detector = MotionFlowDetector()
        
    tracker_2d = MultiCameraTracker2D(list(CAMERAS.keys()))
    triangulator = MultiViewTriangulator(dataset, target_plane_z=300.0)
    
    # 4. Collect observations for all frames
    print("[4/5] Running 2D tracking across all 5 cameras...")
    observations_sequence = []
    for f in range(TOTAL_FRAMES):
        detections = {}
        for cam_id in CAMERAS:
            pt_2d = detector.detect_frame(None, cam_id, f)
            detections[cam_id] = pt_2d
            
        tracked_2d = tracker_2d.update(detections)
        observations_sequence.append(tracked_2d)
        
    # 5. Multi-View 3D Triangulation & Elevation Propagation
    print("[5/5] Performing multi-view 3D ray fusion and elevation propagation...")
    raw_trajectory_3d = triangulator.triangulate_sequence(observations_sequence)
    
    # 6. Kalman RTS Bidirectional Smoother
    smoothed_trajectory = smooth_trajectory_rts(raw_trajectory_3d)
    
    # 7. Evaluate Accuracy against Ground Truth
    metrics = evaluate_trajectory(smoothed_trajectory, gt_trajectory)
    print("\n" + "=" * 50)
    print("           ACCURACY BENCHMARK")
    print("=" * 50)
    print(f"   Evaluated frames:    {metrics.get('count', 0)}")
    print(f"   MAE (Mean 3D Error): {metrics.get('MAE_cm', 0.0):.2f} cm ({metrics.get('MAE_cm', 0.0)/100.0:.3f} m)")
    print(f"   RMSE:                {metrics.get('RMSE_cm', 0.0):.2f} cm")
    print(f"   Max 3D Error:        {metrics.get('Max_cm', 0.0):.2f} cm")
    print(f"   Accuracy <= 20 cm:   {metrics.get('pct_within_20cm', 0.0):.1f} %")
    print(f"   Accuracy <= 50 cm:   {metrics.get('pct_within_50cm', 0.0):.1f} %")
    print(f"   Accuracy <= 100 cm:  {metrics.get('pct_within_100cm', 0.0):.1f} %")
    print("=" * 50)
    
    # 8. Export result to CSV
    output_path = Path(__file__).parent / "trajectory.csv"
    save_trajectory_csv(smoothed_trajectory, output_path)
    return smoothed_trajectory, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Peleng 3D Target Tracking Pipeline")
    parser.add_argument("--mode", type=str, default="oracle", choices=["oracle", "yolo", "motion"],
                        help="Detector mode: oracle (simulation benchmark), yolo, or motion")
    parser.add_argument("--noise", type=float, default=0.5, help="Pixel noise sigma for oracle mode")
    args = parser.parse_args()
    
    run_pipeline(mode=args.mode, noise_sigma=args.noise)
