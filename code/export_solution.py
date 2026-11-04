"""Archives the solution code and resulting trajectory.csv for hackathon submission."""

import os
import zipfile
from pathlib import Path


def create_submission_archive():
    code_dir = Path(__file__).parent
    output_zip = code_dir / "solution.zip"
    
    files_to_pack = [
        "config.py",
        "geometry.py",
        "camera_loader.py",
        "detector.py",
        "tracker.py",
        "triangulator.py",
        "multi_camera_fusion.py",
        "occlusion_kalman.py",
        "cad_model_matcher.py",
        "pose_estimator_6d.py",
        "smoother.py",
        "main.py",
        "run_model.py",
        "requirements.txt",
        "README.md",
    ]
    
    traj_candidates = [
        code_dir / "trajectory.csv",
        code_dir / "trajectory_output.csv",
        Path.home() / ".gemini" / "antigravity" / "scratch" / "trajectory.csv"
    ]
    
    trajectory_file = None
    for cand in traj_candidates:
        if cand.exists():
            trajectory_file = cand
            break

    try:
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as z:
            for fname in files_to_pack:
                fpath = code_dir / fname
                if fpath.exists():
                    z.write(fpath, arcname=f"code/{fname}")
                    print(f"  + Added code/{fname}")
            if trajectory_file:
                z.write(trajectory_file, arcname="trajectory.csv")
                print(f"  + Added trajectory.csv (from {trajectory_file})")
        print(f"\n[+] Successfully created submission archive: {output_zip} ({output_zip.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        fallback = Path.home() / ".gemini" / "antigravity" / "scratch" / "solution.zip"
        with zipfile.ZipFile(fallback, "w", zipfile.ZIP_DEFLATED) as z:
            for fname in files_to_pack:
                fpath = code_dir / fname
                if fpath.exists():
                    z.write(fpath, arcname=f"code/{fname}")
                    print(f"  + Added code/{fname}")
            if trajectory_file:
                z.write(trajectory_file, arcname="trajectory.csv")
                print(f"  + Added trajectory.csv")
        print(f"\n[!] Target {output_zip} not writable inside sandbox ({e}). Created at fallback: {fallback} ({fallback.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    create_submission_archive()
