"""Configuration parameters for Peleng Space Hackathon 3D Multi-Camera Tracker."""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path("D:/SpaceMLData")
CODE_DIR = BASE_DIR / "code"
# Output paths
OUTPUT_DIR = CODE_DIR
OUTPUT_CSV = CODE_DIR / "trajectory_output.csv"


# Ground truth trajectory file
GROUND_TRUTH_CSV = BASE_DIR / "object_transform_SM_AA_track.csv"

# Camera metadata files and video sources
CAMERAS = {
    "Cam1": {
        "csv": BASE_DIR / "object_transform_CineCameraActor1.csv",
        "video": BASE_DIR / "Way1.FinalImage.CineCameraActor1.mp4",
        "default_focal": 35.0,
        "is_static": True
    },
    "Cam2": {
        "csv": BASE_DIR / "object_transform_CineCameraActor2.csv",
        "video": BASE_DIR / "Way1.FinalImage.CineCameraActor2.mp4",
        "default_focal": 60.0,
        "is_static": True
    },
    "Cam3": {
        "csv": BASE_DIR / "object_transform_CineCameraActor3.csv",
        "video": BASE_DIR / "Way1.FinalImage.CineCameraActor3.mp4",
        "default_focal": 35.0,
        "is_static": True
    },
    "Cam4": {
        "csv": BASE_DIR / "object_transform_CineCameraActor4.csv",
        "video": BASE_DIR / "Way1.FinalImage.CineCameraActor4.mp4",
        "default_focal": 50.0,
        "is_static": True
    },
    "Cam5": {
        "csv": BASE_DIR / "object_transform_CineCameraActor5.csv",
        "video": BASE_DIR / "Way1.FinalImage.CineCameraActor5.mp4",
        "default_focal": 20.0,
        "is_static": False  # PTZ camera: pan/yaw changes dynamically
    }
}

# Sensor physical characteristics (Unreal Engine CineCamera Super 35)
SENSOR_WIDTH_MM = 23.76
SENSOR_HEIGHT_MM = 13.365
IMAGE_WIDTH_PX = 1920
IMAGE_HEIGHT_PX = 1080

# Total frames in dataset
TOTAL_FRAMES = 1002

# Target object ground plane constraint
TARGET_PLANE_Z = 300.0

# Output CSV path
OUTPUT_CSV = OUTPUT_DIR / "trajectory.csv"
