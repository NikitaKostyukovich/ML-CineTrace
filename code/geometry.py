"""Geometry, projection, ray casting, and triangulation for Unreal Engine cameras."""

import math
from typing import List, Tuple, Optional


def euler_to_rot_matrix_ue(roll_deg: float, pitch_deg: float, yaw_deg: float) -> List[List[float]]:
    """Convert Unreal Engine Euler angles (Roll, Pitch, Yaw in degrees) into a 3x3 rotation matrix.
    
    Unreal Engine coordinate convention:
        X = Forward, Y = Right, Z = Up.
        R = R_z(Yaw) * R_y(Pitch) * R_x(Roll)
    """
    r_pitch = math.radians(-pitch_deg)
    r_yaw   = math.radians(yaw_deg)
    r_roll  = math.radians(roll_deg)
    
    cp, sp = math.cos(r_pitch), math.sin(r_pitch)
    cy, sy = math.cos(r_yaw),   math.sin(r_yaw)
    cr, sr = math.cos(r_roll),  math.sin(r_roll)
    
    # R_y * R_x:
    m00 = cp
    m01 = sp * sr
    m02 = sp * cr
    
    m10 = 0.0
    m11 = cr
    m12 = -sr
    
    m20 = -sp
    m21 = cp * sr
    m22 = cp * cr
    
    # R_z * (R_y * R_x):
    r00 = cy * m00 - sy * m10
    r01 = cy * m01 - sy * m11
    r02 = cy * m02 - sy * m12
    
    r10 = sy * m00 + cy * m10
    r11 = sy * m01 + cy * m11
    r12 = sy * m02 + cy * m12
    
    r20 = m20
    r21 = m21
    r22 = m22
    
    return [
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
    ]


def project_3d_to_pixel(
    point_world: List[float],
    cam_loc: List[float],
    cam_rot: Tuple[float, float, float],
    focal_length_mm: float,
    sensor_w: float = 23.76,
    sensor_h: float = 13.365,
    img_w: int = 1920,
    img_h: int = 1080
) -> Tuple[Optional[Tuple[float, float]], List[float]]:
    """Project a 3D world coordinate (UE cm) onto 2D image pixel coordinates (u, v).
    
    Returns:
        ((u, v), (x_local, y_local, z_local)) or (None, (x_local, y_local, z_local)) if behind camera.
    """
    R = euler_to_rot_matrix_ue(cam_rot[0], cam_rot[1], cam_rot[2])
    dx = point_world[0] - cam_loc[0]
    dy = point_world[1] - cam_loc[1]
    dz = point_world[2] - cam_loc[2]
    
    # In camera local coordinates: Forward (X), Right (Y), Up (Z)
    x_local = R[0][0]*dx + R[1][0]*dy + R[2][0]*dz
    y_local = R[0][1]*dx + R[1][1]*dy + R[2][1]*dz
    z_local = R[0][2]*dx + R[1][2]*dy + R[2][2]*dz
    
    if x_local <= 0.0:
        return None, [x_local, y_local, z_local]
        
    fx = (focal_length_mm / sensor_w) * img_w
    fy = (focal_length_mm / sensor_h) * img_h
    cx = img_w / 2.0
    cy = img_h / 2.0
    
    u = cx + fx * (y_local / x_local)
    v = cy - fy * (z_local / x_local)
    return (u, v), [x_local, y_local, z_local]


def pixel_to_ray(
    u: float,
    v: float,
    cam_loc: List[float],
    cam_rot: Tuple[float, float, float],
    focal_length_mm: float,
    sensor_w: float = 23.76,
    sensor_h: float = 13.365,
    img_w: int = 1920,
    img_h: int = 1080
) -> Tuple[List[float], List[float]]:
    """Back-project a 2D pixel (u, v) into a normalized 3D ray in world coordinates.
    
    Returns:
        (cam_loc, ray_dir) where ray_dir is a 3-element unit vector [dx, dy, dz].
    """
    fx = (focal_length_mm / sensor_w) * img_w
    fy = (focal_length_mm / sensor_h) * img_h
    cx = img_w / 2.0
    cy = img_h / 2.0
    
    # In camera local frame:
    # Forward = 1.0, Right = (u - cx)/fx, Up = -(v - cy)/fy
    v_local = [1.0, (u - cx) / fx, -(v - cy) / fy]
    
    # Rotate to world coordinates: d_world = R * v_local
    R = euler_to_rot_matrix_ue(cam_rot[0], cam_rot[1], cam_rot[2])
    dx = R[0][0]*v_local[0] + R[0][1]*v_local[1] + R[0][2]*v_local[2]
    dy = R[1][0]*v_local[0] + R[1][1]*v_local[1] + R[1][2]*v_local[2]
    dz = R[2][0]*v_local[0] + R[2][1]*v_local[1] + R[2][2]*v_local[2]
    
    norm = math.sqrt(dx*dx + dy*dy + dz*dz)
    ray_dir = [dx / norm, dy / norm, dz / norm]
    return cam_loc, ray_dir


def ray_plane_intersection(
    cam_loc: List[float],
    ray_dir: List[float],
    plane_z: float = 300.0
) -> Optional[List[float]]:
    """Intersect a 3D ray with horizontal plane Z = plane_z."""
    if abs(ray_dir[2]) < 1e-7:
        return None
    t = (plane_z - cam_loc[2]) / ray_dir[2]
    if t < 0.0:
        return None  # Ray points away from plane
    return [
        cam_loc[0] + t * ray_dir[0],
        cam_loc[1] + t * ray_dir[1],
        plane_z
    ]


def triangulate_rays_midpoint(rays: List[Tuple[List[float], List[float]]]) -> Optional[List[float]]:
    """Triangulate a 3D point by finding the closest point to multiple 3D rays (least squares).
    
    Each ray is represented as (origin, unit_direction).
    For N rays, solves:
        sum_i (I - d_i d_i^T) * X = sum_i (I - d_i d_i^T) * p_i
    """
    if len(rays) == 0:
        return None
    if len(rays) == 1:
        # Fallback to horizontal plane intersection if only 1 ray
        return ray_plane_intersection(rays[0][0], rays[0][1], 300.0)
        
    # Accumulate 3x3 matrix M and 3x1 vector B
    M = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    B = [0.0, 0.0, 0.0]
    
    for origin, dir_vec in rays:
        dx, dy, dz = dir_vec
        # I - d * d^T
        proj = [
            [1.0 - dx*dx, -dx*dy,       -dx*dz],
            [-dy*dx,      1.0 - dy*dy,  -dy*dz],
            [-dz*dx,      -dz*dy,       1.0 - dz*dz]
        ]
        for r in range(3):
            for c in range(3):
                M[r][c] += proj[r][c]
            B[r] += proj[r][0]*origin[0] + proj[r][1]*origin[1] + proj[r][2]*origin[2]
            
    # Solve M * X = B via 3x3 matrix inversion (Cramer's rule)
    det = (
        M[0][0] * (M[1][1]*M[2][2] - M[1][2]*M[2][1]) -
        M[0][1] * (M[1][0]*M[2][2] - M[1][2]*M[2][0]) +
        M[0][2] * (M[1][0]*M[2][1] - M[1][1]*M[2][0])
    )
    if abs(det) < 1e-9:
        # Ill-conditioned, fallback to first ray plane intersection
        return ray_plane_intersection(rays[0][0], rays[0][1], 300.0)
        
    inv_det = 1.0 / det
    invM = [
        [
            (M[1][1]*M[2][2] - M[1][2]*M[2][1]) * inv_det,
            (M[0][2]*M[2][1] - M[0][1]*M[2][2]) * inv_det,
            (M[0][1]*M[1][2] - M[0][2]*M[1][1]) * inv_det
        ],
        [
            (M[1][2]*M[2][0] - M[1][0]*M[2][2]) * inv_det,
            (M[0][0]*M[2][2] - M[0][2]*M[2][0]) * inv_det,
            (M[0][2]*M[1][0] - M[0][0]*M[1][2]) * inv_det
        ],
        [
            (M[1][0]*M[2][1] - M[1][1]*M[2][0]) * inv_det,
            (M[0][1]*M[2][0] - M[0][0]*M[2][1]) * inv_det,
            (M[0][0]*M[1][1] - M[0][1]*M[1][0]) * inv_det
        ]
    ]
    
    x = invM[0][0]*B[0] + invM[0][1]*B[1] + invM[0][2]*B[2]
    y = invM[1][0]*B[0] + invM[1][1]*B[1] + invM[1][2]*B[2]
    z = invM[2][0]*B[0] + invM[2][1]*B[1] + invM[2][2]*B[2]
    
    return [x, y, z]
