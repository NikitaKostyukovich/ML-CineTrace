"""Advanced Kinematic Kalman Filter with Occlusion / Smoke Dead-Reckoning and Mahalanobis Gating."""

import math
from typing import List, Optional, Tuple, Dict
import numpy as np


class OcclusionKinematicKalman:
    """9-state Extended Kalman Filter for 3D Target Tracking:
    
    State vector: [X, Y, Z, Vx, Vy, Vz, Ax, Ay, Az]
    Features:
    - Dead reckoning prediction during total camera occlusion (smoke, terrain, tunnels).
    - Covariance expansion tracking increasing spatial uncertainty.
    - Mahalanobis distance gating to prevent false detections upon target re-emergence.
    """
    
    def __init__(
        self,
        dt: float = 1.0,
        process_noise_accel: float = 0.5,
        measurement_noise: float = 3.0,
        gate_chi2: float = 9.21  # 99% confidence threshold for 2-DOF / 3-DOF
    ):
        self.dt = dt
        self.state_dim = 9
        
        # State vector: [x, y, z, vx, vy, vz, ax, ay, az]
        self.x = np.zeros(9, dtype=np.float64)
        
        # State covariance P
        self.P = np.eye(9, dtype=np.float64) * 50.0
        
        # State transition matrix F (constant acceleration kinematic model)
        self.F = np.eye(9, dtype=np.float64)
        # Position += dt*v + 0.5*dt^2*a
        for i in range(3):
            self.F[i, i + 3] = dt
            self.F[i, i + 6] = 0.5 * dt * dt
            self.F[i + 3, i + 6] = dt
            
        # Process noise covariance Q (continuous white noise acceleration model)
        q = process_noise_accel ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        dt5 = dt4 * dt
        
        self.Q = np.zeros((9, 9), dtype=np.float64)
        for i in range(3):
            self.Q[i, i]         = (dt5 / 20.0) * q
            self.Q[i, i + 3]     = (dt4 / 8.0) * q
            self.Q[i, i + 6]     = (dt3 / 6.0) * q
            self.Q[i + 3, i]     = (dt4 / 8.0) * q
            self.Q[i + 3, i + 3] = (dt3 / 3.0) * q
            self.Q[i + 3, i + 6] = (dt2 / 2.0) * q
            self.Q[i + 6, i]     = (dt3 / 6.0) * q
            self.Q[i + 6, i + 3] = (dt2 / 2.0) * q
            self.Q[i + 6, i + 6] = dt * q
            
        # Measurement matrix H (observes [x, y, z])
        self.H = np.zeros((3, 9), dtype=np.float64)
        for i in range(3):
            self.H[i, i] = 1.0
            
        self.R = np.eye(3, dtype=np.float64) * (measurement_noise ** 2)
        self.gate_chi2 = gate_chi2
        
        self.initialized = False
        self.consecutive_occlusions = 0

    def initialize(self, initial_xyz: List[float], initial_v: Optional[List[float]] = None):
        """Initializes filter state."""
        self.x[:3] = initial_xyz
        if initial_v is not None:
            self.x[3:6] = initial_v
        else:
            self.x[3:6] = [0.0, 0.0, 0.0]
        self.x[6:9] = [0.0, 0.0, 0.0]
        self.P = np.eye(9, dtype=np.float64) * 20.0
        self.initialized = True
        self.consecutive_occlusions = 0

    def predict_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts state forward in time (Dead Reckoning).
        
        During occlusion, state propagates purely via kinematic laws,
        and covariance P naturally grows to reflect position uncertainty.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update_step(self, measurement: Optional[List[float]]) -> Tuple[List[float], bool]:
        """Updates filter with incoming 3D measurement, applying Mahalanobis gating.
        
        Returns:
            (estimated_xyz, was_accepted)
        """
        if measurement is None:
            # Full occlusion / smoke: pure dead reckoning
            self.consecutive_occlusions += 1
            return self.x[:3].tolist(), False
            
        z = np.array(measurement, dtype=np.float64)
        
        if not self.initialized:
            self.initialize(measurement)
            return self.x[:3].tolist(), True
            
        # Innovation y = z - H*x
        y = z - (self.H @ self.x)
        
        # Innovation covariance S = H*P*H^T + R
        S = self.H @ self.P @ self.H.T + self.R
        inv_S = np.linalg.inv(S)
        
        # Mahalanobis distance gating: d_M^2 = y^T * inv(S) * y
        mahalanobis_sq = float(y.T @ inv_S @ y)
        
        if mahalanobis_sq > self.gate_chi2 and self.consecutive_occlusions < 30:
            # Outlier or spurious detection (e.g. smoke reflection) -> reject
            self.consecutive_occlusions += 1
            return self.x[:3].tolist(), False
            
        # Kalman gain K = P * H^T * inv_S
        K = self.P @ self.H.T @ inv_S
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update: P = (I - K*H) * P
        I_KH = np.eye(9, dtype=np.float64) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        self.consecutive_occlusions = 0
        return self.x[:3].tolist(), True

    def get_position(self) -> List[float]:
        return self.x[:3].tolist()

    def get_velocity(self) -> List[float]:
        return self.x[3:6].tolist()

    def get_uncertainty_radius(self) -> float:
        """Returns 1-sigma uncertainty radius (cm) in XY plane."""
        pos_cov_xy = self.P[:2, :2]
        eigvals = np.linalg.eigvalsh(pos_cov_xy)
        return float(math.sqrt(max(0.1, np.max(eigvals))))
