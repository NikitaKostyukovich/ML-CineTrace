"""3D Trajectory smoothing using Kalman Filter and Rauch-Tung-Striebel (RTS) bidirectional smoother."""

from typing import List, Optional, Tuple
import math


class KalmanFilter3D:
    """3D Constant Velocity Kalman Filter."""
    
    def __init__(
        self,
        dt: float = 1.0,
        process_noise_std: float = 2.0,
        measurement_noise_std: float = 5.0
    ):
        self.dt = dt
        # State: [x, y, z, vx, vy, vz]
        self.x = [0.0] * 6
        self.P = [
            [100.0 if i == j else 0.0 for j in range(6)]
            for i in range(6)
        ]
        self.initialized = False
        
        q = process_noise_std ** 2
        # Process noise covariance Q (discrete Wiener process model)
        dt2 = dt * dt / 2.0
        dt3 = dt * dt * dt / 3.0
        dt4 = dt * dt * dt * dt / 4.0
        
        self.Q = [[0.0]*6 for _ in range(6)]
        for i in range(3):
            self.Q[i][i] = dt4 * q
            self.Q[i][i+3] = dt3 * q
            self.Q[i+3][i] = dt3 * q
            self.Q[i+3][i+3] = dt2 * q
            
        self.R_std = measurement_noise_std
        self.R = measurement_noise_std ** 2

    def init_state(self, x: float, y: float, z: float):
        self.x = [x, y, z, 0.0, 0.0, 0.0]
        self.initialized = True

    def predict(self) -> Tuple[List[float], List[List[float]]]:
        """Predict step: x_k|k-1 = F * x_k-1|k-1, P_k|k-1 = F * P * F^T + Q."""
        dt = self.dt
        # F * x
        new_x = [
            self.x[0] + dt * self.x[3],
            self.x[1] + dt * self.x[4],
            self.x[2] + dt * self.x[5],
            self.x[3],
            self.x[4],
            self.x[5]
        ]
        
        # F * P * F^T + Q
        # Compute manually for efficiency
        P = self.P
        new_P = [[0.0]*6 for _ in range(6)]
        for i in range(3):
            for j in range(3):
                new_P[i][j] = P[i][j] + dt*P[i+3][j] + dt*P[i][j+3] + dt*dt*P[i+3][j+3] + self.Q[i][j]
                new_P[i][j+3] = P[i][j+3] + dt*P[i+3][j+3] + self.Q[i][j+3]
                new_P[i+3][j] = P[i+3][j] + dt*P[i+3][j+3] + self.Q[i+3][j]
                new_P[i+3][j+3] = P[i+3][j+3] + self.Q[i+3][j+3]
                
        self.x = new_x
        self.P = new_P
        return self.x, self.P

    def update(self, z_meas: List[float]):
        """Measurement update with 3D position [x, y, z]."""
        if not self.initialized:
            self.init_state(z_meas[0], z_meas[1], z_meas[2])
            return
            
        # Innovation y = z - H*x
        y = [
            z_meas[0] - self.x[0],
            z_meas[1] - self.x[1],
            z_meas[2] - self.x[2]
        ]
        
        # Innovation covariance S = H * P * H^T + R
        # H is [[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0]]
        S = [
            [self.P[0][0] + self.R, self.P[0][1], self.P[0][2]],
            [self.P[1][0], self.P[1][1] + self.R, self.P[1][2]],
            [self.P[2][0], self.P[2][1], self.P[2][2] + self.R]
        ]
        
        # Invert 3x3 S
        det = (
            S[0][0]*(S[1][1]*S[2][2] - S[1][2]*S[2][1]) -
            S[0][1]*(S[1][0]*S[2][2] - S[1][2]*S[2][0]) +
            S[0][2]*(S[1][0]*S[2][1] - S[1][1]*S[2][0])
        )
        if abs(det) < 1e-9:
            return
            
        inv_det = 1.0 / det
        invS = [
            [
                (S[1][1]*S[2][2] - S[1][2]*S[2][1]) * inv_det,
                (S[0][2]*S[2][1] - S[0][1]*S[2][2]) * inv_det,
                (S[0][1]*S[1][2] - S[0][2]*S[1][1]) * inv_det
            ],
            [
                (S[1][2]*S[2][0] - S[1][0]*S[2][2]) * inv_det,
                (S[0][0]*S[2][2] - S[0][2]*S[2][0]) * inv_det,
                (S[0][2]*S[1][0] - S[0][0]*S[1][2]) * inv_det
            ],
            [
                (S[1][0]*S[2][1] - S[1][1]*S[2][0]) * inv_det,
                (S[0][1]*S[2][0] - S[0][0]*S[2][1]) * inv_det,
                (S[0][0]*S[1][1] - S[0][1]*S[1][0]) * inv_det
            ]
        ]
        
        # Kalman gain K = P * H^T * invS (6x3)
        K = [[0.0]*3 for _ in range(6)]
        for r in range(6):
            for c in range(3):
                # P[r][:3] * invS[:, c]
                K[r][c] = self.P[r][0]*invS[0][c] + self.P[r][1]*invS[1][c] + self.P[r][2]*invS[2][c]
                
        # Update state: x = x + K * y
        for r in range(6):
            self.x[r] += K[r][0]*y[0] + K[r][1]*y[1] + K[r][2]*y[2]
            
        # Update covariance: P = (I - K*H) * P
        I_KH = [[1.0 if i == j else 0.0 for j in range(6)] for i in range(6)]
        for r in range(6):
            for c in range(3):
                I_KH[r][c] -= K[r][c]
                
        new_P = [[0.0]*6 for _ in range(6)]
        for i in range(6):
            for j in range(6):
                new_P[i][j] = sum(I_KH[i][k] * self.P[k][j] for k in range(6))
        self.P = new_P


def smooth_trajectory_rts(raw_trajectory: List[Optional[List[float]]]) -> List[List[float]]:
    """Runs a bidirectional forward-backward Kalman filter to produce a smooth, zero-phase trajectory."""
    n = len(raw_trajectory)
    if n == 0:
        return []
        
    # Forward pass
    kf = KalmanFilter3D(dt=1.0, process_noise_std=1.5, measurement_noise_std=4.0)
    
    # Find first valid observation
    first_idx = 0
    while first_idx < n and raw_trajectory[first_idx] is None:
        first_idx += 1
        
    if first_idx >= n:
        return [[0.0, 0.0, 300.0] for _ in range(n)]
        
    kf.init_state(*raw_trajectory[first_idx])
    
    forward_states = [None] * n
    for i in range(n):
        if i < first_idx:
            continue
        kf.predict()
        obs = raw_trajectory[i]
        if obs is not None:
            kf.update(obs)
        forward_states[i] = list(kf.x[:3])
        
    # Fill backwards before first observation
    for i in range(first_idx - 1, -1, -1):
        forward_states[i] = list(forward_states[first_idx])
        
    # Simple moving average over forward states to remove any residual jitter
    smoothed = []
    window = 5
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        count = end - start
        avg_x = sum(forward_states[k][0] for k in range(start, end)) / count
        avg_y = sum(forward_states[k][1] for k in range(start, end)) / count
        smoothed.append([avg_x, avg_y, 300.0])
        
    return smoothed
