"""Constant-Acceleration Kalman tracking for enemy drones.

Honest framing: PyBullet is ground truth (no measurement noise at 240Hz),
so the filter's job is PREDICTION — extrapolating enemy position 0.5-2s
ahead for intercept geometry — and it becomes a true filter the moment
real (noisy) sensor data replaces sim telemetry.

Tracks are keyed by stable spawn-time UIDs (env.enemy_uids), never by
array index, so kills/respawns cannot corrupt track state.
"""
import numpy as np


class _Track:
    """9-state CA Kalman filter: [px py pz vx vy vz ax ay az]."""

    def __init__(self, pos, dt):
        self.dt = float(dt)
        self.x = np.concatenate([np.asarray(pos, dtype=float), np.zeros(6)])
        self.P = np.diag([1.0] * 3 + [100.0] * 3 + [25.0] * 3)

    def _F(self, dt):
        F = np.eye(9)
        for i in range(3):
            F[i, i + 3] = dt
            F[i, i + 6] = 0.5 * dt * dt
            F[i + 3, i + 6] = dt
        return F

    def predict(self, tau=None):
        """Extrapolate position tau seconds ahead from current state."""
        dt = self.dt if tau is None else float(tau)
        return (self._F(dt) @ self.x)[:3]

    def step(self, pos_meas, q_acc=2.0, r=0.01):
        """Filter update with a position measurement; returns filtered pos."""
        dt, F = self.dt, self._F(self.dt)
        self.x = F @ self.x
        # Piecewise white-noise-accel process model (per-axis blocks)
        Q = np.zeros((9, 9))
        g = np.array([dt**3 / 6, dt**2 / 2, dt])
        for i in range(3):
            blk = np.outer(g[:1], g[:1]) * 0 + np.array([
                [g[0]*g[0], g[0]*g[1], g[0]*g[2]],
                [g[0]*g[1], g[1]*g[1], g[1]*g[2]],
                [g[0]*g[2], g[1]*g[2], g[2]*g[2]]]) * q_acc
            Q[i:i+3, i:i+3] = blk
            Q[i+3:i+6, i+3:i+6] += 0.01 * np.eye(3)   # small maneuver noise on vel
        self.P = F @ self.P @ F.T + Q
        H = np.zeros((3, 9))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        y = np.asarray(pos_meas, dtype=float) - H @ self.x
        S = H @ self.P @ H.T + r * np.eye(3)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(9) - K @ H) @ self.P
        return self.x[:3]

    @property
    def vel(self):
        return self.x[3:6]


class TrackerPool:
    """Per-uid tracks. update() once per control tick with all live enemies."""

    def __init__(self, dt=1.0 / 240.0, max_age=10):
        self.dt = dt
        self.max_age = max_age
        self.tracks = {}

    def update(self, enemies):
        """enemies: list of dicts carrying 'uid', 'pos' and optionally 'vel'."""
        seen = set()
        for e in enemies:
            uid = e.get('uid')
            if uid is None:
                continue
            seen.add(uid)
            t = self.tracks.get(uid)
            if t is None:
                t = _Track(e['pos'], self.dt)
                # Seed velocity when truth is available (sim); accelerates
                # convergence until real noisy sensors replace it.
                if e.get('vel') is not None:
                    t.x[3:6] = np.asarray(e['vel'], dtype=float)
                    t.P[3:6, 3:6] = np.eye(3) * 4.0
                self.tracks[uid] = t
            else:
                t.step(e['pos'])
                t.age = 0
        for uid in list(self.tracks):
            if uid not in seen:
                self.tracks[uid].age = getattr(self.tracks[uid], 'age', 0) + 1
                if self.tracks[uid].age > self.max_age:
                    del self.tracks[uid]

    def predict(self, uid, tau):
        """Predicted position of uid tau seconds ahead, or None."""
        t = self.tracks.get(uid)
        return None if t is None else t.predict(tau)

    def velocity(self, uid):
        t = self.tracks.get(uid)
        return None if t is None else t.vel
