"""
hover_env.py
------------
Stage 1 environment: reward the agent for hovering near a fixed point.

Observation layout (21-dim, from PyFlyt QuadX-Hover-v3 in flight_mode=7)
-------------------------------------------------------------------------
Index  Meaning
  0-2  Angular velocities (body frame)
  3-5  Linear velocities  (body frame)
  6-8  Quaternion (x, y, z, w)  — orientation
  9    Altitude error to target
 10-12 Linear velocities (world frame)
 13-15 Target displacement (dx, dy, dz)
 16-20 Motor RPMs (normalised)

We extract position information from the target-displacement indices (13-15)
to compute the shaped reward.

Reward
------
r = -dist(drone, target)                  dense distance penalty
  + STAY_BONUS  if dist < STAY_THRESHOLD  sustained hover bonus
  - EFFORT_COEF * ||action||²             small control effort penalty
"""

from __future__ import annotations

import numpy as np

from .base_drone_env import BaseDroneEnv
from goals.static_point import StaticPoint

# ── Reward hyper-parameters ───────────────────────────────────────────────────
STAY_THRESHOLD = 0.20   # metres — "close enough" for bonus
STAY_BONUS     = 0.10   # extra reward per step while inside threshold
EFFORT_COEF    = 0.005  # penalty per unit of squared action norm

# Target position.  Change to (0.0, 0.0, 2.0) etc. to hover at other heights.
TARGET_POS = np.array([0.0, 0.0, 1.0], dtype=np.float32)


class HoverEnv(BaseDroneEnv):
    """Hover in place at TARGET_POS."""

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__(render_mode=render_mode)
        self.goal = StaticPoint(pos=TARGET_POS)
        self._last_action: np.ndarray = np.zeros(4, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium overrides
    # ------------------------------------------------------------------

    def step(self, action):
        self._last_action = np.asarray(action, dtype=np.float32)
        return super().step(action)

    def reset_goal(self) -> None:
        self.goal = StaticPoint(pos=TARGET_POS)

    # ------------------------------------------------------------------
    # Reward / termination
    # ------------------------------------------------------------------

    def compute_reward(self, obs: np.ndarray, raw_reward: float, info: dict) -> float:
        dist   = self.goal.distance_from_displacement(obs[13:16])
        effort = float(np.dot(self._last_action, self._last_action))
        reward = -dist - EFFORT_COEF * effort
        if dist < STAY_THRESHOLD:
            reward += STAY_BONUS
        return reward

    def is_terminated(self, obs: np.ndarray, info: dict) -> bool:
        return False   # rely on PyFlyt's own crash / timeout termination
