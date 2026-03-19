"""
waypoint_env.py
---------------
Stage 2 environment: fly to a randomly-spawned goal that resets each episode.

Uses PyFlyt/QuadX-Waypoints-v3 under the hood where the env itself provides
a goal vector in observation indices 13-15 (target displacement dx, dy, dz).

Reward
------
r = -dist(drone, waypoint)          dense shaping
  + REACH_BONUS                     if drone reaches within REACH_THRESHOLD
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import PyFlyt.gym_envs  # noqa: F401

from goals.moving_point import MovingPoint

# ── Reward hyper-parameters ───────────────────────────────────────────────────
REACH_THRESHOLD = 0.30   # metres
REACH_BONUS     = 2.00   # one-shot bonus on reaching the waypoint

# PyFlyt env to use for this stage
ENV_ID      = "PyFlyt/QuadX-Waypoints-v3"
FLIGHT_MODE = 7


class WaypointEnv(gym.Env):
    """Navigate to a randomly-placed waypoint goal."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self._inner: gym.Env = gym.make(
            ENV_ID,
            flight_mode=FLIGHT_MODE,
            render_mode=render_mode,
        )
        self.observation_space = self._inner.observation_space
        self.action_space      = self._inner.action_space
        self.render_mode       = render_mode
        self.goal              = MovingPoint()

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self._inner.reset(seed=seed, options=options)
        self.goal.reset()
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        obs, _, terminated, truncated, info = self._inner.step(action)

        dist   = float(np.linalg.norm(obs[13:16]))
        reward = -dist
        if dist < REACH_THRESHOLD:
            reward += REACH_BONUS

        return obs, reward, terminated, truncated, info

    def render(self):
        return self._inner.render()

    def close(self):
        self._inner.close()
