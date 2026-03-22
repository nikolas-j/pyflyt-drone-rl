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

    def __init__(
        self,
        render_mode: str | None = None,
        start_pos: np.ndarray | list[float] | tuple[float, float, float] | None = None,
    ) -> None:
        super().__init__()
        self._inner: gym.Env = gym.make(
            ENV_ID,
            flight_mode=FLIGHT_MODE,
            render_mode=render_mode,
        )
        self.observation_space = self._inner.observation_space
        self.action_space      = self._inner.action_space
        if isinstance(self.action_space, gym.spaces.Box) and self.action_space.dtype != np.float32:
            self.action_space = gym.spaces.Box(
                low=np.asarray(self.action_space.low, dtype=np.float32),
                high=np.asarray(self.action_space.high, dtype=np.float32),
                shape=self.action_space.shape,
                dtype=np.float32,
            )
        self.render_mode       = render_mode
        self.goal              = MovingPoint()
        self._start_pos        = self._coerce_start_pos(start_pos)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        options = dict(options or {})
        requested_start_pos = options.pop("start_pos", None)
        if requested_start_pos is not None:
            self._start_pos = self._coerce_start_pos(requested_start_pos)

        if self._start_pos is not None and hasattr(self._inner.unwrapped, "start_pos"):
            self._inner.unwrapped.start_pos = self._start_pos.copy()

        obs, info = self._inner.reset(seed=seed, options=options or None)
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

    @staticmethod
    def _coerce_start_pos(
        start_pos: np.ndarray | list[float] | tuple[float, float, float] | None,
    ) -> np.ndarray | None:
        if start_pos is None:
            return None
        arr = np.asarray(start_pos, dtype=np.float64)
        if arr.shape == (3,):
            arr = arr.reshape(1, 3)
        if arr.shape != (1, 3):
            raise ValueError(
                "start_pos must be shape (3,) or (1, 3), e.g. [0.0, 0.0, 1.5]"
            )
        return arr
