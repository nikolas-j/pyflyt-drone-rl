"""
hover_env.py
------------
Stage 1 environment: reward the agent for hovering near a fixed point.

Observation layout (21-dim, from PyFlyt QuadX-Hover-v3 in flight_mode=7)
-------------------------------------------------------------------------
Index  Meaning
    0-2  Angular velocities
    3-6  Quaternion (x, y, z, w)
    7-9  Linear velocities
 10-12 World position (x, y, z)
 13-16 Previous action
 17-20 Auxiliary state / motor data

Hover-v3 does not expose a target-displacement vector, so we compute reward
from the drone's world position at indices 10-12.

Reward
------
r = -DIST_COEF * dist(drone, target)            dense distance shaping
    + PROGRESS_COEF * (prev_dist - dist)          reward moving toward target
    + STAY_BONUS  if dist < STAY_THRESHOLD         stronger near-target reward
    - EFFORT_COEF * ||action||²                    optional control penalty

Spawn policy (default)
----------------------
Unless an explicit start_pos is provided, each episode starts from an easy
distribution around z≈1 so PPO can reliably discover climb-to-hover behavior.
"""

from __future__ import annotations

import numpy as np

from .base_drone_env import BaseDroneEnv
from goals.static_point import StaticPoint

# ── Reward hyper-parameters ───────────────────────────────────────────────────
DIST_COEF      = 1.0    # global distance shaping scale
PROGRESS_COEF  = 2.0    # weight for per-step progress (prev_dist - dist)
PROGRESS_CLIP  = 0.20   # clip progress to stabilise rare spikes
STAY_THRESHOLD = 0.20   # metres — "close enough" for hover bonus
STAY_BONUS     = 0.30   # extra reward per step while inside threshold
EFFORT_COEF    = 0.0    # keep at 0 initially to avoid suppressing exploration

# Easy spawn distribution for hover learning (if start_pos not set explicitly)
SPAWN_XY_RANGE = 0.25
SPAWN_Z_MIN    = 0.90
SPAWN_Z_MAX    = 1.10

# Target position.  Change to (0.0, 0.0, 2.0) etc. to hover at other heights.
TARGET_POS = np.array([0.0, 0.0, 2.0], dtype=np.float32)


class HoverEnv(BaseDroneEnv):
    """Hover in place at TARGET_POS."""

    def __init__(
        self,
        render_mode: str | None = None,
        start_pos: np.ndarray | list[float] | tuple[float, float, float] | None = None,
    ) -> None:
        super().__init__(render_mode=render_mode, start_pos=start_pos)
        self.goal = StaticPoint(pos=TARGET_POS)
        self._last_action: np.ndarray = np.zeros(4, dtype=np.float32)
        self._prev_dist: float | None = None
        self._rng = np.random.default_rng()
        self._sample_spawn_each_reset = start_pos is None

    # ------------------------------------------------------------------
    # Gymnasium overrides
    # ------------------------------------------------------------------

    def step(self, action):
        self._last_action = np.asarray(action, dtype=np.float32)
        return super().step(action)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = dict(options or {})

        # Respect explicit starts, otherwise sample an easy episode start.
        if "start_pos" not in options and self._sample_spawn_each_reset:
            options["start_pos"] = self._sample_easy_start_pos()

        obs, info = super().reset(seed=seed, options=options)
        self._prev_dist = self.goal.distance_from_world(obs[10:13])
        info["distance_to_target"] = float(self._prev_dist)
        info["target_pos"] = self.goal.pos.copy()
        return obs, info

    def reset_goal(self) -> None:
        self.goal = StaticPoint(pos=TARGET_POS)

    # ------------------------------------------------------------------
    # Reward / termination
    # ------------------------------------------------------------------

    def shape_reward(self, obs: np.ndarray, raw_reward: float, info: dict) -> float:
        dist = self.goal.distance_from_world(obs[10:13])

        if self._prev_dist is None:
            progress = 0.0
        else:
            progress = float(np.clip(self._prev_dist - dist, -PROGRESS_CLIP, PROGRESS_CLIP))

        effort = float(np.dot(self._last_action, self._last_action))
        reward = -DIST_COEF * dist + PROGRESS_COEF * progress - EFFORT_COEF * effort
        if dist < STAY_THRESHOLD:
            reward += STAY_BONUS

        info["distance_to_target"] = float(dist)
        info["progress_to_target"] = float(progress)
        info["hover_reward"] = float(reward)

        self._prev_dist = dist
        return reward

    def _sample_easy_start_pos(self) -> list[float]:
        x = float(self._rng.uniform(-SPAWN_XY_RANGE, SPAWN_XY_RANGE))
        y = float(self._rng.uniform(-SPAWN_XY_RANGE, SPAWN_XY_RANGE))
        z = float(self._rng.uniform(SPAWN_Z_MIN, SPAWN_Z_MAX))
        return [x, y, z]

    def is_terminated(self, obs: np.ndarray, info: dict) -> bool:
        return False   # rely on PyFlyt's own crash / timeout termination
