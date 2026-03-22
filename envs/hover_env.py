"""
hover_env.py
------------
Stage 1 environment: fly to and hover around a fixed target point.

Model observation (10-dim)
--------------------------
The policy receives only the MVP navigation signal:
- 0:3   relative target vector (target_pos - world_pos)
- 3:6   linear velocity (vx, vy, vz)
- 6:10  orientation quaternion (x, y, z, w)

PyFlyt still provides a raw 21-dim observation internally. This wrapper
transforms raw observations before returning them to PPO.

Reward
------
r = PROGRESS_COEF * clip(prev_dist - dist, -PROGRESS_CLIP, +PROGRESS_CLIP)
    + SUCCESS_BONUS   if dist < SUCCESS_RADIUS
    - CRASH_PENALTY   if inner env terminates

Spawn policy (default)
----------------------
Unless an explicit start_pos is provided, each episode starts from an easy
distribution around z≈1 so PPO can reliably discover climb-to-target behavior.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from .base_drone_env import BaseDroneEnv
from goals.static_point import StaticPoint

# ── Reward hyper-parameters ───────────────────────────────────────────────────
PROGRESS_COEF  = 5.0    # scale progress term to roughly [-1, 1] per step
PROGRESS_CLIP  = 0.20   # clip progress spikes for stable PPO updates
SUCCESS_RADIUS = 0.30   # metres — "inside goal" threshold
SUCCESS_BONUS  = 0.20   # per-step bonus while inside SUCCESS_RADIUS
CRASH_PENALTY  = 5.0    # one-shot penalty when inner env terminates

# Easy spawn distribution for hover learning (if start_pos not set explicitly)
SPAWN_XY_RANGE = 0.25
SPAWN_Z_MIN    = 0.90
SPAWN_Z_MAX    = 1.10

# Target position.  Change to (0.0, 0.0, 2.0) etc. to hover at other heights.
TARGET_POS = np.array([0.0, 3.0, 2.0], dtype=np.float32)

DEFAULT_EPISODE_SECONDS = 12.0
DEFAULT_FLIGHT_DOME_SIZE = 2.0  # Small allowable area since spawn range is only ±0.25m


class HoverEnv(BaseDroneEnv):
    """Hover in place at TARGET_POS."""

    def __init__(
        self,
        render_mode: str | None = None,
        start_pos: np.ndarray | list[float] | tuple[float, float, float] | None = None,
        max_duration_seconds: float = DEFAULT_EPISODE_SECONDS,
        flight_dome_size: float = DEFAULT_FLIGHT_DOME_SIZE,
    ) -> None:
        super().__init__(
            render_mode=render_mode,
            start_pos=start_pos,
            max_duration_seconds=max_duration_seconds,
            flight_dome_size=flight_dome_size,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32,
        )
        self.goal = StaticPoint(pos=TARGET_POS)
        self._last_action: np.ndarray = np.zeros(4, dtype=np.float32)
        self._prev_dist: float | None = None
        self._rng = np.random.default_rng()
        self._sample_spawn_each_reset = start_pos is None

    # ------------------------------------------------------------------
    # Gymnasium overrides
    # ------------------------------------------------------------------

    def step(self, action):
        self._last_action = self._coerce_action(action)
        raw_obs, raw_reward, inner_terminated, truncated, info = self._inner.step(self._last_action)
        info["_inner_terminated"] = bool(inner_terminated)
        info["world_pos"] = np.asarray(raw_obs[10:13], dtype=np.float32).copy()

        reward = self.shape_reward(raw_obs, raw_reward, info)
        terminated = inner_terminated or self.is_terminated(raw_obs, info)

        obs = self._transform_obs(raw_obs)
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = dict(options or {})

        # Respect explicit starts, otherwise sample an easy episode start.
        if "start_pos" not in options and self._sample_spawn_each_reset:
            options["start_pos"] = self._sample_easy_start_pos()

        raw_obs, info = super().reset(seed=seed, options=options)
        info["world_pos"] = np.asarray(raw_obs[10:13], dtype=np.float32).copy()

        self._prev_dist = self.goal.distance_from_world(raw_obs[10:13])
        info["distance_to_target"] = float(self._prev_dist)
        info["target_pos"] = self.goal.pos.copy()

        obs = self._transform_obs(raw_obs)
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

        reward = PROGRESS_COEF * progress
        if dist < SUCCESS_RADIUS:
            reward += SUCCESS_BONUS

        crashed = bool(info.get("_inner_terminated", False))
        if crashed:
            reward -= CRASH_PENALTY

        info["distance_to_target"] = float(dist)
        info["progress_to_target"] = float(progress)
        info["hover_reward"] = float(reward)
        info["hover_success"] = bool(dist < SUCCESS_RADIUS)
        info["hover_crash"] = crashed

        self._prev_dist = dist
        return reward

    def _sample_easy_start_pos(self) -> list[float]:
        x = float(self._rng.uniform(-SPAWN_XY_RANGE, SPAWN_XY_RANGE))
        y = float(self._rng.uniform(-SPAWN_XY_RANGE, SPAWN_XY_RANGE))
        z = float(self._rng.uniform(SPAWN_Z_MIN, SPAWN_Z_MAX))
        return [x, y, z]

    def _transform_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_obs, dtype=np.float32)
        world_pos = raw[10:13]
        rel_target = self.goal.pos.astype(np.float32) - world_pos
        velocity = raw[7:10]
        quaternion = raw[3:7]
        return np.concatenate((rel_target, velocity, quaternion), dtype=np.float32)

    def is_terminated(self, obs: np.ndarray, info: dict) -> bool:
        return False   # rely on PyFlyt's own crash / timeout termination
