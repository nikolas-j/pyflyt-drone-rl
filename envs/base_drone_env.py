"""
base_drone_env.py
-----------------
Thin Gymnasium wrapper around a PyFlyt QuadX environment.

Responsibilities
----------------
- Registers PyFlyt envs with Gymnasium (via the side-effect import).
- Creates the inner env with the correct flight_mode and render_mode.
- Coerces every action to np.float32 so PyFlyt's internal copy() calls
  never fail regardless of what the caller passes.
- Exposes the raw observation and action spaces unchanged so subclasses
  can add their own shaping without fighting the base class.

Subclasses override
-------------------
- `compute_reward(obs, raw_reward, info)` — return a float
- `is_terminated(obs, info)`              — return bool
- `reset_goal()`                          — called at the start of each
                                            episode; subclasses spawn goals
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import PyFlyt.gym_envs  # noqa: F401  Side-effect: registers PyFlyt envs.

ENV_ID     = "PyFlyt/QuadX-Hover-v3"
FLIGHT_MODE = 7   # high-level velocity control: [vx, vy, vz, yaw_rate]


class BaseDroneEnv(gym.Env):
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

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self._inner.reset(seed=seed, options=options)
        self.reset_goal()
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        obs, raw_reward, terminated, truncated, info = self._inner.step(action)

        reward     = self.compute_reward(obs, raw_reward, info)
        terminated = terminated or self.is_terminated(obs, info)

        return obs, reward, terminated, truncated, info

    def render(self):
        return self._inner.render()

    def close(self):
        self._inner.close()

    # ------------------------------------------------------------------
    # Extension points for subclasses
    # ------------------------------------------------------------------

    def compute_reward(self, obs: np.ndarray, raw_reward: float, info: dict) -> float:
        """Override in subclasses. Default: pass PyFlyt's built-in reward through."""
        return float(raw_reward)

    def is_terminated(self, obs: np.ndarray, info: dict) -> bool:
        """Override to add extra termination conditions."""
        return False

    def reset_goal(self) -> None:
        """Override to spawn / reset goal objects at episode start."""
