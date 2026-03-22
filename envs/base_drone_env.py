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
- `shape_reward(obs, raw_reward, info)` — return a float
- `is_terminated(obs, info)`              — return bool
- `reset_goal()`                          — called at the start of each
                                            episode; subclasses spawn goals
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import PyFlyt.gym_envs  # noqa: F401  Side-effect: registers PyFlyt envs.

ENV_ID     = "PyFlyt/QuadX-Hover-v3"
FLIGHT_MODE = 6   # ground-velocity control: [vx, vy, yaw_rate, vz]
DEFAULT_EPISODE_SECONDS = 10.0


class BaseDroneEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        render_mode: str | None = None,
        start_pos: np.ndarray | list[float] | tuple[float, float, float] | None = None,
        max_duration_seconds: float = DEFAULT_EPISODE_SECONDS,
        flight_dome_size: float = 10.0,
    ) -> None:
        super().__init__()
        self._inner: gym.Env = gym.make(
            ENV_ID,
            flight_mode=FLIGHT_MODE,
            render_mode=render_mode,
            max_duration_seconds=max_duration_seconds,
            flight_dome_size=flight_dome_size,
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
        if FLIGHT_MODE == 6 and isinstance(self.action_space, gym.spaces.Box):
            low = np.asarray(self.action_space.low, dtype=np.float32).copy()
            high = np.asarray(self.action_space.high, dtype=np.float32).copy()
            low[3] = -high[3]
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.render_mode       = render_mode
        self._start_pos        = self._coerce_start_pos(start_pos)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        options = dict(options or {})

        # Optional per-episode spawn override: reset(options={"start_pos": [x, y, z]})
        requested_start_pos = options.pop("start_pos", None)
        if requested_start_pos is not None:
            self._start_pos = self._coerce_start_pos(requested_start_pos)

        # PyFlyt QuadX envs consume `unwrapped.start_pos` during begin_reset().
        if self._start_pos is not None and hasattr(self._inner.unwrapped, "start_pos"):
            self._inner.unwrapped.start_pos = self._start_pos.copy()

        obs, info = self._inner.reset(seed=seed, options=options or None)
        self.reset_goal()
        return obs, info

    def step(self, action):
        action = self._coerce_action(action)
        obs, raw_reward, terminated, truncated, info = self._inner.step(action)
        info["_inner_terminated"] = bool(terminated)

        reward     = self.shape_reward(obs, raw_reward, info)
        terminated = terminated or self.is_terminated(obs, info)

        return obs, reward, terminated, truncated, info

    def render(self):
        return self._inner.render()

    def close(self):
        self._inner.close()

    # ------------------------------------------------------------------
    # Spawn position helpers
    # ------------------------------------------------------------------

    def _coerce_action(self, action) -> np.ndarray:
        action_array = np.asarray(action, dtype=np.float32)
        if isinstance(self.action_space, gym.spaces.Box):
            action_array = np.clip(action_array, self.action_space.low, self.action_space.high)
        return action_array

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

    # ------------------------------------------------------------------
    # Extension points for subclasses
    # ------------------------------------------------------------------

    def shape_reward(self, obs: np.ndarray, raw_reward: float, info: dict) -> float:
        """Override in subclasses. Default: pass PyFlyt's built-in reward through."""
        return float(raw_reward)

    def is_terminated(self, obs: np.ndarray, info: dict) -> bool:
        """Override to add extra termination conditions."""
        return False

    def reset_goal(self) -> None:
        """Override to spawn / reset goal objects at episode start."""
