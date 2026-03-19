"""
gate_env.py
-----------
Stage 3 environment: wraps any base env and augments it with gate-crossing
detection.

A "gate" is a vertical rectangle in the world frame:
  - aligned to a fixed X plane: x = gate.x_plane
  - open within Y bounds: gate.y_bounds = (y_min, y_max)
  - open within Z bounds: gate.z_bounds = (z_min, z_max)

Crossing is detected when the drone transitions from one side of the X plane
to the other AND is inside the Y/Z opening at crossing time.

Reward
------
The inner env's reward is preserved unchanged.  On a valid gate pass:
  + GATE_PASS_REWARD  sparse bonus
  + shaping term: -GATE_SHAPING_COEF * dist_to_gate_centre (while approaching)
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from goals.gate import Gate

# ── Reward hyper-parameters ───────────────────────────────────────────────────
GATE_PASS_REWARD   = 5.0
GATE_SHAPING_COEF  = 0.2

# Default gate geometry — override by passing a Gate() to __init__.
DEFAULT_GATE = Gate(
    x_plane  = 3.0,
    y_bounds = (-0.5, 0.5),
    z_bounds = (0.5,  1.5),
)


class GateEnv(gym.Wrapper):
    """
    Gymnasium Wrapper that adds gate-crossing detection on top of any
    compatible drone env (HoverEnv, WaypointEnv, or the raw PyFlyt env).

    Parameters
    ----------
    env  : an already-constructed Gymnasium env
    gate : Gate goal object; defaults to DEFAULT_GATE above
    """

    def __init__(self, env: gym.Env, gate: Gate | None = None) -> None:
        super().__init__(env)
        self.gate = gate or DEFAULT_GATE
        self._prev_x: float | None = None

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        # World-frame position lives at the start of the obs; index 0 is
        # a normalised x-displacement in flight_mode=7 so we infer crossing
        # from the actual PyFlyt inner env's world position via info when
        # available, otherwise fall back to obs[0] as a proxy.
        self._prev_x = self._extract_world_x(obs, info)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        curr_x   = self._extract_world_x(obs, info)
        curr_pos = self._extract_world_pos(obs, info)

        # ── Gate shaping: steer toward gate centre while approaching ──────────
        gate_centre = np.array([
            self.gate.x_plane,
            (self.gate.y_bounds[0] + self.gate.y_bounds[1]) / 2.0,
            (self.gate.z_bounds[0] + self.gate.z_bounds[1]) / 2.0,
        ])
        dist_to_gate = float(np.linalg.norm(curr_pos - gate_centre))
        reward -= GATE_SHAPING_COEF * dist_to_gate

        # ── Sparse bonus on a valid crossing ─────────────────────────────────
        if (
            self._prev_x is not None
            and self.gate.is_crossed(self._prev_x, curr_x, curr_pos)
        ):
            reward += GATE_PASS_REWARD
            info["gate_passed"] = True
            terminated = True   # one gate per episode; remove for multi-gate

        self._prev_x = curr_x
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_world_x(obs: np.ndarray, info: dict) -> float:
        # PyFlyt QuadX obs in flight_mode=7 does not directly expose world XYZ.
        # info["state"] carries the full state when available.
        if "state" in info and info["state"] is not None:
            return float(info["state"][0])
        # Fallback: use obs[13] which is dx-to-target (signed, proxy for x)
        return float(obs[13]) if len(obs) > 13 else 0.0

    @staticmethod
    def _extract_world_pos(obs: np.ndarray, info: dict) -> np.ndarray:
        if "state" in info and info["state"] is not None:
            return np.array(info["state"][:3], dtype=np.float32)
        return np.array([obs[13], obs[14], obs[15]], dtype=np.float32)
