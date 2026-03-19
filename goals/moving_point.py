"""
moving_point.py
---------------
A goal point that can drift, orbit, or follow a programmatic trajectory.

Used by WaypointEnv.  On each episode reset() a new random position is
sampled.  The goal can optionally tick() each timestep to move.

Extend this class to implement more ambitious movement patterns, e.g.:
    class OrbitingPoint(MovingPoint):
        def tick(self, dt): ...position += orbit step...
"""

from __future__ import annotations

import numpy as np


class MovingPoint:
    """
    Parameters
    ----------
    rng_bounds : (low, high) uniform range for each axis when sampling a
                 new random position at reset.  Defaults to a 2m cube at
                 altitude 1-3 m.
    """

    def __init__(
        self,
        rng_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        low  = np.array([-2.0, -2.0,  1.0], dtype=np.float32)
        high = np.array([ 2.0,  2.0,  3.0], dtype=np.float32)
        self._low, self._high = (low, high) if rng_bounds is None else rng_bounds
        self.pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Sample a new random goal position and return it."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.pos = self._rng.uniform(self._low, self._high).astype(np.float32)
        return self.pos

    def tick(self, dt: float = 0.02) -> np.ndarray:
        """
        Advance goal by one timestep.  Base implementation: stationary.
        Override in a subclass for animated goals (orbiting, sinusoidal, etc.)
        """
        return self.pos

    # ------------------------------------------------------------------

    def distance_from_displacement(self, displacement: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(displacement, dtype=np.float32)))
