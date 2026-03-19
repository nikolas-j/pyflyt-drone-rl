"""
static_point.py
---------------
A fixed 3-D target position.

Used by HoverEnv as the hover setpoint.  The position can be set at
construction time and reset each episode if desired.
"""

from __future__ import annotations

import numpy as np


class StaticPoint:
    """
    Parameters
    ----------
    pos : target position in world frame (x, y, z), metres
    """

    def __init__(self, pos: np.ndarray | None = None) -> None:
        self.pos: np.ndarray = (
            np.array(pos, dtype=np.float32)
            if pos is not None
            else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        )

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    def distance_from_world(self, drone_pos: np.ndarray) -> float:
        """Euclidean distance from drone_pos to this target."""
        return float(np.linalg.norm(np.asarray(drone_pos, dtype=np.float32) - self.pos))

    def distance_from_displacement(self, displacement: np.ndarray) -> float:
        """
        Euclidean distance from a (dx, dy, dz) displacement vector.

        PyFlyt's observation indices 13-15 give the vector from the drone
        to the target, so ||displacement|| == distance to target.
        """
        return float(np.linalg.norm(np.asarray(displacement, dtype=np.float32)))
