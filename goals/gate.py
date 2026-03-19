"""
gate.py
-------
A rectangular gate defined by an X-plane and Y/Z opening bounds.

The gate occupies the plane x = x_plane.  A drone "passes through" the gate
when it crosses from the approach side (x < x_plane) to the exit side
(x > x_plane) while within the opening:
    y_bounds[0] <= drone_y <= y_bounds[1]
    z_bounds[0] <= drone_z <= z_bounds[1]

Optionally works for gates along other axes by extending the design here.
"""

from __future__ import annotations

import numpy as np


class Gate:
    """
    Parameters
    ----------
    x_plane  : world-frame X coordinate of the gate plane
    y_bounds : (y_min, y_max) opening in the Y axis
    z_bounds : (z_min, z_max) opening in the Z axis
    """

    def __init__(
        self,
        x_plane:  float,
        y_bounds: tuple[float, float],
        z_bounds: tuple[float, float],
    ) -> None:
        self.x_plane  = float(x_plane)
        self.y_bounds = (float(y_bounds[0]), float(y_bounds[1]))
        self.z_bounds = (float(z_bounds[0]), float(z_bounds[1]))

    # ------------------------------------------------------------------

    @property
    def centre(self) -> np.ndarray:
        """World-frame centre of the gate opening."""
        return np.array([
            self.x_plane,
            (self.y_bounds[0] + self.y_bounds[1]) / 2.0,
            (self.z_bounds[0] + self.z_bounds[1]) / 2.0,
        ], dtype=np.float32)

    def is_crossed(
        self,
        prev_x:   float,
        curr_x:   float,
        curr_pos: np.ndarray,
    ) -> bool:
        """
        Return True when the drone crosses the x_plane in the positive
        direction and is inside the Y/Z opening.

        Parameters
        ----------
        prev_x   : drone world-X at the previous timestep
        curr_x   : drone world-X at the current timestep
        curr_pos : full (x, y, z) world position at current timestep
        """
        # Only count positive-direction crossing (approach → exit)
        crossed_plane = (prev_x - self.x_plane) * (curr_x - self.x_plane) < 0
        if not crossed_plane:
            return False

        y, z = float(curr_pos[1]), float(curr_pos[2])
        in_opening = (
            self.y_bounds[0] <= y <= self.y_bounds[1]
            and self.z_bounds[0] <= z <= self.z_bounds[1]
        )
        return in_opening

    def distance_to_centre(self, pos: np.ndarray) -> float:
        """Euclidean distance from pos to the gate centre."""
        return float(np.linalg.norm(np.asarray(pos, dtype=np.float32) - self.centre))
