"""
waypoint_env.py
---------------
Stage 2 custom environment: open-ended chained waypoint navigation.

Built on the same custom `BaseDroneEnv` stack as HoverEnv (PyFlyt QuadX-Hover-v3)
so control mode, observation transform, and reward shaping are fully owned here.

Behavior
--------
- Random drone spawn in the sky each reset (unless explicit start_pos is provided).
- Spawn one visible transparent goal sphere.
- When the drone reaches the sphere, award a bonus and spawn a new goal.
- Continue until crash or truncation (timeout), not on goal reach.

Observation (10-dim, float32)
-----------------------------
- 0:3   world-frame relative target vector (target_pos - world_pos)
- 3:6   linear velocity (vx, vy, vz)
- 6:10  quaternion (x, y, z, w)
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from .base_drone_env import BaseDroneEnv
from goals.moving_point import MovingPoint

# ── Reward hyper-parameters ───────────────────────────────────────────────────
PROGRESS_COEF = 5.0
PROGRESS_CLIP = 0.20
REACH_RADIUS = 0.15
REACH_BONUS = 1.50
CRASH_PENALTY = 5.0
TIME_PENALTY_PER_STEP = 0.01

# ── Spawn and goal sampling ───────────────────────────────────────────────────
START_XY_RANGE = 1.5
START_Z_MIN = 1.0
START_Z_MAX = 2.5

GOAL_XY_RANGE = 3.0
GOAL_Z_MIN = 1.0
GOAL_Z_MAX = 3.0
GOAL_MIN_SEPARATION = 1.0

# ── Goal visualization ─────────────────────────────────────────────────────────
GOAL_RGBA = (0.2, 0.9, 1.0, 0.35)
GOAL_VISUAL_RADIUS = REACH_RADIUS

# ── Sim ─────────────────────────────────────────────────────────
DEFAULT_EPISODE_SECONDS = 25.0
DEFAULT_FLIGHT_DOME_SIZE = 10.0  # Must accommodate ±3.0m XY goal range (diagonal ~4.24m + buffer)

class WaypointEnv(BaseDroneEnv):
    """Custom chained waypoint navigation environment."""

    metadata = BaseDroneEnv.metadata

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
        goal_low = np.array([-GOAL_XY_RANGE, -GOAL_XY_RANGE, GOAL_Z_MIN], dtype=np.float32)
        goal_high = np.array([GOAL_XY_RANGE, GOAL_XY_RANGE, GOAL_Z_MAX], dtype=np.float32)
        self.goal = MovingPoint(rng_bounds=(goal_low, goal_high))

        self._rng = np.random.default_rng()
        self._sample_spawn_each_reset = start_pos is None
        self._prev_dist: float | None = None
        self._waypoints_reached = 0

        self._goal_visual_shape_id: int | None = None
        self._goal_visual_body_id: int | None = None

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = dict(options or {})

        if "start_pos" not in options and self._sample_spawn_each_reset:
            options["start_pos"] = self._sample_start_pos()

        raw_obs, info = super().reset(seed=seed, options=options)
        world_pos = np.asarray(raw_obs[10:13], dtype=np.float32)

        self._waypoints_reached = 0
        self._clear_goal_visual_handles()
        self._spawn_next_goal_away_from(world_pos)
        self._ensure_goal_visual()
        self._move_goal_visual(self.goal.pos)

        self._prev_dist = self.goal.distance_from_world(world_pos)

        info["world_pos"] = world_pos.copy()
        info["target_pos"] = self.goal.pos.copy()
        info["distance_to_target"] = float(self._prev_dist)
        info["progress_to_target"] = 0.0
        info["goal_reached"] = False
        info["waypoints_reached"] = int(self._waypoints_reached)

        obs = self._transform_obs(raw_obs)
        return obs, info

    def step(self, action):
        action_array = self._coerce_action(action)
        raw_obs, raw_reward, inner_terminated, truncated, info = self._inner.step(action_array)

        info["_inner_terminated"] = bool(inner_terminated)
        world_pos = np.asarray(raw_obs[10:13], dtype=np.float32)
        info["world_pos"] = world_pos.copy()

        dist = self.goal.distance_from_world(world_pos)
        if self._prev_dist is None:
            progress = 0.0
        else:
            progress = float(np.clip(self._prev_dist - dist, -PROGRESS_CLIP, PROGRESS_CLIP))

        reward = PROGRESS_COEF * progress
        goal_reached = dist < REACH_RADIUS
        if goal_reached:
            reward += REACH_BONUS
            self._waypoints_reached += 1
            self._spawn_next_goal_away_from(world_pos)
            self._move_goal_visual(self.goal.pos)
            dist = self.goal.distance_from_world(world_pos)

        crashed = bool(inner_terminated)
        if crashed:
            reward -= CRASH_PENALTY

        reward -= TIME_PENALTY_PER_STEP

        self._prev_dist = dist

        info["target_pos"] = self.goal.pos.copy()
        info["distance_to_target"] = float(dist)
        info["progress_to_target"] = float(progress)
        info["goal_reached"] = bool(goal_reached)
        info["waypoints_reached"] = int(self._waypoints_reached)
        info["waypoint_reward"] = float(reward)

        terminated = inner_terminated or self.is_terminated(raw_obs, info)
        obs = self._transform_obs(raw_obs)
        return obs, reward, terminated, truncated, info

    def close(self):
        self._remove_goal_visual()
        super().close()

    def reset_goal(self) -> None:
        return None

    def is_terminated(self, obs: np.ndarray, info: dict) -> bool:
        return False

    # ------------------------------------------------------------------
    # Observation transform
    # ------------------------------------------------------------------

    def _transform_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_obs, dtype=np.float32)
        world_pos = raw[10:13]
        rel_target = self.goal.pos.astype(np.float32) - world_pos
        velocity = raw[7:10]
        quaternion = raw[3:7]
        return np.concatenate((rel_target, velocity, quaternion), dtype=np.float32)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_start_pos(self) -> list[float]:
        x = float(self._rng.uniform(-START_XY_RANGE, START_XY_RANGE))
        y = float(self._rng.uniform(-START_XY_RANGE, START_XY_RANGE))
        z = float(self._rng.uniform(START_Z_MIN, START_Z_MAX))
        return [x, y, z]

    def _spawn_next_goal_away_from(self, drone_pos: np.ndarray) -> None:
        for _ in range(64):
            candidate = self.goal.reset()
            if np.linalg.norm(candidate - drone_pos) >= GOAL_MIN_SEPARATION:
                return

        direction = drone_pos[:2] / (np.linalg.norm(drone_pos[:2]) + 1e-6)
        fallback = np.array(
            [
                float(np.clip(-direction[0] * GOAL_XY_RANGE, -GOAL_XY_RANGE, GOAL_XY_RANGE)),
                float(np.clip(-direction[1] * GOAL_XY_RANGE, -GOAL_XY_RANGE, GOAL_XY_RANGE)),
                float(np.clip(drone_pos[2] + 0.75, GOAL_Z_MIN, GOAL_Z_MAX)),
            ],
            dtype=np.float32,
        )
        self.goal.pos = fallback

    # ------------------------------------------------------------------
    # Goal visualization
    # ------------------------------------------------------------------

    def _is_render_enabled(self) -> bool:
        return self.render_mode in {"human", "rgb_array"}

    def _get_bullet_client(self):
        inner_unwrapped = getattr(self._inner, "unwrapped", None)
        return getattr(inner_unwrapped, "env", None)

    def _ensure_goal_visual(self) -> None:
        if not self._is_render_enabled():
            return
        if self._goal_visual_body_id is not None:
            return

        bullet = self._get_bullet_client()
        if bullet is None:
            return

        self._goal_visual_shape_id = bullet.createVisualShape(
            shapeType=bullet.GEOM_SPHERE,
            radius=GOAL_VISUAL_RADIUS,
            rgbaColor=GOAL_RGBA,
            specularColor=[0.0, 0.0, 0.0],
        )
        self._goal_visual_body_id = bullet.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=int(self._goal_visual_shape_id),
            baseCollisionShapeIndex=-1,
            basePosition=self.goal.pos.tolist(),
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
        )

    def _move_goal_visual(self, pos: np.ndarray) -> None:
        if self._goal_visual_body_id is None:
            return
        bullet = self._get_bullet_client()
        if bullet is None:
            return
        bullet.resetBasePositionAndOrientation(
            int(self._goal_visual_body_id),
            np.asarray(pos, dtype=np.float32).tolist(),
            [0.0, 0.0, 0.0, 1.0],
        )

    def _remove_goal_visual(self) -> None:
        if self._goal_visual_body_id is None:
            return
        bullet = self._get_bullet_client()
        if bullet is not None:
            try:
                bullet.removeBody(int(self._goal_visual_body_id))
            except Exception:
                pass
        self._clear_goal_visual_handles()

    def _clear_goal_visual_handles(self) -> None:
        self._goal_visual_shape_id = None
        self._goal_visual_body_id = None
