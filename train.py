"""
train.py — Train or play a PPO agent on a PyFlyt drone task.

Usage
─────
  # Train hover from scratch (headless, fast)
  uv run python train.py --task hover --mode train

  # Continue training a saved hover model
  uv run python train.py --task hover --mode train --load models/hover_ppo

  # Watch the trained hover agent fly (opens PyBullet GUI)
  uv run python train.py --task hover --mode play

  # Train the waypoint-navigation task
  uv run python train.py --task waypoint --mode train

  # Train the gate task (wraps hover env with a gate crossing detector)
  uv run python train.py --task gate --mode train
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from envs import HoverEnv, WaypointEnv, GateEnv
from envs.gate_env import GateEnv as _GateEnv
from goals.gate import Gate
from model_registry import (
    format_model_info,
    get_or_create_metadata,
    mark_play,
    mark_train_end,
    mark_train_start,
    save_metadata,
    set_total_timesteps,
)


WAYPOINT_START_POS = [0.0, 0.0, 1.0]


# ── Task registry ─────────────────────────────────────────────────────────────
# Maps --task name to a factory that takes render_mode and returns a gym.Env.
# For GateEnv we wrap HoverEnv: change the inner env here to WaypointEnv for
# a harder variant.

def _make_hover(render_mode):
    return HoverEnv(render_mode=render_mode)

def _make_waypoint(render_mode):
    return WaypointEnv(render_mode=render_mode, start_pos=WAYPOINT_START_POS)

def _make_gate(render_mode):
    inner = HoverEnv(render_mode=render_mode)
    return GateEnv(inner, gate=Gate(x_plane=3.0, y_bounds=(-0.5, 0.5), z_bounds=(0.5, 1.5)))

ENV_FACTORIES = {
    "hover":    _make_hover,
    "waypoint": _make_waypoint,
    "gate":     _make_gate,
}


# ── Paths ─────────────────────────────────────────────────────────────────────
LOG_DIR    = "./logs/"
MODEL_DIR  = "./models/"


# ── PPO defaults ──────────────────────────────────────────────────────────────
PPO_DEFAULTS = dict(
    policy         = "MlpPolicy",
    n_steps        = 1024,
    batch_size     = 128,
    n_epochs       = 5,
    learning_rate  = 2e-4,
    clip_range     = 0.2,
    gamma          = 0.99,
    ent_coef       = 0.01,
    verbose        = 1,
)

DEFAULT_TIMESTEPS = 200_000
PLAY_COORD_PRINT_EVERY = 5


TASK_PROFILES = {
    "hover": {
        "env_id": "PyFlyt/QuadX-Hover-v3",
        "flight_mode": 7,
        "reward_profile": "hover_dense_progress_v1",
    },
    "waypoint": {
        "env_id": "PyFlyt/QuadX-Waypoints-v3",
        "flight_mode": 7,
        "reward_profile": "waypoint_distance_bonus_v1",
    },
    "gate": {
        "env_id": "PyFlyt/QuadX-Hover-v3 + GateEnv",
        "flight_mode": 7,
        "reward_profile": "gate_shaping_sparse_v1",
    },
}


def _load_metadata(task: str, model_path: str) -> dict:
    profile = TASK_PROFILES[task]
    metadata = get_or_create_metadata(
        model_path,
        task=task,
        algorithm="PPO",
        network_description=str(PPO_DEFAULTS["policy"]),
        env_id=profile["env_id"],
        flight_mode=profile["flight_mode"],
        reward_profile=profile["reward_profile"],
    )
    return metadata


def _bootstrap_timesteps_from_checkpoint(model_path: str, metadata: dict) -> dict:
    has_model = os.path.exists(model_path + ".zip")
    total_known = int(metadata.get("total_timesteps", 0)) > 0

    if not has_model or total_known:
        return metadata

    try:
        checkpoint = PPO.load(model_path)
        metadata = set_total_timesteps(metadata, int(checkpoint.num_timesteps))
        save_metadata(model_path, metadata)
    except Exception as exc:
        print(f"[model] WARNING: could not infer timesteps from '{model_path}.zip': {exc}")

    return metadata


def _print_model_info(mode: str, model_path: str, metadata: dict) -> None:
    print(f"[model] mode={mode} path='{model_path}.zip'")
    print(format_model_info(metadata))
    print()


def _extract_world_pos(obs, info: dict) -> tuple[float, float, float] | None:
    if "state" in info and info["state"] is not None:
        state = info["state"]
        if len(state) >= 3:
            return (float(state[0]), float(state[1]), float(state[2]))

    if obs is not None and len(obs) >= 13:
        return (float(obs[10]), float(obs[11]), float(obs[12]))

    return None


# ─────────────────────────────────────────────────────────────────────────────
#  train()
# ─────────────────────────────────────────────────────────────────────────────

def train(task: str, load_path: str | None, total_timesteps: int) -> None:
    os.makedirs(LOG_DIR,   exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Source model for loading metadata visibility (resume or default checkpoint).
    source_model_path = load_path or os.path.join(MODEL_DIR, f"{task}_ppo")
    source_meta = _load_metadata(task, source_model_path)
    source_meta = _bootstrap_timesteps_from_checkpoint(source_model_path, source_meta)
    _print_model_info("train", source_model_path, source_meta)

    # Target model path that this script saves to.
    save_path = os.path.join(MODEL_DIR, f"{task}_ppo")
    save_meta = _load_metadata(task, save_path)
    save_meta = mark_train_start(
        save_meta,
        requested_timesteps=total_timesteps,
        source_model_if_resumed=(source_model_path if load_path else None),
    )
    save_metadata(save_path, save_meta)

    env = Monitor(ENV_FACTORIES[task](render_mode=None), LOG_DIR)

    print("[check_env] Validating environment API...")
    check_env(env, warn=True)
    print("[check_env] All checks passed.\n")

    model_path = source_model_path
    if load_path and os.path.exists(model_path + ".zip"):
        print(f"[train] Resuming from '{model_path}.zip'...")
        model = PPO.load(model_path, env=env, tensorboard_log=LOG_DIR)
    else:
        print("[train] Creating a new PPO model...")
        model = PPO(env=env, tensorboard_log=LOG_DIR, **PPO_DEFAULTS)

    print(f"[train] Task='{task}' | {total_timesteps:,} timesteps")
    print(f"[train] Monitor with:  tensorboard --logdir {LOG_DIR}\n")

    model.learn(
        total_timesteps     = total_timesteps,
        reset_num_timesteps = load_path is None,
        progress_bar        = True,
    )

    model.save(save_path)
    print(f"\n[train] Model saved to '{save_path}.zip'")

    save_meta = _load_metadata(task, save_path)
    save_meta = mark_train_end(
        save_meta,
        total_timesteps=int(model.num_timesteps),
        last_run_timesteps=total_timesteps,
    )
    save_metadata(save_path, save_meta)

    env.close()


# ─────────────────────────────────────────────────────────────────────────────
#  play()
# ─────────────────────────────────────────────────────────────────────────────

def play(task: str, model_path: str | None, n_episodes: int) -> None:
    resolved = model_path or os.path.join(MODEL_DIR, f"{task}_ppo")
    if not os.path.exists(resolved + ".zip"):
        print(f"[play] ERROR: '{resolved}.zip' not found.")
        print(f"[play] Train first:  uv run python train.py --task {task} --mode train")
        return

    metadata = _load_metadata(task, resolved)
    metadata = _bootstrap_timesteps_from_checkpoint(resolved, metadata)
    _print_model_info("play", resolved, metadata)

    print(f"[play] Loading '{resolved}.zip'...")
    model = PPO.load(resolved)

    env = ENV_FACTORIES[task](render_mode="human")

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        print(f"\n[play] ── Episode {ep}/{n_episodes} ──")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

            if steps % PLAY_COORD_PRINT_EVERY == 0 or done:
                world_pos = _extract_world_pos(obs, info)
                if world_pos is not None:
                    x, y, z = world_pos
                    print(f"[play]   step={steps:4d}  pos=({x:+.3f}, {y:+.3f}, {z:+.3f})")

        gate_tag = "  GATE PASSED" if info.get("gate_passed") else ""
        print(f"[play]   reward={total_reward:.2f}  steps={steps}{gate_tag}")

    env.close()
    metadata = _load_metadata(task, resolved)
    metadata = set_total_timesteps(metadata, int(model.num_timesteps))
    metadata = mark_play(metadata, episodes=n_episodes)
    save_metadata(resolved, metadata)
    print("\n[play] Done.")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or play a PPO agent on a PyFlyt drone task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=list(ENV_FACTORIES.keys()),
        default="hover",
        help="Which task / environment to use.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "play"],
        default="train",
        help="'train' runs headless training; 'play' opens the GUI.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help="Total training timesteps (train mode only).",
    )
    parser.add_argument(
        "--load",
        default=None,
        metavar="PATH",
        help="Path to a saved model (.zip without extension) to resume from.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run in play mode.",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train(args.task, args.load, args.timesteps)
    else:
        play(args.task, args.load, args.episodes)
