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


# ── Task registry ─────────────────────────────────────────────────────────────
# Maps --task name to a factory that takes render_mode and returns a gym.Env.
# For GateEnv we wrap HoverEnv: change the inner env here to WaypointEnv for
# a harder variant.

def _make_hover(render_mode):
    return HoverEnv(render_mode=render_mode)

def _make_waypoint(render_mode):
    return WaypointEnv(render_mode=render_mode)

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


# ─────────────────────────────────────────────────────────────────────────────
#  train()
# ─────────────────────────────────────────────────────────────────────────────

def train(task: str, load_path: str | None, total_timesteps: int) -> None:
    os.makedirs(LOG_DIR,   exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = Monitor(ENV_FACTORIES[task](render_mode=None), LOG_DIR)

    print("[check_env] Validating environment API...")
    check_env(env, warn=True)
    print("[check_env] All checks passed.\n")

    model_path = load_path or os.path.join(MODEL_DIR, f"{task}_ppo")
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

    save_path = os.path.join(MODEL_DIR, f"{task}_ppo")
    model.save(save_path)
    print(f"\n[train] Model saved to '{save_path}.zip'")
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

        gate_tag = "  GATE PASSED" if info.get("gate_passed") else ""
        print(f"[play]   reward={total_reward:.2f}  steps={steps}{gate_tag}")

    env.close()
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
