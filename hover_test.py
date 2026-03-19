"""
hover_test.py
-------------
Quick smoke test / telemetry viewer — no training, no model.

Sends the drone a zero-velocity command every step and prints the raw
observation so you can inspect sensor values and verify the environment
is working before training.

Run:
    uv run python hover_test.py
"""

import numpy as np

from envs import HoverEnv


def main() -> None:
    env = HoverEnv(render_mode="human")

    try:
        obs, info = env.reset()
        for step in range(1000):
            # Zero command: stay in place (vx=0, vy=0, vz=0, yaw_rate=0)
            action = np.zeros(4, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"step={step:4d}  reward={reward:+.4f}  obs={obs}")

            if terminated or truncated:
                obs, info = env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()