# PyFlyt Drone Sim

PyFlyt drone simulation with environments, goal types, and training/play entry point.

## Project structure

```
PyFlytDroneSim/
├── envs/
│   ├── base_drone_env.py   # thin PyFlyt wrapper
│   ├── hover_env.py        # Stage 1 — single stationary goal
│   ├── waypoint_env.py     # Stage 2 — chained waypoint navigation
│   └── gate_env.py         # Stage 3 — detector wrapper
├── goals/
│   ├── static_point.py     # fixed XYZ target
│   ├── moving_point.py     # randomly-reset / animatable goal
│   └── gate.py             # 
├── hover_test.py           # smoke test (no training)
├── train.py                # Main entry point (tra)
├── pyproject.toml
└── README.md
```

## 1) Setup with uv

```bash
uv python install 3.12
uv python pin 3.12
uv add "gymnasium==0.29.1" "numpy>=1.26,<2" "PyFlyt==0.28.0" "stable-baselines3>=2.3" "tensorboard" "tqdm" "rich"
```

Or sync from the lock file:

```bash
uv sync
```

## 2) Quick smoke test (no training)

```bash
uv run python hover_test.py
```


## 3) Train a policy

Per-episode run duration defaults to **10 seconds**.

Waypoint and gate tasks use PyFlyt flight mode **6** (`vx, vy, yaw_rate, vz`) --> policy commands horizontal velocity plus vertical climb/descent velocity.


| Task | Command |
|---|---|
| Hover at (0,3,2) | `uv run python train.py --task hover --mode train` |
| Navigate to waypoints | `uv run python train.py --task waypoint --mode train` |
| Fly through gates | `uv run python train.py --task gate --mode train` |

Set a longer per-episode duration (for example, 20s):

```bash
uv run python train.py --task hover --mode train --episode-seconds 20
```

**Resume training from a checkpoint:**

```bash
uv run python train.py --task hover --mode train --load models/hover_ppo
```

### Training flow (checkpoint naming)

- Default save target is task-based: `models/<task>_ppo.zip`.
- Running train **without** `--load` creates/overwrites that default checkpoint.
- Running train **with** `--load models/some_other_name` loads that model for initialization,
	but training is still saved to the default task checkpoint (`models/<task>_ppo.zip`).
- This means a non-default loaded model stays persisted, while the updated result is written
	to the default task-named checkpoint.

**Change training budget:**

```bash
uv run python train.py --task hover --mode train --timesteps 500000
```

**Watch training progress (TensorBoard):**

```bash
tensorboard --logdir ./logs/
```

## 4) Play (watch a trained agent)

```bash
uv run python train.py --task hover --mode play
uv run python train.py --task hover --mode play --episodes 10
uv run python train.py --task hover --mode play --episode-seconds 20
uv run python train.py --task hover --mode play --load models/my_best_hover
```

## Model metadata and progress visibility

`train.py` maintains a per-model sidecar metadata file next to each
checkpoint in `models/`:

- `models/<model_name>.zip`         (SB3 checkpoint)
- `models/<model_name>.meta.json`   (metadata + progress)

Before both train and play modes, the script prints a model info summary
including model name, algorithm/network, task/env/reward profile, and
`total_timesteps` trained so far.




## Tasks

### envs/waypoint_env.py

For training a baseline policy with 100k training steps:
```bash
# ── Reward hyper-parameters ───────────────────────────────────────────────────
PROGRESS_COEF = 5.0
PROGRESS_CLIP = 0.20
REACH_RADIUS = 0.35
REACH_BONUS = 1.50
CRASH_PENALTY = 5.0

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
DEFAULT_FLIGHT_DOME_SIZE = 10.0
```

