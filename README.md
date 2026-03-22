# PyFlyt Drone Sim

PyFlyt drone simulation with environments, goal types, and training/play entry point.

## Project structure

```
PyFlytDroneSim/
├── envs/
│   ├── base_drone_env.py   # thin PyFlyt wrapper (flight_mode=7, action coercion)
│   ├── hover_env.py        # Stage 1 — hold position at (0,0,1)
│   ├── waypoint_env.py     # Stage 2 — navigate to random goals
│   └── gate_env.py         # Stage 3 — detector wrapper (applies to any env)
├── goals/
│   ├── static_point.py     # fixed XYZ target
│   ├── moving_point.py     # randomly-reset / animatable goal
│   └── gate.py             # X-plane gate with Y/Z opening bounds
├── hover_test.py           # zero-command smoke test (no training)
├── train.py                # CLI: --task hover|waypoint|gate  --mode train|play
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

Opens the PyBullet GUI, sends zero velocity commands, and prints telemetry:

```bash
uv run python hover_test.py
```

### Explicit drone spawn position

You can now set the drone spawn point explicitly for hover/waypoint envs:

```python
from envs import HoverEnv

# Fixed spawn for all episodes
env = HoverEnv(start_pos=[1.0, 0.0, 1.8])

# Or override per reset
obs, info = env.reset(options={"start_pos": [2.5, -1.0, 2.0]})
```

`start_pos` accepts shape `(3,)` or `(1, 3)` and maps to PyFlyt's internal `start_pos`.

## 3) Train an agent

| Task | Command |
|---|---|
| Hover at (0,0,1) | `uv run python train.py --task hover --mode train` |
| Navigate to waypoints | `uv run python train.py --task waypoint --mode train` |
| Fly through a gate | `uv run python train.py --task gate --mode train` |

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
uv run python train.py --task hover --mode play --load models/my_best_hover
```

A PyBullet GUI window opens and the agent flies using its learned policy.

## Model metadata and progress visibility

`train.py` maintains a per-model sidecar metadata file next to each
checkpoint in `models/`:

- `models/<model_name>.zip`         (SB3 checkpoint)
- `models/<model_name>.meta.json`   (metadata + progress)

Before both train and play modes, the script prints a model info summary
including model name, algorithm/network, task/env/reward profile, and
`total_timesteps` trained so far.
