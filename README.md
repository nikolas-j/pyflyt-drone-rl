# PyFlyt Drone Sim

Multi-task PyFlyt drone simulation project with clean separation of
environments, goal types, and a single training/play entry point.

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
uv add "gymnasium==0.29.1" "numpy>=1.26,<2" "PyFlyt==0.28.0" "stable-baselines3>=2.3" "rich"
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

## RL Roadmap: Hover → Waypoints → Gates

### Stage 1 — Hover (`envs/hover_env.py`)

Train in `PyFlyt/QuadX-Hover-v3` with shaped reward for holding $(0,0,1)$:

$$r = -\lVert p - p^* \rVert_2 + r_{\text{stay}} - \lambda \lVert a \rVert^2$$

- $r_{\text{stay}} = +0.10$ per step while within 0.20 m of target
- $\lambda = 0.005$ control effort penalty

### Stage 2 — Waypoints (`envs/waypoint_env.py`)

Switch to `PyFlyt/QuadX-Waypoints-v3`.  The goal resets to a new random
position each episode.  `goals/moving_point.py` provides the goal object;
subclass `MovingPoint.tick()` to add animated motion (orbiting, sinusoidal,
etc.).

### Stage 3 — Gates (`envs/gate_env.py`)

`GateEnv` is a `gymnasium.Wrapper` that can wrap *any* inner env.  It
detects when the drone crosses an X-plane gate:

$$(x_{t-1} - x_g)(x_t - x_g) < 0 \quad \text{AND} \quad y,z \in \text{opening bounds}$$

On a valid pass: sparse $+5.0$ reward and episode ends.  Remove the
termination line in `gate_env.py` to chain multiple gates in sequence.