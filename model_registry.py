from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _meta_path(model_path: str) -> str:
    return f"{model_path}.meta.json"


def _default_metadata(
    model_path: str,
    *,
    task: str,
    algorithm: str,
    network_description: str,
    env_id: str,
    flight_mode: int,
    reward_profile: str,
) -> dict[str, Any]:
    now = _utc_now_iso()
    return {
        "model_name": os.path.basename(model_path),
        "model_path": model_path,
        "algorithm": algorithm,
        "network_description": network_description,
        "task": task,
        "env_id": env_id,
        "flight_mode": flight_mode,
        "reward_profile": reward_profile,
        "created_at": now,
        "updated_at": now,
        "total_timesteps": 0,
        "last_run_timesteps": 0,
        "run_count": 0,
        "last_mode": None,
        "source_model_if_resumed": None,
        "last_played_at": None,
        "last_trained_at": None,
        "last_requested_timesteps": 0,
    }


def _merge_missing_fields(target: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    for key, value in defaults.items():
        if key not in target:
            target[key] = value
    return target


def get_or_create_metadata(
    model_path: str,
    *,
    task: str,
    algorithm: str,
    network_description: str,
    env_id: str,
    flight_mode: int,
    reward_profile: str,
) -> dict[str, Any]:
    meta_file = _meta_path(model_path)
    defaults = _default_metadata(
        model_path,
        task=task,
        algorithm=algorithm,
        network_description=network_description,
        env_id=env_id,
        flight_mode=flight_mode,
        reward_profile=reward_profile,
    )

    if os.path.exists(meta_file):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            metadata = defaults
        metadata = _merge_missing_fields(metadata, defaults)
    else:
        metadata = defaults

    metadata["model_name"] = os.path.basename(model_path)
    metadata["model_path"] = model_path
    metadata["task"] = task
    metadata["algorithm"] = algorithm
    metadata["network_description"] = network_description
    metadata["env_id"] = env_id
    metadata["flight_mode"] = flight_mode
    metadata["reward_profile"] = reward_profile
    metadata["updated_at"] = _utc_now_iso()
    return metadata


def save_metadata(model_path: str, metadata: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    metadata = dict(metadata)
    metadata["updated_at"] = _utc_now_iso()
    with open(_meta_path(model_path), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def set_total_timesteps(metadata: dict[str, Any], total_timesteps: int) -> dict[str, Any]:
    metadata["total_timesteps"] = int(max(0, total_timesteps))
    metadata["updated_at"] = _utc_now_iso()
    return metadata


def mark_train_start(
    metadata: dict[str, Any],
    *,
    requested_timesteps: int,
    source_model_if_resumed: str | None,
) -> dict[str, Any]:
    metadata["last_mode"] = "train"
    metadata["last_requested_timesteps"] = int(max(0, requested_timesteps))
    metadata["source_model_if_resumed"] = source_model_if_resumed
    metadata["updated_at"] = _utc_now_iso()
    return metadata


def mark_train_end(
    metadata: dict[str, Any],
    *,
    total_timesteps: int,
    last_run_timesteps: int,
) -> dict[str, Any]:
    metadata["last_mode"] = "train"
    metadata["total_timesteps"] = int(max(0, total_timesteps))
    metadata["last_run_timesteps"] = int(max(0, last_run_timesteps))
    metadata["run_count"] = int(metadata.get("run_count", 0)) + 1
    metadata["last_trained_at"] = _utc_now_iso()
    metadata["updated_at"] = _utc_now_iso()
    return metadata


def mark_play(metadata: dict[str, Any], *, episodes: int) -> dict[str, Any]:
    metadata["last_mode"] = "play"
    metadata["last_played_episodes"] = int(max(0, episodes))
    metadata["last_played_at"] = _utc_now_iso()
    metadata["updated_at"] = _utc_now_iso()
    return metadata


def format_model_info(metadata: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"[model] name={metadata.get('model_name', 'unknown')}",
            f"[model] algorithm={metadata.get('algorithm', 'unknown')}  network={metadata.get('network_description', 'unknown')}",
            f"[model] task={metadata.get('task', 'unknown')}  env={metadata.get('env_id', 'unknown')}  reward={metadata.get('reward_profile', 'unknown')}",
            f"[model] total_timesteps={int(metadata.get('total_timesteps', 0)):,}  runs={int(metadata.get('run_count', 0))}",
        ]
    )
