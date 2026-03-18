from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from lerobot.processor import RobotObservation

ActionMode = Literal["passthrough", "delta"]


@dataclass
class ActionScale:
    low: float
    high: float
    delta_limit: float | None = None


@dataclass
class ActionAdapterConfig:
    action_keys: Sequence[str]
    mode: ActionMode = "passthrough"
    # Native LeKiwi units, not radians.
    # With use_degrees=False, arm joints are typically in [-100, 100], grippers in [0, 100],
    # base velocities are in metric/deg-per-sec units, and the lift is in mm.
    per_key_scales: dict[str, ActionScale] = field(default_factory=dict)


class ActionAdapter:
    def __init__(self, cfg: ActionAdapterConfig):
        self.cfg = cfg

    def _current_value(self, observation: RobotObservation, key: str) -> float:
        if key not in observation:
            raise KeyError(f"Action key '{key}' missing from current observation")
        return float(np.asarray(observation[key], dtype=np.float32).reshape(-1)[0])

    def to_robot_action(self, model_action: np.ndarray, observation: RobotObservation) -> dict[str, float]:
        if model_action.ndim != 1:
            raise ValueError(f"Expected 1D model action, got shape={model_action.shape}")
        if model_action.shape[0] < len(self.cfg.action_keys):
            raise ValueError(
                f"Model output dim {model_action.shape[0]} is smaller than configured action_keys={len(self.cfg.action_keys)}"
            )

        robot_action: dict[str, float] = {}
        for idx, key in enumerate(self.cfg.action_keys):
            value = float(model_action[idx])
            scale = self.cfg.per_key_scales.get(key)
            if self.cfg.mode == "delta":
                current = self._current_value(observation, key)
                delta = value
                if scale and scale.delta_limit is not None:
                    delta = float(np.clip(delta, -scale.delta_limit, scale.delta_limit))
                value = current + delta

            if scale is not None:
                value = float(np.clip(value, scale.low, scale.high))
            robot_action[key] = value
        return robot_action
