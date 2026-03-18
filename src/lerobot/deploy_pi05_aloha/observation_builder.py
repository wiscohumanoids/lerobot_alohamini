from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from lerobot.utils.constants import OBS_IMAGES, OBS_PREFIX, OBS_STATE


@dataclass
class ObservationBuilderConfig:
    camera_to_policy_key: dict[str, str]
    state_key: str = OBS_STATE


class ObservationBuilder:
    def __init__(self, cfg: ObservationBuilderConfig):
        self.cfg = cfg
        for key in self.cfg.camera_to_policy_key.values():
            if not key.startswith(OBS_PREFIX):
                raise ValueError(f"Expected policy observation key, got '{key}'")
        if not self.cfg.state_key.startswith(OBS_PREFIX):
            raise ValueError(f"Expected policy state key, got '{self.cfg.state_key}'")

    @classmethod
    def from_camera_names(cls, camera_names: list[str]) -> "ObservationBuilder":
        return cls(
            ObservationBuilderConfig(
                camera_to_policy_key={name: f"{OBS_IMAGES}.{name}" for name in camera_names},
                state_key=OBS_STATE,
            )
        )

    @staticmethod
    def _to_model_image(frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 frame, got shape={frame.shape}")
        image = frame.astype(np.float32) / 255.0
        return np.transpose(image, (2, 0, 1))

    def build(
        self,
        frames: Mapping[str, NDArray[np.uint8]],
        state: np.ndarray,
        task_text: str,
    ) -> dict[str, Any]:
        batch: dict[str, Any] = {self.cfg.state_key: state}
        for camera_name, policy_key in self.cfg.camera_to_policy_key.items():
            if camera_name not in frames:
                raise KeyError(f"Missing frame for camera '{camera_name}'")
            batch[policy_key] = self._to_model_image(frames[camera_name])
        batch["task"] = task_text
        return batch
