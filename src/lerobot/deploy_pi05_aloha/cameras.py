from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from lerobot.processor import RobotObservation


DEFAULT_LEKIWI_CAMERA_KEYS: tuple[str, ...] = (
    "head_top",
    "head_back",
    "head_front",
    "wrist_left",
    "wrist_right",
)


@dataclass(frozen=True)
class CameraMapping:
    """Map a robot observation image key to the policy observation key.

    Example:
        CameraMapping(robot_key="head_front", policy_key="observation.images.head_front")
    """

    robot_key: str
    policy_key: str


def _ensure_uint8_rgb(image: NDArray[np.generic]) -> NDArray[np.uint8]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape={image.shape}")
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image


def extract_camera_frames(
    observation: RobotObservation,
    camera_keys: Iterable[str],
) -> dict[str, NDArray[np.uint8]]:
    """Extract camera frames from a LeRobot robot observation.

    This repo's AlohaMini/LeKiwi robots already place camera frames directly in the
    observation dict under keys such as ``head_front`` and ``wrist_left``.
    We therefore do *not* create a second camera stack for deployment.
    """

    frames: dict[str, NDArray[np.uint8]] = {}
    for key in camera_keys:
        if key not in observation:
            raise KeyError(f"Camera key '{key}' missing from robot observation")
        frames[key] = _ensure_uint8_rgb(np.asarray(observation[key]))
    return frames
