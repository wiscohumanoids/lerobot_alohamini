from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from lerobot.processor import RobotObservation
from lerobot.robots import Robot, RobotConfig, make_robot_from_config
from lerobot.utils.constants import OBS_STATE

from .cameras import DEFAULT_LEKIWI_CAMERA_KEYS, extract_camera_frames


DEFAULT_LEKIWI_ARM_POS_KEYS: tuple[str, ...] = (
    "arm_left_shoulder_pan.pos",
    "arm_left_shoulder_lift.pos",
    "arm_left_elbow_flex.pos",
    "arm_left_wrist_flex.pos",
    "arm_left_wrist_roll.pos",
    "arm_left_gripper.pos",
    "arm_right_shoulder_pan.pos",
    "arm_right_shoulder_lift.pos",
    "arm_right_elbow_flex.pos",
    "arm_right_wrist_flex.pos",
    "arm_right_wrist_roll.pos",
    "arm_right_gripper.pos",
)

DEFAULT_LEKIWI_BASE_KEYS: tuple[str, ...] = (
    "x.vel",
    "y.vel",
    "theta.vel",
)

DEFAULT_LEKIWI_LIFT_KEYS: tuple[str, ...] = ("lift_axis.height_mm",)

DEFAULT_LEKIWI_STATE_KEYS: tuple[str, ...] = (
    *DEFAULT_LEKIWI_ARM_POS_KEYS,
    *DEFAULT_LEKIWI_BASE_KEYS,
    *DEFAULT_LEKIWI_LIFT_KEYS,
)

# The output action order should match the training dataset exactly.
# In this repo the robot's action_features mirror the state ordering.
DEFAULT_LEKIWI_FULL_ACTION_KEYS: tuple[str, ...] = DEFAULT_LEKIWI_STATE_KEYS
DEFAULT_LEKIWI_ARM_ONLY_ACTION_KEYS: tuple[str, ...] = DEFAULT_LEKIWI_ARM_POS_KEYS


@dataclass
class Pi05AlohaRobotConfig:
    robot_config: RobotConfig
    state_keys: Sequence[str] = field(default_factory=lambda: list(DEFAULT_LEKIWI_STATE_KEYS))
    camera_keys: Sequence[str] = field(default_factory=lambda: list(DEFAULT_LEKIWI_CAMERA_KEYS))


class LeKiwiDeploymentRobot:
    """Thin deployment wrapper around an existing LeRobot Robot.

    Important difference from Cursor's proposal:
    - we reuse the existing robot observation directly
    - we preserve the repo's real action/state key names
    - we do not force an arm-only ``send_position_action`` API
    """

    def __init__(self, cfg: Pi05AlohaRobotConfig):
        self.cfg = cfg
        self._robot: Robot | None = None

    @property
    def robot(self) -> Robot:
        if self._robot is None:
            raise RuntimeError("Robot not connected")
        return self._robot

    def connect(self, calibrate: bool = True) -> None:
        self._robot = make_robot_from_config(self.cfg.robot_config)
        self.robot.connect(calibrate=calibrate)
        self.robot.configure()

    def disconnect(self) -> None:
        if self._robot is None:
            return
        try:
            self._robot.disconnect()
        finally:
            self._robot = None

    def get_observation(self) -> RobotObservation:
        return self.robot.get_observation()

    def get_camera_frames(self, observation: RobotObservation) -> dict[str, np.ndarray]:
        return extract_camera_frames(observation, self.cfg.camera_keys)

    def build_state_vector(self, observation: RobotObservation) -> np.ndarray:
        if OBS_STATE in observation:
            state = np.asarray(observation[OBS_STATE], dtype=np.float32).reshape(-1)
            if state.shape[0] != len(self.cfg.state_keys):
                raise ValueError(
                    f"Robot-provided state has dim {state.shape[0]} but state_keys has {len(self.cfg.state_keys)} entries"
                )
            return state

        values: list[float] = []
        for key in self.cfg.state_keys:
            if key not in observation:
                raise KeyError(f"Missing state key '{key}' in robot observation")
            arr = np.asarray(observation[key], dtype=np.float32).reshape(-1)
            if arr.size != 1:
                raise ValueError(f"State key '{key}' should be scalar-like, got shape {arr.shape}")
            values.append(float(arr[0]))
        return np.asarray(values, dtype=np.float32)

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        sent = self.robot.send_action(action)
        return {k: float(v) for k, v in sent.items() if isinstance(v, (float, int, np.floating, np.integer))}
