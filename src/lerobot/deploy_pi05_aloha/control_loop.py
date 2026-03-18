from __future__ import annotations

import signal
import time
from dataclasses import dataclass, field
from typing import Sequence

from lerobot.robots import RobotConfig

from .action_adapter import ActionAdapter, ActionAdapterConfig, ActionScale
from .aloha_interface import (
    DEFAULT_LEKIWI_ARM_POS_KEYS,
    DEFAULT_LEKIWI_CAMERA_KEYS,
    DEFAULT_LEKIWI_FULL_ACTION_KEYS,
    DEFAULT_LEKIWI_STATE_KEYS,
    LeKiwiDeploymentRobot,
    Pi05AlohaRobotConfig,
)
from .observation_builder import ObservationBuilder
from .policy_runner import Pi05PolicyRunner, PolicyRunnerConfig


@dataclass
class Pi05AlohaControlLoopConfig:
    robot_config: RobotConfig
    pretrained_path: str
    task_text: str
    state_keys: Sequence[str] = field(default_factory=lambda: list(DEFAULT_LEKIWI_STATE_KEYS))
    camera_keys: Sequence[str] = field(default_factory=lambda: list(DEFAULT_LEKIWI_CAMERA_KEYS))
    action_keys: Sequence[str] = field(default_factory=lambda: list(DEFAULT_LEKIWI_FULL_ACTION_KEYS))
    control_hz: float = 10.0
    device: str = "cuda"
    run_seconds: float | None = None
    calibrate: bool = False
    action_mode: str = "passthrough"
    per_key_scales: dict[str, ActionScale] = field(default_factory=dict)


class Pi05AlohaControlLoop:
    def __init__(self, cfg: Pi05AlohaControlLoopConfig):
        self.cfg = cfg
        self._running = False

        self.robot = LeKiwiDeploymentRobot(
            Pi05AlohaRobotConfig(
                robot_config=cfg.robot_config,
                state_keys=cfg.state_keys,
                camera_keys=cfg.camera_keys,
            )
        )
        self.builder = ObservationBuilder.from_camera_names(list(cfg.camera_keys))
        self.policy = Pi05PolicyRunner(
            PolicyRunnerConfig(pretrained_path=cfg.pretrained_path, device=cfg.device)
        )
        self.adapter = ActionAdapter(
            ActionAdapterConfig(
                action_keys=cfg.action_keys,
                mode=cfg.action_mode,  # type: ignore[arg-type]
                per_key_scales=cfg.per_key_scales,
            )
        )

    def _install_sigint_handler(self) -> None:
        def _handler(signum, frame):  # noqa: ARG001
            self._running = False

        signal.signal(signal.SIGINT, _handler)

    def _hold_position_action(self, observation: dict[str, object]) -> dict[str, float]:
        hold: dict[str, float] = {}
        for key in self.cfg.action_keys:
            value = observation.get(key, 0.0)
            hold[key] = float(value)  # type: ignore[arg-type]
        return hold

    def run(self) -> None:
        self._install_sigint_handler()
        self._running = True
        target_period = 1.0 / self.cfg.control_hz
        start = time.perf_counter()

        self.robot.connect(calibrate=self.cfg.calibrate)
        last_observation: dict[str, object] | None = None

        try:
            while self._running:
                tick = time.perf_counter()
                if self.cfg.run_seconds is not None and tick - start >= self.cfg.run_seconds:
                    break

                observation = self.robot.get_observation()
                last_observation = observation
                frames = self.robot.get_camera_frames(observation)
                state = self.robot.build_state_vector(observation)
                batch = self.builder.build(frames=frames, state=state, task_text=self.cfg.task_text)
                action_vec = self.policy.infer(batch)
                robot_action = self.adapter.to_robot_action(action_vec, observation)
                self.robot.send_action(robot_action)

                elapsed = time.perf_counter() - tick
                if elapsed < target_period:
                    time.sleep(target_period - elapsed)
        finally:
            if last_observation is not None:
                try:
                    self.robot.send_action(self._hold_position_action(last_observation))
                except Exception:
                    pass
            self.robot.disconnect()


DEFAULT_ARM_ONLY_SCALES: dict[str, ActionScale] = {
    key: (ActionScale(low=0.0, high=100.0) if "gripper" in key else ActionScale(low=-100.0, high=100.0))
    for key in DEFAULT_LEKIWI_ARM_POS_KEYS
}

DEFAULT_FULL_ACTION_SCALES: dict[str, ActionScale] = {
    **DEFAULT_ARM_ONLY_SCALES,
    "x.vel": ActionScale(low=-0.35, high=0.35),
    "y.vel": ActionScale(low=-0.35, high=0.35),
    "theta.vel": ActionScale(low=-90.0, high=90.0),
    "lift_axis.height_mm": ActionScale(low=0.0, high=300.0),
}
