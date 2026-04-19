#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import logging
import os
import time
from functools import cached_property
from itertools import chain
from typing import Any, Literal
import sys

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_lekiwi import LeKiwiConfig

logger = logging.getLogger(__name__)

from .lift_axis import LiftAxis, LiftAxisConfig

CalibrationArm = Literal["left", "right", "both"]


class LeKiwi(Robot):
    """
    The robot includes a three omniwheel mobile base and a remote follower arm.
    The leader arm is connected locally (on the laptop) and its joint positions are recorded and then
    forwarded to the remote follower arm (after applying a safety clamp).
    In parallel, keyboard teleoperation is used to generate raw velocity commands for the wheels.
    """

    config_class = LeKiwiConfig
    name = "lekiwi"

    def __init__(self, config: LeKiwiConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        if config.arm_profile == "am-arm-6dof":
            left_arm_motors_cfg = {
                "arm_left_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "arm_left_shoulder_lift": Motor(2, "sts3095", norm_mode_body),
                "arm_left_elbow_flex": Motor(3, "sts3095", norm_mode_body),
                "arm_left_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "arm_left_wrist_yaw": Motor(5, "sts3215", norm_mode_body),
                "arm_left_wrist_roll": Motor(6, "sts3215", MotorNormMode.RANGE_M100_100),
                "arm_left_gripper": Motor(7, "sts3215", MotorNormMode.RANGE_0_100),
            }
            right_arm_motors_cfg = {
                "arm_right_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "arm_right_shoulder_lift": Motor(2, "sts3095", norm_mode_body),
                "arm_right_elbow_flex": Motor(3, "sts3095", norm_mode_body),
                "arm_right_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "arm_right_wrist_yaw": Motor(5, "sts3215", norm_mode_body),
                "arm_right_wrist_roll": Motor(6, "sts3215", MotorNormMode.RANGE_M100_100),
                "arm_right_gripper": Motor(7, "sts3215", MotorNormMode.RANGE_0_100),
            }
            self._left_arm_state_keys = (
                "arm_left_shoulder_pan.pos",
                "arm_left_shoulder_lift.pos",
                "arm_left_elbow_flex.pos",
                "arm_left_wrist_flex.pos",
                "arm_left_wrist_yaw.pos",
                "arm_left_wrist_roll.pos",
                "arm_left_gripper.pos",
            )
            self._right_arm_state_keys = (
                "arm_right_shoulder_pan.pos",
                "arm_right_shoulder_lift.pos",
                "arm_right_elbow_flex.pos",
                "arm_right_wrist_flex.pos",
                "arm_right_wrist_yaw.pos",
                "arm_right_wrist_roll.pos",
                "arm_right_gripper.pos",
            )
        elif config.arm_profile == "so-arm-5dof":
            left_arm_motors_cfg = {
                "arm_left_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "arm_left_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "arm_left_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "arm_left_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "arm_left_wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
                "arm_left_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            }
            right_arm_motors_cfg = {
                "arm_right_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "arm_right_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "arm_right_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "arm_right_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "arm_right_wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
                "arm_right_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            }
            self._left_arm_state_keys = (
                "arm_left_shoulder_pan.pos",
                "arm_left_shoulder_lift.pos",
                "arm_left_elbow_flex.pos",
                "arm_left_wrist_flex.pos",
                "arm_left_wrist_roll.pos",
                "arm_left_gripper.pos",
            )
            self._right_arm_state_keys = (
                "arm_right_shoulder_pan.pos",
                "arm_right_shoulder_lift.pos",
                "arm_right_elbow_flex.pos",
                "arm_right_wrist_flex.pos",
                "arm_right_wrist_roll.pos",
                "arm_right_gripper.pos",
            )
        else:
            raise ValueError(
                f"Unknown arm_profile '{config.arm_profile}'. Expected 'so-arm-5dof' or 'am-arm-6dof'."
            )

        self.left_bus = FeetechMotorsBus(
            port=self.config.left_port,
            motors={
                **left_arm_motors_cfg,
                # lift axis
                "lift_axis": Motor(11, "sts3215", MotorNormMode.DEGREES),
            },
            calibration=self.calibration,
        )

        self.right_bus = FeetechMotorsBus(
            port=self.config.right_port,
            motors={
                **right_arm_motors_cfg,
                #"lift_axis": Motor(12, "sts3215", MotorNormMode.DEGREES),
            },
            calibration=self.calibration,
        )

        self.base_bus = FeetechMotorsBus(
            port=self.config.base_port,
            motors={
                "base_left_wheel": Motor(8, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_back_wheel": Motor(9, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_right_wheel": Motor(10, "sts3215", MotorNormMode.RANGE_M100_100),
            },
            calibration=self.calibration,
        )


        self.left_arm_motors  = [m for m in self.left_bus.motors if m.startswith("arm_left_")]
        self.base_motors      = [m for m in self.base_bus.motors if m.startswith("base_")]
        #self.left_arm_motors  = [m for m in self.left_bus.motors        if m.startswith("right_arm_")]

        self.right_arm_motors = [m for m in (self.right_bus.motors if self.right_bus else []) if m.startswith("arm_right_")]

        # self.arm_motors = [motor for motor in self.left_bus.motors if motor.startswith("arm")]
        # self.base_motors = [motor for motor in self.left_bus.motors if motor.startswith("base")]

        self.cameras = make_cameras_from_configs(config.cameras)


        self.lift = LiftAxis(
            LiftAxisConfig(),        
            bus_left=self.left_bus,
            bus_right=self.right_bus,
        )
        # Overcurrent debounce: require N consecutive over-limit reads
        self._overcurrent_count: dict[str, int] = {}
        self._overcurrent_trip_n = 20
        self._last_currents_log_t = 0.0
        self._camera_read_failures: dict[str, int] = {}


    @property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            (
                *self._left_arm_state_keys,
                *self._right_arm_state_keys,
                "x.vel",
                "y.vel",
                "theta.vel",
                "lift_axis.height_mm",   # new
                #"lift_axis.vel",         # new (optional, for debugging)
            ),
            float,
        )

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    # @property
    # def is_connected(self) -> bool:
    #     return self.left_bus.is_connected and all(cam.is_connected for cam in self.cameras.values())
    
    @property
    def is_connected(self) -> bool:
        buses_ok = self.left_bus.is_connected and (self.right_bus.is_connected if self.right_bus else True)
        return buses_ok


    @check_if_already_connected
    def connect(self, calibrate: bool = True, calibrate_arm: CalibrationArm = "both") -> None:
        self.left_bus.connect()
        self.right_bus.connect()
        self.base_bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate(calibrate_arm=calibrate_arm)

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

        self.lift.home()
        print("Lift axis homed to 0mm.")

        

    @property
    def is_calibrated(self) -> bool:
        return self.left_bus.is_calibrated

    def _write_existing_calibration_to_buses(self) -> None:
        logger.info("Writing existing calibration to both buses (trim per-bus caches)")

        calib_left = {k: v for k, v in self.calibration.items() if k in self.left_bus.motors}
        self.left_bus.write_calibration(calib_left, cache=False)
        self.left_bus.calibration = calib_left

        calib_right = {k: v for k, v in self.calibration.items() if k in self.right_bus.motors}
        self.right_bus.write_calibration(calib_right, cache=False)
        self.right_bus.calibration = calib_right

        calib_base = {k: v for k, v in self.calibration.items() if k in self.base_bus.motors}
        self.base_bus.write_calibration(calib_base, cache=False)
        self.base_bus.calibration = calib_base

    def _calibrate_arm_bus(
        self,
        bus: FeetechMotorsBus,
        arm_motors: list[str],
        arm_label: str,
        full_turn_motor: str,
    ) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        if not arm_motors:
            raise RuntimeError(f"{arm_label}_arm_motors is empty; expected configured arm motor names")

        bus.disable_torque(arm_motors)
        for name in arm_motors:
            bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

        full_turn_motors = [full_turn_motor] if full_turn_motor in arm_motors else []
        unknown_range_motors = [m for m in arm_motors if m not in full_turn_motors]

        input(f"Move {arm_label.upper()} arm to the middle of its range of motion, then press ENTER...")
        homing = bus.set_half_turn_homings(unknown_range_motors)
        for motor_name in full_turn_motors:
            homing[motor_name] = 0

        print(
            f"Move {arm_label.upper()} arm joints sequentially through full ROM "
            f"(except '{full_turn_motor}'). Press ENTER to stop..."
        )
        range_mins, range_maxs = bus.record_ranges_of_motion(unknown_range_motors)
        for motor_name in full_turn_motors:
            range_mins[motor_name] = 0
            range_maxs[motor_name] = 4095

        return homing, range_mins, range_maxs

    def _build_motor_calibration(
        self,
        bus: FeetechMotorsBus,
        homing_offsets: dict[str, int],
        range_mins: dict[str, int],
        range_maxs: dict[str, int],
    ) -> dict[str, MotorCalibration]:
        calibration: dict[str, MotorCalibration] = {}
        for name, motor in bus.motors.items():
            calibration[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets.get(name, 0),
                range_min=range_mins.get(name, 0),
                range_max=range_maxs.get(name, 4095),
            )
        return calibration

    def _build_base_calibration(self) -> dict[str, MotorCalibration]:
        if not self.base_motors:
            raise RuntimeError("base_motors is empty; expected names starting with 'base_'")

        self.base_bus.disable_torque(self.base_motors)
        for name in self.base_motors:
            self.base_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

        homing_offsets = {wheel: 0 for wheel in self.base_motors}
        range_mins = {motor_name: 0 for motor_name in self.base_motors}
        range_maxs = {motor_name: 4095 for motor_name in self.base_motors}
        return self._build_motor_calibration(self.base_bus, homing_offsets, range_mins, range_maxs)

    def _require_preserved_calibration(self, calibrate_arm: CalibrationArm) -> None:
        if calibrate_arm == "both":
            return
        if not self.calibration:
            raise RuntimeError(
                f"Single-arm calibration for '{calibrate_arm}' requires an existing calibration file "
                "so the untouched arm and base entries can be preserved."
            )

        if calibrate_arm == "left":
            required_names = [*self.right_bus.motors.keys(), *self.base_bus.motors.keys()]
        else:
            required_names = [*self.left_bus.motors.keys(), *self.base_bus.motors.keys()]

        missing_names = [name for name in required_names if name not in self.calibration]
        if missing_names:
            raise RuntimeError(
                "Existing calibration is missing entries required for single-arm calibration: "
                + ", ".join(missing_names)
            )

    def calibrate(self, calibrate_arm: CalibrationArm = "both") -> None:
        """
        Dual-arm calibration (left arm + chassis on self.left_bus, right arm on self.right_bus):
        - Left arm: position mode → half-turn homing → collect ROM
        - Chassis: no homing; ROM fixed to 0–4095
        - Right arm (if present): position mode → half-turn homing → collect ROM
        - Merge into a single self.calibration, split by bus, write back to both buses, and save
        """
        # If a calibration file already exists: load it and write back, filtering for each bus separately
        if self.calibration:
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, "
                f"or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                self._write_existing_calibration_to_buses()
                return

        self._require_preserved_calibration(calibrate_arm)

        logger.info(f"\nRunning calibration of {self} ({calibrate_arm} selection)")
        updated_calibration = dict(self.calibration)

        if calibrate_arm in ("both", "left"):
            left_homing, l_mins, l_maxs = self._calibrate_arm_bus(
                self.left_bus,
                self.left_arm_motors,
                "left",
                "arm_left_wrist_roll",
            )
            updated_calibration.update(self._build_motor_calibration(self.left_bus, left_homing, l_mins, l_maxs))

        if calibrate_arm in ("both", "right"):
            right_homing, r_mins, r_maxs = self._calibrate_arm_bus(
                self.right_bus,
                self.right_arm_motors,
                "right",
                "arm_right_wrist_roll",
            )
            updated_calibration.update(self._build_motor_calibration(self.right_bus, right_homing, r_mins, r_maxs))

        if calibrate_arm == "both":
            updated_calibration.update(self._build_base_calibration())

        self.calibration = updated_calibration
        self._write_existing_calibration_to_buses()

        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)


    def _apply_fresh_wrist_roll_homing(self) -> None:
        """Compute and apply a homing offset for both wrist_roll motors from their current positions.

        Unlike joints with mechanical stops, wrist_roll is a continuous-rotation joint
        whose encoder angle is arbitrary at startup.  A saved homing offset becomes stale
        the moment the wrist is moved while unpowered, so we re-home it every connect.
        """
        wrist_motors = {
            self.left_bus: "arm_left_wrist_roll",
            self.right_bus: "arm_right_wrist_roll",
        }
        for bus, motor in wrist_motors.items():
            if motor not in bus.motors:
                continue
            bus.write("Homing_Offset", motor, 0)
            pos = bus.read("Present_Position", motor, normalize=False)
            offsets = bus._get_half_turn_homings({motor: pos})
            bus.write("Homing_Offset", motor, offsets[motor])
            if bus.calibration and motor in bus.calibration:
                old = bus.calibration[motor]
                bus.calibration[motor] = MotorCalibration(
                    id=old.id,
                    drive_mode=old.drive_mode,
                    homing_offset=offsets[motor],
                    range_min=old.range_min,
                    range_max=old.range_max,
                )
            logger.info(f"Fresh {motor} homing applied: offset={offsets[motor]}")

    def configure(self):
        # Set-up arm actuators (position mode)
        # We assume that at connection time, arm is in a rest position,
        # and torque can be safely disabled to run calibration.
        self.left_bus.disable_torque()
        self.left_bus.configure_motors()
        for name in self.left_arm_motors:
            self.left_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            self.left_bus.write("P_Coefficient", name, 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            self.left_bus.write("I_Coefficient", name, 0)
            self.left_bus.write("D_Coefficient", name, 32)
            if name.endswith("_gripper"):
                self.left_bus.write("Torque_Limit", name, self.config.gripper_torque_limit)
                logger.info(f"Gripper '{name}' torque limited to {self.config.gripper_torque_limit}/1023")

        for name in self.base_motors:
            self.base_bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value)

        #self.left_bus.enable_torque()

        self.right_bus.disable_torque()
        self.right_bus.configure_motors()
        for name in self.right_arm_motors:
            self.right_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            self.right_bus.write("P_Coefficient", name, 16)
            self.right_bus.write("I_Coefficient", name, 0)
            self.right_bus.write("D_Coefficient", name, 32)
            if name.endswith("_gripper"):
                self.right_bus.write("Torque_Limit", name, self.config.gripper_torque_limit)
                logger.info(f"Gripper '{name}' torque limited to {self.config.gripper_torque_limit}/1023")
        #self.right_bus.enable_torque()

        #self.lift.configure()




    def setup_motors(self) -> None:
        for motor in chain(reversed(self.arm_motors)):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.left_bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.left_bus.motors[motor].id}")
        for motor in chain(reversed(self.base_motors)):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.base_bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.base_bus.motors[motor].id}")


    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = degps * steps_per_deg
        speed_int = int(round(speed_in_steps))
        # Cap the value to fit within signed 16-bit range (-32768 to 32767)
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF  # 32767 -> maximum positive value
        elif speed_int < -0x8000:
            speed_int = -0x8000  # -32768 -> minimum negative value
        return speed_int

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        steps_per_deg = 4096.0 / 360.0
        magnitude = raw_speed
        degps = magnitude / steps_per_deg
        return degps

    def _body_to_wheel_raw(
        self,
        x: float,
        y: float,
        theta: float,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
        max_raw: int = 3000,
    ) -> dict:
        """
        Convert desired body-frame velocities into wheel raw commands.

        Parameters:
          x_cmd      : Linear velocity in x (m/s).
          y_cmd      : Linear velocity in y (m/s).
          theta_cmd  : Rotational velocity (deg/s).
          wheel_radius: Radius of each wheel (meters).
          base_radius : Distance from the center of rotation to each wheel (meters).
          max_raw    : Maximum allowed raw command (ticks) per wheel.

        Returns:
          A dictionary with wheel raw commands:
             {"base_left_wheel": value, "base_back_wheel": value, "base_right_wheel": value}.

        Notes:
          - Internally, the method converts theta_cmd to rad/s for the kinematics.
          - The raw command is computed from the wheels angular speed in deg/s
            using _degps_to_raw(). If any command exceeds max_raw, all commands
            are scaled down proportionally.
        """
        # Convert rotational velocity from deg/s to rad/s.
        theta_rad = theta * (np.pi / 180.0)
        # Create the body velocity vector [x, y, theta_rad].
        velocity_vector = np.array([-x, -y, theta_rad])

        # Define the wheel mounting angles with a -90° offset.
        angles = np.radians(np.array([240, 0, 120]) - 90)
        # Build the kinematic matrix: each row maps body velocities to a wheel’s linear speed.
        # The third column (base_radius) accounts for the effect of rotation.
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Compute each wheel’s linear speed (m/s) and then its angular speed (rad/s).
        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius

        # Convert wheel angular speeds from rad/s to deg/s.
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

        # Scaling
        steps_per_deg = 4096.0 / 360.0
        raw_floats = [abs(degps) * steps_per_deg for degps in wheel_degps]
        max_raw_computed = max(raw_floats)
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        # Convert each wheel’s angular speed (deg/s) to a raw integer.
        wheel_raw = [self._degps_to_raw(deg) for deg in wheel_degps]

        return {
            "base_left_wheel": wheel_raw[0],
            "base_back_wheel": wheel_raw[1],
            "base_right_wheel": wheel_raw[2],
        }

    def _wheel_raw_to_body(
        self,
        left_wheel_speed,
        back_wheel_speed,
        right_wheel_speed,
        wheel_radius: float = 0.05,
        base_radius: float = 0.125,
    ) -> dict[str, Any]:
        """
        Convert wheel raw command feedback back into body-frame velocities.

        Parameters:
          wheel_raw   : Vector with raw wheel commands ("base_left_wheel", "base_back_wheel", "base_right_wheel").
          wheel_radius: Radius of each wheel (meters).
          base_radius : Distance from the robot center to each wheel (meters).

        Returns:
          A dict (x.vel, y.vel, theta.vel) all in m/s
        """

        # Convert each raw command back to an angular speed in deg/s.
        wheel_degps = np.array(
            [
                self._raw_to_degps(left_wheel_speed),
                self._raw_to_degps(back_wheel_speed),
                self._raw_to_degps(right_wheel_speed),
            ]
        )

        # Convert from deg/s to rad/s.
        wheel_radps = wheel_degps * (np.pi / 180.0)
        # Compute each wheel’s linear speed (m/s) from its angular speed.
        wheel_linear_speeds = wheel_radps * wheel_radius

        # Define the wheel mounting angles with a -90° offset.
        angles = np.radians(np.array([240, 0, 120]) - 90)
        m = np.array([[np.cos(a), np.sin(a), base_radius] for a in angles])

        # Solve the inverse kinematics: body_velocity = M⁻¹ · wheel_linear_speeds.
        m_inv = np.linalg.inv(m)
        velocity_vector = m_inv.dot(wheel_linear_speeds)
        x, y, theta_rad = velocity_vector
        
        theta = theta_rad * (180.0 / np.pi)
        return {
            "x.vel": x,
            "y.vel": y,
            "theta.vel": theta,
        }  # m/s and deg/s
    
    def _raw_to_ma(raw):
        try:
            return float(raw) * 6.5
        except Exception:
            return 0.0

    def _read_camera_observation(self, cam_key: str, cam: Any) -> np.ndarray:
        """Read camera frame, tolerating brief transient failures."""
        try:
            frame = cam.async_read()
            self._camera_read_failures[cam_key] = 0
            return frame
        except Exception as e:
            fail_count = self._camera_read_failures.get(cam_key, 0) + 1
            self._camera_read_failures[cam_key] = fail_count

            # Try using the most recent buffered frame for brief glitches.
            try:
                return cam.read_latest(max_age_ms=2000)
            except Exception:
                pass

            if fail_count >= 10:
                raise RuntimeError(
                    f"Camera '{cam_key}' failed {fail_count} consecutive reads. "
                    f"Last error: {e}"
                ) from e

            logger.warning(f"{self} camera '{cam_key}' read failed {fail_count}/10 ({e})")
            raise

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        # Read actuators position for arm and vel for base
        start = time.perf_counter()
        # arm_pos = self.left_bus.sync_read("Present_Position", self.arm_motors)

        #print(f"Left arm motors: {self.left_arm_motors}, Right arm motors: {self.right_arm_motors}")  # debug
        left_pos = self.left_bus.sync_read("Present_Position", self.left_arm_motors)   # left_arm_*


        base_wheel_vel = self.base_bus.sync_read("Present_Velocity", self.base_motors)

        base_vel = self._wheel_raw_to_body(
            base_wheel_vel["base_left_wheel"],
            base_wheel_vel["base_back_wheel"],
            base_wheel_vel["base_right_wheel"],
        )

        right_pos = self.right_bus.sync_read("Present_Position", self.right_arm_motors)  # right_arm_*


        left_arm_state = {f"{k}.pos": v for k, v in left_pos.items()}
        right_arm_state = {f"{k}.pos": v for k, v in right_pos.items()}

        obs_dict = {**left_arm_state, **right_arm_state,**base_vel}
        self.lift.contribute_observation(obs_dict)
        #print(f"Observation dict so far: {obs_dict}")  # debug

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # currents protection
        self.read_and_check_currents(limit_ma=2000, print_currents=True)

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = self._read_camera_observation(cam_key, cam)
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command AlohaMini to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            np.ndarray: the action sent to the motors, potentially clipped.
        """
        # arm_goal_pos = {k: v for k, v in action.items() if k.endswith(".pos")}
        left_pos  = {k: v for k, v in action.items() if k.endswith(".pos") and k.startswith("arm_left_")}
        right_pos = {k: v for k, v in action.items() if k.endswith(".pos") and k.startswith("arm_right_")}


        base_goal_vel = {k: v for k, v in action.items() if k.endswith(".vel")}

        base_wheel_goal_vel = self._body_to_wheel_raw(
            base_goal_vel["x.vel"], base_goal_vel["y.vel"], base_goal_vel["theta.vel"]
        )

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        # if self.config.max_relative_target is not None:
        #     present_pos = self.left_bus.sync_read("Present_Position", self.arm_motors)
        #     goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in arm_goal_pos.items()}
        #     arm_safe_goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)
        #     arm_goal_pos = arm_safe_goal_pos

        self.lift.apply_action(action)

        if left_pos and self.config.max_relative_target is not None:
            present_left = self.left_bus.sync_read("Present_Position", self.left_arm_motors)  # left_arm_*
            gp_left = {k: (v, present_left[k.replace(".pos", "")]) for k, v in left_pos.items()}
            left_pos = ensure_safe_goal_position(gp_left, self.config.max_relative_target)

        if self.right_bus and right_pos and self.config.max_relative_target is not None:
            present_right = self.right_bus.sync_read("Present_Position", self.right_arm_motors)
            gp_right = {k: (v, present_right[k.replace(".pos", "")]) for k, v in right_pos.items()}
            right_pos = ensure_safe_goal_position(gp_right, self.config.max_relative_target)


        # Send goal position to the actuators
        # arm_goal_pos_raw = {k.replace(".pos", ""): v for k, v in arm_goal_pos.items()}
        # self.left_bus.sync_write("Goal_Position", arm_goal_pos_raw)
        # self.left_bus.sync_write("Goal_Velocity", base_wheel_goal_vel)

        # return {**arm_goal_pos, **base_goal_vel}

        #print(f"[{filename}:{lineno}]Sending left_pos:{left_pos}, right_pos:{right_pos}, base_wheel_goal_vel:{base_wheel_goal_vel}")  # debug
    
        if left_pos:
            self.left_bus.sync_write("Goal_Position", {k.replace(".pos", ""): v for k, v in left_pos.items()})
        if self.right_bus and right_pos:
            self.right_bus.sync_write("Goal_Position", {k.replace(".pos", ""): v for k, v in right_pos.items()})
        self.base_bus.sync_write("Goal_Velocity", base_wheel_goal_vel)

        lift_sent = {k: v for k, v in action.items() if k.startswith("lift_axis.")}
        return {**left_pos, **right_pos, **base_goal_vel, **lift_sent}


    def stop_base(self):
        self.base_bus.sync_write("Goal_Velocity", dict.fromkeys(self.base_motors, 0), num_retry=0)
        logger.info("Base motors stopped")

    def disable_arm_torque(self):
        self.left_bus.disable_torque(self.left_arm_motors)
        if self.right_bus:
            self.right_bus.disable_torque(self.right_arm_motors)

    def enable_arm_torque(self):
        self.left_bus.enable_torque(self.left_arm_motors)
        if self.right_bus:
            self.right_bus.enable_torque(self.right_arm_motors)

    def read_and_check_currents(self, limit_ma, print_currents):
        """Read left/right bus currents (mA), print them, and enforce overcurrent protection"""
        scale = 6.5  # sts3215 current unit conversion factor
        left_curr_raw = {}
        left_curr_raw = self.left_bus.sync_read("Present_Current", list(self.left_bus.motors.keys()))
        right_curr_raw = {}
        if getattr(self, "right_bus", None):
            right_curr_raw = self.right_bus.sync_read("Present_Current", list(self.right_bus.motors.keys()))

        now = time.monotonic()
        if print_currents and (now - self._last_currents_log_t >= 1.0):
            left_arr = [int(float(raw) * scale) for raw in left_curr_raw.values()]
            print(f"[Currents][left_bus] {left_arr}")
            if right_curr_raw:
                right_arr = [int(float(raw) * scale) for raw in right_curr_raw.values()]
                print(f"[Currents][right_bus] {right_arr}")
            self._last_currents_log_t = now

        tripped = None
        for name, raw in {**left_curr_raw, **right_curr_raw}.items():
            current_ma = float(raw) * scale

            if current_ma > limit_ma:
                self._overcurrent_count[name] = self._overcurrent_count.get(name, 0) + 1
                print(f"[Overcurrent] {name}: {current_ma:.1f} mA > {limit_ma:.1f} mA ")
            else:
                # reset when it goes back to normal -> "consecutive" semantics
                self._overcurrent_count[name] = 0

            if self._overcurrent_count[name] >= self._overcurrent_trip_n:
                tripped = (name, current_ma, self._overcurrent_count[name])
                break

        if tripped is not None:
            name, current_ma, n = tripped
            print(
                f"[Overcurrent] {name}: {current_ma:.1f} mA > {limit_ma:.1f} mA "
                f"for {n} consecutive reads, disconnecting!"
            )
            try:
                self.stop_base()
            except Exception:
                pass
            try:
                self.disconnect()
            except Exception as e:
                print(f"[Overcurrent] disconnect error: {e}")
            sys.exit(1)


        return {k: round(v * scale, 1) for k, v in {**left_curr_raw, **right_curr_raw}.items()}

    @check_if_not_connected
    def disconnect(self):
        self.stop_base()
        self.left_bus.disconnect(self.config.disable_torque_on_disconnect)
        self.right_bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam_key, cam in self.cameras.items():
            try:
                cam.disconnect()
            except Exception as e:
                logger.warning(f"{self} camera '{cam_key}' disconnect failed: {e}")

        logger.info(f"{self} disconnected.")
