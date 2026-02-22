import logging
import argparse
import inspect
import json
import os
import time
import zmq
import sys
from enum import Enum

from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardSimTeleop, KeyboardSimTeleopConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.robot_utils import precise_sleep

from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.phone_processor import MapPhoneActionToRobotAction
from lerobot.teleoperators.phone.teleop_phone import Phone

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)


class TeleopType(Enum):
    LEADER = 1
    KEYBOARD = 2
    PHONE = 3


def log(msg: str):
    print(f"\033[1;36m{msg}\033[0m")

def error(msg: str):
    print(f"\033[1;31m[ERROR] {msg}\033[0m")

log("Teleop starting...")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--type",
    type=str,
    choices=[member.name.lower() for member in TeleopType],
    help="Type of teleop to use"
)
parser.add_argument("--fps", type=int, default=30, help="Main loop frequency (frames per second)")
parser.add_argument("--leader_id", type=str, default="so101_leader_bi", help="Leader arm device ID")
parser.add_argument(
    "--leader_profile",
    type=str,
    default="so-arm-5dof",
    choices=["so-arm-5dof", "am-arm-6dof"],
    help="Leader arm profile selector.",
)
parser.add_argument("--show_state", action="store_true", help="Continually show target state for debug")

args = parser.parse_args()
if args.type is None or TeleopType[args.type.upper()] is None:
    error("Must include a VALID teleop type with --type!")


TELEOP_TYPE = TeleopType[args.type.upper()]
CMD_PORT = 5555
FPS = args.fps
IP = "host.docker.internal"


match TELEOP_TYPE:
    case TeleopType.LEADER:
        bi_cfg = BiSOLeaderConfig(
            left_arm_config=SOLeaderConfig(
                port="/dev/am_arm_leader_left",
                arm_profile=args.leader_profile,
            ),
            right_arm_config=SOLeaderConfig(
                port="/dev/am_arm_leader_right",
                arm_profile=args.leader_profile,
            ),
            id=args.leader_id,
        )
        teleop = BiSOLeader(bi_cfg)
    case TeleopType.KEYBOARD:
        keyboard_config = KeyboardSimTeleopConfig(id="keyboard")
        teleop = KeyboardSimTeleop(keyboard_config)     # CURRENTLY BROKEN -- FIX LATER!!
    case TeleopType.PHONE:
        phone_config = PhoneConfig(phone_os=PhoneOS.ANDROID)
        teleop = Phone(phone_config)

        ARM_JOINT_NAMES = {
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_yaw",
            "wrist_roll",
            "gripper"
        }

        kinematics_solver = RobotKinematics(
            urdf_path="./assets/SO101/so101_new_calib.urdf",
            target_frame_name="gripper_frame_link",
            joint_names=ARM_JOINT_NAMES
        )

        phone_to_robot_joints_processor = RobotProcessorPipeline[
            tuple[RobotAction, RobotObservation], RobotAction
        ](
            steps=[
                MapPhoneActionToRobotAction(platform=phone_config.phone_os),
                EEReferenceAndDelta(
                    kinematics=kinematics_solver,
                    end_effector_step_sizes={"x": 0.5, "y": 0.5, "z": 0.5},
                    motor_names=ARM_JOINT_NAMES,
                    use_latched_reference=True,
                ),
                EEBoundsAndSafety(
                    end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                    max_ee_step_m=0.10,
                ),
                GripperVelocityToJoint(
                    speed_factor=20.0,
                ),
                InverseKinematicsEEToJoints(
                    kinematics=kinematics_solver,
                    motor_names=ARM_JOINT_NAMES,
                    initial_guess_current_joints=True,
                ),
            ],
            to_transition=robot_action_observation_to_transition,
            to_output=transition_to_robot_action,
        )


        pass


log(f"[SIM TELEOP] Connecting @ command port {CMD_PORT} w/ host IP {IP}")
context = zmq.Context()
cmd_socket = context.socket(zmq.PUSH)
cmd_socket.setsockopt(zmq.CONFLATE, 1)
cmd_socket.connect(f"tcp://{IP}:{CMD_PORT}")


teleop.connect()
if not teleop.is_connected:
    error("Teleop not able to connect!")
    sys.exit(-1)

# Main loop
try:
    while True:
        t0 = time.perf_counter()

        if TELEOP_TYPE is TeleopType.PHONE:
            target_state = phone_to_robot_joints_processor((teleop.get_action(), None))
        else:
            target_state = teleop.get_action()
        cmd_socket.send_string(json.dumps(target_state))

        if args.show_state:
            print(f"\rSent: {target_state}                       ", end="", flush=True)
        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
except Exception as e:
    error(f" Unknown exception: {e}")
finally:
    log("Quitting teleop...")
    cmd_socket.close()
    context.term()
