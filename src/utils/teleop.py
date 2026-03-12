
# FOR REFERENCE ONLY!!!

"""
UNIVERSAL SCRIPT FOR TELEOPERATION IN ISAACSIM TO BE RUN FROM INSIDE THE DOCKER CONTAINER, LOCALLY
"""

CMD_PORT = 5555
IP = "host.docker.internal"

def log(msg: str):
    print(f"\033[1;36m[TELEOP] {msg}\033[0m")

def error(msg: str):
    print(f"\033[1;31m[TELEOP] [ERROR] {msg}\033[0m")

log("Teleop loading...")

#from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardSimTeleop, KeyboardSimTeleopConfig

from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.robot_utils import precise_sleep

log("Imports successful!")

import sys
import tty
import zmq
import math
import time
import json
import select
import termios
import argparse
import threading
import queue
from enum import Enum


KEYBOARD_BINDINGS = {
    'w': ('x.vel', 0.05),
    's': ('x.vel', -0.05),
    'a': ('y.vel', 0.05),
    'd': ('y.vel', -0.05),
    'q': ('theta.vel', 0.05),
    'e': ('theta.vel', -0.05),
    'u': ('lift_axis.height_mm', 2.0),
    'j': ('lift_axis.height_mm', -2.0),
}


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--left_leader_port", type=str, default="/dev/am_arm_leader_left", help="USB device port for left leader arm, if necessary")
    parser.add_argument("--right_leader_port", type=str, default="/dev/am_arm_leader_right", help="USB device port for right leader arm, if necessary")

    try:
        args = parser.parse_args()
    except argparse.ArgumentError as e:
        error(f"Argparse error: {e}")
        sys.exit(-1)


    log(f"Connecting with left_leader_port=\"{args.left_leader_port}\", right_leader_port=\"{args.right_leader_port}\"")
    bi_config = BiSOLeaderConfig(
        left_arm_config=SOLeaderConfig(
            port=args.left_leader_port,
            arm_profile=args.leader_profile,
        ),
        right_arm_config=SOLeaderConfig(
            port=args.right_leader_port,
            arm_profile=args.leader_profile,
        ),
        id=args.leader_id,
    )
    teleop = BiSOLeader(bi_config)


    log(f"Connecting to command port {CMD_PORT} w/ host IP {IP}...")
    context = zmq.Context()
    cmd_socket = context.socket(zmq.PUSH)
    cmd_socket.setsockopt(zmq.CONFLATE, 1)
    cmd_socket.connect(f"tcp://{IP}:{CMD_PORT}")

    log("Connecting teleop...")
    teleop.connect()
    if not teleop.is_connected:
        error("Teleop not able to connect!")
        sys.exit(-1)

    # Main loop


    log("STARTING TELEOP")
    try:
        while True:
            t0 = time.perf_counter()

            arm_state = teleop.get_action()

            target_state =  arm_state

            cmd_socket.send_string(json.dumps(target_state))

            if args.show_state:
                print(target_state)
                #print(f"\rSent: {json.dumps(formatted_state)}                   ", end="", flush=True)
            precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))
    except Exception as e:
        error(f"Unknown exception: {e}")
    finally:
        log("Quitting teleop...")
        cmd_socket.close()
        context.term()

if __name__ == "__main__":
    main()