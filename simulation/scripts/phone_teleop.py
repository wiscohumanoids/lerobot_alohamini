import math
from ast import arg
import sys
import select
import termios
import tty
import time
import json
import zmq
import argparse
import logging
import threading
import time

import hebi
import numpy as np
from teleop import Teleop
import uvicorn

from lerobot.utils.rotation import Rotation


CMD_PORT = 5555
IP = "host.docker.internal"

def log(msg: str):
    print(f"\033[1;36m {msg}\033[0m")

def error(msg: str):
    print(f"\033[1;31m [ERROR] {msg}\033[0m")



def run_no_ssl(self):
    """Run without SSL for local development or behind a reverse proxy."""
    #print(self.__dict__)
    uvicorn.run(self._Teleop__app, host="0.0.0.0", port=8888, log_level="critical")
    print("\n(Connect by local IP on same network)")

Teleop.run = run_no_ssl



JOINT_NAMES = [
    "arm_left_shoulder_pan.pos", "arm_left_shoulder_lift.pos", "arm_left_elbow_flex.pos",
    "arm_left_wrist_flex.pos", "arm_left_wrist_roll.pos", "arm_left_gripper.pos",
    "arm_right_shoulder_pan.pos", "arm_right_shoulder_lift.pos", "arm_right_elbow_flex.pos",
    "arm_right_wrist_flex.pos", "arm_right_wrist_roll.pos", "arm_right_gripper.pos",
]

def get_default_state():
    state = {
        "x.vel": 0.0,
        "y.vel": 0.0,
        "theta.vel": 0.0,
        "lift_axis.height_mm": 300.0
    }
    for name in JOINT_NAMES:
        state[name] = 0.0
    return state


class Phone():

    _calib_pos: np.ndarray | None = None
    _calib_rot_inv: Rotation | None = None

    def __init__(self):
        self._teleop = None
        self._teleop_thread = None
        self._latest_pose = None
        self._latest_message = None
        self._android_lock = threading.Lock()
        self.is_calibrated = True
        self._enabled = True


    def connect(self) -> None:
        log("[PHONE TELEOP] Connecting phone WebXR server...")
        self._teleop = Teleop(host="0.0.0.0", port="8888")
        self._teleop.subscribe(self._android_callback)
        self._teleop_thread = threading.Thread(target=self._teleop.run, daemon=True)
        self._teleop_thread.start()
        #self.calibrate()

    def disconnect(self) -> None:
        self._teleop = None
        if self._teleop_thread and self._teleop_thread.is_alive():
            self._teleop_thread.join(timeout=1.0)
            self._teleop_thread = None
            self._latest_pose = None

    def _read_current_pose(self) -> tuple[bool, np.ndarray | None, Rotation | None, object | None]:
        """
        Reads the latest 6-DoF pose received from the Android device's WebXR session.

        This method accesses the most recent pose data stored by the `_android_callback`. It uses a
        thread lock to safely read the shared `_latest_pose` variable. The pose, a 4x4 matrix, is
        then decomposed into position and rotation, and the configured camera offset is applied.

        Returns:
            A tuple containing:
            - A boolean indicating if a valid pose was available.
            - The 3D position as a NumPy array, or None if no pose has been received yet.
            - The orientation as a `Rotation` object, or None if no pose has been received.
            - The raw 4x4 pose matrix as received from the teleop stream.
        """
        with self._android_lock:
            if self._latest_pose is None:
                return False, None, None, None
            p = self._latest_pose.copy()
            pose = self._latest_pose
        rot = Rotation.from_matrix(p[:3, :3])
        #pos = p[:3, 3] - rot.apply(self.config.camera_offset)
        pos = p[:3, 3]
        return True, pos, rot, pose

    def _wait_for_capture_trigger(self) -> tuple[np.ndarray, Rotation]:
        """
        Blocks execution until the calibration trigger is detected from the Android device.

        This method enters a loop, continuously checking the latest message received from the WebXR
        session. It waits for the user to touch and move their finger on the screen, which generates
        a `move` event. Once this event is detected, the loop breaks and returns the phone's current
        pose.

        Returns:
            A tuple containing the position (np.ndarray) and rotation (Rotation) of the phone at the
            moment the trigger was activated.
        """
        while True:
            with self._android_lock:
                msg = self._latest_message or {}
               # print(msg)

            if bool(msg.get("move", False)):
                ok, pos, rot, _pose = self._read_current_pose()
                if ok:
                    return pos, rot

            time.sleep(0.01)

    def calibrate(self) -> None:
        print(
            "Hold the phone so that: top edge points forward in same direction as the robot (robot +x) and screen points up (robot +z)"
        )
        print("Touch and drag on the WebXR page to capture this pose...\n")

        pos, rot = self._wait_for_capture_trigger()
        self._calib_pos = pos.copy()
        self._calib_rot_inv = rot.inv()
        self._enabled = False
        print("Calibration done\n")


    def _reapply_position_calibration(self, pos: np.ndarray) -> None:
        self._calib_pos = pos.copy()

    def _android_callback(self, pose: np.ndarray, message: dict) -> None:
        print(f"RAW DATA INBOUND: {message}")
        with self._android_lock:
            self._latest_pose = pose
            self._latest_message = message

    def get_action(self) -> dict:
        has_pose, raw_position, raw_rotation, fb_pose = self._read_current_pose()
        if not has_pose or not self.is_calibrated:
            return {}

        # Collect raw inputs (B1 / analogs on iOS, move/scale on Android)
        raw_inputs: dict[str, float | int | bool] = {}
        io = getattr(fb_pose, "io", None)
        if io is not None:
            bank_a, bank_b = io.a, io.b
            if bank_a:
                for ch in range(1, 9):
                    if bank_a.has_float(ch):
                        raw_inputs[f"a{ch}"] = float(bank_a.get_float(ch))
            if bank_b:
                for ch in range(1, 9):
                    if bank_b.has_int(ch):
                        raw_inputs[f"b{ch}"] = int(bank_b.get_int(ch))
                    elif hasattr(bank_b, "has_bool") and bank_b.has_bool(ch):
                        raw_inputs[f"b{ch}"] = int(bank_b.get_bool(ch))

        enable = bool(raw_inputs.get("b1", 0))

        # Rising edge then re-capture calibration immediately from current raw pose
        if enable and not self._enabled and raw_position is not None:
            self._reapply_position_calibration(raw_position)

        # Apply calibration
        pos_cal = self._calib_rot_inv.apply(raw_position - self._calib_pos)
        rot_cal = self._calib_rot_inv * raw_rotation

        self._enabled = enable

        return {
            "phone.pos": pos_cal,
            "phone.rot": rot_cal,
            "phone.raw_inputs": raw_inputs,
            "phone.enabled": self._enabled,
        }





def main():
    parser = argparse.ArgumentParser(description="Standalone PHONE Teleoperation for LeKiwi")
    parser.add_argument("--hide_state", action="store_true", help="Hide the continuously printed target state for cleaner output")

    args = parser.parse_args()


    log(f"[PHONE TELEOP] Connecting @ command port {CMD_PORT} w/ host IP {IP}")
    context = zmq.Context()
    cmd_socket = context.socket(zmq.PUSH)
    cmd_socket.setsockopt(zmq.CONFLATE, 1)
    cmd_socket.connect(f"tcp://{IP}:{CMD_PORT}")

    target_state = get_default_state()
    joint_increment = 0.05

    phone = Phone()

    try:
        phone.connect()
        while True:
            target_state = phone.get_action()
            #cmd_socket.send_string(json.dumps(target_state))
            if not args.hide_state:
                print(f"\rSent: {target_state}            ", end="", flush=True)

    except Exception as e:
        error(f"[PHONE TELEOP] Unknown exception: {e}")

    finally:
        final_stop = get_default_state()
        cmd_socket.send_string(json.dumps(final_stop))

        phone.disconnect()
        cmd_socket.close()
        context.term()

if __name__ == "__main__":
    main()
