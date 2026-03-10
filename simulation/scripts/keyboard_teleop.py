import sys
import tty
import zmq
import math
import time
import json
import select
import termios
import argparse
from ast import arg

CMD_PORT = 5555
IP = "host.docker.internal"

def log(msg: str):
    print(f"\033[1;36m {msg}\033[0m")

def error(msg: str):
    print(f"\033[1;31m [ERROR] {msg}\033[0m")


MSG = """
KEYBOARD MAPPING REFERENCE
==========================

NAVIGATION & LIFT
-----------------
[ Q ] [ W ] [ E ]        [ U ]  (Lift Up)
[ A ] [ S ] [ D ]        [ J ]  (Lift Down)
      [ X ]

W/X : Linear X (Forwards/Backwards)
A/D : Linear Y (Left/Right)
Q/E : Angular  (Rotate)


JOINT CONTROLS (1-12)
---------------------
Keys:      [ 1 ] [ 2 ] [ 3 ] [ 4 ] [ 5 ] [ 6 ]   [ 7 ] [ 8 ] [ 9 ] [ 0 ] [ - ] [ = ]
Direction:  (+)   (+)   (+)   (+)   (+)   (+)     (+)   (+)   (+)   (+)   (+)   (+)
Reverse:   [ ! ] [ @ ] [ # ] [ $ ] [ % ] [ ^ ]   [ & ] [ * ] [ ( ] [ ) ] [ _ ] [ + ] (Shift+Key)

CORRESPONDENCE:
1  -> arm_left_shoulder_pan      7  -> arm_right_shoulder_pan
2  -> arm_left_shoulder_lift     8  -> arm_right_shoulder_lift
3  -> arm_left_elbow_flex        9  -> arm_right_elbow_flex
4  -> arm_left_wrist_flex        0  -> arm_right_wrist_flex
5  -> arm_left_wrist_roll        -  -> arm_right_wrist_roll
6  -> arm_left_gripper           =  -> arm_right_gripper


MISC.
---------------
[ SPACE ] : RESET (All 0, Lift 300)
[   K   ] : VELOCITY STOP (Zeroes x, y, theta)
[ CTRL+C] : QUIT
"""

JOINT_NAMES = [
    "arm_left_shoulder_pan.pos", "arm_left_shoulder_lift.pos", "arm_left_elbow_flex.pos",
    "arm_left_wrist_flex.pos", "arm_left_wrist_roll.pos", "arm_left_gripper.pos",
    "arm_right_shoulder_pan.pos", "arm_right_shoulder_lift.pos", "arm_right_elbow_flex.pos",
    "arm_right_wrist_flex.pos", "arm_right_wrist_roll.pos", "arm_right_gripper.pos",
]

JOINT_KEYS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=']
JOINT_REVERSE_KEYS = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+']

MOVE_BINDINGS = {
    'w': ('x.vel', 0.05),
    's': ('x.vel', -0.05),
    'a': ('y.vel', 0.05),
    'd': ('y.vel', -0.05),
    'q': ('theta.vel', 10.0),
    'e': ('theta.vel', -10.0),
}

LIFT_BINDINGS = {
    'u': ('lift_axis.height_mm', 2.0),
    'j': ('lift_axis.height_mm', -2.0),
}

settings = termios.tcgetattr(sys.stdin)

def getKey():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def limit(val, min_val, max_val):
    return max(min(val, max_val), min_val)

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


def main():
    parser = argparse.ArgumentParser(description="Standalone Keyboard Teleoperation for LeKiwi")
    parser.add_argument("--hide_state", action="store_true", help="Hide the continuously printed target state for cleaner output")

    args = parser.parse_args()


    log(f"[KEYBOARD TELEOP] Connecting @ command port {CMD_PORT} w/ host IP {IP}")
    context = zmq.Context()
    cmd_socket = context.socket(zmq.PUSH)
    cmd_socket.setsockopt(zmq.CONFLATE, 1)
    cmd_socket.connect(f"tcp://{IP}:{CMD_PORT}")

    target_state = get_default_state()
    joint_increment = 0.05

    try:
        print(MSG + "\n")
        while True:
            key = getKey()
            
            if key == '\x03': # CTRL-C
                break

            # RESET Logic
            if key == ' ':
                target_state = get_default_state()
                #print("RESET TO DEFAULTS")

            elif key == 'k':
                target_state["x.vel"] = 0.0
                target_state["y.vel"] = 0.0
                target_state["theta.vel"] = 0.0
                #print("VELOCITY STOP")

            elif key in MOVE_BINDINGS:
                attr, val = MOVE_BINDINGS[key]
                target_state[attr] += val
            
            elif key in LIFT_BINDINGS:
                attr, val = LIFT_BINDINGS[key]
                target_state[attr] += val

            elif key in JOINT_KEYS:
                idx = JOINT_KEYS.index(key)
                target_state[JOINT_NAMES[idx]] += joint_increment
                #print(f"Joint {JOINT_NAMES[idx]}: {target_state[JOINT_NAMES[idx]]:.3f}")

            elif key in JOINT_REVERSE_KEYS:
                idx = JOINT_REVERSE_KEYS.index(key)
                target_state[JOINT_NAMES[idx]] -= joint_increment
                #print(f"Joint {JOINT_NAMES[idx]}: {target_state[JOINT_NAMES[idx]]:.3f}")

            target_state["x.vel"] = limit(target_state["x.vel"], -0.5, 0.5)
            target_state["y.vel"] = limit(target_state["y.vel"], -0.5, 0.5)

            cmd_socket.send_string(json.dumps(target_state))
            if not args.hide_state:
                print(f"\rSent: {target_state}            ", end="", flush=True)

    except Exception as e:
        error(f"[KEYBOARD TELEOP] [ERROR] Unknown exception: {e}")

    finally:
        final_stop = get_default_state()
        cmd_socket.send_string(json.dumps(final_stop))
        cmd_socket.close()
        context.term()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

if __name__ == "__main__":
    main()
