import argparse
import inspect
import os
import time
import termios
import select
import sys
import tty

from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

parser = argparse.ArgumentParser()
parser.add_argument("--no_robot", action="store_true", help="Do not connect robot, only print actions")
parser.add_argument("--no_leader", action="store_true", help="Do not connect leader arm, only perform keyboard-controlled actions.")
parser.add_argument("--fps", type=int, default=30, help="Main loop frequency (frames per second)")
parser.add_argument("--remote_ip", type=str, default="10.139.203.203", help="LeKiwi host IP address")
parser.add_argument("--leader_id", type=str, default="so101_leader_bi", help="Leader arm device ID")
parser.add_argument("--use_rerun", action="store_true", help="Enable Rerun vis (kind of extraneous)")
parser.add_argument("--no_lift", action="store_true", help="Do not send lift-axis commands during teleop.")
parser.add_argument(
    "--arm_profile",
    type=str,
    default="so-arm-5dof",
    choices=["so-arm-5dof", "am-arm-6dof"],
    help="Arm profile selector used for both leader and follower consistency.",
)

args = parser.parse_args()

NO_ROBOT = args.no_robot
NO_LEADER = args.no_leader
USE_RERUN = args.use_rerun
FPS = args.fps
NO_LIFT = args.no_lift

if NO_ROBOT:
    print("NO_ROBOT: robot will not connect, only print actions.")

if NO_LEADER:
    print("NO_LEADER: leader arm will not connect, only print actions.")

if USE_RERUN:
    print("USE_RERUN: Rerun visualization enabled, logging data to Rerun dashboard.")

# Create configs
robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id="my_alohamini")
bi_cfg = BiSOLeaderConfig(
    left_arm_config=SOLeaderConfig(
        port="/dev/ttyACM0",
        arm_profile=args.arm_profile,
    ),
    right_arm_config=SOLeaderConfig(
        port="/dev/ttyACM1",
        arm_profile=args.arm_profile,
    ),
    id=args.leader_id,
)
leader = BiSOLeader(bi_cfg)
robot = LeKiwiClient(robot_config)

# Connection logic
if not NO_ROBOT:
    robot.connect()
else:
    print("NO_ROBOT: robot will not connect, only print actions.")

if not NO_LEADER:
    leader.connect()
else:
    print("NO_LEADER: leader arm will not connect, only print actions.")

# Keyboard control setup
MOVE_BINDINGS = {
    'w': ('x.vel', 0.1),
    's': ('x.vel', -0.1),
    'a': ('y.vel', 0.1),
    'd': ('y.vel', -0.1),
    'q': ('theta.vel', 8.0),
    'e': ('theta.vel', -8.0),
    'u': ('lift_axis.height_mm', 4.0),
    'j': ('lift_axis.height_mm', -4.0),
}

settings = termios.tcgetattr(sys.stdin)
key_last_received = {k: 0 for k in MOVE_BINDINGS.keys()}
KEY_TIMEOUT = 0.2

# Read all available characters from stdin at once without blocking
def get_all_keys():
    keys = []
    tty.setraw(sys.stdin.fileno())
    while True:
        # Check if there is data in stdin (0.0 timeout = non-blocking)
        rlist, _, _ = select.select([sys.stdin], [], [], 0.01)
        if rlist:
            keys.append(sys.stdin.read(1))
#            print("DEBUG ---- read something!")
        else:
            break
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return keys

def limit(val, min_val, max_val):
    return max(min(val, max_val), min_val)

if USE_RERUN:
    init_rerun(session_name="lekiwi_teleop")

# Main loop
action = {
    "x.vel": 0.0,
    "y.vel": 0.0,
    "theta.vel": 0.0
}

while True:
    t0 = time.perf_counter()

    observation = robot.get_observation() if not NO_ROBOT else {}
    arm_actions = leader.get_action() if not NO_LEADER else {}
    arm_actions = {f"arm_{k}": v for k, v in arm_actions.items()}

    current_time = time.time()
    keys_pressed = get_all_keys()

    for k in keys_pressed:
        if k in key_last_received:
            key_last_received[k] = current_time
        if k == '\x1b': # escape for exit
            sys.exit(0)

    current_action = {
        "x.vel": 0.0,
        "y.vel": 0.0,
        "theta.vel": 0.0,
    }

    for k in MOVE_BINDINGS.keys():
        if k in key_last_received and current_time - key_last_received[k] < KEY_TIMEOUT:
            attr, val = MOVE_BINDINGS[k]
            if attr not in current_action:
                current_action[attr] = 0.0
            current_action[attr] += val

    if not NO_LIFT and current_action.get("lift_axis.height_mm") is not None:
        current_action["lift_axis.height_mm"] = limit(current_action["lift_axis.height_mm"], 0.0, 600.0)

    SPACE = ' '
    if SPACE in keys_pressed:
        current_action["x.vel"] = 0.0
        current_action["y.vel"] = 0.0
        current_action["theta.vel"] = 0.0

    action = {**arm_actions, **current_action}
    #action = {**arm_actions, **base_action, **lift_action}
    if USE_RERUN:
        log_rerun_data(observation, action)

    if not NO_ROBOT:
        robot.send_action(action)

    precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    loop_dt = time.perf_counter() - t0
    loop_fps = 1.0 / loop_dt if loop_dt > 0 else float("inf")

    if NO_ROBOT:
        print(f"[fps={loop_fps:.1f}] [NO_ROBOT] action → {action}")
    else:
        print(f"[fps={loop_fps:.1f}] Sent action → {action}")
