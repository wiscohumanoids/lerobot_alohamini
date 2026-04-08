#!/usr/bin/env python3

from datetime import datetime
import argparse
from pathlib import Path
import sys
import time
import termios
import tty
import select
from email import parser

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.robots.alohamini.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.alohamini.lekiwi_client import LeKiwiClient
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.visualization_utils import init_rerun


def get_all_keys():
    """Read all available characters from stdin at once without blocking"""
    keys = []
    settings = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin.fileno())
    while True:
        rlist, _, _ = select.select([sys.stdin], [], [], 0.01)
        if rlist:
            keys.append(sys.stdin.read(1))
        else:
            break
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return keys


def limit(val, min_val, max_val):
    """Limit a value between min and max"""
    return max(min(val, max_val), min_val)


class CustomKeyboardHandler:
    """Custom keyboard handler using raw input for direct teleoperation control"""
    
    def __init__(self):
        self._docker_keyboard_overload = True  # marker attribute to identify this class as a keyboard handler for compatibility with record_loop

        self.MOVE_BINDINGS = {
            'w': ('x.vel', 0.1),
            's': ('x.vel', -0.1),
            'a': ('y.vel', 0.1),
            'd': ('y.vel', -0.1),
            'q': ('theta.vel', 8.0),
            'e': ('theta.vel', -8.0),
            'u': ('lift_axis.height_mm', 4.0),
            'j': ('lift_axis.height_mm', -4.0),
        }
        self.key_last_received = {k: 0 for k in self.MOVE_BINDINGS.keys()}
        self.KEY_TIMEOUT = 0.2
        self.is_connected = True
    
    def connect(self):
        """Placeholder for compatibility with other teleop devices"""
        self.is_connected = True
    
    def disconnect(self):
        """Placeholder for compatibility with other teleop devices"""
        self.is_connected = False
    
    def get_action(self):
        """Get keyboard action with timeout-based key repeat"""
        current_time = time.time()
        keys_pressed = get_all_keys()
        
        # Update last received time for pressed keys
        for k in keys_pressed:
            if k in self.key_last_received:
                self.key_last_received[k] = current_time
            if k == '\x1b':  # escape for exit
                sys.exit(0)
        
        # Build action from current key states
        action = {
            "x.vel": 0.0,
            "y.vel": 0.0,
            "theta.vel": 0.0,
            "lift_axis.height_mm": 0.0
        }
        
        for k in self.MOVE_BINDINGS.keys():
            if k in self.key_last_received and current_time - self.key_last_received[k] < self.KEY_TIMEOUT:
                attr, val = self.MOVE_BINDINGS[k]
                action[attr] += val
        
        action["lift_axis.height_mm"] = 0.0  # disable lift axis
        
        # Space bar to stop
        if ' ' in keys_pressed:
            action["x.vel"] = 0.0
            action["y.vel"] = 0.0
            action["theta.vel"] = 0.0
        
        return action


def main():
    parser = argparse.ArgumentParser(description="Record episodes with bi-arm teleoperation")
    parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset repo_id, e.g. liyitenga/record_20250914225057")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to record")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--episode_time", type=int, default=60, help="Duration of each episode (seconds)")
    parser.add_argument("--reset_time", type=int, default=10, help="Reset duration between episodes (seconds)")
    parser.add_argument("--task_description", type=str, default="My task description4", help="Task description")
    parser.add_argument("--remote_ip", type=str, default="127.0.0.1", help="Robot host IP")
    parser.add_argument("--robot_id", type=str, default="lekiwi_host", help="Robot ID")
    parser.add_argument("--leader_id", type=str, default="so101_leader_bi", help="Leader arm device ID")
    parser.add_argument(
        "--arm_profile",
        type=str,
        default="so-arm-5dof",
        choices=["so-arm-5dof", "am-arm-6dof"],
        help="Arm profile selector used for both leader and follower consistency.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume recording on existing dataset")

    args = parser.parse_args()

    # === Robot and teleop config ===
    robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id=args.robot_id)
    leader_arm_config = BiSOLeaderConfig(
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

    robot = LeKiwiClient(robot_config)
    leader_arm = BiSOLeader(leader_arm_config)
    keyboard = CustomKeyboardHandler()

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # === Dataset setup ===
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    dataset_root = Path(args.dataset.split("/")[-1])

    if args.resume:
        print("Resuming existing dataset:", args.dataset)
        dataset = LeRobotDataset(
            args.dataset,
        )
        dataset.start_image_writer(num_threads=4)
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.dataset,
            fps=args.fps,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
        )
        print(f"Dataset created with id: {dataset.repo_id}")

    # === Connect devices ===
    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    listener, events = init_keyboard_listener()
    init_rerun(session_name="lekiwi_record")

    if not robot.is_connected or not leader_arm.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("Starting record loop...")
    recorded_episodes = 0

    while recorded_episodes < args.num_episodes and not events["stop_recording"]:
        print(f"Recording episode {recorded_episodes + 1} of {args.num_episodes}")

        # === Main record loop ===
        record_loop(
            robot=robot,
            events=events,
            fps=args.fps,
            dataset=dataset,
            teleop=[leader_arm, keyboard],
            control_time_s=args.episode_time,
            single_task=args.task_description,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        # === Reset environment ===
        if not events["stop_recording"] and (
            (recorded_episodes < args.num_episodes - 1) or events["rerecord_episode"]
        ):
            print("Reset the environment")
            record_loop(
                robot=robot,
                events=events,
                fps=args.fps,
                teleop=[leader_arm, keyboard],
                control_time_s=args.reset_time,
                single_task=args.task_description,
                display_data=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

        if events["rerecord_episode"]:
            print("Re-record episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        recorded_episodes += 1

    # === Clean up ===
    print("Stop recording")
    robot.disconnect()
    leader_arm.disconnect()
    keyboard.disconnect()
    listener.stop()
    dataset.finalize()
    dataset.push_to_hub()


if __name__ == "__main__":
    main()
