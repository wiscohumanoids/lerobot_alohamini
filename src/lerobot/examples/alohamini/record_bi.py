#!/usr/bin/env python3

from email import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.robots.alohamini.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.alohamini.lekiwi_client import LeKiwiClient
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.runtime_config import (
    format_args_block,
    format_dataclass_block,
    print_runtime_banner,
    validate_client_fps_vs_cameras,
)
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from datetime import datetime
import argparse
import shutil
from pathlib import Path
import threading
import time


def _init_delete_last_listener(events: dict):
    """Start a pynput listener that flags events['delete_last_saved']=True on Down arrow."""
    try:
        from pynput import keyboard
    except Exception:
        return None

    def on_press(key):
        if key == keyboard.Key.down:
            events["delete_last_saved"] = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def _apply_queued_deletions(dataset: LeRobotDataset, deletion_queue: list[int]) -> LeRobotDataset:
    """Re-encode the dataset once with all queued episodes removed. Returns the new dataset."""
    from lerobot.datasets.dataset_tools import delete_episodes

    original_root = dataset.root
    old_root = Path(str(original_root) + "_old")
    if old_root.exists():
        shutil.rmtree(old_root)
    shutil.move(str(original_root), str(old_root))
    dataset.root = old_root
    return delete_episodes(
        dataset,
        episode_indices=sorted(set(deletion_queue)),
        output_dir=original_root,
        repo_id=dataset.repo_id,
    )


def get_bi_teleop_action(
    robot: LeKiwiClient,
    leader_arm: BiSOLeader,
    keyboard: KeyboardTeleop,
    no_keyboard: bool = True,
) -> dict[str, float]:
    """Collect the current bi-arm and keyboard teleop command."""
    arm_action = leader_arm.get_action()
    arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
    keyboard_action = keyboard.get_action() if not no_keyboard else {}
    base_action = robot._from_keyboard_to_base_action(keyboard_action)
    lift_action = robot._from_keyboard_to_lift_action(keyboard_action)
    return {**arm_action, **base_action, **lift_action}


def save_episode_with_live_preview(
    dataset: LeRobotDataset,
    robot: LeKiwiClient,
    leader_arm: BiSOLeader,
    keyboard: KeyboardTeleop,
    fps: int,
    display_data: bool,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    no_keyboard: bool = True,
) -> None:
    """Save episode while continuously updating rerun camera preview."""
    save_done = threading.Event()
    save_error = {"exception": None}

    def _save_episode() -> None:
        try:
            dataset.save_episode()
        except Exception as exc:  # keep exception to re-raise on main thread
            save_error["exception"] = exc
        finally:
            save_done.set()

    save_thread = threading.Thread(target=_save_episode, daemon=True)
    save_thread.start()

    while not save_done.is_set():
        loop_start = time.perf_counter()
        observation = robot.get_observation()
        teleop_action = get_bi_teleop_action(robot, leader_arm, keyboard, no_keyboard=no_keyboard)
        teleop_action_processed = teleop_action_processor((teleop_action, observation))
        robot_action = robot_action_processor((teleop_action_processed, observation))
        robot.send_action(robot_action)

        if display_data:
            observation_processed = robot_observation_processor(observation)
            log_rerun_data(
                observation=observation_processed,
                action=teleop_action_processed,
                compress_images=True,
            )

        dt_s = time.perf_counter() - loop_start
        sleep_s = max(1.0 / fps - dt_s, 0.0)
        save_done.wait(timeout=sleep_s)

    save_thread.join()
    if save_error["exception"] is not None:
        raise save_error["exception"]

def main():
    parser = argparse.ArgumentParser(description="Record episodes with bi-arm teleoperation")
    parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset repo_id, e.g. liyitenga/record_20250914225057")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to record")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--episode_time", type=int, default=20, help="Duration of each episode (seconds)")
    parser.add_argument("--reset_time", type=int, default=0, help="Reset duration between episodes (seconds)")
    parser.add_argument("--rerecord_reset_time", type=int, default=10, help="Reset duration when rerecording an episode via left arrow (seconds)")
    parser.add_argument("--task_description", type=str, default="My task description4", help="Task description")
    parser.add_argument("--remote_ip", type=str, default="10.139.203.203", help="Robot host IP")
    parser.add_argument("--robot_id", type=str, default="lekiwi_host", help="Robot ID")
    parser.add_argument("--leader_id", type=str, default="so101_leader_bi", help="Leader arm device ID")
    parser.add_argument(
        "--arm_profile",
        type=str,
        default="so-arm-5dof",
        choices=["so-arm-5dof", "am-arm-6dof"],
        help="Arm profile selector used for both leader and follower consistency.",
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default="h264",
        choices=["h264", "hevc", "libsvtav1"],
        help="Video codec used when encoding recorded camera streams.",
    )
    parser.add_argument(
    "--disable_rerun",
        action="store_true",
        help="Disable rerun visualization to avoid native viewer crashes on macOS.",
    )
    parser.add_argument("--rerun_port", type=int, default=9091, help="Port of the running rerun web-viewer")
    parser.add_argument("--keyboard", action="store_true", default=False, help="Enable keyboard teleop for base/lift control")
    parser.add_argument("--setup_time", type=int, default=15, help="Initial setup time in seconds before first episode")
    parser.add_argument("--resume", action="store_true", help="Resume recording on existing dataset")
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip the runtime config confirmation prompt.",
    )

    args = parser.parse_args()

    # === Robot and teleop config ===
    robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id=args.robot_id, no_keyboard=not args.keyboard)
    leader_arm_config = BiSOLeaderConfig(
        left_arm_config=SOLeaderConfig(
            port="/dev/cu.usbmodem5B140323471",
            arm_profile=args.arm_profile,
        ),
        right_arm_config=SOLeaderConfig(
            port="/dev/cu.usbmodem5B140330511",
            arm_profile=args.arm_profile,
        ),
        id=args.leader_id,
    )
    keyboard_config = KeyboardTeleopConfig()

    validate_client_fps_vs_cameras(
        args.fps,
        (cam.fps for cam in robot_config.cameras.values() if cam.fps is not None),
    )

    print_runtime_banner(
        format_args_block("CLI args", args, parser),
        format_dataclass_block("LeKiwiClientConfig", robot_config),
        format_dataclass_block("BiSOLeaderConfig", leader_arm_config),
        format_dataclass_block("KeyboardTeleopConfig", keyboard_config),
        require_confirm=not args.no_confirm,
    )

    robot = LeKiwiClient(robot_config)
    leader_arm = BiSOLeader(leader_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

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
            vcodec=args.vcodec,
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
            vcodec=args.vcodec,
        )
        print(f"Dataset created with id: {dataset.repo_id}")

    # === Connect devices ===
    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    listener, events = init_keyboard_listener()
    events["delete_last_saved"] = False
    delete_listener = _init_delete_last_listener(events)
    deletion_queue: list[int] = []
    last_saved_episode_idx: int | None = None
    if not args.disable_rerun:
        import os
        os.environ["LEROBOT_RERUN_MEMORY_LIMIT"] = "0%"   # no history kept in viewer memory
        os.environ["RERUN_FLUSH_NUM_BYTES"] = "0"           # flush every log call immediately
        init_rerun(session_name="lekiwi_record", ip="localhost", port=args.rerun_port)

    if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("Starting record loop...")

    # Initial setup countdown with live camera preview
    display_data = not args.disable_rerun
    log_say(f"Setup time: {args.setup_time} seconds to get ready")
    for remaining in range(args.setup_time, 0, -1):
        if events["stop_recording"]:
            break
        print(f"\rStarting in {remaining}s... ", end="", flush=True)
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline:
            loop_start = time.perf_counter()
            if display_data:
                observation = robot.get_observation()
                observation_processed = robot_observation_processor(observation)
                log_rerun_data(observation=observation_processed, action=None, compress_images=True)
            dt_s = time.perf_counter() - loop_start
            sleep_s = max(1.0 / args.fps - dt_s, 0.0)
            time.sleep(sleep_s)
    print()

    recorded_episodes = 0

    try:
        while recorded_episodes < args.num_episodes and not events["stop_recording"]:
            if events["delete_last_saved"]:
                events["delete_last_saved"] = False
                if last_saved_episode_idx is not None and last_saved_episode_idx not in deletion_queue:
                    deletion_queue.append(last_saved_episode_idx)
                    log_say(f"Episode {last_saved_episode_idx} queued for deletion")
                    last_saved_episode_idx = None
                else:
                    log_say("No episode to delete or already queued")

            log_say(f"Recording episode {recorded_episodes + 1} of {args.num_episodes}")

            # === Main record loop ===
            record_loop(
                robot=robot,
                events=events,
                fps=args.fps,
                dataset=dataset,
                teleop=[leader_arm, keyboard],
                control_time_s=args.episode_time,
                single_task=args.task_description,
                display_data=not args.disable_rerun,
                display_compressed_images=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

            # === Reset environment ===
            if not events["stop_recording"] and (
                (recorded_episodes < args.num_episodes - 1) or events["rerecord_episode"]
            ):
                reset_seconds = args.rerecord_reset_time if events["rerecord_episode"] else args.reset_time
                log_say(f"Reset the environment ({reset_seconds}s)")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=args.fps,
                teleop=[leader_arm, keyboard],
                control_time_s=reset_seconds,
                    single_task=args.task_description,
                    display_data=not args.disable_rerun,
                    display_compressed_images=True,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            save_episode_with_live_preview(
                dataset=dataset,
                robot=robot,
                leader_arm=leader_arm,
                keyboard=keyboard,
                fps=args.fps,
                display_data=not args.disable_rerun,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                no_keyboard=not args.keyboard,
            )
            recorded_episodes += 1
            last_saved_episode_idx = dataset.meta.total_episodes - 1

        # Save any episode that was fully recorded but not yet saved because
        # stop_recording was set before the loop body could reach save_episode.
        pending_frames = len(dataset.episode_buffer.get("frame_index", []))
        if pending_frames > 0:
            log_say(f"Saving pending episode ({pending_frames} frames) before exit")
            save_episode_with_live_preview(
                dataset=dataset,
                robot=robot,
                leader_arm=leader_arm,
                keyboard=keyboard,
                fps=args.fps,
                display_data=not args.disable_rerun,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                no_keyboard=not args.keyboard,
            )
            recorded_episodes += 1
            last_saved_episode_idx = dataset.meta.total_episodes - 1

        if events["delete_last_saved"]:
            events["delete_last_saved"] = False
            if last_saved_episode_idx is not None and last_saved_episode_idx not in deletion_queue:
                deletion_queue.append(last_saved_episode_idx)
                log_say(f"Episode {last_saved_episode_idx} queued for deletion")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received — saving and cleaning up...")
        pending_frames = len(dataset.episode_buffer.get("frame_index", []))
        if pending_frames > 0:
            print(f"Discarding incomplete episode ({pending_frames} frames).")
            dataset.clear_episode_buffer()

    finally:
        # Always finalize so the parquet footer is properly written.
        log_say("Stop recording")
        listener.stop()
        if delete_listener is not None:
            delete_listener.stop()
        dataset.finalize()
        if deletion_queue:
            log_say(f"Applying {len(deletion_queue)} queued deletion(s): {sorted(set(deletion_queue))}. Re-encoding now")
            dataset = _apply_queued_deletions(dataset, deletion_queue)
        dataset.push_to_hub()

        try:
            robot.disconnect()
        except DeviceNotConnectedError:
            pass
        try:
            leader_arm.disconnect()
        except DeviceNotConnectedError:
            pass
        try:
            keyboard.disconnect()
        except DeviceNotConnectedError:
            pass


if __name__ == "__main__":
    main()
