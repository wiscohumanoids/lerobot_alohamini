#!/usr/bin/env python3

import argparse
import os
import time

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.robots.alohamini.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.alohamini.lekiwi_client import LeKiwiClient
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


CONDITION_NAMES = (
    "primitive_pick",
    "primitive_place",
    "target_u_norm",
    "target_v_norm",
    "target_piece_any",
    "target_piece_o",
    "target_piece_x",
    "place_cell_1",
    "place_cell_2",
    "place_cell_3",
    "place_cell_4",
    "place_cell_5",
    "place_cell_6",
    "place_cell_7",
    "place_cell_8",
    "place_cell_9",
)


def cell_id_to_normalized_center(cell_id: int, rows: int, cols: int) -> tuple[float, float]:
    if not 1 <= cell_id <= rows * cols:
        raise ValueError(f"cell_id must be in [1, {rows * cols}], got {cell_id}")

    zero_based = cell_id - 1
    row = zero_based // cols
    col = zero_based % cols
    u = (col + 0.5) / cols
    v = (row + 0.5) / rows
    return float(u), float(v)


def build_condition_vector(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    vec = np.zeros(len(CONDITION_NAMES), dtype=np.float32)

    if args.primitive == "pick":
        vec[0] = 1.0
        u, v = cell_id_to_normalized_center(args.pick_cell_id, args.grid_rows, args.grid_cols)
        vec[2] = u
        vec[3] = v

        if args.piece_type == "any":
            vec[4] = 1.0
        elif args.piece_type == "o":
            vec[5] = 1.0
        else:
            vec[6] = 1.0

        task = (
            f"Pick the {args.piece_type} block from image grid cell {args.pick_cell_id} "
            f"on a {args.grid_rows}x{args.grid_cols} grid."
        )
    else:
        vec[1] = 1.0
        vec[7 + (args.place_cell_id - 1)] = 1.0
        task = f"Place the held block into tic-tac-toe board cell {args.place_cell_id}."

    return vec, task


def condition_dict(vector: np.ndarray) -> dict[str, float]:
    return {name: float(vector[idx]) for idx, name in enumerate(CONDITION_NAMES)}


def save_episode_with_preview(
    dataset: LeRobotDataset,
    robot: LeKiwiClient,
    fps: int,
    robot_observation_processor,
    display_data: bool,
) -> None:
    while True:
        try:
            dataset.save_episode()
            return
        except RuntimeError:
            # Allow the writer thread to drain while keeping the preview alive.
            loop_start = time.perf_counter()
            if display_data:
                obs = robot.get_observation()
                obs_processed = robot_observation_processor(obs)
                log_rerun_data(observation=obs_processed, action=None, compress_images=True)
            precise_sleep(max(1.0 / fps - (time.perf_counter() - loop_start), 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Record conditioned pick/place demonstrations for AlohaMini.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset repo_id, e.g. user/tictactoe_pick")
    parser.add_argument("--primitive", type=str, required=True, choices=["pick", "place"])
    parser.add_argument("--piece_type", type=str, default="any", choices=["any", "o", "x"])
    parser.add_argument("--pick_cell_id", type=int, default=1, help="Image-grid cell for pick target.")
    parser.add_argument("--grid_rows", type=int, default=8, help="Rows in the pick target image grid.")
    parser.add_argument("--grid_cols", type=int, default=8, help="Cols in the pick target image grid.")
    parser.add_argument("--place_cell_id", type=int, default=5, choices=list(range(1, 10)))
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode_time", type=int, default=12)
    parser.add_argument("--reset_time", type=int, default=5)
    parser.add_argument("--setup_time", type=int, default=10)
    parser.add_argument("--remote_ip", type=str, default="10.139.203.203")
    parser.add_argument("--robot_id", type=str, default="lekiwi_host")
    parser.add_argument("--leader_id", type=str, default="so101_leader_bi")
    parser.add_argument(
        "--arm_profile",
        type=str,
        default="so-arm-5dof",
        choices=["so-arm-5dof", "am-arm-6dof"],
    )
    parser.add_argument("--keyboard", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--disable_rerun", action="store_true")
    parser.add_argument("--rerun_port", type=int, default=9091)
    parser.add_argument("--vcodec", type=str, default="h264", choices=["h264", "hevc", "libsvtav1"])
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    args = parser.parse_args()

    cond_vector, task_description = build_condition_vector(args)
    cond_values = condition_dict(cond_vector)

    robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id=args.robot_id, no_keyboard=not args.keyboard)
    leader_config = BiSOLeaderConfig(
        left_arm_config=SOLeaderConfig(port="/dev/ttyACM0", arm_profile=args.arm_profile),
        right_arm_config=SOLeaderConfig(port="/dev/ttyACM1", arm_profile=args.arm_profile),
        id=args.leader_id,
    )

    robot = LeKiwiClient(robot_config)
    leader = BiSOLeader(leader_config)
    keyboard = KeyboardTeleop(KeyboardTeleopConfig())

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    obs_features[OBS_ENV_STATE] = {
        "dtype": "float32",
        "shape": (len(CONDITION_NAMES),),
        "names": CONDITION_NAMES,
    }
    dataset_features = {**action_features, **obs_features}

    if args.resume:
        dataset = LeRobotDataset(args.dataset, vcodec=args.vcodec)
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

    robot.connect()
    leader.connect()
    keyboard.connect()

    listener, events = init_keyboard_listener()
    display_data = not args.disable_rerun
    if display_data:
        os.environ["LEROBOT_RERUN_MEMORY_LIMIT"] = "0%"
        os.environ["RERUN_FLUSH_NUM_BYTES"] = "0"
        init_rerun(session_name="lekiwi_conditioned_record", ip="localhost", port=args.rerun_port)

    print("Condition vector:")
    for name, value in cond_values.items():
        print(f"  {name}: {value:.3f}")
    print(f"Task: {task_description}")

    log_say(f"Setup time: {args.setup_time} seconds")
    for remaining in range(args.setup_time, 0, -1):
        if events["stop_recording"]:
            break
        print(f"\rStarting in {remaining}s... ", end="", flush=True)
        loop_start = time.perf_counter()
        if display_data:
            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)
            log_rerun_data(observation=obs_processed, action=None, compress_images=True)
        precise_sleep(max(1.0 - (time.perf_counter() - loop_start), 0.0))
    print()

    recorded = 0
    try:
        while recorded < args.num_episodes and not events["stop_recording"]:
            log_say(f"Recording conditioned episode {recorded + 1} of {args.num_episodes}")
            start_episode_t = time.perf_counter()

            while (time.perf_counter() - start_episode_t) < args.episode_time:
                loop_start = time.perf_counter()
                if events["exit_early"]:
                    events["exit_early"] = False
                    break

                obs = robot.get_observation()
                obs_processed = robot_observation_processor(obs)
                obs_with_condition = {**obs_processed, **cond_values}
                observation_frame = build_dataset_frame(dataset.features, obs_with_condition, prefix=OBS_STR)

                arm_action = leader.get_action()
                arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
                keyboard_action = keyboard.get_action() if args.keyboard else {}
                base_action = robot._from_keyboard_to_base_action(keyboard_action)
                lift_action = robot._from_keyboard_to_lift_action(keyboard_action)
                teleop_action = {**arm_action, **base_action, **lift_action}

                processed_action = teleop_action_processor((teleop_action, obs))
                robot_action = robot_action_processor((processed_action, obs))
                robot.send_action(robot_action)

                action_frame = build_dataset_frame(dataset.features, processed_action, prefix=ACTION)
                dataset.add_frame({**observation_frame, **action_frame, "task": task_description})

                if display_data:
                    log_rerun_data(observation=obs_processed, action=processed_action, compress_images=True)

                precise_sleep(max(1.0 / args.fps - (time.perf_counter() - loop_start), 0.0))

            if events["rerecord_episode"]:
                events["rerecord_episode"] = False
                dataset.clear_episode_buffer()
                log_say("Re-recording episode")
                continue

            pending_frames = len(dataset.episode_buffer.get("frame_index", []))
            if pending_frames == 0:
                continue

            save_episode_with_preview(
                dataset=dataset,
                robot=robot,
                fps=args.fps,
                robot_observation_processor=robot_observation_processor,
                display_data=display_data,
            )
            recorded += 1

            if recorded < args.num_episodes and args.reset_time > 0:
                log_say("Reset the scene")
                reset_until = time.perf_counter() + args.reset_time
                while time.perf_counter() < reset_until and not events["stop_recording"]:
                    loop_start = time.perf_counter()
                    if display_data:
                        obs = robot.get_observation()
                        obs_processed = robot_observation_processor(obs)
                        log_rerun_data(observation=obs_processed, action=None, compress_images=True)
                    precise_sleep(max(1.0 / args.fps - (time.perf_counter() - loop_start), 0.0))
    finally:
        log_say("Finalizing dataset")
        if listener is not None:
            listener.stop()
        dataset.finalize()
        if args.push_to_hub:
            dataset.push_to_hub()
        keyboard.disconnect()
        leader.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
