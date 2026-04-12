#!/usr/bin/env python3

import argparse
import threading
import time

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


def log_live_preview_frame(robot: LeKiwiClient, robot_observation_processor) -> None:
    observation = robot.get_observation()
    observation_processed = robot_observation_processor(observation)
    log_rerun_data(observation=observation_processed, action=None, compress_images=True)


def run_task_with_live_preview(
    task,
    robot: LeKiwiClient,
    fps: int,
    display_data: bool,
    robot_observation_processor,
) -> None:
    task_done = threading.Event()
    task_error = {"exception": None}

    def _run_task() -> None:
        try:
            task()
        except Exception as exc:
            task_error["exception"] = exc
        finally:
            task_done.set()

    task_thread = threading.Thread(target=_run_task, daemon=True)
    task_thread.start()

    while not task_done.is_set():
        loop_start = time.perf_counter()
        if display_data:
            log_live_preview_frame(robot, robot_observation_processor)

        dt_s = time.perf_counter() - loop_start
        sleep_s = max(1.0 / fps - dt_s, 0.0)
        task_done.wait(timeout=sleep_s)

    task_thread.join()
    if task_error["exception"] is not None:
        raise task_error["exception"]


def wait_for_reset(
    robot: LeKiwiClient,
    events: dict,
    reset_time_s: int,
    fps: int,
    display_data: bool,
    robot_observation_processor,
) -> None:
    if reset_time_s <= 0:
        return

    log_say(f"Reset the environment. Waiting {reset_time_s} seconds before the next episode.")
    end_time = time.perf_counter() + reset_time_s

    while time.perf_counter() < end_time:
        if events["stop_recording"] or events["exit_early"]:
            events["exit_early"] = False
            break

        loop_start = time.perf_counter()
        if display_data:
            log_live_preview_frame(robot, robot_observation_processor)

        remaining_s = end_time - time.perf_counter()
        dt_s = time.perf_counter() - loop_start
        precise_sleep(min(max(1.0 / fps - dt_s, 0.0), max(remaining_s, 0.0)))


def main():
    parser = argparse.ArgumentParser(description="Evaluate LeKiwi Robot with a pretrained policy")
    parser.add_argument("--num_episodes", type=int, default=2, help="Number of episodes to record")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--episode_time", type=int, default=60, help="Duration of each episode in seconds")
    parser.add_argument("--reset_time", type=int, default=10, help="Idle reset time between episodes in seconds")
    parser.add_argument("--task_description", type=str, default="My task description", help="Description of the task")
    parser.add_argument("--hf_model_id", type=str, required=True, help="HuggingFace model repo id")
    parser.add_argument("--hf_dataset_id", type=str, required=True, help="HuggingFace dataset repo id")
    parser.add_argument("--remote_ip", type=str, default="127.0.0.1", help="LeKiwi host IP address")
    parser.add_argument("--robot_id", type=str, default="lekiwi", help="Robot ID")
    parser.add_argument(
        "--disable_rerun",
        action="store_true",
        help="Disable rerun visualization to avoid native viewer crashes on macOS.",
    )

    args = parser.parse_args()

    # === Robot config ===
    robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id=args.robot_id)
    robot = LeKiwiClient(robot_config)
    robot.connect()

    # === Policy ===
    policy_config = PreTrainedConfig.from_pretrained(args.hf_model_id)
    policy_class = get_policy_class(policy_config.type)
    if policy_config.use_peft:
        from peft import PeftConfig, PeftModel  # type: ignore[reportMissingImports]

        peft_config = PeftConfig.from_pretrained(args.hf_model_id)
        policy = policy_class.from_pretrained(
            pretrained_name_or_path=peft_config.base_model_name_or_path,
            config=policy_config,
        )
        policy = PeftModel.from_pretrained(policy, args.hf_model_id, config=peft_config)
    else:
        policy = policy_class.from_pretrained(args.hf_model_id, config=policy_config)

    # === Dataset features ===
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id=args.hf_dataset_id,
        fps=args.fps,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # === Policy Processors ===
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.hf_model_id,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    listener, events = init_keyboard_listener()
    display_data = not args.disable_rerun
    if display_data:
        init_rerun(session_name="lekiwi_evaluate", ip="127.0.0.1", port=9091)

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    print("Starting evaluate loop...")
    recorded_episodes = 0

    while recorded_episodes < args.num_episodes and not events["stop_recording"]:
        log_say(f"Running inference, recording eval episode {recorded_episodes + 1} of {args.num_episodes}")

        record_loop(
            robot=robot,
            events=events,
            fps=args.fps,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=dataset,
            control_time_s=args.episode_time,
            single_task=args.task_description,
            display_data=display_data,
            display_compressed_images=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        if not events["stop_recording"]:
            run_task_with_live_preview(
                task=dataset.save_episode,
                robot=robot,
                fps=args.fps,
                display_data=display_data,
                robot_observation_processor=robot_observation_processor,
            )
            recorded_episodes += 1

            if recorded_episodes < args.num_episodes:
                wait_for_reset(
                    robot=robot,
                    events=events,
                    reset_time_s=args.reset_time,
                    fps=args.fps,
                    display_data=display_data,
                    robot_observation_processor=robot_observation_processor,
                )

    log_say("Stop recording")
    listener.stop()
    run_task_with_live_preview(
        task=dataset.finalize,
        robot=robot,
        fps=args.fps,
        display_data=display_data,
        robot_observation_processor=robot_observation_processor,
    )
    run_task_with_live_preview(
        task=dataset.push_to_hub,
        robot=robot,
        fps=args.fps,
        display_data=display_data,
        robot_observation_processor=robot_observation_processor,
    )
    robot.disconnect()


if __name__ == "__main__":
    main()
