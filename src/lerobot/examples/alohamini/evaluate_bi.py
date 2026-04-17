#!/usr/bin/env python3

import argparse
import re
import shutil
import threading
import time
from pathlib import Path

from huggingface_hub import HfApi

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


def _ensure_eval_in_dataset_id(dataset_id: str) -> str:
    """Append '_eval' to a dataset repo id if it is not already present."""
    if "eval" in dataset_id.lower():
        return dataset_id
    return f"{dataset_id}_eval"


def _list_existing_eval_versions(hf_model_id: str) -> list[int]:
    """Return version numbers of existing `<hf_model_id>_eval_vN` datasets on HF Hub."""
    if "/" not in hf_model_id:
        raise ValueError(f"hf_model_id must include org/user prefix: {hf_model_id!r}")
    author, model_name = hf_model_id.split("/", 1)
    pattern = re.compile(
        rf"^{re.escape(author)}/{re.escape(model_name)}_eval_v(\d+)$"
    )
    versions: list[int] = []
    for dataset in HfApi().list_datasets(author=author):
        match = pattern.match(dataset.id)
        if match:
            versions.append(int(match.group(1)))
    return versions


def _next_eval_dataset_id(hf_model_id: str) -> str:
    """Compute the next auto-versioned eval dataset repo id for the given model."""
    versions = _list_existing_eval_versions(hf_model_id)
    next_version = max(versions, default=0) + 1
    return f"{hf_model_id}_eval_v{next_version}"


def _resolve_eval_dataset_id(hf_model_id: str, hf_dataset_id: str | None) -> str:
    """Return the eval dataset repo id: auto-version if none supplied, else enforce '_eval'."""
    if hf_dataset_id is None:
        return _next_eval_dataset_id(hf_model_id)
    return _ensure_eval_in_dataset_id(hf_dataset_id)


def _delete_local_dataset(dataset_root: Path | None) -> None:
    """Remove the local cache directory of a dataset after a successful HF push."""
    if dataset_root is None:
        return
    path = Path(dataset_root)
    if path.exists():
        shutil.rmtree(path)


class _InferenceOnlyDatasetStub:
    """Minimal dataset shim for inference-only mode: provides features/fps, no disk writes."""

    def __init__(self, features: dict, fps: int) -> None:
        self.features = features
        self.fps = fps
        self.image_writer = None

    def add_frame(self, frame: dict) -> None:
        return None


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
    parser.add_argument(
        "--hf_dataset_id",
        type=str,
        default=None,
        help="HuggingFace dataset repo id. If omitted, auto-versioned as <hf_model_id>_eval_vN.",
    )
    parser.add_argument("--remote_ip", type=str, default="10.139.203.203", help="LeKiwi host IP address")
    parser.add_argument("--robot_id", type=str, default="lekiwi", help="Robot ID")
    parser.add_argument(
        "--disable_rerun",
        action="store_true",
        help="Disable rerun visualization to avoid native viewer crashes on macOS.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Override number of denoising steps at inference (diffusion only). E.g. 10 for fast DDIM.",
    )
    parser.add_argument(
        "--noise_scheduler_type",
        type=str,
        default=None,
        choices=["DDPM", "DDIM"],
        help="Override noise scheduler type at inference (diffusion only). Use DDIM for faster inference.",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=None,
        help="Override number of action steps executed per policy call (diffusion only).",
    )
    parser.add_argument(
        "--keep_local_dataset",
        action="store_true",
        help="Keep the local eval dataset cache after a successful HuggingFace push (default: delete).",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record rollouts to a dataset and push to HF. Default: inference-only, no recording.",
    )

    args = parser.parse_args()
    if args.record:
        args.hf_dataset_id = _resolve_eval_dataset_id(args.hf_model_id, args.hf_dataset_id)
        print(f"[info] Recording enabled. Using hf_dataset_id: {args.hf_dataset_id}")
    else:
        print("[info] Inference-only mode (no recording). Pass --record to save rollouts.")

    # === Robot config ===
    robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id=args.robot_id)
    robot = LeKiwiClient(robot_config)
    robot.connect()

    # === Policy ===
    policy_config = PreTrainedConfig.from_pretrained(args.hf_model_id)
    if args.num_inference_steps is not None and hasattr(policy_config, "num_inference_steps"):
        policy_config.num_inference_steps = args.num_inference_steps
    if args.noise_scheduler_type is not None and hasattr(policy_config, "noise_scheduler_type"):
        policy_config.noise_scheduler_type = args.noise_scheduler_type
    if args.n_action_steps is not None and hasattr(policy_config, "n_action_steps"):
        policy_config.n_action_steps = args.n_action_steps
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

    if args.record:
        dataset = LeRobotDataset.create(
            repo_id=args.hf_dataset_id,
            fps=args.fps,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
        )
        dataset_stats = dataset.meta.stats
    else:
        dataset = _InferenceOnlyDatasetStub(features=dataset_features, fps=args.fps)
        dataset_stats = None

    # === Policy Processors ===
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.hf_model_id,
        dataset_stats=dataset_stats,
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
        action_word = "recording" if args.record else "running"
        log_say(f"Running inference, {action_word} episode {recorded_episodes + 1} of {args.num_episodes}")

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
            if args.record:
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

    log_say("Stop")
    listener.stop()
    if args.record:
        run_task_with_live_preview(
            task=dataset.finalize,
            robot=robot,
            fps=args.fps,
            display_data=display_data,
            robot_observation_processor=robot_observation_processor,
        )
        dataset_root = getattr(dataset, "root", None)
        run_task_with_live_preview(
            task=dataset.push_to_hub,
            robot=robot,
            fps=args.fps,
            display_data=display_data,
            robot_observation_processor=robot_observation_processor,
        )
        if not args.keep_local_dataset:
            _delete_local_dataset(dataset_root)
            print(f"[info] Deleted local eval dataset: {dataset_root}")
    robot.disconnect()


if __name__ == "__main__":
    main()
