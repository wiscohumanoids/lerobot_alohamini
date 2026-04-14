#!/usr/bin/env python3

import argparse
import time

import numpy as np
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.robots.alohamini.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.alohamini.lekiwi_client import LeKiwiClient
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener, predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, log_say
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
    return float((col + 0.5) / cols), float((row + 0.5) / rows)


def build_condition_vector(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    vec = np.zeros(len(CONDITION_NAMES), dtype=np.float32)
    if args.primitive == "pick":
        vec[0] = 1.0
        vec[2], vec[3] = cell_id_to_normalized_center(args.pick_cell_id, args.grid_rows, args.grid_cols)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a conditioned ACT policy on the real AlohaMini robot.")
    parser.add_argument("--model_path", type=str, required=True, help="Local path or HF repo for the trained policy.")
    parser.add_argument("--primitive", type=str, required=True, choices=["pick", "place"])
    parser.add_argument("--piece_type", type=str, default="any", choices=["any", "o", "x"])
    parser.add_argument("--pick_cell_id", type=int, default=1)
    parser.add_argument("--grid_rows", type=int, default=8)
    parser.add_argument("--grid_cols", type=int, default=8)
    parser.add_argument("--place_cell_id", type=int, default=5, choices=list(range(1, 10)))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode_time", type=int, default=12)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--remote_ip", type=str, default="10.139.203.203")
    parser.add_argument("--robot_id", type=str, default="lekiwi_host")
    parser.add_argument("--disable_rerun", action="store_true")
    args = parser.parse_args()

    cond_vector, task_description = build_condition_vector(args)
    cond_values = condition_dict(cond_vector)

    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=args.remote_ip, id=args.robot_id))
    robot.connect()

    policy_config = PreTrainedConfig.from_pretrained(args.model_path)
    policy_class = get_policy_class(policy_config.type)
    policy = policy_class.from_pretrained(args.model_path, config=policy_config)
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=args.model_path)
    policy.eval()
    policy.to(get_safe_torch_device(policy.config.device))

    _, robot_action_processor, robot_observation_processor = make_default_processors()
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    obs_features[OBS_ENV_STATE] = {
        "dtype": "float32",
        "shape": (len(CONDITION_NAMES),),
        "names": CONDITION_NAMES,
    }
    dataset_features = {**action_features, **obs_features}

    listener, events = init_keyboard_listener()
    if not args.disable_rerun:
        init_rerun(session_name="lekiwi_conditioned_eval", ip="127.0.0.1", port=9091)

    try:
        for episode_idx in range(args.num_episodes):
            if events["stop_recording"]:
                break
            log_say(f"Running conditioned episode {episode_idx + 1} of {args.num_episodes}")
            policy.reset()
            start_t = time.perf_counter()

            while (time.perf_counter() - start_t) < args.episode_time:
                loop_start = time.perf_counter()
                if events["exit_early"]:
                    events["exit_early"] = False
                    break

                obs = robot.get_observation()
                obs_processed = robot_observation_processor(obs)
                obs_with_condition = {**obs_processed, **cond_values}
                observation_frame = build_dataset_frame(dataset_features, obs_with_condition, prefix=OBS_STR)

                action_values = predict_action(
                    observation=observation_frame,
                    policy=policy,
                    device=get_safe_torch_device(policy.config.device),
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    task=task_description,
                    robot_type=robot.robot_type,
                )

                action_dict = make_robot_action(action_values, dataset_features)
                robot_action = robot_action_processor((action_dict, obs))
                robot.send_action(robot_action)

                if not args.disable_rerun:
                    log_rerun_data(observation=obs_processed, action=action_dict, compress_images=True)

                precise_sleep(max(1.0 / args.fps - (time.perf_counter() - loop_start), 0.0))
    finally:
        if listener is not None:
            listener.stop()
        robot.disconnect()


if __name__ == "__main__":
    main()
