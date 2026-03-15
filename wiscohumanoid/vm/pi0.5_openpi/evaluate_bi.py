#!/usr/bin/env python3

from __future__ import annotations

import argparse
from typing import Any
import numpy as np
import requests

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.remote_http.configuration_remote_http import RemoteHTTPPolicyConfig
from lerobot.policies.remote_http_policy import RemoteHTTPPolicy
from lerobot.processor import make_default_processors
from lerobot.processor.converters import (
    observation_to_transition,
    policy_action_to_transition,
    transition_to_observation,
    transition_to_policy_action,
)
from lerobot.processor.pipeline import (
    ActionProcessorStep,
    DataProcessorPipeline,
    ObservationProcessorStep,
    RobotProcessorPipeline,
)
from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun


def _expected_api16_state_order() -> list[str]:
    # OpenPI AlohaMini API16 (dataset/robot interface) order.
    return [
        "arm_left_shoulder_pan.pos",
        "arm_left_shoulder_lift.pos",
        "arm_left_elbow_flex.pos",
        "arm_left_wrist_flex.pos",
        "arm_left_wrist_roll.pos",
        "arm_left_gripper.pos",
        "arm_right_shoulder_pan.pos",
        "arm_right_shoulder_lift.pos",
        "arm_right_elbow_flex.pos",
        "arm_right_wrist_flex.pos",
        "arm_right_wrist_roll.pos",
        "arm_right_gripper.pos",
        "x.vel",
        "y.vel",
        "theta.vel",
        "lift_axis.height_mm",
    ]


def _validate_state_action_order(
    dataset_features: dict[str, dict],
    requested_dof: int | None = None,
) -> tuple[list[str], int]:
    expected_api16 = _expected_api16_state_order()

    state_ft = dataset_features.get("observation.state")
    action_ft = dataset_features.get("action")
    if not isinstance(state_ft, dict) or not isinstance(action_ft, dict):
        raise ValueError("Missing required dataset features: 'observation.state' and/or 'action'.")

    state_names = list(state_ft.get("names") or [])
    action_names = list(action_ft.get("names") or [])

    if state_names != action_names:
        raise ValueError(
            "State/action name order mismatch.\n"
            f"- observation.state.names: {state_names}\n"
            f"- action.names: {action_names}\n"
        )

    state_dim = int(state_ft.get("shape", (0,))[0])
    action_dim = int(action_ft.get("shape", (0,))[0])

    if requested_dof is not None and requested_dof not in (16, 18):
        raise ValueError(f"requested_dof must be 16 or 18, got {requested_dof}")

    if state_dim != action_dim:
        raise ValueError(f"State/action dim mismatch: state={state_dim}, action={action_dim}")
    if state_dim not in (16, 18):
        raise ValueError(f"Only 16/18-DoF are supported, got dim={state_dim}")
    if requested_dof is not None and state_dim != requested_dof:
        raise ValueError(f"Requested {requested_dof}-DoF but robot/dataset features report {state_dim}-DoF")

    if state_dim == 16 and state_names != expected_api16:
        raise ValueError(
            "State/action order does not match OpenPI AlohaMini API16.\n"
            f"- expected: {expected_api16}\n"
            f"- got:      {state_names}\n"
        )

    if not state_names:
        state_names = [f"joint_{i}" for i in range(state_dim)]

    return state_names, state_dim


def _print_camera_info(robot: LeKiwiClient, dataset_features: dict[str, dict], obs_sample: dict[str, Any]) -> None:
    # Dataset feature declared shapes (from robot config).
    for cam in ("head_top", "wrist_left", "wrist_right"):
        k = f"observation.images.{cam}"
        if k in dataset_features:
            print(f"[camera] feature {k} shape={dataset_features[k].get('shape')} dtype={dataset_features[k].get('dtype')}")
        else:
            print(f"[camera] feature {k} missing in dataset_features")

    # Live frames (from robot observation).
    for cam in ("head_top", "wrist_left", "wrist_right"):
        v = obs_sample.get(cam)
        if isinstance(v, np.ndarray):
            print(f"[camera] live {cam} shape={v.shape} dtype={v.dtype}")
        else:
            print(f"[camera] live {cam} missing or not ndarray (type={type(v).__name__})")

    # Mapping sanity (LeRobot -> OpenPI expected camera names)
    mapping = {"head_top": "cam_high", "wrist_left": "cam_left_wrist", "wrist_right": "cam_right_wrist"}
    print(f"[camera] lerobot->openpi mapping: {mapping}")


def _maybe_print_server_metadata(server_url: str, timeout_s: float = 2.0) -> None:
    try:
        resp = requests.get(f"{server_url.rstrip('/')}/metadata", timeout=timeout_s)
        resp.raise_for_status()
        meta = resp.json()
        if isinstance(meta, dict):
            keys = sorted(meta.keys())
            print(f"[server] /metadata keys={keys}")
            # Avoid printing huge blobs.
            for k in ("name", "config", "model", "action_horizon", "action_dim", "robot"):
                if k in meta:
                    print(f"[server] {k}={meta[k]}")
            # Print action normalization info if available
            if "config" in meta and isinstance(meta["config"], dict):
                config = meta["config"]
                if "output_transforms" in config:
                    print(f"[server] output_transforms={config['output_transforms']}")
        else:
            print(f"[server] /metadata returned non-dict: {type(meta).__name__}")
    except Exception as e:
        print(f"[server] failed to fetch /metadata from {server_url!r}: {e}")


def _validate_action_values(
    action_tensor: Any,
    action_dict: dict[str, Any],
    dataset_features: dict[str, dict],
    expected_order: list[str],
    expected_action_dim: int,
    frame_idx: int = 0,
    log_every_n: int = 30,
) -> None:
    """Validate action values: dimension, range, and key matching."""
    import torch

    # Convert to numpy for validation
    if isinstance(action_tensor, torch.Tensor):
        action_array = action_tensor.detach().cpu().numpy()
    else:
        action_array = np.asarray(action_tensor)

    # Flatten if needed
    if action_array.ndim > 1:
        action_array = action_array.flatten()

    # Check dimension
    if len(action_array) != expected_action_dim:
        print(f"[action_validation] ERROR: Action dim mismatch: expected {expected_action_dim}, got {len(action_array)}")
        return

    # Check for NaN/Inf
    if np.any(np.isnan(action_array)) or np.any(np.isinf(action_array)):
        print(f"[action_validation] ERROR: Action contains NaN or Inf values!")
        print(f"[action_validation] action_array={action_array}")

    # Check value ranges (only log periodically to avoid spam)
    if frame_idx % log_every_n == 0:
        action_min = float(np.min(action_array))
        action_max = float(np.max(action_array))
        action_mean = float(np.mean(np.abs(action_array)))

        # Expected ranges (approximate)
        # Joint angles: typically ±π (≈ ±3.14) or ±180°
        # Gripper: typically [0, 1] or specific range
        # Base velocity: typically small (e.g., ±0.5 m/s)
        # Lift: typically in mm, depends on robot

        if expected_action_dim == 16:
            joint_angles = action_array[:12]
            base_vel = action_array[12:15]
            lift = action_array[15:16]
        elif expected_action_dim == 18:
            joint_angles = action_array[:14]
            base_vel = action_array[14:17]
            lift = action_array[17:18]
        else:
            raise ValueError(f"Unsupported action dim for validation: {expected_action_dim}. Expected 16 or 18.")

        warnings = []
        critical_errors = []

        # Check for extremely large values (likely unnormalized or wrong scale)
        if action_max > 100.0 or action_min < -100.0:
            critical_errors.append(
                f"CRITICAL: Action values extremely large (range=[{action_min:.2f}, {action_max:.2f}]). "
                f"This suggests actions may not be properly denormalized or have wrong scale."
            )

        if joint_angles.size and np.any(np.abs(joint_angles) > 10.0):  # > ~3π, clearly abnormal
            warnings.append(f"Joint angles out of range: min={joint_angles.min():.2f}, max={joint_angles.max():.2f}")
        if base_vel.size >= 2 and np.any(np.abs(base_vel[:2]) > 5.0):  # x, y velocity > 5 m/s, clearly abnormal
            warnings.append(f"Base x/y velocity out of range: {base_vel[:2]}")
        if lift.size and np.any(np.abs(lift) > 1000.0):  # Lift > 1000mm, likely abnormal
            warnings.append(f"Lift axis out of range: {lift}")

        if critical_errors:
            print(f"[action_validation] ERROR (frame {frame_idx}):")
            for err in critical_errors:
                print(f"  {err}")
            print(f"  Full action array: {action_array}")
            print(f"  SUGGESTION: Check if server returns normalized actions that need denormalization.")
            print(f"  SUGGESTION: Use --training_dataset_id to load stats for denormalization.")

        if warnings:
            print(f"[action_validation] WARNING (frame {frame_idx}):")
            for w in warnings:
                print(f"  {w}")
            print(f"  Full action: min={action_min:.2f}, max={action_max:.2f}, mean_abs={action_mean:.2f}")

    # Check key matching
    action_names = dataset_features.get("action", {}).get("names", [])
    if len(action_names) == len(expected_order):
        for i, (name, expected) in enumerate(zip(action_names, expected_order)):
            if name != expected:
                if frame_idx == 0:  # Only log once
                    print(f"[action_validation] WARNING: Action name mismatch at index {i}: got '{name}', expected '{expected}'")
    else:
        if frame_idx == 0:
            print(f"[action_validation] WARNING: Action names length mismatch: got {len(action_names)}, expected {len(expected_order)}")

    # Check action_dict keys match robot._state_order
    if frame_idx == 0:
        missing_keys = []
        for key in expected_order:
            if key not in action_dict:
                missing_keys.append(key)
        if missing_keys:
            print(f"[action_validation] WARNING: Missing action keys in dict: {missing_keys}")


def _noop_preprocessor() -> DataProcessorPipeline[dict[str, Any], dict[str, Any]]:
    # Keeps batch dict as-is (preserves 'task' string).
    return DataProcessorPipeline(steps=[], name="noop_preprocessor")


def _noop_postprocessor() -> DataProcessorPipeline[Any, Any]:
    # Identity for PolicyAction (torch.Tensor) by using the PolicyAction converters.
    return DataProcessorPipeline(
        steps=[],
        name="noop_postprocessor",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )


def _maybe_bgr_to_rgb(obs: dict[str, Any], camera_keys: tuple[str, ...]) -> dict[str, Any]:
    out = dict(obs)
    for k in camera_keys:
        img = out.get(k)
        if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[-1] == 3 and img.dtype == np.uint8:
            # OpenCV decode returns BGR by default.
            out[k] = img[..., ::-1].copy()
    return out


class BGRToRGBProcessorStep(ObservationProcessorStep):
    """Convert BGR images to RGB for OpenPI compatibility (training uses RGB from torchvision VideoReader)."""

    def __init__(self, camera_keys: tuple[str, ...] = ("head_top", "wrist_left", "wrist_right")):
        self.camera_keys = camera_keys

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        return _maybe_bgr_to_rgb(observation, self.camera_keys)

    def transform_features(
        self, features: dict[str, Any]
    ) -> dict[str, Any]:
        # No feature shape changes, just color channel reordering.
        return features


class ActionDenormalizeProcessorStep(ActionProcessorStep):
    """Denormalize actions using training dataset statistics.

    This step checks if actions appear to be in normalized range (e.g., [-1, 1] or [0, 1])
    and denormalizes them using dataset statistics if training_stats are provided.
    """

    def __init__(
        self,
        training_stats: dict[str, dict[str, Any]] | None = None,
        action_key: str = "action",
    ):
        self.training_stats = training_stats
        self.action_key = action_key
        self._has_warned = False

    def action(self, action: Any) -> Any:
        import torch

        if self.training_stats is None or self.action_key not in self.training_stats:
            return action

        # Convert to numpy
        if isinstance(action, torch.Tensor):
            action_array = action.detach().cpu().numpy()
        else:
            action_array = np.asarray(action)

        # Flatten if needed
        original_shape = action_array.shape
        if action_array.ndim > 1:
            action_array = action_array.flatten()

        # Check if action appears to be in normalized range
        action_min = float(np.min(action_array))
        action_max = float(np.max(action_array))
        action_mean_abs = float(np.mean(np.abs(action_array)))

        # Heuristic: if values are mostly in [-2, 2] range, might be normalized
        # But if values are very large (hundreds/thousands), likely already denormalized or wrong
        is_likely_normalized = (
            action_mean_abs < 2.0
            and action_min > -5.0
            and action_max < 5.0
            and not self._has_warned
        )

        if is_likely_normalized:
            stats = self.training_stats[self.action_key]
            if "mean" in stats and "std" in stats:
                mean = stats["mean"]
                std = stats["std"]

                # Convert to numpy if needed
                if isinstance(mean, torch.Tensor):
                    mean = mean.numpy()
                if isinstance(std, torch.Tensor):
                    std = std.numpy()

                # Ensure compatible shapes
                if mean.ndim > 1:
                    mean = mean.flatten()
                if std.ndim > 1:
                    std = std.flatten()

                # Pad or slice to match action dimension
                action_dim = len(action_array)
                stats_dim = len(mean)
                if action_dim > stats_dim:
                    # Pad with zeros (mean=0, std=1 for extra dims)
                    mean = np.concatenate([mean, np.zeros(action_dim - stats_dim)])
                    std = np.concatenate([std, np.ones(action_dim - stats_dim)])
                elif action_dim < stats_dim:
                    # Slice to match
                    mean = mean[:action_dim]
                    std = std[:action_dim]

                # Denormalize: x = x_norm * std + mean
                action_denorm = action_array * (std + 1e-6) + mean
                action_array = action_denorm

                if not self._has_warned:
                    print(
                        f"[action_denormalize] Detected normalized actions (range=[{action_min:.3f}, {action_max:.3f}]), "
                        f"denormalizing using training stats. "
                        f"After denorm: range=[{action_array.min():.3f}, {action_array.max():.3f}]"
                    )
                    self._has_warned = True

        # Reshape back to original shape
        action_array = action_array.reshape(original_shape)

        # Convert back to tensor if input was tensor
        if isinstance(action, torch.Tensor):
            return torch.from_numpy(action_array).to(action.device).to(action.dtype)
        return action_array

    def transform_features(self, features: dict[str, Any]) -> dict[str, Any]:
        # No feature shape changes, just value scaling.
        return features


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LeKiwi (AlohaMini) robot with a policy (local or remote).",
        epilog=(
            "  python examples/alohamini/evaluate_bi.py --policy_mode remote_http --server_url http://127.0.0.1:8000 \\\n"
            "    --hf_dataset_id <your_hf_user>/<your_eval_dataset> --task_description \"pickup the rubbish\" \\\n"
            "    --remote_ip 127.0.0.1\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--num_episodes", type=int, default=2, help="Number of episodes to record")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--episode_time", type=int, default=360, help="Duration of each episode in seconds")
    parser.add_argument("--task_description", type=str, default="My task description", help="Description of the task")
    parser.add_argument(
        "--policy_mode",
        type=str,
        default="local_act",
        choices=("local_act", "remote_http"),
        help="Policy mode: local_act (HF ACTPolicy) or remote_http (OpenPI HTTP server).",
    )
    parser.add_argument("--server_url", type=str, default="", help="Remote policy server base URL, e.g. http://127.0.0.1:8000")
    parser.add_argument(
        "--bgr_to_rgb",
        action="store_true",
        default=False,
        help=(
            "Convert OpenCV-decoded BGR images to RGB before inference (default: False). "
            "NOTE: LeRobot datasets recorded with OpenCV (like pick_up_merged) save BGR images "
            "that are incorrectly treated as RGB. Training decodes these as BGR (treated as RGB). "
            "So inference should also send BGR (no conversion) to match training. "
            "Only enable this if your training dataset was recorded with explicit BGR→RGB conversion."
        ),
    )
    parser.add_argument("--hf_model_id", type=str, default="", help="HuggingFace model repo id (required for local_act)")
    parser.add_argument(
        "--hf_dataset_id",
        type=str,
        default="",
        help=(
            "HuggingFace dataset repo id for saving evaluation data (e.g. 'username/eval_dataset'). "
            "If not provided, a local temporary dataset will be created (data saved locally, not pushed to Hub). "
            "Required for policy inference to get dataset features."
        ),
    )
    parser.add_argument("--remote_ip", type=str, default="127.0.0.1", help="LeKiwi host IP address")
    parser.add_argument("--robot_id", type=str, default="lekiwi", help="Robot ID")
    parser.add_argument(
        "--debug_actions",
        action="store_true",
        default=False,
        help="Enable detailed action debugging: log action values, validate ranges, and check for anomalies.",
    )
    parser.add_argument(
        "--training_dataset_id",
        type=str,
        default="",
        help=(
            "Training dataset repo_id (e.g. 'username/pick_up_merged') for loading action statistics. "
            "If provided and actions appear normalized, will attempt to denormalize using dataset stats."
        ),
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        default=False,
        help=(
            "Run inference only without saving evaluation data to disk (no dataset created, no frames/videos written). "
            "Use when disk space is limited or only robot execution is needed. With local_act, --hf_dataset_id is required for normalization stats."
        ),
    )
    parser.add_argument(
        "--robot_dof",
        type=str,
        default="auto",
        choices=("auto", "16", "18"),
        help="Robot action/state DoF. auto=read from robot dataset features (default).",
    )

    args = parser.parse_args()

    if args.policy_mode == "local_act" and not args.hf_model_id:
        raise ValueError("--hf_model_id is required when --policy_mode=local_act")
    if args.policy_mode == "remote_http" and not args.server_url:
        raise ValueError("--server_url is required when --policy_mode=remote_http")
    if args.no_save and args.policy_mode == "local_act" and not args.hf_dataset_id:
        raise ValueError(
            "When using --no_save with --policy_mode=local_act, --hf_dataset_id is required "
            "to load dataset stats for normalization."
        )

    # === Robot config ===
    robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id=args.robot_id)
    robot = LeKiwiClient(robot_config)
    robot.connect()

    # === Dataset features ===
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    # ---- Readiness checks: state/action order + camera resolution ----
    requested_dof = None if args.robot_dof == "auto" else int(args.robot_dof)
    state_order, action_dim = _validate_state_action_order(dataset_features, requested_dof=requested_dof)
    print(f"[info] Detected robot action/state DoF: {action_dim}")

    # When --no_save and local_act, load existing dataset for normalization stats only (no disk write).
    dataset_for_stats = None
    if args.no_save and args.policy_mode == "local_act":
        dataset_for_stats = LeRobotDataset(args.hf_dataset_id)
        print(f"[info] --no_save: using {args.hf_dataset_id} for policy stats only (no evaluation data will be saved).")

    dataset = None
    if not args.no_save:
        # Create dataset: use provided repo_id or a local temporary path
        if args.hf_dataset_id:
            dataset_repo_id = args.hf_dataset_id
            dataset_root = None  # Use default HF_LEROBOT_HOME location
        else:
            # Use a local temporary dataset path (not pushed to Hub)
            from datetime import datetime
            from pathlib import Path
            dataset_repo_id = f"local_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            dataset_root = Path.cwd() / "eval_data" / dataset_repo_id
            print(f"[info] No --hf_dataset_id provided, using local temporary dataset: {dataset_repo_id}")
            print(f"[info] Evaluation data will be saved locally at: {dataset_root}")
            print(f"[info] Data will NOT be pushed to HuggingFace Hub.")

        dataset = LeRobotDataset.create(
            repo_id=dataset_repo_id,
            root=dataset_root,
            fps=args.fps,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
        )
    else:
        print(f"[info] --no_save: inference only, no evaluation data will be written to disk.")

    # Grab a sample observation for camera logging (and optional color conversion sanity).
    obs0 = robot.get_observation()
    if args.bgr_to_rgb:
        obs0 = _maybe_bgr_to_rgb(obs0, ("head_top", "wrist_left", "wrist_right"))
    _print_camera_info(robot, dataset_features, obs0)
    if args.policy_mode == "remote_http" and not isinstance(obs0.get("head_top"), np.ndarray):
        raise ValueError(
            "Remote HTTP 推理要求至少提供 head_top 相机帧（用于映射到 OpenPI images.cam_high），"
            "但当前 obs0['head_top'] 缺失或不是 ndarray。"
        )

    # === Policy ===
    if args.policy_mode == "local_act":
        policy = ACTPolicy.from_pretrained(args.hf_model_id)
        ds_for_policy = dataset if dataset is not None else dataset_for_stats
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy,
            pretrained_path=args.hf_model_id,
            dataset_stats=ds_for_policy.meta.stats,
            preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
        )
    else:
        _maybe_print_server_metadata(args.server_url)
        policy_cfg = RemoteHTTPPolicyConfig(server_url=args.server_url, device="cpu")
        policy = RemoteHTTPPolicy(policy_cfg)
        preprocessor, postprocessor = _noop_preprocessor(), _noop_postprocessor()

        # Check if we need to load training dataset stats for action denormalization
        training_stats = None
        if args.training_dataset_id:
            try:
                print(f"[info] Loading training dataset stats from: {args.training_dataset_id}")
                training_dataset = LeRobotDataset(args.training_dataset_id)
                training_stats = training_dataset.meta.stats
                if "action" in training_stats:
                    action_stats = training_stats["action"]
                    import torch
                    mean = action_stats.get("mean")
                    std = action_stats.get("std")
                    if isinstance(mean, torch.Tensor):
                        mean = mean.numpy()
                    if isinstance(std, torch.Tensor):
                        std = std.numpy()
                    print(
                        f"[info] Training action stats: "
                        f"mean range=[{float(np.min(mean)):.3f}, {float(np.max(mean)):.3f}], "
                        f"std range=[{float(np.min(std)):.3f}, {float(np.max(std)):.3f}]"
                    )
                else:
                    print(f"[warning] Training dataset has no action stats, skipping denormalization setup.")
                    training_stats = None
            except Exception as e:
                print(f"[warning] Failed to load training dataset stats: {e}")
                training_stats = None

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    if args.bgr_to_rgb:
        # Apply color conversion on every loop iteration via record_loop's observation processor.
        # OpenPI training uses RGB (from torchvision VideoReader), but LeKiwiClient returns BGR (from cv2.imdecode).
        # This conversion is REQUIRED for correct inference.
        bgr_to_rgb_step = BGRToRGBProcessorStep(camera_keys=("head_top", "wrist_left", "wrist_right"))
        # Chain the BGR→RGB step after the default processor.
        robot_observation_processor = RobotProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=[*robot_observation_processor.steps, bgr_to_rgb_step],
            to_transition=observation_to_transition,
            to_output=transition_to_observation,
        )

    # Create action validation/denormalization wrapper
    frame_counter = {"count": 0}
    _denormalize_step = None
    if args.policy_mode == "remote_http" and training_stats is not None:
        _denormalize_step = ActionDenormalizeProcessorStep(training_stats=training_stats)

    if args.debug_actions or _denormalize_step is not None:
        from lerobot.policies.utils import make_robot_action as _original_make_robot_action

        def make_robot_action_with_validation(action_tensor: Any, ds_features: dict[str, dict]) -> dict[str, Any]:
            """Wrapper around make_robot_action that adds validation, logging, and optional denormalization."""
            frame_counter["count"] += 1
            frame_idx = frame_counter["count"]

            import torch

            # Apply denormalization if needed
            if _denormalize_step is not None:
                action_tensor = _denormalize_step.action(action_tensor)

            # Convert to numpy for validation
            if isinstance(action_tensor, torch.Tensor):
                action_array = action_tensor.detach().cpu().numpy()
            else:
                action_array = np.asarray(action_tensor)

            # Flatten if needed
            if action_array.ndim > 1:
                action_array = action_array.flatten()

            # Validate and log (only if debugging enabled)
            if args.debug_actions:
                _validate_action_values(
                    action_tensor=action_tensor,
                    action_dict={},  # Will be created below
                    dataset_features=ds_features,
                    expected_order=state_order,
                    expected_action_dim=action_dim,
                    frame_idx=frame_idx,
                    log_every_n=30,
                )

            # Create action dict using original function
            action_dict = _original_make_robot_action(action_tensor, ds_features)

            # Log action dict (only if debugging enabled)
            if args.debug_actions and (frame_idx == 1 or frame_idx % 30 == 0):
                action_min = float(np.min(action_array))
                action_max = float(np.max(action_array))
                print(
                    f"[action_debug] frame={frame_idx}: "
                    f"action_tensor shape={action_tensor.shape if hasattr(action_tensor, 'shape') else 'unknown'}, "
                    f"action_array range=[{action_min:.3f}, {action_max:.3f}], "
                    f"action_dict keys={list(action_dict.keys())[:3]}..."  # Show first 3 keys
                )

            return action_dict

        # Monkey-patch make_robot_action for this session
        import lerobot.policies.utils
        lerobot.policies.utils.make_robot_action = make_robot_action_with_validation
        if args.debug_actions:
            print("[debug] Action validation and logging enabled. Monitoring action values...")
        if _denormalize_step is not None:
            print("[debug] Action denormalization enabled using training dataset stats.")

    listener, events = init_keyboard_listener()
    init_rerun(session_name="lekiwi_evaluate")

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    # Print diagnostic summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"Policy mode: {args.policy_mode}")
    if args.policy_mode == "remote_http":
        print(f"Server URL: {args.server_url}")
        print(f"Action denormalization: {'ENABLED' if _denormalize_step is not None else 'DISABLED'}")
        if _denormalize_step is None and not args.training_dataset_id:
            print(
                "  WARNING: No training dataset stats provided. "
                "If actions appear normalized (range ~[-1, 1]), use --training_dataset_id to enable denormalization."
            )
    print(f"Action debugging: {'ENABLED' if args.debug_actions else 'DISABLED'}")
    print(f"BGR→RGB conversion: {'ENABLED' if args.bgr_to_rgb else 'DISABLED'}")
    print("=" * 80 + "\n")

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
            dataset_features=dataset_features if args.no_save else None,
            save_frames=not args.no_save,
            control_time_s=args.episode_time,
            single_task=args.task_description,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        if not events["stop_recording"] and dataset is not None:
            dataset.save_episode()
            recorded_episodes += 1
        elif not events["stop_recording"] and args.no_save:
            recorded_episodes += 1

    log_say("Stop recording")
    robot.disconnect()
    listener.stop()
    if dataset is not None:
        dataset.finalize()
        if args.hf_dataset_id:
            # Only push to Hub if a valid repo_id was provided
            dataset.push_to_hub()
            print(f"[info] Evaluation data pushed to HuggingFace Hub: {dataset.repo_id}")
        else:
            print(f"[info] Evaluation data saved locally at: {dataset.root}")
            print(f"[info] To push to Hub later, use: dataset.push_to_hub()")
    else:
        print(f"[info] --no_save: no evaluation data was saved.")


if __name__ == "__main__":
    main()
