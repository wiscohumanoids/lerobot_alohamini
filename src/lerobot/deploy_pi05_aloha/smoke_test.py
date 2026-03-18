from __future__ import annotations

import argparse
import time

import numpy as np

from .policy_runner import Pi05PolicyRunner, PolicyRunnerConfig
from .cameras import DEFAULT_LEKIWI_CAMERA_KEYS
from .observation_builder import ObservationBuilder
from .aloha_interface import DEFAULT_LEKIWI_STATE_KEYS


def make_dummy_frames(camera_keys: tuple[str, ...], h: int = 224, w: int = 224) -> dict[str, np.ndarray]:
    return {
        k: np.zeros((h, w, 3), dtype=np.uint8)
        for k in camera_keys
    }


def make_dummy_state(state_keys: tuple[str, ...]) -> np.ndarray:
    return np.zeros((len(state_keys),), dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a quick local PI0.5 inference with dummy inputs.")
    parser.add_argument(
        "--pretrained-path",
        default="lerobot/pi05_base",
        help="HF id or local path",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device, e.g. cpu or cuda",
    )
    parser.add_argument(
        "--task-text",
        default="Pick up the small red cube and place it on the green pad",
        help="Task prompt",
    )
    args = parser.parse_args()

    cfg = PolicyRunnerConfig(
        pretrained_path=args.pretrained_path,
        device=args.device,
    )

    print(f"Loading policy from {args.pretrained_path} on device={args.device} ...")
    start = time.perf_counter()
    try:
        runner = Pi05PolicyRunner(cfg)
    except Exception as e:
        print("Failed to instantiate policy runner:", e)
        return 2
    print(f"Loaded in {time.perf_counter() - start:.1f}s")

    frames = make_dummy_frames(DEFAULT_LEKIWI_CAMERA_KEYS)
    state = make_dummy_state(DEFAULT_LEKIWI_STATE_KEYS)

    builder = ObservationBuilder.from_camera_names(list(DEFAULT_LEKIWI_CAMERA_KEYS))
    batch = builder.build(
        frames=frames,
        state=state,
        task_text=args.task_text,
    )

    print("Running single inference...")
    try:
        action = runner.infer(batch)
    except Exception as e:
        print("Inference failed:", e)
        return 3

    print("Action vector:", action)
    print("Shape:", action.shape)
    print("Dtype:", action.dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
