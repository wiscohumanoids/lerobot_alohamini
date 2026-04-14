#!/usr/bin/env python3
"""
Quantitative metrics for AlohaMini policy evaluation episodes.

`evaluate_bi.py` currently records each episode and pushes the dataset to
the Hub, but produces no numbers anyone can compare across training runs.
Whether a policy is "better" than the previous one is judged by watching the
rollout — slow, subjective, and impossible to track in a sweep.

This module computes per-episode metrics from the in-memory `episode_buffer`
and supports appending them to a CSV log so multiple eval runs can be
compared in a spreadsheet or W&B.

Metrics computed (all derived from action and observation.state arrays):

- **frames** / **duration_s**: episode length in frames and seconds
- **action_path_length**: sum of L2 norms of per-frame action deltas. Low =
  policy barely moved; high = lots of motion (good for active tasks)
- **action_jerk_rms**: RMS of the third derivative of action positions.
  Lower is smoother. Jerky policies often fail in the real world even when
  validation loss looks fine in training.
- **idle_frame_pct**: fraction of frames where the per-frame action delta
  L2 norm is below a small threshold. High idle % can indicate the policy
  has stalled or is waiting on a stuck condition.
- **per_joint_range**: max - min for each joint in the action stream
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# Small absolute threshold (in joint units) below which a per-frame action
# delta is considered "no movement". 0.5 deg is roughly the encoder noise
# floor of the Feetech servos, so anything below this is indistinguishable
# from a stationary command.
DEFAULT_IDLE_DELTA_THRESHOLD = 0.5


@dataclass
class EpisodeMetrics:
    """Quantitative metrics for a single eval episode."""

    episode_index: int
    frames: int
    duration_s: float
    action_path_length: float
    action_jerk_rms: float
    idle_frame_pct: float
    mean_action_std: float
    # joint_name -> range (max - min)
    per_joint_range: dict[str, float] = field(default_factory=dict)

    def as_csv_row(
        self,
        *,
        run_id: str,
        model_id: str,
        dataset_id: str,
        timestamp_iso: str,
    ) -> dict[str, str]:
        """Flatten into a single CSV-friendly row."""
        row = {
            "timestamp": timestamp_iso,
            "run_id": run_id,
            "model_id": model_id,
            "dataset_id": dataset_id,
            "episode_index": str(self.episode_index),
            "frames": str(self.frames),
            "duration_s": f"{self.duration_s:.3f}",
            "action_path_length": f"{self.action_path_length:.3f}",
            "action_jerk_rms": f"{self.action_jerk_rms:.6f}",
            "idle_frame_pct": f"{self.idle_frame_pct:.2f}",
            "mean_action_std": f"{self.mean_action_std:.4f}",
        }
        for joint, rng in self.per_joint_range.items():
            row[f"range__{joint}"] = f"{rng:.3f}"
        return row

    def format(self) -> str:
        return (
            f"  Episode {self.episode_index}: "
            f"frames={self.frames} dur={self.duration_s:.1f}s "
            f"path={self.action_path_length:.1f} "
            f"jerk_rms={self.action_jerk_rms:.4f} "
            f"idle={self.idle_frame_pct:.1f}%"
        )


# Computation ----------------------------------------------------------------


def _stack_arrays(buffer_value: list) -> np.ndarray | None:
    if not buffer_value:
        return None
    try:
        stacked = np.stack(buffer_value)
    except (ValueError, TypeError):
        return None
    if stacked.dtype.kind not in ("f", "i", "u"):
        return None
    return stacked.astype(np.float64)


def _compute_path_length(actions: np.ndarray) -> float:
    """Sum of L2 norms of consecutive action deltas."""
    if actions.shape[0] < 2:
        return 0.0
    deltas = np.diff(actions, axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def _compute_jerk_rms(actions: np.ndarray, fps: int) -> float:
    """RMS of the third time-derivative of action positions.

    Returns 0.0 if the episode is too short to take a third difference.
    Units are (joint units) / s^3.
    """
    if actions.shape[0] < 4 or fps <= 0:
        return 0.0
    dt = 1.0 / fps
    velocity = np.diff(actions, axis=0) / dt
    accel = np.diff(velocity, axis=0) / dt
    jerk = np.diff(accel, axis=0) / dt
    # Per-frame magnitude across all joints, then RMS over time
    magnitude = np.linalg.norm(jerk, axis=1)
    return float(np.sqrt(np.mean(magnitude**2)))


def _compute_idle_pct(actions: np.ndarray, threshold: float) -> float:
    """Percentage of frames where the per-frame action delta is below threshold."""
    if actions.shape[0] < 2:
        return 0.0
    deltas = np.diff(actions, axis=0)
    delta_norms = np.linalg.norm(deltas, axis=1)
    idle = float(np.mean(delta_norms < threshold))
    return idle * 100.0


def compute_episode_metrics(
    episode_buffer: dict,
    features: dict[str, dict],
    fps: int,
    *,
    idle_delta_threshold: float = DEFAULT_IDLE_DELTA_THRESHOLD,
) -> EpisodeMetrics | None:
    """Compute metrics for one episode from the in-memory buffer.

    Returns None if the buffer has no usable action data.
    """
    num_frames = int(episode_buffer.get("size", 0))
    episode_index = int(episode_buffer.get("episode_index", -1))
    if num_frames == 0:
        return None

    actions = _stack_arrays(episode_buffer.get("action", []))
    if actions is None or actions.ndim != 2:
        return None

    joint_names = list(features.get("action", {}).get("names", []))
    if len(joint_names) != actions.shape[1]:
        joint_names = [f"action_{i}" for i in range(actions.shape[1])]

    duration_s = num_frames / max(fps, 1)
    path_length = _compute_path_length(actions)
    jerk_rms = _compute_jerk_rms(actions, fps)
    idle_pct = _compute_idle_pct(actions, idle_delta_threshold)
    mean_action_std = float(actions.std(axis=0).mean())
    per_joint_range = {
        name: float(actions[:, i].max() - actions[:, i].min())
        for i, name in enumerate(joint_names)
    }

    return EpisodeMetrics(
        episode_index=episode_index,
        frames=num_frames,
        duration_s=duration_s,
        action_path_length=path_length,
        action_jerk_rms=jerk_rms,
        idle_frame_pct=idle_pct,
        mean_action_std=mean_action_std,
        per_joint_range=per_joint_range,
    )


# Logging --------------------------------------------------------------------


def append_to_csv(
    csv_path: Path,
    metrics: list[EpisodeMetrics],
    *,
    run_id: str,
    model_id: str,
    dataset_id: str,
) -> None:
    """Append a batch of episode metrics to a CSV log file.

    Creates the file with a header if it doesn't yet exist. Existing files are
    appended to so that metrics from multiple eval runs accumulate over time.
    """
    if not metrics:
        return

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [
        m.as_csv_row(
            run_id=run_id,
            model_id=model_id,
            dataset_id=dataset_id,
            timestamp_iso=timestamp_iso,
        )
        for m in metrics
    ]

    # Use the union of all keys so per-joint range columns survive even if
    # different policies have different joint sets.
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(metrics: list[EpisodeMetrics]) -> str:
    if not metrics:
        return "No eval episodes to summarize."

    durations = np.array([m.duration_s for m in metrics])
    paths = np.array([m.action_path_length for m in metrics])
    jerks = np.array([m.action_jerk_rms for m in metrics])
    idles = np.array([m.idle_frame_pct for m in metrics])

    lines = [
        "=" * 60,
        f"Eval metrics summary ({len(metrics)} episodes)",
        "=" * 60,
        f"  duration_s        mean={durations.mean():.1f}  min={durations.min():.1f}  max={durations.max():.1f}",
        f"  action_path       mean={paths.mean():.1f}  min={paths.min():.1f}  max={paths.max():.1f}",
        f"  action_jerk_rms   mean={jerks.mean():.4f}  min={jerks.min():.4f}  max={jerks.max():.4f}",
        f"  idle_frame_pct    mean={idles.mean():.1f}%  min={idles.min():.1f}%  max={idles.max():.1f}%",
        "=" * 60,
    ]
    for m in metrics:
        lines.append(m.format())
    lines.append("=" * 60)
    return "\n".join(lines)
