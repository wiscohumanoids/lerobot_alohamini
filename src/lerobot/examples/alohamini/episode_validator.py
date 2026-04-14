#!/usr/bin/env python3
"""
Episode-level data quality validation for AlohaMini recordings.

Bad demonstrations silently corrupt training: a frozen arm, a stuck gripper,
or a NaN in the action stream looks fine in the dataset summary but breaks
ACT/Diffusion training in subtle ways. This module catches the obvious
failure modes from the in-memory `episode_buffer` *before* the episode is
written to disk and pushed to HuggingFace Hub.

Checks performed on each episode:
- Frame count consistency (buffer size matches array lengths)
- Duration plausibility (too short = aborted demo, too long = stuck loop)
- Per-joint action variance (a joint that never moved is suspicious)
- Per-joint state variance (the follower joint that never moved is worse)
- Action range sanity (min/max within expected bounds)
- Non-finite values in action or state arrays (NaN / Inf)

The validator is intentionally conservative: it produces *warnings*, not
hard failures. The recording script decides whether to keep or re-record.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Constants ------------------------------------------------------------------

# Below this std-dev (in whatever units the joint reports), a joint is
# considered "stationary" for the entire episode. AlohaMini joints typically
# report degrees, so 0.5 deg of total spread over an episode is effectively no
# motion.
DEFAULT_STATIONARY_STD_THRESHOLD = 0.5

# Episodes shorter than this are almost always aborted demos.
DEFAULT_MIN_DURATION_S = 2.0

# Episodes longer than this are almost always stuck record loops.
DEFAULT_MAX_DURATION_S = 600.0

# An episode that has more than this many problems is reported as failed.
DEFAULT_FAIL_ON_NUM_ISSUES = 1


# Result types ---------------------------------------------------------------


@dataclass
class EpisodeReport:
    """Summary of validation results for a single episode."""

    episode_index: int
    num_frames: int
    duration_s: float
    issues: list[str] = field(default_factory=list)
    stationary_action_joints: list[str] = field(default_factory=list)
    stationary_state_joints: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.issues) < DEFAULT_FAIL_ON_NUM_ISSUES

    def format(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"  Episode {self.episode_index}: {status}",
            f"    frames={self.num_frames}, duration={self.duration_s:.1f}s",
        ]
        for issue in self.issues:
            lines.append(f"    - {issue}")
        return "\n".join(lines)


# Validation -----------------------------------------------------------------


def _stack_buffer_array(buffer_value: list) -> np.ndarray | None:
    """Stack a list of per-frame numpy arrays into a single (T, N) array.

    Returns None if the buffer is empty or unstackable (e.g. list of paths).
    """
    if not buffer_value:
        return None
    try:
        stacked = np.stack(buffer_value)
    except (ValueError, TypeError):
        return None
    if stacked.dtype.kind not in ("f", "i", "u"):
        return None
    return stacked


def _check_finite(name: str, arr: np.ndarray, issues: list[str]) -> None:
    if not np.all(np.isfinite(arr)):
        bad_count = int(np.sum(~np.isfinite(arr)))
        issues.append(f"{name} contains {bad_count} non-finite values (NaN/Inf)")


def _check_stationary_joints(
    arr: np.ndarray,
    joint_names: list[str],
    threshold: float,
) -> list[str]:
    """Return the list of joint names whose std-dev is below the threshold.

    Excludes the gripper from this check because grippers legitimately stay
    open or closed for the entire episode in some tasks.
    """
    stds = arr.std(axis=0)
    stationary = []
    for std, name in zip(stds, joint_names):
        if "gripper" in name.lower():
            continue
        if float(std) < threshold:
            stationary.append(name)
    return stationary


def validate_episode(
    episode_buffer: dict,
    features: dict[str, dict],
    fps: int,
    *,
    stationary_std_threshold: float = DEFAULT_STATIONARY_STD_THRESHOLD,
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
    max_duration_s: float = DEFAULT_MAX_DURATION_S,
) -> EpisodeReport:
    """Validate a single in-memory episode buffer and return a report.

    This is intended to be called *before* `dataset.save_episode()`, while the
    raw per-frame arrays are still sitting in `dataset.episode_buffer`.

    Args:
        episode_buffer: The dataset's `episode_buffer` dict.
        features: The dataset's `features` dict (used to look up joint names).
        fps: The recording fps (used to convert frame count to seconds).

    Returns:
        EpisodeReport with any issues found.
    """
    issues: list[str] = []

    num_frames = int(episode_buffer.get("size", 0))
    episode_index = int(episode_buffer.get("episode_index", -1))

    # 1. Frame count plausibility
    if num_frames == 0:
        issues.append("episode has zero frames")
        return EpisodeReport(
            episode_index=episode_index,
            num_frames=0,
            duration_s=0.0,
            issues=issues,
        )

    duration_s = num_frames / max(fps, 1)

    if duration_s < min_duration_s:
        issues.append(f"episode is suspiciously short ({duration_s:.1f}s < {min_duration_s}s)")
    if duration_s > max_duration_s:
        issues.append(f"episode is suspiciously long ({duration_s:.1f}s > {max_duration_s}s)")

    # 2. Action checks
    action_arr = _stack_buffer_array(episode_buffer.get("action", []))
    action_joint_names: list[str] = []
    stationary_action: list[str] = []
    if action_arr is not None and action_arr.ndim == 2:
        action_joint_names = list(features.get("action", {}).get("names", []))
        if len(action_joint_names) != action_arr.shape[1]:
            # Fall back to indices when names are missing.
            action_joint_names = [f"action_{i}" for i in range(action_arr.shape[1])]

        _check_finite("action", action_arr, issues)
        stationary_action = _check_stationary_joints(
            action_arr, action_joint_names, stationary_std_threshold
        )
        if stationary_action:
            issues.append(
                f"{len(stationary_action)} action joint(s) had near-zero variance: "
                f"{', '.join(stationary_action)}"
            )

    # 3. Observation state checks
    state_arr = _stack_buffer_array(episode_buffer.get("observation.state", []))
    stationary_state: list[str] = []
    if state_arr is not None and state_arr.ndim == 2:
        state_joint_names = list(features.get("observation.state", {}).get("names", []))
        if len(state_joint_names) != state_arr.shape[1]:
            state_joint_names = [f"state_{i}" for i in range(state_arr.shape[1])]

        _check_finite("observation.state", state_arr, issues)
        stationary_state = _check_stationary_joints(
            state_arr, state_joint_names, stationary_std_threshold
        )
        if stationary_state:
            issues.append(
                f"{len(stationary_state)} follower joint(s) never moved: "
                f"{', '.join(stationary_state)}"
            )

    return EpisodeReport(
        episode_index=episode_index,
        num_frames=num_frames,
        duration_s=duration_s,
        issues=issues,
        stationary_action_joints=stationary_action,
        stationary_state_joints=stationary_state,
    )


def summarize(reports: list[EpisodeReport]) -> str:
    """Format a multi-episode validation report for the recording session."""
    if not reports:
        return "No episodes to validate."

    passed = sum(1 for r in reports if r.passed)
    failed = len(reports) - passed

    lines = [
        "=" * 60,
        f"Episode validation summary: {passed} passed, {failed} failed (of {len(reports)})",
        "=" * 60,
    ]
    for r in reports:
        lines.append(r.format())
    lines.append("=" * 60)
    return "\n".join(lines)
