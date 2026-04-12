#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import shutil
from pathlib import Path

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.utils import load_json, write_json
from lerobot.optim.optimizers import load_optimizer_state, save_optimizer_state
from lerobot.optim.schedulers import load_scheduler_state, save_scheduler_state
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)
from lerobot.utils.random_utils import load_rng_state, save_rng_state


def get_step_identifier(step: int, total_steps: int) -> str:
    num_digits = max(6, len(str(total_steps)))
    return f"{step:0{num_digits}d}"


def get_step_checkpoint_dir(output_dir: Path, total_steps: int, step: int) -> Path:
    """Returns the checkpoint sub-directory corresponding to the step number."""
    step_identifier = get_step_identifier(step, total_steps)
    return output_dir / CHECKPOINTS_DIR / step_identifier


def save_training_step(step: int, save_dir: Path) -> None:
    write_json({"step": step}, save_dir / TRAINING_STEP)


def load_training_step(save_dir: Path) -> int:
    training_step = load_json(save_dir / TRAINING_STEP)
    return training_step["step"]


def update_last_checkpoint(checkpoint_dir: Path) -> Path:
    last_checkpoint_dir = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
    if last_checkpoint_dir.is_symlink():
        last_checkpoint_dir.unlink()
    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    last_checkpoint_dir.symlink_to(relative_target, target_is_directory=True)


def get_checkpoint_dirs(checkpoints_dir: Path) -> list[Path]:
    if not checkpoints_dir.exists():
        return []
    return sorted(path for path in checkpoints_dir.iterdir() if path.is_dir() and path.name.isdigit())


def prune_old_checkpoints(checkpoints_dir: Path, keep_last: int) -> list[Path]:
    if keep_last < 1:
        raise ValueError(f"{keep_last=} must be >= 1")

    checkpoint_dirs = get_checkpoint_dirs(checkpoints_dir)
    to_delete = checkpoint_dirs[:-keep_last]
    for checkpoint_dir in to_delete:
        shutil.rmtree(checkpoint_dir)
    return to_delete


def _tensor_nbytes(value) -> int:
    return value.numel() * value.element_size() if isinstance(value, torch.Tensor) else 0


def _nested_tensors_nbytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return _tensor_nbytes(value)
    if isinstance(value, dict):
        return sum(_nested_tensors_nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_nested_tensors_nbytes(v) for v in value)
    return 0


def _estimate_optimizer_state_bytes(
    optimizer: Optimizer | dict[str, Optimizer], fallback_model_bytes: int
) -> int:
    if isinstance(optimizer, dict):
        return sum(_estimate_optimizer_state_bytes(opt, fallback_model_bytes) for opt in optimizer.values())

    state_bytes = _nested_tensors_nbytes(optimizer.state_dict())
    if state_bytes > 0:
        return state_bytes

    # Before the first optimizer step, Adam-like optimizers have no tensor state yet.
    return fallback_model_bytes * 2


def estimate_checkpoint_size_bytes(
    policy: PreTrainedPolicy, optimizer: Optimizer | dict[str, Optimizer]
) -> int:
    model_bytes = _nested_tensors_nbytes(policy.state_dict())
    optimizer_bytes = _estimate_optimizer_state_bytes(optimizer, model_bytes)
    # JSON/config/RNG/processor state is small compared to model + optimizer, but keep some buffer.
    return model_bytes + optimizer_bytes + 256 * 1024**2


def get_checkpoint_size_bytes(checkpoint_dir: Path) -> int:
    return sum(path.stat().st_size for path in checkpoint_dir.rglob("*") if path.is_file())


def _format_gb(size_bytes: int) -> str:
    return f"{size_bytes / 1024**3:.2f} GB"


def ensure_last_checkpoint_symlink_supported(checkpoints_dir: Path) -> None:
    test_root = checkpoints_dir / ".checkpoint_symlink_test"
    target_dir = test_root / "target"
    link_dir = test_root / "link"

    if test_root.exists():
        shutil.rmtree(test_root, ignore_errors=True)

    test_root.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        link_dir.symlink_to(target_dir.relative_to(test_root), target_is_directory=True)
    except OSError as exc:
        raise RuntimeError(
            "Checkpoint preflight failed: this shell cannot create the `checkpoints/last` symlink. "
            "On Windows, run the shell as Administrator or enable Developer Mode before training."
        ) from exc
    finally:
        if link_dir.is_symlink():
            link_dir.unlink()
        if test_root.exists():
            shutil.rmtree(test_root, ignore_errors=True)


def preflight_checkpointing(
    output_dir: Path,
    policy: PreTrainedPolicy,
    optimizer: Optimizer | dict[str, Optimizer],
) -> None:
    checkpoints_dir = output_dir / CHECKPOINTS_DIR
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    ensure_last_checkpoint_symlink_supported(checkpoints_dir)

    latest_checkpoint_size = 0
    checkpoint_dirs = get_checkpoint_dirs(checkpoints_dir)
    if checkpoint_dirs:
        latest_checkpoint_size = get_checkpoint_size_bytes(checkpoint_dirs[-1])

    estimated_checkpoint_size = estimate_checkpoint_size_bytes(policy, optimizer)
    required_free_bytes = max(latest_checkpoint_size, estimated_checkpoint_size)
    free_bytes = shutil.disk_usage(checkpoints_dir).free

    if free_bytes < required_free_bytes:
        raise RuntimeError(
            "Checkpoint preflight failed: not enough free disk space for the next checkpoint. "
            f"Free: {_format_gb(free_bytes)}. "
            f"Required: {_format_gb(required_free_bytes)}."
        )


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    cfg: TrainPipelineConfig,
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    preprocessor: PolicyProcessorPipeline | None = None,
    postprocessor: PolicyProcessorPipeline | None = None,
) -> None:
    """This function creates the following directory structure:

    005000/  #  training step at checkpoint
    ├── pretrained_model/
    │   ├── config.json  # policy config
    │   ├── model.safetensors  # policy weights
    │   ├── train_config.json  # train config
    │   ├── processor.json  # processor config (if preprocessor provided)
    │   └── step_*.safetensors  # processor state files (if any)
    └── training_state/
        ├── optimizer_param_groups.json  #  optimizer param groups
        ├── optimizer_state.safetensors  # optimizer state
        ├── rng_state.safetensors  # rng states
        ├── scheduler_state.json  # scheduler state
        └── training_step.json  # training step

    Args:
        cfg (TrainPipelineConfig): The training config used for this run.
        step (int): The training step at that checkpoint.
        policy (PreTrainedPolicy): The policy to save.
        optimizer (Optimizer | None, optional): The optimizer to save the state from. Defaults to None.
        scheduler (LRScheduler | None, optional): The scheduler to save the state from. Defaults to None.
        preprocessor: The preprocessor/pipeline to save. Defaults to None.
    """
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    policy.save_pretrained(pretrained_dir)
    cfg.save_pretrained(pretrained_dir)
    if cfg.peft is not None:
        # When using PEFT, policy.save_pretrained will only write the adapter weights + config, not the
        # policy config which we need for loading the model. In this case we'll write it ourselves.
        policy.config.save_pretrained(pretrained_dir)
    if preprocessor is not None:
        preprocessor.save_pretrained(pretrained_dir)
    if postprocessor is not None:
        postprocessor.save_pretrained(pretrained_dir)
    save_training_state(checkpoint_dir, step, optimizer, scheduler)


def save_training_state(
    checkpoint_dir: Path,
    train_step: int,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
) -> None:
    """
    Saves the training step, optimizer state, scheduler state, and rng state.

    Args:
        save_dir (Path): The directory to save artifacts to.
        train_step (int): Current training step.
        optimizer (Optimizer | None, optional): The optimizer from which to save the state_dict.
            Defaults to None.
        scheduler (LRScheduler | None, optional): The scheduler from which to save the state_dict.
            Defaults to None.
    """
    save_dir = checkpoint_dir / TRAINING_STATE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    save_training_step(train_step, save_dir)
    save_rng_state(save_dir)
    if optimizer is not None:
        save_optimizer_state(optimizer, save_dir)
    if scheduler is not None:
        save_scheduler_state(scheduler, save_dir)


def load_training_state(
    checkpoint_dir: Path, optimizer: Optimizer, scheduler: LRScheduler | None
) -> tuple[int, Optimizer, LRScheduler | None]:
    """
    Loads the training step, optimizer state, scheduler state, and rng state.
    This is used to resume a training run.

    Args:
        checkpoint_dir (Path): The checkpoint directory. Should contain a 'training_state' dir.
        optimizer (Optimizer): The optimizer to load the state_dict to.
        scheduler (LRScheduler | None): The scheduler to load the state_dict to (can be None).

    Raises:
        NotADirectoryError: If 'checkpoint_dir' doesn't contain a 'training_state' dir

    Returns:
        tuple[int, Optimizer, LRScheduler | None]: training step, optimizer and scheduler with their
            state_dict loaded.
    """
    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
    if not training_state_dir.is_dir():
        raise NotADirectoryError(training_state_dir)

    load_rng_state(training_state_dir)
    step = load_training_step(training_state_dir)
    optimizer = load_optimizer_state(optimizer, training_state_dir)
    if scheduler is not None:
        scheduler = load_scheduler_state(scheduler, training_state_dir)

    return step, optimizer, scheduler
