#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from lerobot.utils.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    OPTIMIZER_PARAM_GROUPS,
    OPTIMIZER_STATE,
    RNG_STATE,
    SCHEDULER_STATE,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    load_training_step,
    preflight_checkpointing,
    prune_old_checkpoints,
    save_checkpoint,
    save_training_state,
    save_training_step,
    update_last_checkpoint,
)


def test_get_step_identifier():
    assert get_step_identifier(5, 1000) == "000005"
    assert get_step_identifier(123, 100_000) == "000123"
    assert get_step_identifier(456789, 1_000_000) == "0456789"


def test_get_step_checkpoint_dir():
    output_dir = Path("/checkpoints")
    step_dir = get_step_checkpoint_dir(output_dir, 1000, 5)
    assert step_dir == output_dir / CHECKPOINTS_DIR / "000005"


def test_save_load_training_step(tmp_path):
    save_training_step(5000, tmp_path)
    assert (tmp_path / TRAINING_STEP).is_file()


def test_load_training_step(tmp_path):
    step = 5000
    save_training_step(step, tmp_path)
    loaded_step = load_training_step(tmp_path)
    assert loaded_step == step


def test_update_last_checkpoint(tmp_path):
    checkpoint = tmp_path / "0005"
    checkpoint.mkdir()
    update_last_checkpoint(checkpoint)
    last_checkpoint = tmp_path / LAST_CHECKPOINT_LINK
    assert last_checkpoint.is_symlink()
    assert last_checkpoint.resolve() == checkpoint


@patch("lerobot.utils.train_utils.save_training_state")
def test_save_checkpoint(mock_save_training_state, tmp_path, optimizer):
    policy = Mock()
    cfg = Mock()
    save_checkpoint(tmp_path, 10, cfg, policy, optimizer)
    policy.save_pretrained.assert_called_once()
    cfg.save_pretrained.assert_called_once()
    mock_save_training_state.assert_called_once()


@patch("lerobot.utils.train_utils.save_training_state")
def test_save_checkpoint_peft(mock_save_training_state, tmp_path, optimizer):
    policy = Mock()
    policy.config = Mock()
    policy.config.save_pretrained = Mock()
    cfg = Mock()
    cfg.use_peft = True
    save_checkpoint(tmp_path, 10, cfg, policy, optimizer)
    policy.save_pretrained.assert_called_once()
    cfg.save_pretrained.assert_called_once()
    policy.config.save_pretrained.assert_called_once()
    mock_save_training_state.assert_called_once()


def test_save_training_state(tmp_path, optimizer, scheduler):
    save_training_state(tmp_path, 10, optimizer, scheduler)
    assert (tmp_path / TRAINING_STATE_DIR).is_dir()
    assert (tmp_path / TRAINING_STATE_DIR / TRAINING_STEP).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / RNG_STATE).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / OPTIMIZER_STATE).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / OPTIMIZER_PARAM_GROUPS).is_file()
    assert (tmp_path / TRAINING_STATE_DIR / SCHEDULER_STATE).is_file()


def test_save_load_training_state(tmp_path, optimizer, scheduler):
    save_training_state(tmp_path, 10, optimizer, scheduler)
    loaded_step, loaded_optimizer, loaded_scheduler = load_training_state(tmp_path, optimizer, scheduler)
    assert loaded_step == 10
    assert loaded_optimizer is optimizer
    assert loaded_scheduler is scheduler


def test_prune_old_checkpoints_keeps_latest(tmp_path):
    checkpoints_dir = tmp_path / CHECKPOINTS_DIR
    old_checkpoint = checkpoints_dir / "000005"
    new_checkpoint = checkpoints_dir / "000010"
    old_checkpoint.mkdir(parents=True)
    new_checkpoint.mkdir(parents=True)
    (old_checkpoint / "marker.txt").write_text("old")
    (new_checkpoint / "marker.txt").write_text("new")

    deleted = prune_old_checkpoints(checkpoints_dir, keep_last=1)

    assert deleted == [old_checkpoint]
    assert not old_checkpoint.exists()
    assert new_checkpoint.exists()


@patch("lerobot.utils.train_utils.ensure_last_checkpoint_symlink_supported")
@patch("lerobot.utils.train_utils.shutil.disk_usage")
def test_preflight_checkpointing_raises_when_disk_is_too_small(
    mock_disk_usage, mock_symlink_supported, tmp_path, optimizer
):
    policy = Mock()
    policy.state_dict.return_value = {"weight": torch.zeros(1024, 1024)}
    mock_disk_usage.return_value = SimpleNamespace(free=1)

    with pytest.raises(RuntimeError, match="not enough free disk space"):
        preflight_checkpointing(tmp_path, policy, optimizer)
