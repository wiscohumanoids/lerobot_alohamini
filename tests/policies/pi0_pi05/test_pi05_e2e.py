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

"""π0.5 (pi05) end-to-end tests: load → preprocess → act → postprocess on CPU (default) or GPU.

Run on CPU (default, CI-friendly):
    pytest tests/policies/pi0_pi05/test_pi05_e2e.py -v

Run on GPU:
    PI05_TEST_DEVICE=cuda pytest tests/policies/pi0_pi05/test_pi05_e2e.py -v

Uses shared fixtures from conftest.py; same harness can be reused for robot or GPU by
setting PI05_TEST_DEVICE or extending fixtures (e.g. real observations).

IMPORTANT: Use the command: `python -m pytest tests/policies/pi0_pi05/test_pi05_e2e.py -v` to run the tests.
"""

import pytest
import torch

from lerobot.policies.pi05 import PI05Policy, make_pi05_pre_post_processors
from lerobot.utils.constants import ACTION


def test_pi05_e2e_load_preprocess_act_postprocess(
    pi05_policy: PI05Policy,
    pi05_processors,
    pi05_dummy_batch: dict,
    pi05_config,
) -> None:
    """Prove π0.5 runs end-to-end: load → preprocess → act → postprocess. Runs on CPU by default."""
    preprocessor, postprocessor = pi05_processors
    device = pi05_config.device

    # Preprocess
    batch = preprocessor(pi05_dummy_batch)

    # Act (select_action: single step from chunk)
    with torch.no_grad():
        action = pi05_policy.select_action(batch)
    action = postprocessor(action)

    # Assert
    assert action is not None
    assert isinstance(action, torch.Tensor)
    expected_dim = pi05_config.output_features[ACTION].shape[0]
    assert action.shape == (1, expected_dim), f"Expected (1, {expected_dim}), got {action.shape}"
    assert action.device.type == "cpu", "Postprocessor should return CPU tensors"
    assert not torch.isnan(action).any() and not torch.isinf(action).any(), "Action had NaN/Inf"


def test_pi05_e2e_predict_action_chunk(
    pi05_policy: PI05Policy,
    pi05_processors,
    pi05_dummy_batch: dict,
    pi05_config,
) -> None:
    """Same pipeline using predict_action_chunk (full chunk). Reusable for RTC/robot."""
    preprocessor, postprocessor = pi05_processors

    batch = preprocessor(pi05_dummy_batch)

    with torch.no_grad():
        actions = pi05_policy.predict_action_chunk(batch)
    actions = postprocessor(actions)

    assert actions is not None
    assert actions.shape == (1, pi05_config.chunk_size, pi05_config.output_features[ACTION].shape[0])
    assert actions.device.type == "cpu"
    assert not torch.isnan(actions).any() and not torch.isinf(actions).any()


def test_pi05_e2e_forward_pass(
    pi05_policy: PI05Policy,
    pi05_processors,
    pi05_dummy_batch: dict,
    pi05_config,
) -> None:
    """Forward pass (training path): preprocess → forward → loss finite."""
    preprocessor, _ = pi05_processors
    batch = preprocessor(pi05_dummy_batch)
    # Forward expects an "action" key in the batch
    batch[ACTION] = torch.randn(
        1, pi05_config.chunk_size, pi05_config.output_features[ACTION].shape[0],
        dtype=torch.float32, device=pi05_config.device,
    )

    loss, loss_dict = pi05_policy.forward(batch)

    assert torch.isfinite(loss).item(), f"Loss not finite: {loss_dict}"
    assert "loss" in loss_dict


def test_pi05_e2e_cpu_explicit() -> None:
    """Explicit CPU e2e: proves π0.5 runs on CPU without relying on env (CI-safe)."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pi05 import PI05Config, PI05Policy, make_pi05_pre_post_processors
    from lerobot.utils.random_utils import set_seed

    set_seed(42)
    config = PI05Config(
        max_action_dim=7,
        max_state_dim=14,
        dtype="float32",
        device="cpu",
    )
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    dataset_stats = {
        "observation.state": {
            "mean": torch.zeros(14), "std": torch.ones(14),
            "q01": -torch.ones(14), "q99": torch.ones(14),
        },
        "action": {
            "mean": torch.zeros(7), "std": torch.ones(7),
            "q01": -torch.ones(7), "q99": torch.ones(7),
        },
        "observation.images.base_0_rgb": {
            "mean": torch.zeros(3, 224, 224), "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224), "q99": torch.ones(3, 224, 224),
        },
    }

    policy = PI05Policy(config)
    policy.eval()
    preprocessor, postprocessor = make_pi05_pre_post_processors(config=config, dataset_stats=dataset_stats)
    batch = {
        "observation.state": torch.randn(1, 14, dtype=torch.float32, device="cpu"),
        "observation.images.base_0_rgb": torch.rand(1, 3, 224, 224, dtype=torch.float32, device="cpu"),
        "task": ["Pick up the object"],
    }
    batch = preprocessor(batch)
    with torch.no_grad():
        action = policy.select_action(batch)
    action = postprocessor(action)

    assert action.shape == (1, 7)
    assert action.device.type == "cpu"
    assert not torch.isnan(action).any()
