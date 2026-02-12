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

"""Shared fixtures for π0.5 (pi05) policy tests. Reusable for CPU, GPU, and future robot tests."""

import os

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi05 import PI05Config, PI05Policy, make_pi05_pre_post_processors
from lerobot.utils.random_utils import set_seed


def _get_pi05_test_device() -> str:
    """Device for π0.5 tests: PI05_TEST_DEVICE env or 'cpu'. Use 'cpu' for CI; set to 'cuda' for GPU."""
    return os.environ.get("PI05_TEST_DEVICE", "cpu").lower()


@pytest.fixture(scope="session")
def pi05_test_device() -> str:
    """Device to run π0.5 tests on. Default 'cpu'; set PI05_TEST_DEVICE=cuda for GPU."""
    return _get_pi05_test_device()


@pytest.fixture
def pi05_config(pi05_test_device: str) -> PI05Config:
    """π0.5 config with device set from PI05_TEST_DEVICE (default CPU)."""
    if pi05_test_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("PI05_TEST_DEVICE=cuda but CUDA is not available")
    set_seed(42)
    config = PI05Config(
        max_action_dim=7,
        max_state_dim=14,
        dtype="float32",
        device=pi05_test_device,
    )
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
    }
    return config


@pytest.fixture
def pi05_dataset_stats() -> dict:
    """Dataset stats for π0.5 normalizer (QUANTILES). Same shape as config features."""
    return {
        "observation.state": {
            "mean": torch.zeros(14),
            "std": torch.ones(14),
            "q01": -torch.ones(14),
            "q99": torch.ones(14),
        },
        "action": {
            "mean": torch.zeros(7),
            "std": torch.ones(7),
            "q01": -torch.ones(7),
            "q99": torch.ones(7),
        },
        "observation.images.base_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        },
    }


@pytest.fixture
def pi05_policy(pi05_config: PI05Config) -> PI05Policy:
    """π0.5 policy instance (from scratch). Call policy.eval() before inference."""
    policy = PI05Policy(pi05_config)
    policy.eval()
    return policy


@pytest.fixture
def pi05_processors(pi05_config: PI05Config, pi05_dataset_stats: dict):
    """Preprocessor and postprocessor pipelines for π0.5."""
    return make_pi05_pre_post_processors(config=pi05_config, dataset_stats=pi05_dataset_stats)


@pytest.fixture
def pi05_dummy_batch(pi05_config: PI05Config) -> dict:
    """Dummy batch for π0.5: state, image, task. Device matches config. Batch size 1."""
    device = pi05_config.device
    batch_size = 1
    return {
        "observation.state": torch.randn(batch_size, 14, dtype=torch.float32, device=device),
        "observation.images.base_0_rgb": torch.rand(
            batch_size, 3, 224, 224, dtype=torch.float32, device=device
        ),
        "task": ["Pick up the object"],
    }
