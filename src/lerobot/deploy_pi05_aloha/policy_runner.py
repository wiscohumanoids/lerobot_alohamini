from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.pi05 import PI05Config, PI05Policy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


@dataclass
class PolicyRunnerConfig:
    pretrained_path: str
    device: str = "cuda"


class Pi05PolicyRunner:
    def __init__(self, cfg: PolicyRunnerConfig):
        self.cfg = cfg
        requested_device = cfg.device
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"

        self.device = torch.device(requested_device)
        self.policy_config = PI05Config.from_pretrained(cfg.pretrained_path)
        self.policy_config.device = str(self.device)

        self.policy: PI05Policy = PI05Policy.from_pretrained(
            pretrained_name_or_path=cfg.pretrained_path,
            config=self.policy_config,
            strict=False,
        )
        self.policy.eval()
        self.policy.to(self.device)

        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] = (
            PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=cfg.pretrained_path,
                config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            )
        )
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] = (
            PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=cfg.pretrained_path,
                config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            )
        )

    def infer(self, raw_observation_batch: dict[str, Any]) -> np.ndarray:
        processed = self.preprocessor(raw_observation_batch)
        with torch.inference_mode():
            action: Tensor = self.policy.select_action(processed)
            action = self.postprocessor(action)
        if action.ndim != 2 or action.shape[0] != 1:
            raise ValueError(f"Expected single-batch action tensor, got shape={tuple(action.shape)}")
        return action[0].detach().cpu().numpy().astype(np.float32)
