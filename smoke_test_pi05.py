import torch
import numpy as np

from lerobot.deploy_pi05_aloha.policy_runner import (
    Pi05PolicyRunner,
    PolicyRunnerConfig
)

print("Starting PI05 smoke test...")

config = PolicyRunnerConfig(
    pretrained_path="/workspace/checkpoints/pi05_base",  # adjust if needed
    device="cpu"  # important for Jetson
)

runner = Pi05PolicyRunner(config)

print("Policy loaded successfully")

# Create minimal fake observation
obs = {
    "state": torch.zeros(1, 10),
    "image": torch.zeros(1, 3, 224, 224)
}

print("Running one inference pass...")

with torch.no_grad():
    action = runner.predict(obs)

print("Action output:", action)

print("Smoke test passed ✅")
