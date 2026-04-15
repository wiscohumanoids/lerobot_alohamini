#!/usr/bin/env python3
"""
Train ACT (Action Chunking Transformer) on wiscohumanoids/colored_cube_v7.

ACT was designed for bimanual ALOHA tasks — exactly this setup.
Optimized for NVIDIA T500 (4 GB VRAM).

Usage:
    python scripts/train_act.py
    python scripts/train_act.py --steps 50000 --batch_size 4 --output outputs/train/act_cube
"""

import argparse
import time
from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--steps",      type=int,   default=80_000,
                    help="Total gradient steps (80k is solid for this dataset size)")
parser.add_argument("--batch_size", type=int,   default=4,
                    help="Batch size — keep at 4 on T500 (4 GB VRAM)")
parser.add_argument("--lr",         type=float, default=1e-5,
                    help="Learning rate (ACT default, proven on ALOHA)")
parser.add_argument("--chunk_size", type=int,   default=100,
                    help="Action chunk size (ACT default for ALOHA)")
parser.add_argument("--output",     type=str,   default="outputs/train/act_colored_cube",
                    help="Directory to save checkpoints")
parser.add_argument("--save_every", type=int,   default=5_000,
                    help="Save a checkpoint every N steps")
parser.add_argument("--log_every",  type=int,   default=100,
                    help="Print loss every N steps")
args = parser.parse_args()

DATASET_REPO = "wiscohumanoids/colored_cube_v7"
OUTPUT_DIR   = Path(args.output)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory // 1024**2} MB)")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
print(f"\nLoading dataset metadata from {DATASET_REPO} ...")
meta = LeRobotDatasetMetadata(DATASET_REPO)

features   = dataset_to_policy_features(meta.features)
output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
input_features  = {k: v for k, v in features.items() if k not in output_features}

print("Input features:")
for k, v in input_features.items():
    print(f"  {k}: shape={v.shape}")
print("Output features:")
for k, v in output_features.items():
    print(f"  {k}: shape={v.shape}")

# delta_timestamps: ACT uses n_obs_steps=1 (no history), chunk_size future actions
delta_timestamps = {
    "observation.images.head_top":   [0.0],
    "observation.images.head_front": [0.0],
    "observation.images.wrist_right":[0.0],
    "observation.state":             [0.0],
    "action": [i / meta.fps for i in range(args.chunk_size)],
}

print(f"\nLoading dataset (this will download videos the first time) ...")
dataset = LeRobotDataset(DATASET_REPO, delta_timestamps=delta_timestamps)
print(f"Dataset size: {len(dataset)} frames")

# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
cfg = ACTConfig(
    input_features=input_features,
    output_features=output_features,
    # Architecture — standard ALOHA settings
    chunk_size=args.chunk_size,
    n_action_steps=args.chunk_size,
    vision_backbone="resnet18",
    pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
    dim_model=512,
    n_heads=8,
    dim_feedforward=3200,
    n_encoder_layers=4,
    n_decoder_layers=1,
    use_vae=True,
    latent_dim=32,
    n_vae_encoder_layers=4,
    kl_weight=10.0,
    dropout=0.1,
    # Training
    optimizer_lr=args.lr,
    optimizer_lr_backbone=args.lr,
    optimizer_weight_decay=1e-4,
    # AMP: T500 supports fp16, cuts VRAM ~40%
    use_amp=True,
    device=device.type,
)

policy = ACTPolicy(cfg)
policy.train()
policy.to(device)

preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=meta.stats)

# Separate LR for backbone (standard ACT practice)
backbone_params = list(policy.model.backbone.parameters())
backbone_ids    = {id(p) for p in backbone_params}
other_params    = [p for p in policy.parameters() if id(p) not in backbone_ids]

optimizer = torch.optim.AdamW(
    [
        {"params": other_params,    "lr": args.lr},
        {"params": backbone_params, "lr": args.lr},  # same lr, can tune separately
    ],
    weight_decay=1e-4,
)

scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

# ---------------------------------------------------------------------------
# Dataloader
# num_workers=2: T500 on WSL2 — more workers tend to stall on video decoding
# ---------------------------------------------------------------------------
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=(device.type == "cuda"),
    drop_last=True,
    persistent_workers=True,
)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print(f"\nTraining ACT for {args.steps:,} steps  |  batch={args.batch_size}  |  "
      f"chunk={args.chunk_size}  |  lr={args.lr}\n")

step     = 0
t_start  = time.perf_counter()
loss_acc = 0.0

while step < args.steps:
    for batch in dataloader:
        batch = preprocessor(batch)

        # ACT expects observation.state as (B, state_dim), not (B, 1, state_dim)
        if "observation.state" in batch and batch["observation.state"].ndim == 3:
            batch["observation.state"] = batch["observation.state"].squeeze(1)

        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            loss, info = policy.forward(batch)

        scaler.scale(loss).backward()
        # Gradient clipping — stabilizes ACT training
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        loss_acc += loss.item()
        step     += 1

        if step % args.log_every == 0:
            elapsed = time.perf_counter() - t_start
            steps_per_sec = step / elapsed
            eta_s = (args.steps - step) / steps_per_sec
            eta_m = eta_s / 60
            print(
                f"step {step:>7,}/{args.steps:,}  "
                f"loss={loss_acc/args.log_every:.4f}  "
                f"({steps_per_sec:.1f} it/s  ETA {eta_m:.0f} min)"
            )
            loss_acc = 0.0

        if step % args.save_every == 0:
            ckpt = OUTPUT_DIR / f"step_{step:07d}"
            policy.save_pretrained(ckpt)
            preprocessor.save_pretrained(ckpt)
            postprocessor.save_pretrained(ckpt)
            print(f"  -> checkpoint saved to {ckpt}")

        if step >= args.steps:
            break

# ---------------------------------------------------------------------------
# Final save
# ---------------------------------------------------------------------------
policy.save_pretrained(OUTPUT_DIR)
preprocessor.save_pretrained(OUTPUT_DIR)
postprocessor.save_pretrained(OUTPUT_DIR)
print(f"\nDone. Final policy saved to {OUTPUT_DIR}")
