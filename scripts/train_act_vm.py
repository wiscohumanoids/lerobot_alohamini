#!/usr/bin/env python3
"""
Self-contained ACT training script for AWS g5.xlarge (A10G, 24 GB VRAM).
Dataset: wiscohumanoids/colored_cube_v7

FAST MODE (recommended): pre-decode videos first, then train from JPEGs on NVMe.
    python3 predecode_dataset.py                        # ~5-10 min, run once
    python3 train_act_vm.py --frames_dir /opt/dlami/nvme/cube_frames ...

SLOW MODE (no pre-decode): streams from video on every step (~1-2 it/s on A10G).
    python3 train_act_vm.py ...

PUSH TO HUB:
    python3 train_act_vm.py --frames_dir /opt/dlami/nvme/cube_frames \\
        --push_to_hub --hf_repo wiscohumanoids/act_colored_cube --hf_token YOUR_TOKEN

PULL BACK TO YOUR MACHINE:
    huggingface-cli download wiscohumanoids/act_colored_cube --local-dir outputs/act_colored_cube
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--steps",       type=int,   default=10_000)
parser.add_argument("--batch_size",  type=int,   default=24,
                    help="A10G fits 24 with 3 cameras. Increase to 32 if VRAM allows.")
parser.add_argument("--lr",          type=float, default=1e-5)
parser.add_argument("--chunk_size",  type=int,   default=100)
parser.add_argument("--output",      type=str,   default="act_colored_cube")
parser.add_argument("--save_every",  type=int,   default=5_000)
parser.add_argument("--log_every",   type=int,   default=100)
parser.add_argument("--frames_dir",  type=str,   default=None,
                    help="Path to pre-decoded JPEG frames (from predecode_dataset.py). "
                         "Much faster than streaming from video.")
parser.add_argument("--push_to_hub", action="store_true")
parser.add_argument("--hf_repo",     type=str,   default=None)
parser.add_argument("--hf_token",    type=str,   default=None)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Install / pin dependencies
# ---------------------------------------------------------------------------
def ensure_packages():
    try:
        import lerobot
        if lerobot.__version__ != "0.4.4":
            print(f"lerobot {lerobot.__version__} found, reinstalling 0.4.4 ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lerobot==0.4.4"])
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lerobot==0.4.4"])
    for pkg in ["huggingface_hub", "av", "pillow"]:
        mod = {"pillow": "PIL", "av": "av", "huggingface_hub": "huggingface_hub"}[pkg]
        try:
            __import__(mod)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

ensure_packages()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import torch
import torch.utils.data
from PIL import Image

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

DATASET_REPO = "wiscohumanoids/colored_cube_v7"
CAMERAS      = ["head_top", "head_front", "wrist_right"]
OUTPUT_DIR   = Path(args.output)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ["LEROBOT_VIDEO_BACKEND"] = "pyav"

# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU : {props.name}  ({props.total_memory // 1024**2} MB VRAM)")
else:
    print("WARNING: no CUDA — training will be extremely slow")

# ---------------------------------------------------------------------------
# Dataset metadata (always needed for stats and fps)
# ---------------------------------------------------------------------------
print(f"\nLoading dataset metadata: {DATASET_REPO}")
meta = LeRobotDatasetMetadata(DATASET_REPO)

input_features = {
    "observation.state":              PolicyFeature(type=FeatureType.STATE,  shape=(16,)),
    "observation.images.head_top":    PolicyFeature(type=FeatureType.VISUAL, shape=(3, 120, 160)),
    "observation.images.head_front":  PolicyFeature(type=FeatureType.VISUAL, shape=(3, 120, 160)),
    "observation.images.wrist_right": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 120, 160)),
}
output_features = {
    "action": PolicyFeature(type=FeatureType.ACTION, shape=(16,)),
}

# ---------------------------------------------------------------------------
# RAM-resident dataset — load everything into memory once, zero I/O at training time
# ---------------------------------------------------------------------------
class RAMDataset(torch.utils.data.Dataset):
    """
    Loads all JPEG frames + state/action into CPU RAM at startup (~8 GB for this dataset).
    After that, __getitem__ is pure tensor slicing — no disk I/O during training.
    GPU stays pegged at 100%.
    """

    def __init__(self, frames_dir: Path, chunk_size: int):
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor

        self.chunk_size = chunk_size

        # --- Load parquet (state + action) ---
        candidates = [
            Path.home() / ".cache" / "huggingface" / "lerobot" / "wiscohumanoids" / "colored_cube_v7",
            Path("/root/.cache/huggingface/lerobot/wiscohumanoids/colored_cube_v7"),
        ]
        data_dir = None
        for c in candidates:
            if (c / "data").exists():
                data_dir = c / "data"
                break
        if data_dir is None:
            raise RuntimeError("lerobot cache not found. Run without --frames_dir once to download.")

        pq_files = sorted(data_dir.rglob("*.parquet"))
        df = pd.concat([pd.read_parquet(p) for p in pq_files])
        df = df.sort_values("index").reset_index(drop=True)
        n  = len(df)

        self.states  = torch.tensor(np.stack(df["observation.state"].values), dtype=torch.float32)  # (N, 16)
        self.actions = torch.tensor(np.stack(df["action"].values),            dtype=torch.float32)  # (N, 16)
        self.ep_idx  = df["episode_index"].values.astype(np.int64)

        # Valid start indices: need chunk_size consecutive frames in same episode
        ep = self.ep_idx
        self.valid = np.array(
            [i for i in range(n - chunk_size) if ep[i + chunk_size - 1] == ep[i]],
            dtype=np.int64,
        )

        # --- Load all images into RAM at half resolution ---
        # Full res (480x640) x3 cams = 111 GB — won't fit on g5.xlarge (16 GB RAM).
        # Half res (240x320) x3 cams = ~7 GB — fits fine.
        # ResNet18 downsamples to 7x10 internally anyway, so half-res loses nothing.
        H, W = 120, 160
        ram_gb = n * 3 * H * W * len(CAMERAS) / 1024**3
        print(f"  Loading {n:,} frames × {len(CAMERAS)} cameras at {H}x{W} into RAM ...")
        print(f"  Estimated RAM: ~{ram_gb:.1f} GB")

        def load_one(args):
            global_idx, cam = args
            path = frames_dir / cam / f"{global_idx:06d}.jpg"
            img  = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
            return np.array(img, dtype=np.uint8)  # (H, W, 3)

        self.images = {}  # cam -> (N, 3, H, W) uint8 tensor
        global_indices = df["index"].values.astype(np.int64)

        for cam in CAMERAS:
            t0 = time.time()
            buf = np.empty((n, H, W, 3), dtype=np.uint8)
            tasks = [(int(global_indices[i]), cam) for i in range(n)]
            with ThreadPoolExecutor(max_workers=16) as ex:
                for i, arr in enumerate(ex.map(load_one, tasks)):
                    buf[i] = arr
                    if (i + 1) % 5000 == 0:
                        print(f"    {cam}: {i+1:,}/{n:,}  ({(i+1)/(time.time()-t0):.0f} frames/s)")
            # (N, H, W, 3) -> (N, 3, H, W), store as uint8 to save RAM
            self.images[cam] = torch.from_numpy(buf.transpose(0, 3, 1, 2))  # (N, 3, H, W)
            print(f"    {cam}: done  ({(time.time()-t0)/60:.1f} min)")

        print(f"  RAM dataset ready: {len(self.valid):,} valid samples")

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        i = int(self.valid[idx])

        images = {
            f"observation.images.{cam}": self.images[cam][i].float() / 255.0
            for cam in CAMERAS
        }

        return {
            **images,
            "observation.state": self.states[i],                          # (16,)
            "action":            self.actions[i : i + self.chunk_size],   # (chunk_size, 16)
            "action_is_pad":     torch.zeros(self.chunk_size, dtype=torch.bool),
        }


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------
if args.frames_dir:
    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        print(f"ERROR: --frames_dir {frames_dir} does not exist.")
        print("Run predecode_dataset.py first.")
        sys.exit(1)
    print(f"\nLoading all frames into RAM from {frames_dir}")
    dataset     = RAMDataset(frames_dir, args.chunk_size)
    num_workers = 0   # data is already in RAM — workers add overhead, not help
    print("Mode: FAST (RAM-resident)")
else:
    print("\nNo --frames_dir given, streaming from video (slow). "
          "Run predecode_dataset.py first for ~5x speedup.")
    delta_timestamps = {
        "observation.images.head_top":    [0.0],
        "observation.images.head_front":  [0.0],
        "observation.images.wrist_right": [0.0],
        "observation.state":              [0.0],
        "action": [i / meta.fps for i in range(args.chunk_size)],
    }
    dataset     = LeRobotDataset(DATASET_REPO, delta_timestamps=delta_timestamps, video_backend="pyav")
    num_workers = 0   # pyav video decoder must run in main process
    print("Mode: SLOW (video decoding per step)")

print(f"Dataset size: {len(dataset):,} samples")

# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
cfg = ACTConfig(
    input_features=input_features,
    output_features=output_features,
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
    optimizer_lr=args.lr,
    optimizer_lr_backbone=args.lr,
    optimizer_weight_decay=1e-4,
    use_amp=True,
    device=device.type,
)

policy = ACTPolicy(cfg)
policy.train()
policy.to(device)

preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=meta.stats)

backbone_params = list(policy.model.backbone.parameters())
backbone_ids    = {id(p) for p in backbone_params}
other_params    = [p for p in policy.parameters() if id(p) not in backbone_ids]

optimizer = torch.optim.AdamW(
    [{"params": other_params,    "lr": args.lr},
     {"params": backbone_params, "lr": args.lr}],
    weight_decay=1e-4,
)
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
    drop_last=True,
    persistent_workers=False,
)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print(f"\nTraining ACT  steps={args.steps:,}  batch={args.batch_size}  "
      f"chunk={args.chunk_size}  lr={args.lr}\n")

step     = 0
t_start  = time.perf_counter()
loss_acc = 0.0

while step < args.steps:
    for batch in dataloader:
        if args.frames_dir:
            # PredecodedACTDataset returns state as (B, 16) already — no squeeze needed
            batch = preprocessor(batch)
        else:
            batch = preprocessor(batch)
            # Video dataset returns state as (B, 1, 16) — squeeze time dim
            if "observation.state" in batch and batch["observation.state"].ndim == 3:
                batch["observation.state"] = batch["observation.state"].squeeze(1)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=(device.type == "cuda")):
            loss, info = policy.forward(batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        loss_acc += loss.item()
        step     += 1

        if step % args.log_every == 0:
            elapsed = time.perf_counter() - t_start
            its     = step / elapsed
            eta_min = (args.steps - step) / its / 60
            print(f"step {step:>7,}/{args.steps:,}  "
                  f"loss={loss_acc/args.log_every:.4f}  "
                  f"({its:.1f} it/s  ETA {eta_min:.0f} min)")
            loss_acc = 0.0

        if step % args.save_every == 0:
            ckpt = OUTPUT_DIR / f"checkpoint_{step:07d}"
            policy.save_pretrained(ckpt)
            preprocessor.save_pretrained(ckpt)
            postprocessor.save_pretrained(ckpt)
            print(f"  -> checkpoint: {ckpt}")

        if step >= args.steps:
            break

# ---------------------------------------------------------------------------
# Final save
# ---------------------------------------------------------------------------
print("\nSaving final policy ...")
policy.save_pretrained(OUTPUT_DIR)
preprocessor.save_pretrained(OUTPUT_DIR)
postprocessor.save_pretrained(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR.resolve()}")

# ---------------------------------------------------------------------------
# Push to Hub
# ---------------------------------------------------------------------------
if args.push_to_hub:
    if not args.hf_repo:
        print("ERROR: --push_to_hub requires --hf_repo")
        sys.exit(1)
    from huggingface_hub import HfApi
    token = args.hf_token or os.environ.get("HF_TOKEN")
    api   = HfApi(token=token)
    print(f"\nUploading to {args.hf_repo} ...")
    api.create_repo(args.hf_repo, repo_type="model", exist_ok=True, private=True)
    api.upload_folder(folder_path=str(OUTPUT_DIR), repo_id=args.hf_repo, repo_type="model")
    print(f"Done. Pull back with:")
    print(f"  huggingface-cli download {args.hf_repo} --local-dir outputs/act_colored_cube")
