#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_ENV_STATE


def make_delta_timestamps(cfg: ACTConfig, fps: int) -> dict[str, list[float]]:
    delta_timestamps = {
        "action": [i / fps for i in cfg.action_delta_indices],
    }

    for key in cfg.image_features:
        delta_timestamps[key] = [0.0]

    if cfg.robot_state_feature is not None:
        delta_timestamps["observation.state"] = [0.0]

    if cfg.env_state_feature is not None:
        delta_timestamps[OBS_ENV_STATE] = [0.0]

    return delta_timestamps


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a conditioned ACT policy on an AlohaMini dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset repo_id, e.g. user/tictactoe_pick")
    parser.add_argument("--root", type=Path, default=None, help="Optional local dataset root.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--policy_repo_id", type=str, default=None, help="Optional HF model repo to push to.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--n_action_steps", type=int, default=50)
    parser.add_argument("--optimizer_lr", type=float, default=1e-5)
    parser.add_argument("--optimizer_lr_backbone", type=float, default=1e-5)
    parser.add_argument("--push_to_hub", action="store_true", default=False)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_metadata = LeRobotDatasetMetadata(args.dataset, root=args.root)
    features = dataset_to_policy_features(dataset_metadata.features)

    input_features = {key: ft for key, ft in features.items() if ft.type is not FeatureType.ACTION}
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}

    if OBS_ENV_STATE not in input_features:
        raise ValueError(
            f"Dataset {args.dataset} is missing {OBS_ENV_STATE}. "
            "Record with record_conditioned_bi.py so the conditioning vector is stored."
        )

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        device=args.device,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        optimizer_lr=args.optimizer_lr,
        optimizer_lr_backbone=args.optimizer_lr_backbone,
        repo_id=args.policy_repo_id,
    )
    policy = ACTPolicy(cfg)
    policy.train()
    policy.to(torch.device(cfg.device))

    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)
    delta_timestamps = make_delta_timestamps(cfg, dataset_metadata.fps)
    dataset = LeRobotDataset(args.dataset, root=args.root, delta_timestamps=delta_timestamps)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=cfg.device != "cpu",
        drop_last=True,
    )

    optimizer = cfg.get_optimizer_preset().build(policy.get_optim_params())

    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % args.log_freq == 0:
                print(
                    f"step={step} loss={loss.item():.4f} "
                    f"l1={loss_dict.get('l1_loss', 0.0):.4f} "
                    f"kld={loss_dict.get('kld_loss', 0.0):.4f}"
                )

            if step > 0 and step % args.save_freq == 0:
                ckpt_dir = args.output_dir / f"checkpoint_{step:06d}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(ckpt_dir)
                preprocessor.save_pretrained(ckpt_dir)
                postprocessor.save_pretrained(ckpt_dir)

            step += 1
            if step >= args.steps:
                done = True
                break

    policy.save_pretrained(args.output_dir)
    preprocessor.save_pretrained(args.output_dir)
    postprocessor.save_pretrained(args.output_dir)

    env_feature = dataset_metadata.features[OBS_ENV_STATE]
    with open(args.output_dir / "conditioning_schema.json", "w") as f:
        json.dump(
            {
                "feature_name": OBS_ENV_STATE,
                "shape": list(env_feature["shape"]),
                "names": list(env_feature["names"]),
                "dataset_repo_id": args.dataset,
            },
            f,
            indent=2,
        )

    if args.push_to_hub:
        if not args.policy_repo_id:
            raise ValueError("--policy_repo_id is required when --push_to_hub is set.")
        policy.push_to_hub(args.policy_repo_id)
        preprocessor.push_to_hub(args.policy_repo_id)
        postprocessor.push_to_hub(args.policy_repo_id)


if __name__ == "__main__":
    main()
