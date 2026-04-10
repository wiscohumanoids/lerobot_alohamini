# Hyperparameter Sweeps for AlohaMini

This guide explains how to run a Weights & Biases (W&B) hyperparameter
sweep over ACT on AlohaMini datasets, and how to turn the results into
empirically-backed training defaults for the team.

## Why sweep

The ACT defaults baked into `lerobot-train` come from the original ALOHA
paper. That paper targeted 6-DOF Trossen arms, a fixed overhead + wrist
camera layout, and Dynamixel servos with very different dynamics from
the 5-DOF Feetech-based arms on AlohaMini. The paper's `kl_weight=10`,
`dim_model=512`, `chunk_size=100`, and `optimizer_lr=1e-5` are a
reasonable starting point but nobody has validated them on our
platform. A single-day Bayesian sweep turns "the paper said so" into
"we checked, and here is what actually works on AlohaMini."

## One-time setup

Install W&B inside your LeRobot virtual environment and log in:

```
pip install wandb
wandb login
```

You will be prompted for an API key. Grab it from
https://wandb.ai/authorize . After login, `wandb` writes the key to
`~/.netrc` and subsequent `wandb agent` invocations are non-interactive.

## Launching a sweep

The sweep config lives at `configs/sweeps/act_alohamini.yaml`. Register
it with W&B and get back a sweep ID:

```
wandb sweep configs/sweeps/act_alohamini.yaml
```

W&B prints something like:

```
wandb: Creating sweep from: configs/sweeps/act_alohamini.yaml
wandb: Created sweep with ID: abc12345
wandb: View sweep at: https://wandb.ai/<you>/lerobot/sweeps/abc12345
wandb: Run sweep agent with: wandb agent <you>/lerobot/abc12345
```

Start an agent on the GPU that will actually run the trials:

```
wandb agent <you>/lerobot/abc12345
```

The agent pulls hyperparameter assignments from the sweep scheduler,
invokes `lerobot-train` with those flags, and reports `train/loss` back
to W&B. It keeps running until the sweep's `run_cap` (20) is hit or you
Ctrl-C it.

### Overriding the dataset per launch

The config hardcodes `dataset.repo_id` to
`wiscohumanoids/alohamini_cube_place_5dof_v38` so that `wandb sweep`
succeeds with no extra flags. To sweep against a different dataset, the
cleanest option is to edit the `value:` on `dataset.repo_id` in the YAML
before running `wandb sweep`. The second-cleanest option is to create a
copy of the config with the new dataset and register that instead:

```
cp configs/sweeps/act_alohamini.yaml configs/sweeps/act_my_dataset.yaml
# edit dataset.repo_id in the copy
wandb sweep configs/sweeps/act_my_dataset.yaml
```

Do not try to override `dataset.repo_id` by passing an extra flag to
`wandb agent` directly; the sweep scheduler owns that parameter once the
sweep is registered.

### Running multiple agents in parallel

A sweep ID is just a queue. You can attach more than one agent to the
same sweep, each on its own GPU or its own machine:

```
# Machine A
wandb agent <you>/lerobot/abc12345

# Machine B
wandb agent <you>/lerobot/abc12345
```

Each agent pulls one trial at a time from the shared scheduler, so
throughput scales roughly linearly with the number of GPUs. The
Bayesian optimizer takes the growing set of finished trials into
account when picking the next assignment, so parallelism does not hurt
sample efficiency the way random search with duplicates would.

## Parameter explanations

| Parameter | Meaning | Why it matters on AlohaMini |
|---|---|---|
| `policy.chunk_size` | How many actions ACT predicts per forward pass. Swept over `[10, 25, 50, 100]`. | Larger chunks give smoother trajectories but make the policy slower to react to surprises. 5-DOF arms with lighter payloads can often tolerate smaller chunks than 6-DOF ALOHA. |
| `policy.kl_weight` | Weight on the KL term in the CVAE loss: `reconstruction_loss + kl_weight * kld_loss`. Log-uniform over `[1.0, 100.0]`. | Higher values push the VAE toward more diverse latent codes (good for multimodal demos); lower values make the policy more deterministic (good for precise manipulation). |
| `policy.optimizer_lr` | Learning rate for the AdamW optimizer on the main transformer. Log-uniform over `[1e-5, 1e-3]`. | The paper default (`1e-5`) was tuned for much larger datasets than AlohaMini typically sees. Sweeping the log range usually finds a better optimum for a few-hour dataset. |
| `policy.dim_model` | Transformer hidden dimension. Swept over `[256, 512]`. | 512 is the paper default. 256 trains roughly 2x faster and often generalizes better on smaller datasets. |
| `batch_size` | Gradient batch size at the `TrainPipelineConfig` level. Swept over `[4, 8, 16]`. | Limited by GPU memory. Larger batches give more stable gradients; smaller batches let you fit more image resolution. |

Everything else (`policy.type=act`, `policy.device=cuda`,
`steps=30000`, `save_freq=30000`, `log_freq=100`, `wandb.enable=true`,
`dataset.video_backend=pyav`) is held fixed so every trial is
comparable.

## Interpreting results

Open the sweep dashboard at the URL `wandb sweep` printed. Two views
are worth your time:

1. **Parallel coordinates plot.** Each trial is a line threading
   through all the hyperparameter axes. Color the lines by
   `train/loss`. The low-loss lines cluster in the region of the search
   space that works; the high-loss lines reveal dead zones to avoid.

2. **Hyperparameter importance panel.** W&B fits a quick random forest
   on top of the trial results and ranks which swept parameters
   actually moved the needle on `train/loss`. On AlohaMini we typically
   see `optimizer_lr` and `kl_weight` dominate; `dim_model` is usually
   a distant third.

Do not just pick the single lowest `train/loss` trial. Look for the
Pareto frontier of `train/loss` vs. wall-clock time — a trial that
reaches within 5% of the best loss in half the training time is often
the better default, because you will run the final training command
many times with different datasets.

## Combining with `eval_metrics.csv`

Train loss is necessary but not sufficient. A policy can overfit the
demo distribution, post a great `train/loss`, and still jitter the
robot in the real world. The eval pipeline
(`src/lerobot/examples/alohamini/evaluate_bi.py`, see PR #4) writes
per-episode metrics to `eval_metrics.csv`, including `action_jerk_rms`
which correlates directly with how comfortable the robot looks at
runtime.

Workflow for declaring a sweep winner:

1. From the sweep dashboard, pick the top three or four trials by
   `train/loss`.
2. For each, re-run `lerobot-train` with the winning hyperparameters
   for the full training schedule (50K+ steps) and save the checkpoint.
3. Run each checkpoint through `evaluate_bi.py` on a held-out task and
   collect the resulting `eval_metrics.csv`.
4. Rank the candidates by `action_jerk_rms` (and any task-success
   metric you care about), not by `train/loss`. The lowest-jerk policy
   that is still task-competent wins and becomes the new team default.

## Cost note

Rough back-of-the-envelope for a single agent on a single GPU:

- 20 trials per sweep (see `run_cap` in the config)
- ~30,000 gradient steps per trial (shorter than a full training run
  on purpose — enough to separate good from bad hyperparameters)
- batch size 4-16, `dim_model` 256-512, chunk size 10-100

On a mid-range NVIDIA GPU that ACT can actually use (an RTX 3090 or
4090, an A4000, or better), a single trial takes roughly 30-60 minutes
and a full 20-trial sweep lands somewhere in the **10-20 GPU-hour**
range. Hyperband early-termination (configured in
`configs/sweeps/act_alohamini.yaml`) can shave 20-40% off that by
killing clearly-losing trials early.

Plan the sweep for overnight on a single workstation, or split it
across two or three agents in parallel and get results back within a
couple of hours. Either way, the 10-20 GPU-hour budget is a one-time
spend per platform change (new servos, new cameras, new arm) — once you
have AlohaMini-specific defaults checked in, normal training runs do
not need to sweep again.
