# 06 — Training and Evaluation

## The three-stage training stack (any modern open VLA)

```
Stage 1: VLM pretraining     (done by Google/HF, you don't redo this)
         PaliGemma / Llama+SigLIP / Eagle-2.5 / SmolVLM
                ↓
Stage 2: VLA pretraining     (done by OpenVLA / PI / NVIDIA, optionally redone)
         OXE + DROID + ... → generalist policy
                ↓
Stage 3: Per-task / per-robot fine-tune  (this is YOU)
         ~10–200 demos → deployed policy
```

Stage 3 is what you actually do 95% of the time. Sometimes stage 2 (rare; you'd need OXE-scale compute).

## Stage 3: fine-tuning — the practical recipe

### OpenVLA + OFT (recommended for OpenVLA-family work)

[Project](https://openvla-oft.github.io/) · code in [openvla/openvla-oft](https://github.com/moojink/openvla-oft).

Default knobs:
- **LoRA rank 32** on attention + MLP (the standard OpenVLA LoRA setup).
- **Batch size 8–32** per GPU (7B model + LoRA fits on a single A100/H100).
- **Learning rate 5e-4** for LoRA params; constant or cosine.
- **Action chunk k=8**, parallel decode, **L1 loss**.
- **Steps**: 30k–150k typically. Plateaus depend on data size; monitor LIBERO or held-out success.
- **Mixed precision**: bf16 throughout.

### π0 / π0-FAST fine-tune via openpi

[Code](https://github.com/Physical-Intelligence/openpi).

Same recipe pattern (LoRA on the VLM backbone; full fine-tune on the action expert). Defaults in the openpi configs are sensible — don't tune them on the first run. Use the provided LeRobot dataset loader.

### SmolVLA fine-tune via LeRobot

[Docs](https://huggingface.co/docs/lerobot/smolvla) · [phospho tutorial](https://docs.phospho.ai/learn/train-smolvla).

CLI-level (note: `--policy.path` loads weights + sets type automatically; do **not** pass `--policy.type` separately):
```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=USER/your-dataset \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/smolvla_yourtask \
  --policy.device=cuda
```

~4 hours on a single A100 for 20k steps. Hub-native; supports `push_to_hub`.

Sim eval (LIBERO protocol — 10 episodes/task, single-env; note the `checkpoints/last/pretrained_model` suffix — `lerobot-eval` expects the saved model dir, not the run dir):
```bash
lerobot-eval \
  --policy.path=outputs/train/smolvla_yourtask/checkpoints/last/pretrained_model \
  --env.type=libero --env.task=libero_spatial \
  --eval.n_episodes=10 --eval.batch_size=1 \
  --env.max_parallel_tasks=1
```

**Important: use the `HuggingFaceVLA/libero` dataset for LIBERO training.** Other LIBERO LeRobot uploads (e.g., `lerobot/libero_spatial_image`) use the feature key `observation.images.wrist_image`, but the LIBERO eval env produces `observation.images.image2` — schema mismatch will crash eval. `HuggingFaceVLA/libero` is preprocessed to match the eval env. (Per [LeRobot LIBERO docs](https://huggingface.co/docs/lerobot/libero).)

### Hyperparameters that matter (in priority order)

1. **Action normalization statistics** — must be computed on your dataset. Common bug source.
2. **Chunk length** — match what you'll deploy with. Don't train k=8 and eval k=50.
3. **LoRA rank and where to apply it** — 16–64 on attention + MLP is typical. Full fine-tune only when you have lots of data.
4. **Steps / data ratio** — 1–5 epochs is usually enough; many epochs overfit.
5. **Vision augmentations** — color jitter, brightness, slight crops. Helps generalization to lighting changes. Don't over-augment, you'll hurt precise tasks.
6. **Image resolution** — match the base model (224 or 256 for most). Higher resolutions rarely worth the cost.

### What NOT to tune on first run

Backbone choice, action head architecture, loss function. Trust the released config; debug the data first.

## Co-training (the π0.5 / Gemini Robotics trick)

To preserve VLM web priors during VLA training, mix non-robot data into the batch:
- Web VQA / image-captioning batches.
- Object detection.
- Spatial reasoning / point prediction (for ER-style training).

Typical ratio: 50–80% robot data, 20–50% web/VQA. Without co-training, the VLM forgets — semantic generalization to unseen objects degrades after enough action data.

**Knowledge insulation** (π0.5) is the harder version: detach action-expert gradients from the VLM backbone entirely, so the VLM stays a clean VLM while the action expert specializes. This decouples training and lets web data + action data be trained on different schedules.

## RL post-training (2025+ trend)

[SimpleVLA-RL (ICLR 2026)](https://github.com/PRIME-RL/SimpleVLA-RL) · [VLA-RFT](https://openreview.net/forum?id=Jaut99EHeu) · [RLinf](https://github.com/RLinf/RLinf)

After SFT, you can fine-tune with RL to fix failure modes that BC won't address:
- **Action-chunked PPO** ([arXiv 2509.25718](https://arxiv.org/pdf/2509.25718)) — adapts PPO to operate on action chunks instead of per-step.
- **VLA-RFT** — RL fine-tuning with verified rewards from world simulators; <400 steps to beat SFT baseline.
- **WoVR** (RLinf integration) — world-model-based RL post-training.

When to consider it:
- SFT plateaued and you have a sim or world model.
- You need robustness to states the BC data doesn't cover well.
- You have engineering capacity to run RL.

When to skip: first prototype, scarce GPU time, no sim. Get SFT working first.

## Evaluation: the part everyone screws up

### Three levels of eval

| Level | Cost | Signal |
|---|---|---|
| **LIBERO / sim benchmark** | minutes | Sanity check; often inflated |
| **SimplerEnv / matched-domain sim** | hours | Real predictor of real-world; the right pre-deployment gate |
| **Real robot, held-out conditions** | days | Ground truth |

### How to run a real eval honestly

- **Test on held-out object instances**, not the same teddy bear from training.
- **Vary initial robot pose** within the natural range.
- **Vary lighting** (different time of day, different overhead lamps).
- **Random distractor objects** in the scene.
- **At least 20–30 trials per condition** for usable success-rate estimates. To reliably distinguish a 60% policy from an 80% policy at 95% confidence, you actually need ~80–100 trials — 20–30 only catches large differences.
- **Report per-condition breakdown**, not just overall mean. The mean hides the failure mode you need to fix.

### Common eval anti-patterns to avoid

- Training and evaluating on the same scene composition. You're measuring memorization.
- Cherry-picking the "good" run from 3 attempts and reporting that.
- Reporting LIBERO 95%+ and concluding the model "works." Read LIBERO-PRO/Plus first.
- No proprioception failures: many "model X fails" reports are actually action-norm or proprio-units bugs.

### Metrics you actually care about

- **Success rate** per task, per condition.
- **Time to success / efficiency** — even when both succeed, faster is real progress.
- **Recovery rate from intentional perturbations** (e.g., move the object mid-grasp).
- **Generalization gap**: train-set success – held-out success. Smaller is better.

## Loss curves and diagnostics

Watch:
- **Action MSE on held-out demos** — primary BC quality signal.
- **Per-dim loss breakdown** — gripper often dominates if not weighted; ensure it doesn't suppress xyz/rpy.
- **Image embedding norm / VLM perplexity on web data** (if co-training) — catches catastrophic forgetting of the VLM.
- **Eval success rate every N steps** — train loss can keep dropping while real eval plateaus. Don't trust train loss alone.

## Time / compute budget for a typical Stage-3 run

- **One task, ~50 demos, SmolVLA fine-tune**: 4–8 hours on 1× A100.
- **One task, ~50 demos, OpenVLA-OFT LoRA**: 8–24 hours on 1× A100.
- **Multi-task, ~500 demos, π0 fine-tune**: 1–3 days on 4–8× A100.
- **OXE-scale pretrain (Stage 2)**: weeks on 64+ GPUs. You probably won't do this.

If a run takes >24 hours on consumer GPUs, you should be using a smaller model or distilled variant. The field has converged hard on "small models + fast iteration" as the better engineering bet for non-foundation-lab teams.
