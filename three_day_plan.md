# VLA: 3-Day Zero-to-Hero Sprint

For a research engineer with **solid ML/DL + transformers**, no robotics background, and an **end goal of deploying a VLA on a real robot**.

**Time budget assumed:** ~10 hours/day for 3 days. Total: ~30 hours.
**Hardware:** one strong GPU (A100 ideal, RTX 4090 OK). Real arm optional — you'll work in sim if not available.

The aggressive cut: only the **load-bearing facts and one end-to-end run-through**. No fluff, no parallel reading paths. Every minute is allocated.

---

## Day 1 — Foundations + read OpenVLA + first sim run

**Goal:** end the day with a complete mental model and a running BC policy in sim.

### Morning (4 hours) — Foundations

Read **in this order, no skipping**:

1. `reference/00_landscape.md` (45 min). Memorize the 7-axis table and the 2023–2026 timeline.
2. `reference/01_robotics_primer.md` (30 min). Internalize: action space conventions, control loop, action chunking.
3. `reference/02_imitation_learning.md` (45 min). Three patterns (BC, ACT, Diffusion Policy) + four action heads.
4. **OpenVLA paper** — <https://arxiv.org/abs/2406.09246> (120 min). Sections 1–5 carefully; ablations skim. Chase down unfamiliar terms (RLDS, Prismatic VLM, action token hijack) as you go.

By lunch you should be able to draw the OpenVLA architecture and explain action tokenization in 60 seconds.

### Afternoon (4 hours) — Install + first run

- Install LeRobot from source (LIBERO extras require it). Fresh `uv` env:
  ```bash
  git clone https://github.com/huggingface/lerobot.git && cd lerobot
  uv venv && source .venv/bin/activate
  uv pip install -e ".[smolvla,libero]"
  uv pip install --upgrade mani-skill  # optional, only if you want ManiSkill envs
  export MUJOCO_GL=egl                  # headless rendering for LIBERO
  ```
- Pull a small LeRobot dataset and **visualize it** (the visualization notebook in `lerobot/examples/`).
- Sanity-check the LIBERO env loads (`python -c "from libero.libero import benchmark; print(benchmark.get_benchmark_dict())"`).
- Launch a **5k-step smoke run** of Diffusion Policy on a LIBERO-Spatial task in `tmux`, logs to `logs/day1_$(date +%H%M).log`. Goal: prove the pipeline works, not get a strong result. Kill it before Day 2's run starts.
  ```bash
  tmux new -s vla_d1
  lerobot-train --policy.type=diffusion \
    --dataset.repo_id=lerobot/libero_spatial_no_noops \
    --steps=5000 --output_dir=outputs/day1_dp \
    2>&1 | tee logs/day1_dp_$(date +%H%M).log
  ```

### Evening (2 hours) — Architectures + action heads

Read in order:
1. `reference/03_vla_architectures.md` end-to-end.
2. `reference/04_action_representations.md` (30 min skim; you mostly need the decision tree).
3. **π0 paper** — <https://www.pi.website/download/pi0.pdf>. Sections 2–4 only. Stop when you understand:
   - Why the action expert is separate from the VLM.
   - What flow matching does in 5–10 steps that a deterministic regression head can't.

Before bed: **kill the Day 1 training run** (`tmux kill-session -t vla_d1`). Day 2 needs the GPU.

**End-of-day checkpoint:** Can you answer (yes/no, out loud):
- What does the OpenVLA action tokenizer do? (Hijacks lowest-freq Llama tokens, 256 bins/dim, 7 dims, AR decode.)
- Why action chunking? (Reduces effective horizon for compounding error; amortizes inference.)
- π0 vs OpenVLA — which is faster at inference and why? (π0 with chunked flow-matching: fewer forward passes than AR token decode.)

If you can't answer all three, re-read before bed.

---

## Day 2 — Datasets, training, and a real fine-tune

**Goal:** fine-tune a VLA on a task you care about. By end of day, you have a checkpoint.

### Morning (3 hours) — Datasets and training recipe

Read:
1. `reference/05_datasets.md` (45 min). Know OXE / DROID / LIBERO / LeRobot format.
2. `reference/06_training_and_eval.md` (45 min). Recipe details + eval honesty.
3. **OpenVLA-OFT paper** — <https://arxiv.org/abs/2502.19645> (90 min). Read carefully; this is the recipe you'll use.

### Midday (1 hour) — Pick your stack

Decision (snap, don't deliberate):
- **Default**: SmolVLA fine-tune on LIBERO-Spatial. Fits the day.
- **Real SO-ARM with pre-collected teleop data**: SmolVLA on your data. Only viable if you already have ~30–50 demos. If not, stay on LIBERO.

(OpenVLA-OFT LoRA needs 8–24 h on A100 per `reference/06_training_and_eval.md` — it does not fit the 3-day plan. Save it for the 1-week plan.)

### Afternoon (5 hours) — Fine-tune

In `tmux` (a 20k-step SmolVLA run is ~4 h on A100; longer on RTX 4090):
```bash
tmux new -s vla_d2
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/libero_spatial_no_noops \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/day2_smolvla \
  --job_name=day2_smolvla \
  --policy.device=cuda \
  2>&1 | tee logs/day2_smolvla_$(date +%H%M).log
```

While training runs, do these in parallel (NOT additive to the 5 h budget):
- Read `reference/08_deployment.md` end-to-end.
- Read **FAST paper** — <https://arxiv.org/abs/2501.09747>, method only.
- Read **π0.5 abstract + intro** — <https://arxiv.org/abs/2504.16054>.
- Sketch the async inference server pattern you'd use to deploy your model.

### Evening (1 hour) — Eval

Once trained, run LIBERO eval:
```bash
lerobot-eval \
  --policy.path=outputs/day2_smolvla \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=20 \
  --eval.batch_size=2
```
Note per-task success rate. If drastically off published, diagnose in this order: (1) action normalization stats, (2) image preprocessing, (3) action convention/units, (4) model. The pipeline is wrong before the model is wrong.

**End-of-day checkpoint:** Working checkpoint + numbers. If success rate is wildly off published, the *pipeline* is broken — debug normalization, image preprocessing, and action conventions before assuming the model is bad.

---

## Day 3 — Deployment, ecosystem, frontier

**Goal:** know how to deploy real, what's frontier, and where to go next.

### Morning (3 hours) — Hardware + deployment

Read:
1. `reference/07_hardware_and_sim.md` (45 min). Hardware mental map.
2. **Implement the inference loop** in code (90 min). Use LeRobot's eval scaffold or write your own:
   - Load checkpoint.
   - Loop: read image → predict chunk → execute chunk[:m] → loop.
   - Add a stopwatch around inference. Log p50, p95 latency.
   - Add chunk buffering so the next inference starts before the current chunk runs out.
3. (45 min) **If you have a real arm**: deploy the loop on the real robot. Run 10 trials. *Calibration assumed done the night before — budget 1–2h on calibration the first time.* Otherwise: deploy in sim with longer rollouts than during training.

### Midday (2 hours) — Ecosystem and frontier

Read:
1. `reference/09_ecosystem_and_sources.md` (45 min).
2. **GR00T N1 paper** — <https://arxiv.org/abs/2503.14734>, sections 1–3 (75 min). Understand the dual-system architecture and the human-video pretraining angle.

### Afternoon (3 hours) — One frontier topic, your pick

Pick ONE and read deeply:
- **RL post-training**: SimpleVLA-RL — <https://github.com/PRIME-RL/SimpleVLA-RL>, paper + repo.
- **Open-world generalization**: π0.5 paper full read — <https://arxiv.org/abs/2504.16054>.
- **Humanoid VLA**: GR00T N1.5 + Helix announcement — <https://research.nvidia.com/labs/gear/gr00t-n1_5/> + <https://www.figure.ai/news/helix>.
- **Cognition-action separation**: Gemini Robotics 1.5 — <https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/>.

If you'd be answering "how do we make VLAs robust" at work next week — pick RL or π0.5. If your stack is humanoid — GR00T. If your stack is long-horizon agentic — Gemini Robotics 1.5.

### Evening (2 hours) — Consolidate

- Write a single-page brief (`SUMMARY.md`): the 7-axis design space, your model choice and why, your fine-tune numbers, your deployment latency budget, the top 3 open problems you'd work on next.
- Browse one of the awesome-VLA repos for ~30 minutes. Star 5 papers from the last 3 months.

---

## Compressed-day pitfalls (where 3-day sprints break)

1. **The fine-tune doesn't converge by end of Day 2.** Causes, in probability order: action normalization wrong; dataset format mismatch; wrong learning rate (defaults are usually fine); too few demos for SmolVLA. **Spend the first hour of Day 3 debugging this** before continuing.
2. **You skip reading a paper in favor of "more coding."** Don't. The coding mostly works; the bugs come from misunderstanding the *spec*. Time spent on the OpenVLA paper saves time on the OpenVLA codebase.
3. **You skip the inference loop on Day 3.** This is where the hardest debugging lives in real deployments. Even sim-only, write the loop.
4. **You try to learn two action heads at once.** Don't. Pick OFT *or* π0 for the deep dive; the other you'll catch later.
5. **You over-research before installing.** Get LeRobot installed in the first hour of Day 1. If install breaks, fix that *now*, not Day 2.

## What I'd do on Day 4 if I had it

Run one paper reproduction (fine-tune on a published dataset, hit ≥80% of the published number). This pins down that your pipeline matches the field's; future debugging gets dramatically easier.
