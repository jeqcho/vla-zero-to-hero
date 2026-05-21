# VLA: 1-Week Zero-to-Hero Plan

For a research engineer with **solid ML/DL + transformers**, no robotics background, and an **end goal of deploying a VLA on a real robot**.

**Time budget assumed:** 5–7 hours/day for 7 days. Total: ~40 hours.
**Hardware assumed:** one consumer GPU (RTX 4080+ ideal, A100 great). For days 5–7, optionally a real arm (SO-ARM100/101 if you can buy one in ~3 days).

This curriculum is structured so each day produces a deliverable. **If a day's deliverable doesn't work, finish it the next morning before moving on.** Do not skip the deliverable.

---

## Day 1 — Field map and robotics primer

**Goal:** know what a VLA is, the design space, and the minimum robotics vocabulary.

**Read** (3 hours):
- `reference/00_landscape.md` (full)
- `reference/01_robotics_primer.md` (full)
- OpenVLA blog post / abstract: <https://openvla.github.io/>

**Do** (2 hours):
- Install LeRobot from source in a fresh `uv` env (LIBERO and SmolVLA need source-install extras):
  ```bash
  git clone https://github.com/huggingface/lerobot.git && cd lerobot
  uv venv && source .venv/bin/activate
  uv pip install -e ".[smolvla,libero]"
  export MUJOCO_GL=egl   # headless rendering for LIBERO
  ```
  *Note: LeRobot's `[libero]` extra is Linux-only. On macOS, install LIBERO manually from <https://github.com/Lifelong-Robot-Learning/LIBERO> or substitute ManiSkill (`uv pip install --upgrade mani-skill`).*
- Walk through LeRobot's "Visualize a Dataset" notebook on any small LeRobot dataset (e.g., `lerobot/aloha_sim_insertion_human`). Confirm you can see images, actions, and a rollout.
- Sanity-check the LIBERO env loads: `python -c "from libero.libero import benchmark; print(benchmark.get_benchmark_dict())"`.
- Sketch the (obs → policy → action) loop on paper. Identify: action dim, control frequency, chunk length, normalization. **You should be able to draw this from memory at the end of the day.**

**Deliverable:** a one-page Markdown note in `notes/day1.md` listing the 7 VLA architecture axes (backbone / head / chunking / S1S2 / action space / proprio / cameras), with one example model for each.

---

## Day 2 — Imitation learning + action representations

**Goal:** understand BC, ACT, Diffusion Policy, and the four action-head patterns from first principles.

**Read** (3 hours):
- `reference/02_imitation_learning.md`
- `reference/04_action_representations.md`
- **ACT paper** — <https://arxiv.org/abs/2304.13705>. Skim sections 1–4 carefully; you can skim ablations.
- **Diffusion Policy** — <https://arxiv.org/abs/2303.04137>. Read section 3 (method); skim experiments.

**Do** (2–3 hours):
- Run ACT or Diffusion Policy training on a LIBERO-Spatial task via LeRobot:
  ```bash
  lerobot-train --policy.type=diffusion --env.type=pusht --steps=10000
  ```
  (or substitute any small LeRobot env on your machine).
- Watch loss curve. Note: action L1/MSE plateaus first; success rate continues to climb.
- **Optional but recommended**: implement a 30-line flow-matching trainer for a toy 2D action distribution. Internalize what the velocity field is.

**Deliverable:** `notes/day2.md` answering, in your own words:
- Why does action chunking work?
- Why does diffusion/flow handle multimodal actions better than MSE?
- What is FAST and why is it cheaper than per-dim binning?

---

## Day 3 — The VLA model zoo and pretraining

**Goal:** know the headline architectures, what's open, what to pick.

**Read** (4 hours):
- `reference/03_vla_architectures.md`
- `reference/05_datasets.md`
- **OpenVLA paper** — <https://arxiv.org/abs/2406.09246>. Full read. Pay attention to: action tokenization scheme, fine-tuning recipe, evaluation methodology.
- **π0 paper** — <https://www.pi.website/download/pi0.pdf>. Sections 2–4. Pay attention to: action expert architecture, flow-matching training.
- **π0.5 paper** (skim) — <https://arxiv.org/abs/2504.16054>. Focus on knowledge insulation + co-training.

**Do** (2 hours):
- Download OpenVLA-7B from HF: `from transformers import AutoModelForVision2Seq; model = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b")`.
- Inspect the architecture (`print(model)`). Find: vision encoder, language model, action token embedding hijack.
- Try one forward pass on a single LIBERO frame; decode the predicted action; print it next to the ground-truth action.

**Deliverable:** a comparison table in `notes/day3.md`: OpenVLA vs. π0 vs. GR00T N1.5 vs. SmolVLA on (backbone, action head, params, training data, when you'd pick it).

---

## Day 4 — Training and evaluation in practice

**Goal:** fine-tune your first VLA. Today is heavy on the deliverable.

**Read** (1.5 hours):
- `reference/06_training_and_eval.md`
- **OpenVLA-OFT paper** — <https://arxiv.org/abs/2502.19645>. Method section (sections 3–4). Understand: parallel decoding, L1 loss, chunking.
- **FAST paper** — <https://arxiv.org/abs/2501.09747>. Skim section 3 on DCT+BPE pipeline.

**Do** (4–5 hours):
- Set up `tmux` for the long-running fine-tune. Logs to `logs/finetune_$(date +%Y%m%d_%H%M%S).log`.
- Choose **one** of:
  - **(easier, fits the day)** Fine-tune SmolVLA on a single LIBERO task (~4 h on A100, longer on RTX 4090).
    ```bash
    lerobot-train --policy.path=lerobot/smolvla_base \
      --dataset.repo_id=HuggingFaceVLA/libero \
      --batch_size=64 --steps=20000 \
      --output_dir=outputs/day4_smolvla --policy.device=cuda
    ```
  - **(real-stakes, overnight)** Run OpenVLA-OFT LoRA fine-tune on a LIBERO task (8–24 h on A100). Reference: <https://openvla-oft.github.io/> · <https://github.com/moojink/openvla-oft>. Start it now; check tomorrow morning.
- Run eval on LIBERO (LIBERO protocol: 10 episodes/task, single-env):
  ```bash
  lerobot-eval \
    --policy.path=outputs/day4_smolvla/checkpoints/last/pretrained_model \
    --env.type=libero --env.task=libero_spatial \
    --eval.n_episodes=10 --eval.batch_size=1 \
    --env.max_parallel_tasks=1
  ```
- Read LIBERO-PRO (<https://arxiv.org/abs/2510.03827>) to understand why LIBERO numbers are over-optimistic.

**If success rate is wildly off published**, the *pipeline* is wrong before the *model* is. Debug in this order: (1) action normalization stats, (2) image preprocessing, (3) action convention/units (EEF vs joint, frame), (4) gripper polarity, (5) chunk-length mismatch train vs eval. Only then assume the model needs more training.

**Deliverable:** working fine-tune checkpoint + LIBERO success-rate table (per-task) in `notes/day4.md`. A failure analysis paragraph: which task(s) did worst and why.

---

## Day 5 — Simulators, hardware, and data collection

**Goal:** stand up a simulator and (if hardware available) a teleop pipeline.

**Read** (1.5 hours):
- `reference/07_hardware_and_sim.md`
- LeRobot SO-100 walkthrough: <https://huggingface.co/docs/lerobot/en/so100>
- ALOHA hardware appendix (skim photos and the leader-follower description; you already read the method on Day 2) — <https://arxiv.org/abs/2304.13705>.

**Do** (5 hours):

**Path A — Real hardware (if SO-ARM101 arrived):**
- Assemble (if not pre-assembled). Run LeRobot calibration (`lerobot-calibrate`). Budget 1–2 h the first time.
- Record ~30 demos of a simple pick-and-place via `lerobot-record` with leader-follower teleop.
- Push the dataset to HF Hub.

**Path B — Sim only (no real arm):**
- Easiest path: collect demos in LIBERO directly. Use the LIBERO-Spatial demo set already loaded on Day 4 as a template, or record new scripted demos in any LIBERO task you choose.
- (Optional, skip if MimicGen install gives trouble — it can eat hours): try MimicGen to scale a small seed set. See <https://github.com/NVlabs/mimicgen>.
- Convert to LeRobot dataset format (see LeRobot docs).

**Deliverable:** a LeRobot-format dataset (30+ episodes) of *your* task — sim or real — sitting in HF Hub or local disk. `notes/day5.md` with the dataset card.

---

## Day 6 — Fine-tune on your task

**Goal:** fine-tune on your own data. Today is GPU-heavy, not loop-heavy.

**Read** (1 hour):
- `reference/08_deployment.md` end-to-end.
- LeRobot deployment guide for your policy.

**Do** (5–6 hours):
- Kick off fine-tune of SmolVLA on your Day 5 dataset, running in `tmux`. Expected wall time: 4–6 h on consumer GPU.
- While it trains:
  - Profile inference latency on the Day 4 checkpoint: time a single forward pass; estimate chunk Hz; sketch the async server pattern.
  - Read 1–2 papers from your awesome-VLA backlog. Avoid context-switching to coding.

**Deliverable:** `notes/day6.md`: training command, expected completion time, latency profile of Day 4 checkpoint.

---

## Day 7 — Deploy + frontier + consolidation

**Goal:** run the inference loop on the new checkpoint; sample one frontier topic; write up the week.

**Morning (3 h) — Deploy your model:**
- Implement the inference loop (LeRobot's `lerobot-eval` or `lerobot-record` is a good starting template).
- Run 10 trials on the real arm (Path A) or in sim (Path B). Per `reference/06_training_and_eval.md`, vary at least one of {object instance, init pose, lighting} across trials.
- Identify the top failure mode in one sentence.

**Midday (2 h) — One frontier paper, your pick:**
- **RL post-training**: SimpleVLA-RL — <https://github.com/PRIME-RL/SimpleVLA-RL>. README + section 3.
- **Humanoid stack**: GR00T N1 paper — <https://arxiv.org/abs/2503.14734>, sections 1–3.
- **Long-horizon agents**: Gemini Robotics 1.5 blog + Gemini Robotics paper — <https://arxiv.org/abs/2503.20020>.

**Evening (1–2 h) — Consolidate:**
- `notes/day7.md`: 10-trial success rate, p95 inference latency, top failure mode, planned next iteration.
- Consolidated `notes/SUMMARY.md` linking each day's deliverables.

---

## After this week

Ladder up:
1. Reproduce one paper's published result on the same dataset and report your delta.
2. Push your fine-tuned model and dataset to HF Hub; write a blog post.
3. Try the OFT recipe on OpenVLA if you used SmolVLA, or vice versa. Compare.
4. Move from a single arm to bimanual (ALOHA / SO-ARM101 bimanual). Same loop, double the action dim.
5. RL post-training in earnest with SimpleVLA-RL or VLA-RFT.

## What you're explicitly NOT doing this week (and why)

- **No classical robotics**: motion planning, dynamics, ROS deep-dive. Skipped to free time for VLA-specific content. Pick up if needed later.
- **No SLAM / mapping**: orthogonal to tabletop VLA.
- **No training an OXE-scale model from scratch**: requires lab-scale compute; not the point of this curriculum.
- **No closed-source models (Gemini Robotics, Helix)**: you can't run them yourself; read papers but don't dwell.
- **No reading every VLA paper**: 80+ have been published. The 12 in the minimum reading list cover 90% of what you need.

## Pacing notes

- If Day 4's fine-tune doesn't converge by end of day, **stop and debug** before Day 5. The pipeline being correct is more important than the schedule.
- Days 5–6 are the most variable: hardware delays, sim install issues, data quality issues all consume time. Budget 50% slack on these days.
- If you fall behind by 1 day, **skip Day 7's RL post-training** rather than compress Day 6. RL on a broken SFT is wasted time.
