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
- Install LeRobot in a fresh `uv` env:
  ```bash
  uv venv && source .venv/bin/activate
  uv pip install lerobot torch torchvision
  ```
- Walk through LeRobot's "Visualize a Dataset" notebook on any small LeRobot dataset (e.g., `lerobot/aloha_sim_insertion_human`). Confirm you can see images, actions, and a rollout.
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
  - **(easier)** Fine-tune SmolVLA on a single LIBERO task (~4 hours on 1× A100).
  - **(real-stakes)** Run OpenVLA-OFT LoRA fine-tune on a LIBERO task. Reference: <https://openvla-oft.github.io/>
- Run eval on LIBERO. Compare to published numbers. Diagnose if off.
- Read LIBERO-PRO (<https://arxiv.org/abs/2510.03827>) to understand why LIBERO numbers are over-optimistic.

**Deliverable:** working fine-tune checkpoint + LIBERO success-rate table (per-task) in `notes/day4.md`. A failure analysis paragraph: which task(s) did worst and why.

---

## Day 5 — Simulators, hardware, and data collection

**Goal:** stand up a simulator and (if hardware available) a teleop pipeline.

**Read** (1.5 hours):
- `reference/07_hardware_and_sim.md`
- LeRobot SO-100 walkthrough: <https://huggingface.co/docs/lerobot/en/so100>
- ALOHA paper (skim setup, not method again) — <https://arxiv.org/abs/2304.13705>.

**Do** (5 hours):

**Path A — Real hardware (if SO-ARM101 arrived):**
- Assemble (if not pre-assembled). Run LeRobot calibration.
- Record ~30 demos of a simple pick-and-place via leader-follower or VR teleop.
- Push the dataset to HF Hub.

**Path B — Sim only:**
- Install ManiSkill 3 (`pip install mani-skill`). Run a pick-place env headless.
- Use a scripted policy or MimicGen to generate 50 trajectories.
- Convert to LeRobot dataset format.

**Deliverable:** a LeRobot-format dataset (30+ episodes) of *your* task — sim or real — sitting in HF Hub or local disk. `notes/day5.md` with the dataset card.

---

## Day 6 — Fine-tune on your task + deploy

**Goal:** fine-tune on your own data and run the inference loop end-to-end.

**Read** (1 hour):
- `reference/08_deployment.md`
- LeRobot deployment guide for your policy.

**Do** (5–6 hours):
- Kick off fine-tune of SmolVLA (or OpenVLA-OFT) on your Day 5 dataset, running in `tmux`. Expected wall time: 4–6 hours on consumer GPU.
- While it trains:
  - Profile inference latency: time a single forward pass; estimate chunk Hz; sketch the async server pattern.
  - Implement the inference loop (LeRobot's `lerobot-eval` is a good starting template).
- Once trained: run real inference (Path A) or sim eval (Path B). Record 10 trials.
- Identify top failure mode. Try one mitigation: more data, action chunk size, image augmentation.

**Deliverable:** `notes/day6.md`: 10-trial success rate, p95 inference latency, top failure mode, planned next iteration.

---

## Day 7 — RL post-training, frontier topics, and consolidation

**Goal:** know what's possible beyond SFT; write up your week.

**Read** (3 hours):
- `reference/09_ecosystem_and_sources.md`
- **SimpleVLA-RL paper** — <https://github.com/PRIME-RL/SimpleVLA-RL>. Skim README + paper section 3.
- **GR00T N1 paper** — <https://arxiv.org/abs/2503.14734>. Read sections 1–3 (dual-system architecture; data mixture).
- **Gemini Robotics blog + paper abstract** — <https://arxiv.org/abs/2503.20020>. Understand the ER (embodied reasoning) variant.
- One survey, your pick: <https://arxiv.org/abs/2505.04769>

**Do** (2–3 hours):
- Try one RL post-training iteration on your fine-tuned model if you have a sim env (SimpleVLA-RL pattern). Even a single epoch — see if eval moves.
- Browse the awesome-VLA lists; pick 3 papers from the last 3 months and skim abstracts. The field moves; this becomes a habit.
- Write a 1-page "where I am now and where I'd go next" doc (`notes/day7.md`).

**Deliverable:** consolidated `notes/SUMMARY.md` linking each day's deliverables. **You can now have a real conversation about VLA architecture, training, and deployment with anyone in the field.**

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
