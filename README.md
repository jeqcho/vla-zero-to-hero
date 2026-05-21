# VLA: Zero to Hero

A no-fluff curriculum for a research engineer learning Vision-Language-Action models for real-robot deployment.

## Pick a plan

- **[`three_day_plan.md`](three_day_plan.md)** — Aggressive 3-day sprint (~30 hrs). End-to-end fine-tune + deploy.
- **[`one_week_plan.md`](one_week_plan.md)** — 7-day plan (~40 hrs). Same outcome with more room for paper reading, hardware setup, and iteration.

## Reference

- **[`reference/`](reference/README.md)** — Compact reference on architectures, datasets, training, deployment, hardware. Read in order or use as lookup.

## Audience

You have solid ML/DL + transformers. You have minimal robotics background. Your end goal is building/deploying a VLA on a real robot — not surveying the field, not publishing the next foundation model.

## What this curriculum optimizes for

1. **Load-bearing only.** Every section maps to a step in "fine-tune and deploy a VLA on hardware." If it doesn't, it was cut.
2. **Open source first.** OpenVLA, π0/openpi, SmolVLA, LeRobot, Isaac GR00T — code you can run. Closed-source models (Gemini Robotics, Helix) are covered conceptually but not relied on.
3. **Recent (2024–2026).** RT-1-era material is mentioned for context, not assigned reading.
4. **Honest evaluation.** LIBERO is a sanity check, not ground truth. The curriculum trains the habit of held-out real evals from day 1.

## What's deliberately out of scope

- Classical robotics: kinematics deep-dive, dynamics, motion planning, ROS internals.
- SLAM, mapping, locomotion, lower-body humanoid control.
- Training foundation models from scratch on OXE-scale data.
- Closed-source model internals you can't verify or run.
- Exhaustive paper-by-paper history. The 12 key papers in [`reference/09_ecosystem_and_sources.md`](reference/09_ecosystem_and_sources.md) cover ~90% of what you need.

## Layout

```
.
├── README.md                  ← you are here
├── three_day_plan.md          ← 3-day sprint
├── one_week_plan.md           ← 7-day plan
└── reference/
    ├── README.md              ← reference index
    ├── 00_landscape.md        ← what a VLA is, taxonomy, 2023–2026 timeline
    ├── 01_robotics_primer.md  ← minimum robotics for ML folks
    ├── 02_imitation_learning.md ← BC, ACT, Diffusion Policy
    ├── 03_vla_architectures.md  ← RT-2 → OpenVLA → π0 → GR00T → Helix → Gemini
    ├── 04_action_representations.md ← discrete, FAST, flow matching, parallel decode
    ├── 05_datasets.md         ← OXE, DROID, LIBERO, LeRobot data, synthetic
    ├── 06_training_and_eval.md ← LoRA / OFT fine-tune, RL post-train, eval honesty
    ├── 07_hardware_and_sim.md ← SO-100, ALOHA, Franka, humanoids; LIBERO, ManiSkill, Isaac
    ├── 08_deployment.md       ← real-time inference, async server, quantization
    └── 09_ecosystem_and_sources.md ← codebases, papers, awesome lists, follows
```
