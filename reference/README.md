# VLA Reference (Zero-to-Hero)

Curated reference for a research engineer building/deploying Vision-Language-Action models on real robots.
**Bias:** practical, recent (2023–2026), open-source-first, deploy-on-hardware oriented.

## Reading order

1. [00_landscape.md](00_landscape.md) — what a VLA is, taxonomy, and the 2023–2026 timeline (read first).
2. [01_robotics_primer.md](01_robotics_primer.md) — minimum robotics for ML folks (kinematics, EE pose, gripper, control loop).
3. [02_imitation_learning.md](02_imitation_learning.md) — BC, ACT, Diffusion Policy. These are the parent ideas.
4. [03_vla_architectures.md](03_vla_architectures.md) — RT-2 → OpenVLA → π0/π0.5 → GR00T → Helix → Gemini Robotics.
5. [04_action_representations.md](04_action_representations.md) — discrete tokens, FAST, flow matching, action chunking.
6. [05_datasets.md](05_datasets.md) — Open X-Embodiment, DROID, BridgeV2, LIBERO, LeRobot community data.
7. [06_training_and_eval.md](06_training_and_eval.md) — pretraining, LoRA / OFT fine-tuning, RL post-training, evaluation.
8. [07_hardware_and_sim.md](07_hardware_and_sim.md) — SO-100, ALOHA, Franka, humanoids; LIBERO, ManiSkill, Isaac Lab, RoboCasa.
9. [08_deployment.md](08_deployment.md) — real-time inference, async action server, quantization, sim-to-real.
10. [09_ecosystem_and_sources.md](09_ecosystem_and_sources.md) — LeRobot, openpi, OpenVLA codebases, awesome lists, where to follow the field.

## Companion study plans

- [`../one_week_plan.md`](../one_week_plan.md) — 7-day curriculum.
- [`../three_day_plan.md`](../three_day_plan.md) — 3-day zero-to-hero sprint.

## Naming conventions used here

- **VLM** — Vision-Language Model (e.g., PaliGemma, LLaVA). Image+text → text.
- **VLA** — Vision-Language-Action. Adds an action head to a VLM. Image+text → robot action sequence.
- **EEF / EE** — End-effector (gripper / tool tip).
- **DoF** — Degrees of freedom. 6-DoF EE pose = (x, y, z, roll, pitch, yaw); +1 for gripper.
- **OXE** — Open X-Embodiment dataset (multi-robot ~1M trajectories).
- **OFT** — Optimized Fine-Tuning (the parallel-decoding recipe for OpenVLA).
- **S1/S2** — Dual-system: System 1 fast actor (~50–200 Hz), System 2 slow reasoner (~5–10 Hz).
