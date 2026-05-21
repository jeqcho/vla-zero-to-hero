# 05 — Datasets

What exists, what's in each, what to use them for.

## Open X-Embodiment (OXE) — the "ImageNet of robotics"

[Paper](https://arxiv.org/abs/2310.08864) · [Project](https://robotics-transformer-x.github.io/) · [GitHub](https://github.com/google-deepmind/open_x_embodiment)

- **~1M trajectories**, **22 robot embodiments**, **60 datasets pooled from 34 labs** (2023-10).
- Format: **RLDS** (RL Datasets, TFDS-based). Standardized observation/action schema across embodiments.
- Action convention (when present): 7D EEF deltas + gripper.
- **What every modern open VLA is pretrained on (or initialized from).** OpenVLA, Octo, π0 all eat OXE.

Practical:
- Download via TFDS or the GitHub repo's scripts. Mirrors on GCS.
- **Quality is heterogeneous.** OpenVLA's training mixed-curated a subset (the "rt-x mix") with quality filtering.
- Common gotcha: action conventions (joint vs. EEF, absolute vs. delta) drift across sub-datasets. Always check `data_loader.action_proprio_normalization_type`.

## DROID — high-quality teleop on Franka

- **~76K demos / ~350 hours / 564 scenes** across **52 buildings / 18 labs**, all on Franka Panda arms, unified protocol. [Paper](https://arxiv.org/abs/2403.12945).
- Quality is much more consistent than OXE (single embodiment, standard setup).
- Excellent for fine-tuning generalists, evaluating cross-scene generalization.
- Distributed as part of OXE; also available standalone.

## BridgeData V2

- ~60K trajectories on a WidowX 250 6-DoF arm across diverse household manipulation.
- The default "small generalist" pretraining set when you can't afford full OXE.
- Used as the eval target in many SimplerEnv / Octo comparisons.

## LIBERO — the dominant simulation benchmark

[Code](https://github.com/Lifelong-Robot-Learning/LIBERO) · papers below

Four task suites in MuJoCo / Robosuite:
- **LIBERO-Spatial** — 10 tasks × 50 demos. Same objects, different spatial arrangements.
- **LIBERO-Object** — 10 tasks × 50 demos. Same task, different objects.
- **LIBERO-Goal** — 10 tasks × 50 demos. Same scene/objects, different goals.
- **LIBERO-Long (LIBERO-100)** — **100 long-horizon, multi-step tasks** × ~50 demos.

Why everyone uses it:
- Standardized eval, easy to install (`pip install libero`).
- Has both demonstrations (for fine-tuning) and an evaluator (for measuring success).
- Almost every 2024–2026 VLA paper reports LIBERO numbers.

Caveats — **be skeptical of headline LIBERO numbers**:
- [LIBERO-PRO](https://arxiv.org/abs/2510.03827): under reasonable perturbations (objects, init state, instructions, environment), reported gains shrink dramatically. Memorization confound.
- [LIBERO-Plus](https://arxiv.org/abs/2510.13626): similar story — robustness to object layout / camera / language perturbation is much worse than headline.
- Treat LIBERO as a *minimum* bar, not a real-world predictor.

## SimplerEnv — closing the sim-to-real gap

[GitHub](https://github.com/simpler-env/SimplerEnv)

Recreates real evaluation environments (Google Robot, WidowX/Bridge) in simulation with cosmetic + dynamics matching. Lets you benchmark a checkpoint *quickly* before doing the expensive real-robot eval. Good predictor of real success, much better than LIBERO.

If a paper reports SimplerEnv numbers, that's evidence the authors care about real-world transferability.

## LeRobot community datasets

The HuggingFace Hub has a growing collection of small task-specific datasets uploaded by the community (LeRobot tag). SmolVLA's pretraining data is **only** these. Useful for:
- Quick reproductions of specific tasks.
- A pattern to emulate when uploading your own data.

Browse: `https://huggingface.co/datasets?other=lerobot`

## Synthetic data — the underrated category

- **MimicGen** ([paper](https://arxiv.org/abs/2310.17596)) — given a small set of seed demonstrations, automatically segment, retarget, and replay them across new object poses/layouts. Generates 100K+ trajectories from ~10 seeds.
- **RoboCasa** ([project](https://robocasa.ai/)) — kitchen-scale simulated environments (2500+ assets, 100+ tasks) with MimicGen-generated demos.
- **GR00T-Dreams blueprint** (NVIDIA) — synthetic trajectory generation via world foundation models. GR00T N1.5 was developed in 36 hours of synthetic data gen vs. ~3 months of manual collection.
- **DexMimicGen** — extends MimicGen to bimanual / dexterous.

Use these for:
- Augmenting small real datasets.
- Pretraining + small-real fine-tune.
- Stress-testing OOD generalization with controlled perturbations.

## Human video pretraining (the 2025+ frontier)

Robots are slow to collect data; humans aren't. Several models exploit this:

- **GR00T N1.5** uses FLARE to align with future *latents* of human videos (frame-prediction is too expensive).
- **GR00T N1.7** pretrains on **20K hours of EgoScale ego-centric human video**.
- The trick is a **relative EEF action representation** that lets human hand-wrist trajectories serve as proxies for robot EEF trajectories.

If your task involves manipulation with broad object generalization, expect human-video-pretrained checkpoints (GR00T N1.7+) to be your best starting point.

## How to read a dataset entry (RLDS quick spec)

Each step has:
```
observation:
  image: H×W×3 uint8 (one or more cameras)
  natural_language_instruction: string
  state: optional proprioception (qpos, EEF pose, gripper width)
action: float[7] or float[14] etc.    # exact dim is dataset-specific
is_first / is_last / is_terminal: booleans
discount, reward: usually unused for VLA
```

Episodes are grouped via `tf.data.Dataset.from_generator` over an iterable of steps. OpenVLA's `prismatic/vla/datasets/` and Octo's `octo/data/` are reference implementations.

## LeRobot dataset format

[Spec](https://huggingface.co/docs/lerobot/main/en/lerobot-dataset-v3)

Simpler than RLDS. The de-facto format for new VLA work:
- Parquet files for non-image data (states, actions, timestamps).
- MP4 files for camera streams (compressed).
- `meta/info.json` describes schema; `meta/stats.json` has normalization stats.
- First-class HuggingFace Hub integration: `dataset.push_to_hub("user/my-task")`.

If you're collecting new data with a real robot via LeRobot, you'll produce this format by default — `lerobot-record` writes it. openpi and SmolVLA both consume LeRobot datasets directly. **This is the format to standardize on for new work in 2026.**

## How much data do you actually need

| Scenario | Demos | Why |
|---|---|---|
| Per-task ACT, ALOHA-style | 20–50 | Standard ACT result |
| OpenVLA fine-tune on a new task | 10–25 | Pretrained generalist priors do most of the work |
| SmolVLA fine-tune | ~50 episodes recommended | Smaller model needs more per-task |
| π0 base model from scratch | 10,000+ hours, 7 platforms, 68 tasks | PI scale; you won't do this |
| Gemini Robotics On-Device adapt | 50–100 | Their stated number |

Rule of thumb: **start with ~50 demos**, train, look at failure cases, collect more demos that target those failures. Iterate. This is much more effective than blindly collecting 1000 demos up front.
