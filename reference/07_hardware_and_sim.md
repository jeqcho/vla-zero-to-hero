# 07 — Hardware and Simulators

What real robots and sim platforms cost, what they do, what they're good for. Skewed toward what an open-source VLA practitioner actually buys / installs in 2026.

## Real-robot platforms (tabletop manipulation)

### SO-100 / SO-101 / SO-ARM100/101 (Hugging Face × The Robot Studio)

[SO-100 docs](https://huggingface.co/docs/lerobot/en/so100) · [SO-ARM repo](https://github.com/TheRobotStudio/SO-ARM100)

- **6-DoF 3D-printed arm with parallel-jaw gripper.**
- **$100–$500 per arm** depending on assembly state and tariffs.
- SO-100: bus servos at 19.5 kg·cm @ 7.4V. SO-101: upgraded to 30 kg·cm @ 12V (better payload, less friction).
- **Native LeRobot integration** (lerobot-record, lerobot-train, lerobot-eval all out-of-box).
- Sold as kits via WaveShare, Hiwonder, Seeed Studio, AliExpress.
- **Bimanual SO-ARM101 kit** ~$220–$240.

**Use this if:** you want to start tomorrow with no prior robotics infrastructure, on a budget, and you're OK with a small payload (<500g). The dominant low-friction entry point in 2026.

### ALOHA / ALOHA 2 (Stanford / Google) and Mobile ALOHA

[ALOHA paper](https://arxiv.org/abs/2304.13705) · [ALOHA 2](https://aloha-2.github.io/) · [Mobile ALOHA](https://mobile-aloha.github.io/)

- **Bimanual 6-DoF teleop platform.** Two ViperX 300 arms (followers) puppeteered by two WidowX 250 leaders.
- ~$20k (originally); ALOHA 2 adds gravity comp, improved grippers; Mobile ALOHA adds a holonomic base.
- **The bimanual fine-manipulation reference.** Standard testbed for ACT and many bimanual VLAs.
- Four cameras: two wrist + two overhead.

**Use this if:** you need bimanual capability, contact-rich tasks (cooking, folding, threading), and you have $20k+.

### Franka Panda / Research 3

- 7-DoF arm. The academic-research standard. Smooth torque control, F/T sensing at flange.
- ~$30k+.
- Most DROID data is on Franka. Many papers' real-world results are on Franka.
- ROS + libfranka driver. Now well-integrated into LeRobot.

**Use this if:** you're in a lab that already has one, or you need precise force control / torque sensing.

### UR5 / UR5e (Universal Robots)

- 6-DoF collaborative arm. Industrial-grade reliability, easy programming.
- ~$25k–$45k.
- [GELLO teleop framework](https://wuphilipp.github.io/gello_site/) often paired with UR5e for low-cost bimanual setups.
- Common in academia and industry. Slower control loop than Franka but more robust.

### xArm 7 (UFACTORY)

- 7-DoF arm, ~$10k–$15k. Popular budget Franka alternative.
- Many OXE sub-datasets used xArm.
- Less precise than Franka, more affordable.

### Trossen WidowX 250 (6-DoF)

- 6-DoF. ~$3k.
- Used in BridgeData V2 and many low-cost setups.

### Humanoids (out of scope for this curriculum; awareness only)

Notable platforms: Fourier GR-1 / 1X NEO (GR00T-deployed), Unitree G1/H1 (research-accessible), Figure 02/03 (closed, Helix), Tesla Optimus (closed). For humanoid VLA work in 2026, start in simulation (Isaac Lab + GR00T) — hardware costs $50k–$200k+ and is partnership-gated.

## Teleop interfaces (the "joystick" for collecting data)

- **ALOHA / ALOHA 2 / Mobile ALOHA leader-follower** — best for bimanual fine motor. ~$2k extra (the leader arms).
- **GELLO** ([project](https://wuphilipp.github.io/gello_site/)) — generalizable, low-cost (uses Dynamixel motors), 3D-printable. Drop-in for arbitrary follower arms.
- **VR controllers** (Quest 2/3) — fastest to set up; jerkier; common for Bridge-style data. Frameworks: Open Teleop, oculus_reader.
- **3D mouse (SpaceMouse)** — old-school; good for slow precise tasks.
- **Direct kinesthetic teaching** — physically move the arm; works for cobots like UR / Panda; lower data rate but most natural.

ALOHA's leader-follower pattern produces the highest-quality demos by a wide margin. If you're doing serious work, that or GELLO are the defaults.

## Simulators

### LIBERO

[Code](https://github.com/Lifelong-Robot-Learning/LIBERO).
- MuJoCo + Robosuite based. Four task suites (Spatial / Object / Goal / Long).
- The de-facto sim benchmark. Easy install, fast.
- Limitations: relatively narrow rendering, memorization risk (see LIBERO-PRO / Plus).

### SimplerEnv

[Code](https://github.com/simpler-env/SimplerEnv).
- Recreates real environments (Google Robot, WidowX/Bridge) in sim with matched dynamics and cosmetics.
- Goal: predict real-world success without burning hours on a real robot.
- Empirically much better than LIBERO at predicting real-world performance.

### ManiSkill 3

[Paper](https://arxiv.org/html/2410.00425v1) · [Code](https://github.com/haosulab/ManiSkill).
- **GPU-parallelized** simulation and rendering. Thousands of envs in parallel for RL.
- First general benchmark with fast RL from visual inputs on complex manipulation.
- Good fit for RL post-training pipelines (works with RLPD, RFCL, etc.).

### Isaac Lab + Isaac GR00T + Isaac Lab-Arena (NVIDIA)

[Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/main/index.html) · [GR00T repo](https://github.com/NVIDIA/Isaac-GR00T) · [Isaac Lab-Arena × LeRobot](https://huggingface.co/blog/nvidia/generalist-robotpolicy-eval-isaaclab-arena-lerobot).
- High-fidelity physics + photorealistic rendering on NVIDIA hardware.
- **Whole-body control + teleoperation** for humanoids (Isaac Lab 2.3+).
- Isaac Lab-Arena: VLA-policy eval framework integrated with HF LeRobot. Supports GR00T, π0, SmolVLA out of the box.
- Free, but needs an RTX GPU (and is heavy).

### RoboCasa

[Project](https://robocasa.ai/) · [Paper](https://arxiv.org/abs/2406.02523).
- Kitchen-scale household simulation. **2,500+ 3D assets, 100+ tasks.**
- Built on Robosuite; uses MimicGen to scale demonstrations.
- Supports BC, diffusion policies, VLA frameworks, RL with action sequence modeling.

### Gazebo / PyBullet / Drake / MuJoCo Playground

- Legacy / niche options. Use these only if you have a specific reason (Drake for trajectory optimization; PyBullet for quick prototypes).
- For VLA work in 2026, LIBERO / ManiSkill / Isaac Lab cover ~all needs.

## Compute checklist for hardware deployment

| Compute target | What runs on it |
|---|---|
| **Off-robot inference server** (RTX 4090 / A100) | OpenVLA 7B, π0, GR00T-class models at 10–50 Hz. Connect via gRPC / WebSocket. |
| **On-robot Jetson AGX Orin / Thor** | SmolVLA 450M, π0 (with quantization), GR00T On-Device variants. 5–20 Hz typical. |
| **CPU / NPU only** | LiteVLA-Edge-class (<256M) at ~5–10 Hz. Niche. |

Most production-grade real-robot demos in 2026 use an off-robot inference server + on-robot action executor over a fast LAN. Quantized on-robot inference is improving but still trades quality for latency. See `08_deployment.md` for numbers.

## What hardware to buy if you're starting fresh today (2026)

Most pragmatic ladder:

1. **No budget**: $0 — Run SmolVLA / OpenVLA in LIBERO / ManiSkill sim. Get the workflow muscle memory first.
2. **~$200 budget**: SO-ARM100 single arm + USB cam. Fine-tune SmolVLA on a tabletop pick-and-place.
3. **~$2k budget**: SO-ARM101 bimanual kit + 2× cameras. Replicate an ACT-style bimanual task.
4. **~$30k budget**: Franka Research 3 + GELLO teleop. Now you're in DROID territory and can join real research conversations.
5. **Lab-scale**: ALOHA 2 or Mobile ALOHA for bimanual + base; reach for humanoid only after this is solid.

Don't skip rungs. Trying to debug a humanoid with no manipulation muscle memory burns months.
