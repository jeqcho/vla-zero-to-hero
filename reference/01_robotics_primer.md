# 01 — Robotics Primer for ML Researchers

The only robotics you need to bootstrap into VLAs. If you already know forward/inverse kinematics, skim and skip.

## The robot, in one diagram

```
[ Base ] -- joint1 -- link1 -- joint2 -- link2 -- ... -- jointN -- [ End-Effector (EEF) + Gripper ]
                                                                     ^
                                                                     these are what we usually command
```

A manipulator has **N joints** (typically 6 or 7) and an **end-effector** (gripper / hand). Most VLA work targets a **6- or 7-DoF arm with a parallel-jaw gripper**.

## Two ways to command a robot

| Action space | What you predict | Pros | Cons |
|---|---|---|---|
| **Joint space** (qpos / Δqpos) | One scalar per joint per timestep | Direct, no IK needed | Embodiment-specific |
| **EEF space (task space)** | 6D pose (xyz + RPY) + gripper open/close | Embodiment-agnostic; matches "move the hand to X" intuition | Requires inverse kinematics (IK) controller underneath |

**Default in most modern VLAs:** EEF deltas. The OXE-canonical 7D action is `[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]`, often normalized per-dim to [-1, 1] via 1st/99th-percentile quantiles.

**Bimanual:** double the action dim. ALOHA-style 14D = 2× (6 joints + 1 gripper).

**Humanoid (e.g., GR00T):** up to ~30+ DoF (torso, neck, two arms, two hands, sometimes legs). Modern humanoid VLAs typically partition the action space (upper body via policy, legs via classical controller) or use relative EEF representations for human↔robot transfer.

## Forward and inverse kinematics (FK / IK), in 30 seconds

- **FK:** joints → EEF pose. Closed-form, fast, exact. `pose = FK(q)`.
- **IK:** desired EEF pose → joints. Generally no closed form (multiple solutions, singularities). Solved numerically (Jacobian pseudo-inverse, damped least squares) by the robot's controller — you do not implement this yourself.
- **Practical fact:** when you command EEF, the robot stack runs IK at its control rate (often 250 Hz–1 kHz) and tracks. You command at 10–100 Hz; the low-level controller fills in.

## The control loop (this is the loop a VLA replaces)

```
loop at 30–100 Hz:
  obs = read_cameras() + read_proprioception()    # images + qpos + EEF pose + gripper state
  action_chunk = policy(obs, language)            # ← the VLA
  for a in action_chunk[:execute_horizon]:
    robot.send_command(a)                          # IK + low-level PID
    sleep(1 / control_hz)
```

**Action chunking** = predict `k` actions per forward pass, execute some prefix open-loop, then re-predict. Reduces compounding error and amortizes inference cost. ACT used `k=100`; π0 uses `k=50`; OpenVLA-OFT uses `k=8`. Tradeoff: bigger `k` = less reactive, more efficient.

## What proprioception means in this context

The scalar inputs the robot knows about itself, fed alongside images:
- `qpos` — joint positions (N values).
- `qvel` — joint velocities (often optional).
- `EEF pose` — 6D pose of the gripper.
- `gripper width` or open/close state.
- `force/torque` at the wrist (rare; only on F/T-sensored robots).

**OpenVLA does not consume proprioception**; π0 and most newer models do. Whether to include it is a real design choice — without it, the model is purely vision-driven (good for cross-embodiment, bad for occluded tasks).

## Cameras

Two conventions you will see constantly:
- **Third-person** (a.k.a. **scene / static**) — fixed camera looking at the workspace. Usually wide FOV.
- **Wrist** (a.k.a. **EE-mounted**) — camera attached near the gripper. Provides close-up of the contact.

Most VLAs take **2–3 RGB images** (often resized to 224×224 or 256×256) per step. ALOHA and Mobile ALOHA use 4 cameras (2 wrist + 2 overhead). Depth is rarely used in current open VLAs — RGB only is the dominant pattern.

## Gripper

In open VLAs, gripper is **binary** (open=0 / close=1) or **scalar continuous** (width 0–1). Even though real grippers can be force-controlled, VLA action vectors almost always treat gripper as 1 dimension. Don't over-think it.

## Control frequencies (sanity numbers)

| Layer | Typical rate | Who runs it |
|---|---|---|
| Low-level motor PID | 250 Hz – 1 kHz | Robot firmware |
| IK / Cartesian controller | 100–500 Hz | Robot driver (ROS, robotiq, franka) |
| **VLA policy inference** | **5–50 Hz (action chunk × execute horizon)** | Your GPU |
| Dual-system S2 (planner) | 1–10 Hz | Your GPU |

If your VLA's chunked inference is 5 Hz with chunk=20, you're commanding actions at ~100 Hz effective. This is fine for tabletop manipulation; not fine for contact-rich dynamic tasks.

## Coordinate frames you must not confuse

- **World** — fixed, usually the robot's base. What humans usually think in.
- **Base** — robot's mounting frame. Often = world for tabletop setups.
- **EEF / tool** — the gripper's frame.
- **Camera** — each camera's optical frame.

A 6D pose is meaningful only with a frame attached. "Move 5 cm forward" — in *whose* frame? Standard convention: **EEF-relative deltas are in the current EEF frame** unless stated otherwise.

## Things you do NOT need to learn for VLA work (yet)

- SLAM, structure-from-motion. VLAs do not build maps.
- Classical motion planning (RRT, OMPL). The low-level controller handles short trajectories.
- Dynamics / Lagrangian mechanics. Modern VLAs treat the robot kinematically; the controller handles dynamics.
- ROS internals beyond basic node/topic semantics — most VLA codebases (LeRobot, openpi) bypass ROS entirely.

You can pick these up later if you go into mobile manipulation, locomotion, or high-DoF humanoid lower-body control. For tabletop VLA fine-tuning, you don't need them.

## Two terms you will hear and should understand

- **Compounding error** (a.k.a. **covariate shift**): a behavior-cloning policy trained on expert states drifts away from them at test time, and its errors compound. This is why ACT and Diffusion Policy work — chunking + multimodal outputs + temporal ensembling mitigate it. Recovery from compounding error is still an open problem.
- **Distribution shift** (here, **embodiment shift**): the test robot, lighting, or object set differs from training. The cross-embodiment OXE training regime + per-robot fine-tuning is the current standard mitigation.
