# 02 — Imitation Learning Foundations

Everything in a modern VLA's action head descends from three ideas: BC, ACT, and Diffusion Policy. Understand these three and the rest is variation.

## Behavior Cloning (BC)

The full algorithm: supervised learning of action from observation.

```
loss = E_{(o, a) ~ demos} [ || π_θ(o) - a ||² ]    # for continuous a
loss = E_{(o, a) ~ demos} [ CE(π_θ(o), a) ]        # for discretized a
```

Pros: trivial, sample-efficient on the training distribution, no environment interaction needed.
Cons: **compounding error / covariate shift** — small per-step errors push the robot off the expert state distribution, and out there the model has never been trained. It does not recover.

### Why BC works at all in modern VLA

The "BC doesn't work" folk wisdom is from 2010s-era policies with 1–5 dim observations. Modern fixes:
- High-capacity vision encoders + web-pretrained features → behavior cloning generalizes much further than people expected.
- **Action chunking** — predict and execute k steps before re-querying, reducing the effective horizon by ~k.
- **Multimodal action heads** (diffusion, flow) — capture multiple valid expert behaviors instead of averaging them into garbage.
- **Lots of diverse demos** (OXE-scale) — covariate shift is empirical, not theoretical; with enough states covered, drift is bounded.

## ACT — Action Chunking Transformer (Zhao et al., RSS 2023)

[Paper](https://arxiv.org/abs/2304.13705) · [Code/aloha](https://github.com/tonyzhaozh/act)

The training algorithm:
1. CVAE encoder takes the action sequence + style latent, encodes to z.
2. Transformer decoder conditions on z + observations (camera images, qpos) and predicts a **chunk of k future actions** (k=100 in the paper; 1 sec at 50 Hz).
3. At test time, sample z from prior; **temporal ensembling** averages action predictions from overlapping chunks via exponential weighting.

Why it works:
- k-step chunks reduce effective horizon for compounding error.
- CVAE captures multimodality (multiple ways to do the same task).
- Temporal ensembling smooths discrete chunk boundaries.

Result: 80–90% success on contact-rich bimanual tasks (battery slotting, ziploc bag opening) with ~50 demos. ACT is still the strong baseline for any new task with bimanual data — try it before reaching for a giant VLA.

## Diffusion Policy (Chi et al., RSS 2023)

[Paper](https://arxiv.org/abs/2303.04137) · [Code](https://github.com/real-stanford/diffusion_policy) · [Project](https://diffusion-policy.cs.columbia.edu/)

Action prediction as conditional denoising:

```
Train: sample (o, a*); add Gaussian noise to a* over T steps; train ε_θ(o, a_t, t) to predict added noise.
Test:  start from a ~ N(0, I); iterate a_{t-1} = denoise(a_t, o, t) for T steps.
```

Equivalent to learning the score `∇_a log p(a|o)` and following Langevin dynamics. The model outputs an action **chunk** (`a_t ∈ R^{k × action_dim}`), so it's also a chunking method. DDIM-style inference at 5–20 steps works fine.

Why it's strong:
- **Handles multimodal actions** natively — diffusion does not collapse modes (unlike MSE).
- **High-dim actions** are no harder than low-dim, since you denoise the whole chunk.
- **Stable training** — no GAN / RL instability.

Wins on 12 benchmark tasks with avg +46.9% over prior SOTA.

In VLA-land, the diffusion-head pattern survives in:
- **Octo** — diffusion head on a transformer trunk.
- **π0, π0.5** — flow-matching head (a continuous-time variant of diffusion).
- **GR00T N1/N1.5** — diffusion-transformer (DiT) action expert.
- **RDT-1B** — diffusion in the trunk itself.

## Flow Matching, in 60 seconds

A simpler diffusion sibling adopted by π0, π0.5, π0.6:

```
Sample t ~ U(0,1), a0 ~ N(0, I), a1 ~ data.
Define linear interpolation: at = (1-t)·a0 + t·a1.
Train v_θ(at, t, obs) to predict the true velocity (a1 - a0).
Test: integrate da/dt = v_θ(a, t, obs) from t=0 to t=1.
```

Vs. diffusion: trains on velocities instead of scores; ODE-integrable in fewer steps (often 5–10); empirically very stable for continuous action chunks at 50 Hz. Mechanically the same role as a diffusion head.

## L1-Regression + Parallel Decoding (the OpenVLA-OFT pattern)

[OpenVLA-OFT paper](https://arxiv.org/abs/2502.19645). The pendulum has swung partially back: discrete-token autoregressive heads (RT-2, OpenVLA) are slow to decode; diffusion/flow heads are great but multi-step inference adds latency. OFT shows that for some tabletop tasks, plain **L1 regression with bidirectional attention** and **parallel chunk prediction** matches or beats the fancy heads while being 26× faster.

Setup: feed `k` learned action queries; attend bidirectionally; one forward pass produces a `k × action_dim` chunk; trained with L1 loss. No autoregression, no denoising loop.

## Decision tree for picking an action head (today's defaults)

```
Are you fine-tuning OpenVLA?
  → Use OFT (parallel decode + L1 + chunking). 26x faster, +20% success on LIBERO.

Are you fine-tuning π0 / π0-FAST?
  → Use what shipped: flow matching for π0; FAST tokenizer + AR for π0-FAST.
    π0 is smoother for dexterous high-Hz tasks; π0-FAST is faster to train.

Building a small open VLA from scratch on consumer GPU?
  → SmolVLA's recipe: flow-matching head on a small (450M) backbone. Or DiT à la GR00T.

Bimanual ALOHA-style task, no VLA needed (custom robot, novel task)?
  → Start with ACT or Diffusion Policy. Cheaper. Often enough.
```

## Demonstrations: how many, what quality

A few practical numbers from the literature:
- ACT: ~50 demos → 80–90% on contact-rich bimanual tasks.
- OpenVLA fine-tune: ~10–25 demos per task can suffice for in-distribution generalization.
- Gemini Robotics On-Device: 50–100 demos for adaptation to new task.
- SmolVLA author guidance: ~50 episodes minimum, more is better.
- π0: trained on 7 robot platforms, 68 unique tasks, 10,000+ hours.

**Quality > quantity, but only up to a point.** Demonstrations should cover the test distribution (different initial positions, lighting, distractors). Inconsistent demonstrators hurt. ALOHA-style leader-follower teleop gives smooth high-quality data; VR teleop is faster to collect but jerkier.

## Read these papers in this order

1. ACT (Zhao et al. 2023) — short, foundational, makes chunking intuitive.
2. Diffusion Policy (Chi et al. 2023) — clean writing, sets up everything downstream.
3. OpenVLA (Kim et al. 2024) — first open VLA, sets the codebase you'll likely touch.
4. π0 (PI 2024) — flow-matching action experts; current SOTA dexterity.
5. OpenVLA-OFT (Kim et al. 2025) — when AR + discrete is dethroned, why, and what replaces it.

If you only have time for one: **OpenVLA**. It is the rosetta stone of the modern VLA stack.
