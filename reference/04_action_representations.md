# 04 — Action Representations

How a VLA actually emits motor commands. This is the single most active design axis in 2024–2026.

## The four mainstream patterns

| Pattern | Loss / sampling | Inference cost | Smoothness | Used by |
|---|---|---|---|---|
| **Discrete per-dim binning (256 bins/dim, AR)** | CE over tokens | k×d sequential forward passes | OK | RT-2, OpenVLA (stock) |
| **FAST tokenization (DCT + BPE, AR)** | CE over BPE tokens | ~few tokens per chunk; AR | Good | π0-FAST |
| **Flow matching / diffusion** | velocity / score regression; multi-step ODE/DDIM at test | 5–20 integration steps | Excellent | π0, π0.5, Octo, GR00T (DiT), RDT-1B, SmolVLA |
| **L1 regression + parallel decode** | L1 over chunk floats; 1 forward pass | 1 pass | Good | OpenVLA-OFT, Helix S1 |

All four are typically wrapped in **action chunking** (predict `k` future actions; execute open-loop).

## Pattern 1 — Discrete per-dimension binning (the RT-2 / OpenVLA way)

Each action dimension's range is split into 256 bins (covering 1st–99th percentile of training data, clipped). Each timestep's `d`-dim action becomes `d` discrete tokens. A 7-DoF chunk of length `k` is `7k` tokens, decoded autoregressively.

**How OpenVLA gets away with no new tokenizer:** it hijacks the lowest-frequency 256 tokens of the Llama-2 tokenizer (rare words/subwords) and reassigns them to action bins. The VLM doesn't need to learn new embeddings.

Pros: simple; reuses LLM machinery and losses; principled.
Cons:
- AR decoding is slow (one token at a time → `7k` sequential forward passes).
- Each dimension treated independently → loses temporal/cross-dim correlations.
- Bin quantization caps precision at ~1/256 of action range.

## Pattern 2 — FAST: Frequency-space Action Sequence Tokenization

[Paper](https://arxiv.org/abs/2501.09747) · [Project](https://www.pi.website/research/fast) · [HF tokenizer](https://huggingface.co/physical-intelligence/fast)

The realization: per-dim per-timestep binning is wasteful because action chunks are smooth low-frequency signals. Just like JPEG, the bulk of the information is in low DCT coefficients.

Algorithm:
1. Normalize actions (1st/99th percentile → [-1,1] per dim) for cross-embodiment robustness.
2. **Discrete Cosine Transform** along time, per dimension. Get DCT coefficients (mostly low-freq).
3. Scale-and-round (quantize) — low coefficients get more bits than high.
4. **BPE** compresses the quantized DCT integer sequence into a short token sequence.
5. The VLA learns to emit this short token sequence; decoded by inverse FAST.

Result:
- **~10× compression** vs. per-dim binning → fewer tokens per chunk → much cheaper AR decode.
- **5× training speedup** for the same final quality.
- "Universal" — works across many embodiments without retraining the tokenizer when using **FAST+** (trained on 1M trajectories).
- Pairs well with autoregressive VLMs: π0-FAST achieves comparable performance to π0 (flow matching), with simpler training infra.

## Pattern 3 — Diffusion and flow matching

Both produce continuous action chunks via iterative refinement.

### Diffusion (DDPM-style, used in Diffusion Policy, Octo, RDT-1B)
- Forward: `a_t = √(α_t)·a_0 + √(1-α_t)·ε`, ε ~ N(0, I).
- Train: predict ε from `(a_t, t, obs)` with MSE loss.
- Test: iterate denoise with DDPM (T~100) or DDIM (T~5–20) until clean.

### Flow matching (used in π0, π0.5, π0.6)
- Linear interpolant: `a_t = (1-t)·a_0 + t·a_1`, a_0 ~ N(0, I), a_1 from data.
- Train: predict velocity `(a_1 - a_0)` from `(a_t, t, obs)` with MSE.
- Test: integrate ODE `da/dt = v_θ(a, t, obs)` from t=0→1 (Euler, 5–10 steps).

Why VLAs prefer diffusion / flow over discrete:
- **Multimodal action distributions** are captured natively (multiple valid grasps, two ways to fold a sleeve).
- **Smooth continuous output** at high control rates (50–100 Hz) — no quantization artifacts.
- **High action dim** (humanoids, bimanual) handled cleanly.

Cost: inference requires multiple forward passes (5–20). Mitigated by:
- Distilled one-step variants — see SnapFlow ([arXiv 2604.05656](https://arxiv.org/pdf/2604.05656)), one-step flow VLAs.
- Asynchronous inference: predict next chunk while current chunk executes.

### "Action expert" pattern

π0 and GR00T factor the action head into a smaller dedicated transformer (the "action expert"), separate from the VLM backbone. The VLM produces a sequence of latent tokens; the action expert reads those + noise + proprio and runs the diffusion/flow process. This:
- Decouples VLM and action update rates.
- Lets you swap action experts (joint vs. EEF; 7D vs. 30D) without retraining the VLM.
- Enables "knowledge insulation" (π0.5) — VLM gradients can be detached from the action expert.

## Pattern 4 — L1 regression + parallel decoding (OpenVLA-OFT)

[Paper](https://arxiv.org/abs/2502.19645) · [Project](https://openvla-oft.github.io/)

The minimalist alternative to fancy heads. The trick is **architectural**, not statistical:
- Feed `k` learned action queries into the transformer (k = chunk length).
- Use **bidirectional attention** over the action queries (drop the causal mask within the chunk).
- One forward pass outputs `k × d` floats. Train with **L1 loss**.

That's it. No tokens, no denoising, no flow ODE. Just a regression head that emits the whole chunk in parallel.

Numbers: **26× faster** inference and **3× lower latency** than stock OpenVLA, +20% on LIBERO. For tabletop tasks where multimodality is mild, this is the new strong baseline.

When this fails: highly multimodal tasks (multiple equally-good grasps in different directions). L1 regression averages modes; you'll see hesitation or mode confusion.

## Action chunking, in any of the four patterns

Predict a chunk of `k` future actions; execute open-loop for `m ≤ k` steps; re-predict. Three useful knobs:

- **k (chunk length)**: ACT 100 · π0 ~50 · OpenVLA-OFT 8 · GR00T similar. Bigger = less reactive but more efficient.
- **m (execute horizon)**: often = k (fully open-loop within chunk) or = k/2 (overlapping chunks for smoother handoff).
- **Temporal ensembling** (ACT): when chunks overlap, weighted-average overlapping predictions, exponential weight toward the most recent. Smooths chunk boundaries.

Chunking is the single most important inference-time trick — it amortizes inference cost (one forward pass → many actions executed) and shortens the effective imitation horizon.

## Normalization, the under-appreciated detail

Every modern VLA computes **per-dimension 1st and 99th percentiles** over the training set and normalizes actions to [-1, 1]. At inference:
- Predicted normalized action → unnormalize with stored stats → send to robot.
- **If you fine-tune on a new robot with a different action range, recompute statistics.** A surprisingly common deployment bug is forgetting to do this.

OpenVLA and openpi both ship utilities for this (`compute_dataset_stats.py` or similar).

## How to choose

```
You need fast inference and your task is mostly unimodal     → L1 + parallel decode (OFT)
You need maximum dexterity / multimodality, can afford 5–10 step decode → flow matching (π0)
You're using a VLM you don't want to fork              → FAST tokenizer + AR
You're explicitly comparing to RT-2 / OpenVLA in a paper  → discrete per-dim bins
```

In practice almost no one ships per-dim binning anymore. The 2026 default for new VLA work is **flow matching with action chunking** (small models) or **L1 + parallel decode** (when you're starting from OpenVLA weights).
