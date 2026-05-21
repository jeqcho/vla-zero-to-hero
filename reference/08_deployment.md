# 08 — Deployment, Latency, Real-Time

How to take a trained VLA and run it on a robot at the rates the controller wants.

## The inference loop you're actually targeting

```
Loop at robot control rate (30–100 Hz):
  1. Read latest images from N cameras (last frame only; don't queue stale frames).
  2. Read proprioception (qpos, gripper) at the same instant as images.
  3. Look up the next action in the current chunk (or trigger a new inference if exhausted).
  4. Send action to robot via driver (libfranka / lerobot.robot / ROS / custom).
  5. Background: when ~half the chunk is executed, kick off the next inference.
```

**The async server pattern** (used by openpi and most production deployments):
- **Robot process** owns timing; holds an action buffer; sends actions at the controller rate.
- **Inference process** lives on a beefier machine; takes obs over the wire, returns action chunks.
- Buffer is sized so the next chunk arrives before the current chunk runs out. Latency budget = chunk execution time minus inference time.

This decouples physical control rate from inference rate. **Don't try to run VLA inference at the controller frequency.** It will not work.

## Realistic latency / throughput numbers

| Model + config | Hardware | Latency or rate |
|---|---|---|
| **OpenVLA 7B stock**, INT4 quantized | Jetson AGX Orin | ~3 FPS (bottleneck: AR token decode, not memory) [LiteVLA-Edge paper] |
| OpenVLA 7B fp16 | Jetson AGX Orin | doesn't fit comfortably in 64 GB shared memory; rarely run at this precision on-device |
| **OpenVLA + OFT (parallel decode + chunk)** | A100 / RTX 4090 | ~25× speedup → chunk in ~30 ms |
| **π0 (flow matching, 5–10 ODE steps)** | RTX 4090 | 50 Hz effective control (with chunking) |
| **SmolVLA 450M** | Consumer GPU | Real-time on a single GPU |
| **LiteVLA-Edge (~256M, 4-bit GGUF)** | Jetson Orin | **150 ms per chunk (~6.6 Hz)** end-to-end |
| **LiteVLA-H dual-rate** | Jetson AGX Orin | **S1: 50 ms (19.7 Hz); S2: 150–165 ms (~6.6 Hz)** |
| **Gemini Robotics On-Device** | Robot-local | Designed to be real-time on-device |

Key takeaway: **action chunking is non-negotiable.** Inference is the bottleneck; chunking amortizes it across many control steps.

## Where the latency actually goes

For autoregressive token-emitting VLAs (RT-2, stock OpenVLA), most of the wall-clock is the **prefill** of the vision encoder + LLM (one big forward pass over the entire context) **before** action tokens start decoding. AR action decoding adds another ~7k token forward passes.

Mitigations in order of impact:
1. **Parallel decode the action chunk** (OFT). Removes the AR loop entirely.
2. **KV cache reuse across chunks** when context changes minimally — supported in modern transformer inference servers (vLLM, SGLang).
3. **Quantization** (INT8 / INT4). Most impact on memory + bandwidth; some on compute. Watch for accuracy regressions on dexterous tasks.
4. **Distill to a smaller model** (TinyVLA, SmolVLA). Big inference win at some capability cost.
5. **Off-robot inference** + LAN. Removes the on-robot constraint, but adds network latency (typically <5 ms LAN; >50 ms WiFi is risky).

## Quantization for VLA, in practice

- **Weight-only quantization (W8A16, W4A16) via bitsandbytes / AWQ / GGUF**: usually safe to INT8, slight degradation at INT4 (especially for fine motor precision).
- **Activation quantization (W8A8)**: more aggressive; benchmark per-task.
- **Avoid quantizing the action head.** It's small and precision-critical. Quantize the VLM backbone.
- **Always re-evaluate after quantization** — small accuracy losses in the VLM translate to specific failure modes (e.g., gripper timing) at the action level.

[LiteVLA-Edge](https://arxiv.org/abs/2603.03380) is a good case study: FP32 SFT → INT4 GGUF + llama.cpp on Jetson. Achieves 150 ms end-to-end. The pattern (train high-precision, deploy low-precision) is the standard.

## Sim-to-real and domain randomization

If you train on sim (RoboCasa / LIBERO / Isaac Lab) and deploy real, you need to bridge the gap. The current best practices:

1. **Domain randomization** during training: lighting, textures, camera pose perturbations, robot dynamics randomization. ManiSkill and Isaac Lab both support this easily.
2. **Match cameras to sim**: same FOV, same image resolution, same color balance. A drift here causes most "model X doesn't work on my robot" reports.
3. **Match action conventions**: if sim uses 7D EEF deltas in world frame and your real driver expects EEF frame, you need to convert. **Test this with a hand-coded action first** before blaming the model.
4. **Mixed sim+real training**: a few real demos go a long way (often 5–20 real demos on top of sim pretraining recovers most of the gap).
5. **Calibrate normalization on real data**: recompute the per-dim percentiles on your real robot's command range.

## What to fix when "it doesn't work on the real robot"

In rough probability order:

1. **Action normalization mismatch.** You used sim stats; the robot expects different units or range. Print the predicted actions and compare to what you sent during teleop.
2. **Camera mismatch.** Resolution, white-balance, FOV all different from training. View a test image through your loader.
3. **Image preprocessing inconsistency.** Train used ImageNet mean/std; inference uses raw [0,1]. Or vice versa.
4. **Wrong action frame.** EEF-relative vs. base-relative vs. world. Look at your robot driver docs.
5. **Wrong gripper convention.** Train: 0=open / 1=close. Robot: 0=close / 1=open. Or `[-1, 1]` instead of `[0, 1]`.
6. **Chunk-length mismatch.** Trained k=8; deployed asking for k=50. Some heads handle this; many don't.
7. **Latency / timing**: actions arrive late, robot stops responding. Check chunk execution vs. inference time.
8. **Then, finally**, model quality. If 1–7 are clean and it still doesn't work, you have a data or model problem.

## Production deployment checklist (the one I'd want)

- [ ] Async inference server runs on the same machine for a week without crashing.
- [ ] Action latency p95 measured and within budget (target: < (chunk_length × control_period) / 2).
- [ ] Image pipeline integrity: render a saved obs through the same preprocessing as training.
- [ ] Normalization stats are loaded from the *training* dataset stats, not recomputed on inference data.
- [ ] Safety stop / E-stop wired up at the robot driver level, not in the inference process.
- [ ] Recovery behavior defined: what does the robot do if inference times out? (Default: hold position.)
- [ ] Logging: dump (obs, predicted action, executed action, success label) for every trial. You'll need this for failure analysis.
- [ ] At least 20 held-out trials passed before declaring deployment "working."

## On-robot vs off-robot — what to choose

**Off-robot inference (LAN-attached GPU box) is almost always the right answer** in 2026 for any 1B+ model, because:
- You can iterate on the model without redeploying to the robot.
- You get full-precision quality.
- LAN latency (~3–5 ms) is small vs. chunk execution time.
- One GPU box can serve multiple robots.

**On-robot inference** is only correct when:
- Network is unreliable (mobile robots, outdoor, telepresence).
- Hard latency requirements <30 ms end-to-end.
- You're committed to a quantized small model anyway.

Default plan: off-robot first; only consider on-robot once the full system works end-to-end and you have a specific latency or autonomy reason to move.
