# 00 — The VLA Landscape

## What a VLA is (one paragraph)

A **VLA** is a single neural network that maps `(camera images, language instruction, [proprioception])` → `robot action chunk`. Architecturally it is a VLM (a multimodal transformer pretrained on web image-text) with an **action head** that emits low-level commands (joint deltas or 6-DoF EEF pose + gripper) at 30–100 Hz. Train it on robot demonstration trajectories, optionally co-train on web/VQA data to preserve VLM priors, fine-tune for your robot.

## Canonical architecture choices (the design space)

| Axis | Options | Used by |
|---|---|---|
| **Backbone** | LLM-style (Llama-2 7B), VLM (PaliGemma 3B, Eagle-2/2.5, Gemma) | OpenVLA (Llama+SigLIP); π0/π0.5 (PaliGemma); GR00T N1 (Eagle-2), N1.5+ (Eagle-2.5); Helix (custom 7B VLM) |
| **Action head** | Discrete-token autoregression; flow matching / diffusion; L1 regression w/ parallel decoding | OpenVLA, RT-2 (discrete); π0, GR00T, RDT-1B (diffusion/flow); OpenVLA-OFT, Helix S1 (regression) |
| **Chunking** | Predict next k actions in one shot, execute open-loop | Almost everyone now: ACT k=100; π0 k=50; OpenVLA-OFT k=8 (single arm) / k=25 (bimanual) |
| **System decomposition** | Single tower; dual-system S1 (fast) + S2 (slow) | Single: OpenVLA, π0. Dual: GR00T N1, Helix, Gemini Robotics 1.5 |
| **Action space** | Joint positions / deltas; EEF Δpose + gripper; both | OXE convention: EEF Δxyz + ΔRPY + gripper (7D) |
| **Proprioception** | Used or ignored | Ignored: OpenVLA stock. Used: π0, GR00T, SmolVLA, most newer models |
| **Cameras** | Third-person only, +wrist, multi-view | Single 3rd-person: OpenVLA. 3rd + wrist: π0, SmolVLA. Multi: ALOHA-trained (2 wrist + 2 overhead) |

## 2023–2026 timeline (only models that matter)

- **2023-03** — [**Diffusion Policy**](https://arxiv.org/abs/2303.04137) (Chi et al.). Score-based action denoising. Sets the diffusion-head baseline.
- **2023-04** — [**ACT + ALOHA**](https://arxiv.org/abs/2304.13705) (Zhao et al.). Action Chunking Transformer + low-cost bimanual teleop hardware. Defines "predict k actions at once" pattern.
- **2023-07** — [**RT-2**](https://robotics-transformer2.github.io/) (Google). First real VLA: PaLI-X / PaLM-E with actions-as-tokens. Closed weights.
- **2023-10** — [**Open X-Embodiment + RT-X**](https://arxiv.org/abs/2310.08864). Pools 60 datasets, ~1M trajectories, 22 embodiments. Becomes the "ImageNet of robotics."
- **2024-05** — [**Octo**](https://octo-models.github.io/). Open transformer trained on 800k OXE traj with a diffusion action head.
- **2024-06** — [**OpenVLA**](https://arxiv.org/abs/2406.09246) (Stanford/Google). 7B, Llama-2 + SigLIP/DINO, discrete action tokens. Trained on a 970k curated OXE subset. Beats RT-2-X (55B) by 16.5%. The community reference codebase.
- **2024-10** — [**π0**](https://www.pi.website/download/pi0.pdf) (Physical Intelligence). ~3.3B (PaliGemma 3B + 300M flow-matching action expert). 50 Hz dexterous control. Followed by **openpi** release.
- **2024-10** — [**RDT-1B**](https://arxiv.org/abs/2410.07864). 1B-parameter diffusion transformer for bimanual.
- **2025-01** — [**π0-FAST**](https://www.pi.website/research/fast). DCT-based action tokenizer; 5× training speedup over per-dim binning.
- **2025-02** — [**Helix**](https://www.figure.ai/news/helix) (Figure). Dual-system S2 (7B VLM at 7–9 Hz) + S1 (80M policy at ~200 Hz). First high-rate full-upper-body humanoid control from a VLA.
- **2025-02** — [**OpenVLA-OFT**](https://openvla-oft.github.io/). Parallel decoding + L1 + chunking → 26× faster inference, 76.5%→97.1% on LIBERO.
- **2025-03** — [**Gemini Robotics + Gemini Robotics-ER**](https://arxiv.org/abs/2503.20020) (DeepMind). Gemini 2.0 with action modality. ER = embodied reasoning variant.
- **2025-03** — [**GR00T N1**](https://arxiv.org/abs/2503.14734) (NVIDIA). Open dual-system humanoid foundation model. ~2.2B total (1.34B Eagle-2 VLM + 0.86B DiT action expert).
- **2025-04** — [**π0.5**](https://arxiv.org/abs/2504.16054). Open-world generalization via co-training on web + high-level semantics + "knowledge insulation" (action gradients don't flow into VLM).
- **2025-06** — [**SmolVLA 450M**](https://huggingface.co/blog/smolvla). Small open VLA for consumer hardware; pretrained on lerobot-tagged community data.
- **2025-07** — [**Gemini Robotics On-Device**](https://deepmind.google/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/). Runs locally on the robot; adapts in 50–100 demos.
- **2025-05** — [**GR00T N1.5**](https://research.nvidia.com/labs/gear/gr00t-n1_5/). Adds FLARE (future-latent alignment), upgrades VLM to Eagle-2.5, learns from human video. 13.1%→38.3% on DreamGen.
- **2025-09** — [**Gemini Robotics 1.5**](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/). Adds chain-of-thought before acting.
- **2025-11** — [**π0.6**](https://website.pi-asset.com/pi06star/PI06_model_card.pdf) (model card). Latest PI release; further open-world push.
- **2026-Q1** — [**GR00T N1.7**](https://github.com/NVIDIA/Isaac-GR00T). 20K hr EgoScale human video pretraining, relative-EEF action representation for human↔robot transfer.

## What still doesn't work well in 2026

(Recurring across every survey — know these so you don't oversell.)

1. **Recovery from failure mid-task.** Slip a grasp, fumble an object — most VLAs don't reliably re-engage.
2. **On-robot real-time inference for ≥7B models.** OpenVLA stock = ~3 FPS on Jetson AGX Orin even at INT4. Mitigations: OFT parallel decode, async server off-robot, smaller models (SmolVLA, GR00T N1.5 ~2B), quantization.
3. **Long-horizon planning.** Helped by S2 chain-of-thought, but still brittle past ~30 sec horizons without hierarchy.
4. **Robust eval.** LIBERO success rates are inflated by memorization; see [LIBERO-PRO](https://arxiv.org/abs/2510.03827) and LIBERO-Plus for perturbation tests.
5. **Cross-embodiment without per-robot fine-tuning.** OXE training helps but per-robot 20–200 episode fine-tune is still standard practice.

## What "deploy on a real robot" means in practice (the checklist this curriculum optimizes for)

You need to be able to:
1. Pick a base model that fits your hardware budget (latency + RAM + compute).
2. Collect 20–200 demonstrations via teleop on your robot.
3. Get the demos into LeRobot dataset format (or RLDS for OpenVLA-stack).
4. Run fine-tuning (LoRA, or OFT for OpenVLA, or `openpi` finetune for π0).
5. Deploy with a real-time loop — usually an off-robot inference server + on-robot action executor, or quantized on-device.
6. Evaluate honestly (held-out objects, positions, lighting; not just training distribution).

Each of the reference docs that follows targets one of these steps.
