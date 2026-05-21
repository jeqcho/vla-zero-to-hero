# 03 — VLA Architectures

A field guide to the models you must know, organized by lineage. Pick one to fine-tune; understand the rest as variations on the same theme.

## The reference architecture (what they all share)

```
                  ┌──────────────────────────────┐
images (N cams)──►│                              │
text (lang.) ────►│   Vision-Language backbone   │──► hidden tokens ──► [action head] ──► action chunk (k × d)
proprio (opt.) ──►│   (VLM, e.g. PaliGemma 3B)   │
                  └──────────────────────────────┘
```

Variations are along three axes:
- **Backbone size / family**: Llama-2 7B, PaliGemma 3B, Eagle 2.5 (~3B), Gemma, Gemini 2.0.
- **Action head**: discrete-token AR, diffusion, flow matching, L1 regression w/ parallel decode.
- **System decomposition**: single tower vs. dual-system (S1 fast actor + S2 slow reasoner).

## RT-2 (Google, 2023)

[Project](https://robotics-transformer2.github.io/)

Closed-weights but conceptually foundational. PaLM-E or PaLI-X co-trained on web VQA and robot data. **Actions are discretized into 256 bins per dimension and emitted as text tokens** ("move to 0.12 0.05 -0.03 ..."). Decoded autoregressively at 1–5 Hz.

Why it mattered: first demonstration that VLM web priors transfer to robot control (emergent semantic reasoning: "pick up the extinct animal" → toy dinosaur).

Why it lost mindshare: closed weights, slow inference, no good handling of action multimodality.

## RT-X / Open X-Embodiment (Google + 33 labs, 2023-10)

[Paper](https://arxiv.org/abs/2310.08864) · [Project](https://robotics-transformer-x.github.io/) · [GitHub](https://github.com/google-deepmind/open_x_embodiment)

The dataset effort (covered in 05_datasets.md) and two models: RT-1-X (architecture-from-scratch) and RT-2-X (VLM-initialized, 55B). RT-2-X established the OXE-trained generalist baseline.

## Octo (UC Berkeley, 2024-05)

[Project](https://octo-models.github.io/) · [Code](https://github.com/octo-models/octo)

Open transformer trained on 800K OXE trajectories. **Diffusion head** over the action chunk. Multi-camera, multi-embodiment. ~93M to ~1.4B params.

Strengths: clean codebase, designed for fine-tuning to new robots, supports adding/dropping input modalities post-hoc.
Weaknesses: smaller backbone limits semantic transfer compared to OpenVLA.

## OpenVLA (Stanford / Google / TRI / Berkeley, 2024-06)

[Paper](https://arxiv.org/abs/2406.09246) · [Code](https://github.com/openvla/openvla) · [HF model](https://huggingface.co/openvla/openvla-7b)

The community reference. 7B params. Backbone = **Prismatic VLM** (Llama-2 7B language + SigLIP + DINOv2 fused vision tower). Trained on OXE.

Action head: 256 discrete bins per dim, 7D EEF deltas, emitted as 7 tokens autoregressively. Hijacks the lowest-frequency 256 tokens of the Llama tokenizer to represent action bins.

Inputs: third-person image (224×224) + language instruction. **No proprioception**, no wrist cam (in the base model).

Numbers worth remembering:
- Beats RT-2-X (55B) by 16.5% across 29 evaluation tasks.
- LIBERO base success: ~76.5%. With OFT recipe: ~97.1%.
- Stock inference: ~3 FPS on Jetson AGX Orin INT4; ~6–8 Hz on RTX 4090.
- Supports LoRA fine-tuning (the standard).

**This is the codebase to learn first if you want to ship a VLA.** Fine-tune scripts, dataset converters (RLDS), eval harnesses all exist.

### OpenVLA-OFT (Stanford, 2025-02)

[Paper](https://arxiv.org/abs/2502.19645) · [Project](https://openvla-oft.github.io/)

Recipe-level improvement on top of OpenVLA — same backbone, new head/training:
- **Parallel decoding**: emit all `k × 7` action floats in one forward pass with bidirectional attention.
- **L1 regression loss** instead of cross-entropy over bins.
- **Action chunking** k=8.
- 25–50× inference speedup; +20% LIBERO success.

Now the recommended starting point for OpenVLA fine-tuning.

## π0 / π0-FAST / π0.5 / π0.6 (Physical Intelligence)

Codebase: **[openpi](https://github.com/Physical-Intelligence/openpi)** — production-quality, BC + flow matching, fine-tune scripts, async inference server, LeRobot integration.

### π0 (2024-10) [paper](https://www.pi.website/download/pi0.pdf)
- Backbone: **PaliGemma 3B** (SigLIP vision + Gemma language).
- Action expert: **300M flow-matching transformer**.
- 50 Hz dexterous control. Trained on 7 platforms / 68 tasks / 10k+ hours.
- Tasks demonstrated: laundry folding, table bussing, grocery bagging, box assembly.

### π0-FAST (2025-01) [paper](https://arxiv.org/abs/2501.09747)
- Same PaliGemma backbone, but **autoregressive action emission with FAST tokenizer** (DCT + BPE).
- 5× training speedup vs. per-dim binning.
- Trade: AR decode is slower per-action than flow matching at inference, but training is much cheaper.

### π0.5 (2025-04) [paper](https://arxiv.org/abs/2504.16054)
- Adds **knowledge insulation**: action expert gradients do NOT flow back into the VLM backbone. The VLM is co-trained on web data + FAST action tokens; the action expert produces continuous actions on top.
- Co-training on web data + high-level semantic prediction → robust open-world generalization.
- Demoed cleaning new homes never seen in training.

### π0.6 (2025-11) [model card](https://website.pi-asset.com/pi06star/PI06_model_card.pdf)
- Latest release; further improvements on open-world generalization. Read the model card for current numbers.

## GR00T N1 / N1.5 / N1.7 (NVIDIA, 2025–2026)

Code: **[Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)** · N1 [paper](https://arxiv.org/abs/2503.14734) · [N1.5 project](https://research.nvidia.com/labs/gear/gr00t-n1_5/)

Open humanoid foundation model. Dual-system architecture:
- **System 2**: VLM (Eagle 2.5 in N1.5) at ~5–10 Hz. Scene + language understanding.
- **System 1**: DiT (diffusion transformer) action expert generating fluid motor actions in real time.

Trained on heterogeneous mixture: real robot trajectories + human video + synthetic data (RoboCasa, MimicGen, GR00T-Dreams blueprint).

### N1.5 advances over N1
- VLM upgraded to Eagle 2.5 (better grounding / physical understanding).
- **FLARE** (Future LAtent Representation alignment): aligns to future *embeddings*, not future *frames*. Cheaper than world-model rollout, enables learning from human video.
- 13.1% → 38.3% success on 12 DreamGen tasks.

### N1.7 (2026)
- **EgoScale**: pretrained on 20K hours of human ego-centric video.
- Relative-EEF action representation consistent across humans and robots → direct transfer of manipulation priors from human video.

Deployed on Fourier GR-1 and 1X humanoids for bimanual household tasks.

## Helix (Figure AI, 2025-02)

[Announcement](https://www.figure.ai/news/helix). Closed weights. Architecturally interesting.

Pure dual-system, but tightly coupled:
- **S2**: 7B-parameter VLM at **7–9 Hz**. Outputs a continuous latent vector capturing intent.
- **S1**: 80M-parameter transformer policy at **~200 Hz**, cross-attending to S2's latent. Emits full upper-body continuous control (wrists, torso, head, individual fingers).

First VLA reported to output high-rate continuous control over **all** upper-body DoF including individual fingers, on a real humanoid. Generalizes across thousands of household objects without object-specific training.

Helix 02 (2026-02) extends to a "general-purpose humanoid system" per [Figure announcement](https://www.luigifreda.com/2026/02/02/figure-ai-announces-helix-02-a-general-purpose-humanoid-system/).

## Gemini Robotics / -ER / 1.5 (Google DeepMind, 2025+)

[Gemini Robotics paper](https://arxiv.org/abs/2503.20020) · [Blog](https://deepmind.google/models/gemini-robotics/) · [1.5 blog](https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/) · [On-Device blog](https://deepmind.google/blog/gemini-robotics-on-device-brings-ai-to-local-robotic-devices/)

Three flavors:
- **Gemini Robotics** — main VLA, Gemini 2.0 with action modality.
- **Gemini Robotics-ER** — embodied-reasoning variant; emphasizes spatial / 3D reasoning, point prediction, trajectory planning. Useful as a VLM-only system that another action layer consumes.
- **Gemini Robotics 1.5** — chain-of-thought before acting; emits intermediate plans then actions.
- **Gemini Robotics On-Device** — local-inference version; adapts in 50–100 demos.

Closed weights but documented behavior. The "thinks before acting" pattern is the obvious near-term direction for long-horizon work.

## SmolVLA (Hugging Face, 2025-06)

[Blog](https://huggingface.co/blog/smolvla) · [Doc](https://huggingface.co/docs/lerobot/smolvla) · [Base](https://huggingface.co/lerobot/smolvla_base)

450M-param VLA. Backbone: SmolVLM-2 family. Pretrained **only on open community LeRobot datasets** (everything tagged `lerobot` on the Hub). Action head: flow matching.

The fast-iteration choice if you have one consumer GPU. Fine-tune in ~4 hours on a single A100 (20K steps). Runs on consumer hardware.

## Other notable open VLAs (memorize at the bullet level)

- **RDT-1B** (2024-10, [arXiv](https://arxiv.org/abs/2410.07864)) — 1B diffusion transformer for bimanual. Diffusion happens in the trunk, not a separate head.
- **CogACT** (Microsoft, [code](https://github.com/microsoft/CogACT)) — cognition + action separation; supports starting from OpenVLA weights.
- **DexVLA** (2025-02, [arXiv](https://arxiv.org/abs/2502.05855)) — VLM + plug-in billion-param diffusion expert. Focuses on dexterous tasks.
- **TinyVLA** (2024-09, [arXiv](https://arxiv.org/abs/2409.12514)) — small, fast, no pretraining stage. Outperforms OpenVLA on speed/data-efficiency; tradeoff is some semantic generalization.
- **VLA-0** (2025-10, [arXiv](https://arxiv.org/pdf/2510.13054)) — minimal-modification VLA; uses a stock VLM with very simple action handling. Worth tracking for the "how small can we make the action head" question.

## Picking your starter model (today, 2026)

| Your situation | Pick |
|---|---|
| One GPU, SO-100 / consumer arm, want fast iteration | **SmolVLA** + LeRobot |
| 1–8 GPUs, OXE-style 7-DoF arm (Franka, UR5, xArm), want strongest open generalist | **π0** or **π0-FAST** via `openpi` |
| Want the canonical research baseline / paper comparisons | **OpenVLA** (use OFT recipe) |
| Building a humanoid stack | **GR00T N1.5/N1.7** (open) or just simulate to start; Helix/Gemini are closed |
| Bimanual + ALOHA hardware + a target task | Start with **ACT**; reach for π0 only if ACT plateaus |
