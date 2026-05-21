# 09 — Ecosystem and Sources

The codebases you'll actually use, the papers worth reading, and the channels to follow.

## Codebases (in priority order)

### 🤗 LeRobot — the integration layer

- Repo: <https://github.com/huggingface/lerobot>
- Docs: <https://huggingface.co/docs/lerobot/>
- The hardware-agnostic Python interface. Supports SO-100/101, ALOHA, Trossen, Franka, UR5e, Unitree, more.
- Ships `lerobot-record`, `lerobot-train`, `lerobot-eval` CLIs.
- Hosts SmolVLA, ACT, Diffusion Policy, π0, π0-FAST as first-class policies.
- LeRobot dataset format is the de-facto standard for new VLA data.

**If you do nothing else, learn this codebase.** It is the entry point.

### `openpi` — Physical Intelligence's open release

- Repo: <https://github.com/Physical-Intelligence/openpi>
- Production-quality: π0, π0-FAST, π0.5 checkpoints + training and fine-tuning scripts.
- Async inference server for real-robot deployment.
- LeRobot dataset compatibility.
- Apache 2.0 license.

If you want the strongest open VLA in 2026, this is it.

### `openvla` — the OpenVLA codebase

- Repo: <https://github.com/openvla/openvla>
- 7B Llama-2+SigLIP+DINOv2 VLA. RLDS data loaders. LoRA fine-tuning scripts.
- OFT recipe: <https://github.com/moojink/openvla-oft>
- The canonical research baseline. Cite this in any paper.

### `Isaac-GR00T` — NVIDIA's open humanoid stack

- Repo: <https://github.com/NVIDIA/Isaac-GR00T>
- GR00T N1, N1.5, N1.7 checkpoints. Eagle-2.5 VLM, DiT action expert.
- Integrates with Isaac Lab and Isaac Lab-Arena.

### Reference for action heads

- ACT: <https://github.com/tonyzhaozh/act>
- Diffusion Policy: <https://github.com/real-stanford/diffusion_policy>
- FAST tokenizer: <https://huggingface.co/physical-intelligence/fast>

### Simulators

- LIBERO: <https://github.com/Lifelong-Robot-Learning/LIBERO>
- SimplerEnv: <https://github.com/simpler-env/SimplerEnv>
- ManiSkill 3: <https://github.com/haosulab/ManiSkill>
- Isaac Lab: <https://isaac-sim.github.io/IsaacLab/>
- RoboCasa: <https://robocasa.ai/>
- MimicGen: <https://github.com/NVlabs/mimicgen>

### RL post-training

- SimpleVLA-RL (ICLR 2026): <https://github.com/PRIME-RL/SimpleVLA-RL>
- RLinf: <https://github.com/RLinf/RLinf>

## Papers — the minimum reading list

If you have one weekend, read these in order:

1. **OpenVLA** (Kim et al., 2024) — <https://arxiv.org/abs/2406.09246>
2. **π0** (Black et al., 2024) — <https://www.pi.website/download/pi0.pdf>
3. **π0.5** (Black et al., 2025) — <https://arxiv.org/abs/2504.16054>
4. **FAST** (Pertsch et al., 2025) — <https://arxiv.org/abs/2501.09747>
5. **OpenVLA-OFT** (Kim et al., 2025) — <https://arxiv.org/abs/2502.19645>
6. **GR00T N1** (NVIDIA, 2025) — <https://arxiv.org/abs/2503.14734>
7. **Gemini Robotics** (DeepMind, 2025) — <https://arxiv.org/abs/2503.20020>
8. **Open X-Embodiment / RT-X** (Collaboration, 2023) — <https://arxiv.org/abs/2310.08864>

Foundation reading (older, still load-bearing):

9. **ACT + ALOHA** (Zhao et al., 2023) — <https://arxiv.org/abs/2304.13705>
10. **Diffusion Policy** (Chi et al., 2023) — <https://arxiv.org/abs/2303.04137>
11. **RT-2** (Brohan et al., 2023) — <https://robotics-transformer2.github.io/>
12. **PaliGemma** (Beyer et al., 2024) — <https://arxiv.org/abs/2407.07726> (only if you don't know the VLM)

Surveys (when you want a map):

- "Vision-Language-Action Models: Concepts, Progress, Applications and Challenges" (Sapkota et al., 2025) — <https://arxiv.org/abs/2505.04769>
- "VLA Models for Robotics: A Review Towards Real-World Applications" (2025) — <https://arxiv.org/html/2510.07077v1>

## Live "awesome" lists (track these — fields move monthly)

- <https://github.com/yueen-ma/Awesome-VLA> — main academic survey list.
- <https://github.com/keon/awesome-physical-ai> — broader physical AI / world models / VLA.
- <https://github.com/jonyzhang2023/awesome-embodied-vla-va-vln> — embodied AI focus.
- <https://github.com/Psi-Robot/Awesome-VLA-Papers> — action-tokenization survey companion.
- <https://github.com/Denghaoyuan123/Awesome-RL-VLA> — RL of VLA.

## Conferences and venues (in priority order)

1. **CoRL** (Conference on Robot Learning) — the primary venue.
2. **RSS** (Robotics: Science and Systems) — Diffusion Policy, ACT, OFT all published here.
3. **ICRA** (IEEE International Conference on Robotics and Automation) — broader, more classical.
4. **ICLR / NeurIPS / ICML** — ML side; many VLA papers (OpenVLA, SimpleVLA-RL) appear here.
5. **CVPR** — vision side; embodied/VLA tracks growing.

Twitter / X: new releases still break here first. Three high-signal accounts: **@physical_int** (PI releases), **@GoogleDeepMind** (Gemini Robotics), **@NVIDIAAIDev** (GR00T + Isaac). Individual researcher handles rot fast; better to follow the labs.

HuggingFace: <https://huggingface.co/lerobot> is the org. New model releases drop here.

## Blogs / writeups worth following

- Physical Intelligence blog: <https://www.pi.website/blog>
- NVIDIA GEAR (Generalist Embodied Agent Research): <https://research.nvidia.com/labs/gear/>
- Google DeepMind Robotics: <https://deepmind.google/research/areas/robotics-and-control/>
- Phospho.ai blog: <https://blog.phospho.ai/> (great walkthroughs of ACT / π0 internals)
- LearnOpenCV LeRobot tutorials: <https://learnopencv.com/vision-language-action-models-lerobot-policy/>
- Hugging Face robotics blog: <https://huggingface.co/blog?tag=robotics>

## Tutorials / hands-on getting-started

- SmolVLA tutorial: <https://docs.phospho.ai/learn/train-smolvla>
- LeRobot SO-100 walkthrough: <https://huggingface.co/docs/lerobot/en/so100>
- LeRobot imitation-learning on real robots: <https://huggingface.co/docs/lerobot/main/en/il_robots>
- openpi quick start: see the `README.md` in <https://github.com/Physical-Intelligence/openpi>

## Where to ask questions

- LeRobot Discord (linked from HF org page) — active, friendly, fast turnaround.
- HF forums (`huggingface.co/discussions`).
- GitHub issues on the respective repos — maintained by serious labs.
- r/robotics, r/MachineLearning — broader but lower signal.

## Field velocity note

VLA is moving fast — major releases every 2–3 months in 2025–2026. Treat any "state of the art" claim more than 6 months old with skepticism. The model that's SOTA when you start this curriculum may not be when you finish it. **The first-principles material (action heads, datasets, eval methodology) ages much better than specific model rankings.** Optimize your reading time accordingly.
