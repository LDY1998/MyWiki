# Learning Plan: AI Infrastructure for Large-Scale RL, Robotics, and World Models

A software/data-layer focused roadmap for "Physical AI" infra — skipping ROS, middleware, and hardware. Built around the 2026 stack: GPU-parallel simulation, distributed RL training, world-model pretraining, and high-throughput policy serving.

> **Hardware assumption for this plan:** 1× RTX 3070 (8 GB VRAM), 1 node. That's enough to *learn the infra* — not enough to train frontier models. See [Working Within an 8 GB Budget](#working-within-an-8-gb-budget-rtx-3070) below for the adapted strategy, scaled-down exercises, and when to rent cloud GPUs.

---

## Mental Model: The Physical AI Infra Stack

Think of physical AI training as five tightly-coupled layers. The bottleneck is almost always the interface between two layers (e.g., sim → trainer, trainer → inference engine).

```
┌───────────────────────────────────────────────┐
│ 5. Serving / Policy inference (vLLM, TRT-LLM) │
├───────────────────────────────────────────────┤
│ 4. RL orchestration (verl, OpenRLHF, AReaL)   │
├───────────────────────────────────────────────┤
│ 3. Training backends (Megatron, FSDP2, NeMo)  │
├───────────────────────────────────────────────┤
│ 2. World models + data curation (Cosmos, 1XWM)│
├───────────────────────────────────────────────┤
│ 1. Simulation & data generation (Isaac/MJX)   │
└───────────────────────────────────────────────┘
```

Key insight from the verl paper: **80–90% of RL training time is sample generation**. This drives every major architectural choice (hybrid engines, colocated actor/rollout, async RL).

---

## Phase 0 — Foundations (1–2 weeks)

Before touching infra, make sure these primitives are solid:

- **Distributed primitives**: NCCL collectives, all-reduce vs. all-gather, ring vs. tree, bandwidth-bound vs. latency-bound ops.
- **Parallelism taxonomy**: DP / TP / PP / SP / EP / CP. Know where each saves memory vs. where it costs bandwidth.
- **GPU memory math**: parameters + gradients + optimizer state + activations. Be able to back-of-envelope a 70B model on 8×H100.
- **Ray basics**: actors, placement groups, object store. Ray is the glue in most modern RL frameworks.

Resource: [Everything about Distributed Training and Efficient Finetuning](https://sumanthrh.com/post/distributed-and-efficient-finetuning/)

---

## Phase 1 — Distributed Training Backends (2–3 weeks)

This is the substrate every RL/world-model system sits on.

### What to learn
- **FSDP2** (per-parameter DTensor sharding) — the modern PyTorch-native path.
- **Megatron-LM** — TP + PP + SP; still the reference for efficient transformer training.
- **DeepSpeed ZeRO-1/2/3** — conceptually important even if you end up on FSDP.
- **TorchTitan** — shows how FSDP2 + TP + PP + `torch.compile` + Float8 compose cleanly.

### Hands-on
1. Train a small GPT with TorchTitan on 2 GPUs; toggle TP and PP and watch memory/throughput.
2. Reproduce a ZeRO-3 run, then port it to FSDP2. Understand what changed.
3. Profile with `torch.profiler` / Nsight; identify whether you're compute- or comm-bound.

Resources:
- [TorchTitan paper](https://arxiv.org/html/2410.06511v1)
- [Megatron-DeepSpeed](https://github.com/deepspeedai/Megatron-DeepSpeed)
- [DeepSpeed vs FSDP in 2026](https://vrlatech.com/deepspeed-vs-pytorch-fsdp-which-distributed-training-framework-in-2026/)

---

## Phase 2 — GPU-Parallel Simulation as a Data Source (2–3 weeks)

Simulation *is* the data pipeline for robotics RL. Modern sims are GPU-resident and produce millions of env-steps/sec.

### What to learn
- **Isaac Lab** (NVIDIA) — modular, supports PhysX / Warp / Newton / MuJoCo backends. The de-facto robot-learning framework in 2026.
- **MuJoCo Playground / MJX / MuJoCo Warp** — MJWarp gives 252× (locomotion) and 475× (manipulation) speedups on Blackwell.
- **mjlab** — Isaac Lab's manager API on top of MuJoCo Warp, bridging the two ecosystems.
- **Newton 1.0** (NVIDIA GTC 2026 GA) — production engine on Warp + OpenUSD.
- **ManiSkill3** — GPU-parallel manipulation benchmarking.

### Hands-on
1. Run an Isaac Lab locomotion env at 4096 parallel environments; graph throughput vs. batch size.
2. Port the same task to MJX/MJWarp; compare step-rate and gradient variance.
3. Build a domain-randomization pipeline (lighting, friction, mass) and measure sim-to-real proxy metrics.

Resources:
- [Isaac Lab paper (2025)](https://arxiv.org/html/2511.04831v1)
- [MuJoCo Playground](https://playground.mujoco.org/)
- [Newton Physics Engine 2026](https://byteiota.com/newton-physics-engine-475x-faster-robot-simulation-2026/)
- [mjlab](https://github.com/mujocolab/mjlab)
- [ManiSkill3 paper](https://arxiv.org/html/2410.00425v1)

---

## Phase 3 — Large-Scale RL Orchestration (3–4 weeks)

This is where physical AI infra diverges from vanilla LLM training: you need to co-schedule **actor training + rollout inference + reward models** on the same cluster.

### Frameworks to study
- **verl** (ByteDance) — 3D-HybridEngine eliminates redundancy between training and generation; Megatron backend; production-grade.
- **OpenRLHF** — Ray + vLLM + DeepSpeed; v0.10 (Apr 2026) adds VLM RLHF for models like Qwen3.5.
- **AReaL** (inclusionAI) — "AReaL-SEA" self-evolving data engine; async RL.
- **SimpleVLA-RL** (ICLR 2026) — RL scaling specifically for Vision-Language-Action models.
- **Slime, Verifiers** — round out the 2026 landscape.

### Concepts that matter
- **Hybrid Engine / colocated rollout**: same GPUs run both training and inference phases.
- **Async RL**: decouple rollout workers from learner to hide generation latency.
- **Resharding**: moving weights from training layout (TP/PP) to inference layout (TP only) efficiently.
- **PPO vs. GRPO vs. DAPO vs. REINFORCE++**: know when each is preferred; GRPO dominates LLM-RL in 2026.

### Hands-on
1. Run OpenRLHF GRPO on a 7B model with a single reward model; trace the Ray actor graph.
2. Reproduce a verl run with 3D-HybridEngine; measure generation throughput vs. naive split.
3. Swap in a VLA policy (SimpleVLA-RL) — now your rollouts produce image+action trajectories.

Resources:
- [verl](https://github.com/volcengine/verl)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) • [paper](https://arxiv.org/html/2405.11143v6)
- [AReaL](https://github.com/inclusionAI/AReaL)
- [SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL)
- [Anatomy of RL Frameworks](https://www.hanifleo.com/anatomy-of-rl-frameworks/)
- [OpenRLHF vs veRL deep dive](https://langcopilot.com/posts/2025-11-06-openrlhf-vs-verl-ray-framework-deep)
- [Open Source RL Libraries for LLMs — Anyscale](https://www.anyscale.com/blog/open-source-rl-libraries-for-llms)
- [RL Infrastructure 2025 — Introl](https://introl.com/blog/reinforcement-learning-infrastructure-rlhf-robotics-gpu-clusters-2025)

---

## Phase 4 — World Foundation Models (3–4 weeks)

World models are the generative substrate that lets you train policies without infinite real-world rollouts.

### Landscape
- **NVIDIA Cosmos** — open WFM platform, 9,000T tokens pretraining, Cosmos Tokenizer (8× compression, 12× faster), Cosmos Reason 2 (Feb 2026) for spatio-temporal reasoning.
- **NeMo Curator** — video curation pipeline: 20M hours in 14 days on Blackwell vs. 3+ years on CPU.
- **1X World Model (1XWM)** — diffusion video model + Inverse Dynamics Model; fine-tuned on 70h of NEO data; forecasts contact-rich humanoid futures with learned value head.
- **DreamerV3 / scalable world models** — the academic lineage (Hafner et al.).
- **Genie 3 / Genie Envisioner** — action-conditioned video models; policy eval and distillation.

### Concepts
- **Video tokenization** (discrete vs. continuous; temporal compression ratios).
- **Inverse Dynamics Models** (pixels → actions bridge).
- **Latent rollout for policy search** (training inside the world model — "dreaming").
- **Evaluation via generated futures**: "Evaluating bits, not atoms."

### Hands-on
1. Use Cosmos + NeMo Curator to curate ~100h of video into training tokens.
2. Fine-tune a Cosmos WFM on a narrow domain; sample futures conditioned on an action sequence.
3. Train a tiny DreamerV3 on a Isaac Lab task; compare sample efficiency to model-free PPO.

Resources:
- [NVIDIA Cosmos](https://www.nvidia.com/en-us/ai/cosmos/) • [GitHub](https://github.com/nvidia-cosmos) • [paper](https://arxiv.org/abs/2501.03575)
- [Cosmos blog — scaling synthetic data](https://developer.nvidia.com/blog/scale-synthetic-data-and-physical-ai-reasoning-with-nvidia-cosmos-world-foundation-models/)
- [Physical AI Data Factory Blueprint](https://nvidianews.nvidia.com/news/nvidia-announces-open-physical-ai-data-factory-blueprint-to-accelerate-robotics-vision-ai-agents-and-autonomous-vehicle-development)
- [1X World Model](https://www.1x.tech/discover/1x-world-model) • [PDF](https://www.1x.tech/1x-world-model.pdf)
- [Training Agents Inside Scalable World Models (Hafner & Yan 2025)](https://arxiv.org/pdf/2509.24527)
- [Awesome-World-Model](https://github.com/LMD0311/Awesome-World-Model)

---

## Phase 5 — Robot Data Layer & VLA Training (2–3 weeks)

Real-robot data is the other half of the data equation. The 2026 standard is LeRobot's format.

### What to learn
- **LeRobot dataset format** — HuggingFace's unified schema for multi-modal robot trajectories.
- **OpenVLA** — 7B VLA: SigLIP+DINOv2 vision → projector → Llama-2-7B → discrete action tokens; trained on 970k real demos.
- **SmolVLA** — efficient VLA on LeRobot community data.
- **OpenPI / Pi0, ACT** — the other common policies downstream of LeRobot.
- **Data-factory blueprints** — AWS Batch-based pipelines for embodied AI; NVIDIA's Physical AI Data Factory.

### Hands-on
1. Ingest a LeRobot dataset; write a dataloader that streams video + proprio + language into an FSDP trainer.
2. Fine-tune OpenVLA or SmolVLA on a small task; benchmark action-token accuracy.
3. Build a sim-to-real data mixing schedule (X% sim from Isaac Lab + Y% real from LeRobot).

Resources:
- [OpenVLA](https://openvla.github.io/) • [paper](https://arxiv.org/abs/2406.09246)
- [SmolVLA blog](https://huggingface.co/blog/smolvla)
- [LeRobot + ROCm fine-tuning guide](https://rocm.blogs.amd.com/artificial-intelligence/rocm-lerobot/README.html)
- [EmbodiFlow LeRobot pipeline](https://io-ai.tech/platform/en/guides/Pipeline/LeRobot/)
- [Embodied AI on AWS Batch](https://aws.amazon.com/blogs/spatial/embodied-ai-blog-series-part-1/)

---

## Phase 6 — Inference & Policy Serving (1–2 weeks)

RL rollouts and deployed policies both need a fast inference engine. In 2026 the choice is usually vLLM, SGLang, or TensorRT-LLM.

### What to learn
- **vLLM** — PagedAttention, continuous batching; easiest path, used inside OpenRLHF and verl.
- **SGLang** — RadixAttention for prompt/KV-cache reuse; best for structured, tool-use, multi-step agent prompts.
- **TensorRT-LLM** — max throughput on Hopper/Blackwell if you can tolerate compile time.
- **Weight resharding across serving ↔ training**: the hidden bottleneck in hybrid-engine RL.
- **VLA-specific serving**: action-chunking, low-latency decoding, on-robot quantization.

### Hands-on
1. Serve a 7B policy with vLLM; measure p50/p99 latency at different concurrencies.
2. Repeat with SGLang; observe RadixAttention cache hits when prompts share prefixes.
3. Benchmark a compiled TRT-LLM engine; quantify the build-time vs. throughput tradeoff.

Resources:
- [vLLM vs TensorRT-LLM vs SGLang — H100 benchmarks 2026](https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/)
- [Best LLM Inference Engines in 2026 — Yotta Labs](https://www.yottalabs.ai/post/best-llm-inference-engines-in-2026-vllm-tensorrt-llm-tgi-and-sglang-compared)
- [Choosing an Inference Framework — BentoML](https://bentoml.com/llm/getting-started/choosing-the-right-inference-framework)

---

## Phase 7 — Capstone: End-to-End Physical-AI Pipeline (2–4 weeks)

Pick one of these and drive it through the full stack:

1. **Sim-only RL at scale**: Isaac Lab (4k envs) → PPO/GRPO in verl → vLLM-served VLA evaluation → produce a policy with measurable sim success.
2. **World-model-in-the-loop RL**: Curate video with NeMo Curator → pretrain a small Cosmos-style WFM → train a Dreamer-style agent inside it → distill back into a reactive policy.
3. **VLA post-training**: Start from OpenVLA → RL post-train with SimpleVLA-RL on LeRobot data → serve through SGLang → benchmark on a held-out task suite.

Deliverables for the capstone: a written architecture doc, a throughput/cost table (steps/sec, $/M-steps), and a failure-mode analysis.

---

## Cross-Cutting Topics (learn continuously)

- **Storage & data**: Parquet/WebDataset/Lance for trajectory data; tiered storage for video (hot NVMe → warm S3); streaming dataloaders that don't stall the trainer.
- **Observability**: Weights & Biases + custom RL dashboards (rollout throughput, KL, reward variance, GPU SM util).
- **Cluster**: SLURM vs. Ray vs. Kubernetes+Kueue for mixed training/rollout workloads; gang scheduling; topology-aware placement for NVLink domains.
- **Cost**: $/successful-policy, not $/GPU-hour. Rollout generation dominates — that's where optimization pays.

---

## Reading / Watching Queue (quick wins)

- "Anatomy of RL Frameworks" — Hanif Leoputera
- "Open Source RL Libraries for LLMs" — Anyscale blog
- verl PyTorch Conference Europe 2026 + NVIDIA GTC 2026 talks
- NVIDIA Cosmos Technical Blog posts (WFM, Reason 2, Data Factory)
- 1X World Model PDF (best single read on humanoid world models)
- Hafner & Yan 2025 — "Training Agents Inside of Scalable World Models"

---

## Working Within an 8 GB Budget (RTX 3070)

An 8 GB consumer card can't fit a 7B model even in FP16 (14 GB weights alone). Don't fight that — pick exercises where the *infra concept* is what matters, and scale down the model. For the rest, rent.

### What the 3070 **can** do locally
- **MJX / MuJoCo Playground / mjlab** — GPU-parallel sim of classic control, locomotion, and small manipulation tasks at thousands of envs. Ideal for Phases 2–3.
- **Isaac Lab on small tasks** — Cartpole, Ant, Anymal-C locomotion at reduced env counts (~512–1024). Requires Linux + recent driver.
- **Small-model RL** — PPO/GRPO on 0.5B–1.5B models with QLoRA 4-bit + gradient checkpointing + FlashAttention. OpenRLHF and verl both support this path.
- **LoRA/QLoRA fine-tuning** of SmolVLA, ACT, small Dreamer variants.
- **Inference** of ≤3B models at 4-bit via vLLM or llama.cpp; ≤7B via AWQ/GPTQ with aggressive offload.
- **All the *reading, profiling, and Ray-cluster-of-one* exercises** — the distributed primitives don't need big GPUs to learn.

### What you should **rent cloud GPUs** for
Use [RunPod](https://www.runpod.io/), [Vast.ai](https://vast.ai/), [Lambda](https://lambdalabs.com/), or [Modal](https://modal.com/) on-demand. A single H100 at ~$2–3/hr or an 8× A100 pod at ~$10–15/hr is the cheapest way to do the "real" runs.

- **Phase 1 capstone**: one multi-GPU TorchTitan run (2–4 hours on an 8× A100 pod) to *feel* TP+PP+FSDP.
- **Phase 3 capstone**: one end-to-end verl/OpenRLHF GRPO run on a 7B model (4–8 hours on 8× H100).
- **Phase 4**: Cosmos fine-tuning or NeMo Curator at any real scale.
- **Phase 5**: Full OpenVLA fine-tune (7B, needs ≥40 GB).

Budget tip: Write and debug everything locally at tiny scale, then launch a cloud run only when the pipeline is proven. A well-prepared 4-hour rental teaches more than 40 hours of thrashing.

### Scaled-down substitutes per phase

| Phase | Local substitute on 3070 |
|-------|--------------------------|
| 1. Distributed backends | TorchTitan in `--local` mode with a 125M GPT; simulate multi-rank with `CUDA_VISIBLE_DEVICES` tricks; read the code. |
| 2. Simulation | MJX cartpole/humanoid at 4096 envs; Isaac Lab Ant at 512 envs. Both fit in 8 GB. |
| 3. RL orchestration | OpenRLHF with Qwen2.5-0.5B + QLoRA + vLLM, single-GPU Ray cluster. Study hybrid-engine code even if you can't run 70B. |
| 4. World models | DreamerV3 on DMC/Atari (fits easily); tiny latent-diffusion video model on toy data; *read* the Cosmos paper rather than retrain it. |
| 5. VLA / LeRobot | SmolVLA (450M) LoRA fine-tune on a LeRobot community dataset; ACT on PushT. |
| 6. Serving | vLLM / SGLang / llama.cpp serving a 3B model at 4-bit; measure latency/throughput curves. |
| 7. Capstone | "Sim-only RL at scale" capstone is fully doable locally using MJX + a small policy. |

### Essential local tooling
- **`bitsandbytes`** — 4-bit / 8-bit quantized optimizers and weights.
- **`peft`** — LoRA/QLoRA adapters.
- **`flash-attn`** — fits more into 8 GB; 3070 supports FA2.
- **`torch.compile`** + mixed precision (BF16 where supported, else FP16).
- **`nvitop`** / `nvidia-smi dmon` — know your memory and SM util at all times.
- **WSL2 caveat**: some sims (Isaac Sim GUI) are happier on native Linux; MJX and headless Isaac Lab work fine under WSL2.

### Recommended cloud workflow
1. Develop locally with a "tiny" config (1 GPU, 125M model, 64 envs).
2. Commit + push; launch the same code on a cloud pod via a launcher script.
3. Use Modal or RunPod "serverless" for short bursts; Lambda/Vast for multi-hour runs.
4. Always checkpoint to S3/R2 so you can spot-interrupt cheaply.

---

## Extended Resource Library

### Foundational Courses (free)
- [Stanford CS234 — Reinforcement Learning](https://web.stanford.edu/class/cs234/) (Emma Brunskill)
- [UC Berkeley CS285 — Deep RL](https://rail.eecs.berkeley.edu/deeprlcourse/) (Sergey Levine) — the canonical deep-RL course.
- [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course) — practical, PPO → advanced.
- [Hugging Face Robotics Course / LeRobot tutorials](https://huggingface.co/learn) — hands-on with SmolVLA, ACT.
- [Stanford CS336 — Language Modeling from Scratch](https://stanford-cs336.github.io/) — build a modern training stack end-to-end.
- [MIT 6.S191 — Intro to Deep Learning](http://introtodeeplearning.com/) — good refresher.
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/) — ML infra and ops fundamentals.
- [NVIDIA DLI — Robot Learning with Isaac Lab](https://www.nvidia.com/en-us/training/) (some free modules).

### Distributed Training & Infra Reads
- [HuggingFace — The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) — single best reference for parallelism strategies in 2026.
- [EleutherAI — Transformer Math 101](https://blog.eleuther.ai/transformer-math/) — memorize the arithmetic.
- [Stas Bekman — ML Engineering book](https://github.com/stas00/ml-engineering) — free, endlessly useful.
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) (Google / JAX team).
- [Cerebras / MosaicML blog archive](https://www.databricks.com/blog/category/generative-ai) — practical large-training writeups.
- [PyTorch TorchTitan docs](https://github.com/pytorch/torchtitan)
- [PyTorch distributed tutorials](https://pytorch.org/tutorials/beginner/dist_overview.html)

### Key Papers (in rough reading order)
**RL & RLHF foundations**
- [PPO (Schulman 2017)](https://arxiv.org/abs/1707.06347)
- [DPO](https://arxiv.org/abs/2305.18290), [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300), [DAPO](https://arxiv.org/abs/2503.14476)
- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [SRL: Scaling Distributed RL to 10k+ cores](https://openreview.net/forum?id=lajn1iROCu)

**Distributed / training infra**
- [Megatron-LM](https://arxiv.org/abs/1909.08053), [Megatron-Turing NLG](https://arxiv.org/abs/2201.11990)
- [ZeRO](https://arxiv.org/abs/1910.02054), [ZeRO-Infinity](https://arxiv.org/abs/2104.07857)
- [TorchTitan](https://arxiv.org/html/2410.06511v1)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691), [FlashAttention-3](https://arxiv.org/abs/2407.08608)

**RL frameworks**
- [OpenRLHF paper](https://arxiv.org/abs/2405.11143)
- [verl / HybridFlow paper](https://arxiv.org/abs/2409.19256)
- [Anyscale — Open Source RL Libraries for LLMs](https://www.anyscale.com/blog/open-source-rl-libraries-for-llms)

**Robotics / VLA**
- [RT-1](https://arxiv.org/abs/2212.06817), [RT-2](https://arxiv.org/abs/2307.15818), [RT-X / Open X-Embodiment](https://arxiv.org/abs/2310.08864)
- [OpenVLA](https://arxiv.org/abs/2406.09246)
- [π0 (Physical Intelligence)](https://www.physicalintelligence.company/blog/pi0)
- [ACT — Learning Fine-Grained Bimanual Manipulation](https://tonyzhaozh.github.io/aloha/)
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)
- [SimpleVLA-RL (ICLR 2026)](https://arxiv.org/abs/2509.09674)

**Simulation**
- [Isaac Gym](https://arxiv.org/abs/2108.10470)
- [Isaac Lab (2025)](https://arxiv.org/html/2511.04831v1)
- [MuJoCo Playground](https://arxiv.org/html/2502.08844v1)
- [ManiSkill3](https://arxiv.org/html/2410.00425v1)

**World models**
- [World Models — Ha & Schmidhuber 2018](https://arxiv.org/abs/1803.10122) (the original).
- [Dreamer](https://arxiv.org/abs/1912.01603), [DreamerV2](https://arxiv.org/abs/2010.02193), [DreamerV3](https://arxiv.org/abs/2301.04104).
- [Training Agents Inside Scalable World Models (Hafner & Yan 2025)](https://arxiv.org/pdf/2509.24527) — the DreamerV4 paper in spirit.
- [Genie](https://arxiv.org/abs/2402.15391), [Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/).
- [Cosmos WFM Platform](https://arxiv.org/abs/2501.03575).
- [UniSim / Learning Interactive Real-World Simulators](https://arxiv.org/abs/2310.06114).
- [1X World Model Technical Report](https://www.1x.tech/1x-world-model.pdf).
- [Video Generation Models in Robotics — Survey](https://arxiv.org/html/2601.07823v1).
- [Awesome-World-Model list](https://github.com/LMD0311/Awesome-World-Model).

### Blogs & Newsletters to Follow
- [Lilian Weng's blog](https://lilianweng.github.io/) — best deep-dives on RL/agent topics.
- [Sergey Levine's lab blog (BAIR)](https://bair.berkeley.edu/blog/).
- [Chip Huyen — ML infra posts](https://huyenchip.com/blog/).
- [Anyscale blog](https://www.anyscale.com/blog) — Ray-centric infra content.
- [NVIDIA Technical Blog — Robotics / Physical AI tag](https://developer.nvidia.com/blog/tag/robotics/).
- [HuggingFace blog — LeRobot tag](https://huggingface.co/blog?tag=robotics).
- [Interconnects (Nathan Lambert)](https://www.interconnects.ai/) — RLHF and post-training commentary.
- [Jack Clark — Import AI](https://importai.substack.com/).
- [Sebastian Raschka's Ahead of AI](https://magazine.sebastianraschka.com/).
- [1X tech blog](https://www.1x.tech/discover).
- [Physical Intelligence blog](https://www.physicalintelligence.company/blog).

### Code Repos worth Reading End-to-End
- [cleanrl](https://github.com/vwxyzjn/cleanrl) — single-file RL implementations; unmatched for *learning*.
- [tianshou](https://github.com/thu-ml/tianshou) — modular PyTorch RL.
- [TorchRL](https://github.com/pytorch/rl) — PyTorch's RL library.
- [Dreamer implementations](https://github.com/danijar/dreamerv3).
- [LeRobot](https://github.com/huggingface/lerobot).
- [OpenPI (Physical Intelligence)](https://github.com/Physical-Intelligence/openpi).
- [nanoGPT / nanoVLM](https://github.com/karpathy/nanoGPT) — for grounding yourself before tackling the big frameworks.

### Communities
- r/reinforcementlearning, r/MachineLearning
- LeRobot Discord, HuggingFace Discord
- Eleuther Discord (distributed training channels)
- NVIDIA Isaac / Omniverse forums
- PyTorch forums and the `#distributed` channel

---

## Suggested Cadence

| Week | Focus |
|------|-------|
| 1–2  | Phase 0 + Phase 1 (distributed basics + FSDP2/Megatron) |
| 3–5  | Phase 2 (Isaac Lab / MJX at scale) |
| 6–9  | Phase 3 (verl / OpenRLHF deep dive) |
| 10–13| Phase 4 (Cosmos + world models) |
| 14–16| Phase 5 (LeRobot + VLA) |
| 17–18| Phase 6 (serving) |
| 19–22| Phase 7 (capstone) |

Total: ~5 months at serious part-time pace.
