## Learning Plan: AI Infrastructure for Large-Scale RL, Robotics, and World Models

A software/data-layer roadmap for "Physical AI" infra — skipping ROS, middleware, and hardware. Front-loaded on the **2026 state of RL, world models, robot foundation models, and VLA**, then drilling into how those models are trained and (especially) **served**.

> **Priority shift from earlier drafts:**
> 1. Lead with the current landscape of RL / WFM / Robot FM / VLA — what exists, who builds it, what wins.
> 2. Cover training of those models, but treat large-scale distributed training as supporting infra rather than a primary topic.
> 3. **Inference and policy serving is the deepest part of this plan.** Rollouts dominate RL cost, and deployed policies are nothing but inference.

> **Hardware assumption:** 1× RTX 3070 (8 GB VRAM), 1 node. See [Working Within an 8 GB Budget](#working-within-an-8-gb-budget-rtx-3070) below.

---

## Mental Model: The Physical AI Stack

Four layers, top-down. Bottlenecks live at the interfaces.

```
┌────────────────────────────────────────────────────┐
│ 4. Policy inference / serving (vLLM, SGLang, TRT)  │  ← deepest focus
├────────────────────────────────────────────────────┤
│ 3. RL post-training (verl, OpenRLHF, AReaL)        │
├────────────────────────────────────────────────────┤
│ 2. Foundation models: WFM + Robot FM + VLA         │
├────────────────────────────────────────────────────┤
│ 1. Simulation & data (Isaac/MJX, LeRobot)          │
└────────────────────────────────────────────────────┘
```

Key insight from the verl paper: **80–90% of RL training time is sample generation**. That is why inference matters more than training for physical AI in 2026.

---

## Phase 0 — Foundations: Deep RL & Robot Learning (2 weeks)

Build the conceptual base before touching any 2026 framework.

### Core courses
- **Stanford CS224R — Deep Reinforcement Learning** (Chelsea Finn) — the spine of this plan. Imitation learning, policy gradients, actor-critic, model-based RL, offline RL, robot learning. https://cs224r.stanford.edu/
- **UC Berkeley CS285 — Deep Reinforcement Learning** (Sergey Levine) — deeper, longer, canonical companion.
- **Hugging Face Deep RL Course** — practical PPO → advanced; good lab partner.

### What you should be able to do at the end
- Explain PPO, DPO, GRPO, DAPO and when each is preferred.
- Describe imitation learning and behavior cloning, and how VLAs build on them.
- Sketch a model-based RL loop and where world models slot in.
- Read a 2026 RL/robotics paper and not get lost in the algorithm section.

### Hands-on
1. Implement PPO from scratch on a Gym task using `cleanrl` as the reference.
2. Run a small GRPO loop on a tiny LM (Qwen2.5-0.5B) following an OpenRLHF tutorial.
3. Behavior-clone an ACT or Diffusion Policy model on a LeRobot dataset.

---

## Phase 1 — 2026 Landscape: RL / WFM / Robot FM / VLA (2 weeks)

Survey the field before learning to train or serve any of it. The deliverable here is *understanding*, not code.

### What to map out

**RL for foundation models**
- Algorithms: GRPO (DeepSeek-R1, V3) is the dominant LLM-RL choice; PPO baseline; DPO/ORPO for preference; DAPO/GMPO/Pass@k for stability and exploration.
- Stacks: verl (ByteDance, 3D-HybridEngine), OpenRLHF (Ray + vLLM + DeepSpeed), AReaL (async RL with self-evolving data), SimpleVLA-RL (VLA-specific).

**World foundation models**
- NVIDIA Cosmos (open WFM platform; Cosmos Tokenizer; Cosmos Reason 2 for spatio-temporal reasoning).
- 1X World Model (humanoid-specific diffusion + Inverse Dynamics Model).
- DreamerV3 → Hafner & Yan 2025 ("Training Agents Inside Scalable World Models" — DreamerV4 in spirit).
- Genie 2 / Genie 3 — action-conditioned video models for policy eval and distillation.

**Robot foundation models & VLA**
- OpenVLA (7B): SigLIP+DINOv2 → Llama-2-7B → discrete action tokens.
- π0 / OpenPI (Physical Intelligence) — flow-matching policy.
- SmolVLA — efficient open VLA on LeRobot data.
- ACT, Diffusion Policy — common downstream policies.
- RT-1 / RT-2 / RT-X / Open X-Embodiment — the lineage that proved cross-embodiment data scales.

### How to study this phase
Read papers, blog posts, and tech reports. Produce a 2-page landscape map under `wiki/maps/`. Walk into Phase 2–4 already knowing *what's worth your time*.

### Resources
- [NVIDIA Cosmos](https://www.nvidia.com/en-us/ai/cosmos/) • [paper](https://arxiv.org/abs/2501.03575) • [scaling synthetic data](https://developer.nvidia.com/blog/scale-synthetic-data-and-physical-ai-reasoning-with-nvidia-cosmos-world-foundation-models/)
- [1X World Model PDF](https://www.1x.tech/1x-world-model.pdf)
- [OpenVLA](https://openvla.github.io/) • [paper](https://arxiv.org/abs/2406.09246)
- [π0 — Physical Intelligence](https://www.physicalintelligence.company/blog/pi0)
- [verl](https://github.com/volcengine/verl) • [HybridFlow paper](https://arxiv.org/abs/2409.19256)
- [OpenRLHF paper](https://arxiv.org/abs/2405.11143)
- [AReaL](https://github.com/inclusionAI/AReaL)
- [SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL)
- [Awesome-World-Model](https://github.com/LMD0311/Awesome-World-Model)
- [Anatomy of RL Frameworks — Hanif Leoputera](https://www.hanifleo.com/anatomy-of-rl-frameworks/)
- [Open Source RL Libraries for LLMs — Anyscale](https://www.anyscale.com/blog/open-source-rl-libraries-for-llms)

---

## Phase 2 — World Foundation Models (2–3 weeks)

How WFMs are tokenized, trained, and used.

### What to learn
- **Video tokenization**: discrete vs. continuous; temporal compression ratios (Cosmos Tokenizer is 8× compression / 12× faster).
- **Inverse Dynamics Models** — pixels → actions bridge.
- **Latent rollout for policy search** — training inside the world model ("dreaming").
- **Evaluation via generated futures** — "evaluating bits, not atoms."
- **Curation pipelines** — NeMo Curator (20M video-hours in 14 days on Blackwell).

### Hands-on
1. Use Cosmos + NeMo Curator to curate ~100h of video into training tokens.
2. Fine-tune a Cosmos WFM on a narrow domain; sample futures conditioned on an action sequence.
3. Train a tiny DreamerV3 on an Isaac Lab task; compare sample efficiency to model-free PPO.

### Resources
- [Cosmos GitHub](https://github.com/nvidia-cosmos)
- [Physical AI Data Factory Blueprint](https://nvidianews.nvidia.com/news/nvidia-announces-open-physical-ai-data-factory-blueprint-to-accelerate-robotics-vision-ai-agents-and-autonomous-vehicle-development)
- [Training Agents Inside Scalable World Models (Hafner & Yan 2025)](https://arxiv.org/pdf/2509.24527)
- [Dreamer](https://arxiv.org/abs/1912.01603) • [DreamerV2](https://arxiv.org/abs/2010.02193) • [DreamerV3](https://arxiv.org/abs/2301.04104)

---

## Phase 3 — Robot Foundation Models & VLA Training (2–3 weeks)

The other half of the foundation-model side: the policy itself.

### What to learn
- **LeRobot dataset format** — HuggingFace's unified schema for multi-modal robot trajectories.
- **VLA architectures** — vision encoder (SigLIP/DINOv2) + projector + LLM backbone + action head (discrete tokens, flow matching, or diffusion).
- **Action representations** — discrete tokens vs. continuous chunks vs. diffusion targets.
- **Cross-embodiment data** — RT-X / Open X-Embodiment lessons.
- **Data-factory pipelines** — AWS Batch-based pipelines for embodied AI.

### Hands-on
1. Ingest a LeRobot dataset; write a dataloader that streams video + proprio + language.
2. Fine-tune SmolVLA or OpenVLA on a small task; benchmark action-token accuracy.
3. Build a sim-to-real data mixing schedule (X% sim from Isaac Lab + Y% real from LeRobot).

### Resources
- [LeRobot](https://github.com/huggingface/lerobot)
- [SmolVLA blog](https://huggingface.co/blog/smolvla)
- [OpenPI / π0](https://github.com/Physical-Intelligence/openpi)
- [LeRobot + ROCm fine-tuning](https://rocm.blogs.amd.com/artificial-intelligence/rocm-lerobot/README.html)
- [EmbodiFlow LeRobot pipeline](https://io-ai.tech/platform/en/guides/Pipeline/LeRobot/)
- [Embodied AI on AWS Batch](https://aws.amazon.com/blogs/spatial/embodied-ai-blog-series-part-1/)

---

## Phase 4 — RL Post-Training for Foundation Models (2–3 weeks)

Once you have a base policy, RL is how it gets sharper. The 2026 stacks all combine Ray + an inference engine + a training backend.

### Core concepts
- **Hybrid Engine / colocated rollout** — same GPUs run both training and inference phases.
- **Async RL** — decouple rollout workers from learner to hide generation latency.
- **Resharding** — moving weights from training layout (TP/PP) to inference layout (TP only).
- **GRPO vs. PPO vs. DAPO** — when each wins; GRPO dominates LLM-RL in 2026.
- **VLA-RL specifics** — SimpleVLA-RL (ICLR 2026), rollouts that produce image+action trajectories.

### Hands-on
1. Run OpenRLHF GRPO on a 0.5B–1.5B model with QLoRA + a single reward model; trace the Ray actor graph.
2. Reproduce a small-scale verl run with 3D-HybridEngine; measure generation throughput vs. naive split.
3. Swap in a VLA policy with SimpleVLA-RL.

### Resources
- [verl](https://github.com/volcengine/verl)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [OpenRLHF vs verl deep dive](https://langcopilot.com/posts/2025-11-06-openrlhf-vs-verl-ray-framework-deep)
- [RL Infrastructure 2025 — Introl](https://introl.com/blog/reinforcement-learning-infrastructure-rlhf-robotics-gpu-clusters-2025)

---

## Phase 5 — Inference & Policy Serving (4–6 weeks) — Primary Focus

The deepest part of the plan. Modern physical AI is inference-bound: rollouts dominate RL cost, deployed policies are pure inference, and on-robot constraints make latency a first-class concern.

### Engines and what makes them different
- **vLLM** — PagedAttention, continuous batching; the easy default; embedded inside OpenRLHF and verl.
- **SGLang** — RadixAttention for prompt/KV-cache reuse; best for structured, tool-use, multi-step agent prompts.
- **TensorRT-LLM** — max throughput on Hopper/Blackwell when you can tolerate compile time.
- **llama.cpp / mlx-lm** — small, on-device, edge.

### Concepts to master
- **PagedAttention** — block-based KV cache; fragmentation, eviction.
- **RadixAttention** — prefix-tree KV reuse; when it pays off.
- **Continuous batching / iteration-level scheduling** — the throughput lever.
- **Speculative decoding** (Medusa, EAGLE, lookahead) — the latency lever.
- **Prefill–decode disaggregation** — DistServe-style architectures.
- **Quantization** — AWQ, GPTQ, FP8, INT4; per-tensor vs. per-channel; calibration vs. weight-only.
- **Weight resharding training ↔ serving** — the hidden bottleneck in hybrid-engine RL.
- **VLA-specific serving** — action chunking, low-latency decoding, on-robot quantization, multi-modal prefill.

### Hands-on
1. Serve a 3B–7B policy with vLLM; measure p50/p99 latency at different concurrencies; sweep `max_num_seqs` and watch the curve.
2. Repeat with SGLang; design a prompt set that exercises RadixAttention and quantify cache-hit gains.
3. Build a TRT-LLM engine; quantify build-time vs. throughput tradeoff.
4. Quantize the same model with AWQ and GPTQ; compare quality (perplexity / a small downstream eval) and speed.
5. Implement speculative decoding with a small draft model; measure tokens/sec uplift.
6. Serve a VLA at low latency: action chunking, AWQ 4-bit, batch=1; measure end-to-end action-step latency at a 30 Hz control rate.

### Resources
- [vLLM docs](https://docs.vllm.ai/) • [GitHub](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [vLLM vs TensorRT-LLM vs SGLang — H100 benchmarks 2026](https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/)
- [Best LLM Inference Engines in 2026 — Yotta Labs](https://www.yottalabs.ai/post/best-llm-inference-engines-in-2026-vllm-tensorrt-llm-tgi-and-sglang-compared)
- [Choosing an Inference Framework — BentoML](https://bentoml.com/llm/getting-started/choosing-the-right-inference-framework)
- [Lilian Weng — Large Transformer Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
- [AWQ paper](https://arxiv.org/abs/2306.00978) • [GPTQ paper](https://arxiv.org/abs/2210.17323)
- [Medusa](https://arxiv.org/abs/2401.10774) • [EAGLE](https://arxiv.org/abs/2401.15077)
- [DistServe](https://arxiv.org/abs/2401.09670)

---

## Phase 6 — Training Infrastructure (Reference, 1–2 weeks)

Distributed training matters, but for this plan it's *supporting* infra — enough to read papers and run small experiments, not enough to compete with Megatron specialists. GPU-parallel simulation lives here too because it's the sample factory feeding training.

### What to learn at a high level
- **FSDP2** (per-parameter DTensor sharding) — modern PyTorch path.
- **Megatron-LM / Megatron-Core** — TP + PP + SP reference.
- **TorchTitan** — composes FSDP2 + TP + PP + `torch.compile` + Float8.
- **DeepSpeed ZeRO 1/2/3** — conceptually important for the OpenRLHF training stack.
- **GPU-parallel simulation** — Isaac Lab, MJX / MuJoCo Warp, mjlab, Newton 1.0, ManiSkill3.

### Hands-on (light)
1. Train a small GPT with TorchTitan on 2 simulated ranks; toggle TP and PP and watch memory/throughput.
2. Profile with `torch.profiler` / Nsight; identify whether you're compute- or comm-bound.
3. Run an Isaac Lab locomotion env at 4096 parallel environments; compare throughput on MJX/MJWarp.

### Resources
- [TorchTitan](https://github.com/pytorch/torchtitan) • [paper](https://arxiv.org/html/2410.06511v1)
- [Megatron-DeepSpeed](https://github.com/deepspeedai/Megatron-DeepSpeed)
- [Isaac Lab (2025)](https://arxiv.org/html/2511.04831v1)
- [MuJoCo Playground](https://playground.mujoco.org/)
- [mjlab](https://github.com/mujocolab/mjlab)
- [HuggingFace — Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- [EleutherAI — Transformer Math 101](https://blog.eleuther.ai/transformer-math/)

---

## Phase 7 — Capstone: End-to-End Physical-AI Pipeline (3–5 weeks)

Pick one and drive it through the full stack. **Every capstone option must include a serious inference benchmark.**

1. **VLA inference deep dive** — take OpenVLA or SmolVLA → quantize (AWQ / GPTQ / FP8) → serve with vLLM and SGLang → benchmark p99 latency at a 30 Hz control rate → bonus: speculative decoding for action tokens.
2. **World-model-in-the-loop RL** — curate video with NeMo Curator → pretrain a small Cosmos-style WFM → train a Dreamer-style agent inside it → serve the resulting policy through vLLM → distillation eval.
3. **Sim-only RL with serving focus** — Isaac Lab → GRPO in OpenRLHF → vLLM-served policy with continuous batching → end-to-end p50/p99 latency report at varying concurrencies.

Deliverables for any capstone: a written architecture doc, a **latency / throughput / quality table** (p50/p99, tokens/sec, $/M-steps), a quantization ablation, and a failure-mode analysis.

---

## Cross-Cutting Topics (learn continuously)

- **Storage & data**: Parquet/WebDataset/Lance for trajectory data; tiered storage for video (hot NVMe → warm S3); streaming dataloaders that don't stall the trainer or rollout worker.
- **Observability**: Weights & Biases + custom RL/inference dashboards (rollout throughput, KL, reward variance, GPU SM util, KV cache hit rate, prefill latency).
- **Cluster**: SLURM vs. Ray vs. Kubernetes+Kueue for mixed training/rollout/serving workloads; gang scheduling; topology-aware placement.
- **Cost**: $/successful-policy and $/M-tokens-served, not $/GPU-hour. Rollout and serving dominate — that's where optimization pays.

---

## Reading / Watching Queue (quick wins)

- Chelsea Finn — Stanford CS224R lectures.
- Sergey Levine — UC Berkeley CS285 lectures.
- "Anatomy of RL Frameworks" — Hanif Leoputera.
- "Open Source RL Libraries for LLMs" — Anyscale blog.
- vLLM and SGLang design docs / blog posts.
- Lilian Weng — "Large Transformer Inference Optimization."
- NVIDIA Cosmos technical blog series (WFM, Reason 2, Data Factory).
- 1X World Model PDF.
- Hafner & Yan 2025 — "Training Agents Inside of Scalable World Models."

---

## Working Within an 8 GB Budget (RTX 3070)

An 8 GB consumer card can't fit a 7B model even in FP16 (14 GB weights alone). Don't fight that — pick exercises where the *infra concept* is what matters, scale down the model, and rent for the rest.

### What the 3070 **can** do locally
- **MJX / MuJoCo Playground / mjlab** — GPU-parallel sim of classic control, locomotion, and small manipulation tasks at thousands of envs.
- **Isaac Lab on small tasks** — Cartpole, Ant, Anymal-C locomotion at reduced env counts (~512–1024).
- **Small-model RL** — PPO/GRPO on 0.5B–1.5B models with QLoRA 4-bit + gradient checkpointing + FlashAttention.
- **LoRA/QLoRA fine-tuning** of SmolVLA, ACT, small Dreamer variants.
- **Inference** of ≤3B models at 4-bit via vLLM or llama.cpp; ≤7B via AWQ/GPTQ with aggressive offload.
- **All the *reading, profiling, and Ray-cluster-of-one* exercises** — the infra concepts don't need big GPUs to learn.

### What you should **rent cloud GPUs** for
Use [RunPod](https://www.runpod.io/), [Vast.ai](https://vast.ai/), [Lambda](https://lambdalabs.com/), or [Modal](https://modal.com/) on-demand. A single H100 at ~$2–3/hr or an 8× A100 pod at ~$10–15/hr is the cheapest way to do "real" runs.

- **Phase 2**: Cosmos fine-tuning or NeMo Curator at any real scale.
- **Phase 3**: Full OpenVLA fine-tune (7B, needs ≥40 GB).
- **Phase 4**: One end-to-end verl/OpenRLHF GRPO run on a 7B model (4–8 hours on 8× H100).
- **Phase 5**: TRT-LLM compilation and 7B-class inference benchmarks at production scale.
- **Phase 6**: One multi-GPU TorchTitan run (2–4 hours on an 8× A100 pod) to *feel* TP+PP+FSDP.

Budget tip: write and debug everything locally at tiny scale, then launch a cloud run only when the pipeline is proven. A well-prepared 4-hour rental teaches more than 40 hours of thrashing.

### Scaled-down substitutes per phase

| Phase | Local substitute on 3070 |
|-------|--------------------------|
| 0. Deep RL foundations | `cleanrl` PPO on Gym; Qwen2.5-0.5B GRPO via OpenRLHF; ACT/Diffusion Policy BC on a small LeRobot task. |
| 1. Landscape | Reading, mapping. No GPU needed. |
| 2. World models | DreamerV3 on DMC/Atari; tiny latent-diffusion video model on toy data; *read* the Cosmos paper rather than retrain it. |
| 3. Robot FMs / VLA | SmolVLA (450M) LoRA fine-tune on a LeRobot community dataset; ACT on PushT. |
| 4. RL post-training | OpenRLHF with Qwen2.5-0.5B + QLoRA + vLLM, single-GPU Ray cluster. Study hybrid-engine code even if you can't run 70B. |
| 5. Inference (primary) | vLLM / SGLang / llama.cpp serving a 3B model at 4-bit; full latency/throughput sweeps; AWQ/GPTQ ablations; speculative decoding with a small draft. |
| 6. Training infra | TorchTitan in `--local` mode with a 125M GPT; simulate multi-rank with `CUDA_VISIBLE_DEVICES`; MJX/Isaac Lab small envs. |
| 7. Capstone | The "VLA inference deep dive" capstone is fully doable locally on a small VLA. |

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
- [Stanford CS224R — Deep Reinforcement Learning](https://cs224r.stanford.edu/) (Chelsea Finn) — primary RL course for this plan.
- [UC Berkeley CS285 — Deep RL](https://rail.eecs.berkeley.edu/deeprlcourse/) (Sergey Levine) — canonical companion.
- [Stanford CS234 — Reinforcement Learning](https://web.stanford.edu/class/cs234/) (Emma Brunskill).
- [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course) — practical, PPO → advanced.
- [Hugging Face Robotics Course / LeRobot tutorials](https://huggingface.co/learn) — hands-on with SmolVLA, ACT.
- [Stanford CS336 — Language Modeling from Scratch](https://stanford-cs336.github.io/) — build a modern training stack end-to-end.
- [MIT 6.S191 — Intro to Deep Learning](http://introtodeeplearning.com/) — refresher.
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/) — ML infra and ops fundamentals.
- [NVIDIA DLI — Robot Learning with Isaac Lab](https://www.nvidia.com/en-us/training/) (some free modules).

### Inference & Serving (read deeply)
- [Lilian Weng — Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
- [vLLM blog and docs](https://blog.vllm.ai/)
- [SGLang blog](https://lmsys.org/blog/)
- [DistServe](https://arxiv.org/abs/2401.09670) — prefill/decode disaggregation.
- [Sarathi-Serve](https://arxiv.org/abs/2403.02310) — chunked prefill.
- [AWQ](https://arxiv.org/abs/2306.00978), [GPTQ](https://arxiv.org/abs/2210.17323), [SmoothQuant](https://arxiv.org/abs/2211.10438).
- [Medusa](https://arxiv.org/abs/2401.10774), [EAGLE](https://arxiv.org/abs/2401.15077).

### Distributed Training & Infra (reference)
- [HuggingFace — The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook).
- [EleutherAI — Transformer Math 101](https://blog.eleuther.ai/transformer-math/).
- [Stas Bekman — ML Engineering book](https://github.com/stas00/ml-engineering).
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) (Google / JAX team).
- [PyTorch TorchTitan](https://github.com/pytorch/torchtitan).
- [PyTorch distributed tutorials](https://pytorch.org/tutorials/beginner/dist_overview.html).

### Key Papers (in rough reading order)

**RL & RLHF foundations**
- [PPO (Schulman 2017)](https://arxiv.org/abs/1707.06347)
- [DPO](https://arxiv.org/abs/2305.18290), [GRPO / DeepSeekMath](https://arxiv.org/abs/2402.03300), [DAPO](https://arxiv.org/abs/2503.14476)
- [InstructGPT](https://arxiv.org/abs/2203.02155)

**RL frameworks**
- [OpenRLHF paper](https://arxiv.org/abs/2405.11143)
- [verl / HybridFlow paper](https://arxiv.org/abs/2409.19256)

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
- [Training Agents Inside Scalable World Models (Hafner & Yan 2025)](https://arxiv.org/pdf/2509.24527).
- [Genie](https://arxiv.org/abs/2402.15391), [Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/).
- [Cosmos WFM Platform](https://arxiv.org/abs/2501.03575).
- [UniSim / Learning Interactive Real-World Simulators](https://arxiv.org/abs/2310.06114).
- [1X World Model Technical Report](https://www.1x.tech/1x-world-model.pdf).
- [Awesome-World-Model list](https://github.com/LMD0311/Awesome-World-Model).

**Distributed / training infra (background)**
- [Megatron-LM](https://arxiv.org/abs/1909.08053), [Megatron-Turing NLG](https://arxiv.org/abs/2201.11990)
- [ZeRO](https://arxiv.org/abs/1910.02054), [ZeRO-Infinity](https://arxiv.org/abs/2104.07857)
- [TorchTitan](https://arxiv.org/html/2410.06511v1)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691), [FlashAttention-3](https://arxiv.org/abs/2407.08608)

### Blogs & Newsletters to Follow
- [Lilian Weng's blog](https://lilianweng.github.io/) — best deep-dives on RL/agent topics and inference optimization.
- [Sergey Levine's lab blog (BAIR)](https://bair.berkeley.edu/blog/).
- [Chelsea Finn's group page](https://ai.stanford.edu/~cbfinn/).
- [Chip Huyen — ML infra posts](https://huyenchip.com/blog/).
- [Anyscale blog](https://www.anyscale.com/blog) — Ray-centric infra content.
- [NVIDIA Technical Blog — Robotics / Physical AI tag](https://developer.nvidia.com/blog/tag/robotics/).
- [HuggingFace blog — LeRobot tag](https://huggingface.co/blog?tag=robotics).
- [Interconnects (Nathan Lambert)](https://www.interconnects.ai/) — RLHF and post-training commentary.
- [Sebastian Raschka's Ahead of AI](https://magazine.sebastianraschka.com/).
- [1X tech blog](https://www.1x.tech/discover).
- [Physical Intelligence blog](https://www.physicalintelligence.company/blog).

### Code Repos worth Reading End-to-End
- [cleanrl](https://github.com/vwxyzjn/cleanrl) — single-file RL implementations; unmatched for *learning*.
- [TorchRL](https://github.com/pytorch/rl).
- [Dreamer implementations](https://github.com/danijar/dreamerv3).
- [LeRobot](https://github.com/huggingface/lerobot).
- [OpenPI (Physical Intelligence)](https://github.com/Physical-Intelligence/openpi).
- [vLLM](https://github.com/vllm-project/vllm) — read the scheduler and PagedAttention paths.
- [SGLang](https://github.com/sgl-project/sglang) — read the RadixCache implementation.
- [nanoGPT / nanoVLM](https://github.com/karpathy/nanoGPT) — for grounding before tackling the big frameworks.

### Communities
- r/reinforcementlearning, r/MachineLearning
- LeRobot Discord, HuggingFace Discord
- vLLM Discord, SGLang Discord
- NVIDIA Isaac / Omniverse forums
- PyTorch forums and the `#distributed` channel

---

## Suggested Cadence

| Week  | Focus |
|-------|-------|
| 1–2   | Phase 0 — Deep RL foundations (Chelsea Finn / Sergey Levine + cleanrl) |
| 3–4   | Phase 1 — 2026 landscape survey |
| 5–7   | Phase 2 — World foundation models |
| 8–10  | Phase 3 — Robot FMs / VLA training |
| 11–13 | Phase 4 — RL post-training |
| 14–19 | **Phase 5 — Inference & Serving** (primary focus) |
| 20–21 | Phase 6 — Training infra reference |
| 22–26 | Phase 7 — Capstone with serving benchmarks |

Total: ~6 months at serious part-time pace, weighted toward inference.
