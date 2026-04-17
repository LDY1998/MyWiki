# Phase 0 — Foundations: Distributed Primitives, Parallelism, GPU Memory, Ray

**Duration:** 1–2 weeks (10–20 focused hours)
**Hardware:** 1× RTX 3070 (8 GB). Everything here is designed to run locally — we simulate "multi-GPU" with multiple processes on one GPU + CPU (`gloo`) backends.

**Goal:** By the end of this phase you can read any 2026 paper / framework doc (verl, TorchTitan, Cosmos) and the parallelism and communication sections make immediate sense. You don't need to have *trained* a 70B model — you need to know exactly where every byte of VRAM goes and what every collective does.

---

## Learning Objectives (the checklist)

You're done with Phase 0 when you can, without looking anything up:

1. Draw the memory breakdown of a 1B transformer in Adam FP16 training, to within ~15%.
2. Explain the difference between `all_reduce`, `all_gather`, `reduce_scatter`, `broadcast`, and when each is used.
3. Describe DP / TP / PP / SP / CP / EP and *which dimension each shards*.
4. Explain why FSDP/ZeRO-3 is "all-gather forward, reduce-scatter backward."
5. Explain ring vs. tree all-reduce and when each wins (bandwidth vs. latency bound).
6. Write a Ray actor, pass tensors between actors, and explain what `ray.put` / `ray.get` do to the object store.
7. Back-of-envelope: "Can a 7B model fit on 1× RTX 3070 with QLoRA + gradient checkpointing?" (Answer: yes, just barely. Know why.)

---

## Week 1 — Distributed Primitives & Parallelism Theory

### Topics
- **Collectives**: all-reduce, all-gather, reduce-scatter, broadcast, scatter, gather, point-to-point send/recv.
- **Topology**: ring vs. tree algorithms, bandwidth vs. latency, NVLink vs. PCIe vs. Ethernet.
- **The parallelism zoo**: DP, TP (tensor), PP (pipeline), SP (sequence), CP (context), EP (expert).
- **ZeRO-1 / 2 / 3** and how ZeRO-3 ≈ FSDP.
- **Mixed precision**: FP32 master weights, BF16 vs. FP16 activations, loss scaling.

### Primary reading (do these in order)

1. **[HuggingFace — The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)** — start here. Read the first ~40% (through TP/PP/FSDP). This is the best single 2026 reference.
2. **[EleutherAI — Transformer Math 101](https://blog.eleuther.ai/transformer-math/)** — memorize the memory and FLOP formulas.
3. **[Jay Alammar — Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)** — if your transformer mental model is rusty.
4. **[NVIDIA — Efficient Large-Scale Language Model Training on GPU Clusters (Megatron paper)](https://arxiv.org/abs/2104.04473)** — the canonical TP+PP reference. Read sections 1–4 carefully.
5. **[ZeRO paper](https://arxiv.org/abs/1910.02054)** — read sections 1–5. Focus on the memory-state partitioning figure.
6. **[PyTorch FSDP tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)** and **[FSDP2 overview](https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)**.
7. **[NCCL collectives docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html)** — skim the list, understand the communication patterns.

### Optional (pick 1–2)
- [How to Scale Your Model (JAX scaling book)](https://jax-ml.github.io/scaling-book/) — chapters 1–3.
- [Stas Bekman — ML Engineering book, Parallelism chapter](https://github.com/stas00/ml-engineering/tree/master/training/model-parallelism).
- [Sebastian Raschka — Understanding parameter counts](https://magazine.sebastianraschka.com/).
- Lilian Weng — [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/).

### Short exercises (try to do these without looking things up)

- **Memory math drills.** For each, compute params / grad / optimizer / activation memory:
  - 125M GPT, seq 1024, BF16+FP32 master, Adam, batch 8.
  - 1.3B GPT, seq 2048, BF16 mixed, Adam.
  - 7B Llama, seq 4096, what's the *minimum* VRAM with ZeRO-3 on 8 GPUs?
  - 7B Llama, QLoRA (4-bit base + LoRA adapters) on 1× RTX 3070 — does it fit?
- **Collective identification.** For each pseudo-code snippet below, name the collective:
  - "each rank has a partial gradient; each rank needs the full averaged gradient" → ?
  - "rank 0 has weights; all ranks need a copy" → ?
  - "each rank has a shard of weights; we need the full matrix temporarily for forward" → ?
  - "each rank has a full gradient; we want each rank to end up owning 1/N of the averaged gradient" → ?

(Answers: all-reduce, broadcast, all-gather, reduce-scatter.)

---

## Week 2 — Hands-On: Collectives, FSDP, and Ray

All hands-on here runs on your single 3070. We use two tricks to learn "distributed" locally:
- **Multi-process single-GPU**: launch 2–4 processes that all use `cuda:0` and talk over NCCL. Small models still fit.
- **CPU / gloo backend**: for anything where you just want to see the collective pattern, use `gloo`. No GPU needed.

### Setup

```bash
# fresh venv
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy einops matplotlib wandb ray[default]
```

Verify:
```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

### Exercise 1 — Raw collectives (1–2 hours)

Write `collectives_demo.py`. For each collective (broadcast, all-reduce, all-gather, reduce-scatter, barrier), launch 4 processes with `torchrun --nproc_per_node=4 --standalone collectives_demo.py` using `backend="gloo"` on CPU. Print before/after tensors per rank.

Reference: [PyTorch distributed tutorial](https://pytorch.org/tutorials/beginner/dist_overview.html), [Writing Distributed Applications](https://pytorch.org/tutorials/intermediate/dist_tuto.html).

**Success criterion:** you can predict the output of any collective before running it.

### Exercise 2 — Hand-rolled Data Parallel (2–3 hours)

Train a tiny MLP on MNIST (or a nanoGPT char-model on tinyshakespeare). Do **not** use `DistributedDataParallel`. Manually:

1. Split the dataset across ranks.
2. Forward + backward locally.
3. `all_reduce` the gradients with `op=SUM`, divide by world size.
4. Optimizer step.

Verify the loss curves match a single-process run with the same effective batch size. Now wrap the model in `torch.nn.parallel.DistributedDataParallel` and confirm you get the same numbers.

**Success criterion:** you know, to the line, what DDP is doing under the hood.

### Exercise 3 — From DDP to FSDP (2–3 hours)

Take a slightly larger model (a 50–100M param GPT — nanoGPT is perfect) and:

1. Train with DDP. Note peak memory.
2. Switch to `fully_shard` (FSDP2). Note peak memory.
3. Toggle activation checkpointing. Note peak memory again.
4. Plot: parameters vs. peak VRAM for each config.

Reference: [nanoGPT](https://github.com/karpathy/nanoGPT), [FSDP2 docs](https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html).

**Success criterion:** you can explain the memory numbers you observed using the formulas from Transformer Math 101.

### Exercise 4 — Ray basics (1–2 hours)

Work through the official [Ray Core Walkthrough](https://docs.ray.io/en/latest/ray-core/walkthrough.html), then build this small system:

- One `Learner` actor holding model weights.
- Four `RolloutWorker` actors that pull weights, run a dummy "rollout" (just sample trajectories from a `gym.make("CartPole-v1")`), and return batches.
- A main loop that collects rollouts asynchronously via `ray.wait`, then "trains" (no-op gradient step) and broadcasts new weights.

This is exactly the actor topology inside OpenRLHF / verl, but with no learning — pure plumbing.

**Success criterion:** your main loop never blocks on a single worker; slow workers don't stall fast ones.

---

## Milestone Project — "Mini-Distributed-Trainer"

Combine everything above into one repo. This is your Phase 0 portfolio artifact.

### Deliverable

A single repo `mini-dist-trainer/` that contains **three deliverables**:

#### 1. `memcalc.py` — Transformer memory calculator

A CLI that given a model config prints the memory breakdown:

```bash
python memcalc.py --params 1.3e9 --seq 2048 --batch 4 --dtype bf16 --optimizer adam --parallelism zero3 --world-size 8
```

Output:
```
Parameters:         2.48 GB  (1.3B × 2 bytes, sharded 1/8)
Gradients:          2.48 GB  (same shape, sharded)
Optimizer (fp32×2): 9.92 GB  → 1.24 GB/rank
Activations:       ~3.1 GB   (with activation ckpt: ~0.7 GB)
Total per GPU:     ~7.9 GB
```

Supports: `none | zero1 | zero2 | zero3 | tp | pp` (you can fake TP/PP as "divide params by degree" — the point is the mental model).

**Bonus:** Add a `--answer-q` mode that answers:
- "Will 7B fit on 1×RTX 3070 with QLoRA?"
- "What's the minimum world size for 70B BF16 training with ZeRO-3?"

#### 2. `train_gpt.py` — Three backends for the same tiny GPT

Take nanoGPT (or a 10M-param GPT you write yourself) and train it on tinyshakespeare in **three modes**, selectable by `--mode`:

- `--mode ddp` — manual all-reduce data parallel (from Exercise 2).
- `--mode fsdp` — `fully_shard` (FSDP2) with activation checkpointing.
- `--mode manual-zero` — your own ZeRO-1 implementation (shard optimizer state across ranks, all-gather before step, reduce-scatter gradients). ~100 lines.

Run each on 4 simulated ranks (4 processes, all on `cuda:0` — batch size small enough to fit). Log:
- Peak memory per rank.
- Throughput (tokens/sec).
- Loss curve (all three must match to within noise).

Produce a README with a table comparing the three and one-paragraph analysis.

#### 3. `ray_rollout/` — Async rollout system skeleton

From Exercise 4, cleaned up. One `Learner`, N `RolloutWorker`s, a `ReplayBuffer` actor, and a main loop. No actual learning yet — just the infra. Add:
- Weight sync every K steps (broadcast pattern).
- Worker failure tolerance (`max_task_retries`).
- Metrics to W&B or a local JSONL log.

This is the skeleton we'll fill in during Phase 3.

### README requirements

Write a `README.md` that includes:
- **Memory budget walkthrough**: show the memory math for one of your runs, both predicted (from `memcalc.py`) and measured (from `torch.cuda.max_memory_allocated()`). Explain any gap.
- **Collective trace**: for the FSDP path, describe *every* collective that happens in one forward-backward-step cycle.
- **One open question** you still have. (The goal is honest self-assessment — Phase 0 is done when you know what you don't know.)

### Time budget
- `memcalc.py`: 2–3 hours
- `train_gpt.py`: 4–6 hours (the ZeRO-1 implementation is the bulk)
- `ray_rollout/`: 2–3 hours
- README + writeup: 1–2 hours

**Total: ~12–14 hours.** If it takes 25, that's fine — you're learning. If it takes 5, you probably skipped the manual ZeRO — go back and do it.

---

## Self-Assessment Quiz (take this before declaring Phase 0 done)

Answer these from memory. If you miss more than 2, revisit the corresponding reading.

1. Why is FSDP's forward pass an `all_gather` and its backward a `reduce_scatter`?
2. In ZeRO-3, what's sharded and what's replicated?
3. If I have an 8-GPU node with NVLink and a 32-GPU cluster across 4 nodes with 100 Gb/s Ethernet, where does tensor parallelism work well and where does it break down?
4. Why does pipeline parallelism introduce "bubbles"? What does 1F1B scheduling do about them?
5. What's the difference between BF16 and FP16, and why does BF16 not need loss scaling?
6. For a 7B model training with Adam in BF16+FP32-master, roughly how many bytes per parameter do you need across all states? (Answer: ~18.)
7. Why does `ray.put` exist — what problem does it solve vs. just passing args?
8. If you set `NCCL_P2P_DISABLE=1`, what gets slower and why?
9. What is activation checkpointing trading off?
10. Given a single RTX 3070, what's the largest dense model you can *fine-tune* (LoRA) and the largest you can *infer* (4-bit)? Justify with memory math.

---

## What You Can Skip in Phase 0 (and come back to later)

- **Expert Parallelism (EP)** — only matters once you touch MoE models. Defer to Phase 3 if you hit it.
- **Context Parallelism (CP) / Ring Attention** — long-context training trick. Defer to Phase 1.
- **Float8 / FP8 training** — modern efficiency win but not conceptually core. Phase 1.
- **InfiniBand / topology-aware scheduling** — relevant only at >1 node. Phase 7.
- **Custom CUDA kernels / Triton** — useful but orthogonal; a separate rabbit hole.

---

## Graduation Criteria

You can move to Phase 1 when:

- [ ] Your `mini-dist-trainer` repo exists and all three deliverables run.
- [ ] You scored ≥8/10 on the self-assessment quiz.
- [ ] You can read the [TorchTitan paper](https://arxiv.org/html/2410.06511v1) section on parallelism composition and follow it with no confusion.
- [ ] You can read a verl or OpenRLHF config file and roughly predict what the cluster layout will look like.

Ship the repo to GitHub with a short blog post or README writeup. That artifact is your receipt that Phase 0 happened — and it's exactly the kind of thing you'll extend (not rewrite) in every subsequent phase.
