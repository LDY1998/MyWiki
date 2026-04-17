# Data Parallelism (DP / DDP)

The baseline distributed training pattern: every rank holds a full replica of the model; each rank processes a different micro-batch; gradients are averaged via `AllReduce` before the optimizer step.

## The DDP communication pattern

Per training step:
1. Forward (local).
2. Backward (local); as each bucket of gradients becomes ready, launch an async `AllReduce`.
3. Optimizer step (local), using the averaged gradients.

Overlap of backward compute with gradient `AllReduce` is what makes DDP "free" at reasonable scales.

## When DP is not enough

- **Memory**: full model + grads + optimizer on every GPU → doesn't fit past ~1–2B params on an 80 GB card with Adam. Solved by [[ZeRO]] / [[FSDP]].
- **Scale**: AllReduce latency grows with N. Beyond a few hundred GPUs you combine DP with [[Tensor Parallelism]] / [[Pipeline Parallelism]] to reduce the DP world size.

## References

- [[Ultra-Scale Playbook]]
- [[NCCL Collectives]]
- [[ZeRO]]
