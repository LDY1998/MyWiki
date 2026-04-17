# ZeRO (Zero Redundancy Optimizer)

A strategy for eliminating the memory redundancy of [[Data Parallelism]] by *sharding* model state across data-parallel ranks instead of replicating it. Introduced by DeepSpeed; the intellectual parent of [[FSDP]].

## The three stages

Let N be the DP world size. Each GPU normally holds params (P), gradients (G), optimizer states (O) fully.

| Stage | Partitions | Per-GPU memory | Extra comm vs. DDP |
|-------|-----------|----------------|--------------------|
| ZeRO-1 | O only | P + G + O/N | 0 |
| ZeRO-2 | O + G | P + G/N + O/N | 0 |
| ZeRO-3 | O + G + P | P/N + G/N + O/N | ~1.5× |

The win grows with N. For Adam in BF16+FP32-master, optimizer states (the momentum+variance+master copy) are ~12 bytes/param, so ZeRO-1 alone saves a lot.

## ZeRO-3 = FSDP

ZeRO-3's pattern:
1. **Forward**: AllGather params for current layer → compute → discard.
2. **Backward**: AllGather params for current layer → compute grad → ReduceScatter grad.
3. **Optimizer step**: each rank updates its shard.

That's exactly what [[FSDP]] does. FSDP2 uses per-parameter `DTensor` sharding, which composes cleanly with [[Tensor Parallelism]].

## When ZeRO hurts

ZeRO-3's 1.5× communication can bottleneck on slow interconnects. Real clusters combine ZeRO (across DP axis) with [[Tensor Parallelism]] (within a node) and [[Pipeline Parallelism]] (across nodes) to land in sensible bandwidth regimes.

## References

- [[ZeRO Paper]]
- [[FSDP]]
- [[Ultra-Scale Playbook]]
