# FSDP (Fully Sharded Data Parallel)

PyTorch's native implementation of [[ZeRO]]-3. Shards parameters, gradients, and optimizer states across data-parallel ranks, materializing full parameters only transiently during forward and backward.

## Lifecycle of a wrapped module

1. Forward: `AllGather` parameters → compute → free local full params.
2. Backward: `AllGather` params → compute gradients → `ReduceScatter` grads → free.
3. Optimizer step: operates on local shard only.

## FSDP1 vs FSDP2

- **FSDP1** (`FullyShardedDataParallel`): flattens each wrapped module's params into a single "FlatParameter." Simple but makes composition with [[Tensor Parallelism]] awkward.
- **FSDP2** (`fully_shard`): per-parameter sharding using `DTensor`. Cleanly composes with TP, PP, and Float8. Used by [[TorchTitan]]. Preferred in 2026.

## Key config decisions

- **Auto-wrap granularity**: per transformer block is the standard; too fine (per linear) = too many gathers; too coarse (whole model) = no sharding benefit.
- **`MixedPrecisionPolicy`**: param dtype, reduce dtype, buffer dtype — often BF16 compute with FP32 reduce for stability.
- **Activation checkpointing**: usually applied *inside* the wrap target to recover activation memory.
- **Offload**: CPU offload for optimizer states is a last resort — massive slowdown.

## Memory intuition

For a 7B model with Adam on N GPUs:
- DDP per-GPU: ~112 GB (doesn't fit).
- ZeRO-1 per-GPU: ~28 + 84/N GB.
- FSDP / ZeRO-3 per-GPU: ~112/N GB + activations.

So on 8×H100 (80 GB), 7B fits comfortably under FSDP but not DDP.

## References

- [[ZeRO]]
- [[PyTorch FSDP Tutorial]]
- [[TorchTitan]]
