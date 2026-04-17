# Tensor Parallelism (TP)

Intra-layer model parallelism: a single matmul is split across multiple GPUs. Introduced by Megatron-LM. Best used *inside* a node where NVLink makes the required `AllReduce`s cheap.

## The Megatron recipe for a transformer block

- **MLP**: first linear is column-parallel (no comm needed after split), second linear is row-parallel (needs `AllReduce` across ranks at the end).
- **Self-attention**: partition heads across TP ranks (each rank owns a slice of Q, K, V heads); output projection is row-parallel → `AllReduce`.

Net cost: **2 AllReduces per transformer block per forward, 2 per backward.**

## Where TP hurts

- Cross-node TP: AllReduces go over Ethernet/InfiniBand instead of NVLink → often a net loss.
- Very small models: communication overhead dominates.
- TP degree must divide the number of heads and the hidden size evenly.

## Sequence Parallelism (SP)

A refinement: shard the sequence dimension of LayerNorm / Dropout / residual-add ops that TP leaves replicated. Reduces activation memory without changing the TP communication pattern. Worth enabling whenever TP is enabled.

## References

- [[Megatron-LM Paper]]
- [[Ultra-Scale Playbook]]
- [[NCCL Collectives]]
