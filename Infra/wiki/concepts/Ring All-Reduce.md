# Ring All-Reduce

The bandwidth-optimal algorithm for AllReduce on a ring of N GPUs. Two phases: ReduceScatter (N-1 steps) then AllGather (N-1 steps). Per-GPU communicated bytes = `2M(N-1)/N`, which approaches `2M` for large N — independent of how many GPUs participate.

## Why it matters

- "Free scaling" intuition for DDP: adding GPUs doesn't increase per-GPU AllReduce time (beyond a small constant).
- When ring breaks: very small messages (latency dominates → tree algorithm wins), or heterogeneous links (NVLink + PCIe in the same ring = you go at PCIe speed).

## Tree and double-binary-tree

NCCL uses a double binary tree for small messages and a ring (or rings) for large. Selected automatically based on size, topology, and `NCCL_ALGO`.

## References

- [[NCCL User Guide]]
- [[NCCL Collectives]]
