# NCCL Collectives

The six communication primitives that every distributed training framework ultimately calls. Understanding their semantics — and their communication cost — is the atomic unit of reasoning about parallelism.

## The six primitives

| Collective | Before | After |
|---|---|---|
| **Broadcast** | rank 0 has `x` | all ranks have `x` |
| **Reduce** | all ranks have `x_i` | rank 0 has `Σ x_i` |
| **AllReduce** | all ranks have `x_i` | all ranks have `Σ x_i` |
| **AllGather** | each rank has shard `x_i` | all ranks have `[x_0, x_1, …, x_{N-1}]` |
| **ReduceScatter** | all ranks have full `x_i` | each rank has `(Σ x_i)[i-th shard]` |
| **Send/Recv** | — | point-to-point |

Identity: `AllReduce = ReduceScatter + AllGather`. This is why FSDP's memory win isn't "free" — it pays the same communication volume as DDP, split across two phases.

## Cost (ring algorithm)

For message size `M` and world size `N`:

- AllReduce: `2M × (N-1)/N` bytes per GPU total.
- AllGather / ReduceScatter: `M × (N-1)/N` each.
- Broadcast: `M × (N-1)/N`.

Ring saturates bandwidth; tree algorithms win for small messages (latency-bound).

## Where each shows up

- [[Data Parallelism]] → AllReduce on gradients.
- [[FSDP]] forward → AllGather on parameters.
- [[FSDP]] backward → ReduceScatter on gradients.
- [[Tensor Parallelism]] → two AllReduce per transformer block.
- [[Pipeline Parallelism]] → Send/Recv between stages.
- Checkpoint load → Broadcast from rank 0.

## Environment knobs

- `NCCL_DEBUG=INFO` — print topology and algorithm selection.
- `NCCL_P2P_DISABLE=1` — force traffic through host memory (debugging).
- `NCCL_ALGO=Ring|Tree` — force an algorithm.
- `NCCL_PROTO=Simple|LL|LL128` — protocol for small/large messages.

## References

- [[NCCL User Guide]]
- [[Ultra-Scale Playbook]]
- [[Ring All-Reduce]]
