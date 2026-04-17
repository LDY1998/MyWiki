# Parallelism Strategies

A taxonomy of how work is split across GPUs when training (or serving) large models. Real clusters compose multiple axes.

## The axes

| Axis | Shards what | Comm per step | Best used |
|------|-------------|---------------|-----------|
| [[Data Parallelism]] (DP) | Batch | `AllReduce` of grads | Always, outermost |
| [[ZeRO]] / [[FSDP]] | Optimizer/grad/param across DP ranks | `AllGather` + `ReduceScatter` | Replace replicated DP |
| [[Tensor Parallelism]] (TP) | Matmuls within a layer | `AllReduce` per block | Inside a node (NVLink) |
| Sequence Parallelism (SP) | Sequence dim of non-TP ops | Free with TP | Always with TP |
| Context Parallelism (CP) | Attention sequence | Ring exchange of K/V | Long context (>32k) |
| [[Pipeline Parallelism]] (PP) | Layers across stages | `Send`/`Recv` | Across nodes |
| Expert Parallelism (EP) | MoE experts | `All-to-all` | MoE models |

## The composition pattern

A typical 2026 large-model training layout on a multi-node H100 cluster:

```
Outer:   DP (ZeRO-1 or FSDP across nodes)
Mid:     PP (between nodes)
Inner:   TP + SP (inside a node, across NVLink)
```

The guiding principle: **match the collective's bandwidth to the interconnect's bandwidth**. TP's AllReduces are large and frequent → put them on NVLink. DP's AllReduce can tolerate slower links → outermost.

## For small-scale / RL training

RL frameworks ([[OpenRLHF]], verl) often skip PP entirely and lean on FSDP + TP because:
- Rollout models reshape dynamically between training and inference layouts — PP makes that painful.
- Models are usually ≤70B, and ZeRO-3 + TP is enough.

## References

- [[Ultra-Scale Playbook]]
- [[Megatron-LM Paper]]
- [[ZeRO]]
