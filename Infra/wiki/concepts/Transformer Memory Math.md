# Transformer Memory Math

The back-of-envelope formulas for how much VRAM a transformer training (or inference) job needs.

## Parameters

For a GPT-style decoder with `n_layers` (L), `d_model` (D), `vocab` (V), context `C`:

```
N ≈ 12 · L · D²           (attn + MLP, MLP hidden = 4D)
  + 2 · V · D              (token + output embedding; or just V·D if tied)
```

For Llama-style with gated MLPs (hidden ≈ 2.67D × 3 projections), the MLP factor is closer to 8D² per layer.

## Training memory (mixed-precision Adam)

Per parameter, per rank (unsharded):

| State | Bytes |
|---|---|
| BF16/FP16 params | 2 |
| BF16/FP16 grads | 2 |
| FP32 master params | 4 |
| FP32 Adam m | 4 |
| FP32 Adam v | 4 |
| **Total** | **16** |

Add ~2 bytes for BF16 grad staging in some implementations → "~18 bytes per param" rule of thumb.

## Activation memory (per transformer layer, per sequence)

Roughly:

```
act_per_layer ≈ s · b · h · (34 + 5 · a · s / h)         (Megatron formula)
```

where `s` = seq, `b` = batch, `h` = hidden, `a` = num heads. The `5as/h` term is attention softmax memory — dominates at long context. [[FlashAttention]] removes it.

**Activation checkpointing** trades recompute for memory: O(√L) instead of O(L) activations stored, at the cost of one extra forward per layer.

## Inference / KV cache

```
kv_cache_bytes = 2 · L · s · (h / n_kv_heads_ratio) · bytes_per_elem · batch
```

For Llama-2-7B at fp16, 4k context, batch 1: ~2 GB. This is why serving long-context at scale is memory-bound.

## Sharding math

Under [[ZeRO]]-3 / [[FSDP]] on N DP ranks: divide params + grads + optimizer by N. Activations are **not** sharded by FSDP alone — that requires [[Tensor Parallelism]] + Sequence Parallelism.

## References

- [[Transformer Math 101]]
- [[Ultra-Scale Playbook]]
- [[Lilian Weng - Inference Optimization]]
