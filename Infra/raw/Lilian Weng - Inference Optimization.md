---
title: Large Transformer Model Inference Optimization
source: https://lilianweng.github.io/posts/2023-01-10-inference-optimization/
author:
  - "[[Lilian Weng]]"
published: 2023-01-10
created: 2026-04-16
description: "Comprehensive survey of inference-side optimizations: KV cache, quantization, speculative decoding, attention variants, batching."
tags:
  - inference
  - kv-cache
  - quantization
  - survey
---

# Large Transformer Model Inference Optimization

Lilian Weng's survey of inference-time optimizations. Optional in Phase 0 — this territory becomes central in Phase 6 (serving).

## Summary

- **KV cache**: memory formula, paged attention, cache quantization.
- **Quantization**: post-training int8/int4, GPTQ, AWQ, SmoothQuant.
- **Pruning and distillation**.
- **Attention variants**: multi-query, grouped-query, FlashAttention.
- **Speculative decoding** and Medusa-style parallel decoding.
- **Batching strategies**: static, dynamic, continuous (a.k.a. "in-flight") batching.

## Why it's in Phase 0 (as optional reading)

Inference memory math is easier to internalize than training memory math — good warm-up. Also: the KV-cache formula is exactly what you'll need to answer "can I serve a 7B model on my RTX 3070?"

## Connections

- Extends [[Transformer Math 101]] into the inference regime.
- Feeds directly into Phase 6 serving choices (vLLM, SGLang, TensorRT-LLM).
