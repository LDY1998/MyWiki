---
title: How to Scale Your Model
source: https://jax-ml.github.io/scaling-book/
author:
  - "[[Google DeepMind]]"
  - "[[JAX team]]"
published: 2024
created: 2026-04-16
description: "A JAX-native book on how LLMs are trained at scale, with deep treatments of roofline analysis, sharding, and TPU-specific details that translate directly to GPUs."
tags:
  - jax
  - tpu
  - parallelism
  - reference
---

# How to Scale Your Model

A free online book from Google / JAX authors. Complements the HuggingFace [[Ultra-Scale Playbook]] with a more mathematical, roofline-oriented perspective. TPU-centric in its examples but the underlying principles are hardware-agnostic.

## Summary

Chapters cover:
- Roofline analysis: compute-bound vs. memory-bound vs. communication-bound regimes.
- All the parallelism axes, framed as "which mesh dimension is each tensor sharded along?"
- How to reason about per-device arithmetic intensity.
- Attention, FFN, and embedding sharding patterns.
- Inference-specific scaling (KV cache, speculative decoding).

## Why it's worth reading alongside Ultra-Scale Playbook

The Playbook is framework-flavored (PyTorch/Nanotron). Scaling Book is math-flavored (JAX `shard_map`, tensor meshes). Reading both gives stereoscopic vision on the same material — the same ideas expressed in two idioms, which is exactly what you want when the ideas are the point.

## How to use it in Phase 0

Optional. Read chapters 1–3 in Week 1 if you finish the Playbook early.

## Connections

- Parallel to [[Ultra-Scale Playbook]].
- Roofline material reinforces [[Transformer Math 101]].
