---
title: Transformer Math 101
source: https://blog.eleuther.ai/transformer-math/
author:
  - "[[Quentin Anthony]]"
  - "[[Stella Biderman]]"
  - "[[Hailey Schoelkopf]]"
published: 2023-04-18
created: 2026-04-16
description: "Closed-form formulas for compute, memory, and communication cost of transformer training. The arithmetic you need to reason about any distributed training setup."
tags:
  - memory-math
  - compute
  - reference
---

# Transformer Math 101

An EleutherAI blog post that distills transformer training into a handful of formulas you can apply by hand. Essential prerequisite for reading framework internals.

## Summary

Gives closed-form expressions for:

- **Parameter count** as a function of `n_layers`, `d_model`, `vocab`, and whether you tie embeddings.
- **Compute (FLOPs)**: the canonical `6 × N × D` approximation for training (`2ND` forward, `4ND` backward) and `2 × N × D` for inference.
- **Training memory**: params, gradients, optimizer states, activations — separately and summed, under FP32 / FP16+FP32-master / BF16 / mixed Adam. The famous "16–20 bytes per parameter" rule comes from here.
- **Activation memory** as a function of batch × sequence × hidden, and how activation checkpointing reduces it.
- **Inference memory**: KV cache sizing, attention bandwidth.
- **Communication cost** of [[Data Parallelism]], [[ZeRO]], [[Tensor Parallelism]], [[Pipeline Parallelism]].

## Why it matters

Every capacity decision (will this fit? how many GPUs do I need? what's the bottleneck?) reduces to these formulas. Without them you're guessing; with them you can predict peak memory to within ~15% before launching a run.

## Key numbers to memorize

- Mixed-precision Adam: ~16–20 bytes/param for model+grad+optimizer (2+2+4+4+4 = 16 minimum, often higher with momentum buffers).
- 6 FLOPs per token per parameter for training.
- KV cache per token per layer: `2 × d_model × bytes_per_element`.

## How to use it for Phase 0

Work the exercises in [[Phase 0 Foundations]] ("Memory math drills") using these formulas directly, no code.

## Connections

- Operationalized by [[Ultra-Scale Playbook]] and [[ZeRO]].
- The "6ND" approximation underlies the [[Chinchilla Scaling Laws]].
