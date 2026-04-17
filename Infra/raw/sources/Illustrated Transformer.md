---
title: The Illustrated Transformer
source: http://jalammar.github.io/illustrated-transformer/
author:
  - "[[Jay Alammar]]"
published: 2018-06-27
created: 2026-04-16
description: "Visual walkthrough of the Transformer architecture. Still the most referenced intuition primer for attention, multi-head attention, positional encoding, and the encoder/decoder stack."
tags:
  - foundations
  - transformer
  - primer
---

# The Illustrated Transformer

A blog post that explains the Transformer architecture through step-by-step diagrams. Not distributed-training content — it's a *prerequisite* mental model.

## Summary

Walks through:
- Input embeddings and positional encoding.
- Scaled dot-product attention (Q, K, V).
- Multi-head attention: why split heads, how outputs are concatenated.
- The full encoder block (attention + FFN + residuals + layer norm).
- Decoder and cross-attention.
- Final linear + softmax for token prediction.

Every step has a diagram showing tensor shapes.

## Why it's in Phase 0

If your mental model of attention shapes is fuzzy, every later discussion (TP shards heads across ranks, SP shards sequence, FSDP gathers per-layer weights) will slide off. 30 minutes here pays for itself many times over.

## Connections

- Prerequisite for [[Tensor Parallelism]] (which shards along the head dimension).
- Prerequisite for [[FlashAttention]] (a drop-in optimized attention kernel).
