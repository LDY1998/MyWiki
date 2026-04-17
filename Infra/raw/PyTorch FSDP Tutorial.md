---
title: Getting Started with Fully Sharded Data Parallel (FSDP / FSDP2)
source: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
author:
  - "[[PyTorch team]]"
published: 2024
created: 2026-04-16
description: "Official PyTorch tutorial for FSDP + the FSDP2 `fully_shard` API docs. The hands-on entry point for the ZeRO-3 pattern in PyTorch."
tags:
  - pytorch
  - fsdp
  - tutorial
---

# PyTorch FSDP Tutorial (and FSDP2 docs)

The official PyTorch walkthroughs for Fully Sharded Data Parallel. Read alongside:
- FSDP1 tutorial: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- FSDP2 API: https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
- Advanced FSDP tutorial: https://pytorch.org/tutorials/intermediate/FSDP_advanced_tutorial.html

## Summary

Shows how to:
- Wrap a model with `FullyShardedDataParallel` (FSDP1) or `fully_shard` (FSDP2).
- Choose an auto-wrap policy (per-layer vs. per-transformer-block).
- Enable activation checkpointing via `apply_activation_checkpointing`.
- Save and load sharded checkpoints.
- Mix in mixed precision (`MixedPrecisionPolicy`).

FSDP2 is the modern path: it uses per-parameter `DTensor` sharding instead of the FSDP1 "FlatParameter." This composes cleanly with [[Tensor Parallelism]] and [[Pipeline Parallelism]] — which is why [[TorchTitan]] uses FSDP2 throughout.

## Why read the tutorial *after* the ZeRO paper

The API is trivial; the concepts aren't. If you read the tutorial first without the [[ZeRO]] mental model, you'll get a checklist of incantations with no understanding of when they'll bite you (e.g., why wrap at transformer-block granularity, not per-layer; why `MixedPrecisionPolicy` applies differently to params vs. reduce dtype).

## How to use it in Phase 0

Exercise 3 in [[Phase 0 Foundations]] ("From DDP to FSDP") is directly based on this tutorial.

## Connections

- Implementation of [[ZeRO]] stage 3.
- Composes with [[Tensor Parallelism]] in [[TorchTitan]].
