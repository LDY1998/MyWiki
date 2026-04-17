---
title: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
source: https://arxiv.org/abs/1910.02054
author:
  - "[[Samyam Rajbhandari]]"
  - "[[Microsoft DeepSpeed]]"
published: 2019-10-04
created: 2026-04-16
description: "The Zero Redundancy Optimizer. Introduces stage-1/2/3 sharding of optimizer states, gradients, and parameters across data-parallel ranks. The intellectual ancestor of FSDP."
tags:
  - paper
  - zero
  - sharding
  - deepspeed
---

# ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

The Microsoft/DeepSpeed paper that showed you can keep the simplicity of data parallelism but eliminate its memory redundancy by sharding model state across ranks.

## Summary

Observes that standard [[Data Parallelism]] replicates three kinds of state on every GPU — optimizer states (O), gradients (G), and parameters (P) — and partitions them incrementally:

- **ZeRO-1 (Pos)**: partition optimizer states only. ~4× memory reduction with no extra communication.
- **ZeRO-2 (Pos+g)**: also partition gradients. ~8× reduction, same communication volume as DDP.
- **ZeRO-3 (Pos+g+p)**: also partition parameters. ~Nx reduction (where N = world size), at the cost of ~1.5× communication (all-gather weights on forward, all-gather on backward, reduce-scatter grads).

ZeRO-Infinity (follow-up) extends this with CPU / NVMe offload.

## Why it's foundational

- ZeRO-3 ≈ [[FSDP]]. Understanding ZeRO makes FSDP's API obvious.
- The "gather weights just-in-time, discard after use" pattern is the dominant memory strategy in 2026 frameworks.

## Key insight

The apparent "free lunch" of ZeRO-3 is not free: you pay 1.5× the communication volume of DDP. On fast NVLink domains this is fine; across slower interconnects it can bottleneck. This is why real clusters combine ZeRO with [[Tensor Parallelism]] and [[Pipeline Parallelism]] rather than pushing ZeRO-3 across all GPUs.

## What to read

Sections 1–5. Skim the empirical results; the partitioning figure is the most important artifact.

## Connections

- [[FSDP]] is the PyTorch-native implementation of ZeRO-3 (with per-parameter sharding in FSDP2).
- Complementary to [[Tensor Parallelism]] (ZeRO shards across the DP axis; TP shards within a layer).
- [[Ultra-Scale Playbook]] has the clearest modern walkthrough of ZeRO's communication pattern.
