---
title: ML Engineering (Open Book)
source: https://github.com/stas00/ml-engineering
author:
  - "[[Stas Bekman]]"
published: continuously updated
created: 2026-04-16
description: "Battle-scarred field notes on large-model training: parallelism, debugging, hardware pitfalls, NCCL tuning, SLURM, checkpointing. Assumes you've already read the theory."
tags:
  - ml-engineering
  - training
  - reference
---

# ML Engineering — Stas Bekman

A continuously updated open-source book drawn from Stas's experience on BLOOM-176B, IDEFICS, and various HuggingFace training runs. Reads like the operator's manual no one else wrote down.

## Summary

Chapters cover:
- **Model parallelism** cheat sheet: which axis, which framework, which config knobs.
- **Debugging distributed training**: how to diagnose NaN losses, hangs, rank mismatches, NCCL timeouts.
- **Hardware**: GPU topology, NVLink vs. PCIe, InfiniBand, storage throughput requirements.
- **SLURM** recipes for multi-node jobs.
- **Checkpointing** at scale: what goes wrong and how to structure it.
- **Performance**: profiling workflows, common bottleneck patterns.

## Why it matters

The [[Ultra-Scale Playbook]] tells you what to do; Stas tells you what goes wrong when you try. Together they cover the "happy path" and the "everything's on fire" path.

## How to use it in Phase 0

Read the parallelism cheat sheet and the debugging chapter. The rest is reference material you'll come back to when you hit specific problems in Phase 1+.

## Connections

- Complements [[Ultra-Scale Playbook]] and [[Megatron-LM Paper]] with operational reality.
