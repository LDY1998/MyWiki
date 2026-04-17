---
title: "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"
source: https://arxiv.org/abs/2104.04473
author:
  - "[[Deepak Narayanan]]"
  - "[[NVIDIA]]"
published: 2021-04-09
created: 2026-04-16
description: "The canonical reference for composing tensor, pipeline, and data parallelism on GPU clusters. Introduces PTD-P parallelism and the 1F1B pipeline schedule used everywhere since."
tags:
  - paper
  - tensor-parallelism
  - pipeline-parallelism
  - megatron
---

# Megatron-LM: Efficient Large-Scale LM Training on GPU Clusters

The NVIDIA paper that defined how to train trillion-parameter transformers by composing [[Tensor Parallelism]], [[Pipeline Parallelism]], and [[Data Parallelism]] — "PTD-P" parallelism.

## Summary

Key contributions:

1. **Tensor Parallelism** for transformer blocks: splits the MLP and attention matmuls column-wise then row-wise so that only two `all_reduce` calls per layer are needed. Best run *inside* a node (NVLink).
2. **Pipeline Parallelism with interleaved 1F1B schedule**: stages execute one forward then one backward per microbatch, with virtual stages that reduce the "pipeline bubble" proportional to `1/v`.
3. **Composition**: TP for intra-node, PP for inter-node, DP on top of both. Demonstrates 502 PFLOP/s sustained on 3072 A100s training a 1T parameter model.

## Why it's foundational

Every modern large-training codebase (Megatron-LM, Megatron-DeepSpeed, NeMo, [[TorchTitan]], Colossal-AI) either implements or inherits these ideas. Reading this paper is the cheapest way to understand why frameworks are shaped the way they are.

## What to read

Sections 1–4 are the core. Section 5 (scheduler) and 6 (evaluation) are worth skimming. Skip the earlier PTD derivations if short on time.

## Key formulas

- Pipeline bubble fraction: `(p-1) / (m)` for naive schedule, `(p-1) / (v × m)` for interleaved, where `p` = pipeline stages, `m` = microbatches, `v` = virtual stages per device.
- TP communication per layer: two `all_reduce` over hidden size in forward, two in backward.

## Connections

- Extends [[Megatron Paper (original)]] (2019) which introduced TP.
- Complemented by [[ZeRO]] (sharding the DP dimension) and [[FSDP]].
- Implemented end-to-end in [[TorchTitan]], verl, and NeMo.
