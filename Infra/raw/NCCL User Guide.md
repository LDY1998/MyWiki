---
title: NCCL User Guide — Collective Operations
source: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html
author:
  - "[[NVIDIA]]"
published: continuously updated
created: 2026-04-16
description: "Authoritative docs for NCCL: the GPU collective communication library underneath every PyTorch distributed training job."
tags:
  - nccl
  - collectives
  - reference
---

# NCCL User Guide

NCCL (NVIDIA Collective Communications Library) is the backend that actually moves tensors between GPUs in every major framework. Understanding its primitives — not the C API, just the *semantics* — is the single highest-ROI reference for Phase 0.

## Summary

The primitives:

| Collective | Semantics |
|------------|-----------|
| `Broadcast` | Rank 0 sends its buffer; all ranks end up with a copy. |
| `Reduce` | All ranks contribute; rank 0 gets the sum (or other op). |
| `AllReduce` | All ranks contribute; all ranks get the reduced result. Equivalent to `ReduceScatter` + `AllGather`. |
| `AllGather` | Each rank contributes a shard; all ranks end up with the concatenation. |
| `ReduceScatter` | All ranks contribute full buffers; each rank ends up owning a reduced shard. |
| `Send/Recv` | Point-to-point. |

The guide also documents:
- Topology detection (NVLink / PCIe / IB) and algorithm choice (ring vs. tree vs. double binary tree).
- Environment variables (`NCCL_DEBUG`, `NCCL_P2P_DISABLE`, `NCCL_ALGO`, `NCCL_PROTO`) — knowing these is the difference between "training is mysteriously slow" and "training is mysteriously slow for a *reason*."
- Bandwidth characteristics: bus bandwidth ≈ algorithm bandwidth × `2(N-1)/N` for ring all-reduce.

## Why it matters

Every framework-level pattern resolves to some combination of these:
- DDP = `AllReduce` on gradients.
- FSDP forward = `AllGather` on params.
- FSDP backward = `ReduceScatter` on grads.
- TP = two `AllReduce` per transformer block.
- PP = `Send`/`Recv` between stages.

If you can map a framework API call to the exact collective it will issue, you can reason about its performance.

## How to use it

Skim the overview and the collectives reference. You don't need to memorize the C API — just the operations table above.

## Connections

- Operations underlie [[ZeRO]], [[FSDP]], [[Tensor Parallelism]], [[Pipeline Parallelism]].
- [[Ring All-Reduce]] is the most common underlying algorithm.
