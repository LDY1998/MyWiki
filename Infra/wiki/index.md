---
title: Infra Wiki - Index
type: map
status: active
tags:
  - index
  - ai-infra
updated: 2026-04-17
---

# Infra Wiki - Index

Curated knowledge base for AI infrastructure targeting large-scale RL, robotics, and world models.

## Maintenance

- [[Wiki Maintenance Schema]] - repo structure, page types, ingestion workflow, and link conventions.
- [[Infra Wiki Log]] - append-only record of ingests, reorgs, lint passes, and major synthesis work.
- [[Physical AI Infra]] - map of the current learning path and core topic clusters.
- [[llm-wiki]] - source idea for the persistent LLM-maintained wiki pattern.

## Learning plans

- [[physical-ai-infra-learning-plan|Physical AI Infra Learning Plan]] - the 8-phase roadmap.
- [[phase-0-foundations|Phase 0 - Foundations]] - distributed primitives, parallelism, GPU memory, Ray.

## Concepts

### Distributed primitives
- [[NCCL Collectives]] - the six primitives every framework calls.
- [[Ring All-Reduce]] - the bandwidth-optimal AllReduce algorithm.

### Parallelism
- [[Parallelism Strategies]] - taxonomy of DP / TP / PP / SP / CP / EP and how to compose them.
- [[Data Parallelism]] - DDP baseline.
- [[ZeRO]] - stage-1/2/3 sharding of model state.
- [[FSDP]] - PyTorch-native ZeRO-3 (FSDP1 and FSDP2).
- [[Tensor Parallelism]] - Megatron-style intra-layer sharding.
- [[Pipeline Parallelism]] - 1F1B and interleaved schedules.

### Memory & math
- [[Transformer Memory Math]] - formulas for params, grads, optimizer, activations, KV cache.

## Source summaries

### Core
- [[Ultra-Scale Playbook]] - HF systematic guide to all parallelism axes.
- [[Transformer Math 101]] - EleutherAI memory/compute/comm formulas.
- [[Illustrated Transformer]] - Jay Alammar's architecture primer.
- [[Megatron-LM Paper]] - NVIDIA PTD-P parallelism.
- [[ZeRO Paper]] - DeepSpeed memory-partitioned DP.
- [[PyTorch FSDP Tutorial]] - FSDP1 + FSDP2 API.
- [[NCCL User Guide]] - NVIDIA collective communication library.

### Optional
- [[JAX Scaling Book]] - Google/JAX perspective on scaling.
- [[ML Engineering Book]] - Stas Bekman's field notes.
- [[Lilian Weng - Inference Optimization]] - inference-side optimizations survey.

## Repository areas

- `raw/_inbox/` - new sources waiting for ingestion.
- `raw/sources/` - ingested raw source documents.
- `raw/assets/` - local attachments and images.
- `wiki/` - maintained synthesis layer.
- `plans/` - learning plans and phase roadmaps.
- `tools/` - helper scripts and generated reports.
- `archive/` - inactive or superseded material.

## Status

Phase 0 in progress as of 2026-04-16. Milestone: `mini-dist-trainer` repo (see [[phase-0-foundations]]).
