# Infra Wiki — Index

Curated knowledge base for AI infrastructure targeting large-scale RL, robotics, and world models.

## Learning plans

- [[physical-ai-infra-learning-plan|Physical AI Infra Learning Plan]] — the 8-phase roadmap.
- [[phase-0-foundations|Phase 0 — Foundations]] — distributed primitives, parallelism, GPU memory, Ray.

## Concepts

### Distributed primitives
- [[NCCL Collectives]] — the six primitives every framework calls.
- [[Ring All-Reduce]] — the bandwidth-optimal AllReduce algorithm.

### Parallelism
- [[Parallelism Strategies]] — taxonomy of DP / TP / PP / SP / CP / EP and how to compose them.
- [[Data Parallelism]] — DDP baseline.
- [[ZeRO]] — stage-1/2/3 sharding of model state.
- [[FSDP]] — PyTorch-native ZeRO-3 (FSDP1 and FSDP2).
- [[Tensor Parallelism]] — Megatron-style intra-layer sharding.
- [[Pipeline Parallelism]] — 1F1B and interleaved schedules.

### Memory & math
- [[Transformer Memory Math]] — formulas for params, grads, optimizer, activations, KV cache.

## Raw sources (Week 1 reading)

### Core
- [[Ultra-Scale Playbook]] — HF systematic guide to all parallelism axes.
- [[Transformer Math 101]] — EleutherAI memory/compute/comm formulas.
- [[Illustrated Transformer]] — Jay Alammar's architecture primer.
- [[Megatron-LM Paper]] — NVIDIA PTD-P parallelism.
- [[ZeRO Paper]] — DeepSpeed memory-partitioned DP.
- [[PyTorch FSDP Tutorial]] — FSDP1 + FSDP2 API.
- [[NCCL User Guide]] — NVIDIA collective communication library.

### Optional
- [[JAX Scaling Book]] — Google/JAX perspective on scaling.
- [[ML Engineering Book]] — Stas Bekman's field notes.
- [[Lilian Weng - Inference Optimization]] — inference-side optimizations survey.

## Status

Phase 0 in progress as of 2026-04-16. Milestone: `mini-dist-trainer` repo (see [[phase-0-foundations]]).
