---
title: The Ultra-Scale Playbook
source: https://huggingface.co/spaces/nanotron/ultrascale-playbook
author:
  - "[[HuggingFace Nanotron team]]"
published: 2025
created: 2026-04-16
description: "Open-source reference for training LLMs across thousands of GPUs. Covers every parallelism axis (DP, TP, PP, SP, CP, EP), FSDP/ZeRO, and how to compose them."
tags:
  - distributed-training
  - parallelism
  - reference
---

# The Ultra-Scale Playbook

An interactive, open-source book from the HuggingFace Nanotron team that systematically walks through everything needed to train a modern LLM at scale. Widely regarded as the single best 2026 reference for [[Parallelism Strategies]].

## Summary

Starts from single-GPU training and builds up layer by layer:

1. Single GPU: memory breakdown, recomputation, mixed precision.
2. Data Parallelism: DDP and its communication pattern.
3. [[ZeRO]] / [[FSDP]]: sharding optimizer states, gradients, and parameters.
4. [[Tensor Parallelism]]: intra-layer partitioning of matmuls.
5. Sequence / Context Parallelism: sharding the sequence dimension (including Ring Attention).
6. [[Pipeline Parallelism]]: inter-layer partitioning, 1F1B, interleaved schedules.
7. Expert Parallelism for MoE.
8. How to **compose** these into 4D/5D parallelism on real clusters.

Includes interactive diagrams and back-of-envelope memory/throughput calculators. The composition chapter is the payoff — it's what makes [[TorchTitan]] and Megatron configs legible.

## Why it's the starting point

Most other resources teach one technique at a time. The Playbook's contribution is the *systematic composition* view: which techniques are orthogonal, which conflict, and how the communication pattern of one interacts with the memory budget of another.

## How to use it for Phase 0

Read through FSDP (chapters 1–4) in Week 1. Revisit TP/PP/Sequence in Phase 1. The MoE chapter can be deferred.

## Connections

- Foundational for [[FSDP]], [[ZeRO]], [[Tensor Parallelism]], [[Pipeline Parallelism]].
- Complements [[Transformer Math 101]] (the Playbook shows the systems, Transformer Math gives the arithmetic).
