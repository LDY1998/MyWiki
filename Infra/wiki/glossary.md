---
title: Glossary
type: glossary
status: active
tags:
  - glossary
updated: 2026-04-17
---

# Glossary

Shared vocabulary for the Infra wiki.

## Terms

- [[NCCL Collectives|Collective]] - communication primitive used by distributed training frameworks.
- [[Data Parallelism|Data parallelism]] - replicate model weights and shard data across ranks.
- [[Tensor Parallelism|Tensor parallelism]] - shard tensor operations within model layers.
- [[Pipeline Parallelism|Pipeline parallelism]] - shard model layers across stages.
- [[ZeRO]] - partition optimizer state, gradients, and parameters across data-parallel ranks.
- [[FSDP]] - PyTorch-native fully sharded data parallel implementation.

