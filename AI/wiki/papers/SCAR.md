# SCAR: Shapley Credit Assignment for More Efficient RLHF

- **Arxiv**: [2505.20417](https://arxiv.org/abs/2505.20417)
- **Authors**: Meng Cao, Shuyuan Zhang, Xiaojun Chang, Doina Precup
- **Date**: May 2025

## Summary

Applies Shapley values from cooperative game theory to distribute a sequence-level reward among constituent tokens or text spans based on their principled marginal contributions. Creates dense per-token reward signals without training auxiliary critique models or requiring fine-grained human annotations.

## Key Techniques

1. **Shapley Value Decomposition**: Treats each token/span as a "player" in a cooperative game; the sequence reward is the coalition value; each player's credit is their average marginal contribution across all possible orderings
2. **Provable Optimality Preservation**: Mathematically proves that the Shapley-distributed rewards preserve the original optimal policy
3. **Dense Reward Signals**: Converts a single sequence-level scalar into per-token dense rewards, enabling more informative policy gradients

## Key Insight

Game theory provides a principled, axiomatic framework for fair credit distribution — Shapley values are the unique allocation satisfying efficiency, symmetry, additivity, and null-player properties. Unlike heuristic dense reward methods, this gives a theoretically grounded decomposition.

## Key Results

- Faster convergence and higher final reward scores vs standard RLHF and attention-based dense reward baselines
- Evaluated on sentiment control, text summarization, and instruction tuning

## Connections

- Offers a game-theoretic alternative to the value-network approach of [[VAPO]] and Monte Carlo approach of [[VinePPO]]
- Addresses the same [[Credit Assignment]] problem as [[TEMPO]], [[GRPO-lambda]], and [[SPO]]
- Unlike [[GRPO]] variants, SCAR works within the standard RLHF/PPO framework rather than extending GRPO
