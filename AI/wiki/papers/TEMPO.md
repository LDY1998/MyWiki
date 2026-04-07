# TEMPO: Exploiting Tree Structure for Credit Assignment in RL Training of LLMs

- **Arxiv**: [2509.18314](https://arxiv.org/abs/2509.18314)
- **Authors**: UMass Amherst
- **Date**: September 2025
- **Venue**: ICLR 2026

## Summary

Proposes TEMPO (Tree-Estimated Mean Prefix Value for Policy Optimization), a critic-free algorithm that augments [[GRPO]]'s group-relative signal with branch-gated temporal-difference corrections derived from a prefix tree. At non-branching tokens, TEMPO reduces to GRPO; at branching tokens (where responses diverge), it injects precise token-level credit without a learned value network or extra rollouts.

## Key Techniques

1. **Prefix-to-Tree (P2T)**: Converts a group of sampled responses into a prefix tree and computes nonparametric prefix values V(s) by aggregating descendant outcomes
2. **Branch-Gated TD Credit**: TD corrections are only applied at branching points in the tree — tokens where responses diverge — since shared prefixes carry zero differential signal
3. **No Extra Cost**: Exploits structure already present in GRPO's group sampling; no additional rollouts, value networks, or reward models needed

## Key Insight

If all responses in a group share a prefix, those tokens don't matter for credit assignment — only the divergence points do. The prefix tree naturally identifies these critical decision points.

## Key Results

- Outperforms PPO and GRPO on Qwen3-1.7B/4B across math (MATH, GSM-HARD, AMC23) and medical (MedQA, MedMCQA, MMLU-Medical) benchmarks
- Reaches higher validation accuracy with less wall-clock time than baselines
- Code: [github.com/fatebreaker/tempo](https://github.com/fatebreaker/tempo)

## Connections

- Directly extends [[GRPO]] by adding fine-grained credit at zero extra cost
- Complements [[VinePPO]]'s Monte Carlo approach — both avoid value networks, but TEMPO uses tree structure instead of extra rollouts
- Addresses the same coarse [[Credit Assignment]] problem as [[GRPO-lambda]], [[SPO]], and [[SCAR]]
