# VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks

- **Arxiv**: [2504.05118](https://arxiv.org/abs/2504.05118)
- **Authors**: ByteDance Seed
- **Date**: April 2025
- **Raw**: [[VAPO_2504.05118]]

## Summary

Presents VAPO (Value-based Augmented PPO), the first value-model-based RL framework to outperform value-free methods (GRPO, DAPO) on long chain-of-thought reasoning. Achieves 60.4 on AIME 2024 with Qwen-32B, surpassing [[DAPO]] by 10+ points. Identifies three core challenges of using value models with long CoT: value model bias, heterogeneous sequence lengths, and sparse rewards.

## Key Techniques

1. **Value-Pretraining**: Warm up the value network with Monte Carlo returns before RL training to fix initialization bias
2. **Decoupled-GAE**: Use lambda=1.0 for value updates (unbiased) and adaptive lambda for policy updates (faster convergence)
3. **Length-Adaptive GAE**: Dynamically adjusts lambda based on sequence length: lambda = 1 - 1/(alpha * length), ensuring uniform TD-error distribution
4. **Token-Level Policy Gradient Loss**: From [[DAPO]], equal weight to all tokens
5. **Clip-Higher**: From [[DAPO]], asymmetric clipping for exploration
6. **Positive Example LM Loss**: NLL loss on correct responses as auxiliary objective (weight=0.1)
7. **Group-Sampling**: From [[GRPO - DeepSeek-R1]], fewer prompts with more samples each

## Key Results

| Method | AIME24 |
|--------|--------|
| Vanilla PPO | 5 |
| DeepSeek-R1-Zero | 47 |
| DAPO | 50 |
| **VAPO** | **60** |

- Zero training crashes across multiple runs
- Smoother training curves and superior length scaling vs DAPO

## Connections

- Argues value-based > value-free if challenges are addressed (contrasts with [[VinePPO]]'s finding that value networks are inaccurate)
- Integrates techniques from [[DAPO]] (Clip-Higher, Token-Level Loss) and [[GRPO - DeepSeek-R1]] (Group-Sampling)
- Both VAPO and [[VinePPO]] address credit assignment, but VAPO uses a trained value network while VinePPO uses Monte Carlo sampling
