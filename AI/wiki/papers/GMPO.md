# GMPO: Geometric-Mean Policy Optimization

- **Arxiv**: [2507.20673](https://arxiv.org/abs/2507.20673)
- **Authors**: Zhao et al. (UCAS, CUHK, HKUST, Microsoft Research)
- **Date**: July 2025
- **Raw**: [[GMPO_2507.20673]]

## Summary

Identifies that [[GRPO - DeepSeek-R1]]'s arithmetic mean of token-level rewards is sensitive to outlier importance sampling ratios, causing unstable policy updates. Proposes GMPO, which maximizes the geometric mean instead, inherently reducing sensitivity to outliers. Outperforms GRPO by 4.1% average on math benchmarks with 7B models.

## Key Insight

The arithmetic mean used in GRPO amplifies extreme importance sampling ratios. As training progresses, these ratios expand, destabilizing updates. The geometric mean naturally dampens outliers, maintaining stable ratio ranges while allowing larger clipping bounds for better exploration.

## Method

- Replace arithmetic mean with geometric mean in the GRPO objective for token-level rewards
- Allows larger asymmetric clipping range (epsilon_1, epsilon_2) for greater exploration
- Theoretically proven: narrower value range, more robust gradients
- Maintains smaller KL divergence and higher entropy than GRPO

## Key Results

- GMPO-7B: +4.1% average over GRPO on AIME24, AMC, MATH500, Minerva, OlympiadBench
- +1.4% on Geometry3K multimodal reasoning
- More stable importance sampling ratios throughout training
- Higher token entropy = better exploration

## Connections

- Direct improvement over [[GRPO - DeepSeek-R1]]'s objective function
- Tackles the same entropy collapse problem as [[DAPO]]'s Clip-Higher, but from a different angle (objective function vs. clipping)
- Orthogonal to [[VAPO]]'s value-based improvements — could potentially be combined
- Complementary to [[GRESO]]'s efficiency improvements
