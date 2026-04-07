# Pass@k Training for Adaptively Balancing Exploration and Exploitation

- **Arxiv**: [2508.10751](https://arxiv.org/abs/2508.10751)
- **Authors**: Chen et al. (Renmin University, ByteDance Seed)
- **Date**: August 2025
- **Raw**: [[Pass-at-k_2508.10751]]

## Summary

Standard RLVR uses Pass@1 as reward, which causes policies to converge to conservative local optima. Proposes using Pass@k as the reward instead — a response is rewarded if any k-subset contains a correct answer. This shifts the optimization toward exploration, preventing early convergence. Derives an efficient analytical solution and shows exploration and exploitation are not inherently conflicting.

## Key Insight

Pass@1 reward creates a harsh binary signal that punishes exploration. Pass@k is more forgiving — it rewards the model as long as it can sometimes produce a correct answer, encouraging diverse reasoning paths. The analytical derivation shows Pass@k Training essentially designs a different advantage function.

## Method

1. Use Pass@k (probability that at least one of k samples is correct) as reward instead of Pass@1
2. Efficient implementation via bootstrap sampling
3. Analytical derivation enables efficient training without actually sampling k responses
4. Adaptive training based on policy entropy to balance exploration/exploitation phases
5. Can combine Pass@1 and Pass@k training phases

## Key Results

- Boosts exploration ability, leading to continuous improvement in subsequent training
- Surpasses native RLVR on Enigmata, math tasks, and maze tasks
- Works with Qwen2.5-7B-Ins and other models
- Opens direction of "advantage function design" for RLVR

## Connections

- Addresses the exploration collapse also tackled by [[DAPO]] (Clip-Higher) and [[GMPO]] (geometric mean)
- Built on [[GRPO - DeepSeek-R1]] training framework
- Complementary to [[GRESO]]'s efficiency — Pass@k changes what to optimize, GRESO changes which prompts to skip
- Different approach to exploration than [[VAPO]]'s Positive Example LM Loss
