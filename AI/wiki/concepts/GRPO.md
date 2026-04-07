# Group Relative Policy Optimization (GRPO)

A value-free RL algorithm introduced in [[GRPO - DeepSeek-R1]] that eliminates the need for a [[Value Network]] by estimating advantages through group-level reward normalization.

## How It Works

For a given prompt q, GRPO samples a group of G responses {o_i}. Each response gets a reward R_i. The advantage for response i is:

```
A_i = (R_i - mean({R_i})) / std({R_i})
```

This advantage is assigned uniformly to all tokens in the response. The policy is updated via a clipped objective (like [[PPO]]):

```
J_GRPO = E[ min(r_t * A_i, clip(r_t, 1-ε, 1+ε) * A_i) - β * D_KL(π_θ || π_ref) ]
```

where r_t = π_θ(o_t) / π_θ_old(o_t) is the [[Importance Sampling Ratio]].

## Why It Matters

- Removes the [[Value Network]] entirely, halving GPU memory requirements
- Simple to implement and scale
- Foundation for most subsequent LLM RL algorithms

## Known Issues

- **[[Entropy Collapse]]**: The symmetric clipping range limits exploration, causing the policy to become deterministic too early ([[DAPO]], [[GMPO]])
- **Coarse credit assignment**: All tokens in a response share the same advantage, unlike [[PPO]] with [[GAE]] which assigns per-token advantages ([[VinePPO]], [[VAPO]])
- **Wasted rollouts**: Many prompts produce zero-variance rewards (all correct or all wrong), contributing no gradient signal ([[GRESO]])
- **Sample-level loss**: GRPO averages loss per-sequence first, then across sequences — this underweights long responses ([[DAPO]]'s Token-Level Loss fixes this)

## Variants

| Variant | Key Change |
|---------|-----------|
| [[DAPO]] | Clip-Higher + Dynamic Sampling + Token-Level Loss |
| [[GMPO]] | Geometric mean instead of arithmetic mean |
| [[Pass@k Training]] | Pass@k reward instead of Pass@1 |
| [[GRESO]] | Pre-rollout prompt filtering |
| [[StarPO - RAGEN]] | Extension to multi-turn agent trajectories |

## References

- Shao et al., "DeepSeekMath" (2024) — original GRPO formulation
- [[GRPO - DeepSeek-R1]] — large-scale application to reasoning
