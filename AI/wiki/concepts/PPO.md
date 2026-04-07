# Proximal Policy Optimization (PPO)

The dominant RL algorithm for LLM finetuning. Uses a clipped surrogate objective to prevent large, destabilizing policy updates.

## Core Objective

```
L_CLIP(θ) = E[ min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t) ]
```

where:
- r_t = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) is the [[Importance Sampling Ratio]]
- A_t is the advantage estimated via [[GAE]] using a [[Value Network]]
- ε is the clipping range (typically 0.2)

The clipping ensures the policy doesn't change too much in a single update.

## In the LLM Context

Language generation is modeled as a token-level MDP:
- **State**: prompt + tokens generated so far
- **Action**: next token from vocabulary
- **Reward**: sparse, only at the final token (e.g., correctness)
- **Dynamics**: deterministic (appending a token)

PPO requires a separate **value network** (critic) of similar size to the policy, roughly doubling GPU memory. This motivated value-free alternatives like [[GRPO]].

## PPO vs GRPO

| Aspect | PPO | [[GRPO]] |
|--------|-----|------|
| Advantage | Per-token via [[GAE]] + [[Value Network]] | Per-response via group reward normalization |
| Memory | 2x (policy + value network) | 1x (policy only) |
| Credit assignment | Fine-grained (token-level) | Coarse (all tokens share same advantage) |
| Complexity | Higher | Lower |

## Improvements for LLM RL

- [[VAPO]]: Makes PPO work for long-CoT by fixing value model bias, adding Length-Adaptive GAE
- [[VinePPO]]: Replaces value network with Monte Carlo estimates
- [[DAPO]]: Inherits PPO's clipping idea but decouples it (Clip-Higher)

## References

- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
