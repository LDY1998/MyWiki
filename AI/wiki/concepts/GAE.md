# Generalized Advantage Estimation (GAE)

A technique for computing per-token advantage estimates in [[PPO]], trading off bias and variance via a parameter λ.

## Formula

```
A_t = Σ_{l=0}^{T-t-1} (γλ)^l * δ_{t+l}
```

where δ_t = R(s_t, a_t) + γ * V(s_{t+1}) - V(s_t) is the TD error, and V is the [[Value Network]].

## Bias-Variance Tradeoff

- **λ = 0**: A_t = δ_t (low variance, high bias — relies entirely on value network)
- **λ = 1**: A_t = Monte Carlo return - V(s_t) (high variance, low bias — uses actual returns)
- **λ = 0.95**: Standard setting, balances both

## Problem with Long CoT

For long reasoning chains (T >> 1), GAE with fixed λ < 1 causes the reward signal to decay exponentially:

```
Reward propagation to token t ≈ λ^(T-t) * R(s_T, <EOS>)
```

This means early tokens receive near-zero advantage signal, making optimization ineffective for long sequences.

## Solutions in the Literature

| Paper | Approach |
|-------|----------|
| [[VAPO]] | **Length-Adaptive GAE**: λ = 1 - 1/(α * length), adapts per-sequence |
| [[VAPO]] | **Decoupled-GAE**: λ=1.0 for value updates, adaptive λ for policy |
| [[VinePPO]] | Bypasses GAE entirely with Monte Carlo value estimates |
| [[GRPO]] | Eliminates GAE — uses group-level reward normalization instead |

## References

- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2016)
