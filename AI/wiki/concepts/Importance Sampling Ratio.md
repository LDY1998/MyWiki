# Importance Sampling Ratio

The ratio between the current policy's probability and the old policy's probability for a given token. Central to [[PPO]] and [[GRPO]].

## Definition

```
r_t(θ) = π_θ(o_t | q, o_{<t}) / π_θ_old(o_t | q, o_{<t})
```

- r_t ≈ 1: current and old policies agree on this token
- r_t >> 1: current policy assigns much higher probability (token became more likely)
- r_t << 1: current policy assigns much lower probability (token became less likely)

## Role in Policy Optimization

The policy gradient is weighted by this ratio:

```
∇J ∝ r_t * A_t
```

This allows reusing data from the old policy (off-policy correction). But extreme ratios cause unstable updates.

## Clipping

[[PPO]]/[[GRPO]] clip the ratio to prevent instability:

```
clip(r_t, 1-ε, 1+ε)
```

Standard: ε = 0.2 (symmetric). [[DAPO]]/[[VAPO]] use **asymmetric clipping** (Clip-Higher): ε_low = 0.2, ε_high = 0.28 to preserve exploration.

## Instability Problem ([[GMPO]])

As training progresses, the importance sampling ratio range expands. Tokens with outlier ratios dominate the arithmetic mean objective, causing erratic policy updates. [[GMPO]]'s geometric mean dampens these outliers.

## Connection to KL Divergence

Large importance sampling ratios indicate the policy has diverged significantly from the old/reference policy. This is related to [[Entropy Collapse]] — as the policy becomes deterministic, some tokens see extreme ratio changes.
