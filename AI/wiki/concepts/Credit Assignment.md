# Credit Assignment

The fundamental problem of determining which actions (tokens/steps) in a sequence contributed to the final outcome. Central challenge in RL for LLM reasoning.

## Why It's Hard

In reasoning tasks, models generate long chains of thought before receiving a single binary reward (correct/incorrect). Most steps are routine — only a few are critical. Accurately attributing credit to the right steps is essential for efficient learning.

## Approaches

### Value Network ([[PPO]], [[VAPO]])
Train a critic model to estimate expected future reward at each token. Compute advantage via [[GAE]].
- **Pro**: Fine-grained, per-token credit
- **Con**: Value networks are inaccurate for reasoning tasks ([[VinePPO]] shows they barely beat random baselines)
- **Fix**: [[VAPO]] makes value networks work via Value-Pretraining, Decoupled-GAE, Length-Adaptive GAE

### Monte Carlo Estimation ([[VinePPO]])
Fork K independent continuations from each intermediate state, average their returns.
- **Pro**: Unbiased estimates, no value network needed
- **Con**: Computationally expensive (K extra rollouts per state)
- **Result**: 9x fewer gradient steps to reach PPO's peak

### Group-Relative ([[GRPO]])
Sample G responses per prompt, normalize rewards across the group.
- **Pro**: Simple, no value network
- **Con**: All tokens in a response share the same advantage (coarse)

### Step-Level ([[Step-DPO]])
Identify the first erroneous step, optimize at that granularity.
- **Pro**: Directly targets the error location
- **Con**: Requires error localization pipeline, offline only

### Tree-Based ([[TEMPO]])
Build a prefix tree from grouped responses; apply TD credit only at branching points where responses diverge.
- **Pro**: Fine-grained credit at zero extra cost over GRPO — exploits structure already in group sampling
- **Con**: Credit resolution limited by tree branching density

### Eligibility Traces ([[GRPO-lambda]])
Adapt classic λ-returns to GRPO using token log-probabilities as trace weights.
- **Pro**: Finer than GRPO with minimal added complexity, no value network
- **Con**: Approximation quality depends on log-probability as a credit proxy

### Game-Theoretic ([[SCAR]])
Use Shapley values to decompose sequence reward into per-token contributions based on marginal contributions.
- **Pro**: Axiomatic fairness guarantees, provably preserves optimal policy
- **Con**: Shapley computation is expensive (requires approximation at scale)

### Segment-Level ([[SPO]])
Partition responses into segments (e.g., reasoning steps) and estimate advantages per segment.
- **Pro**: Better balance than token-level (noisy) or response-level (coarse)
- **Con**: Segmentation quality matters; fixed chunking may miss natural boundaries

### Macro Actions ([[MA-RLHF]])
Group tokens into multi-token macro actions to reduce temporal distance to rewards.
- **Pro**: Simple, no overhead, scales well (2B–27B)
- **Con**: Fixed chunking, no content awareness

## The Spectrum

```
Coarse ←——————————————————————————————————————————————→ Fine-grained
GRPO    GRPO-λ    MA-RLHF/SPO    TEMPO      Step-DPO    VinePPO    SCAR    VAPO/PPO+GAE
(response) (traces)  (segment)  (branch-pt)   (step)    (step-MC)  (Shapley)  (token)
```

Finer credit assignment generally yields better performance but at higher computational cost. Recent work (especially [[TEMPO]]) challenges this tradeoff by extracting fine-grained credit from structure already present in group sampling.
