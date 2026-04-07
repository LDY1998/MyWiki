# Process Reward Model (PRM)

A reward model that provides feedback at each **reasoning step** rather than only at the end of a complete response. Contrasts with Outcome Reward Models (ORMs) that give a single scalar for the final answer.

## Why It's Appealing

- Denser reward signal than [[Sparse Rewards]] — credit is assigned per step, not per response
- Natural fit for [[Chain-of-Thought]] reasoning where intermediate steps matter
- Enables test-time search, reranking, and debugging by scoring partial solutions

## DeepSeek's Objections ([[GRPO - DeepSeek-R1]])

DeepSeek listed step-level PRMs as an **unsuccessful attempt** in R1 training. Their three core objections:

### 1. Step boundaries are messy
For free-form reasoning, what counts as a "step" is slippery. Assigning correct local rewards becomes noisy and gameable.

### 2. Step labels are expensive and unreliable
Dense per-step human labels don't scale. Automatic labeling (e.g., Monte Carlo estimation) often generalizes poorly.

### 3. PRMs invite reward hacking during RL
The policy learns to exploit locally high-reward steps, especially when the PRM is imperfect. Sum-form credit assignment makes this worse.

As of the R1 v2 revision (January 2026) and the R1-0528 release, DeepSeek still does not report successfully adopting step-level PRMs. Their stack continues to lean on outcome-level and verifiable rewards.

## The Repair Work (2025–2026)

The field largely accepted DeepSeek's pain points but proposed targeted fixes:

### Fixing step boundaries
- **ThinkPRM**: Generative verifier that "thinks" through step validity, tolerating fuzzier boundaries. Needs far fewer process labels than classic discriminative PRMs.
- **FunPRM**: Function-as-step supervision with final-solution signals to purify noisy partial-step rewards.

### Reducing label dependence
- **PRIME**: Learns implicit process rewards online from rollouts + outcome labels, avoiding explicit step labels entirely.
- **Qwen PRM Lessons** ("The Lessons of Developing Process Reward Models in Mathematical Reasoning"): Shows Monte Carlo label synthesis is weaker than expected; proposes consensus filtering and better evaluation.

### Preventing reward hacking
- **PURE**: Argues hacking partly comes from sum-form [[Credit Assignment]]; proposes min-form credit assignment to stabilize PRM-based RL.
- **SCOPE**: Uses PRMs surgically — identifies the first bad step in failed rollouts and salvages partially correct trajectories, rather than using PRM as the whole reward.
- **SGPO**: Step-wise judge finds the first incorrect step and guides optimization from there, keeping outcome verification in the loop.

## Current Consensus

1. **Naive step-level PRM training is brittle** — DeepSeek's warnings hold up
2. **PRMs can still help** for reranking, search, failure recovery, and constrained domains
3. The trend is **hybridization**: implicit rewards, generative verifiers, better credit assignment, and using outcome verification to keep the PRM honest

```
Pure ORM ←————————————————————————→ Pure PRM
DeepSeek-R1    PRIME/SCOPE    ThinkPRM    Classic PRM
(outcome only)  (hybrid)     (generative)  (dense step labels)
```

## Connections

- Directly related to [[Credit Assignment]] — PRMs are one approach to finer-grained credit
- [[Step-DPO]] uses step-level supervision in the preference optimization setting
- [[VinePPO]] achieves step-level credit via Monte Carlo sampling instead of a learned PRM
- [[TEMPO]] gets fine-grained credit from tree structure, sidestepping PRMs entirely
