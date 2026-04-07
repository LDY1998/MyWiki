# Agentic LLM RL — Algorithms Wiki

This wiki covers reinforcement learning algorithms for LLM reasoning, based on papers from the [Awesome-AgenticLLM-RL-Papers](https://github.com/xhyumiracle/Awesome-AgenticLLM-RL-Papers) collection (Sec 2.7: Algorithms).

## Papers

### Foundational
- [[GRPO - DeepSeek-R1]] — GRPO algorithm; pure RL reasoning without SFT (Jan 2025)

### Value-Free Improvements (GRPO variants)
- [[DAPO]] — Clip-Higher, Dynamic Sampling, Token-Level Loss; fixes entropy collapse (Mar 2025)
- [[GMPO]] — Geometric mean objective; stabilizes importance sampling ratios (Jul 2025)
- [[Pass@k Training]] — Pass@k as reward for exploration-exploitation balance (Aug 2025)
- [[GRESO]] — Pre-rollout filtering; 2x training speedup by skipping uninformative prompts (Jun 2025)
- [[TEMPO]] — Tree-based credit at branching points; zero extra cost over GRPO (Sep 2025, ICLR 2026)
- [[GRPO-lambda]] — Eligibility traces for GRPO via token log-probabilities (Oct 2025)

### Value-Based Methods
- [[VAPO]] — Value-based PPO with Length-Adaptive GAE; outperforms DAPO by 10+ points (Apr 2025)
- [[VinePPO]] — Monte Carlo value estimates; shows value networks are inaccurate (Oct 2024)
- [[SCAR]] — Shapley value decomposition of sequence rewards into per-token credit (May 2025)
- [[SPO]] — Segment-level advantage estimation between token and response granularity (May 2025, NeurIPS 2025)
- [[MA-RLHF]] — Macro actions reduce temporal credit distance in RLHF (Oct 2024, ICLR 2025)

### Preference Optimization
- [[Step-DPO]] — Step-level DPO for long-chain reasoning; 10K pairs, 3% gain on MATH (Jun 2024)
- [[ORPO]] — Reference-free preference optimization via odds ratio (Mar 2024)

### Multi-Turn / Agent RL
- [[StarPO - RAGEN]] — GRPO extended to multi-turn agent interactions (Apr 2025)

## Key Themes

### Credit Assignment
How to determine which tokens/steps contributed to a correct answer?
- **Value networks** ([[VAPO]]): Train a critic, but requires careful techniques to avoid bias
- **Monte Carlo** ([[VinePPO]]): Sample continuations from intermediate states, unbiased but compute-heavy
- **Group-relative** ([[GRPO - DeepSeek-R1]]): Use group reward normalization, simple but coarse
- **Step-level** ([[Step-DPO]]): Localize first error, optimize at step granularity
- **Tree-based** ([[TEMPO]]): Credit at prefix-tree branching points, zero extra cost
- **Eligibility traces** ([[GRPO-lambda]]): Classic λ-returns adapted to GRPO
- **Game-theoretic** ([[SCAR]]): Shapley values for principled per-token reward decomposition
- **Segment-level** ([[SPO]]): Middle ground between token and response granularity
- **Macro actions** ([[MA-RLHF]]): Coarsen action space to reduce temporal distance

### Entropy Collapse / Exploration
Preventing the policy from becoming too deterministic too early:
- **Clip-Higher** ([[DAPO]]): Asymmetric clipping to preserve exploration
- **Geometric mean** ([[GMPO]]): Dampen outlier importance ratios
- **Pass@k reward** ([[Pass@k Training]]): Reward diversity of correct reasoning paths
- **No KL penalty** ([[DAPO]]): Let model diverge freely from initial policy

### Training Efficiency
- **Pre-rollout filtering** ([[GRESO]]): Skip prompts predicted to be uninformative
- **Dynamic Sampling** ([[DAPO]]): Filter zero-variance prompts after rollout
- **Monte Carlo** ([[VinePPO]]): Fewer gradient steps despite slower per-iteration

## AIME 2024 Leaderboard (Qwen2.5-32B base)

| Method | Score | Type |
|--------|-------|------|
| Vanilla PPO | 5 | Value-based |
| GRPO (R1-Zero) | 47 | Value-free |
| DAPO | 50 | Value-free |
| **VAPO** | **60** | Value-based |

## Concepts

### Algorithms
- [[GRPO]] — Group Relative Policy Optimization, the foundational value-free algorithm
- [[PPO]] — Proximal Policy Optimization, the value-based baseline
- [[DPO]] — Direct Preference Optimization, offline alignment

### Core Mechanisms
- [[GAE]] — Generalized Advantage Estimation, bias-variance tradeoff via λ
- [[Credit Assignment]] — Which tokens/steps caused the outcome?
- [[Process Reward Model]] — Step-level rewards; DeepSeek's failed attempt and the field's repair work
- [[Value Network]] — Critic model for estimating expected reward
- [[Importance Sampling Ratio]] — Policy ratio that drives updates and instability

### Challenges
- [[Entropy Collapse]] — Policy becomes deterministic, killing exploration
- [[Sparse Rewards]] — Single binary reward after thousands of tokens
- [[Chain-of-Thought]] — Long CoT reasoning and why it breaks standard RL

## Source Data

Raw paper markdown files are in `../raw/`. Each paper summary in `papers/` links back to its raw source.
