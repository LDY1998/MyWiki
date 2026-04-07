# RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning

- **Arxiv**: [2504.20073](https://arxiv.org/abs/2504.20073)
- **Authors**: Wang et al. (Stanford, Microsoft, NYU, etc.)
- **Date**: April 2025
- **Raw**: [[StarPO_2504.20073]]

## Summary

Introduces StarPO (Star Policy Optimization), a framework for multi-turn agent RL training, and RAGEN, a system for studying self-evolution in LLM agents. Unlike single-turn reasoning (math), multi-turn agent settings involve interleaved reasoning and environment interaction, introducing new instability patterns. Proposes StarPO-S with instance filtering and gradient shaping to stabilize training.

## Key Contributions

- **StarPO framework**: Generalizes GRPO to multi-turn agent trajectories with reasoning-guided trajectory-level optimization
- **New instability pattern**: Multi-turn agent RL suffers from "echo trap" — models get stuck in repetitive action loops
- **StarPO-S**: Stabilized variant with instance filtering (remove high-variance trajectories) and gradient shaping
- **RAGEN system**: Built on verl for systematic study of agent RL training

## Key Findings

- Reasoning (think tokens) improves generalization but fades without fine-grained rewards in multi-turn settings
- Generating useful trajectories (diverse, informative) is critical for RL training
- PPO fails in certain multi-turn environments (Frozen Lake) where GRPO-based methods succeed

## Connections

- Extends [[GRPO - DeepSeek-R1]] from single-turn reasoning to multi-turn agent interactions
- Addresses similar training instability issues as [[DAPO]] but in the agent setting
- The trajectory filtering connects to [[GRESO]]'s prompt filtering idea
