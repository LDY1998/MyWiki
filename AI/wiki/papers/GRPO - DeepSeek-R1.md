# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

- **Arxiv**: [2501.12948](https://arxiv.org/abs/2501.12948)
- **Authors**: DeepSeek-AI
- **Date**: January 2025
- **Raw**: [[GRPO_DeepSeek-R1_2501.12948]]

## Summary

Introduces DeepSeek-R1-Zero and DeepSeek-R1, the first open models to demonstrate that reasoning capabilities can be incentivized purely through RL without SFT. DeepSeek-R1-Zero is trained via large-scale RL (using [[GRPO]]) directly on the base model, naturally emerging with self-verification, reflection, and long chain-of-thought behaviors. DeepSeek-R1 adds cold-start data and multi-stage training to fix readability and language-mixing issues, achieving performance comparable to OpenAI o1-1217.

## Key Contributions

- **GRPO algorithm**: Group Relative Policy Optimization eliminates the value network by estimating advantage through group-level reward normalization across G sampled responses
- **Pure RL reasoning**: First open demonstration that reasoning emerges from RL alone (no SFT), with "aha moment" self-evolution behaviors
- **Multi-stage pipeline**: Cold start data -> reasoning RL -> rejection sampling + SFT -> all-scenario RL
- **Distillation**: 6 dense models (1.5B-70B) distilled from R1, with the 32B distilled model outperforming QwQ-32B-Preview

## Key Results

- AIME 2024: 71.0% pass@1 (R1-Zero), 79.8% (R1), 86.7% with majority voting
- Comparable to OpenAI o1-1217 on math, code, and science reasoning
- Distilled 14B model outperforms open-source QwQ-32B-Preview

## Unsuccessful Attempts

- **[[Process Reward Model]]**: DeepSeek tried step-level PRMs but found step boundaries messy, labels expensive, and the approach prone to reward hacking. Still listed as unsuccessful in v2 (Jan 2026). See [[Process Reward Model]] for how the field has since addressed each objection.

## Connections

- Foundational algorithm for [[DAPO]], [[VAPO]], [[GRESO]], [[GMPO]], [[Pass@k Training]]
- [[DAPO]] identifies and fixes entropy collapse issues in GRPO
- [[VAPO]] argues value-based methods can outperform GRPO with proper training
- [[VinePPO]] addresses the same credit assignment problem from a Monte Carlo perspective
