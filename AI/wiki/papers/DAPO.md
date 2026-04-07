# DAPO: An Open-Source LLM Reinforcement Learning System at Scale

- **Arxiv**: [2503.14476](https://arxiv.org/abs/2503.14476)
- **Authors**: ByteDance Seed, Tsinghua AIR, HKU
- **Date**: March 2025
- **Raw**: [[DAPO_2503.14476]]

## Summary

Proposes Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO), a fully open-source RL system that achieves 50 points on AIME 2024 with Qwen2.5-32B, outperforming DeepSeek-R1-Zero-Qwen-32B (47 points) in 50% fewer training steps. Identifies and solves key issues in naive [[GRPO]]: entropy collapse, reward noise, and training instability.

## Key Techniques

1. **Clip-Higher**: Decouples clipping into epsilon_low and epsilon_high; raises epsilon_high to prevent entropy collapse and maintain exploration
2. **Dynamic Sampling**: Filters out prompts where all G responses get the same reward (zero variance), improving training efficiency
3. **Token-Level Policy Gradient Loss**: Averages loss over all tokens equally instead of per-sequence first, preventing long sequences from being underweighted
4. **Overlong Reward Shaping**: Penalizes truncated (overlong) responses with soft penalties instead of hard failure, reducing reward noise
5. **No KL penalty**: Removes KL divergence term since long-CoT models diverge significantly from the initial model by design

## Key Results

- AIME 2024: 50 points (Qwen2.5-32B), outperforming DeepSeek-R1-Zero (47)
- Achieved with 50% fewer training steps than R1-Zero
- Built on verl framework, fully open-sourced

## Connections

- Builds on [[GRPO - DeepSeek-R1]] as baseline, fixing its training issues
- [[VAPO]] adopts Clip-Higher and Token-Level Loss from DAPO, adding value-based methods on top
- [[GRESO]] further improves DAPO's Dynamic Sampling with pre-rollout filtering
- [[GMPO]] tackles the same instability via geometric mean instead of clipping
