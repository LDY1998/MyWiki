# MA-RLHF: Reinforcement Learning from Human Feedback with Macro Actions

- **Arxiv**: [2410.02743](https://arxiv.org/abs/2410.02743)
- **Date**: October 2024
- **Venue**: ICLR 2025

## Summary

Introduces macro actions (multi-token chunks) into RLHF to reduce the temporal distance between actions and rewards, facilitating faster and more accurate credit assignment. Achieves parity with vanilla RLHF 1.7-2x faster without additional computational overhead.

## Key Techniques

1. **Macro Action Formulation**: Groups consecutive tokens into macro actions, reducing the effective sequence length and temporal credit assignment horizon
2. **No Overhead**: The macro action framework integrates into standard RLHF without requiring extra models, rollouts, or significant compute

## Key Insight

By coarsening the action space from individual tokens to multi-token chunks, the temporal distance between an action and its reward shrinks, making credit assignment easier — a classic RL trick applied to the RLHF setting.

## Key Results

- Reaches reward parity with vanilla RLHF 1.7-2x faster
- Robust scalability from 2B to 27B parameters
- No additional computational overhead vs standard RLHF

## Connections

- Addresses [[Credit Assignment]] through action-space coarsening rather than reward decomposition ([[SCAR]]) or advantage refinement ([[TEMPO]], [[GRPO-lambda]])
- Conceptually related to [[SPO]]'s segment-level approach but operates on fixed multi-token chunks rather than content-aware segments
- Works within standard RLHF/PPO rather than the [[GRPO]] family
