# GRPO-λ: Credit Assignment improves LLM Reasoning

- **Arxiv**: [2510.00194](https://arxiv.org/abs/2510.00194)
- **Date**: October 2025

## Summary

Introduces GRPO-λ, an extension of [[GRPO]] that brings eligibility traces (the λ-return from classic RL) into the value-free framework. Uses token-level log-probabilities as trace weights applied after each sequence generation, enabling finer credit assignment without requiring a value network.

## Key Techniques

1. **λ-Return Reformulation**: Approximates learning from λ-returns using eligibility traces computed from token-level log-probabilities
2. **Trace Weighting Variations**: Introduces multiple weighting schemes for the λ-return and their applications to the eligibility trace
3. **Post-Generation Application**: Traces are applied after complete sequence generation, maintaining GRPO's simplicity

## Key Insight

Classic RL's eligibility traces can be adapted to the GRPO setting by using the model's own token log-probabilities as a proxy for temporal credit, bridging the gap between response-level and token-level advantage estimation.

## Key Results

- 30-40% improved performance during RL training on LLaMA-3.1 and Qwen-2
- ~3 point improvement over GRPO on average across AIME24, Math500, OlympiadMath, MinervaMath, AMC
- 4.5 point improvement on the 7B model

## Connections

- Direct extension of [[GRPO]] with minimal additional complexity
- Addresses the same coarse [[Credit Assignment]] as [[TEMPO]], [[SPO]], and [[SCAR]], but through classic RL machinery (eligibility traces) rather than tree structure or game theory
- Simpler than [[VAPO]]'s approach of training a full value network
