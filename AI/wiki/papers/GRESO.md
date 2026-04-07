# GRESO: Efficient RL for LLM Reasoning via Selective Rollouts

- **Arxiv**: [2506.02177](https://arxiv.org/abs/2506.02177)
- **Authors**: Zheng et al. (CMU, LLNL, UIUC, Meta AI)
- **Date**: June 2025
- **Raw**: [[GRESO_2506.02177]]

## Summary

Observes that a large portion of GRPO training rollouts are wasted on uninformative prompts (where all sampled responses get the same reward). Proposes GRESO (GRPO with Efficient Selective Rollout), a pre-rollout filtering algorithm that predicts and skips uninformative prompts using reward training dynamics. Achieves up to 2.4x rollout speedup and 2.0x total training speedup without accuracy loss.

## Key Insight

Prompts that produce zero-variance rewards (all correct or all wrong) contribute no gradient signal. These prompts show strong temporal consistency — if uninformative in one epoch, likely uninformative in the next. This enables predictive filtering before expensive rollout.

## Method

1. Track reward dynamics per prompt across training epochs
2. Use temporal correlation to predict which prompts will be uninformative
3. Probabilistic pre-rollout filtering: skip predicted zero-variance prompts
4. Lightweight and online — adds minimal overhead

## Key Results

- Up to 2.4x rollout speedup, 2.0x total training speedup
- No accuracy degradation on MATH500, AMC, Gaokao, Minerva, Olympiad Bench
- Tested on Qwen2.5-Math-1.5B/7B and DeepSeek-R1-Distill-Qwen-1.5B

## Connections

- Extends [[DAPO]]'s Dynamic Sampling (which filters after rollout) to pre-rollout filtering
- Built on the [[GRPO - DeepSeek-R1]] training loop
- Orthogonal to algorithm improvements like [[VAPO]], [[GMPO]] — can be combined with them
