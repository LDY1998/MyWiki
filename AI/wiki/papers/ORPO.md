# ORPO: Reference-free Monolithic Preference Optimization with Odds Ratio

- **Arxiv**: [2403.07691](https://arxiv.org/abs/2403.07691)
- **Authors**: Hong, Lee, Thorne (KAIST AI)
- **Date**: March 2024
- **Raw**: [[Online-RPO_2403.07691]]

## Summary

Proposes Odds Ratio Preference Optimization (ORPO), a reference-free alignment method that combines SFT and preference optimization in a single training phase. Unlike DPO which requires a separate reference model, ORPO adds an odds-ratio-based penalty to the standard cross-entropy loss, making it computationally efficient while maintaining alignment quality.

## Key Insight

SFT alone increases the probability of both preferred and undesirable outputs. ORPO addresses this by adding a penalty term based on the odds ratio between preferred and rejected responses, which naturally suppresses undesirable generations during SFT.

## Method

- Combines SFT cross-entropy loss with an odds-ratio preference penalty
- No reference model needed (unlike DPO)
- Single training phase (unlike RLHF's multi-stage pipeline)
- Odds ratio provides better gradient properties than simple probability ratio

## Key Results

- Mistral-ORPO-7B competitive with models trained via SFT+DPO
- Strong performance on AlpacaEval, MT-Bench
- Computationally more efficient than DPO (no reference model forward passes)

## Connections

- Alternative to [[Step-DPO]]'s approach — ORPO is reference-free while Step-DPO focuses on step-level granularity
- Simpler than [[GRPO - DeepSeek-R1]] pipeline (no online RL needed) but less powerful for reasoning
- Part of the broader DPO-variant family alongside [[Step-DPO]]
