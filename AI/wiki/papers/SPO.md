# SPO: Segment Policy Optimization

- **Arxiv**: [2505.23564](https://arxiv.org/abs/2505.23564)
- **Date**: May 2025
- **Venue**: NeurIPS 2025

## Summary

Proposes Segment Policy Optimization (SPO), which partitions responses into segments and estimates advantages at the segment level — an intermediate granularity between token-level (PPO) and response-level (GRPO). Achieves a better balance between credit precision and estimation accuracy.

## Key Techniques

1. **Flexible Segment Partition**: Divides responses into meaningful segments (e.g., reasoning steps) rather than fixed-length chunks
2. **Segment Advantage Estimation**: Computes advantages per segment rather than per token or per response
3. **Probability-Mask Strategy**: Novel masking approach for policy optimization using segment-level advantages

## Key Insight

Token-level methods (PPO) have fine-grained credit but suffer from inaccurate critics. Response-level methods (GRPO) have accurate but coarse credit. Segments hit the sweet spot — large enough for reliable estimation, small enough for meaningful credit differentiation.

## Key Results

- Short CoT: 6-12 percentage point accuracy improvement over PPO and GRPO on GSM8K
- Long CoT: 7-11 percentage point accuracy improvement over GRPO on MATH500 (2K and 4K context)
- Code: [github.com/AIFrameResearch/SPO](https://github.com/AIFrameResearch/SPO)

## Connections

- Directly addresses the coarse [[Credit Assignment]] of [[GRPO]]
- Conceptually similar to [[MA-RLHF]]'s macro actions but with flexible, content-aware segmentation
- Complements [[TEMPO]]'s tree-based approach — SPO uses explicit segmentation while TEMPO uses implicit branching structure
