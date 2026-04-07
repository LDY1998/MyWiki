# Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning of LLMs

- **Arxiv**: [2406.18629](https://arxiv.org/abs/2406.18629)
- **Authors**: Lai et al. (CUHK, HIT-SZ, SmartMore)
- **Date**: June 2024
- **Raw**: [[Step-DPO_2406.18629]]

## Summary

Shows that vanilla DPO provides minimal benefits for long-chain mathematical reasoning because it can't localize errors in multi-step answers. Proposes Step-DPO, which treats individual reasoning steps as the optimization unit. With only 10K step-wise preference pairs and <500 training steps, achieves ~3% accuracy gain on MATH for 70B+ models. Applied to Qwen2-72B-Instruct, achieves 70.8% MATH and 94.0% GSM8K, surpassing GPT-4-1106.

## Key Insight

In long-chain reasoning, most errors appear midway through the solution, not at the beginning. Vanilla DPO compares complete answers, making it hard to pinpoint where the error occurred. Step-DPO identifies the first erroneous step and optimizes at that granularity.

## Method

1. **Error Collection**: Generate responses, identify those with wrong final answers
2. **Step Localization**: Find the first erroneous reasoning step k in the wrong answer
3. **Rectification**: From the correct prefix s_1...s_{k-1}, sample new continuations; keep those reaching correct answers as s_win
4. **In-distribution data**: Self-generated corrections outperform human/GPT-4 corrections (out-of-distribution data suffers from gradient decay)

## Key Results

- Qwen2-72B-Instruct + Step-DPO: 70.8% MATH, 94.0% GSM8K
- Surpasses GPT-4-1106, Claude-3-Opus, Gemini-1.5-Pro
- Only 10K preference pairs needed
- Larger models benefit more from Step-DPO

## Connections

- Addresses fine-grained credit assignment like [[VinePPO]], but through preference optimization rather than RL
- Complementary to [[GRPO - DeepSeek-R1]] — Step-DPO is offline (DPO-based) while GRPO is online RL
- The step-level supervision idea connects to process reward models discussed in [[GRPO - DeepSeek-R1]]'s unsuccessful attempts section
