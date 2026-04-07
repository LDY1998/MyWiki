# Direct Preference Optimization (DPO)

An offline alignment method that directly optimizes on preference pairs without training a separate reward model or doing online RL. Simpler than [[PPO]]-based RLHF but less effective for reasoning.

## Core Idea

Given preference pairs (y_win, y_lose) for a prompt x, DPO optimizes:

```
L_DPO = -log σ(β * [log π_θ(y_win|x)/π_ref(y_win|x) - log π_θ(y_lose|x)/π_ref(y_lose|x)])
```

This implicitly defines a reward model and optimizes the policy in closed form.

## Limitation for Reasoning

DPO compares complete responses holistically. For long-chain reasoning, this fails because:
- Most of a wrong answer may be correct up to one error step
- DPO can't localize where the error occurred
- The model struggles to distinguish preferred from undesirable outputs (see [[Step-DPO]] Fig. 2)

## Variants for Reasoning

| Method | Key Improvement |
|--------|----------------|
| [[Step-DPO]] | Optimize at individual reasoning step level |
| [[ORPO]] | Remove reference model dependency entirely |

## DPO vs Online RL

| Aspect | DPO | Online RL ([[GRPO]], [[PPO]]) |
|--------|-----|-----|
| Data | Offline preference pairs | Online rollouts |
| Exploration | None (fixed dataset) | Active (generates new responses) |
| Reward model | Implicit | Explicit (rule-based or learned) |
| Reasoning gains | Minimal | Significant |
| Simplicity | Higher | Lower |

[[GRPO - DeepSeek-R1]] and subsequent work show that online RL substantially outperforms DPO for reasoning tasks, likely because exploration is critical for discovering complex reasoning patterns.

## References

- Rafailov et al., "Direct Preference Optimization" (2023)
