# Value Network (Critic)

A separate neural network trained alongside the policy to estimate the expected cumulative reward from any intermediate state. Used in [[PPO]] and [[VAPO]] for computing advantages via [[GAE]].

## How It Works

Given a state s_t (prompt + tokens so far), the value network predicts:

```
V(s_t) = E[total reward | starting from s_t, following policy π]
```

This prediction is used to compute the TD error and advantage for policy updates.

## Problems in LLM Reasoning

[[VinePPO]] showed that value networks have fundamental issues:

1. **Low accuracy**: Only ~65% correct predictions (where "correct" = within 0.05 of ground truth)
2. **Poor ranking**: Near-chance performance at identifying which next step leads to higher value
3. **Error accumulation**: Prediction errors increase as reasoning progresses, because later states are more diverse and harder to generalize to
4. **Memory cost**: Requires a full model-sized network + optimizer states, roughly doubling GPU memory (up to 112GB for 7B LLMs)

## Making Value Networks Work ([[VAPO]])

[[VAPO]] proposes several fixes:

| Technique | Problem Solved |
|-----------|---------------|
| Value-Pretraining | Initialization bias from reward model |
| Decoupled-GAE | Different λ for value vs policy updates |
| Length-Adaptive GAE | Reward signal decay in long sequences |

With these, [[VAPO]] achieves SOTA (60 on AIME), arguing value-based methods have higher ceiling than value-free.

## The Debate: Value Network vs No Value Network

| Approach | Proponents | Argument |
|----------|-----------|----------|
| Keep it | [[VAPO]] | Finer credit assignment, higher ceiling |
| Replace with MC | [[VinePPO]] | Value networks are inaccurate, MC is unbiased |
| Remove entirely | [[GRPO]], [[DAPO]] | Simpler, good enough with group rewards |
