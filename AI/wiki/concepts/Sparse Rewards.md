# Sparse Rewards

In LLM reasoning RL, rewards are typically binary (correct/incorrect) and only assigned at the final token. All intermediate tokens receive zero reward. This creates a fundamental challenge for learning.

## The Problem

A model generates hundreds or thousands of tokens in a chain-of-thought, but only receives a single bit of feedback at the end. This makes it hard to determine:
- Which tokens were helpful vs harmful
- Where reasoning went wrong
- How to improve specific steps

Combined with long CoT, sparsity is amplified — the model must explore more to find correct solutions, but receives feedback less frequently per token.

## Solutions

### Rule-Based Rewards ([[DAPO]], [[GRPO - DeepSeek-R1]])
Use verifiable tasks (math, code) where correctness can be checked automatically. Binary reward: +1 for correct, -1 for incorrect. Simple but extremely sparse.

### Overlong Reward Shaping ([[DAPO]])
Instead of hard failure for truncated responses, apply a soft penalty. This gives gradient signal even for responses that ran out of tokens.

### Group Sampling ([[GRPO]], [[VAPO]])
Sample multiple responses per prompt to create contrastive signals. Even with sparse rewards, having both correct and incorrect responses for the same prompt provides learning signal.

### Positive Example LM Loss ([[VAPO]])
Add an auxiliary NLL loss on correct responses, ensuring the model doesn't forget how to generate good solutions even when rewards are rare.

### Process Reward Models
Train a model to provide per-step rewards (not just outcome). Attempted by [[GRPO - DeepSeek-R1]] but listed as an "unsuccessful attempt" — hard to scale and prone to reward hacking.

### Pass@k Reward ([[Pass@k Training]])
Softens the binary reward by evaluating whether any subset of k samples is correct, providing a more informative signal.

## The Exploration Connection

Sparse rewards exacerbate [[Entropy Collapse]]: if the model rarely finds correct solutions, it gets few positive signals, reinforcing conservative behavior. This makes exploration techniques ([[DAPO]]'s Clip-Higher, [[GMPO]]'s geometric mean) essential.
