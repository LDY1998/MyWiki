# Chain-of-Thought (CoT) Reasoning

The paradigm where LLMs generate explicit step-by-step reasoning before arriving at a final answer. Long CoT is the key capability targeted by all algorithms in this wiki.

## Short vs Long CoT

- **Short CoT**: A few reasoning steps (e.g., GSM8K grade-school math). Standard methods work fine.
- **Long CoT**: Hundreds to thousands of tokens of reasoning (e.g., AIME competition math). Creates unique challenges for RL training.

## Why Long CoT Breaks Standard RL

1. **[[Sparse Rewards]]**: Single reward after thousands of tokens
2. **[[GAE]] decay**: Reward signal decays exponentially backward through long sequences
3. **[[Value Network]] bias**: Harder to predict value accurately over long horizons
4. **[[Entropy Collapse]]**: More tokens = more opportunities for the policy to become deterministic
5. **Variable length**: Responses range from short to very long, requiring algorithms to handle both ([[VAPO]]'s Length-Adaptive GAE)

## Key Result: DeepSeek-R1-Zero

[[GRPO - DeepSeek-R1]] showed that long CoT reasoning can emerge naturally from pure RL without any supervised demonstrations. The model spontaneously develops:
- Self-verification ("let me check this...")
- Reflection ("wait, that's wrong...")
- Extended exploration of solution paths

## Test-Time Scaling

Long CoT enables test-time scaling: more compute at inference (longer reasoning chains) = better answers. This is the paradigm shift introduced by OpenAI o1 and replicated by [[GRPO - DeepSeek-R1]].
