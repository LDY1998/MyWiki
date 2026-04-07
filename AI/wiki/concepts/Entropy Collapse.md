# Entropy Collapse

A failure mode in LLM RL training where the policy's token distribution becomes overly deterministic, killing exploration and causing training to stagnate.

## What Happens

During [[GRPO]]/[[PPO]] training, the model's output entropy (diversity of token probabilities) steadily decreases. When it drops too low:
1. The model commits to a narrow set of reasoning patterns
2. It can no longer explore alternative solution paths
3. Performance plateaus or degrades
4. Training may crash due to numerical instability

## Why It Happens

In standard [[GRPO]] with symmetric clipping (1-ε, 1+ε):
- Correct tokens get probability increases (capped at 1+ε)
- Incorrect tokens get probability decreases (floored at 1-ε)
- The asymmetry compounds: low-probability tokens keep shrinking but high-probability tokens are capped
- Net effect: entropy monotonically decreases

## Solutions

### Clip-Higher ([[DAPO]], [[VAPO]])
Decouple the clipping range: use smaller ε_low (e.g., 0.2) but larger ε_high (e.g., 0.28). This gives low-probability tokens more room to increase, maintaining exploration.

### Geometric Mean ([[GMPO]])
Replace arithmetic mean with geometric mean in the objective. This dampens outlier importance sampling ratios that drive entropy collapse, allowing the use of even larger clipping ranges.

### Pass@k Reward ([[Pass@k Training]])
By rewarding diverse correct paths (not just the most likely one), the policy is incentivized to maintain higher entropy.

### Remove KL Penalty ([[DAPO]])
The KL penalty anchors the policy to the initial model, which may itself have low entropy. Removing it allows the policy to explore more freely in long-CoT scenarios.

## Monitoring

Track policy entropy during training. Healthy training shows gradual, controlled entropy decrease. A sharp drop signals collapse.

```
Healthy:   ████████████▓▓▓▓▓▓▒▒▒▒░░  (gradual)
Collapse:  ████████▓░                  (cliff)
```
