https://arxiv.org/pdf/2501.12948 i read the deepseek paper and am insterested in the unsuccessful attempt that they mention they are trying step level process reward model, is there any update on that now?

Thought for 57s

Yes, but the update is a bit of a split-screen story.

**From DeepSeek specifically:** I do **not** see a public sign that DeepSeek has since come back and said, “we solved step-level PRMs and now use them in R1-style training.” In the revised **v2** of the R1 paper, posted on **January 4, 2026**, they still keep **Process Reward Model (PRM)** in the “Unsuccessful Attempts” section and repeat the same core objections: step definition is hard, intermediate-step correctness is hard to label at scale, and neural PRMs tend to invite reward hacking during large-scale RL.

Their later public releases do not look like a reversal either. The **DeepSeek-R1-0528** model card says the update came from “increased computational resources” and “algorithmic optimization mechanisms during post-training,” but it does not say they switched to step-level PRMs. And in **DeepSeek-Prover-V2** from late 2025, DeepSeek explicitly says they use **binary correct-or-incorrect feedback as the primary reward supervision**, which points more toward outcome/verifiable rewards than a full step-level PRM pipeline.

So the cleanest answer is:

**No public DeepSeek update says they successfully adopted step-level process reward models for their main reasoning training stack.** Their public artifacts still lean toward outcome-level or verifiable rewards.

That said, the **research field** absolutely did not leave PRMs in the attic. It turned into a bustling repair shop:

- **Qwen’s “The Lessons of Developing Process Reward Models in Mathematical Reasoning”** argued PRMs are still useful, but building them well is harder than people first thought. They found common Monte Carlo data synthesis weak, proposed better data filtering, and released stronger math PRMs.
- **PRIME** tried to dodge one of DeepSeek’s complaints by learning **implicit process rewards** from rollouts plus outcome labels, instead of relying on expensive explicit step labels.
- **ThinkPRM** showed that a **generative verifier** can work surprisingly well with far fewer process labels than classic discriminative PRMs.
- **PURE** tackled DeepSeek’s reward-hacking objection more directly. Its claim is that part of the hacking comes from the usual **sum-style credit assignment**, and that **min-form credit assignment** makes PRM-based RL much more stable.
- **SCOPE** used PRMs in a narrower, more surgical way: not as the whole reward engine, but to identify the first bad step in failed rollouts and salvage partially correct trajectories.

But the fog has not lifted completely. There are also newer results arguing that PRMs can still **misguide reasoning models**, especially native long-thinking models, and may help less than expected unless the setup is very careful.

My take: DeepSeek’s “unsuccessful attempt” has aged less like a dead end and more like an early warning label. The field’s current consensus seems to be:

1. **Naive step-level PRM training is brittle.**
2. **PRMs can still help**, especially for reranking, search, failure recovery, or in constrained domains.
3. The promising trend is **hybridization**: implicit rewards, better credit assignment, verifier-style PRMs, and using PRMs as a helper rather than the single throne room reward signal.

So if your question is “was DeepSeek simply wrong about PRMs?”, I’d say **no**. If your question is “has the community made progress on exactly the problems they pointed out?”, then **yes, quite a bit**.

If you want, I can next do a tighter comparison of **DeepSeek’s objections vs. how PRIME / ThinkPRM / PURE / SCOPE each try to patch one specific failure mode**.

sure

Thought for 14s

Yep. Here’s the tighter map, with DeepSeek’s objections on the left and what later work tried on the right.

**Bottom line first:** I still do **not** see a public DeepSeek update saying they successfully brought step-level PRMs into the main R1 line. Their paper still presents PRMs as an unsuccessful path for that training setup, and their later public R1-0528 release notes improvements from more compute and post-training optimization without saying they switched to step-level PRMs.

## 1. DeepSeek objection: **step boundaries are messy**

DeepSeek’s complaint was not just “PRMs are bad.” It was more like: _for free-form reasoning, what counts as a step is slippery_, so assigning correct local rewards becomes noisy and gameable.

Later work attacked that in a few different ways:

- **ThinkPRM** leans into a **generative verifier** that “thinks” through whether a step is valid, instead of using only a compact discriminative scorer. The idea is that richer verification can tolerate fuzzier step boundaries better, while also needing far fewer process labels than classic PRMs.
- **FunPRM** goes even more directly at the granularity problem by proposing **function-as-step** supervision and then using final-solution signals to purify noisy partial-step rewards. That is basically an attempt to stop the “what exactly is a step?” issue from poisoning the whole signal.

**Interpretation:** this objection has not been “solved,” but newer work is trying to make the step unit more semantic or more robust to noise, instead of pretending token chunks are clean reasoning atoms.

## 2. DeepSeek objection: **step labels are expensive and unreliable**

This one aged like steel. A lot of later papers basically agree with DeepSeek that dense process labels are a headache.

- **Qwen’s PRM lessons paper** says common **Monte Carlo estimation** for generating process labels often generalizes worse than stronger labeling approaches, and that standard Best-of-N evaluation can overstate PRM quality. They propose **consensus filtering** plus better evaluation to get a stronger open PRM.
- **Scaling Automated Process Verifiers** also frames the same bottleneck plainly: dense per-step human labels do not scale, and automatically labeled PRMs had shown limited gains.
- **ThinkPRM** is again relevant here because its whole sales pitch is “use far fewer process labels.”
- **PRIME** takes the bolder detour: instead of requiring explicit step labels, it learns **implicit process rewards** online from rollouts plus outcome labels. That is almost a direct reply to DeepSeek’s “this labeling pipeline is too brittle and expensive” concern.

**Interpretation:** the field mostly accepted DeepSeek’s criticism here. The response has been to reduce dependence on explicit step labels, not to make giant human-labeled PRM pipelines the norm.

## 3. DeepSeek objection: **PRMs invite reward hacking during RL**

This is the spiciest one, and several papers tackle it head-on.

- **PURE** argues the hacking is not only because PRMs are imperfect, but because the standard **sum-form credit assignment** in RL makes it easy for the model to exploit locally high-reward steps. Their fix is **min-form credit assignment**, which they report stabilizes PRM-based RL much better.
- **PRIME** also exists partly because online PRM training is especially vulnerable to hacking when process labels are weak. By backing into process rewards from outcomes, it tries to avoid a fragile separately trained PRM becoming the candy machine the policy learns to shake.
- **Verified Critical Step Optimization** uses PRMs more conservatively: PRM identifies likely critical steps, but only alternatives that actually lead to successful completion are kept as training data. In other words, PRM proposes where to look, while **verified outcomes** decide what survives.
- **SGPO** similarly uses a step-wise judge to find the **first incorrect step** and guide optimization from there, rather than handing total authority to a dense step reward stream.

**Interpretation:** this is where the field shifted most. The newer mood is not “train on raw PRM scores and pray.” It is closer to: **use PRMs as scaffolding, diagnostics, search guidance, or partial credit, and keep outcome verification in the loop**. Tiny judges, not philosopher-kings 👑

## 4. So was DeepSeek “wrong”?

I’d say **not really**.

DeepSeek’s public claim was never that process supervision is useless in principle. It was that their **step-level PRM attempt** was not worth it in their large-scale RL setting because of step ambiguity, label difficulty, and hacking. The later literature mostly **confirms those pain points**, while proposing narrower fixes and hybrids rather than a clean reversal.

## 5. What seems true as of now

My best synthesis is:

- **For pure large-scale RL training**, outcome-level or verifiable rewards still look simpler and more robust than naive step-level PRMs. DeepSeek’s public trajectory still points that way.
- **For test-time scaling, reranking, search, debugging, and step diagnosis**, PRM-like verifiers are very alive and getting better. ThinkPRM, SGPO, and verified-critical-step style methods all push in that direction.
- The frontier is increasingly **hybrid**: implicit process rewards, generative verifiers, better credit assignment, and using outcome verification to keep the PRM honest.

So the “update” is not **DeepSeek later succeeded with PRMs**. It is more: **the community spent 2025 into 2026 trying to build safer, cheaper, less gameable versions of the exact thing DeepSeek said was hard.**

I can also turn this into a compact table: **DeepSeek failure mode → later paper → core trick → what it probably means for future RL reasoning systems**.