


## Further Reading:
### Control-Oriented Visual / Multimodal Representation for Robotics: Paper List

  

这是一份围绕 **control-oriented visual representation / visuomotor representation / embodied visual representation / multisensory representation** 的论文阅读清单。

  

核心问题是：机器人需要的视觉表示不只是“图像里有什么”，而是“这个视觉状态对动作、接触、身体状态和任务目标意味着什么”。

  

---

  

### 1. 视觉表示用于机器人控制


|Paper|关键词|为什么读|
|---|---|---|
|[R3M: A Universal Visual Representation for Robot Manipulation](https://arxiv.org/abs/2203.12601)|human video pretraining, manipulation representation, Ego4D|经典入门论文。用人类第一视角视频预训练视觉表示，再作为 frozen perception module 给机器人 policy 使用。适合理解“为什么人类操作视频能帮助机器人控制”。|
|[Real-World Robot Learning with Masked Visual Pre-training / MVP](https://tetexiao.com/projects/mvp)|MAE, masked visual pretraining, real-world robot learning|用 masked autoencoder 风格的视觉预训练帮助真实机器人控制。适合理解 self-supervised visual pretraining 如何迁移到 motor control。|
|[Where are we in the search for an Artificial Visual Cortex for Embodied Intelligence? / VC-1](https://arxiv.org/abs/2303.18240)|embodied AI benchmark, visual foundation model, CortexBench|系统比较预训练视觉模型在 locomotion、navigation、dexterous manipulation、mobile manipulation 等 embodied AI 任务上的表现。重要结论：没有一个视觉 backbone 在所有任务上都稳赢。|


---

  

### 2. Control-Centric / Reward-Centric Representation


|Paper|关键词|为什么读|
|---|---|---|
|[VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training](https://arxiv.org/abs/2210.00030)|visual reward, value representation, goal-conditioned RL, human videos|把 human video representation learning 看成 offline goal-conditioned RL 问题。重点不只是学 feature，而是学可以当 dense reward / value 用的视觉表示。|
|[LIV: Language-Image Representations and Rewards for Robotic Control](https://arxiv.org/abs/2306.00958)|vision-language reward, language-conditioned robotics, action-free video|学 joint vision-language representation 和 reward。适合理解语言如何帮助机器人视觉表示更贴近任务和控制。|
|[Language-Driven Representation Learning for Robotics / Voltron](https://arxiv.org/abs/2302.12766)|language-driven visual representation, human videos, robotics|和 R3M / LIV 同一族，重点是用语言监督帮助视觉表示学到更任务相关、更高层的 features。适合接在 R3M 后读。|

---

  

### 3. 机器人数据 + Proprioception / Action 联合训练

| Paper                                                                                                                                          | 关键词                                                                         | 为什么读                                                                                                                                                              |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Robots Pre-train Robots: Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets / MCR](https://arxiv.org/abs/2410.22325) | DROID, proprioception, action dynamics, manipulation-centric representation | 非常贴近“联合 sensor/action data 训练”的问题。用机器人数据预训练视觉 encoder，并通过 proprioceptive state-action dynamics、action prediction、time contrastive loss 让表示更 manipulation-centric。 |
| [DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset](https://arxiv.org/abs/2403.12945)                                                | large-scale robot data, real-world demos, multi-scene manipulation          | 不是 encoder 方法本身，但很重要。它提供大量真实机器人 demonstration，是后续 robot representation / policy pretraining 的关键数据基础。                                                              |


  

---

  

### 4. 3D / Geometry-Oriented Representation

  

| Paper                                                                                                                           | 关键词                                                   | 为什么读                                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| [3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations](https://arxiv.org/abs/2403.03954) | point cloud, 3D representation, diffusion policy      | 如果你关心“控制需要几何，而不只是语义”，这篇很重要。它把 sparse point cloud 的 compact 3D representation 接到 diffusion policy 上，强调 3D representation 对真实机器人泛化很关键。 |
| [3D-MVP: 3D Multiview Pretraining for Robotic Manipulation](https://arxiv.org/abs/2406.18158)                                   | multi-view, MAE, 3D pretraining, robotic manipulation | 把 MVP / MAE 的思想推进到 3D multi-view robot manipulation。适合和 3D Diffusion Policy 对照读。                                                     |

  

---

  

### 5. Visuo-Tactile / Multisensory Representation

  

| Paper                                                                                                                                             | 关键词                                                                           | 为什么读                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| [Self-Supervised Visuo-Tactile Pretraining to Locate and Follow Garment Features](https://www.roboticsproceedings.org/rss19/p018.pdf)             | visual-tactile contrastive learning, deformable object, garment manipulation  | 用 paired visual/tactile data 做 cross-modal learning。适合理解触觉如何补足视觉看不到的接触信息，尤其是布料等 deformable object。                   |
| [VITaL Pretraining: Visuo-Tactile Pretraining for Tactile and Non-Tactile Manipulation Policies](https://arxiv.org/abs/2403.11898)                | tactile, imitation learning, visuo-tactile pretraining                        | 研究 visuo-tactile pretraining 如何帮助 tactile policy，甚至帮助 inference 时不用 tactile 的 vision-only policy。适合理解触觉预训练如何迁移到下游控制。 |
| [Touch in the Wild: Learning Fine-Grained Manipulation with Visuo-Tactile Pretraining](https://binghao-huang.github.io/touch_in_the_wild/)        | large visuo-tactile corpus, proprioception, downstream imitation learning     | 用 image-tactile pairs 做预训练，得到 joint visuo-tactile representation，再结合 robot proprioceptive states 用于下游 manipulation。  |
| [Visual-tactile pretraining and online multitask learning for contact-rich manipulation](https://www.science.org/doi/10.1126/scirobotics.ady2869) | tactile, dexterous hand, contact-rich manipulation, online multitask learning | 聚焦多指手和接触丰富的任务。适合深入理解 dexterous manipulation 中视觉、触觉和在线学习的结合。                                                          |

  

---

  

### 6. 通用视觉 Foundation Model 用于机器人

  

这些模型不一定专门为控制训练，但经常作为 robot policy 的 visual backbone 或对照组。

  

| Paper                                                                                                     | 关键词                                                             | 为什么读                                                                                                        |
| --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)           | self-supervised vision, dense features, visual foundation model | DINOv2 不是机器人专用，但 dense visual features 很强，经常被拿来接 robot policy。适合理解通用 self-supervised vision backbone 能提供什么。 |
| [Theia: Distilling Diverse Vision Foundation Models for Robot Learning](https://arxiv.org/abs/2407.20179) | distillation, diverse vision foundation models, robot learning  | 思路很有趣：不是押一个视觉模型，而是把多个 vision foundation model 的 features 蒸馏成一个更适合 robot learning 的表示。                       |

  

---

  

### 推荐阅读顺序

  

如果你想循序渐进，可以按这个顺序读：

  

如果你想循序渐进，可以按这个顺序读：

| 顺序  | Paper                             | 目的                                                                           |
| --- | --------------------------------- | ---------------------------------------------------------------------------- |
| 1   | R3M                               | 理解人类视频预训练如何帮助 robot manipulation。                                            |
| 2   | MVP                               | 理解 masked visual pretraining 如何迁移到真实机器人控制。                                   |
| 3   | VC-1                              | 建立 embodied visual representation 的全局地图。                                     |
| 4   | VIP                               | 理解视觉表示如何变成 reward / value。                                                   |
| 5   | LIV                               | 理解 vision-language representation 如何服务 robotic control。                      |
| 6   | MCR                               | 理解如何联合 robot action / proprioception 训练 manipulation-centric representation。 |
| 7   | DROID                             | 理解大规模真实机器人数据集的形态。                                                            |
| 8   | 3D Diffusion Policy               | 理解 3D representation 如何增强 diffusion policy。                                  |
| 9   | 3D-MVP                            | 理解 multi-view / 3D masked pretraining。                                       |
| 10  | VITaL / Visuo-Tactile Pretraining | 理解视觉 + 触觉如何帮助 contact-rich manipulation。                                     |

  

---

  

### 最重要的三篇

  

| Paper | 核心价值                                                                        |
| ----- | --------------------------------------------------------------------------- |
| R3M   | 人类视频 → robot manipulation representation。                                   |
| VC-1  | 系统评估 embodied AI visual foundation models。                                  |
| MCR   | 机器人 proprioception / action dynamics → manipulation-centric representation。 |
  

---

  

### 一句话总结

  

这条研究线可以粗略分成几层：

  

| 方向                      | 解决的问题                                        |
| ----------------------- | -------------------------------------------- |
| R3M / MVP / VC-1        | 机器人能不能借用大规模视觉预训练？                            |
| VIP / LIV               | 视觉表示能不能更像 reward / value，而不只是 image feature？ |
| MCR / DROID             | 能不能从机器人自己的动作、本体状态和轨迹中学控制相关表示？                |
| 3D / Visuo-Tactile work | 能不能从“看图”升级到理解几何、接触和物理交互？                     |
  

真正的机器人视觉不应该只是识别物体，而应该学会看见 **affordance、geometry、contact、dynamics、embodiment**。普通视觉 encoder 是眼睛，control-oriented encoder 是长在手腕神经上的眼睛。