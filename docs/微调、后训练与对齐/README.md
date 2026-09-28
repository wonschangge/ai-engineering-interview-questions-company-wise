---
type: topic
id: finetuning
title: 微调、后训练与对齐
order: 5
summary: 从 RLHF/DPO/GRPO 的推导到 LoRA/QLoRA 的参数与显存账，覆盖把基座模型变成可交付产品所需的训练与对齐判断。
total: 12
updated: 2026-09-28
---

# 微调、后训练与对齐

这是「跨公司高频问题」里的第五个专题，共 12 道题，**已全部上线**。它回答的是**模型的行为怎么被塑造**：
偏好怎么变成梯度、要不要真的做 RL、参数该改多少、显存够不够、以及「到底该提示、该检索还是该微调」。

## 题解清单

| # | 题目 | 一句话看点 |
| --- | --- | --- |
| 01 | [请端到端讲一遍 RLHF：reward model、策略优化、KL 惩罚。](01-端到端讲一遍%20RLHF：reward%20model、策略优化与%20KL%20惩罚.md) | Bradley-Terry 偏好损失 → PPO 的 clip 目标 → KL 的三层作用；四模型同场的工程代价 |
| 02 | [什么是 DPO？为什么它在许多实验室里取代了基于 PPO 的 RLHF？什么情况下 online RL 仍然更好？](02-什么是%20DPO？为什么它取代了基于%20PPO%20的%20RLHF.md) | 从 KL 约束的闭式解反解出奖励，把 RL 变成分类；offline 的边界正是 online RL 的用武之地 |
| 03 | [解释 GRPO，以及为什么在规模化场景下去掉 value network 很关键。](03-解释%20GRPO%20以及为什么去掉%20value%20network%20很关键.md) | 组内标准化替代 value 基线：省一整个同规模模型及其优化器状态，且天然契合可验证奖励 |
| 04 | [从数学上解释 LoRA 的分解。它为什么有效，以及如何选择秩 r？](04-从数学上解释%20LoRA%20的分解与秩的选择.md) | $\Delta W = BA$ 的参数量账 + 低内在秩的经验依据；$r$ 的扫描流程与四个配套旋钮 |
| 05 | [QLoRA 是如何降低显存占用的？它需要做哪些 quantization 权衡？](05-QLoRA%20如何降低显存占用以及它需要哪些量化权衡.md) | NF4 + 双重量化 + 分页优化器三件套；7B/13B/70B 的显存账与「什么时候别用」 |
| 06 | [比较 LoRA、prefix tuning、prompt tuning 和 full fine-tuning。分别在什么场景下选择它们？](06-比较%20LoRA、prefix%20tuning、prompt%20tuning%20与全量微调.md) | 权重空间 vs 激活空间两条路线；软提示占上下文且与 prefix cache 冲突 |
| 07 | [什么是灾难性遗忘？在 fine-tuning 过程中如何缓解它？](07-什么是灾难性遗忘？如何在微调中缓解.md) | 数据/参数/目标/流程四层缓解；实证口径是 1B–7B 尺度上的持续指令微调 |
| 08 | [Prompting、RAG 还是 fine-tuning：请给出你的决策框架，并把成本和 latency 一并考虑进去。](08-Prompting、RAG%20还是%20fine-tuning：决策框架与成本权衡.md) | 先问「缺知识还是缺行为」；给出微调的盈亏平衡点算法与可复算算例 |
| 09 | [算一下用 Adam 以 bf16 全量 fine-tuning 一个 7B 模型所需的 GPU 显存。换成 LoRA 呢？](09-算一下%207B%20模型%20bf16%20全量微调与%20LoRA%20的%20GPU%20显存.md) | 16 bytes/param 的经典账 + ZeRO 分片；LoRA 省的是梯度与优化器状态而不是权重 |
| 10 | [什么是 RLVR（RL with verifiable rewards）？它在哪些方面胜过学到的 reward model？](10-什么是%20RLVR%20以及它为何胜过学到的%20reward%20model.md) | 规则奖励精确、可规模化、不可欺骗；但只覆盖可验证任务，且验证器本身会被钻空子 |
| 11 | [解释 RLHF 中的 reward hacking，以及各实验室如何应对它。](11-解释%20RLHF%20中的%20reward%20hacking%20以及各实验室如何应对.md) | Goodhart 定律的实证曲线；目标/数据/评测/流程四层防线与症状指标 |
| 12 | [什么是 distillation？如何基于大模型构建一个能力强的小模型？](12-什么是%20distillation？如何基于大模型构建能力强的小模型.md) | logit 蒸馏的损失与温度 vs 今天主流的序列级/推理蒸馏；能力天花板与错误继承 |

## 这个专题的知识地图

按「从偏好到部署」的顺序，12 道题分成四组：

| 组 | 主题 | 题目 |
| --- | --- | --- |
| A. 偏好优化三件套 | RLHF 的完整链路、DPO 的推导与边界、GRPO 的省显存设计 | 01、02、03 |
| B. 参数高效微调 | LoRA 的数学与调参、QLoRA 的量化权衡、PEFT 家族选型 | 04、05、06 |
| C. 决策与成本 | 提示/检索/微调的判断框架、显存与训练成本的定量账 | 08、09 |
| D. 风险与收益 | 灾难性遗忘、reward hacking、可验证奖励、蒸馏 | 07、11、10、12 |

**建议顺序**：`01 → 02 → 03` 是偏好优化主线（也是被问得最深的一条，面试常要求现场推 DPO 的式子）；
`04 → 05 → 06` 是工程主线（参数量、显存、选型）；`08 → 09` 是「要不要做、做不做得起」的决策；
`07 → 11` 是两个失败模式（学了新的忘了旧的、把奖励刷爆）；`10 → 12` 是当下最热的两种收益路径。

## 谁在问这个专题

按 `README.md` 的统计，本专题的题目出现在 13 家公司的面试中：**Hugging Face、Scale AI、Mistral AI（各 3 题）**
覆盖最广——从 DPO/QLoRA 的实现细节到显存账与 RLHF 全链路；**Sarvam AI 与阿里巴巴（各 2 题）** 关注 PEFT 选型、
RLVR 与蒸馏（与它们训练多语言/自研模型的路线一致）；**DeepSeek** 问 GRPO；**OpenAI、Microsoft、Databricks、Glean、Cohere**
问「提示/RAG/微调」的决策框架；**Apple** 问 PEFT 家族；**智谱 AI** 问 RLVR。

## 自测清单

能不看资料回答下面这些问题，说明这个专题过关了：

- RLHF 的三个阶段各需要什么数据？KL 项拦住的到底是什么？
- 从 KL 约束的 RLHF 目标出发，怎么推出 DPO 的损失？配方项为什么能消掉？
- DPO 是 offline 的，这个性质具体带来哪三个弱点？什么信号说明该换 online RL？
- GRPO 的优势是怎么算的？去掉 value network 省下了什么、代价是什么？
- LoRA 的可训练参数量怎么算？$r$ 从 8 变到 64，参数量与效果各怎么变？
- QLoRA 的三个创新分别解决什么问题？4-bit 底座为什么还能保持质量？
- prefix tuning 与 prompt tuning 的差别在哪一层？为什么软提示会占上下文预算？
- 微调一个 7B 模型，Adam + bf16 每参数要多少字节？其中 LoRA 省掉了哪几项？
- 为什么「LoRA 让 70B 单卡可训」这个说法不严谨？
- 什么任务适合 RLVR？它的三类失效模式是什么？
- reward hacking 的六个常见症状是什么？为什么 KL 系数是最重要的旋钮之一？
- 蒸馏在 LLM 时代的主流形态是什么？为什么 logit 蒸馏不常用？
- 什么时候该提示、什么时候该检索、什么时候该微调？盈亏平衡点怎么算？

## 参考资料

- 本专题题解的主要来源为 Outcome School 的博客与对齐/参数高效微调方向的原始论文（InstructGPT、PPO、DPO、DeepSeekMath（GRPO）、
  DeepSeek-R1、LoRA、QLoRA、Prefix-Tuning、P-Tuning v2、ZeRO、LIMA、Self-Instruct、Concrete Problems in AI Safety、
  Reward Model Overoptimization、Constitutional AI、DistilBERT 等），每题文末列出具体文章、作者与链接。
- 题面与公司分布来自本仓库的 [README.md](../../README.md) 与 [README.zh-CN.md](../../README.zh-CN.md)。
