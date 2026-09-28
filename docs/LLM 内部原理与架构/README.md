---
type: topic
id: llm-internals
title: LLM 内部原理与架构
order: 1
summary: 从 attention 的数学一路到 KV cache、位置编码与 MoE，覆盖大模型内部结构的面试主线；也是其它所有专题的前置知识。
total: 16
updated: 2026-09-28
---

# LLM 内部原理与架构

这是「跨公司高频问题」里的第一个专题，共 16 道题（当前已上线前 3 道）。它回答的是**模型内部到底发生了什么**：
注意力怎么算、显存花在哪里、为什么某些结构设计能省显存、位置信息如何注入、容量如何扩展。

## 这个专题的知识地图

按依赖顺序，16 道题可以分成 5 组：

| 组 | 主题 | 题目 |
| --- | --- | --- |
| A. 注意力数学（起点） | 缩放因子、注意力分布为什么会失效 | 01 缩放因子 |
| B. 推理期的显存与注意力变体 | KV cache 与显存公式、MQA/GQA、MLA | 02 KV cache、03 MQA/GQA、04 MLA |
| C. 计算效率 | 为什么 FlashAttention 快、MoE 如何扩容量 | 05 FlashAttention、10 MoE |
| D. 位置与归一化、激活 | 位置编码演进、RoPE 与长上下文外推、pre-LN/RMSNorm、SwiGLU | 07 位置编码、08 RoPE 与 YaRN、14 pre-LN/RMSNorm、15 SwiGLU |
| E. 训练与生成的宏观视角 | 分词、预训练/微调/偏好优化、采样策略、缩放定律、长上下文失效、一次前向传播全流程 | 06 BPE、11 训练三阶段、12 采样、09 Chinchilla、13 lost-in-the-middle、16 前向传播 |

**建议顺序**：01 → 02 → 03 → 04 是一条完整主线（注意力数学 → 显存公式 → 结构压缩），面试里经常连着问；
07、08 是另一条（位置信息）；其余可按目标公司挑。

## 已上线题解

| # | 题目 | 一句话看点 |
| --- | --- | --- |
| 01 | [解释 scaled dot-product attention，以及为什么 1/sqrt(d_k) 缩放因子很重要。](01-解释%20scaled%20dot-product%20attention，以及为什么%201-sqrt(d_k)%20缩放因子很重要.md) | 点积方差恰为 $d_k$，除以 $\sqrt{d_k}$ 把方差拉回 1，避免 softmax one-hot 与梯度消失 |
| 02 | [什么是 KV cache？它在规模化场景下的内存影响是什么？请推导公式。](02-什么是%20KV%20cache？它在规模化场景下的内存影响是什么？请推导公式.md) | `2 × L × H_kv × d_head × S × b × bytes`，并用 LLaMA-3-70B 算出每 token 320 KiB |
| 03 | [什么是 Multi-Query Attention（MQA）和 Grouped-Query Attention（GQA）？它们牺牲了什么？](03-什么是%20Multi-Query%20Attention（MQA）和%20Grouped-Query%20Attention（GQA）？它们牺牲了什么.md) | 组内共享 K/V、保留各自的 Q；用注意力多样性换显存与吞吐 |

其余 13 道题的题解在计划中，站点会以「待撰写」标记列出，方便按专题刷题时先看清单。

## 谁在问这个专题

按 `README.md` 的统计，本专题的题目出现在 Anthropic、OpenAI、Google DeepMind、Meta、xAI、Mistral AI、Cohere、
DeepSeek、Moonshot AI、智谱 AI、阿里巴巴、Sarvam AI、Microsoft、Amazon、Apple、NVIDIA、Together AI、
Character.AI、Databricks 等公司的面试中。其中 **KV cache（02）被 8 家公司问到**，是本专题出现频率最高的题；
**MoE（10）被 6 家问到**；**RoPE / 位置编码（07、08）** 集中在 Meta、Moonshot AI、阿里巴巴。

## 自测清单

能不看资料回答下面这些问题，说明这个专题过关了：

- 为什么是 $\sqrt{d_k}$ 而不是 $d_k$？推导里用了哪个假设？
- KV cache 每 token 的显存怎么算？70B 模型 32k 上下文、bf16、batch=1 大概多少显存？
- GQA 里 `num_attention_heads=32, num_key_value_heads=8` 意味着什么？KV cache 缩小几倍？
- RoPE 为什么能编码相对位置？YaRN 解决的问题是什么？
- 为什么把 LayerNorm 放到 pre-block？RMSNorm 省掉了什么？
- SwiGLU 相比 GELU 的 MLP 多了什么、为什么值得多花参数？

## 参考资料

- 本专题题解的主要来源为 Outcome School 的博客（每题文末列出具体文章与链接）。
- 题面与公司分布来自本仓库的 [README.md](../../README.md) 与 [README.zh-CN.md](../../README.zh-CN.md)。
