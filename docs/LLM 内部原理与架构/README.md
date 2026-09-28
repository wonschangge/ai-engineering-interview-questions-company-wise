---
type: topic
id: llm-internals
title: LLM 内部原理与架构
order: 1
summary: 从 attention 的数学一路到 KV cache、位置编码、MoE 与逐张量的前向传播，覆盖大模型内部结构的面试主线；也是其它所有专题的前置知识。
total: 16
updated: 2026-09-28
---

# LLM 内部原理与架构

这是「跨公司高频问题」里的第一个专题，共 16 道题，**已全部上线**。它回答的是**模型内部到底发生了什么**：
注意力怎么算、显存花在哪里、结构设计如何省显存、位置信息如何注入、容量如何扩展、一次前向传播里张量怎么流动。

## 题解清单

| # | 题目 | 一句话看点 |
| --- | --- | --- |
| 01 | [解释 scaled dot-product attention，以及为什么 1/sqrt(d_k) 缩放因子很重要。](01-解释%20scaled%20dot-product%20attention，以及为什么%201-sqrt%28d_k%29%20缩放因子很重要.md) | 点积方差恰为 $d_k$（两条推导路径），除以 $\sqrt{d_k}$ 把方差拉回 1，避免 softmax one-hot 与梯度消失 |
| 02 | [什么是 KV cache？它在规模化场景下的内存影响是什么？请推导公式。](02-什么是%20KV%20cache？它在规模化场景下的内存影响是什么？请推导公式.md) | 显存公式 `2·L·H_kv·d_head·S·b·bytes` 逐项推导；LLaMA-3-70B 每 token 320 KiB、32k 需 10 GiB |
| 03 | [什么是 Multi-Query Attention（MQA）和 Grouped-Query Attention（GQA）？它们牺牲了什么？](03-什么是%20Multi-Query%20Attention（MQA）和%20Grouped-Query%20Attention（GQA）？它们牺牲了什么.md) | 组内共享 K/V、每个 head 保留自己的 Q；用注意力视角多样性换显存与带宽，倍数随头数与组数变化 |
| 04 | [什么是 Multi-head Latent Attention（MLA）？DeepSeek 为什么引入它？](04-什么是%20Multi-head%20Latent%20Attention（MLA）？DeepSeek%20为什么引入它.md) | 低秩 latent + 解耦 RoPE；同形状相对 MHA 压 56.9×，论文 93.3% 的口径要连部署 6 bit 一起算 |
| 05 | [解释 FlashAttention。它并没有减少 FLOPs，那为什么更快？](05-解释%20FlashAttention。它并没有减少%20FLOPs，那为什么更快.md) | IO-aware：分块 + online softmax 把 HBM 流量降约 3.7×，FLOPs 一个没省（反向还更多） |
| 06 | [Byte Pair Encoding 是如何工作的？它有哪些失效场景（数字、代码、非拉丁文字）？](06-Byte%20Pair%20Encoding%20是如何工作的？它有哪些失效场景（数字、代码、非拉丁文字）.md) | 贪心合并算法逐步手算；数字、代码缩进、非拉丁文字三类失效场景与 tokenizer 公平性 |
| 07 | [什么是 transformer 中的位置编码？它是如何演进的（sinusoidal → learned → RoPE → ALiBi）？](07-什么是%20transformer%20中的位置编码？它是如何演进的.md) | 从「置换等变」讲起，四种方案怎么加位置、能不能外推，一张演进对比表收口 |
| 08 | [解释 RoPE，以及位置插值 / YaRN 如何把上下文扩展到训练长度之外。](08-解释%20RoPE，以及位置插值、YaRN%20如何把上下文扩展到训练长度之外.md) | 旋转矩阵导出相对位置；PI / NTK-aware / YaRN 三档外推的分区策略与代价 |
| 09 | [Chinchilla 缩放定律说了什么？它与更早的缩放直觉有何不同？](09-Chinchilla%20缩放定律说了什么？它与更早的缩放直觉有何不同.md) | 给定算力时 $N$ 与 $D$ 同比例增长（约 20 tokens/参数），以及今天为什么反过来「过度训练」小模型 |
| 10 | [什么是 mixture-of-experts 架构？它如何在不增加 FLOPs 的情况下扩展容量？](10-什么是%20mixture-of-experts%20架构？它如何在不增加%20FLOPs%20的情况下扩展容量.md) | 总参数 vs 激活参数、路由与负载均衡、训练 all-to-all 通信与推理期显存代价 |
| 11 | [解释 pre-training、supervised fine-tuning 与 preference optimisation 之间的区别。](11-解释%20pre-training、supervised%20fine-tuning%20与%20preference%20optimisation%20的区别.md) | 三个阶段的监督信号从哪来：自监督 → 指令微调 → 偏好优化（RLHF 三段式 / DPO / GRPO） |
| 12 | [比较 greedy、beam search、top-k、top-p 与 temperature 采样。它们各自在什么情况下会失败？](12-比较%20greedy、beam%20search、top-k、top-p%20与%20temperature%20采样.md) | 同一条「温度缩放 → 截断 → 归一化 → 采样」流水线，五种策略的失效模式与场景参数 |
| 13 | [长上下文中的 lost-in-the-middle 问题是什么？该如何解决？](13-长上下文中的%20lost-in-the-middle%20问题是什么？该如何解决.md) | U 形位置偏差的受控实验、机制解释，以及 RAG 的落地动作：少塞、精排、两端放 |
| 14 | [为什么现代 transformer 把 LayerNorm 放在 pre-block 位置？RMSNorm 是什么？](14-为什么现代%20transformer%20把%20LayerNorm%20放在%20pre-block%20位置？RMSNorm%20是什么.md) | pre-LN 让梯度沿恒等路径直通；RMSNorm 去掉均值中心化，省一次 reduction 与一组 bias |
| 15 | [解释 SwiGLU，以及为什么现代 LLM 的 MLP 模块中用门控激活取代了 ReLU/GELU。](15-解释%20SwiGLU，以及为什么现代%20LLM%20的%20MLP%20模块中用门控激活取代了%20ReLU、GELU.md) | 门控 FFN 的形式与参数账：隐藏维为什么取 8/3 d 才与原始 FFN 参数量相当 |
| 16 | [请逐个张量地讲一遍 decoder-only transformer 一次前向传播中发生的过程。](16-逐个张量地讲一遍%20decoder-only%20transformer%20的一次前向传播.md) | 综合题：从 `[B,S]` 到 `[B,S,V]` 的逐张量形状表，附带参数量与 FLOPs 量级 |

## 这个专题的知识地图

按依赖顺序，16 道题可以分成 5 组：

| 组 | 主题 | 题目 |
| --- | --- | --- |
| A. 注意力数学（起点） | 缩放因子、注意力分布为什么会失效 | 01 |
| B. 推理期的显存与注意力变体 | KV cache 与显存公式、MQA/GQA、MLA | 02、03、04 |
| C. 计算效率 | 为什么 FlashAttention 快、MoE 如何扩容量 | 05、10 |
| D. 位置、归一化与激活 | 位置编码演进、RoPE 与长上下文外推、pre-LN/RMSNorm、SwiGLU | 07、08、14、15 |
| E. 训练与生成的宏观视角 | 分词、训练三阶段、采样、缩放定律、长上下文失效、一次前向传播全流程 | 06、09、11、12、13、16 |

**建议顺序**：`01 → 02 → 03 → 04` 是一条完整主线（注意力数学 → 显存公式 → 结构压缩），面试里经常连着问；
`07 → 08` 是第二条（位置信息如何注入与扩展）；`14 → 15 → 16` 是第三条（把 block 拆开、再串成一次前向传播）。
其余题目可以按目标公司挑着看。

## 谁在问这个专题

按 `README.md` 的统计，本专题的题目出现在 Anthropic、OpenAI、Google DeepMind、Meta、xAI、Mistral AI、Cohere、
DeepSeek、Moonshot AI、智谱 AI、阿里巴巴、Sarvam AI、Microsoft、Amazon、Apple、NVIDIA、Together AI、
Character.AI、Databricks 等公司的面试中。其中 **KV cache（02）被 8 家公司问到**，是本专题出现频率最高的题；
**MoE（10）被 6 家问到**；**RoPE / 位置编码（07、08）** 集中在 Meta、Moonshot AI、阿里巴巴；
**FlashAttention（05）** 来自 Together AI 这类推理基础设施团队；**lost-in-the-middle（13）** 出现在 Moonshot AI 的长上下文岗位上。

## 自测清单

能不看资料回答下面这些问题，说明这个专题过关了：

- 为什么是 $\sqrt{d_k}$ 而不是 $d_k$？推导里用了哪个假设？
- KV cache 每 token 的显存怎么算？70B 模型 32k 上下文、bf16、batch=1 大概多少显存？
- GQA 里 `num_attention_heads=32, num_key_value_heads=8` 意味着什么？KV cache 缩小几倍？
- MLA 压缩的是什么？为什么 RoPE 必须「解耦」出来单独处理？
- FlashAttention 既然没减少 FLOPs，快在哪里？online softmax 为什么和普通 softmax 数值等价？
- BPE 的合并顺序为什么重要？中文一句话为什么比英文更费 token？
- sinusoidal 位置编码为什么「被认为能外推」？ALiBi 又省掉了什么？
- RoPE 的内积为什么只依赖相对位置？PI、NTK-aware、YaRN 分别在改什么？
- Chinchilla 说的 20 tokens/参数是铁律吗？为什么今天的小模型要训练远超这个比例？
- MoE 的「总参数」与「激活参数」分别决定了什么？为什么推理时它不一定省显存？
- SFT 之后为什么还需要偏好优化？PPO 里的 KL 项在防什么？
- 为什么 beam search 在开放式生成上不如采样？top-p 相比 top-k 自适应在哪？
- 上下文窗口是 128k，为什么把关键信息放中间仍然可能被忽略？工程上怎么放？
- pre-LN 为什么能省掉 warmup 的烦恼？RMSNorm 去掉均值为什么还能工作？
- SwiGLU 的隐藏维为什么取 $8d/3$？不这么取会怎样？
- 从 `[B,S]` 的 token id 到 `[B,S,V]` 的 logits，中间哪些张量是 $S^2$ 量级？哪些是逐 token 的？

## 参考资料

- 本专题题解的主要来源为 Outcome School 的博客与若干原始论文（每题文末列出具体文章、作者与链接）。
- 题面与公司分布来自本仓库的 [README.md](../../README.md) 与 [README.zh-CN.md](../../README.zh-CN.md)。
