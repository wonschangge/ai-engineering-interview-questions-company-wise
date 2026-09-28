---
type: topic
id: inference-serving
title: 推理、服务与 GPU 性能
order: 2
summary: 从 prefill/decode 的算术强度到 batching、PagedAttention、量化、并行策略与容量规划，覆盖把大模型跑成线上服务所需的全部判断题。
total: 15
updated: 2026-09-28
---

# 推理、服务与 GPU 性能

这是「跨公司高频问题」里的第二个专题，共 15 道题，**已全部上线**。它回答的是**模型怎么变成服务**：
钱花在算力还是带宽上、显存怎么分配、批怎么调度、精度降到哪一档、延迟与吞吐怎么取舍、成本怎么降下来。

## 题解清单

| # | 题目 | 一句话看点 |
| --- | --- | --- |
| 01 | [解释 prefill 和 decode 两个阶段。为什么 prefill 是计算受限的，而 decode 是内存带宽受限的？](01-解释%20prefill%20与%20decode%20两个阶段为什么一个计算受限一个带宽受限.md) | 用算术强度（FLOPs/byte）判断瓶颈：prefill ≈ $S$，decode ≈ 1，而 H100 的拐点约 295 |
| 02 | [什么是 continuous（in-flight）batching？为什么它取代了 static batching？](02-什么是%20continuous%20batching？为什么它取代了%20static%20batching.md) | 把批的边界从「请求级」降到「token 级」；step 级模拟给出 slot 利用率对比 |
| 03 | [PagedAttention 是如何工作的？它解决了 KV cache 碎片化的什么问题？](03-PagedAttention%20是如何工作的？它解决了%20KV%20cache%20碎片化的什么问题.md) | block + block table 的分页式显存管理，把浪费压到最后一个 block，并支持块级共享与换出 |
| 04 | [什么是 speculative decoding？为什么输出质量能够保持？什么情况下它没有帮助？](04-什么是%20speculative%20decoding？为什么输出质量能保持.md) | 拒绝采样让输出分布与 target 严格一致；收益取决于接受率、$\gamma$ 与算力余量 |
| 05 | [解释 prefix caching / prompt caching。什么时候应该使用它？什么会导致已缓存的 prefix 失效？](05-解释%20prefix%20caching%20与%20prompt%20caching.md) | 因果注意力使前缀 KV 可复用；一个空格、一个时间戳就能让后面全部失效 |
| 06 | [对比用于 serving 的 FP16、BF16、FP8、INT8、INT4 和 FP4。每往下降一档会损失什么？](06-对比%20FP16、BF16、FP8、INT8、INT4%20与%20FP4%20量化.md) | 浮点靠指数拿范围、定点靠 scale 拿范围；每降一档损失的精度、显存、速度与工程复杂度 |
| 07 | [对比 tensor、pipeline、data、sequence 和 expert 并行。什么时候需要组合使用它们？](07-对比%20tensor、pipeline、data、sequence%20与%20expert%20并行.md) | 切什么、通信在哪、通信量随什么增长；给出并行度选择的判断顺序 |
| 08 | [估算服务一个 70B 模型所需的 GPU 显存：权重、KV cache、activations、碎片化。](08-估算服务一个%2070B%20模型所需的%20GPU%20显存.md) | 四块账分别建模再合并，最后落到「这台机器能放多少并发」 |
| 09 | [什么是 TTFT、TPOT、ITL 和 throughput？它们之间如何相互权衡？](09-什么是%20TTFT、TPOT、ITL%20与%20throughput%20以及它们如何权衡.md) | 四个指标的定义与换算；用 goodput（SLO 内的有效吞吐）而不是裸吞吐做验收 |
| 10 | [做一下 roofline 计算：在 batch size 为 1 时，一块 H100 服务 70B 模型每秒能产出多少 token？](10-做%20roofline%20计算：batch%20size%20为%201%20时一块%20H100%20服务%2070B%20模型每秒产出多少%20token.md) | 带宽 ÷ 每 token 需读字节数；FP8/INT4/TP=8 三种口径的上下界与实测折扣 |
| 11 | [什么时候你会选择 vLLM、SGLang、TensorRT-LLM，还是自研技术栈？](11-什么时候选择%20vLLM、SGLang、TensorRT-LLM%20还是自研栈.md) | 三者的优化着力点不同：通用调度 / 前缀树与结构化输出 / 编译期 kernel 融合 |
| 12 | [你会如何把 LLM 服务成本降低 10 倍？列举所有可用的手段并排序。](12-如何把%20LLM%20服务成本降低%2010%20倍：手段清单与排序.md) | 先写成本公式再排序；用「缓存 → 调度 → 量化 → 路由 → 产品限制」凑出 10× 的乘法账 |
| 13 | [一次没有改动模型的部署之后，你的 p99 latency 翻了一倍。请讲一遍排查过程。](13-p99%20latency%20翻倍后的排查过程.md) | 先分辨排队变慢还是计算变慢；队列、缓存、流量结构、配置四类假设 + 二分法 |
| 14 | [什么是 chunked prefill？为什么它能在混合流量下改善 tail latency？](14-什么是%20chunked%20prefill？它如何改善混合流量下的尾延迟.md) | 把长 prefill 切块并与 decode 混批（Sarathi-Serve），用一点 TTFT 换 ITL 的平稳 |
| 15 | [解释 disaggregated prefill/decode 服务，以及它在什么情况下才划算。](15-解释%20disaggregated%20prefill-decode%20服务及其适用条件.md) | 两阶段资源画像相反时分离才划算；KV cache 跨节点传输是收益的第一道门槛 |

## 这个专题的知识地图

按依赖顺序，15 道题可以分成 6 组：

| 组 | 主题 | 题目 |
| --- | --- | --- |
| A. 性能基础 | 两个阶段的瓶颈差异、roofline 估算、指标体系 | 01、10、09 |
| B. 调度与显存 | batching、分页显存、chunked prefill、P/D 分离 | 02、03、14、15 |
| C. 精度与加速 | 量化格式、投机解码 | 06、04 |
| D. 复用与成本 | 前缀缓存、成本优化 | 05、12 |
| E. 并行与容量 | 五种并行、显存容量规划 | 07、08 |
| F. 工程实战 | 引擎选型、p99 排障 | 11、13 |

**建议顺序**：`01 → 10` 先把「瓶颈在哪」算清楚（这是后面所有优化的判断依据）；`02 → 03 → 14 → 15` 是服务的调度主线；
`06 → 08` 是显存与精度的组合拳；`12 → 13` 是面试里最像真实工作的两道综合题。

## 谁在问这个专题

按 `README.md` 的统计，本专题的题目出现在 17 家公司的面试中。**NVIDIA（10 题）与 Together AI（7 题）**
是最高频的提问方，这与岗位性质一致：推理基础设施团队会连着追问 roofline、并行策略、显存账与调度细节；
**Moonshot AI 与 Amazon（各 3 题）** 关注长上下文与容量/成本；**Google DeepMind、Meta** 问并行策略；
**OpenAI、Databricks、Perplexity** 问 p99 排障；**Groq** 关注 P/D 分离。

## 自测清单

能不看资料回答下面这些问题，说明这个专题过关了：

- prefill 和 decode 的算术强度分别是多少？为什么 batch 变大会让 decode 从带宽受限转向计算受限？
- 一块 H100 跑 70B 模型，batch=1 时理论上每秒能出多少 token？扣掉哪些开销后是多少？
- continuous batching 相比 static batching 省下来的到底是什么？它需要什么前置条件？
- PagedAttention 的 block table 解决了哪三类浪费？block size 取大取小各自坏在哪里？
- 投机解码为什么能不改变输出分布？什么情况下它会变慢？
- prefix cache 什么会失效？为什么「动态内容要放到 prompt 最后」？
- FP8 与 INT8 的区别是什么？为什么激活比权重更难量化？
- TP/PP/DP/SP/EP 各自的通信发生在哪里、通信量随什么增长？为什么 TP 通常不超过单机 8 卡？
- 8×H100 服务 70B 模型、8k 上下文，最多能放多少并发？这个数字被哪几项卡住？
- TTFT 与 TPOT 分别在什么场景下更重要？为什么验收要用 goodput？
- 把成本降 10 倍，你的前三步是什么？为什么它们不能简单相乘？
- p99 翻倍时，你怎么在 10 分钟内判断是排队问题还是计算问题？
- chunked prefill 牺牲了什么、换来了什么？
- 什么条件下 P/D 分离才划算？KV cache 传输量怎么估？

## 参考资料

- 本专题题解的主要来源为 Outcome School 的博客与推理方向的原始论文（vLLM、Sarathi-Serve、DistServe、Speculative Decoding、Megatron-LM、ZeRO 等），每题文末列出具体文章、作者与链接。
- 题面与公司分布来自本仓库的 [README.md](../../README.md) 与 [README.zh-CN.md](../../README.zh-CN.md)。
