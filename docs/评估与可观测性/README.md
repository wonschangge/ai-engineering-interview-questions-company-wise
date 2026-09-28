---
type: topic
id: evaluation
title: 评估与可观测性
order: 6
summary: 从 LLM-as-judge 的偏差与校准、eval set 的构建、幻觉度量到回归门禁、benchmark 污染、traces 与在线 A/B，覆盖「怎么知道系统到底行不行」这套能力。
total: 10
updated: 2026-09-28
---

# 评估与可观测性

这是「跨公司高频问题」里的第六个专题，共 10 道题，**已全部上线**。它回答的是**怎么知道自己做得对**：
分数该信谁、没有标注怎么办、幻觉怎么量、什么改动能上线、线上到底发生了什么。

## 题解清单

| # | 题目 | 一句话看点 |
| --- | --- | --- |
| 01 | [设计一个 LLM-as-judge 评估。它有哪些已知偏差，你如何纠正？](01-设计一个%20LLM-as-judge%20评估：它有哪些已知偏差以及如何纠正.md) | 把 judge 当有系统偏差的仪器：位置/长度/自我增强偏差的纠正手段与校准流程 |
| 02 | [在没有标注好的 ground truth 且专家成本高昂时，你如何构建 eval set？](02-在缺少%20ground%20truth%20且专家成本高昂时如何构建%20eval%20set.md) | 「可判定的替代信号」+ 合成数据 + 少量人工校正（PPI）；100–500 条起步的分层设计 |
| 03 | [在生产环境 RAG 系统中，你如何检测并度量幻觉？](03-在生产环境%20RAG%20系统中如何检测并度量幻觉.md) | 接地失败 vs 事实性幻觉；原子事实分解、自一致性采样、搜索增强评估与线上症状指标 |
| 04 | [设计一道回归门禁（regression gate），用来决定 prompt 或模型改动能否上线。](04-设计一道回归门禁来决定%20prompt%20或模型改动能否上线.md) | 分层测试集 + 事前阈值 + 统计判定 + flaky 水位；judge 版本必须与模型版本一起锁 |
| 05 | [为什么 benchmark 分数在提升，用户却说系统变差了？请列举原因。](05-为什么%20benchmark%20分数在提升用户却说系统变差了.md) | 代理指标与目标脱钩的七类原因；分层评测 + 线上指标仲裁 + 榜单机制偏差 |
| 06 | [什么是 benchmark 污染，你如何防范？](06-什么是%20benchmark%20污染以及如何防范.md) | 四种污染来源、六种检测手段（含 TS-Guessing）、数据/评测/流程三层防范 |
| 07 | [生产环境 LLM 系统需要哪些可观测性：traces、spans、成本、反馈？](07-生产环境%20LLM%20系统需要哪些可观测性.md) | 四层遥测（metrics/logs/traces/质量与成本）；trace 必须带版本标注才能归因 |
| 08 | [你如何在生产环境管理 prompt 版本与回滚？](08-如何在生产环境管理%20prompt%20版本与回滚.md) | prompt 是代码：发布版本（release bundle）+ 原子切换 + 灰度 + 缓存 key 失效 |
| 09 | [设计在线评估：你记录什么日志、采样什么、对什么做 A/B？](09-设计在线评估：记录什么日志、采样什么、对什么做%20A-B.md) | 离线→影子→灰度→全量的阶梯；随机化单位、护栏指标与「不能拿来做实验」的红线 |
| 10 | [与评估单个模型回复相比，你会如何评估一个 agent？](10-与评估单个模型回复相比如何评估一个%20agent.md) | 结果/过程/成本三层指标 + pass^k 一致性；可执行环境与终态校验（τ-bench、WebArena、SWE-bench、GAIA） |

## 这个专题的知识地图

按「从单点到系统、从离线到线上」组织，10 道题分成四组：

| 组 | 主题 | 题目 |
| --- | --- | --- |
| A. 怎么打分 | judge 的设计与偏差、没有标注时怎么建评测集 | 01、02 |
| B. 度量什么 | 幻觉、agent 的轨迹与一致性 | 03、10 |
| C. 怎么守住 | 回归门禁、prompt 版本与回滚、在线实验 | 04、08、09 |
| D. 怎么不被骗 | benchmark 与目标的脱钩、污染与防范 | 05、06 |
| 基础设施 | traces/spans/成本/反馈的四层遥测 | 07 |

**建议顺序**：`01 → 02 → 03` 是「把测量做出来」（judge 校准 → 评测集 → 专项指标）；
`04 → 08 → 09` 是「把测量接进研发与发布流程」（门禁 → 版本 → 在线实验）；
`05 → 06` 是「别被数字骗了」（代理指标失配 → 数据污染）；
`07` 与 `10` 分别补上基础设施与 agent 这两个维度。

## 谁在问这个专题

按 `README.md` 的统计，本专题的题目出现在 11 家公司的面试中：**Perplexity、Anthropic、Cognition、智谱 AI、Scale AI（各 2 题）**
是最集中的提问方——Perplexity 问 judge 与在线评估，Anthropic 问幻觉度量与回归门禁，Cognition 问 benchmark 与用户感受的背离，
Scale AI 与智谱 AI 问污染与 agent 评估；**Cohere、Harvey** 问没有标注时怎么建 eval set（法律场景专家贵）；
**OpenAI、Cursor、阿里巴巴、Moonshot AI** 分别问幻觉检测与 agent 评估。

## 自测清单

能不看资料回答下面这些问题，说明这个专题过关了：

- LLM-as-judge 的四类已知偏差是什么？位置偏差和长度偏差分别怎么纠正？
- 怎么证明一个 judge 可以用来做上线门禁？（一致性、可重复性、金标准回归）
- 没有 ground truth、专家又贵，你的 eval set 从哪来？少量人工标注怎么用在刀刃上？
- 「接地失败」和「事实性幻觉」的区别是什么？各自的度量手段有什么不同？
- 为什么不能只用 LLM judge 测幻觉？自一致性采样的原理是什么？
- 回归门禁的裁决规则怎么写？为什么必须先测出 flaky 水位？
- benchmark 涨了但用户说变差，列出至少五类原因与诊断顺序。
- benchmark 污染的四种来源是什么？TS-Guessing 在测什么？
- 一条 LLM 请求的 trace 里必须记录哪些字段？哪些绝对不能记？
- prompt 版本回滚要连带回滚什么？为什么语义缓存的 key 必须包含 prompt 版本？
- 在线 A/B 的随机化单位该选请求、会话还是用户？哪些东西不能拿来做实验？
- 评估 agent 时，结果层、过程层、成本层分别看什么指标？pass^k 为什么比平均成功率更有信息量？

## 参考资料

- 本专题题解的主要来源为 Outcome School 的博客与评估方向的原始论文/官方文档（MT-Bench、Prometheus 2、ARES、Ragas、FActScore、
  SelfCheckGPT、SAFE、LiveBench、TS-Guessing 的污染研究、The Leaderboard Illusion、Chatbot Arena、τ-bench、WebArena、GAIA、
  SWE-bench，以及 OpenTelemetry GenAI 语义约定、openai/evals、promptfoo、LangSmith、Arize Phoenix），每题文末列出具体文章、作者与链接。
- 与评估直接相关的另外两处题解：RAG 的 [评估流水线](../RAG%20与检索/04-如何评估%20RAG%20流水线的质量：分别评估%20retrieval%20与%20generation.md) 与 Agent 的
  [循环终止与可靠性](../Agent%20与工具调用/09-什么能让%20agent%20循环正确终止以及如何限制成本与步数.md)。
- 题面与公司分布来自本仓库的 [README.md](../../README.md) 与 [README.zh-CN.md](../../README.zh-CN.md)。
