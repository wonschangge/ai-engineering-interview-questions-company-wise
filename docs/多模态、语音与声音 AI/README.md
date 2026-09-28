---
type: topic
id: multimodal
title: 多模态、语音与声音 AI
order: 8
summary: 从 VLM 的视觉接入方式、视频的 token 与显存账，到实时语音的延迟预算、barge-in、级联与原生之争、ASR/TTS 评测、code-switching、流式 TTS 与 diarisation，覆盖模态工程的全链路。
total: 10
updated: 2026-09-28
---

# 多模态、语音与声音 AI

这是「跨公司高频问题」里的第八个专题，共 10 道题，**已全部上线**。它回答的是**文本之外的那几种模态怎么进模型、怎么算账**：
图像变成多少 token、视频为什么是上下文问题、语音延迟花在哪、副语言信息值不值得为它放弃可控性。

## 题解清单

| # | 题目 | 一句话看点 |
| --- | --- | --- |
| 01 | [视觉语言模型是如何把图像输入到 LLM 中的：projector、cross-attention，还是原生 token？](01-视觉语言模型如何把图像输入到%20LLM.md) | 三条路线的机制与代价；视觉 token 数线性于像素数，直接吃上下文与 KV 预算 |
| 02 | [从图像扩展到视频时，会发生哪些变化？](02-从图像扩展到视频时会发生哪些变化.md) | token = 帧数 × 每帧 token；60 万 token 的 KV cache 约 183 GiB，所以是系统问题 |
| 03 | [为实时语音 agent 做 latency 预算：VAD、ASR、LLM、TTS、网络。时间都花在哪里了？](03-为实时语音%20agent%20做%20latency%20预算.md) | 六段相加的算式；VAD 的静音确认是最大隐性成本，出路是流水线化而非各段压极限 |
| 04 | [为语音 agent 设计 barge-in / 打断处理机制。](04-为语音%20agent%20设计%20barge-in%20打断处理机制.md) | barge-in 是五个环节；背信道不该打断、AEC 防自打断、历史要截到「用户听到的位置」 |
| 05 | [级联式 ASR+LLM+TTS 与原生 speech-to-speech 的对比：请为双方各做论证。](05-级联式语音管线与原生%20speech-to-speech%20的对比.md) | 级联的可控性 vs 原生的低延迟与副语言信息；混合路线（inner monologue、按意图路由） |
| 06 | [除了 WER 之外，你如何评估 ASR 质量？在不存在唯一正确输出时，又如何评估 TTS 质量？](06-除了%20WER%20之外如何评估%20ASR%20与%20TTS%20质量.md) | WER 的四个缺陷 + 分群/实体/流式指标；TTS 用「人工受控校准 + 自动代理回归 + 下游任务仲裁」 |
| 07 | [在生产环境的 ASR 系统中，你如何处理 code-switching 与口音？](07-在生产环境的%20ASR%20系统中如何处理%20code-switching%20与口音.md) | 语言边界无标记 + 共享音素冲突；数据/表征/适配三路手段，以及 MER 口径必须先写清 |
| 08 | [解释流式 TTS 的 chunking 与 jitter buffer 大小设定。](08-解释流式%20TTS%20的%20chunking%20与%20jitter%20buffer%20大小设定.md) | chunking 是延迟问题、buffer 是稳定性问题；buffer 大小 = 抖动分布的目标分位数 |
| 09 | [设计一个 diarisation 系统，并说明你如何归属角色，而不仅仅是聚类。](09-设计一个%20diarisation%20系统并说明如何归属角色.md) | 聚类给 SPEAKER_00/01，角色归属要回答谁是医生；DER 的三个分量与重叠语音口径 |
| 10 | [你会如何在一个索引中对图像、视频和文本构建多模态 retrieval？](10-如何在一个索引中对图像、视频和文本构建多模态%20retrieval.md) | 双塔/单塔/晚交互三条路线；ColPali 式页面图像检索绕开脆弱的版面解析 |

## 这个专题的知识地图

按「视觉 → 语音 → 音频结构 → 检索」四组排列：

| 组 | 主题 | 题目 |
| --- | --- | --- |
| 视觉理解 | 图像如何进 LLM、视频带来了什么变化 | 01、02 |
| 实时语音 | 延迟预算、打断、级联 vs 原生 | 03、04、05 |
| 语音质量与结构 | ASR/TTS 评测、code-switching 与口音、流式合成、说话人分离 | 06、07、08、09 |
| 跨模态检索 | 图像/视频/文本的统一索引 | 10 |

**建议顺序**：`01 → 02` 是视觉主线（也是与推理专题衔接最紧的一段：视觉 token 就是上下文预算）；
`03 → 04 → 05` 是语音主线（延迟 → 打断 → 架构选择，面试里常连着问）；
`06 → 07 → 09` 是「语音的质量与结构」（评测 → 鲁棒性 → 说话人）；
`08` 是流式合成的工程细节，`10` 把模态接回检索与 RAG。

与其它专题的接口：视觉 token 直接进 KV cache、与显存预算强耦合，接 **[推理与 GPU 性能的显存估算](../推理、服务与%20GPU%20性能/08-估算服务一个%2070B%20模型所需的%20GPU%20显存.md)**；
页面图像检索接 **[RAG 的表格与多栏 PDF](../RAG%20与检索/10-在%20retrieval%20流水线中如何处理表格、图表与多栏%20PDF.md)**；
原生语音的护栏缺口接 **[安全与 guardrails](../安全、安保与负责任%20AI/04-为面向消费者的助手设计%20guardrails.md)**。

## 谁在问这个专题

按 `README.md` 的统计，本专题的题目出现在 5 家公司的面试中，且高度集中：
**ElevenLabs（5 题）** 几乎包下语音侧——延迟预算、barge-in、级联 vs 原生、ASR/TTS 评测、流式 TTS；
**Meta（2 题）** 问 VLM 的视觉接入与视频扩展；**Sarvam AI（2 题）** 问延迟预算与 code-switching（与其多语言语音路线一致）；
**Abridge（2 题）** 问 code-switching 与 diarisation（医疗转写场景）；**阿里巴巴（Qwen）** 问 VLM 的视觉接入方式。

## 自测清单

能不看资料回答下面这些问题，说明这个专题过关了：

- VLM 把图像接进 LLM 的三条路线分别是什么？各自的代表工作与代价？
- 视觉 token 数由什么决定？一张图大致占多少 token、占多少 KV cache？
- 视频为什么不是「更大的图像」？举一个具体的 token 与显存算例。
- 实时语音的首音延迟由哪几段组成？哪一段最容易被忽略、为什么？
- 为什么「把每段都压到极限」不如「把链路流水线化」？
- barge-in 有哪五个必须实现的环节？背信道为什么不该打断？
- 如何区分「用户真的在插话」与「环境噪声/回声」？
- 为级联与原生各给三条论证，并说出各自的致命短板。
- WER 有哪些缺陷？ASR 还该看哪些指标？
- TTS 没有唯一正确答案，那评测组合应该怎么搭？
- code-switching 与口音是两个什么问题？各自的处理手段？
- 为什么公开的多语言模型在 code-switching 上会退化，怎么修？
- chunking 与 jitter buffer 分别优化什么？buffer 该按什么规则定大小？
- diarisation 的 DER 由哪三部分组成？「角色归属」比聚类多做了什么？
- 多模态检索的三条路线怎么选？跨模态分数能直接相加吗？

## 参考资料

- 本专题题解的主要来源为 Outcome School 的博客与多模态/语音方向的原始论文与规范（ViT、LLaVA、Flamingo、BLIP-2、Qwen2-VL、LongVILA、Video-ChatGPT、
  Moshi、AudioPaLM、Seamless、Whisper、USM、Attention-Guided Adaptation、TALCS、ISCSLP 2022 挑战、SSL 语码转换、
  CLIP、SigLIP、ImageBind、ColPali、FastSpeech 2、VITS、StyleTTS 2、pyannote.audio、EEND、说话人分离综述，以及 RFC 3550 与 Silero VAD），
  每题文末列出具体文章、作者与链接。
- 题面与公司分布来自本仓库的 [README.md](../../README.md) 与 [README.zh-CN.md](../../README.zh-CN.md)。
