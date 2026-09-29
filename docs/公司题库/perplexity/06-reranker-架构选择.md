---
type: question
id: perplexity-06
company: Perplexity
topic: rag
order: 6
question: 讨论 reranker 的架构选择：cross-encoder、ColBERT、基于 LLM 的方案。
question_en: Discuss reranker architecture choices: cross-encoder, ColBERT, and LLM-based approaches.
asked_at: []
level: 高阶
tags: [reranker, cross-encoder, ColBERT, LLM 重排, 算力预算]
sources:
  - title: ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT（延伸）
    url: https://arxiv.org/abs/2004.12832
    author: Khattab & Zaharia
    published: 2020-04-27
  - title: Passage Re-ranking with BERT（延伸）
    url: https://arxiv.org/abs/1901.04085
    author: Nogueira & Cho
    published: 2019-01-31
  - title: RankGPT: Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents（延伸）
    url: https://arxiv.org/abs/2304.09542
    author: Sun et al.
    published: 2023-04-19
related: [perplexity-05, perplexity-07, perplexity-09, rag-02, openai-16]
updated: 2026-09-28
---

## 一句话答案

> **三种方案的本质差别是"算力花在哪里"**：**ColBERT 把算力放在离线（文档向量预计算）、cross-encoder 放在在线（逐篇前向）、LLM 放在在线且极贵**——**所以"能负担多少候选"差三个数量级。**
> **★ 量化一：三种方案的算力与延迟（50 个候选）**
> （H100 bf16 989 TFLOPs、MFU 40% → 有效 **396 TFLOPs**）
> | 方案 | **单次算力** | **延迟** | 可行？ |
> | --- | --- | --- | --- |
> | **cross-encoder（110M，50 篇 × 512 token）** | **5.63 TF** | **14.2 ms** | **可行** |
> | **ColBERT（在线只算 MaxSim）** | **≈0** | **≈0 ms** | **可行** |
> | **LLM reranker（7B，50 篇 × 1000 token）** | **700 TF** | **1769.5 ms** | **不可行** |
> | **LLM reranker（70B，50 篇 × 1000 token）** | **7000 TF** | **17694.6 ms** | **不可行** |
> **读法**：**ColBERT 的在线算力几乎可忽略（文档向量离线算）**、**cross-encoder 约 14.2ms**，**而 70B 的 LLM reranker 要 17.7 秒**——**所以 LLM rerank 只能用于 top-5 量级**。
> **★ 量化二：反推"每种方案能负担多少候选"（300ms 预算）**
> | 方案 | 单篇算力 | **300ms 能处理** | 结论 |
> | --- | --- | --- | --- |
> | **cross-encoder（110M，512 token）** | 112.64 GF | **1,054 篇** | **够用** |
> | **LLM reranker（7B，1000 token）** | 14,000 GF | **8 篇** | **只能重排 top-8** |
> | **LLM reranker（70B，1000 token）** | 140,000 GF | **0.8 篇** | **一篇都不行** |
> **读法**：**300ms 预算下 cross-encoder 能处理约 1,054 篇、7B LLM 只能 8 篇、70B 连 1 篇都不到**——**所以"LLM 重排"必须配"先粗排到 top-10"**（**这是两级架构的量化依据**）。
> **★ 量化三：索引体积（ColBERT 的多向量代价）**
> | 方案 | 每文档 | **10 亿文档** |
> | --- | --- | --- |
> | **单向量（768 维 fp32）** | **3.0 KB** | **3.1 TB** |
> | 单向量（768 维 int8） | 0.8 KB | 0.8 TB |
> | **ColBERT（32 × 128 fp32）** | **16.0 KB** | **16.4 TB** |
> | ColBERT（32 × 128 int8） | 4.0 KB | 4.1 TB |
> **读法**：**ColBERT 的索引是单向量的约 5.3 倍**（**fp32：16.4 vs 3.1 TB；int8：4.1 vs 0.8 TB，同为 5 倍**）——**量化到 int8 能整体降 4 倍，但相对单向量仍是 5 倍**。**所以"用 ColBERT"的代价是存储与内存带宽**。
> **★ 三者的定位**：
> | 方案 | **算力在哪** | **质量** | **适用** |
> | --- | --- | --- | --- |
> | **双塔（bi-encoder）** | **离线** | 基准 | **召回**（**串 [[perplexity-09]]**） |
> | **ColBERT** | **离线为主 + 在线 MaxSim** | **中上** | **大规模、延迟敏感** |
> | **cross-encoder** | **在线（逐篇）** | **高** | **top-50–100 精排** |
> | **LLM reranker** | **在线（极贵）** | **最高** | **top-5–10** |
> **读法**：**"算力放在离线还是在线"是第一判据**——**离线能做的就别放到在线**（**这是 ColBERT 的核心思想**）。
> **★ 推荐架构（四级）**：
> | 级 | 方法 | 候选数 |
> | --- | --- | --- |
> | ① | **混合召回（BM25 + 稠密）** | **1,000+** |
> | ② | **ColBERT 或轻量 cross-encoder** | **100** |
> | ③ | **cross-encoder（110M）** | **20** |
> | ④ | **LLM reranker（7B）** | **5** |
> 一句话判据：**"先算'每篇的算力' → 用延迟预算反推'能处理多少候选' → 算力尽量放离线 → 四级漏斗：召回 1000 → 100 → 20 → 5"**。

## 面试官在考什么

- **★ 是否算"每篇算力"**：**能否用 FLOPs 反推候选数**（本机：**1,054 / 8 / 0.8**）——**这是本题的分水岭**。
- **★ 是否知道 ColBERT 的算力在离线**：**能否指出"在线只算 MaxSim"**。
- **★ 是否知道 LLM 重排极贵**：**能否量化"70B 要 17.7 秒"**。
- **索引体积**：**能否指出"ColBERT 是单向量的 5 倍"**。
- **质量排序**：**能否给出"LLM ≥ cross-encoder > ColBERT > 双塔"**。
- **四级漏斗**：**能否设计"1000 → 100 → 20 → 5"**。
- **量化**：**能否指出"int8 降 4 倍但相对仍是 5 倍"**。
- **延迟预算**：**能否指出"重排的预算通常 200–300ms"**。
- **批处理**：**能否指出"cross-encoder 可以批处理，LLM 重排不行"**。
- **诚实**：**承认"LLM 重排的质量优势在部分任务上并不明显"**。

**常见错误答案**

- **只说"LLM 重排质量最好"**（**不算成本**）。
- **用 LLM 重排 50 篇**（**延迟爆炸**——**这正是 [[perplexity-05]] 的根因**）。
- **不知道 ColBERT 的算力在离线**。
- **忽略索引体积**（**10 亿文档下 16 TB**）。
- **不做分级**（**一步到位用最贵的**）。
- **不考虑批处理**（**cross-encoder 可以批，LLM 不行**）。
- **不做质量-延迟的权衡曲线**。
- **认为"新方法一定更好"**（**忽略工程约束**）。

## 原理与推导

### 1. ★ 三种方案的算力位置

| 方案 | 离线 | 在线 |
| --- | --- | --- |
| **双塔** | **文档向量** | **内积** |
| **ColBERT** | **每 token 向量** | **MaxSim（$O(L\cdot Q)$）** |
| **cross-encoder** | **无** | **每对 $(q,d)$ 一次前向** |
| **LLM** | **无** | **一次大 prefill** |

**读法**：**"离线能做的就别放到在线"**——**这是 ColBERT 的设计哲学**（**它把"token 级交互"的算力挪到了离线**）。

### 2. ★ 算力公式

$$\text{cross-encoder}=N\times2\times P_{\text{bert}}\times L\qquad\text{LLM}=2\times P_{\text{llm}}\times N\times L_{\text{doc}}$$

| 参数 | cross-encoder | LLM |
| --- | --- | --- |
| $P$ | **110M** | **7B / 70B** |
| $L$ | **512** | **1000** |
| $N$ | 50 | 50 |

**读法**：**"LLM 的 $P$ 比 BERT 大 64–640 倍"**——**所以同样处理 50 篇，算力差 100 倍以上**（**本机 700 TF vs 5.63 TF**）。

### 3. ColBERT 的 MaxSim

$$\text{score}(q,d)=\sum_{i\in q}\max_{j\in d}\ \mathbf{E}_{q_i}\cdot\mathbf{E}_{d_j}^\top$$

| 项 | 复杂度 |
| --- | --- |
| **离线** | **$O(|d|\cdot\dim)$ 每文档** |
| **在线** | **$O(|q|\cdot|d|\cdot\dim)$** |
| **本机**（$|q|{=}32,|d|{=}128$） | **$32\times128\times128\times2\approx1$ MFLOPs** |

**读法**：**"在线只是矩阵乘"**——**所以它的延迟几乎可忽略**（**代价是索引大 5 倍**）。

### 4. 索引体积

| 方案 | 每文档 | 10 亿 |
| --- | --- | --- |
| **单向量 fp32** | **3.0 KB** | **3.1 TB** |
| **ColBERT fp32** | **16.0 KB** | **16.4 TB** |
| **ColBERT int8** | **4.0 KB** | **4.1 TB** |

**读法**：**"多向量 = 多 5 倍存储"**——**而"内存带宽"也会成为瓶颈**（**因为检索要读更多字节**）。

### 5. 质量-成本曲线

| 方案 | 相对质量 | 相对成本 |
| --- | --- | --- |
| **双塔** | 1.00 | **1×** |
| **ColBERT** | **1.05–1.10** | **5×（存储）** |
| **cross-encoder** | **1.15–1.25** | **100×（算力）** |
| **LLM** | **1.20–1.30** | **10,000×（算力）** |

**读法**：**"质量提升是递减的、成本是指数增长的"**——**所以"到哪一级停"是业务决策**（**本机的四级漏斗是最实用的折中**）。

### 6. 四级漏斗

| 级 | 方法 | 候选 | 延迟预算 |
| --- | --- | --- | --- |
| ① | **混合召回** | **1,000+** | **100–150 ms** |
| ② | **ColBERT / 轻量 CE** | **100** | **20–30 ms** |
| ③ | **cross-encoder 110M** | **20** | **15–30 ms** |
| ④ | **LLM reranker 7B** | **5** | **150–200 ms** |
| **合计** | —— | **5** | **约 300 ms** |

**读法**：**"每一级只处理上一级的 1/10–1/5"**——**而总延迟控制在 300ms 内**（**串 [[perplexity-05]] 的预算分解**）。

## 数值与代码验证

### 表 1：算力与延迟、能负担的候选数、索引体积（见代码输出）

| 项 | 数值 |
| --- | --- |
| 见输出 | 见输出 |

### 可运行代码

```python
print('① 三种 reranker 的算力与延迟（对 50 个候选）')
PEAK=989e12; MFU=0.40   # H100 bf16
print(f'  设定：H100 bf16 {PEAK/1e12:.0f} TFLOPs、MFU {MFU:.0%} -> 有效 {PEAK*MFU/1e12:.0f} TFLOPs')
print(f'  {"方案":<34} {"单次算力":>14} {"延迟":>10} {"可行?":<14}')
rows=[]
# cross-encoder：BERT-base 110M，每个候选 512 token
for name,flops,note in (
  ('cross-encoder（110M，50 篇 x 512 token）', 50*2*110e6*512, '**逐篇前向**'),
  ('ColBERT（在线只算 MaxSim）', 50*32*128*2, '**文档向量离线算**'),
  ('LLM reranker（7B，50 篇 x 1000 token）', 2*7e9*50*1000, '**一次 prefill**'),
  ('LLM reranker（70B，50 篇 x 1000 token）', 2*70e9*50*1000, '**一次 prefill**')):
    lat=flops/(PEAK*MFU)
    ok='**可行**' if lat<0.3 else ('**勉强**' if lat<1.0 else '**不可行**')
    rows.append((name,flops,lat,ok))
    print(f'  {name:<34} {flops/1e12:>11.2f} TF {lat*1000:>8.1f} ms {ok:<14}')
print('  读法：**ColBERT 的在线算力几乎可忽略（文档向量离线算）**；**cross-encoder 约 14.2ms**；')
print('        而**70B 的 LLM reranker 要 17.7 秒（完全不可行）** —— 所以 LLM rerank **只能用于 top-5 量级**')
print()
print('② 反推"每种方案能负担多少候选"（给定 300ms 的 rerank 预算）')
BUDGET=0.30
print(f'  {"方案":<34} {"单篇算力":>12} {"300ms 能处理":>13} 说明')
for name,per in (('cross-encoder（110M，512 token）',2*110e6*512),
                 ('LLM reranker（7B，1000 token）',2*7e9*1000),
                 ('LLM reranker（70B，1000 token）',2*70e9*1000)):
    n=BUDGET*PEAK*MFU/per
    print(f'  {name:<34} {per/1e9:>9.2f} GF {n:>13.0f} 篇 '
          f'{"**够用**" if n>=50 else ("**只能重排 top-"+str(int(n))+"**" if n>=1 else "**一篇都不行**")}')
print('  读法：**300ms 预算下 cross-encoder 能处理约 1,054 篇、7B LLM 只能处理 8 篇、70B 连 1 篇都不到（0.8 篇）** ——')
print('        所以**「LLM 重排」必须配「先粗排到 top-10」**（**这是两级架构的量化依据**）')
print()
print('③ 索引体积：ColBERT 的"多向量"代价')
DOCS=1_000_000_000
print(f'  {"方案":<28} {"每文档":>12} {"10 亿文档":>12} 说明')
for name,per,note in (('单向量（768 维 fp32）',768*4,'**3.1 GB/百万**'),
                      ('单向量（768 维 int8）',768*1,'量化后 1/4'),
                      ('ColBERT（32 x 128 fp32）',32*128*4,'**16.4 KB/文档**'),
                      ('ColBERT（32 x 128 int8）',32*128*1,'量化后 1/4')):
    print(f'  {name:<28} {per/1024:>9.1f} KB {per*DOCS/1e12:>9.1f} TB {note}')
print('  读法：**ColBERT 的索引是单向量的约 5.3 倍**（fp32：16.4 TB vs 3.1 TB；int8：4.1 TB vs 0.8 TB，**同为 5 倍**）——')
print('        量化到 int8 能**整体降 4 倍**（16.4 → 4.1 TB），但**相对单向量仍是 5 倍**；')
print('        所以"用 ColBERT"的代价是**存储与内存带宽**（**这是它的主要权衡**）')
```

预期输出要点（实跑）：① **算力与延迟**（50 候选）：cross-encoder **5.63 TF / 14.2 ms**、ColBERT **≈0 / ≈0 ms**、7B LLM **700 TF / 1769.5 ms**、70B LLM **7000 TF / 17694.6 ms**；② **300ms 能处理**：cross-encoder **1,054 篇**、7B **8 篇**、70B **0.8 篇**；③ **索引体积**（10 亿文档）：单向量 fp32 **3.1 TB**、ColBERT fp32 **16.4 TB**、int8 **4.1 TB / 0.8 TB**。

## 常见追问

- **追问**：LLM 重排到底值不值？
  - 要点：**看"它用在多少篇上"**：① **top-5 用 7B 重排**（**约 177ms**，**可接受**）；② **top-50 用 7B**（**1.77 秒**，**不可接受**）；③ **所以"值不值"取决于"用在漏斗的哪一级"**。**读法**：**"LLM 重排只适合最后一跳"**——**这是本机量化最直接的结论**。
- **追问**：为什么 cross-encoder 能批处理而 LLM 不能？
  - 要点：**因为计算形态不同**：① **cross-encoder 是 $N$ 个独立的前向**（**可拼成一个 batch**）；② **LLM 重排通常把 50 篇拼进一个 prompt**（**是一次长 prefill，天然是一个序列**）；③ **也可以用"50 个独立请求"批处理**（**但每个都要完整 prefill，不省**）。**读法**：**"LLM 重排的算力是 $O(N\times L)$ 而 cross-encoder 是 $O(N\times L)$"**——**两者量级相同但 $P$ 差 64 倍**。
- **追问**：ColBERT 的延迟真的可忽略吗？
  - 要点：**算力可忽略，但内存带宽不是**：① **每文档 16 KB**（**50 篇 = 800 KB**）；② **从内存读 800 KB 约 0.1ms**（**100 GB/s**）；③ **但如果是 1,000 篇候选**（**16 MB**）**就要 0.16ms**（**仍可忽略**）。**读法**：**"ColBERT 的瓶颈是存储与内存，不是算力"**——**这与它"算力放离线"的设计一致**。
- **追问**：怎么评估 reranker 的质量提升？
  - 要点：**三条**：① **nDCG@10 或 MRR**（**排序指标**）；② **端到端指标**（**答案质量**——**因为 reranker 是中间环节**）；③ **A/B 实验**（**最终判据**）。**读法**：**"reranker 的指标提升不等于答案质量提升"**——**因为 LLM 可能对排序不敏感**（**串 [[perplexity-07]]**）。
- **追问**：如果只有 CPU 呢？
  - 要点：**结论完全改变**：① **cross-encoder 在 CPU 上要慢 50–100 倍**（**50 篇约 1 秒**）；② **ColBERT 的在线部分仍然快**（**矩阵乘**）；③ **LLM 完全不可行**。**读法**：**"硬件约束会改变架构选择"**——**所以"目标环境是什么"是必须先问的**（**串 [[gdm-19]]**）。
- **追问**：这道题与"TTFT 恶化"有什么关系？
  - 要点：**它正是那个根因**：① **[[perplexity-05]] 的根因是"重排从 cross-encoder 20 篇变成 LLM 50 篇"**；② **本机的量化解释了为什么它会涨 1300ms**（**7B 处理 50 篇要 1.77 秒**）；③ **所以两道题共享同一个算力模型**。**读法**：**"把算力模型算清楚，就能预判发版的影响"**——**这是这两道题的联系**。

## 相关题目

- [[perplexity-05]]：TTFT 从 1.2s 恶化到 3s——**本机量化的直接应用**。
- [[perplexity-07]]：50 个候选只留 10 个——**"如何取舍"**。
- [[perplexity-09]]：1000 亿网页的检索流水线——**召回级的架构**。
- [[rag-02]]：向量索引与倒排——**双塔与 ANN**。
- [[openai-16]]：企业搜索系统——**混合检索的实测**。

## 参考资料与归属

- **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT（延伸）** —— Khattab & Zaharia，2020-04-27：<https://arxiv.org/abs/2004.12832>。**延迟交互（late interaction）与 MaxSim** 是本篇第 1、3 节的依据。
- **Passage Re-ranking with BERT（延伸）** —— Nogueira & Cho，2019-01-31：<https://arxiv.org/abs/1901.04085>。**cross-encoder 精排** 是本篇第 2 节的依据。
- **RankGPT: Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents（延伸）** —— Sun et al.，2023-04-19：<https://arxiv.org/abs/2304.09542>。**LLM 重排的质量与成本** 是本篇第 5 节的依据。
- **延伸来源说明**：表 1 与可运行代码中的全部数值（H100 989 TFLOPs、MFU 40%、BERT-base 110M、512 token、50 篇候选、ColBERT 32×128、7B/70B LLM 与 1000 token、768 维单向量、10 亿文档）都是**按本仓库统一常数与本机公式计算的显式假设**；**算力、延迟、候选数、索引体积都是直接计算**（**可复现**）。**⚠️ "MFU 40%"是理想值**——**真实系统的 cross-encoder 批处理效率、LLM prefill 效率都会低于此**（**所以真实延迟会比本机估算更高**）；**"质量相对值 1.00–1.30"是文献中的典型区间**（**不是本机实测**）。**可迁移的结论是"算力尽量放离线、用延迟预算反推候选数、四级漏斗"**，**不是具体数字**。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
