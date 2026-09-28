---
type: question
id: multimodal-01
topic: 多模态、语音与声音 AI
order: 1
question: 视觉语言模型是如何把图像输入到 LLM 中的：projector、cross-attention，还是原生 token？
question_en: How do VLMs feed images into an LLM: projector, cross-attention, or native tokens?
asked_at: [Meta, 阿里巴巴（Qwen）]
level: 进阶
tags: [vlm, projector, cross-attention, 视觉-token]
sources:
  - title: Multimodal AI
    url: https://outcomeschool.com/blog/multimodal-ai
    author: Amit Shekhar (Outcome School)
    published: 2026-05-10
  - title: Decoding Vision Transformer (ViT)
    url: https://outcomeschool.com/blog/decoding-vision-transformer-vit
    author: Amit Shekhar (Outcome School)
    published: 2026-04-15
  - title: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale（延伸）
    url: https://arxiv.org/abs/2010.11929
    author: Dosovitskiy et al. (ViT, ICLR 2021)
    published: 2020-10-22
  - title: Visual Instruction Tuning（LLaVA）（延伸）
    url: https://arxiv.org/abs/2304.08485
    author: Liu et al. (NeurIPS 2023)
    published: 2023-04-17
  - title: "Flamingo: a Visual Language Model for Few-Shot Learning（延伸）"
    url: https://arxiv.org/abs/2204.14198
    author: Alayrac et al. (NeurIPS 2022)
    published: 2022-04-29
  - title: "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models（延伸）"
    url: https://arxiv.org/abs/2301.12597
    author: Li et al. (ICML 2023)
    published: 2023-01-30
related: [multimodal-02, multimodal-10, llm-internals-05, inference-serving-03]
updated: 2026-09-28
---

## 一句话答案

> 三条路线都真实存在，区别在于「视觉特征在哪一层进入 LLM」：**projector / 适配器**（LLaVA、BLIP-2）把 patch 特征映射成若干视觉 token，直接拼进 LLM 的输入序列；**cross-attention**（Flamingo）不改 LLM 权重，把 gated cross-attention 层插进已有 LM 层之间；**原生 token / 早融合**（Qwen2-VL）让视觉 token 与文本 token 从第一层起就共用同一套 attention。
> 今天的开源主流是 projector 与原生 token，因为视觉 token 进入同一序列后，KV cache、prefix caching、continuous batching 这些推理优化可以原样复用；cross-attention 的表达力不差，代价是架构与推理栈都要单独处理。
> 三条路线的账最终都落在同一处：**一张图占多少 token**。视觉 token 数与像素数成正比，吃的是同一个上下文预算——[[llm-internals-13]] 讲塞进长窗口的信息为什么未必被用好，[[inference-serving-03]] 讲这些 token 怎么变成 KV cache 的显存与带宽。

## 面试官在考什么

- 能不能给出三条路线各自的代表工作与机制，而不是笼统地说「都行」：projector 是「映射进序列」，cross-attention 是「插层」，原生 token 是「从第一层就同源」。
- 能不能把「视觉 token 数」当成一等公民来算：ViT 把图切成 16×16 patch（论文口径），分辨率翻倍 token 变 4 倍；这些 token 会进 KV cache、会和文本抢上下文。
- 能不能说清代价：projector 的定长瓶颈、cross-attention 的推理不友好、原生 token 的上下文成本。
- 知不知道 LLaVA 式训练流程为什么先对齐再指令微调，以及跳过对齐会发生什么。
- 有没有生产视角：预处理一致性、token 预算、不同分辨率的 batching 浪费、同一张图的视觉前缀缓存。

常见错误答案：

- 「VLM 就是把图像 embedding 和文本 embedding 拼起来」。拼接发生在 token 层而不是句向量层，而且拼的是「一串」token，不是「一个」向量——这决定了后面的所有成本。
- 「Flamingo 的做法已经过时，所以 cross-attention 是错的方向」。它解决的是「接一个完全冻结的强 LLM」这个问题，gated 结构加零初始化的设计动机值得讲清楚。
- 「视觉 token 不占上下文，模型会自己压缩」。恰恰相反，所有省 token 的手段（切图 tile、token 合并、定长 query）都是显式工程决策。

## 原理与推导

### 1. 共同的前半段：视觉编码器把图变成一串 patch 特征

多模态系统的通用配方是「每种模态各自的 encoder → 在共享嵌入空间里对齐 → 同一个 Transformer 推理并生成」：文本侧是 tokenizer 加 embedding，视觉侧就是下面这条 patch 化流水线，三条路线的差别只在视觉特征以什么形式、在哪一层进入语言模型。

ViT 的出发点是「把图当句子读」：把图像切成不重叠的 $P \times P$ patch（原论文取 $P=16$），每块 patch 展平成向量后做一次线性投影得到 embedding，加上位置编码，再过 Transformer encoder。$N \times N$ 像素的图像得到的 patch 数是

$$T = \left(\frac{N}{P}\right)^2$$

代码里这次的线性投影用 kernel size $P$、stride $P$ 的卷积实现，等价于「切块 + 投影」。原论文 ViT-Base 的口径是 224×224 输入、16×16 patch，于是 $14 \times 14 = 196$ 个 patch，加 1 个可学习的 CLS token 共 197 个 token（分类读 CLS 的输出，VLM 里通常只把 patch token 送进 LLM）；ViT-Base 约 86M 参数、12 层。核心结论只有一句：**token 数正比于像素数**，分辨率翻倍就是 4 倍 token。CLIP 的 ViT-L/14 换成 $P=14$：224² 得 $16 \times 16 = 256$ 个 patch，336² 得 $24 \times 24 = 576$ 个 patch——LLaVA 系列用的正是后者。

三条路线在这之后才分叉。

### 2. 路线一：projector / 适配器（LLaVA）

机制最直白：拿视觉编码器输出的 patch 特征矩阵 $X \in \mathbb{R}^{T \times d_v}$，过一个可训练的小模块 $g(\cdot)$ 映射到 LLM 的词嵌入维度 $d_{lm}$，得到 $T$ 个「视觉 token」，在 prompt 里图像该出现的位置把它们与文本 token 拼成同一条序列送进 LLM（实现上通常先按模板把图像位置写成占位符，再在 tokenize 之后用视觉 token 填充该位置）：

$$H_{\text{vis}} = g(X) \in \mathbb{R}^{T \times d_{lm}}, \qquad \text{input} = [\,H_{\text{vis}} \,;\, E_{\text{text}}\,]$$

LLaVA 的原始版本里 $g$ 就是一个线性层（把 CLIP ViT-L/14 的 1024 维映射到 LLM 的隐层维度），训练时冻结视觉编码器与 LLM，只训这个投影；LLaVA-1.5 起换成两层 MLP + GELU，「MLP projector」说的就是后者。论文口径（arXiv:2304.08485 摘要）：在 GPT-4 生成的合成多模态指令跟随数据上取得相对 GPT-4 的 **85.1% 相对分**（relative score，评测方式是让 GPT-4 对模型回复打分，不是绝对准确率）；在 ScienceQA 上微调后，「LLaVA + GPT-4」协同达到 **92.53%** 的新 SOTA。这两个数字常被当成「小模型打赢大模型」来引用，要注意它们的口径差别很大。

BLIP-2 把「适配器」这个想法推到极致：它不投影全部 patch，而是让一组**可学习的 query** 通过 cross-attention 从冻结的图像编码器输出里抽取信息，这组结构叫 Q-Former（32 个 query），输出是**定长**的视觉表示，再线性映射进冻结 LLM 的输入空间。因为图像编码器与 LLM 都冻结，只训这个轻量桥接模块，论文口径（arXiv:2301.12597 摘要）：zero-shot VQAv2 上比 Flamingo-80B 高 **8.7%**，同时**可训练参数少 54 倍**。

这个对比是本题的核心张力：定长 query 让成本与分辨率解耦（任何图都是 32 个 token），但也把「图里有多少细节」这件事压成了一个固定的信息瓶颈——文档里的细字、密集表格这种需要分辨率的任务会先掉在这里。

### 3. 路线二：cross-attention（Flamingo）

Flamingo 完全不改 LLM 的权重，也不把视觉 token 塞进文本序列，而是**在已有的 LM 层之间插入新的层**：

- **Perceiver Resampler** 把变长的视觉输入压成定长的 latent（论文用 64 个 latent），这样图像与视频可以共用同一条下游通路；
- **gated cross-attention** 层插在冻结 LM 的层之间，用 tanh 门控控制新层对主干的注入强度，门控参数**初始化为 0**——训练第一步模型的输出与原始 LM 完全一致，视觉信息是被「逐渐打开」的，这让冻结主干的训练稳定下来。

这条路线的卖点是不动底座、易插拔、能接入一个已经完全训好的强 LLM，并且天然支持任意交错的图文序列（交错网页数据是 Flamingo 的 few-shot 能力的来源）。论文口径（arXiv:2204.14198 摘要）：单模型在 few-shot 设置下拿到新的 SOTA，在**多个 benchmark 上超过用几千倍任务数据微调的模型**。

代价在工程侧：一是架构复杂，插入层意味着参数量、显存与并行切分都要单独设计；二是新模态有自己的 K/V，标准推理栈里那条「所有 token 共享一条 KV cache、按 block 分页管理」的路径用不上，prefix caching 与 PagedAttention 这类优化需要额外改造才能覆盖它。

### 4. 路线三：原生 token / 早融合（Qwen2-VL）

原生 token 的意思是：视觉 token 与文本 token 走**同一套 attention、同一套位置编码、同一套并行与缓存策略**，从第一层起就没有区别。Qwen2-VL 是这条路线当前最常被引用的代表（arXiv:2409.12191 摘要）：

- **Naive Dynamic Resolution**：按图像的原始分辨率产生**不同数量**的视觉 token，不再把图缩放到固定尺寸。论文把这一条与「更贴近人类感知」一起作为卖点，并说明图像与视频采用统一范式处理。
- **M-RoPE（Multimodal Rotary Position Embedding）**：把位置编码分解到时间、高度、宽度三个维度，让文本、图像、视频的位置信息在同一套机制里融合。
- 论文同时报告了 2B / 8B / 72B 三个规模上的缩放规律研究。

这一步换来的是架构上的干净：flash attention（[[llm-internals-05]]）、continuous batching、paged KV cache、prefix caching 全部原样可用。代价也直接——**token 数量不再由架构决定，而是由用户上传的图决定**，服务端必须自己设预算。

### 5. 把三条路线的账放在一张表里

| 维度 | projector（LLaVA / BLIP-2） | cross-attention（Flamingo） | 原生 token（Qwen2-VL） |
| --- | --- | --- | --- |
| 视觉特征进入位置 | LLM 输入 embedding 层 | 插在 LLM 层与层之间 | 与文本同序列、同层 |
| 是否改 LLM 权重 | 可冻可解冻：BLIP-2 全程冻结 LLM，LLaVA 第二阶段解冻 | 否（冻结主干） | 是（同序列联合训练） |
| 视觉 token 数 | 由 projector 决定（BLIP-2 定长 32） | 不进序列，压成定长 latent（64） | 由分辨率决定，可变 |
| 推理栈复用 | 好（就是普通 decoder） | 差（需单独实现插入层） | 最好（与纯文本 LLM 一致） |
| 上下文占用 | 中（576 量级） | 不占文本序列 | 高（可到几万） |
| 主要风险 | 重采样信息瓶颈 | 架构与推理复杂度 | 上下文与 KV 显存 |

### 6. 训练流程：对齐 → 指令微调 →（可选）偏好对齐，以及为什么顺序不能反

1. **对齐阶段（alignment）**：冻结视觉编码器与 LLM，只训 projector，数据是图文对（caption 类）。目的是让视觉 token 落进 LLM 已经学好的语义空间里，而不是让 LLM 去适应一堆噪声。
2. **指令微调阶段（instruction tuning）**：视觉编码器继续冻结，解冻 LLM（或只加 LoRA，见 [[finetuning-04]]），用多模态指令数据训练，让模型学会按指令使用视觉信息。
3. **可选的偏好对齐**：在多模态偏好数据上做偏好优化，对齐有用性/无害性等目标（见 [[finetuning-02]]）。

顺序不能反的原因很具体：projector 是随机初始化的，第一阶段之前它输出的是无意义向量；此时若直接联合训练，LLM 会收到一批「看起来像 token、实际是噪声」的输入并被拉离预训练分布，表现为语言能力回退、训练不稳定。先对齐把噪声变成有结构的信号，联合训练才安全。

## 数值与代码验证

### 1. 视觉 token 数：正比于像素数

| 配置 | patch 切法 | patch 数 | token 数（ViT 侧） |
| --- | --- | --- | --- |
| ViT-Base（原论文） | 224²，16×16 | $14^2 = 196$ | 197 |
| CLIP ViT-L/14 @224 | 224²，14×14 | $16^2 = 256$ | 256 |
| CLIP ViT-L/14 @336（LLaVA 式） | 336²，14×14 | $24^2 = 576$ | 576 |
| 16×16 patch 切 1024² | 1024²，16×16 | $64^2 = 4096$ | 4096 |
| 16×16 patch 切 4K（3840×2160） | 3840×2160，16×16 | $240 \times 135 = 32{,}400$ | 32,400 |

表里的 token 数按 ViT 侧的序列长度算，ViT-Base 的 197 含 1 个 CLS；VLM 通常只把 patch token 送进 LLM，LLaVA 的 576 就是纯 patch 数，计不计 CLS 只差 1 个 token，不影响量级。

关键比值：边长翻倍 → token 数 ×4（1024² 的 4096 是 512² 的 1024 的 4 倍）。这条线性关系是后文所有显存与延迟数字的来源。

### 2. KV cache：用 [[llm-internals-02]] 的口径复算

口径统一为 LLaMA-3-70B：80 层、GQA 8 个 KV 头、head_dim 128、bf16，单 token KV 为 $2 \times 80 \times 8 \times 128 \times 2$ B $= 327{,}680$ B $= 320$ KiB（与 [[llm-internals-02]]、[[inference-serving-03]] 一致）。视觉 token 与文本 token 走同一套 attention，就按同一口径计费：

| 场景 | 视觉 token | 单序列 KV 增量 | 折合 80 GB 卡（十进制口径） |
| --- | --- | --- | --- |
| LLaVA 单图 336² | 576 | 180 MiB | 0.0024 张 |
| 一分钟视频抽 8 帧（每帧 576） | 4,608 | 1.41 GiB | 0.019 张 |
| 一分钟视频抽 64 帧 | 36,864 | 11.25 GiB | 0.15 张 |
| 单图 4K 原分辨率切 16×16 | 32,400 | 9.89 GiB | 0.13 张 |
| 16×16 patch 切 4096² 的极端情况 | 65,536 | 20.0 GiB | 0.27 张 |
| 长视频 544 帧（每帧 256 token，长视频输入的常见量级） | 139,264 | 42.5 GiB | 0.57 张 |

最后一行说明问题不在「单请求能不能放下」，而在**并发**：42.5 GiB 的单序列 KV 意味着这张卡上再也放不下第二条序列。[[inference-serving-03]] 的 PagedAttention 解决的是碎片，不解决总量；总量只能靠减 token、量化 KV 或加卡。这也是 [[multimodal-02]] 里视频场景的核心矛盾。

### 3. prefill 算力：与文本同一套公式

沿用仓库既有口径（[[inference-serving-15]]）：prefill 的主导项是 $2 \cdot P \cdot S$，$P = 70\text{e}9$、H100 SXM bf16 稠密算力 989 TFLOPs、8 卡 TP、MFU 0.45，则 $S = 8192$ 时约 322 ms。视觉 token 是序列的一部分，按同一条线性关系外推：

| $S$（视觉 token 为主） | $2PS$ | 8×H100 @45% MFU |
| --- | --- | --- |
| 576 | 80.6 TFLOPs | 22.6 ms |
| 4,608 | 645 TFLOPs | 181 ms |
| 36,864 | 5.16 PFLOPs | 1.45 s |
| 139,264 | 19.5 PFLOPs | 5.48 s |

这张表算的是权重矩阵乘法项 $2PS$，是 prefill 的下界：真实延迟还要加上注意力的二次项（$\propto S^2$，长序列时不可忽略）、KV cache 写入与通信开销，所以长视频那一行的实测值会比 5.48 s 更高。

同一台机器上，纯文本 8k 上下文的首 token 延迟是几百毫秒量级，一张图只加 22.6 ms——**视觉编码本身不是瓶颈**。按 CLIP ViT-L/14 @336 的规模估算（24 层、$d=1024$、MLP 4096、577 个 token），每 token 每层约 $12d^2$ 的主干矩阵乘，整塔约 $12 \times 1024^2 \times 577 \times 24 \approx 174$ GMACs（按 2 FLOPs/MAC 折合约 348 GFLOPs），在 989 TFLOPs 峰值面前约 0.35 ms。瓶颈永远在「这些 token 到了 LLM 里之后」。

### 4. 一段可运行的 token 预算代码

```python
def visual_tokens(h, w, patch, merge=1, tiles=1):
    """估算一张图进 LLM 的视觉 token 数。
    patch: 视觉编码器的 patch 边长（ViT 16 / CLIP ViT-L 14）
    merge: pixel-shuffle 之类的合并倍率，2 表示 2x2 合并成 1 个 token
    tiles: 切图 tile 数（高分辨率场景下每块 tile 独立编码）
    """
    per_tile = (h // patch) * (w // patch)
    return tiles * (per_tile // (merge ** 2))

KV_PER_TOKEN = 327680  # B, LLaMA-3-70B GQA-8 bf16, 见 llm-internals-02

for h, w, patch, merge, tiles in [
    (336, 336, 14, 1, 1),      # LLaVA 式单图
    (224, 224, 16, 1, 1),      # ViT-Base
    (336, 336, 14, 1, 4),      # 672x672 切成 4 块 336 tile 后逐块编码
    (1024, 1024, 16, 2, 1),    # 动态分辨率 + 2x2 合并
]:
    n = visual_tokens(h, w, patch, merge, tiles)
    print(f"{h}x{w} patch{patch} merge{merge} tiles{tiles}: "
          f"{n:>7,d} tok, KV {n * KV_PER_TOKEN / 1024**3:6.2f} GiB")

# 336x336 patch14 merge1 tiles1:     576 tok, KV   0.18 GiB
# 224x224 patch16 merge1 tiles1:     196 tok, KV   0.06 GiB
# 336x336 patch14 merge1 tiles4:   2,304 tok, KV   0.70 GiB
# 1024x1024 patch16 merge2 tiles1:   1,024 tok, KV   0.31 GiB
```

后两行是「把高分辨率图喂进去」的两种代价：**切图**把 token 数按 tile 数线性放大（672×672 切 4 块 336 tile，576 → 2304），**合并**把 token 数按倍率平方缩小（1024² 的 4096 → 1024）。前者保细节、后者保预算，工程上经常两个一起用。计算时注意口径：tile 数由「切图规则 + 是否补边」决定，每块 tile 还要被 resize 到编码器接受的边长，所以 $h$、$w$ 填的是**编码器的输入边长**，不是原图边长；$h$、$w$ 不能被 patch 整除时按实际补边后的网格算。

### 5. 把预算控制在窗口内的四种手段

| 手段 | 机制 | 代价 |
| --- | --- | --- |
| 动态分辨率（min/max pixels 上限） | 按原图比例产生 token，固定尺寸下的欠采样与过采样都避免 | token 数不可控，必须由服务端设上下限 |
| 降采样输入 | 缩小图，token 随像素线性降 | 细字、小目标直接消失 |
| 切图 tile | 每块 tile 独立编码后拼接 | token × tile 数；全局关系要额外一层融合 |
| token 合并 / 池化 | 相邻 patch 特征合并（如 2×2 变 1） | 空间分辨率下降，需重训或至少微调 |
| 定长 query（Q-Former 式） | cross-attention 抽固定个数 query | 与分辨率解耦，但形成信息瓶颈 |

## 常见追问

- **追问**：为什么多数开源 VLM 选 projector 而不是 cross-attention？
  - 要点：projector 之后模型就是一个普通 decoder，训练栈（FSDP/DeepSpeed）、推理栈（vLLM/SGLang 的 paged KV、continuous batching、prefix caching）、量化与并行方案全部原样复用；cross-attention 要改模型结构，也就等于要在推理引擎里为它单独写一套 kernel 与调度。表达力不是主要差距，工程量才是。
- **追问**：Q-Former 的 query 数怎么选？多了少了各会怎样？
  - 要点：query 数是「视觉信息带宽」这个旋钮。BLIP-2 用 32 个；调大能装更多细节、但每个 query 都会进 LLM 的上下文并计 KV；调小省预算但会丢信息。选择依据是任务所需的细粒度——OCR/表格类任务对定长瓶颈最敏感，这也是后来动态分辨率路线流行的原因。
- **追问**：视觉 token 能压缩吗？压缩会损失什么？
  - 要点：能。四条常见路径：降采样、切图 tile、token 合并 / 池化（pixel-shuffle 一类的空间合并）、Q-Former 式定长 query。前两条只改预处理与排布，不动模型参数；后两条改变了特征的空间对应关系与抽取方式，需要训练侧配合（至少微调 projector），否则精度会掉。压缩丢的是空间细节，坐标定位与密集文本读取是最先退化的能力。
- **追问**：视觉 token 放在文本前面还是后面？交错输入有什么特殊处理？
  - 要点：LLaVA 式做法是放在 `<image>` 占位符出现的位置，也就是「图文按阅读顺序交错」。把图像固定在 prompt 前段（system 之后、历史之前）能让它成为一段稳定前缀，prefix cache 更容易命中；交错放置则要求位置编码与注意力掩码能表达「哪段文本对应哪张图」，多图场景还需要分隔 token 来标记边界（[[multimodal-10]] 讨论检索场景下这套约定的用法）。
- **追问**：同一张图用户连续追问 5 轮，能省掉什么？预处理不一致又会怎样？
  - 要点：图像 token 是前缀，第 2 轮起它的 KV 完全没变，可以复用——这就是 prefix caching 的典型场景，省掉的是每轮重算视觉编码与 prefill 的钱（[[inference-serving-05]]）。注意前提：预处理必须逐字节一致（尺寸、归一化、切图规则、甚至 resize 的插值方式），否则前缀哈希不同，缓存必然 miss。
  - 要点：预处理与训练不一致则更糟，模型不会报错，只是静默掉点。典型的不一致包括 resize 的插值方式、归一化的均值方差、通道顺序、切图的重叠与先后顺序、pad 值、以及 token 合并的边界对齐；表现是「benchmark 还行、线上某类图很差」这种难定位的退化。
- **追问**：批处理时不同分辨率怎么处理？
  - 要点：同一 batch 内序列长度不同会 pad 到最长，短序列的算力被浪费；视觉 token 数量的方差比文本大得多（336² 的 576 与 1152² 按 16×16 patch 切出的 $72^2 = 5184$ 可能进同一批），浪费因此被放大。工程上有三条路：按 token 数分桶（长度相近的同批）、把图统一缩放到固定 tile 数、或用 [[inference-serving-02]] 的 continuous batching 让长短请求各自进出（batch 大小与吞吐的关系见 [[inference-serving-10]]）。

## 公司变体

- **Meta**：先把出处认准——Flamingo（cross-attention + Perceiver Resampler）出自 DeepMind，LLaVA 一系的 projector 路线出自微软与威斯康星大学，都不是 Meta 的工作，张冠李戴比答不出更扣分。这家公司的问法更偏**架构机制与工程取舍**：gated cross-attention 为什么把门控初始化为 0、Perceiver Resampler 为什么压成定长、交错图文序列的注意力掩码怎么设计、以及「不改底座接一个冻结 LLM」与「改底座统一训练」各自的代价。回答时给出结构图与张量形状，比只讲结论更容易过。
- **阿里巴巴（Qwen）**：对应的是 Qwen2-VL 这条原生 token 路线，问法会更偏**工程取舍 + 位置编码设计**：动态分辨率下 token 数为什么不可控、M-RoPE 把时间/高度/宽度拆开解决什么问题、图像与视频如何用统一范式处理、以及服务端怎么给视觉 token 设预算。准备时把「token 数 → 上下文 → KV 显存」这条链算成具体数字。

## 相关题目

- [[multimodal-02]]：从图像扩展到视频时会发生哪些变化——同一套 token 账在时间维度上的放大。
- [[multimodal-10]]：图像、视频与文本的混合检索索引，涉及视觉向量与生成式 VLM 的分工。
- [[llm-internals-02]]：KV cache 的内存公式与 320 KiB/token 口径的来源。
- [[inference-serving-03]]：PagedAttention 如何处理 KV cache 的碎片化（它不解决总量）。
- 延伸阅读：[[llm-internals-13]]（长上下文的 lost-in-the-middle 与「少塞、精排」）、[[inference-serving-05]]（prefix caching 与 prompt caching）、[[inference-serving-02]]（continuous batching）、[[finetuning-04]]（LoRA）、[[finetuning-02]]（DPO）。

## 参考资料与归属

1. [What is Multimodal AI? How It Works and Where It Is Used](https://outcomeschool.com/blog/multimodal-ai)，Amit Shekhar（Outcome School），2026-05-10。用于第 1 节开头的整体框架：每种模态各自的 encoder 把输入变成 embedding、在共享嵌入空间里对齐、再交给一个共同的 Transformer 推理并生成。该文「多模态更重更慢、必须按模态分别评估、模态不是越多越好」的成本提示，与本文第 5 节把视觉 token 预算当成显式工程决策的立场一致；它的「多模态理解 / 跨模态生成 / 跨模态对齐」三类系统划分与本题主线无关，未展开。
2. [What is a Vision Transformer (ViT) and How Does It Work?](https://outcomeschool.com/blog/decoding-vision-transformer-vit)，Amit Shekhar（Outcome School），2026-04-15。用于第 1 节的 ViT 细节：224×224 切成 16×16 patch 得到 14×14=196 个 patch、每块展平成 768 维后做线性投影（实现为 kernel/stride 均为 16 的卷积）、加 CLS token 与位置编码、过 12 层 encoder、用 CLS 做分类；以及「patch 越小 token 越多、self-attention 越贵，16×16 是折中」与 ViT-Base/Large/Huge 的规模口径。
3. [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)（延伸），Dosovitskiy et al.（ViT，ICLR 2021），2020-10-22。来源 2 未给出原论文口径的确认，第 1 节的 patch 化公式 $T=(N/P)^2$、patch 序列 + 位置编码作为纯 Transformer 输入、以及「大数据预训练后迁移到中小 benchmark 表现优异且训练算力更少」的结论取自该论文摘要。
4. [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)（延伸），Liu et al.（NeurIPS 2023），2023-04-17。第 2 节的 LLaVA 机制与数字：用纯语言 GPT-4 生成多模态图文指令数据、用一个投影把视觉编码器与 LLM 连起来端到端训练；85.1% 是与 GPT-4 在合成多模态指令跟随数据上的**相对分**，92.53% 是在 ScienceQA 上微调后 LLaVA 与 GPT-4 协同的准确率。这两个口径在来源 1、2 中都没有，属于延伸来源。
5. [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)（延伸），Alayrac et al.（NeurIPS 2022），2022-04-29。第 3 节的 cross-attention 路线：连接预训练视觉模型与语言模型的架构改造、处理任意交错图文序列、图像与视频统一接入；few-shot 下取得新 SOTA 且「在多个 benchmark 上超过用几千倍任务数据微调的模型」的结论取自摘要。Perceiver Resampler 压缩变长视觉输入、gated cross-attention 插层与门控零初始化的细节属于该论文正文，来源 1、2 未覆盖。
6. [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)（延伸），Li et al.（ICML 2023），2023-01-30。第 2 节的 Q-Former 机制与数字：Querying Transformer 两阶段预训练（先从冻结图像编码器学表示，再从冻结语言模型学生成）；zero-shot VQAv2 上比 Flamingo-80B 高 8.7%、可训练参数少 54 倍，均取自摘要。
7. [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)（延伸），Wang et al.（arXiv:2409.12191，2024-09-18）。第 4 节的原生 token 路线口径取自该文摘要：Naive Dynamic Resolution 让不同分辨率产生不同数量的视觉 token、M-RoPE 融合文本/图像/视频的位置信息、图像与视频采用统一范式处理，以及在 2B / 8B / 72B 三个规模上研究缩放规律；来源 1、2 未覆盖这部分。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
