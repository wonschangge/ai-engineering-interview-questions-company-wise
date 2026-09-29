---
type: question
id: alibaba-09
company: 阿里巴巴（Qwen）
topic: multimodal
order: 9
question: Qwen2.5-VL 使用原生动态分辨率的 ViT，配合 window attention 与多模态 RoPE。为什么用原生分辨率而不是固定切块，MRoPE 又编码了什么？
question_en: Qwen2.5-VL uses a native dynamic-resolution ViT with window attention and multimodal RoPE. Why native resolution instead of fixed tiling, and what does MRoPE encode?
asked_at: []
level: 高阶
tags: [vlm, 动态分辨率, window-attention, m-rope, 视觉-token]
sources:
  - title: 多模态 AI
    url: https://outcomeschool.com/blog/multimodal-ai
    author: Amit Shekhar (Outcome School)
    published: 
  - title: 解读 Vision Transformer (ViT)
    url: https://outcomeschool.com/blog/decoding-vision-transformer-vit
    author: Amit Shekhar (Outcome School)
    published: 
  - title: Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution（延伸）
    url: https://arxiv.org/abs/2409.12191
    author: Wang et al.
    published: 2024-09-18
  - title: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale（延伸）
    url: https://arxiv.org/abs/2010.11929
    author: Dosovitskiy et al. (ViT, ICLR 2021)
    published: 2020-10-22
related: [multimodal-01, multimodal-02, multimodal-10, coding-02, inference-serving-02]
updated: 2026-09-28
---

## 一句话答案

> 三件事各解决一个瓶颈，而且是**同一个瓶颈链上的三环**：
> ① **原生动态分辨率**解决「固定切块把信息弄丢」——它按图像原始宽高比产生**不同数量**的视觉 token，不再把图缩放到固定边长再用网格切碎（论文口径：Naive Dynamic Resolution，让模型按原始分辨率处理图像，并与视频统一范式）；
> ② **window attention** 解决「token 数上去之后注意力二次成本」——把全局注意力限制在窗口内，把 $O(N^2)$ 压到 $O(Nw)$；
> ③ **MRoPE** 解决「图像、视频、文本共处一个序列时位置语义混乱」——把位置编码分解到**时间、高、宽**三个分量，让同一个 patch 的坐标是 $(t,i,j)$ 而不是一个扁平序号。
> 一句话总结取舍：动态分辨率买的是**保真**（版面、小字、长宽比），付的是**token 预算不可预测**；window attention 买的是**成本**，付的是**长程依赖**；MRoPE 买的是**结构清晰**，付的是**位置编码实现复杂度**。

## 面试官在考什么

- 是否理解**「token 数线性于像素数」这条链**：ViT 把图切成 patch，patch 数就是 token 数，$N=HW/p^2$（再按 merge 缩小 $m^2$ 倍）。串起「分辨率翻倍 → token 数 ×4 → KV 显存 ×4 → prefill 时间 ×4」这条因果链，比背结论有说服力。
- 能否讲清**固定切块的真实代价**：为了统一尺寸必须缩放（小字变糊）或切 tile（语义单元被切断、补边浪费算力、切块数随分辨率线性膨胀）。
- **window attention 与滑窗注意力（SWA）的区别**：前者是视觉编码器内部的局部注意力（为了省二次成本，需要少量跨窗口/全局层来交换信息），后者是 LLM 解码时只回看最近 $w$ 个 token 的 KV（为了把 KV cache 变成常数）。两者动机不同、作用位置不同，混为一谈就露怯。
- **MRoPE 到底编码什么**：不是「给图像加位置」，而是把一组位置 id 拆成三维，让文本（三分量同步推进）、图像（$t$ 固定、$i,j$ 按网格）、视频（$t$ 按帧推进）在同一套旋转矩阵下共存。
- 是否知道这条路线带来的**工程副作用**：token 数随请求变化 → 批处理里 padding 与调度效率下降（串 [[inference-serving-02]]）。

**常见错误答案**

- 「动态分辨率就是把 resize 去掉」——还要处理 patch 网格对齐、补边、以及 token 数与调度器的配合；去掉 resize 只是第一步。
- 「window attention 让模型只能看局部，所以效果必然更差」——过度的局部化确实会伤长程依赖，所以工程上保留少量全局层或跨窗口机制；重点是**成本-能力可以被设计成折中**，不是非黑即白。
- 「MRoPE 就是 2D RoPE 加一维」——要说出三个分量各自承担什么：时间用于帧序、高宽用于空间网格，且文本 token 上三者必须退化成等价的 1D 行为，否则文本位置语义会被破坏。

## 原理与推导

### 1. 固定切块的三个代价（为什么值得换）

| 代价 | 机制 | 后果 |
| --- | --- | --- |
| 缩放损失 | 把任意长宽比缩放到固定边长，压缩比不同 | 小字、表格线、公式下标糊掉；细长图（票据、截图）损失最大 |
| 切块断裂 | 切成固定 tile 后按顺序拼接 | 跨 tile 的语义单元（一行表格、一句话）被切断，模型要靠位置猜接缝 |
| 补边浪费 | 不整除时补零到网格 | 极端长宽比下有效像素占比很低，算力白烧（一张 1000×100 的图补成 1000×1000 就是 10× 浪费） |

论文对 Qwen2-VL 的表述是把这一条作为核心设计（Naive Dynamic Resolution），目标是让视觉表示「更贴近人类感知」，并让图像与视频走**统一范式**。

### 2. 动态分辨率下的 token 数：先算清这条链

ViT 的输入是 patch 序列（论文的 16×16 口径），所以

$$N_{\text{patch}}=\frac{H}{p}\cdot\frac{W}{p},\qquad N_{\text{token}}=\frac{N_{\text{patch}}}{m^{2}}=\frac{HW}{p^{2}m^{2}}$$

其中 $m$ 是视觉 token 合并倍率（把 $m\times m$ 个相邻 patch 合并成一个 LLM 侧 token）。这条式子有三个直接推论，必须能当场说：

1. **边长翻倍 → token ×4**（面积翻倍倍率是平方）；
2. **合并倍率 $m=2$ → token 降为 $1/4$**，这是「保细节还是保预算」的旋钮；
3. **token 数决定下游一切**：KV 显存、prefill FLOPs（$2NP$）、注意力成本（$O(N^2)$ 项），以及调度器里这一条请求占多少 batch 预算。

### 3. window attention：把二次项换成一次项

全注意力每个 token 要和所有 token 算分，单层成本 ∝ $N^2d$；窗口注意力把每个 token 的可见范围限制在窗口 $w$ 内，成本 ∝ $Nwd$。比值是

$$\frac{\text{窗口}}{\text{全局}}=\frac{w}{N}$$

$N=4096$、$w=1024$ 时省 4 倍；$N=16384$ 时省 16 倍——**token 越多，窗口的收益越大**，这也解释了为什么这条设计与动态分辨率是配套的：先让 token 数变多（保真），再用窗口注意力把成本压回来。

代价与补偿：① 纯局部注意力会丢长程依赖，所以需要**少量全局层或跨窗口交换**（例如隔层全局、或用可学习的聚合 token）；② 窗口大小是新的超参，太小伤能力、太大省不下成本；③ 实现上要处理窗口边界与 $N$ 不被 $w$ 整除的 padding。

### 4. MRoPE：位置语义的三维分解

RoPE 的做法是对 $q,k$ 的每一对维度做角度为 $\theta_i = \text{pos}\cdot b^{-2i/d}$ 的旋转，因此「位置」本质上是一组角度。当序列里混着文本、图像、视频时，用一个扁平的位置序号会把空间结构抹平：图像的第 3 个 patch 与第 4 个 patch 可能是同一行也可能是换行后的第一个，扁平序号区分不了。

MRoPE 的做法是**把维度切成三段，分别用三个位置 id**：

| 模态 | 时间 $\text{pos}_t$ | 高度 $\text{pos}_h$ | 宽度 $\text{pos}_w$ |
| --- | --- | --- | --- |
| 文本 | 递增 | = 时间（保持一致，退化为 1D RoPE） | = 时间 |
| 图像 | 固定（同一张图内不变） | 行号 $i$ | 列号 $j$ |
| 视频 | 帧号 $t$ | 行号 $i$ | 列号 $j$ |

要点有三个：① 文本上三个分量必须同步，否则纯文本的位置语义会被破坏；② 图像上时间分量固定，等于告诉模型「这些都是同一时刻的空间 token」；③ 视频上时间分量按帧推进，天然表达「第 $t$ 帧的 $(i,j)$ 位置」——这就是为什么它能同时处理图像与视频（统一范式）。

### 5. 与 KV cache、批处理的交互（工程侧）

- **KV cache**：视觉 token 与文本 token 在 LLM 里是同一批 KV，所以每 token 的成本完全相同。按仓库统一常数，8B 级骨干约 **128 KiB/token**、70B 级约 **320 KiB/token**；一张 1024²、merge=2 的图是 1024 token，即 128 MiB / 320 MiB。**视觉输入是 KV 预算的大户**，这也是「图像前缀能不能缓存」这个优化（同一张图被多次提问时复用其 KV，串 [[inference-serving-05]]）在多模态场景收益特别大的原因。
- **批处理效率**：动态分辨率意味着同一批里每个请求的视觉 token 数不同。连续批处理（串 [[inference-serving-02]]）能吸收这种不均匀，但 padding 与 kernel 形状仍会造成浪费；极端长宽比（很宽的截图）会让单请求 token 数差异达到数十倍，调度器需要按 token 而不是按请求数计费。

## 数值与代码验证

### 表 1：分辨率 → token 数（$p=16$，merge 为 $m$）

| 输入 | $N_{\text{patch}}$ | $m=1$ | $m=2$（常见） | $m=4$ |
| --- | --- | --- | --- | --- |
| 512×512 | 1024 | 1024 | **256** | 64 |
| 1024×1024 | 4096 | 4096 | **1024** | 256 |
| 2048×2048 | 16384 | 16384 | **4096** | 1024 |
| 1000×100（极端长宽比） | 396 | 396 | **99** | 25 |

口径：$N=HW/p^2/m^2$，向下取整按实际补边后的网格算。第 4 行说明动态分辨率对细长图的意义：固定切块会把它补成方形（约 10× 浪费），动态分辨率只按实际 patch 数计价。

### 表 2：token 数 → 下游成本（70B 级骨干，8×H100 @ MFU 45% = 3.56 PFLOPs/s）

| 场景 | 视觉 token | prefill FLOPs $2NP$ | prefill 时间 | KV（70B，320 KiB/token） |
| --- | --- | --- | --- | --- |
| 512² 图，merge=2 | 256 | $3.58\times10^{13}$ | 10.1 ms | 80 MiB |
| 1024² 图，merge=2 | 1024 | $1.43\times10^{14}$ | 40.3 ms | 320 MiB |
| 1024² 图，merge=1 | 4096 | $5.73\times10^{14}$ | 161 ms | 1.25 GiB |
| 2048² 图，merge=2 | 4096 | $5.73\times10^{14}$ | 161 ms | 1.25 GiB |
| 长视频 544 帧 × 256 token | 139,264 | $1.95\times10^{16}$ | **5.48 s** | 42.5 GiB |

口径说明：prefill FLOPs 用 $2NP$（$N$ 为 token 数、$P$ 为参数量）；时间按 8 卡峰值 × MFU 45% 计；KV 用仓库统一常数（70B 级 320 KiB/token）。最后一行的 5.48 s 与 42.5 GiB 是本仓库多模态专题的同一口径，可直接对照。

### 表 3：window attention 的成本比

| 序列长度 $N$ | 全注意力相对成本 | 窗口 $w=512$ | $w=1024$ | $w=2048$ |
| --- | --- | --- | --- | --- |
| 1024 | 1.00 | 0.50 | 1.00（无收益） | 1.00 |
| 4096 | 1.00 | 0.125 | **0.25** | 0.50 |
| 16384 | 1.00 | 0.031 | **0.063** | 0.125 |

口径：比值 $=w/N$（忽略常数与边界项）。读法：**窗口只有在 $N>w$ 时才省**，且省的比例随 $N$ 增大而增大。$N=1024$、$w=1024$ 时没有收益，这解释了为什么窗口注意力必须与「提高分辨率/帧数」配套才有意义。

### 可运行代码

```python
# 复算表 1-表 3：token 数、KV、prefill 成本、窗口收益
KiB = 1024; GiB = 1024**3
P_70B, TFLOPS, MFU, GPUS = 70e9, 989e12, 0.45, 8
KV_8B, KV_70B = 128 * KiB, 320 * KiB          # 仓库统一常数：每 token KV

def tokens(h, w, p=16, m=2):
    return (h // p) * (w // p) // (m * m)

print("表 1：分辨率 -> token 数")
for h, w in ((512, 512), (1024, 1024), (2048, 2048), (1000, 100)):
    row = " ".join(f"m={m}: {tokens(h, w, 16, m):>6,d}" for m in (1, 2, 4))
    print(f"  {h}x{w:<5} patch={(h//16)*(w//16):>7,d}  {row}")

print("\n表 2：token -> prefill 与 KV（70B 级，8xH100 @ MFU 45%）")
eff = TFLOPS * MFU * GPUS
for name, n in (("512^2 m=2", 256), ("1024^2 m=2", 1024), ("1024^2 m=1", 4096),
                ("2048^2 m=2", 4096), ("544 帧 x 256", 139264)):
    flops = 2 * n * P_70B
    print(f"  {name:<12} {n:>7,d} tok  {flops:.2e} FLOPs  "
          f"{flops/eff*1000:8.1f} ms  KV(70B)={n*KV_70B/GiB:6.2f} GiB  KV(8B)={n*KV_8B/GiB:6.3f} GiB")

print("\n表 3：window attention 的成本比 w/N")
print(f"{'N':>7} {'w=512':>9} {'w=1024':>9} {'w=2048':>9}")
for n in (1024, 4096, 16384):
    row = " ".join(f"{min(1.0, w/n):>9.3f}" for w in (512, 1024, 2048))
    print(f"{n:>7,d} {row}")

# 视觉塔 vs LLM：谁才是瓶颈（配合 merge 的token 数）
VIT_L, VIT_D, VIT_MLP, N_PATCH = 24, 1024, 4096, 576
vit_flops = 2 * N_PATCH * (12 * VIT_D**2) * VIT_L      # 约 12*d^2 的主干矩阵乘
llm_flops = 2 * tokens(1024, 1024) * P_70B
print(f"\n视觉塔(336px CLIP-L 规模) ≈ {vit_flops:.3e} FLOPs；"
      f"1024^2/merge2 的 1024 个视觉 token 进 70B 后 ≈ {llm_flops:.3e} FLOPs，"
      f"比值 {llm_flops/vit_flops:.0f}x")
```

预期输出要点：表 1 中 `1000x100` 在 merge=2 下只有 99 个 token（对比补成方形要 396×4/4=396 个 patch 量级的浪费）；表 2 最后一行打印 `5.48 s`（与仓库多模态专题一致）；表 3 显示 $N=1024,w=1024$ 时比值为 1.000（无收益）；最后一行显示视觉塔与 LLM 侧的成本比在**百倍量级**——**瓶颈永远在「视觉 token 进入 LLM 之后」**，这与仓库既有多模态专题的结论一致。

## 常见追问

- **追问**：动态分辨率会不会让显存不可预测？
  - 要点：会。所以工程上要设**每请求视觉 token 上限**，超限就降采样或分块，并把预算按 token（而不是按请求）计入调度；这也是动态分辨率与连续批处理必须一起设计的原因。
- **追问**：window attention 与 LLM 里的滑窗注意力是一回事吗？
  - 要点：不是。前者在视觉编码器内部，用窗口换掉注意力二次成本，需要少量全局层补齐长程信息；后者在自回归解码时把 KV cache 限制在最近 $w$ 个 token，让每步成本不随上下文增长（串 [[multimodal-02]] 与推理专题）。
- **追问**：MRoPE 在纯文本上会不会破坏原来的位置行为？
  - 要点：不会，只要三个分量在文本上同步递增——此时三个分量的旋转等价于一维位置编码。实现上必须保证这条退化性质，否则文本能力会掉。
- **追问**：图像需要时间分量吗？
  - 要点：需要「固定值」这个语义。同一张图所有 patch 的时间分量相同，等于告诉模型这批 token 属于同一时刻；视频里让它按帧递增，同一套编码就同时支持两种输入。
- **追问**：为什么不干脆用固定高分辨率 + 大 token 预算？
  - 要点：成本是平方级的：token 数线性于像素，但注意力与 KV 的代价更高；固定高分辨率对所有请求都贵，而动态分辨率只对真正高分辨率的图贵。按需付费比统一加价更合理。
- **追问**：视觉 token 太多时怎么压？
  - 要点：三条路——提高 merge 倍率（$m$ 从 2 到 4 就是 4 倍）、只保留关键区域（先做区域检测再编码）、以及前缀缓存（同一张图复用已算好的 KV，串 [[inference-serving-05]]）。

## 相关题目

- [[multimodal-01]]：VLM 把图像接进 LLM 的三条路线与视觉 token 的显存账，本题是它在「动态分辨率」这一支上的深入。
- [[multimodal-02]]：从图像到视频的变化（帧采样、长上下文、KV 账），本题表 2 的长视频行与它同口径。
- [[multimodal-10]]：多模态检索的索引设计；页面图像检索（ColPali 式）正是「保留原分辨率、不做脆弱版面解析」的另一条应用。
- [[coding-02]]：GQA 与 KV 头的计算，解释了 KV 每 token 字节数是怎么来的。
- [[inference-serving-02]]：连续批处理如何吸收动态分辨率带来的 token 数不均。

## 参考资料与归属

- **多模态 AI** —— Outcome School 博客（该条来源未署个人作者与日期）：<https://outcomeschool.com/blog/multimodal-ai>。多模态模型的基本拼装方式（编码器 + 投影 + LLM）与「视觉信息最终要变成 token 进入语言模型」的总纲转述自这篇。
- **解读 Vision Transformer (ViT)** —— Outcome School 博客（该条来源未署个人作者与日期）：<https://outcomeschool.com/blog/decoding-vision-transformer-vit>。patch 切分、位置编码与「图像被当作 token 序列处理」的机制转述自这篇。
- **Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution（延伸）** —— Wang et al.，2024-09-18：<https://arxiv.org/abs/2409.12191>。Naive Dynamic Resolution（按原始分辨率产生不同数量视觉 token）、M-RoPE（把位置编码分解到时间/高/宽三个维度）、图像与视频统一处理范式，以及 2B/8B/72B 的规模序列来自这篇论文摘要。
- **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale（延伸）** —— Dosovitskiy et al.（ViT，ICLR 2021），2020-10-22：<https://arxiv.org/abs/2010.11929>。16×16 patch 化与「纯 Transformer 直接作用于 patch 序列」的原始设定来自这篇。
- **延伸来源说明**：表 1 的 token 算式与取值、表 2 的 FLOPs/KV 与时间、表 3 的窗口成本比、以及「视觉塔与 LLM 侧成本相差百倍量级」的结论，均按本仓库统一常数（ViT patch 16、H100 bf16 dense 989 TFLOPs、70B 级每 token KV 320 KiB、8B 级 128 KiB、MFU 45%、$2NP$ 口径）自行推算，其中长视频行的时间口径（8 卡节点）与本仓库多模态专题保持一致。Qwen2.5-VL 相对 Qwen2-VL 的 window attention 细节，其公开表述来自该系列的技术报告与模型卡；本仓库**未把该报告列入来源数组，也未据此改动任何来源的 title/author/published**。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
