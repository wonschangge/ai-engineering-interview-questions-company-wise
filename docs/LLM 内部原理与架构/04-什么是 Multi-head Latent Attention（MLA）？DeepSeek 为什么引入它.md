---
type: question
id: llm-internals-04
topic: LLM 内部原理与架构
order: 4
question: 什么是 Multi-head Latent Attention（MLA）？DeepSeek 为什么引入它？
question_en: What is Multi-head Latent Attention (MLA) and why did DeepSeek introduce it?
asked_at: [DeepSeek, Moonshot AI]
level: 高阶
tags: [mla, kv-cache, deepseek, 低秩压缩]
sources:
  - title: KV Cache Compression
    url: https://outcomeschool.com/blog/kv-cache-compression
    author: Amit Shekhar (Outcome School)
    published: ""
  - title: DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model（延伸）
    url: https://arxiv.org/abs/2405.04434
    author: DeepSeek-AI
    published: 2024-05-07
related: [llm-internals-02, llm-internals-03, llm-internals-10]
updated: 2026-09-28
---

## 一句话答案

> MLA 不靠“减少 K/V head 数”省显存，而是把每层每 token 的 K、V 联合下投影成一个低秩 latent 向量 $c_t^{KV}$（DeepSeek-V2 取 $d_c=512$），cache 里只存它和一小段承载位置的 64 维键，合计 576 个元素。生成时用 up-projection 还原 K/V，而这两个 up-projection 可以吸收进相邻的 $W^Q$、$W^O$，所以不必真的展开 K/V。按 DeepSeek-V2 配置（60 层、128 头、$d_h=128$）复算：同形状 MHA 每 token 要存 1,966,080 个元素，MLA 只存 34,560 个，缩小 56.9 倍；head 数一个没减，所以它压缩的是“存储表示”而不是“注意力视角”，质量反而比 MHA 好。

## 面试官在考什么

- 是否把 KV cache 的大小 $2 n_h d_h l$ 当成结构性事实，能否一眼指出 MLA 动的是公式里的哪一项。
- 能否区分两条压缩路线：GQA/MQA 改小 head 数 $n_h$（K/V 视角变少，质量要还债）；MLA 不动 $n_h$，改的是“每 token 存多少个元素”，用低秩瓶颈换存储。
- 能否写出 down/up projection 三式，并说清“cache 里只有 latent”的真正依据是 up-projection 的**吸收**，而不是“用的时候再展开”——展开出来的 K/V 如果还要存一份，等于没省。
- 是否知道 decoupled RoPE 是被吸收这个优化**逼出来的**：它不是可选细节，而是“能不能吸收”与“要不要位置信息”之间的唯一出路。这一点区分“读过论文”和“背过概念”。
- 能不能自己按 $d_c+d_h^R$ 和层数算一遍每 token 元素数，并说清论文摘要里 93.3% 的口径——它相对的是 DeepSeek 67B，且叠加了部署期的 6 bit 量化，与“相对同形状 MHA 的 98.2%”不是一回事。

常见错误答案：

- 把 MLA 说成“更激进的 GQA”或“所有 head 共享一份 K/V”。MLA 的 128 个 head 各有独立的 K/V 投影，共享的只是压缩后的 latent 存储。
- 说 MLA 是“有损压缩、会掉质量”。论文 Table 9 的同规模消融里，大 MoE 的 MLA 版在 BBH/MMLU/C-Eval/CMMLU 上全面高于 MHA 版（如 MMLU 59.0 对 57.5），小 MoE 上也只有 C-Eval 一项低 0.7 分，而它的 KV cache 只有 MHA 版的 14%。

## 原理与推导

### 1. 瓶颈：decode 每步都要重读整份历史 K/V

decode 生成第 $t$ 个 token 时，历史每个 token 的 K、V 都要参与打分与加权。MHA 每 token 每层缓存 $2 n_h d_h$ 个元素，$l$ 层即 $2 n_h d_h l$。DeepSeek-V2 的 $n_h=128$、$d_h=128$、$l=60$：单 token 就是 1,966,080 个元素，BF16 下 3.75 MiB。这条路径上 GPU 不缺算力，缺的是把 cache 从显存搬进 SM 的带宽，所以它是 memory-bound：压缩 cache 同时省显存和提高吞吐（公式推导与量级见 [[llm-internals-02]]）。

### 2. 四条压缩路线，以及“减 head”的代价

省 KV cache 有四条路，压缩对象、代价和能否叠加都不一样。MLA 属于最后一类，也是唯一在**不动 head 数**的前提下把 cache 压到 MQA 量级的路线：

| 路线 | 压缩对象 | 需要重训 | 典型压缩比 | 质量代价 | 能否与其它手段叠加 |
| --- | --- | --- | --- | --- | --- |
| 量化 | 每个元素占多少 bit（16→8 或 4） | 否 | 2×–4× | 很小；低于 4 bit 后误差明显 | 能 |
| Token 淘汰 | 留多少个 token（固定预算） | 否 | cache 大小固定，不随长度增长 | 被淘汰的信息永久丢失，精确召回类任务会失效 | 能 |
| 共享 K/V（MQA/GQA） | 有几份 K/V（$n_h \to n_g$） | 是 | GQA 常见 4×–8×；MQA 可到 $n_h$ 倍 | GQA 很小，MQA 明显 | 能 |
| 低秩压缩（MLA） | 每 token 存多少个元素 | 是 | 取决于配置；DeepSeek-V2 相对同形状 MHA 为 56.9× | 论文中优于 MHA | 能 |

量化把 16 bit 压到 8 bit 是 2×、压到 4 bit 是 4×，但 cache 仍随 token 线性增长；token 淘汰是唯一让 cache 真正不增长的路线，代价是不可逆的信息丢失（首个 token 作为 attention sink 必须保留，淘汰它会让输出退化）。MLA 与它们正交：模型用 MLA，部署时还能对 latent 再做量化、再叠淘汰。

MQA/GQA 把 $n_h$ 换成更小的组数 $n_g$：GQA-8 降到 122,880，MQA 降到 15,360。省得很直接，但 K/V 的“视角数”一起变少。DeepSeek-V2 论文附录 D.1 的同规模 dense 7B 消融给出了代价：

| 指标（7B dense） | MQA | GQA（8 组） | MHA |
| --- | --- | --- | --- |
| BBH（EM, 3-shot） | 33.2 | 35.6 | 37.0 |
| MMLU（Acc., 5-shot） | 37.9 | 41.2 | 45.2 |
| C-Eval（Acc., 5-shot） | 30.0 | 37.7 | 42.9 |
| CMMLU（Acc., 5-shot） | 34.6 | 38.4 | 43.5 |

MHA 全面领先，说明“减视角”这条路要还质量债。MLA 的目标就是绕开这笔债（GQA/MQA 的机制细节见 [[llm-internals-03]]）。

### 3. 第二条路线：低秩联合压缩，K 与 V 共用一个 latent

不动 head 数，改为压缩“每 token 存什么”。用一个下投影把输入压成 latent，再用上投影分别还原 K 与 V：

$$c_t^{KV} = W^{DKV} h_t,\qquad k_t^C = W^{UK} c_t^{KV},\qquad v_t^C = W^{UV} c_t^{KV}$$

其中 $W^{DKV}\in\mathbb{R}^{d_c\times d}$，$W^{UK},W^{UV}\in\mathbb{R}^{n_h d_h\times d_c}$，且 $d_c \ll n_h d_h$。“联合”指 K 和 V 共用同一个 latent，而不是各压各的。DeepSeek-V2 取 $d_c = 512$。

### 4. 关键：up-projection 可以吸收，所以不用展开 K/V

只把 latent 存起来、需要时展开，展开出的 K/V 还得再存一份，省不下来。真正的依据是打分和加权本来就不需要显式的 K/V。按 head 分块，$i$ 为 head 下标：

$$(W_i^Q h_t)^\top (W_i^{UK} c_j^{KV}) = h_t^\top \underbrace{\big(W_i^{Q\top} W_i^{UK}\big)}_{\text{推理前预计算}} c_j^{KV}$$

value 侧同理，$W_i^O W_i^{UV}$ 也可以预先合并：

$$o_{t,i} = \sum_{j\le t} a_{tj}\, W_i^O W_i^{UV} c_j^{KV} = W_i^O W_i^{UV}\Big(\sum_{j\le t} a_{tj} c_j^{KV}\Big)$$

于是 $W^{UK}$ 吸收进 $W^Q$、$W^{UV}$ 吸收进 $W^O$，推理期只需要 $c_j^{KV}$——这就是“cache 里只有 latent”的全部依据。相应地，吸收后每个 head 的 query 变成 $d_c=512$ 维，$n_h d_c = 65{,}536$ 维，是原始 $n_h d_h = 16{,}384$ 维的 4 倍，Q 侧投影的 FLOPs 上升，这是省显存换来的一笔账。

### 5. RoPE 与低秩压缩冲突

RoPE 是位置相关的旋转。若把 RoPE 作用在 $k_t^C$ 上，打分变成 $(W_i^Q h_t)^\top R_{t-j}\, (W_i^{UK} c_j^{KV})$：旋转矩阵依赖当前 token $t$ 与前缀 token $j$ 的相对距离，卡在 $W^Q$ 与 $W^{UK}$ 中间，而矩阵乘法不可交换，$W^{UK}$ 就没法预先乘进 $W^Q$。结果是每一步都要为全部前缀重算 Key，吸收带来的收益全部作废。

### 6. 解法：解耦 RoPE，让位置走一条不参与吸收的窄通道

把每个 head 的维度拆成两部分：内容部分走 latent（可吸收），位置部分用一个额外的小维度显式承载：

$$q_{t,i}^R = \mathrm{RoPE}\big(W_i^{QR} c_t^Q\big),\qquad k_t^R = \mathrm{RoPE}\big(W^{KR} h_t\big)$$

$$q_{t,i} = \big[\,q_{t,i}^C\,;\,q_{t,i}^R\,\big],\qquad k_{t,i} = \big[\,k_{t,i}^C\,;\,k_t^R\,\big]$$

$$o_{t,i} = \sum_{j\le t}\mathrm{softmax}_j\!\left(\frac{q_{t,i}^\top k_{j,i}}{\sqrt{d_h + d_h^R}}\right) v_{j,i}^C$$

三个容易考到的细节：

- $k_t^R$ **没有 head 下标**：$W^{KR}\in\mathbb{R}^{d_h^R\times d}$，所有 head 共用同一段位置键，位置分支退化成 MQA 形态，每个 token 只需多存 $d_h^R=64$ 个元素。
- 拼接后每个 head 的打分维度是 $d_h+d_h^R=192$，所以分母是 $\sqrt{192}$ 而不是 $\sqrt{128}$——和 [[llm-internals-01]] 的方差论证同源，缩放因子要跟着有效维度走。
- 内容项无位置依赖，可以吸收；位置项无法吸收，但只有 64 维，显式算也不贵。

至此每层每 token 的 cache 是 $d_c + d_h^R = 576$ 个元素。把它写成 $4.5\,d_h l$，正好等于 $2 n_g d_h l$ 在 $n_g=2.25$ 时的值：**MLA 的 cache 相当于一个只有 2.25 组的 GQA，但每个 head 仍有自己独立的 K/V。**

### 7. 两个配套设计

Query 也做低秩压缩（$c_t^Q = W^{DQ}h_t$，$d_c'=1536$；DeepSeek-V2-Lite 不做这项压缩），但这**不省 cache**，只为降低训练期 activation 显存。另外 latent 之后额外加 RMSNorm、在宽度瓶颈处乘缩放因子，用来稳住低秩压缩带来的输出尺度变化。

### 8. 为什么和 MoE 配套

MoE 省的是每 token 的 FLOPs（236B 总参数只激活 21B），但完全不减少 KV cache；MLA 减少的是每步要从显存读的 KV 字节，但不减少权重读取量。两者打的是不同的瓶颈，可以叠加，这也是 DeepSeek-V2 能同时拿出 42.5% 训练成本下降和 5.76 倍吞吐的原因（MoE 侧见 [[llm-internals-10]]）。

## 数值与代码验证

复算口径：DeepSeek-V2 配置 $l=60$、$n_h=128$、$d_h=128$、$d_c=512$、$d_h^R=64$；下表是**元素个数**，不含存储精度（论文 Table 1 同样按元素数计）。

| 方案 | 每层每 token | 每 token（60 层） | 相对 MHA | BF16 每 token |
| --- | --- | --- | --- | --- |
| MHA | $2\cdot128\cdot128=32{,}768$ | 1,966,080 | 100% | 3.75 MiB |
| GQA（8 组） | $2\cdot8\cdot128=2{,}048$ | 122,880 | 6.25% | 240 KiB |
| MQA | $2\cdot128=256$ | 15,360 | 0.78% | 30 KiB |
| **MLA** | $512+64=576$ | **34,560** | **1.76%** | 67.5 KiB |

相对同形状 MHA 的削减是 $(1-1/56.89)=98.2\%$。128K 上下文、batch 1 时，MLA cache 在 BF16 下约 8.44 GiB，而同形状 MHA 要约 480 GiB——后者在当前硬件上根本跑不起来。

### 摘要里 93.3% 的口径（容易记错的一处）

论文摘要写的是“相对 DeepSeek 67B 减少 93.3% KV cache”。这个数字**不是**相对 MHA，也不只是架构压缩，需要把部署精度一起算进来才能复现：

| 口径 | 计算 | 削减 |
| --- | --- | --- |
| 只看元素数：MLA vs DeepSeek 67B（GQA-8、95 层，每 token 194,560） | $1-34{,}560/194{,}560$ | 82.2% |
| 元素数 + 精度：MLA 按论文 3.2.3 节的部署平均 6 bit，67B 按 BF16 | $1-\dfrac{34{,}560\times6}{194{,}560\times16}$ | **93.3%** |
| 同形状 MHA（60 层、128 头） | $1-34{,}560/1{,}966{,}080$ | 98.2% |
| 同形状 GQA-8（60 层、128 头） | $1-122{,}880/1{,}966{,}080$ | 93.75% |

第二行复算得 93.34%，与摘要的 93.3% 吻合，这也解释了为什么论文强调"actually deployed"的对比里同时列了 FP8 权重与 6 bit KV cache 量化。顺带提醒最后一行的巧合：同形状 GQA-8 相对 MHA 恰好也是 93.75%，和 93.3% 数值接近但完全不是一回事，引用数字时必须带口径。

### 代码：cache 元素数与“吸收”的等价性

```python
# 1) 复算每 token 的 cache 元素数（DeepSeek-V2 配置）
l, nh, dh, dc, dhR = 60, 128, 128, 512, 64
rows = {"MHA": 2*nh*dh*l, "GQA-8": 2*8*dh*l, "MQA": 2*dh*l, "MLA": (dc+dhR)*l}
base = rows["MLA"]
for k, v in rows.items():
    print(f"{k:6s} {v:9,d} elem/token  {v*2/1024:8.1f} KiB(BF16)  vs MLA {v/base:5.2f}x")
# MHA    1,966,080 elem/token    3840.0 KiB(BF16)  vs MLA 56.89x
# GQA-8    122,880 elem/token     240.0 KiB(BF16)  vs MLA  3.56x
# MQA       15,360 elem/token      30.0 KiB(BF16)  vs MLA  0.44x
# MLA       34,560 elem/token      67.5 KiB(BF16)  vs MLA  1.00x
# 注：源博客里常引的"约 16 倍"是另一个口径——32 个 head、latent 512、
# 且不额外存 64 维 RoPE 键：2*32*128/512 = 16。
```

```python
import torch
torch.manual_seed(0)
# 缩小版尺寸便于跑通；真实值为 d=5120, nh=128, dh=128, dc=512, dhR=64
d, nh, dh, dc = 512, 8, 64, 128
Wq  = torch.randn(nh*dh, d) / d**0.5
Wuk = torch.randn(nh*dh, dc) / dc**0.5
Wuv = torch.randn(nh*dh, dc) / dc**0.5
Wo  = torch.randn(d, nh*dh) / (nh*dh)**0.5
h_t, c_j = torch.randn(d), torch.randn(dc)      # h_t 当前 token；c_j 前缀 token 的 latent

# 路径 A（显式）：先由 latent 还原 K，再打分
score_a = ((Wq @ h_t).view(nh, dh) * (Wuk @ c_j).view(nh, dh)).sum(-1)

# 路径 B（吸收）：W^UK 乘进 W^Q，query 变成 dc 维，直接和 cache 里的 latent 点积
W_quk = torch.stack([Wq.view(nh, dh, d)[i].T @ Wuk.view(nh, dh, dc)[i] for i in range(nh)])
score_b = (torch.einsum('ndc,d->nc', W_quk, h_t) * c_j).sum(-1)
print("score  max diff =", (score_a - score_b).abs().max().item())    # 4.8e-06

# value 侧：W^O_i W^UV_i 直接作用在 latent 的加权和上
a = torch.softmax(torch.randn(nh), -1)
W_ouv = torch.stack([Wo[:, i*dh:(i+1)*dh] @ Wuv.view(nh, dh, dc)[i] for i in range(nh)])
out_a = Wo @ (a.view(nh, 1) * (Wuv @ c_j).view(nh, dh)).reshape(-1)
out_b = torch.einsum('n,nc,ndc->d', a, c_j.expand(nh, dc), W_ouv)
print("output max diff =", (out_a - out_b).abs().max().item())        # 2.4e-07
```

两个 diff 都在 float32 误差量级，说明吸收是恒等变形而非近似。另外：吸收后 $n_h d_c=65{,}536$ 维的 query 投影是新增开销，所以 prefill 阶段（compute-bound）通常不吸收，直接按式展开 K/V 更划算；只有 decode（memory-bound）才走吸收形态，实现上因此是两套 kernel。

## 常见追问

- **追问**：MLA 的 cache 和 GQA-2.25 组等价，为什么质量能比 MHA 还好？
  - 要点：等价的是**存储量**，不是**计算**。GQA 里组内 head 共用同一份 K/V，K/V 的秩上界被组数卡死；MLA 的每个 head 有独立的 $W_i^{UK}$、$W_i^{UV}$ 切片，从同一个 latent 解出各自的 K/V，只是“存储时共享”。MLA 对 K 投影确实有秩约束：$\mathrm{rank}(W^{UK}W^{DKV})\le\min(d_c,\,d)$，而 MHA 的 $W^K$ 秩可达 $\min(n_h d_h,\,d)$——$d_c \ge \min(n_h d_h,d)$ 时能精确复现任意 $W^K$，DeepSeek-V2 取 $d_c=512$ 而 $\min(n_h d_h,d)=5120$，是刻意留出的瓶颈。质量没有因此变差：论文 Table 9 里参数更少的 MLA 版（247.4B 总参、21.5B 激活）在 BBH/MMLU/C-Eval/CMMLU 上全面高于 MHA 版（250.8B、25.0B），例如 MMLU 59.0 对 57.5。论文只给了这个经验结果，没有解释增益来自哪里——把 MLA 更强说成“低秩瓶颈的正则效应”属于推测，回答时最好标明这是自己的解读。
- **追问**：为什么 RoPE 的 key 要所有 head 共享？每个 head 一份不行吗？
  - 要点：可以，但 cache 会变成 $n_h d_h^R = 8192$ 个元素/层/token，压缩收益几乎全丢。位置是低熵的全局信号，没有“每个 head 需要不同位置视角”的必要；论文 Eq.15 的 $W^{KR}$ 形状里根本没有 head 下标，这是刻意设计。代价是位置信息只通过一个 64 维共享子空间注入。
- **追问**：既然 RoPE 和吸收冲突，那干脆不用位置编码，或者用可学习的绝对位置编码行不行？
  - 要点：结构上可行——绝对位置编码是加在输入 $h_t$ 上的，不会插在 $W^Q$ 与 $W^{UK}$ 之间，吸收照常成立。放弃它有两个原因：绝对位置编码的外推能力差，扩展到 128K 上下文要靠 RoPE 一类方法（插值、YaRN），而 DeepSeek 67B 已验证过 RoPE。解耦 RoPE 让两者兼得，代价只是每 token 多存 64 个元素（RoPE 与外推见 [[llm-internals-08]]）。
- **追问**：MLA 能不能加到已经训练好的模型上？
  - 要点：不能直接加。低秩瓶颈改变了 K/V 投影的秩上界，必须从头训练才有效；GQA 有 uptraining 这种便宜的迁移路径，MLA 没有。补救办法是先用低秩分解近似已有的 K/V 投影再继续训练——公开工作 TransMLA 就是把 GQA 权重转成 MLA 结构，报告在 LLaMA-2-7B 上压缩约 93% 的 KV cache，代价是约 6B token 的继续训练（<https://arxiv.org/abs/2502.07864>）。
- **追问**：训练时 MLA 的收益在哪里？
  - 要点：训练期没有 KV cache 瓶颈，收益主要是低秩瓶颈省下的 activation 显存（尤其 query 侧的低秩压缩）与 MoE 的稀疏计算；MLA 的显存红利几乎全在推理期。论文里的量化结果是：相对 DeepSeek 67B，每万亿 token 训练 GPU 小时从 300.6K 降到 172.8K（省 42.5%，其中 MoE 贡献为主），单节点 8 张 H800 的生成吞吐超过 50K tokens/s，是 67B 的 5.76 倍。
- **追问**：MLA 之后还有更狠的压缩吗？
  - 要点：两个方向——一是继续压 latent 维度或对其量化，二是跨 token 压缩（把连续多个 token 合成一个 cache 条目）。后者与“每个 token 独立存一份”的假设直接冲突，是目前长上下文 serving 的活跃方向。

## 公司变体

- **DeepSeek**：这道题的主场。偏数学推导与论文细节——他们必须能讲清 $d_c=4d_h$、$d_h^R=d_h/2$ 的取值、吸收为什么成立、RoPE 为什么与低秩压缩冲突，以及各消融表的数字。工程侧同样会被问，因为 DeepSeek 开源过 MLA 的 decode kernel（FlashMLA），kernel 层面的问题（吸收后两套 kernel、cache 布局、分页）都在射程内。
- **Moonshot AI**：偏工程落地与 serving 视角。这家公司公开的推理基础设施工作以 KV cache 为中心（分离式架构 Mooncake），所以常见问法是“MLA 的 cache 布局怎么设计”“prefill 与 decode 为什么用不同形态”“从 GQA 迁到 MLA 值不值”。

以上是依据两家公司公开技术输出的取向判断，不是对具体面试流程的描述。

## 相关题目

- [[llm-internals-02]]：KV cache 的显存公式 $2\times L\times H_{kv}\times d_{head}\times S\times b\times\text{bytes}$，是本题的前置；先能手推，再谈 MLA 改掉哪一项。
- [[llm-internals-03]]：MQA/GQA 用减少 K/V 视角换显存，正是 MLA 要绕开的那笔质量债。
- [[llm-internals-01]]：解耦 RoPE 后打分维度变成 $d_h+d_h^R=192$，缩放因子为什么是 $\sqrt{192}$ 要从这里推。
- [[llm-internals-08]]：RoPE 的旋转矩阵与相对位置性质，是理解“为什么不能对低秩 K 直接做 RoPE”的基础。
- [[llm-internals-10]]：MoE 省 FLOPs、MLA 省带宽，两者打不同瓶颈，DeepSeek-V2 把它们配套使用。

## 参考资料与归属

- Amit Shekhar (Outcome School)，《KV Cache Compression》，2026-09-12，<https://outcomeschool.com/blog/kv-cache-compression>
- DeepSeek-AI，《DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model（延伸）》，2024-05-07，<https://arxiv.org/abs/2405.04434>

标注（延伸）的一条用于补齐公式、配置与实验数字：第 2 节的四类压缩路线对比、低秩压缩的直觉与那则 16 倍示例来自博客；第 3 节的式子与维度、第 6 节的 $d_c+d_h^R$ 与“等价 2.25 组 GQA”、第 4 节的元素数表与 93.3% 口径复算、以及各消融数字，均取自（延伸）的 DeepSeek-V2 论文（配置见论文 3.1.2 节，KV cache 对比见 Table 1 与附录 D，部署精度与吞吐见 3.2.3 节）。常见追问里 TransMLA 的一行数据来自其论文摘要，链接已在正文给出。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
