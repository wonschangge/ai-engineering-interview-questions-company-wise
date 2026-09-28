---
type: question
id: llm-internals-10
topic: LLM 内部原理与架构
order: 10
question: 什么是 mixture-of-experts 架构？它如何在不增加 FLOPs 的情况下扩展容量？
question_en: What is a mixture-of-experts architecture and how does it scale capacity without scaling FLOPs?
asked_at: [Mistral AI, Cohere, DeepSeek, Moonshot AI, 智谱 AI, 阿里巴巴]
level: 进阶
tags: [moe, 稀疏激活, 路由, 负载均衡]
sources:
  - title: Mixture of Experts Explained
    url: https://outcomeschool.com/blog/mixture-of-experts
    author: Amit Shekhar (Outcome School)
  - title: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity（延伸）
    url: https://arxiv.org/abs/2101.03961
    author: Fedus et al.
    published: 2021-01-11
  - title: Mixtral of Experts（延伸）
    url: https://arxiv.org/abs/2401.04088
    author: Jiang et al. (Mistral AI)
    published: 2024-01-08
related: [llm-internals-04, llm-internals-09, llm-internals-16]
updated: 2026-09-28
---

## 一句话答案

> MoE 把 Transformer block 里的 FFN 换成 $N$ 个并行的专家 FFN 加一个路由器，每个 token 只走其中 top-$k$ 个。于是**总参数**（要装进显存的部分）随 $N$ 线性增长，而**每 token FLOPs** 只由被选中的 $k$ 个专家决定，attention 部分完全不变。这就是「扩容量不扩算力」的全部机制：容量按参数量算，成本按激活参数量算。代价是显存、通信与训练难度——MoE 省算力，不省内存。

## 面试官在考什么

- 能否把「总参数」与「激活参数」这两个账本分清楚，并用同一个式子说明白 FLOPs 由谁决定。答不出这个区分，后面全是空话。
- 是否知道 MoE 替换的**只是 FFN**，attention、embedding、norm 仍然共享——这正是 Mixtral 8x7B 约 46.7B（口语里常说 47B）而不是 8×7=56B 的原因。
- 负载均衡：为什么路由会坍缩、auxiliary loss 与 router z-loss 各自修什么、capacity factor 与 token dropping 是什么。这是从「背过」到「做过」的分界线。
- 系统侧代价：专家并行的 all-to-all 通信量随 batch 与序列长度增长；显存不降反升；小 batch 下专家利用率低、延迟抖动。
- 代表模型的设计差异：Switch Transformer 的 top-1、Mixtral 的 top-2、DeepSeekMoE 的细粒度专家 + 共享专家 + 无辅助损失负载均衡。

**常见错误答案**

- 「MoE 参数少所以快。」MoE 的参数**更多**，只是每 token 只用一小部分。省的是 FLOPs 与带宽，不是参数量。
- 「MoE 省显存。」恰恰相反，推理时必须放下全部专家权重，Mixtral 光权重就要约 87 GiB 的 fp16，比同激活量（约 12.9B）的 dense 模型更吃显存。
- 「专家是人分工好的，有的管数学有的管代码。」没有任何人工标注，训练出的特化往往落在标点、词形这类低层模式上。

## 原理与推导

### 架构位置：只换 FFN，attention 不动

标准 Transformer block 是 `attention → FFN` 两个子层。MoE 版本只把第二个子层替换掉：

```text
dense block                          MoE block
  x ──► MHA ──► + ──► FFN ──► +        x ──► MHA ──► + ──► Router ──► top-k ──┐
                     │                                                       │
                     v                                            E1 .. EN ───┴──► 加权求和 ──► +
```

隐藏状态进入路由器得到 $N$ 个分数，选 top-$k$ 个专家分别做 FFN，再按分数加权相加；没被选中的专家这一步完全不执行。attention、embedding、所有 norm 与最后的输出投影都被所有专家共享。

### 计算路径与公式

第 $t$ 个 token 的隐藏状态记作 $x_t \in \mathbb{R}^{d}$，第 $i$ 个专家的路由器权重记作 $e_i$，则

$$
s_{i,t} = \mathrm{softmax}_i\!\left(x_t^\top e_i\right),\qquad
\mathcal{T}_t = \mathrm{TopK}\!\left(\{s_{j,t}\}_{j=1}^{N},\,k\right)
$$

$$
g_{i,t} =
\begin{cases}
\dfrac{s_{i,t}}{\sum_{j \in \mathcal{T}_t} s_{j,t}}, & i \in \mathcal{T}_t \\[2mm]
0, & \text{otherwise}
\end{cases}
\qquad
y_t = \sum_{i \in \mathcal{T}_t} g_{i,t}\, E_i(x_t)
$$

三个容易说错的细节：

1. **softmax 在全 $N$ 个专家上做**，顺序是「先归一化、再选 top-k」。Mixtral 的实现选出 top-k 之后还对这 $k$ 个分数**再归一化**一次，使 gate 之和为 1，加权求和的尺度与 $k$ 无关；也有的实现直接用原始 $s_{i,t}$（此时 $k$ 个 gate 之和 $\le 1$，分布越尖锐越接近 1）。
2. **top-k 是硬选择**，$g$ 中有 $N-k$ 个零，只有 $k$ 个专家参与前向。$k=1$ 即 Switch Transformer 的 Switch layer，$k=2$ 是 Mixtral 的配置。
3. **加权求和是路由器唯一的可导通路**。top-k 的选择本身不可导，梯度只能经 $g_{i,t}$ 传回 $W_r$；去掉加权，路由器就学不到东西。

### 参数量与 FLOPs：两个账本

每个专家就是一个标准 FFN，含 gate、up、down 三个矩阵。若中间维度为 $d_{ff}$，单个专家参数量是 $3\,d\,d_{ff}$（忽略 bias）。于是

$$
P_{\text{total}} \approx \underbrace{P_{\text{shared}}}_{\text{attention + embedding + norm}} + L \cdot N \cdot 3\,d\,d_{ff},
\qquad
P_{\text{active}} \approx P_{\text{shared}} + L \cdot k \cdot 3\,d\,d_{ff}
$$

前向每 token 的矩阵乘 FLOPs 约为 $2\,P_{\text{active}}$（乘加各算一次，推理单次前向口径；训练约再乘 3，因为包含反向）。所以

$$
\frac{\text{每 token FLOPs}_{\text{MoE}}}{\text{同总参数 dense}} \approx \frac{P_{\text{active}}}{P_{\text{total}}} \approx \frac{k}{N}
\quad\text{（当共享主干可忽略时）}
$$

这就是题眼的完整答案：**容量按专家数 $N$ 增长，算力按每次激活的专家数 $k$ 增长，两者被解耦。** 把 $N$ 从 1 加到 8（$k$ 固定为 2）：专家参数涨 8 倍，总参数因共享主干不变只涨约 6.4 倍（而不是 8 倍），每 token FLOPs 只涨 1.8 倍——Mixtral 的 46.7B 对 12.9B 就是这个比值。

### 负载均衡：为什么需要辅助损失

路由器只优化语言建模 loss，最优解可能是把所有 token 都送给少数几个专家——其余专家拿不到梯度，永远不会变好，这叫**路由坍缩（routing collapse）**。工程上有三个抓手：

**1. auxiliary load balancing loss（Switch / GShard、DeepSeekMoE）。** 设 $f_i$ 是实际分给专家 $i$ 的 token 比例（$\sum_i f_i = 1$），$P_i$ 是 batch 内所有 token 对专家 $i$ 的平均路由概率（$\sum_i P_i = 1$）：

$$
\mathcal{L}_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i,
\qquad
f_i = \frac{1}{T k}\sum_{t} \mathbb{1}\!\left[i \in \mathcal{T}_t\right],
\qquad
P_i = \frac{1}{T}\sum_{t} s_{i,t}
$$

完全均衡时 $f_i = P_i = 1/N$，这一项退化成 $\sum_i P_i^2$，由 Cauchy–Schwarz（等号只在 $P_i$ 全相等时成立）取值 $1/N$，乘上 $N$ 恰好等于 $\alpha$，与专家数无关。但要注意这只说明均衡点的取值是 $\alpha$：$f_i$ 是离散计数，实现里按常数处理（stop-gradient），于是这一项对 $P$ 是线性的，梯度只经过 $P_i$：

$$
\frac{\partial \mathcal{L}_{\text{aux}}}{\partial P_i} = \alpha N f_i
$$

**这一步要读准**：辅助损失只动路由概率 $P$，并把 $P$ **推向当前的实际分配 $f$**，而不是直接把 $f$ 推向均匀——负载均衡的功劳不在这一项本身，而在它与主 loss、z-loss 的联合动态。实战口径是：$\alpha$ 取 $10^{-2}$ 量级（Switch 用 $0.01$，Mixtral 的配置里是 $0.02$），配合 z-loss 压住 logits 尺度；Switch 的实验里 token 丢弃率通常低于 1%，而且论文强调这个比例与专家数无关。想深挖的话，DeepSeek-V3 改用「不加 loss、只调 bias」的路线，正是因为干扰梯度与均衡目标互相拉扯。

**2. router z-loss（ST-MoE 提出，常与 aux 一起开）。**

$$
\mathcal{L}_{z} = \beta \cdot \frac{1}{T}\sum_{t} \left(\mathrm{logsumexp}_i\, \ell_{i,t}\right)^2
$$

其中 $\ell_{i,t}$ 是路由器 logits，$\mathrm{logsumexp}$ 就是 softmax 分母的对数。它不直接管均衡，管的是数值稳定：logits 爆炸会让 softmax 饱和成 one-hot，路由失去梯度。Switch 的做法是把路由器的局部计算提到 float32，其余保持 bfloat16，避免把 float32 张量送进 all-to-all。

**3. capacity factor 与 token dropping。** 为了让每个专家的张量形状静态可编译，训练时给每个专家设一个配额（Switch 原文的公式是这里 $k=1$ 的特例）：

$$
\text{expert capacity} = \left(\frac{\text{token 数} \times k}{\text{专家数}}\right) \times \text{capacity factor}
$$

超出配额的 token 被丢弃，其表示经残差连接直接进下一层（即这一层对它是恒等映射）。capacity factor 大于 1 留出缓冲；调大减少丢弃但浪费算力与显存。DeepSeek-V3 走的是另一条路：它**不做 token dropping**，改用无辅助损失的负载均衡——给每个专家一个标量 bias $b_i$，只加在 top-k 的选择分数上

$$
\mathcal{T}_t = \mathrm{argtop}_k\!\left(s_{i,t} + b_i\right)
$$

每步按负载更新 $b_i \leftarrow b_i + \gamma \cdot \mathrm{sign}(\bar{\ell} - \ell_i)$（$\ell_i$ 是该专家本步收到的 token 数，$\bar\ell$ 是均值，DeepSeek-V3 取 $\gamma = 10^{-3}$），过载的专家 bias 被压低、欠载的被抬高；加权时仍用原始 $s_{i,t}$。bias 不进主 loss，因此不向路由器注入干扰梯度，推理时直接冻结。DeepSeek-V3 另外保留了一个极小的 sequence-wise aux（$\alpha = 10^{-4}$），用来压单条序列内部的尖峰。

### 训练侧瓶颈：all-to-all

专家并行的基本流程是：路由 → **all-to-all dispatch** 把 token 发到持有目标专家的 GPU → 专家计算 → **all-to-all combine** 把结果送回原位。以 bf16 为例，每个 token 在**每一层**要搬的字节数是

$$
4 \times k \times d \times 2\ \text{B}
\quad(\text{dispatch 与 combine 各两个方向})
$$

取 $d = 4096$、$k = 2$：$4 \times 2 \times 4096 \times 2 = 64\ \text{KiB}$/token/层。这个量随 batch 与序列长度**线性增长**，而且必须跨节点走 InfiniBand，于是通信与计算的比例成了 MoE 训练的核心约束。DeepSeek-V3 的应对是用 DualPipe 让计算与通信重叠，再用 node-limited routing 限制每个 token 最多发往 4 个节点，官方报告在整个预训练过程中没有出现不可恢复的 loss spike。

## 数值与代码验证

### 复算 Mixtral 8x7B 的 46.7B / 12.9B 口径

口径说明：共享主干 $=$ embedding（词表 32000 × 4096，0.131 B）$+$ 输出头（config 里 `tie_word_embeddings: false`，再 0.131 B）$+$ 32 层 attention（$d_{model}=4096$、32 个 Q head、8 个 KV head、$d_{head}=128$，$W_q,W_o$ 为 $4096^2$、$W_k,W_v$ 为 $8 \times 128 \times 4096$，合计 1.342 B）+ 所有 norm（0.0003 B）+ 32 个路由器（每个 $4096 \times 8$，合计 0.001 B）；专家部分 $= 32 \times 8 \times 3 \times 4096 \times 14336 = 45.097$ B。逐项算出来是：

| 组成 | 参数量 | 说明 |
| --- | --- | --- |
| **共享主干** | **1.606 B** | embedding + 输出头 + attention + norm + router，所有专家共用 |
| 专家（32 × 8 个） | 45.097 B | 单专家 176.2 M，单层 1409.3 M |
| **总参数** | **46.703 B** | 官方口径 46.7B（口语说 47B） |
| **每 token 激活** | **12.880 B** | 主干 + 2/8 的专家，即官方口径的 12.9B（口语说 13B） |
| 激活比例 | 27.6% | 专家部分只激活 2/8 = 25.0%，主干是 100% |

两个最容易讲错的点：**为什么不是 8 × 7 = 56B**（只有 FFN 被复制 8 份，attention、embedding、输出头与 norm 都是共享的，$1.61 + 45.10 = 46.70$）；**12.9B 里已经包含全部共享主干**（$1.61 + 2/8 \times 45.10 = 12.88$），不能说成「12.9B 专家参数」。口径上还有两个坑：attention 是 GQA（8 个 KV head），$W_k,W_v$ 不能按 $4096^2$ 算；输出头也没和 embedding tie——两处都按「全尺寸 + tie」估会多出约 0.67B，得到虚高的 47.4B / 13.6B。

### dense vs MoE 对比表

FLOPs 用「前向每 token $\approx 2P$」的粗口径（估规模足够，忽略 attention 随序列长度的二次项与路由开销）：

| 对比项 | dense 同规模 | MoE（Mixtral 8x7B 配置） |
| --- | --- | --- |
| 总参数 | 46.7 B | 46.7 B |
| 每 token 激活参数 | 46.7 B | 12.9 B |
| 每 token 前向 FLOPs | ≈ 93.4 GFLOPs | ≈ 25.8 GFLOPs（约 0.28×） |
| 与同 FLOPs 的 dense 相比 | 12.9B 参数 | 3.6× 的参数量（= 46.7 / 12.9） |
| 权重显存（fp16） | 87.0 GiB | 87.0 GiB（**不省**） |
| 每 token 需读取的权重 | 87.0 GiB | 24.0 GiB（主干 + 被激活的 $k/N$ 专家） |
| 通信 | 无额外通信 | 每层 all-to-all，≈ 64 KiB/token（$d=4096,k=2$，bf16） |

最后两行的差别值得展开说：MoE 并没有减少空闲显存（权重全都要驻留），减少的是**每 token 要从 HBM 搬进计算单元的权重**——所以它省的是带宽与算力，只有在 batch 足够大、算力成为瓶颈时这份节省才变成吞吐。训练的复杂度、多卡专家并行的必要性也都随 MoE 一起到来。

### 路由、辅助损失、bias 与容量丢弃

一段可运行的验证（逐专家循环只为看清数学，生产实现用 grouped GEMM + all-to-all）：

```python
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

class TopKMoE(nn.Module):
    def __init__(self, d_model, d_ff, n_experts, top_k=2):
        super().__init__()
        self.N, self.K = n_experts, top_k
        self.W_r  = nn.Linear(d_model, n_experts, bias=False)          # 路由器就是一个线性层
        self.up   = nn.Parameter(torch.randn(n_experts, d_ff, d_model) / d_model ** 0.5)
        self.down = nn.Parameter(torch.randn(n_experts, d_model, d_ff) / d_ff ** 0.5)

    def forward(self, x, bias=None):
        logits = self.W_r(x)                                           # 原始 logits，z-loss 用它
        z = logits - logits.max(-1, keepdim=True).values               # 数值稳定，等价于原 softmax
        scores = F.softmax(z, dim=-1)                                  # 在全 N 个专家上归一化
        pick = scores if bias is None else scores + bias               # bias 只改「谁被选」
        _, idx = pick.topk(self.K, dim=-1)
        gates = scores.gather(-1, idx)
        gates = gates / gates.sum(-1, keepdim=True)                    # 被选中的 K 个上重归一化
        y = torch.zeros_like(x)
        for slot in range(self.K):
            for e in range(self.N):
                m = idx[:, slot] == e
                if m.any():
                    y[m] += gates[m, slot:slot + 1] * F.gelu(x[m] @ self.up[e].t()) @ self.down[e].t()
        return y, gates.detach(), idx, scores, logits

def aux_loss(scores, idx, N, K, alpha=1e-2):
    f = F.one_hot(idx, N).float().sum(1).mean(0) / K                   # sum_i f_i = 1
    return alpha * N * (f.detach() * scores.mean(0)).sum(), f          # f 按常数处理

model = TopKMoE(64, 256, 8, 2)
y, gates, idx, scores, logits = model(torch.randn(512, 64))
loss, f = aux_loss(scores, idx, 8, 2)
print("token0 选中", idx[0].tolist(), "gate", [round(float(g), 3) for g in gates[0]],
      "和 =", round(float(gates[0].sum()), 6))
print("f =", [round(float(v), 3) for v in f], "均衡值 =", round(1 / 8, 3))
print("aux =", round(float(loss.detach()), 5), "| z-loss =",
      round(float((torch.logsumexp(logits, -1) ** 2).mean().detach()), 3))

def update_bias(bias, load, gamma=1e-3):                               # DeepSeek-V3 的 aux-free
    return bias + gamma * torch.sign(load.float().mean() - load)       # 过载 -> bias 降低

bias = torch.zeros(8)
for _ in range(2):
    bias = update_bias(bias, torch.tensor([900, 880, 860, 180, 170, 160, 150, 140]))
print("bias =", [round(float(b), 4) for b in bias])

T, N, K, cf = 4096, 8, 2, 1.25
cap = int(T * K / N * cf)                                              # 1280
load = torch.tensor([1600, 1100, 1000, 800, 700, 600, 450, 350])       # 一个偏斜的负载
print(f"capacity = {cap} | 分配总数 {int(load.sum())} | 丢弃 {int((load - load.clamp(max=cap)).sum())}")
```

实测输出与手算一致：

- token0 选中 2 个专家，两个 gate 之和为 `1.0`；
- 均摊负载下 `aux = 0.01002`，与均衡点取值 $\alpha = 0.01$ 相差不到 0.2%；`z-loss ≈ 5.00`（即 $\mathrm{logsumexp} \approx 2.24$，与 $\ln 8 = 2.08$ 同量级，训练中会被继续压低）；
- bias 走两步后为 `[-0.002, -0.002, -0.002, +0.002, +0.002, +0.002, +0.002, +0.002]`，过热的三个专家被压低、欠载的五个被抬高；
- `capacity = 4096 × 2 / 8 × 1.25 = 1280`，偏斜负载下丢弃 320 个分配，占 6600 个总分配的 4.85%——这就是「capacity 调小则丢 token、调大则浪费算力」的具体样子。

## 常见追问

- **追问**：MoE 到底省了什么？
  - 要点：省的是每 token 的**计算量与显存带宽**，不省显存占用。全部专家权重都必须加载；decode 阶段每 token 只需读取主干与 $k$ 个被激活专家的权重，专家部分的读取量约为 $k/N$。它不省参数量、也不省总训练显存。
- **追问**：为什么小 batch 下 MoE 的加速不明显？
  - 要点：decode 是显存带宽受限的。batch=1 时，无论 dense 还是 MoE，每 token 都要把激活到的权重从 HBM 读一遍，MoE 读得少但两者都远达不到算力峰值；只有 batch 变大、算力成为瓶颈时，$k/N$ 的算力节省才转成吞吐。这也是 MoE 主要出现在大规模高吞吐服务、而不是端侧的原因。
- **追问**：expert capacity 调大还是调小？
  - 要点：调大减少 token 丢弃（保住质量）但填充的空槽浪费算力与显存，调小反过来。Switch 的经验是丢弃率要压到 1% 以下，并倾向用更小的 capacity factor（1.0–1.25）。DeepSeek-V3 干脆取消了 token dropping。
- **追问**：auxiliary loss 的系数 $\alpha$ 怎么选？
  - 要点：Switch 用 $10^{-2}$，Mixtral 的配置是 $0.02$，DeepSeekMoE 的消融实验用 expert-level $10^{-2}$。太小压不住坍缩，太大则均衡梯度会干扰语言建模目标、抑制专家特化——这是一条互相拉扯的曲线。所以要区分 expert-level（防坍缩，取小）与 device-level（保设备不空转，可取大）两个 loss。
- **追问**：MoE 的 scaling law 看哪个参数量？
  - 要点：看**激活参数**，它对应计算预算。总参数只是存储与容量。因此不能拿 MoE 的总参数量去套 Chinchilla 式的「参数-数据」配比，详见 [[llm-internals-09]]。
- **追问**：微调 MoE 有什么坑？
  - 要点：路由器行为在微调中会漂移，可能出现负载重新倾斜或某些专家被弃用；小数据集上专家层更容易过拟合，Switch 的做法是只在专家 FFN 上加更高的 dropout，而不是全局加大 dropout。

## 公司变体

- **Mistral AI**：偏工程实现，几乎一定会落到 Mixtral 的具体数字上——8 个专家、每 token 选 2 个、46.7B 总参数（口语 47B）、12.9B 激活（口语 13B），以及「为什么不是 56B」；也可能追问这份权重的显存账（约 87 GiB fp16）与多卡部署。
- **Cohere**：偏企业私有化部署的经济账。常见问法是「旗舰模型总参数约为激活参数的 10 倍，为什么这适合私有部署、代价在哪」——要同时说清私有部署卡的是**显存**（全部专家都要放下，成本反而高）与算力利用率（高吞吐 batch 下 MoE 才划算），结论是 MoE 适合集中式高吞吐服务，小 batch 场景吃亏。
- **DeepSeek**：偏论文深挖。公开题目方向包括 DeepSeekMoE 与标准 top-2 MoE 的区别（细粒度专家分割 + 共享专家隔离）、无辅助损失负载均衡解决了什么、bias 技巧如何生效，以及现场用 PyTorch 实现带共享专家的 top-k 路由并指出效率与正确性陷阱。DeepSeek-V3 官方报告给出 671B 总参数、37B 激活这两个可引用数字，以及全程没有不可恢复 loss spike 的稳定性结论。
- **Moonshot AI（Kimi）**：偏架构与系统成本的结合。公开题目提到 K2 是 1T 参数、每 token 约激活 32B、数百个专家，要求解释路由机制与训练它的系统成本。重点在专家并行的 all-to-all、专家放置与负载均衡如何一起决定实际吞吐。
- **智谱 AI（GLM）**：偏好参数账本与实现细节。公开题目以 GLM-4.5 的「355B 总参数 / 32B 激活」问这样拆分的收益与代价，并要求现场实现 top-k 路由后对比辅助损失与 loss-free 两种均衡方案——两条路线都要能写出来。
- **阿里巴巴（Qwen）**：偏选型取舍。公开题目给的是同一家的两个选项——30B-A3B 的 MoE 与 32B 的 dense，问什么时候选 MoE。答题框架应是：显存能放下 30B 全部专家、吞吐由 batch 撑得起、追求更高「每 FLOPs 质量」时选 MoE；单请求低延迟、显存受限或端侧部署时选 dense。

## 相关题目

- [[llm-internals-04]]：MLA 与 MoE 在 DeepSeek 系列里是同时出现的一对设计，一个压低 KV cache、一个压低每 token FLOPs，问「DeepSeek 的推理成本优化」时会一起考。
- [[llm-internals-09]]：Chinchilla 式缩放定律按参数量与 token 数配比，MoE 下必须换成激活参数口径，否则总参数会被严重误用。
- [[llm-internals-16]]：逐张量走一遍 decoder-only 的一次前向，把 MoE 层放回完整数据流里看，就不会把「只换 FFN」记成「换掉整个 block」。

## 参考资料与归属

- Amit Shekhar (Outcome School)，《What is Mixture of Experts (MoE) and How Does It Work?》，2026-04-09，<https://outcomeschool.com/blog/mixture-of-experts>
- William Fedus, Barret Zoph, Noam Shazeer，《Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity》（延伸），2021-01-11，<https://arxiv.org/abs/2101.03961>
- Albert Q. Jiang et al. (Mistral AI)，《Mixtral of Experts》（延伸），2024-01-08，<https://arxiv.org/abs/2401.04088>
- Barret Zoph et al.，《ST-MoE: Designing Stable and Transferable Sparse Expert Models》（延伸，用于 router z-loss 的出处与作用），2022-02-17，<https://arxiv.org/abs/2202.08906>
- DeepSeek-AI，《DeepSeek-V3 Technical Report》（延伸，用于 aux-free 负载均衡、细粒度专家与 671B/37B 数字），2024-12-27，<https://arxiv.org/abs/2412.19437>
- Damai Dai et al. (DeepSeek-AI)，《DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models》（延伸，用于细粒度专家分割、共享专家隔离与 expert-level $\alpha_1 = 0.01$），2024-01-11，<https://arxiv.org/abs/2401.06066>
- Wang et al.，《Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts》（延伸，用于 bias 更新规则与 $\gamma = 10^{-3}$），2024-08-28，<https://arxiv.org/abs/2408.15664>

正文中的架构与公式来自上述资料。Mixtral 的 46.703B / 12.880B、capacity 与 token dropping 的数值、all-to-all 字节数、dense vs MoE 的 FLOPs 表是按上述口径自行复算的结果，与源文的取整口径（47B / 13B）一致但不完全相同；「辅助损失把概率推向当前分配、而不是直接推向均匀」属于对公式与实现的分析，源文只写了 encourages uniform routing。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
