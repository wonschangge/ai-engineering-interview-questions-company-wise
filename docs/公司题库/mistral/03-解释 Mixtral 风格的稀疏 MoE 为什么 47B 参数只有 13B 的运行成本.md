---
type: question
id: mistral-03
company: Mistral AI
topic: llm-internals
order: 3
question: 解释 Mixtral 风格的稀疏 mixture-of-experts 模型是如何工作的。为什么一个约 47B 参数的模型，运行成本大致只相当于约 13B 的模型？
question_en: Explain how a Mixtral-style sparse mixture-of-experts model works. Why does a ~47B-parameter model run at roughly the cost of a ~13B one?
asked_at: []
level: 进阶
tags: [moe, 稀疏激活, Mixtral, 数值推导, 系统设计]
sources:
  - title: MoE 详解
    url: https://outcomeschool.com/blog/mixture-of-experts
  - title: Mixtral of Experts（延伸）
    url: https://arxiv.org/abs/2401.04088
    author: Jiang et al. (Mistral AI)
    published: 2024-01-08
  - title: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity（延伸）
    url: https://arxiv.org/abs/2101.03961
    author: Fedus et al.
    published: 2021-01-11
  - title: KV Cache in LLMs
    url: https://outcomeschool.com/blog/kv-cache-in-llms
    author: Amit Shekhar (Outcome School)
    published: 2026-03-27
related: [llm-internals-10, llm-internals-02, llm-internals-16, llm-internals-09]
updated: 2026-09-29
---

## 一句话答案

> MoE 把每个 Transformer block 里的 FFN 换成 $N=8$ 个并行专家 FFN 加一个路由器：隐藏状态经路由器得到全部 8 个专家的分数，取 top-2，再在这 2 个上把 gate 重归一化到和为 1，输出的加权和进残差；attention、embedding、norm 与输出头全部共享。Mixtral 8x7B 因此是 32 层 × 8 个专家的 45.097B 专家参数 + 1.606B 共享主干 = 46.703B（口语 47B），每 token 只激活主干与 2/8 的专家 = 12.880B（口语 13B），激活比例 27.6%。「运行成本≈13B」指的是每 token 的算力与显存带宽：前向约 25.8 GFLOP、decode 每 token 只读约 24.0 GiB 权重，两项都恰好与一个 13B dense 模型相当；但显存里必须常驻全部 87.0 GiB 权重——MoE 省算力与带宽，不省显存。

## 面试官在考什么

- 能否把**总参数（容量与存储）**与**激活参数（每 token 计算）**两本账分开，并立刻落到 Mixtral 的具体配置上：8 个专家、每 token top-2、32 层。
- 是否知道 MoE 替换的**只是 FFN**，attention、embedding、norm、输出头都共享——这直接推出「不是 8 个模型集成」和「不是 $8\times7=56$B」两个结论。
- 能否当场把 46.7B / 12.9B 算出来：单专家 $3\,d\,d_{ff}$、GQA 的 $W_k,W_v$ 是 $H_{kv}d_{head}\times d$ 而不是 $d\times d$、输出头没有与 embedding tie。只背结论的人在这里会露出破绽。
- 是否说得出「相当于 13B」的**边界**：省算力与带宽是两笔不同的账，兑现条件也不同（batch 大小决定哪一笔变成速度），而显存反而更吃紧。
- 训练与系统侧是否做过：auxiliary load balancing loss 与 router z-loss 各修什么、capacity factor 与 token dropping 的代价、专家并行的 all-to-all 通信量、推理侧专家负载不均导致的长尾。

**常见错误答案**

- 「47B 只有 13B 的成本，所以 13B 的显存就够。」权重必须全部驻留，bf16 下是 87.0 GiB，比同激活量（12.9B）的 dense 模型（约 24.2 GiB）贵 3.6 倍。
- 「MoE 就是 8 个 7B 模型投票，8 份独立模型。」被复制 8 份的只有 FFN；如果是 8 份独立模型，参数量会是 $8\times7=56$B 而不是 46.7B。
- 「batch=1 时 MoE 会快 3.6 倍。」相对同总参 dense，每 token 读取量确实从 87.0 GiB 降到 24.0 GiB，但专家权重是 8 份碎片化矩阵、小 batch 下 grouped GEMM 效率不高，再叠加路由与 all-to-all 的固定开销，实测单请求延迟的改善明显小于理想带宽比。

## 原理与推导

### 只换 FFN：MoE 在 block 里的位置

```text
dense block                                MoE block（Mixtral 的每一层）
  x ─► RMSNorm ─► GQA ─► + ─► RMSNorm ─► FFN ─► +
  x ─► RMSNorm ─► GQA ─► + ─► RMSNorm ─► Router ─► top-2 ─┬─► E_a ─┐
                                                          └─► E_b ─┴─► 加权和 ─► +
```

dense block 是 `attention → FFN` 两个子层，MoE 版本只把第二个子层换掉。Mixtral 8x7B 的 32 层**全部**是 MoE 层，每层 8 个专家、每个 token 选 2 个；attention 是 GQA（32 个 Q head、8 个 KV head、$d_{head}=128$、上下文 32k）。共享的还有输入 embedding、输出头（配置里 `tie_word_embeddings: false`，与 embedding 不共享）与全部 RMSNorm；**每层有自己的路由器**，层间不共享，所以同一个 token 在不同层被送去的专家可以完全不同。这也是本题最容易失分的地方：MoE 不是把模型复制 8 份，而是只把 FFN 子层做成 8 选 2 的稀疏版本；「专家」也没有人工分工，训练里没有任何主题标签，学到的特化通常落在标点、词形这类低层模式上。

### 路由：全 8 个专家上 softmax，取 top-2，再重归一化

第 $t$ 个 token 的隐藏状态为 $x_t \in \mathbb{R}^{d}$，路由器 $W_r \in \mathbb{R}^{N \times d}$ 的第 $i$ 行记作 $e_i$：

$$
\ell_{i,t} = x_t^\top e_i,\qquad
s_{i,t} = \mathrm{softmax}_i(\ell_{i,t}),\qquad
\mathcal{T}_t = \mathrm{TopK}\big(\{s_{j,t}\}_{j=1}^{N},\, k\big)
$$

$$
g_{i,t} = \frac{s_{i,t}}{\sum_{j \in \mathcal{T}_t} s_{j,t}}\ (i \in \mathcal{T}_t),
\qquad
y_t = \sum_{i \in \mathcal{T}_t} g_{i,t}\, E_i(x_t)
$$

三个必须说准的细节：

1. **先在全 $N$ 个专家上 softmax、再选 top-$k$**；选定之后只在被选中的 $k$ 个上**重归一化**，使 gate 之和恒为 1，输出的尺度与 $k$ 无关。另一种写法是直接对 top-$k$ 的 logits 做 softmax——两者数学上完全相同（把 $\exp(\ell_i)/Z$ 在子集上重新归一化时 $Z$ 被约掉），实测差异只在 float32 的 $10^{-7}$ 量级；源文写的是后一种口径，两者不矛盾。
2. **top-$k$ 是硬选择**：8 个 gate 里 6 个是 0，那 6 个专家这一步一次矩阵乘都不做。$k=1$ 是 Switch Transformer 的配置，$k=2$ 是 Mixtral 的配置。
3. **选择本身不可导**，梯度只能经 gate 回传路由器——这是 MoE 训练不稳的根源，后面的 aux loss 与 z-loss 都是为它服务的。

### 两个账本：容量按专家数增长，算力按激活数增长

每个专家是一个标准 SwiGLU FFN，含 gate、up、down 三个矩阵。忽略 bias 与 norm，单个专家是 $3\,d\,d_{ff}$：

$$
P_{\text{total}} = \underbrace{P_{\text{shared}}}_{\text{embedding} + \text{lm head} + \text{attention} + \text{norm} + \text{router}} + L \cdot N \cdot 3\,d\,d_{ff},
\qquad
P_{\text{active}} = P_{\text{shared}} + L \cdot k \cdot 3\,d\,d_{ff}
$$

推理每 token 前向的矩阵乘 FLOPs 约为 $2\,P_{\text{active}}$（一次乘加算两次；训练还要乘约 3 计入反向），因此

$$
\frac{\text{每 token FLOPs}_{\text{MoE}}}{\text{每 token FLOPs}_{\text{同总参 dense}}}
= \frac{P_{\text{active}}}{P_{\text{total}}}
= \frac{1.606 + \frac{2}{8}\times 45.097}{46.703} = 0.276,
\qquad
\frac{P_{\text{total}}}{P_{\text{active}}} = \frac{46.703}{12.880} = 3.63
$$

右边那个比值就是题眼的机制：**同样的每 token 算力，MoE 换到 3.6 倍的参数容量——容量按 $N$ 增长、算力按 $k$ 增长，两者被解耦**。同一套数字里还能读出两个比例：把每层专家数从 1 个（等效 dense，含主干共 7.243B）加到 8 个，专家参数涨 8 倍，总参数只涨 $46.703/7.243 = 6.45$ 倍（共享主干不随 $N$ 变），每 token FLOPs 只涨 $12.880/7.243 = 1.78$ 倍。

### 「相当于 13B」的边界：省算力与带宽，不省显存

Mixtral 相对同总参（46.7B）dense 模型省的是两笔账，兑现条件不同：

- **算力账**：$2 \times 12.880\text{B} = 25.8$ GFLOP/token，对 dense 的 $2 \times 46.703\text{B} = 93.4$ GFLOP/token 是 $0.28\times$。
- **带宽账**（decode 每 token 要读的权重）：$2 \times 12.880\text{B} = 24.0$ GiB，对 dense 的 87.0 GiB 是 $0.28\times$。而一个 13B dense 模型每 token 要读 $13\text{B} \times 2 = 24.2$ GiB——**与 Mixtral 的 24.0 GiB 几乎相同**，这就是「运行成本≈13B」在带宽口径下的来源。
- **不省的一笔**：常驻显存。全部 256 个专家都要放进显存，87.0 GiB（bf16），单张 80GB 卡放不下，必须多卡切分；训练还要加梯度与优化器状态（AdamW 的 fp32 一阶/二阶动量加 fp32 主权重约 12 B/参数，46.7B 约 560 GB），所以 MoE 训练离不开专家并行与 ZeRO 一类的切分。

兑现条件要讲清：batch=1 的 decode 算术强度只有约 1 FLOP/byte，远低于 H100 bf16 的算力拐点（约 295 FLOP/byte，口径见 [[llm-internals-02]]），延迟由字节数决定，MoE 省下的**算力**在这一档几乎不变成速度，能兑现的只有**带宽**那一半。真正把 $0.28\times$ 变成吞吐的是算力成为瓶颈的大 batch 高吞吐服务——prefill 天然 compute-bound，decode 要 batch 大到数百（权重只读一次、被 batch 摊薄）。这也是 MoE 出现在集中式大吞吐服务而不是端侧的原因：端侧既放不下 87 GiB 权重，又是 batch=1，算力节省毫无用处。

### 训练侧的稳定性：aux loss、z-loss、capacity

路由器只优化语言建模 loss，最优解可能是把所有 token 送给少数几个专家——其余专家拿不到梯度、永远不变好，这叫**路由坍缩**。三个抓手：

**1. auxiliary load balancing loss**（GShard / Switch）。$f_i$ 是专家 $i$ 实际分到的 token 比例，$P_i$ 是 batch 内所有 token 对专家 $i$ 的平均路由概率：

$$
\mathcal{L}_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i P_i,
\qquad
f_i = \frac{1}{Tk}\sum_t \mathbb{1}\big[i \in \mathcal{T}_t\big],
\qquad
P_i = \frac{1}{T}\sum_t s_{i,t}
$$

$f_i$ 在实现里按常数处理（stop-gradient），于是这一项对 $P_i$ 的梯度是 $\alpha N f_i$：梯度下降的拉力正比于当前负载，负载重于加权平均的专家其路由概率被压低、概率流向负载轻的专家——这才是均衡压力的来源；这一项对 $P$ 是线性的，单独看会退化，只有与主 loss 联合才稳定。完全均衡时 $f_i = P_i = 1/N$，该项恰好取 $\alpha$——Switch 用 $0.01$，**Mixtral 的配置是 `router_aux_loss_coef: 0.02`**。太小压不住坍缩，太大则均衡梯度干扰语言建模目标、抑制专家特化。

**2. router z-loss**（ST-MoE 提出，常与 aux 同开）压的是路由器 logits 的尺度，$\mathcal{L}_z = \beta \cdot \frac{1}{T}\sum_t (\mathrm{logsumexp}_i\, \ell_{i,t})^2$：logits 爆炸会让 softmax 饱和成 one-hot，路由失去梯度。Switch 的做法是把路由器的局部计算提到 float32，其余保持 bfloat16，避免把 float32 张量送进 all-to-all。

**3. capacity factor 与 token dropping**。为了让每个专家的张量形状静态可编译，训练时给每个专家一个配额 $\text{capacity} = \frac{T \times k}{N} \times \text{capacity factor}$；超出配额的 token 被丢弃，其表示经残差直接进下一层。factor 大于 1 留缓冲：调大减少丢弃但空槽浪费算力与显存，调小反过来。Switch 的经验是把丢弃率压到 1% 以下；DeepSeek-V3 干脆取消 token dropping，改用只调 bias、不加辅助损失的负载均衡。

### 系统侧：专家并行的 all-to-all 与两级负载均衡

专家并行的工作流是「路由 → all-to-all dispatch 把 token 发到持有目标专家的 GPU → 专家计算 → all-to-all combine 送回」，bf16 下每个 token 每层要搬 $4 \times k \times d \times 2\ \text{B} = 4 \times 2 \times 4096 \times 2 = 64$ KiB（dispatch 与 combine 各两个方向）。这个量随 batch 与序列长度**线性增长**，而且必须跨节点走 InfiniBand，于是通信/计算比成为 MoE 训练的核心约束（应对手段是计算通信重叠与 node-limited routing）：batch 128、序列 4096 的单步有 524,288 个 token，每层要搬 32 GiB，32 层合计约 1.0 TiB。

推理侧的问题不同：专家分布不均会让部分 GPU 过热、其余空转，而且**延迟长尾由最热的那张卡决定**，所以需要两级均衡——expert-level 的 loss 防路由坍缩（取小），device-level 的 loss 保证设备不空转（可取大）。负载随请求波动的抖动，会让 MoE 服务的尾延迟比 dense 更难看。

### 别把 KV cache 的账混进来

MoE 只改 FFN，**完全不改变 KV cache**。Mixtral 8x7B 是 32 层、8 个 KV head、$d_{head}=128$、bf16，按 $M = 2 L H_{kv} d_{head} S \cdot \text{bytes}$ 算，每 token 是 $2\times32\times8\times128\times2 = 128$ KiB，32k 上下文单序列 4 GiB。单看这个数不大，但 cache 是**每条序列各自一份**、随并发线性增长：约 22 条并发 32k 会话的 cache 总量（88 GiB）就与 87.0 GiB 的全部权重相当。真正缩小这一项的是 GQA/MQA/MLA（见 [[llm-internals-02]]、[[llm-internals-03]]），不是 MoE；长上下文高并发时 KV cache 的读取还会**稀释** MoE 的权重节省，这一点在数值验证的并发读取量对照里量化。

## 数值与代码验证

### 复算 Mixtral 8x7B 的 46.703B / 12.880B

口径：`hidden_size=4096`、`intermediate_size=14336`、`num_hidden_layers=32`、`num_attention_heads=32`、`num_key_value_heads=8`（$d_{head}=128$）、`num_local_experts=8`、`num_experts_per_tok=2`、`vocab_size=32000`、`tie_word_embeddings=false`、bf16。

| 组成 | 参数量 | 说明 |
| --- | --- | --- |
| 输入 embedding | 0.131 B | $32000 \times 4096$，全体共享 |
| 输出头 lm_head | 0.131 B | 同样 $32000 \times 4096$，配置里没有 tie |
| 32 层 GQA attention | 1.342 B | 每层 $W_q,W_o$ 各 $4096^2$，$W_k,W_v$ 各 $8\times128\times4096$ |
| 32×2 个 RMSNorm + final norm | 0.0003 B | 266,240 个参数 |
| 32 个路由器 | 0.001 B | 每个 $4096 \times 8$，层间不共享 |
| **共享主干** | **1.606 B** | 上面五项之和，所有专家共用 |
| 专家（32 × 8 = 256 个） | 45.097 B | 单专家 $3\times4096\times14336 = 176.2$ M，单层 1409.3 M |
| **总参数** | **46.703 B** | 官方口径 46.7B（口语 47B） |
| **每 token 激活** | **12.880 B** | 主干 + 2/8 的专家（激活比例 27.6%），官方口径 12.9B（口语 13B） |

两个口径坑：**12.880B 里已经包含全部共享主干**（$1.606 + 2/8 \times 45.097$），不能说成「12.9B 专家参数」；attention 是 GQA，若把 $W_k,W_v$ 按 $4096^2$ 算、输出头又假设与 embedding tie，会多出约 0.67B，得到虚高的 47.4B / 13.6B。

### 两个账本与「相当于 13B」的对照

| 口径 | Mixtral 8x7B | 同总参 dense（46.7B） | 同激活量 dense（12.9B） |
| --- | --- | --- | --- |
| 每 token 前向 FLOPs | 25.8 GFLOP | 93.4 GFLOP | 25.8 GFLOP |
| decode 每 token 读权重（bf16） | 24.0 GiB | 87.0 GiB | 24.2 GiB |
| 常驻权重（bf16） | 87.0 GiB | 87.0 GiB | 24.2 GiB |
| 相对同总参 dense 的算力与容量 | 0.28× / 3.63× | 1× / 1× | 0.28× / 1× |
| 每层额外通信 | 64 KiB/token all-to-all | 无 | 无 |

decode 一步要读的是「权重 + 每条序列各自的 KV cache」，下面这张表说明 32k 上下文下 MoE 的 3.6 倍优势会被 KV 稀释：

| 并发 32k 会话 | KV cache 合计 | MoE 每步读取 | dense 46.7B 每步读取 | 比值 |
| --- | --- | --- | --- | --- |
| 1 | 4 GiB | 28.0 GiB | 91.0 GiB | 0.31× |
| 8 | 32 GiB | 56.0 GiB | 119 GiB | 0.47× |
| 32 | 128 GiB | 152 GiB | 215 GiB | 0.71× |

上下文越长、并发越高，每步的字节越被 KV cache 主导，MoE 相对 dense 的带宽优势越小；此时决定吞吐的是 cache 的读取与容量规划，不是专家的稀疏性。

### 代码：参数账本、FLOPs、带宽与 KV cache

```python
d, d_ff, L, N, k = 4096, 14336, 32, 8, 2       # Mixtral-8x7B-v0.1 的 config.json
H, H_kv, d_head, V = 32, 8, 128, 32000
B, GiB = 2, 1024 ** 3                          # bf16：每参数 2 字节

emb     = V * d                                # 输入 embedding
lm_head = V * d                                # 输出头（配置里没有 tie）
attn    = L * (d * d + H_kv * d_head * d + H_kv * d_head * d + d * d)   # Wq, Wk, Wv, Wo
norms   = L * 2 * d + d                        # 每层 2 个 RMSNorm + final norm
router  = L * d * N                            # 每层一个 4096 x 8 的路由器
shared  = emb + lm_head + attn + norms + router
expert  = 3 * d * d_ff                         # 单专家：gate / up / down
experts = L * N * expert
total   = shared + experts
active  = shared + k / N * experts             # 每 token 真正参与的参数

print(f"共享 {shared/1e9:.3f}B | 总 {total/1e9:.3f}B | 激活 {active/1e9:.3f}B（{active/total:.1%}）")
print(f"容量 {total/active:.2f}x | FLOPs/token {2*active/1e9:.1f} vs {2*total/1e9:.1f} GFLOP"
      f" | 比值 {active/total:.3f}")
print(f"权重 {total*B/GiB:.1f} GiB | decode 每 token 读 {active*B/GiB:.1f} GiB"
      f" | 13B dense 读 {13e9*B/GiB:.1f} GiB")

kv = 2 * L * H_kv * d_head * B                 # 2(K,V) x 层数 x KV 头 x head_dim x 字节
print(f"KV {kv/1024:.0f} KiB/token | 32k 单序列 {kv*32768/GiB:.1f} GiB"
      f" | 等权重并发 {total*B/(kv*32768):.1f} 条 | all-to-all {4*k*d*B/1024:.0f} KiB/token/层")
```

```text
共享 1.606B | 总 46.703B | 激活 12.880B（27.6%）
容量 3.63x | FLOPs/token 25.8 vs 93.4 GFLOP | 比值 0.276
权重 87.0 GiB | decode 每 token 读 24.0 GiB | 13B dense 读 24.2 GiB
KV 128 KiB/token | 32k 单序列 4.0 GiB | 等权重并发 21.7 条 | all-to-all 64 KiB/token/层
```

### 代码：路由、aux loss 与 capacity

```python
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)

d, N, K = 64, 8, 2                      # 专家数 N=8、每 token 选 top-2，与 Mixtral 同构
W_r = nn.Linear(d, N, bias=False)       # 路由器就是一个 d x N 的线性层
W_r.weight.requires_grad_(False)
x = torch.randn(4096, d)

logits = W_r(x)                         # [T, 8] 原始 logits，router z-loss 用它
scores = F.softmax(logits, -1)          # 在全部 8 个专家上归一化
top_s, idx = scores.topk(K, -1)         # 取 top-2
gates = top_s / top_s.sum(-1, keepdim=True)          # 在被选中的 2 个上重归一化
gates_alt = F.softmax(logits.gather(-1, idx), -1)    # 另一种等价写法
print(f"gate 之和 {float(gates.sum(-1).min()):.6f} ~ {float(gates.sum(-1).max()):.6f}"
      f" | 两种写法最大差 {float((gates - gates_alt).abs().max()):.1e}")
print("token0 选中", idx[0].tolist(), "gate =", [round(float(g), 3) for g in gates[0]])

f = F.one_hot(idx, N).float().sum(1).mean(0) / K     # 每个专家实际分到的 token 比例
P = scores.mean(0)                                   # 全 batch 的平均路由概率
aux = lambda f, P, a=0.02: a * N * (f * P).sum()
skew = torch.tensor([0.50, 0.20, 0.10, 0.05, 0.05, 0.04, 0.03, 0.03])
print(f"aux：随机路由 {float(aux(f, P)):.4f} | 完全均衡 "
      f"{float(aux(torch.full((N,), 1/N), torch.full((N,), 1/N))):.4f} | 偏斜 {float(aux(skew, skew)):.4f}")
print(f"z-loss {float((torch.logsumexp(logits, -1) ** 2).mean()):.3f}"
      f" | logsumexp 均值 {float(torch.logsumexp(logits, -1).mean()):.3f}（ln 8 = 2.079）")

T, cf = 4096, 1.25                     # 一个 batch 里 4096 个 token 进某一 MoE 层
cap = int(T * K / N * cf)
load = torch.tensor([1400, 1300, 1200, 1100, 1000, 900, 700, 592])   # sum = 8192 = T*k
drop = int((load - load.clamp(max=cap)).sum())
print(f"capacity {cap} | 分配总数 {int(load.sum())} | 丢弃 {drop}（{drop/load.sum():.1%}）")
```

```text
gate 之和 1.000000 ~ 1.000000 | 两种写法最大差 8.9e-08
token0 选中 [2, 0] gate = [0.704, 0.296]
aux：随机路由 0.0200 | 完全均衡 0.0200 | 偏斜 0.0493
z-loss 4.999 | logsumexp 均值 2.225（ln 8 = 2.079）
capacity 1280 | 分配总数 8192 | 丢弃 140（1.7%）
```

读法：gate 之和恒为 1、两种归一化写法等价；均衡路由下 aux 精确落在 $\alpha = 0.02$，偏斜到 $f_i=P_i=[0.50,0.20,0.10,\dots]$ 时升到 0.0493，说明这一项确实在惩罚不均衡；capacity 取 $4096\times2/8\times1.25 = 1280$，中等偏斜下丢 1.7%，贴近 Switch 提到的 1% 量级目标。与源文口径对照：源文给的是取整口径（约 47B、约 13B、$8\times7$ 不等于 56B、省算力不省显存、aux loss 与 expert capacity 的作用），本篇的 46.703B / 12.880B / 25.8 GFLOP / 24.0 GiB / 87.0 GiB / 128 KiB 是按上面写明的 config 口径自行复算的，与源文一致但更精确；源文说 gate 权重来自「只对被选中的专家做 softmax」，与「全量 softmax 后重归一化」等价，前文已给出数值验证。Switch 的 $\alpha = 0.01$、capacity factor 与「丢弃率压到 1% 以下」、Mixtral 的 `router_aux_loss_coef: 0.02` 分别来自两篇延伸来源。

## 常见追问

- **追问**：为什么是 47B 而不是 $8\times7=56$B？13B 里又包含什么？
  - 要点：被复制 8 份的只有 FFN，attention、embedding、输出头与 norm 都共享，所以是 $1.606 + 45.097 = 46.703$B。每 token 激活的 12.880B 是「主干 1.606B + 2/8 的专家 11.274B」，已经含主干，不是「13B 专家」。两处口径坑是 GQA 的 K/V 维度和未 tie 的输出头，算错会虚高约 0.67B。
- **追问**：softmax 是在 8 个专家上做，还是在被选中的 2 个上做？
  - 要点：先在全 $N$ 个上归一化再取 top-$k$、然后只在选中的 $k$ 个上重归一化，是 Mixtral 实现里的一步；直接对 top-$k$ 的 logits 做 softmax 与之数学等价，gate 之和恒为 1。要说清「重归一化」这一步——少了它，加权和的尺度会随 $k$ 与路由尖锐度变化。
- **追问**：既然「相当于 13B」，为什么 batch=1 时不觉得快 3.6 倍？
  - 要点：batch=1 的 decode 算术强度约 1 FLOP/byte，是带宽受限的，MoE 省下的算力在这一档几乎不变成速度；能兑现的只有带宽那一半（24.0 GiB vs 87.0 GiB），而专家权重是 8 份碎片化矩阵、grouped GEMM 在小 batch 下效率不高，再叠加路由与通信开销，实测延迟改善小于理想比值。把 $0.28\times$ 变成吞吐需要大 batch。
- **追问**：aux loss 的系数与 capacity factor 怎么调？
  - 要点：$\alpha$ 太小压不住坍缩，太大抑制专家特化（Switch $0.01$、Mixtral $0.02$）；capacity factor 调大减少 token 丢弃但空槽浪费算力，Switch 倾向 1.0–1.25 并把丢弃率压到 1% 以下。DeepSeek-V3 走另一条路：不加辅助损失、只按负载调专家 bias（$\gamma = 10^{-3}$），并取消 token dropping。
- **追问**：微调 MoE 有什么坑？
  - 要点：路由器行为会漂移，SFT 之后负载可能重新倾斜甚至弃用部分专家，要监控每个专家收到的 token 分布；小数据集上专家层更容易过拟合，Switch 的做法是只提高专家 FFN 的 dropout，而不是全局加大。另外专家并行下的微调要保证路由与并行布局不被破坏。
- **追问**：为什么不能拿 46.7B 去套 Chinchilla 的约 20 tokens/参数？
  - 要点：总参数只是存储与容量，计算预算对应的是激活参数，MoE 的缩放要额外引入专家数维度：把总参数代进 $6ND$ 会**高估**训练算力（真正参与前向与反向的只有激活参数），按 20 tokens/参数 配数据则会把 token 预算放大约 3.6 倍（见 [[llm-internals-09]]）。同理，MoE 的收益要与 KV cache 的账分开：长上下文高并发时 cache 读取会稀释专家的稀疏性（见 [[llm-internals-02]]）。

## 相关题目

- [[llm-internals-10]]：MoE 的通用机制、负载均衡与系统代价的完整推导，数字口径与本题一致（46.703B / 12.880B / 87.0 GiB）；本题是它在 Mistral AI 这道公司题上的落地版，重点在「运行成本≈13B」的边界。
- [[llm-internals-02]]：KV cache 的 $2LH_{kv}d_{head}S$ 公式、128 KiB/token 的复算方式，以及 H100 的算力拐点约 295 FLOP/byte 都来自这篇；MoE 不动 KV cache，两笔账必须分开算。
- [[llm-internals-16]]：把 MoE 层放回逐张量的前向流水线，才能一眼看清哪些张量是共享的、被替换的到底是哪一个子层。
- [[llm-internals-09]]：缩放定律在 MoE 上要换成激活参数口径，否则 46.7B 会被当成 46.7B dense 来配数据。

## 参考资料与归属

- Amit Shekhar (Outcome School)，《What is Mixture of Experts (MoE) and How Does It Work?》，2026-04-09，<https://outcomeschool.com/blog/mixture-of-experts>：MoE 的定位（只替换 FFN）、路由器与 top-2 加权求和、总参数与激活参数的区别、47B / 13B 与「为什么不是 56B」、负载均衡与 expert capacity、MoE 不省显存的结论。
- Albert Q. Jiang et al. (Mistral AI)，《Mixtral of Experts》（延伸），2024-01-08，<https://arxiv.org/abs/2401.04088>：Mixtral 8x7B 的配置（32 层、8 个专家、每 token top-2）与官方 46.7B / 12.9B 口径。
- William Fedus, Barret Zoph, Noam Shazeer，《Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity》（延伸），2021-01-11，<https://arxiv.org/abs/2101.03961>：$k=1$ 的稀疏配置、auxiliary load balancing loss 及其系数 $\alpha = 0.01$、expert capacity 与 token dropping 的经验（丢弃率压到 1% 以下）。
- Amit Shekhar (Outcome School)，《KV Cache in LLMs》，2026-03-27，<https://outcomeschool.com/blog/kv-cache-in-llms>：KV cache 的显存公式与「每条序列各一份、随并发线性增长」的规模化口径，用于最后一节的边界说明。
- Mixtral-8x7B 的 config 取值（`num_local_experts: 8`、`num_experts_per_tok: 2`、`num_key_value_heads: 8`、`intermediate_size: 14336`、`router_aux_loss_coef: 0.02`、`tie_word_embeddings: false`）取自该模型在 Hugging Face 上的 `config.json`；46.703B / 12.880B、25.8 与 93.4 GFLOP、24.0 与 87.0 GiB、128 KiB/token、64 KiB/token/层等数字是按该口径自行复算的，与源文的取整口径一致但不完全相同。router z-loss 的出处（ST-MoE）与 DeepSeek-V3 的无辅助损失负载均衡、取消 token dropping 属于补充背景，用于说明这两条机制的上下游。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
