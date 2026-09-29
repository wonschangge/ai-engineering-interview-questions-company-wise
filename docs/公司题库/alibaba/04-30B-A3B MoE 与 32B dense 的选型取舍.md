---
type: question
id: alibaba-04
company: 阿里巴巴（Qwen）
topic: llm-internals
order: 4
question: Qwen 同时提供 dense 与 MoE 模型（30B 约 3B 激活；235B 约 22B 激活）。什么时候你会选 30B-A3B 的 MoE，而不是 32B 的 dense 模型？
question_en: Qwen ships both dense and MoE models (30B with ~3B active; 235B with ~22B active). When would you pick the 30B-A3B MoE over a 32B dense?
asked_at: []
level: 进阶
tags: [MoE, 稀疏激活, 模型选型, 推理成本, 显存与吞吐]
sources:
  - title: MoE 详解
    url: https://outcomeschool.com/blog/mixture-of-experts
    author: 
    published: 
  - title: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity（延伸）
    url: https://arxiv.org/abs/2101.03961
    author: Fedus et al.
    published: 2021-01-11
  - title: Mixtral of Experts（延伸）
    url: https://arxiv.org/abs/2401.04088
    author: Jiang et al. (Mistral AI)
    published: 2024-01-08
  - title: Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws（延伸）
    url: https://arxiv.org/abs/2401.00448
    author: Sardana et al.
    published: 2023-12-31
  - title: Continuous Batching in LLMs
    url: https://outcomeschool.com/blog/continuous-batching-in-llms
    author: Amit Shekhar (Outcome School)
    published: 
related: [llm-internals-10, llm-internals-09, mistral-03, inference-serving-02]
updated: 2026-09-28
---

## 一句话答案

> 结论先给：服务是**高并发、吞吐与成本优先**、显存预算与 32B dense 同档时，选 30B-A3B 的 MoE——它总参 30.5319B、每 token 激活 3.3528B（10.98%），容量/算力比 9.11×，bf16 权重 56.87 GiB 对 Qwen3-32B dense 的 61.02 GiB 反而略小 7%（每 token 前向 6.71 对 65.52 GFLOP，KV cache 96 对 256 KiB）。
> 所以这组模型不是 Mixtral 那种「用显存换算力」，而是「同样显存预算下用更稀疏的激活换 9.11× 容量」；兑现条件必须讲清：batch 撑到几十以上、专家并行压在单机 NVLink 域内、能接受负载不均带来的 p99 抖动。decode 的算术强度上界只有 2.08 FLOPs/byte，batch=1 与长上下文下 MoE 省的是带宽而不是算力，实测加速远小于 9.8×；反过来，单请求低延迟、端侧或单卡部署、需要密集微调定制、集群缺少 all-to-all 预算，就选 32B dense。

## 面试官在考什么

- 能不能把两本账分开并落到 config：总参数（容量与存储）$P_{\text{total}}\approx P_{\text{shared}}+L\,N\,3\,d\,d_{ff}$，激活参数（每 token 计算）$P_{\text{active}}\approx P_{\text{shared}}+L\,k\,3\,d\,d_{ff}$，并且知道 3.3528B 里已经含全部共享主干。
- 是否知道 MoE 只替换 FFN：attention、embedding、RMSNorm、输出头与 KV cache 全部不动，所以本题的 KV 账与 [[llm-internals-02]] 完全一致，被换掉的只有 FFN 那一段算法。
- 会不会当场算：30.5319B / 3.3528B / 9.11×、56.87 对 61.02 GiB、96 对 256 KiB、6.71 对 65.52 GFLOP。只背「MoE 省算力」的人给不出这几个数。
- 是否知道本组模型与 Mixtral 8x7B 的结论相反：Mixtral 是容量 3.63× 但显存贵 3.6 倍，Qwen3-30B-A3B 是 9.11× 容量、显存同档甚至略低。这是本题的分水岭。
- 兑现条件是否讲透：MoE 的算力优势只在 prefill 与大 batch 的高吞吐档变成速度，KV 稀释、all-to-all 通信、专家负载不均的尾延迟是三个必须主动提的代价。
- 训练、微调、量化与缩放口径是否想过：$6N_{\text{active}}D$、路由器漂移、拿总参数套 Chinchilla 会把数据预算放大约 9 倍。

**常见错误答案**

- 「30B-A3B 就是一个 3B 模型，显存只要 6 GB。」错的。3.3528B 是每 token 参与计算的参数（含共享主干），全部 6144 个专家的权重必须常驻，bf16 下 56.87 GiB。
- 「MoE 省显存，所以 MoE 部署更省」与「MoE 每 token 快 9.8 倍」。本组 MoE 显存只是略小；算力优势只有在大 batch 的 prefill 档才兑现，decode 的带宽比还会被 KV 稀释。

## 原理与推导

### 1. 机制与公式：MoE 只替换 FFN，48 层无一例外

```text
dense block（Qwen3-32B 的每一层）              MoE block（Qwen3-30B-A3B 的每一层）
  x ─► RMSNorm ─► GQA ─► + ─► RMSNorm ─► FFN ─► +      x ─► ... ─► Router ─► top-8 ─┬─► E1 ─┐
                                                                                     └─► E8 ─┴─► 加权和 ─► +
```

- 被替换的只有 FFN 子层；attention（含 QK-Norm）、输入 embedding、RMSNorm、输出头全部共享。**每层有独立路由器、层间不共享**，同一个 token 在 48 层里会被送到完全不同的专家组合。
- `decoder_sparse_step=1` 表示 48 层全部是 MoE 层，不存在「部分层稀疏」的口径混用，下面所有账都按 48 层全 MoE 算。
- config 里**没有** `shared_expert_intermediate_size`，即 Qwen3 的 MoE 取消了共享专家，与 Qwen2-MoE、DeepSeekMoE 的「共享专家 + 细粒度专家」不是同一套设计。这一点必须先声明，否则按共享专家口径算会把参数量多算一大截。
- 单专家是 SwiGLU 的三个矩阵（gate/up/down），$3\,d\,d_{ff}$，忽略 bias 与 norm。推理**每 token 前向**的矩阵乘约 $2P_{\text{active}}$ FLOPs（一次乘加算两次）；训练用 $6N_{\text{active}}D$，其中 $6=2$（前向）$+4$（反向）。
- Qwen3-30B-A3B 的 config 口径：`hidden_size=2048`、`num_hidden_layers=48`、`num_attention_heads=32`、`num_key_value_heads=4`、`head_dim=128`、`moe_intermediate_size=768`、`num_experts=128`、`num_experts_per_tok=8`、`norm_topk_prob=true`、`router_aux_loss_coef=0.001`、`vocab_size=151936`、`tie_word_embeddings=false`、`torch_dtype=bfloat16`、`max_position_embeddings=40960`。对照 Qwen3-32B dense：`hidden_size=5120`、64 层、64 个 Q 头、8 个 KV 头、`head_dim=128`、`intermediate_size=25600`、`vocab_size=151936`、`tie_word_embeddings=false`；两边都**没有 tie**，151936×2048 与 151936×5120 的 lm_head 都与 embedding 各自独立，算参数量时不能省掉输出头。

### 2. 路由：全 128 上 softmax，取 top-8，再重归一化

$$
s_{i,t}=\mathrm{softmax}_i\!\left(x_t^\top e_i\right),\qquad
\mathcal{T}_t=\mathrm{TopK}\!\left(\{s_{j,t}\}_{j=1}^{128},\,8\right),\qquad
g_{i,t}=\frac{s_{i,t}}{\sum_{j\in\mathcal{T}_t}s_{j,t}},\qquad
y_t=\sum_{i\in\mathcal{T}_t}g_{i,t}E_i(x_t)
$$

- 先在全 128 个专家上做 softmax，再取 top-8；`norm_topk_prob=true` 表示在被选中的 8 个上重归一化，使 gate 之和恒为 1，加权和的尺度与 $k$ 无关。
- top-8 是**硬选择**：另外 120 个专家这一步一次矩阵乘都不做。
- 选择本身不可导，梯度只能经加权系数回传路由器，所以加权求和这一步不能省。
- 系数口径不许混用：Qwen3-30B-A3B 的 `router_aux_loss_coef=0.001`，比 Switch 的 0.01、Mixtral 的 0.02 小一个量级。原因是粒度——128 选 8 每 token 只碰 6.25% 的专家，8 选 2 要碰 25%，均衡压力天然更小。aux loss 防的是**路由坍缩**，不是强制均匀；引用 $\alpha$ 时必须写明是哪个模型的哪个值。

### 3. 两本账与 Mixtral 的对照（本题的题眼）

复算：单专家 $3\times2048\times768=4{,}718{,}592\approx4.72$ M，6144 个专家合计 28.991B；共享主干 1.5409B（embedding 0.3112B + 输出头 0.3112B + 48 层 attention 0.9060B + 48 个路由器 0.0126B + QK-Norm 0.000012B）；总参 $1.5409+28.991=30.5319$ B（官方 30.5B），每 token 激活 $1.5409+(8/128)\times28.991=3.3528$ B（官方 3.3B），激活比例 10.98%，容量/算力比 9.11×。

对照 dense 32B：embedding 与输出头 1.5558B、attention 6.0398B、FFN 25.1658B，合计 32.7615B（官方 32.8B）。

题眼是与 Mixtral 8x7B 的对照：Mixtral 是总参 46.703B / 激活 12.880B、容量 3.63×，但 bf16 权重 87.0 GiB 比同激活量的 12.9B dense（24.2 GiB）贵 3.6 倍，那是**用显存换算力**；Qwen3-30B-A3B 是总参 30.5319B / 激活 3.3528B、容量 9.11×，权重 56.87 GiB 与 dense 32B 的 61.02 GiB 同档甚至略低 7%。**同样显存预算下，MoE 用更稀疏的激活换到 9.11× 容量；它的代价从显存搬到了系统复杂度（专家并行通信、负载均衡、尾延迟）与训练、微调难度上。**这是本题的第一句结论。

### 4. KV cache 与带宽稀释：MoE 完全不改 KV

MoE 只改 FFN，KV cache 的口径与 dense 完全一样，$M=2LH_{kv}d_{\text{head}}S\cdot\text{bytes}$：

- 每 token：MoE $2\times48\times4\times128\times2=98{,}304$ B = **96 KiB**；dense $2\times64\times8\times128\times2=262{,}144$ B = **256 KiB**（与仓库既有的 Qwen3-32B 口径一致），比值 2.67×。
- 32k 单序列：MoE 3.0 GiB，dense 8.0 GiB。每卡留 60 GiB 给 KV 时，并发上限 20 条对 7 条（与 [[alibaba-03]] 的 Qwen3-32B 口径一致：7.5 向下取整为 7 条）。
- decode 每步读取 = 权重 $+b\times$ 单序列 KV。b=1 时 MoE 9.2 GiB 对 dense 69.0 GiB（7.5×），b=32 时 102.2 对 317.0（3.1×），b=128 时 390.2 对 1085.0（2.8×），b 继续增大收敛到 KV 之比 2.67×。**MoE 的带宽优势随并发与上下文长度被 KV 稀释；长上下文高并发时决定吞吐的是 KV 容量与读取，不是稀疏性。**
- 算术强度上界（这是「MoE 为什么不一定更快」的定量解释）：每步 FLOPs $=2P_{\text{active}}b$，字节 $=2P_{\text{active}}+b\,k_{kv}S$，$b\to\infty$ 时收敛到 $2P_{\text{active}}/(k_{kv}S)$，即 MoE 2.08、dense 7.63 FLOPs/byte，两者都远低于 roofline 295 FLOPs/byte。**32k 上下文下 decode 无论并发多大都是带宽受限**，MoE 省下的算力不会在 decode 变成速度，只能在 prefill 与大 batch 高吞吐档兑现。
- 单请求延迟上限（带宽下界，H100 3.35 TB/s、$b=1$）：$S=0$ 时 MoE 6.71 GB / 3.35 TB/s = 2.00 ms/token（约 500 token/s），dense 19.56 ms（约 51 token/s）；$S=32\text{k}$ 时 2.96 ms 对 22.12 ms。这是上限而不是预测，实测还要乘 grouped GEMM 效率、路由与通信开销。

### 5. 三笔账：TTFT、吞吐与成本、通信

**① TTFT。** prefill 8k 的 FLOPs 分两块：线性项 $2P_{\text{active}}S$ 是 54.9 对 536.8 TFLOPs；attention 的二次项 $4LS^2d$ 是 26.4 对 88.0 TFLOPs（因果掩码可再减半，这里保守不减）；合计 81.3 对 624.7 TFLOPs，比值 0.130——理想的 9.8× 被不稀疏的 attention 二次项稀释成 7.7×。按 H100 bf16 dense 989 TFLOPs、MFU 40%（有效 395.6 TFLOPs）得 TTFT 约 206 ms 对 1579 ms。

**② 吞吐与成本。** 算力受限档（chunked prefill + 大 batch）的单位 GPU 吞吐比取 9.8×（理想）到 7.7×（含 attention）；按 \$2/GPU·h 的样例算式，MoE 58,995 token/s → \$0.0094/百万 token，dense 6,038 token/s → \$0.092/百万 token，差 9.8×。decode 档按表 4 的稀释口径，b=128 时只差 2.8×。**MoE 的成本优势兑现条件是 batch 撑得起**：长请求在 decode 阶段长期占住 slot，混池时尾延迟与吞吐互相牵制（机制见 [[inference-serving-02]]）；batch=1 的服务里省的是带宽不是算力，收益远小于 9.8×。

**③ 通信与尾延迟。** 专家并行的 all-to-all 每 token 每层搬 $4k\,d\cdot2\ \text{B}=4\times8\times2048\times2=128$ KiB，×48 层 = 6.00 MiB/token；在 NVLink（按 900 GB/s 量级）上是 7.0 µs，在 400 Gb/s InfiniBand（50 GB/s）上是 125.8 µs——而 MoE 单 token 的计算量 6.71 GFLOP / 395.6 TFLOPs 只有 17 µs。**EP 一旦跨节点，通信就主导单请求延迟**：专家并行要尽量压在单机 NVLink 域内（8 卡内），跨节点要么改成 TP + 专家复制，要么只在高吞吐离线档使用。通信之外还有负载不均：专家分布不均会让部分 GPU 过热、其余空转，**长尾由最热的那张卡决定**，需要两级均衡（expert-level 防坍缩取小、device-level 防空转可取大）并监控每个专家收到的 token 分布——这是 dense 的均匀负载完全没有的运维负担。

### 6. 训练、微调、量化与缩放口径

- **训练算力**：$6N_{\text{active}}D$，MoE 只有 dense 的 $3.3528/32.7615=10.2\%$。若达到同等质量需要 $k$ 倍 token，训练算力比 $=0.102k$，盈亏平衡点 $k=9.8$——把 $k$ 当讨论变量写出来就是「MoE 更吃数据」的定量形式；不要断言具体 $k$，它由数据质量与评测档位决定。
- **训练显存不省**：Adam 混合精度按 14 B/参数（bf16 权重 2 + fp32 主权重 4 + 一阶动量 4 + 二阶动量 4）计，MoE 427 GB 对 dense 459 GB；按 16 B/参数（含 fp32 梯度）是 489 对 524 GB；再叠加激活与 all-to-all 缓冲后两者同档。
- **训练侧额外成本**：6.00 MiB/token 的 all-to-all、负载均衡调参、capacity factor 与 token dropping（Switch 的口径是把丢弃率压到 1% 以下、倾向 1.0–1.25 的 factor；DeepSeek-V3 取消 token dropping，改用只调 bias 的无辅助损失方案）。
- **微调**：路由器行为会漂移，SFT 之后负载可能重新倾斜甚至弃用部分专家，必须监控专家负载分布；小数据集上专家层更容易过拟合，Switch 的做法是只提高专家 FFN 的 dropout 而不是全局加大；LoRA 要讲清作用范围（attention 还是专家 FFN）与并行布局下 checkpoint 合并的麻烦。这是 dense 32B 的既有优势，选型时要如实计入。
- **量化**：4 bit 权重 MoE 15.3 GB 对 dense 16.4 GB，两者都要全部权重常驻，量化不能把 MoE 变成「小显存模型」；KV 量化与 MoE 正交（MoE 不动 KV）。路由器 logits 对尺度敏感（router z-loss 的存在就是证据），专家权重分布差异大，per-tensor 量化更容易伤到冷门专家——这条是工程判断，不是论文结论。
- **缩放定律口径**：MoE 的缩放必须换成激活参数（见 [[llm-internals-09]]），拿总参数套 20 tokens/参数会把数据预算放大约 9 倍。把推理成本写进目标函数后（Beyond Chinchilla-Optimal 的口径），「过训小模型」与「用稀疏换容量」是同一类决策——MoE 就是把推理成本前置到架构里的做法，这是本题的理论收口。

### 7. 决策清单与生产坑

选 30B-A3B MoE 当且仅当：

- (a) 服务是吞吐、成本优先的高并发在线场景，batch 能撑到几十以上，按第 5 节的算式便宜 3–10×；
- (b) 显存预算与 dense 32B 同档（56.87 对 61.02 GiB），但希望 KV 少 2.67×、同卡多跑 2.67× 的会话（20 对 7 条 32k 会话）；
- (c) 质量瓶颈在「容量」而不是「单 token 深度」，即知识覆盖、多语言、长尾与代码 API 这类靠容量吃下的任务；
- (d) 集群有 NVLink 域内的专家并行与专家均衡监控能力；
- (e) 能接受 p99 尾延迟抖动与更复杂的部署、回滚。

选 32B dense 当且仅当：

- (a) 单请求低延迟或端侧、单卡部署（MoE 必须常驻全部 128 个专家，4 bit 后仍 15.3 GB）；
- (b) 微调、定制密集，要避开路由漂移与专家过拟合；
- (c) 集群通信弱，没有 all-to-all 预算；
- (d) 对尾延迟稳定性与既有量化、投机解码生态依赖强。

生产坑按踩到概率排序：

1. 把「MoE 省显存」说反——本组模型 MoE 显存略小，但仍是 56.87 GiB，不是「3B 的显存」；把「激活 3.3528B」当成「3.3528B 模型」（激活参数含全部共享主干，且总参数仍要全部驻留）。
2. 用总参数套 Chinchilla，数据预算放大约 9 倍；把 KV 的账混进 MoE 的稀疏性（KV 由 $L$、$H_{kv}$、$d_{\text{head}}$ 决定，MoE 不动它）。
3. 忽略专家负载抖动的尾延迟，以及 EP 跨节点的 all-to-all（6.00 MiB/token 对 17 µs 的计算）。
4. 忘记路由器漂移的监控；把 batch=1 的理论带宽比当成实测加速（要乘 grouped GEMM 效率与路由开销）。

## 数值与代码验证

### 表 1：参数账本（config 口径见第 1 节，bf16 即 2 B/参数）

| 组成 | Qwen3-30B-A3B | Qwen3-32B dense |
| --- | --- | --- |
| 单专家 / 单层 FFN | $3\times2048\times768=4{,}718{,}592\approx4.72$ M | $3\times5120\times25600=393.2$ M |
| 专家总量 / 全部 FFN | $48\times128\times4{,}718{,}592=28.991$ B | $64\times3\times5120\times25600=25.1658$ B |
| embedding + 输出头 | $2\times151936\times2048=0.6223$ B | $2\times151936\times5120=1.5558$ B |
| attention | $48\times(2048{\cdot}4096+2\times2048{\cdot}512+4096{\cdot}2048)=0.9060$ B | $64\times(5120{\cdot}8192+2\times5120{\cdot}1024+8192{\cdot}5120)=6.0398$ B |
| 路由器 | $48\times2048\times128=0.0126$ B | 无 |
| QK-Norm | $48\times2\times128\approx0.000012$ B | $64\times2\times128\approx0.000016$ B |
| **总参数** | **30.5319B**（官方 30.5B） | **32.7615B**（官方 32.8B） |
| **每 token 激活** | $1.5409+\frac{8}{128}\times28.991=$ **3.3528B**（官方 3.3B） | 全部 32.7615B（100%） |

### 表 2：两本账与 Mixtral 8x7B 的对照

| 口径 | Qwen3-30B-A3B | Qwen3-32B dense | Mixtral 8x7B |
| --- | --- | --- | --- |
| 总参 / 激活 | 30.5319B / 3.3528B | 32.7615B / 32.7615B | 46.703B / 12.880B |
| 每 token 前向 FLOPs（$2P_{\text{active}}$） | 6.71 GFLOP | 65.52 GFLOP | 25.8 GFLOP |
| 容量/算力比 | 9.11× | 1× | 3.63× |
| bf16 权重（常驻） | 61.06 GB = 56.87 GiB | 65.52 GB = 61.02 GiB | 87.0 GiB |
| decode 每 token 读权重 | 6.71 GB = 6.245 GiB | 61.02 GiB | 24.0 GiB |

### 表 3：KV cache 与并发上限（$S=32\text{k}$、每卡留 60 GiB）

| 模型 | $L$ | $H_{kv}$ | 每 token KV | 32k 单序列 | 并发上限 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-30B-A3B | 48 | 4 | $2\times48\times4\times128\times2=98{,}304$ B = **96 KiB** | 3.0 GiB | $60\text{GiB}/(96\text{KiB}\times32768)=20.0$ → **20 条** |
| Qwen3-32B | 64 | 8 | $2\times64\times8\times128\times2=262{,}144$ B = **256 KiB** | 8.0 GiB | $=7.5$ → **7 条**（与 [[alibaba-03]] 一致） |

### 表 4：decode 带宽稀释、算术强度与延迟下界

| 口径 | Qwen3-30B-A3B | Qwen3-32B dense | 比值 |
| --- | --- | --- | --- |
| 每步读取 $b=1$（权重 + 1×KV） | 6.25 + 3.0 = 9.2 GiB | 61.02 + 8.0 = 69.0 GiB | 7.5× |
| $b=8$ | 30.2 GiB | 125.0 GiB | 4.1× |
| $b=32$ | 102.2 GiB | 317.0 GiB | 3.1× |
| $b=128$ | 390.2 GiB | 1085.0 GiB | 2.8× |
| $b\to\infty$ 的极限 | — | — | 收敛到 8.0/3.0 = **2.67×** |
| 算术强度上界 $2P_{\text{active}}/(k_{kv}S)$ | 2.08 FLOPs/byte | 7.63 FLOPs/byte | 均 ≪ 295 |
| $b=1,S=0$ 延迟下界 | 2.00 ms/token（500 token/s） | 19.56 ms（51 token/s） | 9.8× |
| $b=1,S=32\text{k}$ 延迟下界 | 2.96 ms/token | 22.12 ms | 7.5× |

### 表 5：prefill 8k 分解、TTFT 与成本

| 项 | Qwen3-30B-A3B | Qwen3-32B dense | 比值 |
| --- | --- | --- | --- |
| 线性项 $2P_{\text{active}}S$ | 54.9 TFLOPs | 536.8 TFLOPs | 0.102 |
| attention 二次项 $4LS^2d$ | 26.4 TFLOPs | 88.0 TFLOPs | 0.300 |
| 合计（$S=8192$） | 81.3 TFLOPs | 624.7 TFLOPs | 0.130 |
| TTFT（989 TFLOPs、MFU 0.4） | 206 ms | 1579 ms | 7.7× |
| 算力受限吞吐（395.6 TFLOPs） | 58,995 token/s | 6,038 token/s | 9.8× |
| 成本（\$2/GPU·h） | \$0.0094/百万 token | \$0.092/百万 token | 9.8× |
| all-to-all | 128 KiB/token/层，6.00 MiB/token | 无 | NVLink 7.0 µs / IB 125.8 µs 对计算 17 µs |

### 表 6：训练与量化

| 口径 | Qwen3-30B-A3B | Qwen3-32B dense | 说明 |
| --- | --- | --- | --- |
| 训练算力比 $6N_{\text{active}}D$ | 0.102 | 1 | 平衡点 $k=9.8$ |
| 训练显存 14 B/参数 | 427 GB | 459 GB | Adam：2+4+4+4 |
| 训练显存 16 B/参数 | 489 GB | 524 GB | 含 fp32 梯度 |
| 4 bit 权重 | 15.3 GB | 16.4 GB | 全部专家都要常驻 |

### 代码：参数账本、KV、稀释表、prefill 与通信

```python
KiB, MiB, GiB, GB = 1024, 1024**2, 1024**3, 10**9

# Qwen3-30B-A3B: hidden 2048 / 48 层 / 32 个 Q 头 / 4 个 KV 头 / head_dim 128 / 专家 768 x 128 / top-8
d, L, Hq, Hkv, dh, dff, N, k, V = 2048, 48, 32, 4, 128, 768, 128, 8, 151936
# Qwen3-32B dense: hidden 5120 / 64 层 / 64 个 Q 头 / 8 个 KV 头 / FFN 25600
D2, L2, Hq2, Hkv2, dff2 = 5120, 64, 64, 8, 25600

expert  = 3 * d * dff
experts = L * N * expert
shared  = 2*V*d + L*(d*Hq*dh + 2*d*Hkv*dh + Hq*dh*d) + L*d*N + L*2*dh
total, active = shared + experts, shared + k/N*experts
attn2   = L2 * (D2*(Hq2*dh) + 2*D2*(Hkv2*dh) + (Hq2*dh)*D2)
total2  = 2*V*D2 + attn2 + L2*3*D2*dff2 + L2*2*dh

print(f"单专家 {expert/1e6:.2f}M | 专家总量 {experts/1e9:.3f}B | 共享主干 {shared/1e9:.4f}B")
print(f"MoE 总参 {total/1e9:.4f}B 激活 {active/1e9:.4f}B ({active/total:.2%}) 容量 {total/active:.2f}x")
print(f"dense 总参 {total2/1e9:.4f}B | FLOPs/token {2*active/1e9:.2f} vs {2*total2/1e9:.2f} GFLOP")
print(f"bf16 权重 {total*2/GiB:.2f} vs {total2*2/GiB:.2f} GiB | 比 {total/total2:.3f}")

k_moe, k_dense, S = 2*L*Hkv*dh*2, 2*L2*Hkv2*dh*2, 32768
print(f"KV {k_moe/KiB:.0f} vs {k_dense/KiB:.0f} KiB/token | 32k 单序列 {k_moe*S/GiB:.1f} vs {k_dense*S/GiB:.1f} GiB")
print(f"并发上限 {60*GiB/(k_moe*S):.2f}->{int(60*GiB/(k_moe*S))} vs {60*GiB/(k_dense*S):.2f}->{int(60*GiB/(k_dense*S))}")
for b in (1, 8, 32, 128):
    m, dd = active*2/GiB + b*k_moe*S/GiB, total2*2/GiB + b*k_dense*S/GiB
    print(f"b={b:3d} 每步读取 {m:7.1f} vs {dd:7.1f} GiB | {dd/m:.1f}x")
BW = 3.35e12                                        # H100 HBM 3.35 TB/s
print(f"算术强度上界 {2*active/(k_moe*S):.2f} vs {2*total2/(k_dense*S):.2f} FLOPs/byte（roofline 295）")
print(f"b=1 延迟下界 {active*2/BW*1e3:.2f} vs {total2*2/BW*1e3:.2f} ms | 32k {(active*2+k_moe*S)/BW*1e3:.2f} vs {(total2*2+k_dense*S)/BW*1e3:.2f} ms")

S8, eff = 8192, 989e12*0.4                          # prefill 8k；989 TFLOPs dense x MFU 0.4
lin_m, lin_d = 2*active*S8, 2*total2*S8
quad_m, quad_d = 4*L*S8**2*d, 4*L2*S8**2*D2
print(f"prefill 8k {lin_m/1e12:.1f}+{quad_m/1e12:.1f}={(lin_m+quad_m)/1e12:.1f} vs"
      f" {lin_d/1e12:.1f}+{quad_d/1e12:.1f}={(lin_d+quad_d)/1e12:.1f} TFLOPs 比 {(lin_m+quad_m)/(lin_d+quad_d):.3f}")
print(f"TTFT {(lin_m+quad_m)/eff*1e3:.0f} vs {(lin_d+quad_d)/eff*1e3:.0f} ms"
      f" | 吞吐 {eff/(2*active):.0f} vs {eff/(2*total2):.0f} token/s")
a2a = 4*k*d*2                                        # dispatch + combine，每 token 每层
print(f"all-to-all {a2a/KiB:.0f} KiB/token/层 | {a2a*L/MiB:.2f} MiB/token"
      f" | NVLink {a2a*L/900e9*1e6:.1f} us | IB50 {a2a*L/50e9*1e6:.1f} us | 计算 {2*active/eff*1e6:.0f} us")
print(f"训练比 {active/total2:.3f} 平衡点 k={total2/active:.1f}"
      f" | 14B/参 {total*14/GB:.0f} vs {total2*14/GB:.0f} GB | 4bit {total*0.5/GB:.1f} vs {total2*0.5/GB:.1f} GB")
```

```text
单专家 4.72M | 专家总量 28.991B | 共享主干 1.5409B
MoE 总参 30.5319B 激活 3.3528B (10.98%) 容量 9.11x
dense 总参 32.7615B | FLOPs/token 6.71 vs 65.52 GFLOP
bf16 权重 56.87 vs 61.02 GiB | 比 0.932
KV 96 vs 256 KiB/token | 32k 单序列 3.0 vs 8.0 GiB
并发上限 20.00->20 vs 7.50->7
b=  1 每步读取     9.2 vs    69.0 GiB | 7.5x
b=  8 每步读取    30.2 vs   125.0 GiB | 4.1x
b= 32 每步读取   102.2 vs   317.0 GiB | 3.1x
b=128 每步读取   390.2 vs  1085.0 GiB | 2.8x
算术强度上界 2.08 vs 7.63 FLOPs/byte（roofline 295）
b=1 延迟下界 2.00 vs 19.56 ms | 32k 2.96 vs 22.12 ms
prefill 8k 54.9+26.4=81.3 vs 536.8+88.0=624.7 TFLOPs 比 0.130
TTFT 206 vs 1579 ms | 吞吐 58995 vs 6038 token/s
all-to-all 128 KiB/token/层 | 6.00 MiB/token | NVLink 7.0 us | IB50 125.8 us | 计算 17 us
训练比 0.102 平衡点 k=9.8 | 14B/参 427 vs 459 GB | 4bit 15.3 vs 16.4 GB
```

### 代码：路由与 norm_topk_prob 的行为

```python
import torch, torch.nn.functional as F
torch.manual_seed(0)
d, N, K = 64, 128, 8                    # 与 Qwen3-30B-A3B 同构：128 选 8
W_r = torch.randn(d, N, requires_grad=True)
x = torch.randn(4096, d)
logits = x @ W_r
scores = F.softmax(logits, -1)          # 先在全 128 个专家上 softmax
top_s, _ = scores.topk(K, -1)           # 再取 top-8
gates = top_s / top_s.sum(-1, keepdim=True)     # norm_topk_prob=true：重归一化，gate 之和恒为 1
print(f"gate 之和 {gates.sum(-1).min().item():.6f} ~ {gates.sum(-1).max().item():.6f}")
print(f"每 token 只碰 {K/N:.2%} 的专家；8 选 2 是 25.00%")
(gates * torch.randn(4096, K)).sum().backward()  # 选择不可导，梯度只经加权系数回传
print(f"路由器梯度范数 {float(W_r.grad.norm()):.3e}")
```

```text
gate 之和 1.000000 ~ 1.000000
每 token 只碰 6.25% 的专家；8 选 2 是 25.00%
路由器梯度范数 1.406e+02
```

读法：gate 之和恒为 1 说明加权输出的尺度与 $k$ 无关（直接对 top-k logits 做 softmax 是等价写法，float32 下差异只在 $10^{-7}$ 量级）；梯度范数非零说明路由器的学习信号完全来自加权系数，这正是 aux loss 与 z-loss 要服务的对象。表 1 到表 6 的全部数字与上面两段脚本的输出逐项一致，口径写在每张表的表头与正文第 1 节。

## 常见追问

- **追问**：30B-A3B 是不是「3B 模型」，显存只要 6 GB 多？
  - 要点：不是。3.3528B 是每 token 参与计算的参数，且已包含全部共享主干（1.5409B），不是「3.35B 专家参数」。全部 6144 个专家的权重都要常驻，bf16 下 56.87 GiB，4 bit 后仍 15.3 GB。激活参数决定算力，总参数决定显存与容量，两本账不能混。
- **追问**：既然每 token 算力只有 dense 的 10.2%，为什么 batch=1 时不快 9.8 倍？
  - 要点：b=1 时 decode 是带宽受限的，算术强度上界只有 2.08 FLOPs/byte，远低于 roofline 295。省下的算力在这一档几乎不变速度，能兑现的只有权重读取那一半（6.25 对 61.02 GiB）；而专家权重是 128 份碎片化矩阵，grouped GEMM 在小 batch 下效率不高，再叠加路由与 all-to-all 开销（128 KiB/token/层，跨节点 125.8 µs 对计算 17 µs），实测延迟改善明显小于理论带宽比。把省下的算力变成吞吐需要 batch 大到几百。
- **追问**：为什么 Qwen3 的 `router_aux_loss_coef` 只有 0.001，比 Switch 和 Mixtral 小一个量级？共享专家又去哪了？
  - 要点：粒度不同。128 选 8 时每 token 只碰 6.25% 的专家，8 选 2 要碰 25%，前者天然更容易均衡；aux loss 防的是路由坍缩（少数专家吃掉全部 token、其余拿不到梯度），不是强制均匀。引用 $\alpha$ 必须写明模型与取值：Qwen3-30B-A3B 是 0.001，Switch 是 0.01，Mixtral 是 0.02。config 里没有 `shared_expert_intermediate_size`，说明 Qwen3 取消了共享专家，与 Qwen2-MoE、DeepSeekMoE 的「共享专家 + 细粒度专家」不是同一套设计——所以参数量账必须按「主干 + 全部专家」算，不能另加一个共享专家项。
- **追问**：长上下文高并发时，MoE 的优势还剩多少？
  - 要点：被 KV 稀释。MoE 不动 KV cache，96 对 256 KiB/token 是 2.67×；b=1、32k 时总读取比还有 7.5×，b=128 时只剩 2.8×，b 继续增大收敛到 2.67×。此时决定吞吐的是 KV 容量与读取，不是稀疏性——这也是同一批流量里 MoE 的并发上限（20 条）比 dense（7 条）高 2.67× 的价值所在。
- **追问**：什么情况下必须选 32B dense？
  - 要点：单请求低延迟或端侧、单卡部署（MoE 要常驻 128 个专家）；微调定制密集，要避开路由漂移与专家过拟合；集群通信弱、没有 all-to-all 预算；以及依赖既有量化与投机解码生态、对尾延迟稳定性要求高的场景。dense 的负载天然均匀，没有最热专家决定长尾的问题。

## 相关题目

- [[llm-internals-10]]：MoE 的通用机制、路由、负载均衡与系统代价的母题；本题是它在 Qwen 这道选型题上的落地版，重点落在「同显存换容量」。
- [[llm-internals-09]]：Chinchilla 的 20 tokens/参数口径只对训练算力最优；MoE 的缩放必须换成激活参数，否则数据预算被放大约 9 倍。
- [[mistral-03]]：Mixtral 8x7B 的 46.703B / 12.880B / 87.0 GiB 与「用显存换算力」的完整推导，是本题对照组的来源。
- [[inference-serving-02]]：continuous batching 的调度机制，解释长请求占住 slot 时 MoE 的成本优势为什么依赖大 batch。
- [[alibaba-03]]：Qwen3 的思考预算与 KV 并发上限口径，表 3 里 dense 的 7 条并发与它保持一致。

## 参考资料与归属

- **MoE 详解** —— Outcome School 博客（该条来源未署个人作者）：<https://outcomeschool.com/blog/mixture-of-experts>。MoE 的定位（只替换 FFN）、路由器与 top-k 加权求和、总参数与激活参数的区别、负载均衡与 expert capacity、MoE 不省内存的定性结论均转述自这篇。
- **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity（延伸）** —— Fedus et al.，2021-01-11：<https://arxiv.org/abs/2101.03961>。$k=1$ 的稀疏配置、auxiliary load balancing loss 及其系数 $\alpha=0.01$、capacity factor 与「丢弃率压到 1% 以下」、只提高专家 FFN dropout 的微调经验来自这篇。
- **Mixtral of Experts（延伸）** —— Jiang et al. (Mistral AI)，2024-01-08：<https://arxiv.org/abs/2401.04088>。Mixtral 8x7B 的 8 专家 top-2 配置、`router_aux_loss_coef=0.02` 与官方 46.7B / 12.9B 口径，是本题对照组的依据。
- **Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws（延伸）** —— Sardana et al.，2023-12-31：<https://arxiv.org/abs/2401.00448>。把推理成本写进目标函数后「过训小模型」与「用稀疏换容量」成为同一类决策的口径来自这篇。
- **Continuous Batching in LLMs** —— Amit Shekhar (Outcome School)：<https://outcomeschool.com/blog/continuous-batching-in-llms>。长请求在 decode 阶段长期占住 slot、每步重读权重与 KV、混池时尾延迟与吞吐互相牵制的机制来自这篇。
- **延伸来源说明**：Qwen3-30B-A3B 与 Qwen3-32B 的 config.json 取值（层数、KV 头数、`moe_intermediate_size=768`、`num_experts=128`、`num_experts_per_tok=8`、`norm_topk_prob=true`、`router_aux_loss_coef=0.001`、`tie_word_embeddings=false`、`max_position_embeddings=40960`、`decoder_sparse_step=1`）取自公开模型配置，未列入上面的来源数组；30.5319B / 3.3528B / 32.7615B / 6.71 与 65.52 GFLOP / 56.87 与 61.02 GiB / 96 与 256 KiB / 9.11× / 2.08 与 7.63 FLOPs/byte / 81.3 与 624.7 TFLOPs / 6.00 MiB/token 等数字是按该口径自行复算的结果（脚本见 `.work/lane-b/alibaba-04-recompute.py`）。DeepSeek-V3 的无辅助损失负载均衡与取消 token dropping 属于补充背景，用于说明这两个机制的上下游。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
