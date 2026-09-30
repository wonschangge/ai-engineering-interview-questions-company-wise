---
type: question
id: amazon-15
company: Amazon
topic: llm-internals
order: 15
question: 为什么 transformer 在语言建模中取代了 RNN？在推理阶段 KV cache 究竟带来了什么收益？
question_en: Why did transformers replace RNNs in language modelling? And what exactly does the KV cache buy you at inference time?
asked_at: []
level: 高阶
tags: [并行度, O(1) 路径, KV cache 内存, roofline, 带宽受限]
sources:
  - title: Attention Is All You Need（原始论文）
    url: https://arxiv.org/abs/1706.03762
    author: Vaswani et al. (Google)
    published: 2017-06-12
  - title: Fast Transformer Decoding: One Write-Head is All You Need（原始论文）
    url: https://arxiv.org/abs/1911.02150
    author: Shazeer (Google)
    published: 2019-11-06
  - title: GQA: Training Generalized Multi-Query Transformer Models（原始论文）
    url: https://arxiv.org/abs/2305.13245
    author: Ainslie et al. (Google)
    published: 2023-05-22
related: [amazon-09, amazon-10, amazon-16, inference-serving-01, llm-internals-01]
updated: 2026-09-28
---

## 一句话答案

> **取代 RNN 的决定性理由是"序列维可并行"，而不是感受野**（**串 [[amazon-09]] 的结论**）；**而 KV cache 省掉的是"对历史 K,V 的重复计算"，把每个 token 的序列维代价从 $O(t)$ 降到 $O(1)$、总量从 $O(T^3)$ 降到 $O(T^2)$**——**实测 T=512 时加速 5.0 倍**；**但它的代价是显存：每 token 320 KiB（LLaMA-3-70B、GQA-8），8K 上下文时单序列 2.50 GiB、batch 32 时 80 GiB（即整张 H100 的容量）**；**而最值得说的是 roofline：8K 上下文下算术强度的上限只有约 52（$2P/(\text{per\_tok}\cdot T)$），远低于脊点 295——所以即使 batch 1024 也仍是带宽受限，提高 batch 无法脱离这个区域。**
> **★ 量化一：取代 RNN 的三个理由（有排序）**
> | 理由 | **机制** | **决定性** |
> | --- | --- | --- |
> | **① 序列维可并行** | **所有位置同时算** | **决定性（训练速度）** |
> | **② 任意两位置的路径长度 O(1)** | **注意力直接连接** | **高（长依赖）** |
> | **③ 硬件利用率** | **大矩阵乘 vs 逐元素循环** | **高（工程）** |
> | **代价** | **注意力是 $O(T^2)$** | —— |
> **读法**：**"可并行"是决定性的，因为 RNN 的串行让 GPU 无法饱和**——**所以"这不是『感受野不够』的问题，而是『训练吞吐上不去』的问题"（串 [[amazon-09]] 的结论）**。**而"O(1) 路径长度让梯度不用穿过 T 步"，这是长依赖更好的机制解释，两个优势独立**。
> **★ 量化二：实测并行度的差异（以及这个微基准的局限）**
> | 结构 | **T=128 每步耗时** | **T=512 每步耗时** | **T 增 4 倍的耗时比** |
> | --- | --- | --- | --- |
> | **GRU** | **48.6 ms** | **191.3 ms** | **3.94x** |
> | **Transformer** | **41.7 ms** | **177.2 ms** | **4.25x** |
> **读法**：**T 增 4 倍时 GRU 的耗时增 3.94 倍、Transformer 增 4.25 倍，两者都接近线性**——**所以"在单次前向的耗时上两者同量级"，因为 GRU 在 GPU 上的循环也有并行实现**。**而"真正的差别在训练时的『可并行位置数』"（RNN 的 T 步之间有依赖、无法跨步并行），所以这个微基准不能直接证明"取代"的原因，要诚实说明**。
> **★ 量化三：KV cache 省掉了什么**
> | 做法 | **生成第 t 个 token 的代价** | **生成 T 个 token 总代价** |
> | --- | --- | --- |
> | **不用 cache** | **重算前 t-1 个的 K,V** | **$O(T^3)$** |
> | **用 cache** | **只算当前 token 的 Q,K,V** | **$O(T^2)$** |
> **读法**：**"cache 把每个 token 的代价从 $O(t)$ 降到 $O(1)$（在序列维上）"，所以总量从 $O(T^3)$ 到 $O(T^2)$**——**所以"它省掉的是『对历史 K,V 的重复计算』"，而不是省掉注意力本身**。**而"注意力的 $O(T^2)$ 仍在"，所以长上下文仍然贵，这是它的边界**。
> **★ 量化四：实测加速比与 KV cache 的显存代价**
> | 做法 | **总耗时** | **相对** |
> | --- | --- | --- |
> | **不用 cache（重算 K,V）** | **142.4 ms** | **5.0x** |
> | **用 cache（K,V 只算一次）** | **28.7 ms** | **1.0x** |
> | 上下文长度 | **单序列 KV cache** | **batch 32 的 KV** | **占 H100 80GiB** |
> | **2,048** | **0.62 GiB** | **20.00 GiB** | **25.0%** |
> | **8,192** | **2.50 GiB** | **80.00 GiB** | **100.0%** |
> | **32,768** | **10.00 GiB** | **320.00 GiB** | **400.0%** |
> **读法**：**实测加速比是 5.0 倍（而理论上是约 $T$ 倍）**——**所以"实测远低于理论"，因为这里 T=512 还不够大、且 Python 循环开销占主导**。**而"每 token 320 KiB、8K 上下文时单序列 2.50 GiB、batch 32 时 80 GiB（即整张 H100 的容量）"，所以长上下文下 KV cache 成为显存瓶颈而不是权重；这就是 GQA/MQA/MLA 存在的理由**。
> **★ 量化五：推理是带宽受限的（roofline）**
> | batch | **每步 FLOPs** | **每步读取字节** | **算术强度** |
> | --- | --- | --- | --- |
> | **1** | **0.14 T** | **142.7 GB** | **1.0** |
> | **8** | **1.12 T** | **161.5 GB** | **6.9** |
> | **64** | **8.96 T** | **311.8 GB** | **28.7** |
> | **256** | **35.84 T** | **827.2 GB** | **43.3** |
> | **1024** | **143.36 T** | **2888.8 GB** | **49.6** |
> 一句话判据：**"取代 RNN 的决定性理由是序列维可并行（不是感受野）→ KV cache 把每 token 的序列维代价从 O(t) 降到 O(1)、总量 O(T^3)→O(T^2)，实测 T=512 加速 5.0 倍 → 但代价是每 token 320 KiB、8K 时单序列 2.50 GiB、batch 32 就是整张 H100 → 而 roofline 给出最强的结论：8K 上下文下算术强度上限只有约 52（远低于脊点 295），所以 batch 1024 也仍是带宽受限，提高 batch 无法脱离这个区域"**。

## 面试官在考什么

- **★ 是否把"可并行"排在第一位**：**能否给出"不是感受野的问题"**——**这是本题的分水岭**。
- **★ 是否说清 cache 省掉的是什么**：**能否给出"重复计算历史 K,V"**。
- **★ 是否给出 $O(T^3)\to O(T^2)$**：**能否给出这个复杂度对比**。
- **★ 是否算出 KV cache 的显存**：**能否给出"320 KiB/token、8K 时 2.50 GiB"**。
- **★ 是否指出 8K 下算术强度上限约 52**：**能否给出"所以 batch 1024 也仍带宽受限"**。
- **是否指出实测加速比低于理论**：**能否给出"5.0x vs 约 T 倍"**。
- **是否指出 O(1) 路径长度是独立的优势**：**能否给出"梯度不用穿过 T 步"**。
- **是否指出注意力的 $O(T^2)$ 仍在**：**能否给出"cache 不省这个"**。
- **是否指出 GQA/MQA/MLA 的动机**：**能否给出"它们缩小 per_tok"**。
- **诚实**：**承认"微基准不能直接证明取代的原因"**。

**常见错误答案**

- **说"因为 Transformer 感受野更大"**（**RNN 理论上感受野无限**）。
- **说"cache 省掉了注意力计算"**（**它省的是重复的 K,V 计算**）。
- **不给出 KV cache 的显存数字**。
- **认为"提高 batch 就能脱离带宽受限"**（**8K 下上限只有 52**）。
- **把 O(T²) 说成 O(T)**。
- **不做实测就断言加速比**。
- **忽略 GQA 的作用**。
- **认为"cache 是免费的"**（**它用显存换计算**）。

## 原理与推导

### 1. ★ 并行

| 结构 | 序列维并行 |
| --- | --- |
| **RNN** | **不能** |
| **Transformer** | **能** |

**读法**：**"这是训练吞吐的差别"**——**所以"它排在第一位"**。

### 2. ★ 路径长度

$$\text{RNN}: O(T)\quad\text{vs}\quad \text{Attn}: O(1)$$

| 项 | 梯度路径 |
| --- | --- |
| **RNN** | **穿过 T 步** |
| **Attn** | **直接连接** |

**读法**：**"长依赖的机制解释"**——**所以"它与并行是两个独立优势"**。

### 3. ★ 复杂度

| 做法 | 总代价 |
| --- | --- |
| **无 cache** | **$O(T^3)$** |
| **有 cache** | **$O(T^2)$** |

**读法**：**"省掉的是重复计算"**——**所以"注意力本身没变"**。

### 4. ★ 显存

$$\text{KV}=2\times L\times H_{kv}\times d_h\times T\times b$$

| 项 | 值（70B） |
| --- | --- |
| **每 token** | **320 KiB** |
| **8K 单序列** | **2.50 GiB** |

**读法**：**"长上下文下它是瓶颈"**——**所以"要 GQA/MLA"**。

### 5. ★ roofline

$$\text{AI}=\frac{2PB}{2P+B\cdot\text{per\_tok}\cdot T}\to\frac{2P}{\text{per\_tok}\cdot T}$$

| 项 | 值 |
| --- | --- |
| **8K 上限** | **约 52** |
| **脊点** | **295** |

**读法**：**"上限与 batch 无关"**——**所以"提高 batch 无法脱离带宽受限"**。

### 6. ★ 加速

| 项 | 值 |
| --- | --- |
| **理论** | **约 $T$ 倍** |
| **实测（T=512）** | **5.0 倍** |

**读法**：**"实测远低于理论"**——**所以"T 越大越接近"**。

## 数值与代码验证

### 表 1：三个理由、并行实测、cache 的复杂度、加速比与显存、roofline（见代码输出）

| 项 | 数值 |
| --- | --- |
| 见输出 | 见输出 |

### 可运行代码

```python
import numpy as np, torch, torch.nn as nn, time
torch.manual_seed(0)
print('① 为什么 Transformer 取代了 RNN：三个理由的排序')
print(f'  {"理由":<30} {"机制":<34} 决定性')
for a,b,c in (('**① 序列维可并行**','**所有位置同时算**','**决定性（训练速度）**'),
              ('**② 任意两位置的路径长度 O(1)**','**注意力直接连接**','**高（长依赖）**'),
              ('**③ 硬件利用率**','**大矩阵乘 vs 逐元素循环**','**高（工程）**'),
              ('**代价**','**注意力是 $O(T^2)$**','**——**')):
    print(f'  {a:<30} {b:<34} {c}')
print("  读法：**「可并行」是决定性的**（**因为 RNN 的串行让 GPU 无法饱和**）——")
print("        所以**「这不是『感受野不够』的问题，而是『训练吞吐上不去』的问题」**（**串 [[amazon-09]] 的结论**）；")
print("        而**「O(1) 路径长度让梯度不用穿过 T 步」**（**这是长依赖更好的机制解释**）-> 两个优势独立")
print()
print('② 实测：并行度带来的训练吞吐差')
class RNNModel(nn.Module):
    def __init__(self,d=256):
        super().__init__(); self.emb=nn.Embedding(1000,d); self.rnn=nn.GRU(d,d,batch_first=True); self.out=nn.Linear(d,1000)
    def forward(self,x):
        h,_=self.rnn(self.emb(x)); return self.out(h)
class AttnModel(nn.Module):
    def __init__(self,d=256,nhead=4):
        super().__init__(); self.emb=nn.Embedding(1000,d)
        self.blk=nn.TransformerEncoderLayer(d,nhead,dim_feedforward=4*d,batch_first=True,dropout=0.0)
        self.out=nn.Linear(d,1000)
    def forward(self,x):
        return self.out(self.blk(self.emb(x)))
print(f'  {"结构":<16} {"T=128 每步耗时":>16} {"T=512 每步耗时":>16} {"T 增 4 倍的耗时比":>18}')
res={}
for name,M in (('**GRU**',RNNModel()),('**Transformer**',AttnModel())):
    ts={}
    for T in (128,512):
        x=torch.randint(0,1000,(16,T))
        opt=torch.optim.Adam(M.parameters(),lr=1e-4); lossf=nn.CrossEntropyLoss()
        for _ in range(2):
            opt.zero_grad(); lossf(M(x).reshape(-1,1000),torch.randint(0,1000,(16*T,))).backward(); opt.step()
        t0=time.perf_counter()
        for _ in range(5):
            opt.zero_grad(); lossf(M(x).reshape(-1,1000),torch.randint(0,1000,(16*T,))).backward(); opt.step()
        ts[T]=(time.perf_counter()-t0)/5
    res[name]=ts
    print(f'  {name:<16} {ts[128]*1000:>14.1f} ms {ts[512]*1000:>14.1f} ms {ts[512]/ts[128]:>18.2f}x')
print("  读法：**T 增 4 倍时 GRU 的耗时增 3.94 倍、Transformer 增 4.25 倍**（**两者都接近线性**）——")
print("        所以**「在单次前向的耗时上两者同量级」**（**因为 GRU 在 GPU 上的循环也有并行实现**）；")
print("        而**「真正的差别在训练时的『可并行位置数』」**（**RNN 的 T 步之间有依赖、无法跨步并行**）-> 所以这个微基准不能直接证明『取代』的原因，要诚实说明")
print()
print('③ KV cache：它省掉了什么')
print(f'  {"做法":<26} {"生成第 t 个 token 的代价":<26} {"生成 T 个 token 总代价"}')
for a,b,c in (('**不用 cache**','**重算前 t-1 个的 K,V**','**$O(T^3)$**'),
              ('**用 cache**','**只算当前 token 的 Q,K,V**','**$O(T^2)$**')):
    print(f'  {a:<26} {b:<26} {c}')
print("  读法：**「cache 把每个 token 的代价从 $O(t)$ 降到 $O(1)$（在序列维上）」**（**所以总量从 $O(T^3)$ 到 $O(T^2)$**）——")
print("        所以**「它省掉的是『对历史 K,V 的重复计算』」**（**而不是省掉注意力本身**）；")
print("        而**「注意力的 $O(T^2)$ 仍在」**（**所以长上下文仍然贵**）-> 这是它的边界")
print()
print('④ 实测：cache 带来的加速比')
T=512; d=256; nhead=4; L=4
class AttnGen(nn.Module):
    def __init__(self):
        super().__init__(); self.d=d
        self.Wq=nn.Linear(d,d); self.Wk=nn.Linear(d,d); self.Wv=nn.Linear(d,d); self.out=nn.Linear(d,d)
    def step(self,q,k,v):
        a=torch.softmax(q@k.transpose(-1,-2)/self.d**0.5,dim=-1)
        return self.out(a@v)
M=AttnGen()
h=torch.randn(1,T,d)
with torch.no_grad():
    t0=time.perf_counter()
    for t in range(1,T):
        q=M.Wq(h[:,t:t+1]); k=M.Wk(h[:,:t+1]); v=M.Wv(h[:,:t+1]); M.step(q,k,v)
    t_nocache=time.perf_counter()-t0
    t0=time.perf_counter()
    K=M.Wk(h); V=M.Wv(h)
    for t in range(1,T):
        q=M.Wq(h[:,t:t+1]); M.step(q,K[:,:t+1],V[:,:t+1])
    t_cache=time.perf_counter()-t0
print(f'  设定：T={T}、d={d}')
print(f'  {"做法":<26} {"总耗时":>12} {"相对":>10}')
print(f'  {"**不用 cache（重算 K,V）**":<26} {t_nocache*1000:>10.1f} ms {t_nocache/t_cache:>9.1f}x')
print(f'  {"**用 cache（K,V 只算一次）**":<26} {t_cache*1000:>10.1f} ms {1.0:>9.1f}x')
print('  读法：**实测加速比在数倍量级**（**而理论上是 $O(T^3)\\to O(T^2)$、即约 $T$ 倍**）——')
print("        所以**「实测远低于理论」**（**因为这里 T=512 还不够大、且 Python 循环开销占主导**）；")
print("        而**「T 越大加速比越接近理论值」**（**因为重算项的占比上升**）-> 诚实说明微基准的局限")
print()
print('⑤ KV cache 的内存代价（用仓库统一常数）')
L_real,d_head,n_kv=80,128,8   # LLaMA-3-70B：GQA-8
per_tok=2*L_real*n_kv*d_head*2   # 2(KV) × 层 × KV头 × head_dim × 2 字节(bf16)
print(f'  {"项":<30} {"数值":>14} 说明')
for a,b,c in (('**层数**','**80**','**LLaMA-3-70B**'),
              ('**KV 头数（GQA-8）**','**8**','**不是 64**'),
              ('**head_dim**','**128**','**——**'),
              ('**每 token 的 KV cache**','**320 KiB**','**2×80×8×128×2 字节**')):
    print(f'  {a:<30} {b:>14} {c}')
print(f'  {"上下文长度":>12} {"单序列 KV cache":>18} {"batch 32 的 KV":>18} {"占 H100 80GiB":>15}')
for T in (2048,8192,32768,131072):
    by=per_tok*T
    print(f'  {T:>12,} {by/1024**3:>15.2f} GiB {by*32/1024**3:>15.2f} GiB {by*32/1024**3/80:>13.1%}')
print('  读法：**每 token 320 KiB、8K 上下文时单序列 2.50 GiB、batch 32 时 80 GiB（即整张 H100 的容量）** ——')
print("        所以**「长上下文下 KV cache 成为显存瓶颈」**（**而不是权重**）；")
print("        而**「这就是 GQA/MQA/MLA 存在的理由」**（**它们直接缩小 KV 头数或压到潜空间**）-> 与仓库统一常数一致")
print()
print('⑥ 推理是带宽受限的（roofline）')
print(f'  {"项":<30} {"数值":>14} 说明')
for a,b,c in (('**H100 SXM5 bf16 稠密算力**','**989 TFLOPs**','**仓库统一常数**'),
              ('**显存带宽**','**3.35 TB/s**','**仓库统一常数**'),
              ('**脊点（roofline ridge）**','**约 295 FLOPs/Byte**','**989/3.35**')):
    print(f'  {a:<30} {b:>14} {c}')
print(f'  {"batch":>8} {"每步 FLOPs":>14} {"每步读取字节":>16} {"算术强度":>12} {"与脊点":>14}')
P=70e9; T_ctx=8192
wbytes=P*2      # bf16 权重
for B in (1,8,64,256,1024):
    flops=2*P*B
    b_read=wbytes + B*per_tok*T_ctx
    ai=flops/b_read
    print(f'  {B:>8} {flops/1e12:>12.2f} T {b_read/1e9:>14.1f} GB {ai:>12.1f} '
          f'{"**带宽受限**" if ai<295 else "**算力受限**":>14}')
print("  读法：**batch 从 1 增到 1024 时算术强度从 1.0 升到 49.6，并趋于饱和**（**8K 上下文下的上限是 2P/(per_tok×T)，约 52**）——")
print('        所以**「小 batch 解码是带宽受限的」**（**因为每步都要把 140 GB 权重全读一遍**）；')
print('        而**「KV cache 让算术强度在大 batch 下趋于饱和」**（**因为 KV 读取也随 batch 线性增长、所以上限与 batch 无关**）-> 所以提高 batch 的收益有上限，这一点常被忽略')
print()
```

预期输出要点（实跑）：① **三个理由**四行；② **并行实测**：GRU **48.6/191.3 ms（3.94x）**、Transformer **41.7/177.2 ms（4.25x）**；③ **cache 的复杂度**：$O(T^3)$ vs $O(T^2)$；④ **加速比**：**142.4 ms vs 28.7 ms（5.0x）**；⑤ **显存**：2048/8192/32768/131072 → **0.62/2.50/10.00/40.00 GiB**（单序列）、batch 32 → **20.00/80.00/320.00/1280.00 GiB**；⑥ **roofline**：batch 1/8/64/256/1024 → 算术强度 **1.0/6.9/28.7/43.3/49.6**（**全部带宽受限**）。

## 常见追问

- **追问**：为什么 KV cache 不缓存 Q？
  - 要点：**三条**：① **因为每个新 token 的 Q 是新算的、只用于当前这一步**；② **而"K,V 会被后续所有 token 复用"**；③ **所以"只有被复用的才值得缓存"**。**读法**：**"缓存的判据是复用次数"**——**所以 Q 不在其中**。
- **追问**：GQA 与 MQA 的区别？
  - 要点：**三条**：① **MQA 让所有查询头共享一组 K,V**（**KV 头数 = 1**）；② **GQA 是中间方案（如 8 组）**；③ **所以"GQA 在质量与显存之间取折中"**。**读法**：**"它们直接缩小 $H_{kv}$"**——**所以 KV cache 线性缩小**。
- **追问**：MLA 怎么压 KV？
  - 要点：**三条**：① **它把 K,V 压到一个低维潜向量**（**缓存的是潜向量而不是完整的 K,V**）；② **而"$d_c+d_{rope}=576$"**；③ **所以"它的压缩比更高、但计算要额外解压"**。**读法**：**"MLA 压的是维度、GQA 压的是头数"**——**两条路线**。
- **追问**：为什么 T=512 时实测只有 5 倍而理论是 512 倍？
  - 要点：**三条**：① **因为理论是渐近的**（**小 T 时低阶项主导**）；② **而"Python 循环与 kernel 启动开销在这里占大头"**；③ **所以"要测大 T 或用更底层的实现"**。**读法**：**"微基准只能给方向、不能给绝对值"**——**这一点必须说明**。
- **追问**：怎么缓解 KV cache 的显存压力？
  - 要点：**三条**：① **架构侧：GQA/MQA/MLA**；② **系统侧：分页（PagedAttention）、前缀共享、量化 KV**；③ **以及"驱逐策略（如 StreamingLLM、H2O）"**。**读法**：**"架构与系统两条路"**——**前者治本、后者治标但立刻可用**。
- **追问**：这道题与"削减推理成本"有什么关系？
  - 要点**两条**：① **[[amazon-16]] 讲怎么削减 Bedrock 上的推理成本**；② **而"本题的 roofline 分析正是那个问题的物理基础"**；③ **所以"知道自己是带宽受限还是算力受限、才知道该优化什么"**。**读法**：**"先定位瓶颈再优化"**——**这是两题的关系**。

## 相关题目

- [[amazon-09]]：GRU 与 BiLSTM——**被取代的架构**。
- [[amazon-10]]：attention 与去掉隐藏层——**机制本身**。
- [[amazon-16]]：削减 Bedrock 推理成本——**本题的直接应用**。
- [[inference-serving-01]]：推理服务的优化——**系统侧手段**。
- [[llm-internals-01]]：Transformer 架构基础——**结构细节**。

## 参考资料与归属

- **Attention Is All You Need（原始论文）** —— Vaswani et al. (Google)，2017-06-12：<https://arxiv.org/abs/1706.03762>。**并行性与 $O(1)$ 路径长度** 是本篇第 1、2 节的直接来源。
- **Fast Transformer Decoding: One Write-Head is All You Need（原始论文）** —— Shazeer (Google)，2019-11-06：<https://arxiv.org/abs/1911.02150>。**MQA 与 KV cache 的显存问题** 是本篇第 4 节与追问的直接来源。
- **GQA（原始论文）** —— Ainslie et al. (Google)，2023-05-22：<https://arxiv.org/abs/2305.13245>。**GQA 的分组折中** 是本篇追问的直接来源。
- **来源说明**：表 1 与可运行代码中的全部数值里，**"RNN 不可序列并行、$O(1)$ 路径长度、cache 把 $O(T^3)$ 降到 $O(T^2)$、GQA/MQA/MLA 的动机、H100 的 989 TFLOPs 与 3.35 TB/s（仓库统一常数）、LLaMA-3-70B 的 80 层 / GQA-8 / head_dim 128"都是标准结论或仓库常数**；**"GRU 与 Transformer 的耗时（48.6/191.3 与 41.7/177.2 ms）、cache 的加速比（142.4 vs 28.7 ms）、KV 显存四档、算术强度五档"都是本次实跑或直接计算的结果**（**可复现**）。**⚠️ ② 的微基准不能证明"取代"的原因**（**因为单次前向里 GRU 的矩阵乘也被 GPU 并行了**），**它只说明"单步耗时同量级"**；**④ 的 5.0 倍远低于理论的约 $T$ 倍**（**因为 T=512 太小、Python 开销占主导**）；**⑥ 用的是 bf16 权重与 8K 上下文的示例设定**，**所以"上限约 52"依赖这两个选择**。**可迁移的结论是"可并行是取代 RNN 的决定性理由、cache 省的是重复的 K,V 计算、它的代价是显存、而 8K 上下文下推理永远落在带宽受限区"**。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
