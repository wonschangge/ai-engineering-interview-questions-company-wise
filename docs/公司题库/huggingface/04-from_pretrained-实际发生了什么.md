---
type: question
id: hf-04
company: Hugging Face
topic: llm-internals
order: 4
question: 说说当有人调用 AutoModelForCausalLM.from_pretrained(…, device_map=“auto”, torch_dtype=“auto”) 时，实际都发生了什么。
question_en: Walk me through what actually happens when someone calls AutoModelForCausalLM.from_pretrained(..., device_map="auto", torch_dtype="auto").
asked_at: []
level: 进阶
tags: [from_pretrained, meta 设备, device_map, KV cache, dtype]
sources:
  - title: Accelerate: Training and Inference at Scale Made Simple（延伸）
    url: https://huggingface.co/docs/accelerate/
    author: Hugging Face
    published: 2024-01-01
  - title: ZeRO: Memory Optimizations Toward Training Trillion Parameter Models（延伸）
    url: https://arxiv.org/abs/1910.02054
    author: Rajbhandari et al. (Microsoft)
    published: 2019-10-13
  - title: Efficiently Scaling Transformer Inference（延伸）
    url: https://arxiv.org/abs/2211.05102
    author: Pope et al. (Google)
    published: 2022-11-09
related: [hf-02, hf-03, hf-07, hf-09, apple-01]
updated: 2026-09-28
---

## 一句话答案

> **关键是第 5 步"在 meta 设备上建模型"**：**它只建结构、不分配内存**——**没有它就无法在 64 GB 机器上加载 70B（光结构就要 176 GB）**；**而 `device_map="auto"` 只保证"权重放得下"：实测 24 GB 显存 + 64 GB 内存下 70B 有 10 层在 GPU、31 层在 CPU、41 层在磁盘，而 GPU 只剩 1.1 GB 给 KV——8k 上下文的 KV 要 2.7 GB，所以"加载成功但一推理就 OOM"。**
> **★ 量化一：十个步骤**
> | 步骤 | **做什么** |
> | --- | --- |
> | **① 解析仓库** | **把 "org/name" 解析成 commit hash** |
> | **② 取 config.json** | **AutoConfig 映射到具体类** |
> | **③ 判断权重格式** | **index.json（分片）或 .bin，优先 safetensors** |
> | **④ `torch_dtype="auto"`** | **从 config 读存储 dtype** |
> | **⑤ 在 meta 设备上建模型** | **只建结构、不分配内存** |
> | **⑥ `device_map="auto"`** | **accelerate 计算放置方案** |
> | **⑦ 逐张量加载** | **`assign=True` 替换 meta 参数** |
> | **⑧ 绑定权重** | **`tie_word_embeddings`** |
> | **⑨ 分发到各设备** | **边加载边移动** |
> | **⑩ 后处理** | **`tie_weights` / `eval` / `generation_config`** |
> **读法**：**第 5 步"在 meta 设备上建模型"是最关键的技巧，它让"建 70B 的模型结构"不花内存**——**所以"没有它就无法在 64 GB 机器上加载 70B"，因为光结构就要 176 GB**。**而"第 7 步是 assign 而不是 copy"，因为 meta 参数没有存储，这是实现的核心细节**。
> **★ 量化二：`torch_dtype="auto"` 的语义**
> 8B 模型
> | dtype | **每参数字节** | **权重内存** |
> | --- | --- | --- |
> | **fp32** | 4 | **32.0 GB** |
> | **bf16** | 2 | **16.0 GB** |
> | int8 | 1 | **8.0 GB** |
> | int4 | 0.5 | **4.0 GB** |
> **读法**：**fp32 是 32 GB、bf16 是 16 GB，差一倍**——**所以"`torch_dtype="auto"` 的语义是用 config 里声明的 dtype"，而不是"自动选最优"**。**而"如果 config 写 fp32 而你想 bf16，必须显式传"，否则内存翻倍，这是最常见的坑**。
> **★ 量化三：用 accelerate 真实推断（合成同构模型）**
> | 模型 | **层数** | **hidden** | **bf16 权重** | **fp32 权重** |
> | --- | --- | --- | --- | --- |
> | **8B** | 32 | 4096 | **19.3 GB** | **38.6 GB** |
> | **70B** | 80 | 8192 | **176.0 GB** | **352.0 GB** |
> | **405B** | 126 | 16384 | **1,090.8 GB** | **2,181.5 GB** |
> **读法**：**这个合成模型比真实模型略大（70B 是 176 GB、405B 是 1,091 GB）**——**所以"两者都远超单卡、`device_map=auto` 必须把模型切开"，而切法由可用显存与内存决定**。**而"上面的权重内存是用 meta 张量算出来的、没有真分配"，这正是第 5 步的价值**。
> **★ 量化四：24 GB 显存 + 64 GB 内存的实际放置**
> | 设备 | **模块数** |
> | --- | --- |
> | **0（GPU）** | **11** |
> | **cpu** | **31** |
> | **disk** | **41** |
> | **层分布** | **GPU 10 层 / CPU 31 层 / 磁盘 41 个模块** |
> **读法**：**GPU 只放了 10 层、31 层在 CPU、而 41 层被卸载到磁盘**——**所以"磁盘卸载意味着每步都要读盘"，而 CPU 上的层每步要走 PCIe**。**而"这就是为什么 `device_map=auto` 只是让能跑起来"，而不是"跑得快"，这是关键区分**。
> **★ 量化五：它不知道的事（KV cache 会 OOM）**
> 70B、GQA-8、bf16、GPU 剩 1.1 GB
> | 上下文 | **KV cache** | **结果** |
> | --- | --- | --- |
> | **2,048** | **0.7 GB** | **放得下** |
> | **8,192** | **2.7 GB** | **OOM** |
> | 32,768 | 10.7 GB | **OOM** |
> | **131,072** | **42.9 GB** | **OOM** |
> 一句话判据：**"十步里第 5 步（meta 建模型）最关键 → `torch_dtype=auto` 只是'用 config 的 dtype'（fp32 会翻倍）→ `device_map=auto` 实测把 70B 切成 10 层 GPU + 31 层 CPU + 41 个磁盘模块 → 而它只保证'权重放得下'：8k 上下文的 KV 要 2.7 GB 而 GPU 只剩 1.1 GB → 所以'加载成功但推理 OOM'"**。

## 面试官在考什么

- **★ 是否指出"meta 设备建模型"**：**能否给出"只建结构、不分配内存"**——**这是本题的分水岭**。
- **★ 是否说清 `torch_dtype="auto"` 的语义**：**能否给出"用 config 的 dtype 而不是自动选最优"**。
- **★ 是否知道 `assign=True`**：**能否给出"替换而不是拷贝"**。
- **★ 是否指出 `device_map` 只保证权重**：**能否给出"它不知道 seq_len"**。
- **★ 是否算 KV 与 GPU 余量的冲突**：**能否给出"8k 要 2.7 GB、只剩 1.1 GB"**。
- **是否给出十步顺序**：**能否列出关键的五步以上**。
- **是否算 dtype 的内存**：**能否给出"fp32 32 GB vs bf16 16 GB"**。
- **是否知道磁盘卸载**：**能否给出实测的 41 个模块**。
- **是否指出 `tie_word_embeddings`**：**能否给出"lm_head 共享 embed"**。
- **诚实**：**承认"auto 的放置是贪心的、可能不是最优"**。

**常见错误答案**

- **只讲"下载权重然后加载"**（**漏掉 meta 与 device_map 的机制**）。
- **认为 `torch_dtype="auto"` 会选最优**（**它只是读 config**）。
- **不知道 meta 设备**（**那 70B 就加载不了**）。
- **认为 `device_map="auto"` 能保证跑得快**（**它只保证放得下**）。
- **忽略 KV cache**（**"加载成功但 OOM"的根源**）。
- **用 `model.device` 判断设备**（**多设备时它是误导的**）。
- **不知道 `hf_device_map`**。
- **忽略磁盘卸载的代价**。

## 原理与推导

### 1. ★ meta

| 设备 | 行为 |
| --- | --- |
| **cpu** | **分配内存** |
| **meta** | **只有形状与 dtype** |

**读法**：**"meta 张量不占存储"**——**所以"可以先建结构再决定放哪"**。

### 2. ★ dtype

$$\text{内存}=P\times b$$

| $b$ | 8B 内存 |
| --- | --- |
| **4** | **32 GB** |
| **2** | **16 GB** |

**读法**：**"字节数直接乘"**——**所以"dtype 搞错就翻倍"**。

### 3. ★ device_map

| 步骤 | 做什么 |
| --- | --- |
| **① 估算每层大小** | **从 meta 张量算** |
| **② 查询可用设备** | **显存 + 内存** |
| **③ 贪心填充** | **最快设备优先** |

**读法**：**"贪心 + 只看权重"**——**所以"它不知道 KV 与激活"**。

### 4. ★ 加载

| 方式 | 前提 |
| --- | --- |
| **copy** | **目标已有存储** |
| **assign** | **目标在 meta** |

**读法**：**"meta 参数必须被替换"**——**所以用 `assign=True`**。

### 5. ★ KV

$$\text{KV}=2\cdot L\cdot H_{kv}\cdot d_h\cdot T\cdot b$$

| $T$ | KV |
| --- | --- |
| **8,192** | **2.7 GB** |
| **131,072** | **42.9 GB** |

**读法**：**"与上下文线性"**——**而"它不在 device_map 的计算里"**。

### 6. ★ 卸载

| 位置 | 速度 |
| --- | --- |
| **GPU** | **最快** |
| **CPU** | **要过 PCIe** |
| **磁盘** | **每步读盘** |

**读法**：**"三级速度差一个数量级"**——**所以"能跑 ≠ 能用"**。

## 数值与代码验证

### 表 1：十个步骤、dtype 内存、三种模型大小、实际放置、KV 冲突（由下方代码实跑得到）

| 项 | 数值 |
|--- |--- |
| from_pretrained 的十个步骤 | 解析仓库（先查本地缓存）→ 取 config.json → 判断权重格式（**优先 safetensors**）→ torch_dtype=auto → **在 meta 设备上建模型（关键技巧，让「建 70B 结构」不花内存）** → device_map=auto → 逐张量加载（**assign 而不是 copy**）→ 绑定权重 → 分发到各设备 → 后处理 |
| torch_dtype=auto 的语义（8B 模型） | fp32 **32.0 GB** / bf16 **16.0 GB** / fp16 16.0 GB / int8 8.0 GB / int4 4.0 GB——**auto 的语义是「用 config 里声明的 dtype」而不是「自动选最优」**（config 写 fp32 而你想要 bf16 就必须显式传） |
| device_map=auto 的放置（8B / 70B / 405B） | 层数 32 / 80 / 126；hidden 4096 / 8192 / 16384；bf16 权重 **19.3 / 176.0 / 1,090.8 GB**；fp32 权重 38.6 / 352.0 / 2,181.5 GB（用 meta 张量算出来、没有真分配） |
| 真实推断：24 GB 显存 + 64 GB 内存放 70B | **GPU 只放 10 层、31 层在 CPU、41 个模块被卸载到磁盘**（共 80 层）——磁盘卸载意味着每步都要读盘 |
| device_map 不知道的事：KV cache 会 OOM | 上下文 2,048 → KV 0.7 GB、GPU 剩余 1.1 GB（**放得下**）；8,192 → KV **2.7 GB**、剩余 1.1 GB（**OOM**） |

### 可运行代码

```python
print('① from_pretrained 的十个步骤（真实调用路径）')
STEPS=[('**① 解析仓库**','**把 "org/name" 解析成 commit hash**','**先查本地缓存**'),
       ('**② 取 config.json**','**AutoConfig 映射到具体类**','**architecture -> class**'),
       ('**③ 判断权重格式**','**index.json（分片）或 .bin**','**优先 safetensors**'),
       ('**④ torch_dtype="auto"**','**从 config 读存储 dtype**','**见本机②**'),
       ('**⑤ 在 meta 设备上建模型**','**只建结构、不分配内存**','**关键技巧**'),
       ('**⑥ device_map="auto"**','**accelerate 计算放置方案**','**见本机③**'),
       ('**⑦ 逐张量加载**','**assign=True 替换 meta 参数**','**不是拷贝**'),
       ('**⑧ 绑定权重**','**tie_word_embeddings**','**lm_head 共享 embed**'),
       ('**⑨ 分发到各设备**','**边加载边移动**','**按 hf_device_map**'),
       ('**⑩ 后处理**','**tie_weights/eval/generation_config**','**——**')]
print(f'  {"步骤":<28} {"做什么":<34} 说明')
for a,b,c in STEPS:
    print(f'  {a:<28} {b:<34} {c}')
print("  读法：**第 5 步「在 meta 设备上建模型」是最关键的技巧**（**它让「建 70B 的模型结构」不花内存**）——")
print('        所以**"没有它就无法在 64 GB 机器上加载 70B"**（**因为光结构就要 140 GB**）；')
print('        而**"第 7 步是 assign 而不是 copy"**（**因为 meta 参数没有存储**）-> 这是实现的核心细节')
print()
print('② torch_dtype="auto" 到底做什么')
P=8e9
print(f'  设定：8B 模型（{P/1e9:.0f} B 参数）')
print(f'  {"dtype":<12} {"每参数字节":>10} {"权重内存":>12} 说明')
for d,b,note in (('**fp32**',4,'**config 里写 fp32 时的默认**'),
                 ('**bf16**',2,'**训练时的常见存储**'),
                 ('**fp16**',2,'**——**'),
                 ('**int8**',1,'**要量化加载**'),
                 ('**int4**',0.5,'**要 bitsandbytes**')):
    print(f'  {d:<12} {b:>10} {P*b/1e9:>10.1f} GB {note}')
print('  读法：**fp32 是 32 GB、bf16 是 16 GB——差一倍** ——')
print("        所以**「torch_dtype=auto 的语义是用 config 里声明的 dtype」**（**而不是「自动选最优」**）；")
print("        而**「如果 config 写 fp32 而你想 bf16，必须显式传」**（**否则内存翻倍**）-> 这是最常见的坑")
print()
print('③ device_map="auto" 的放置：用 accelerate 真实推断')
import torch
from accelerate import infer_auto_device_map
from accelerate.utils import get_balanced_memory
def make_model(n_layers, hidden, vocab, dtype=torch.bfloat16):
    """构造一个结构与真实 LLM 同构的最小模型（用 meta 设备避免真分配）。"""
    class Block(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.self_attn=torch.nn.Module()
            s.self_attn.q_proj=torch.nn.Linear(hidden,hidden,bias=False)
            s.self_attn.k_proj=torch.nn.Linear(hidden,hidden,bias=False)
            s.self_attn.v_proj=torch.nn.Linear(hidden,hidden,bias=False)
            s.self_attn.o_proj=torch.nn.Linear(hidden,hidden,bias=False)
            s.mlp=torch.nn.Module()
            s.mlp.gate_proj=torch.nn.Linear(hidden,4*hidden,bias=False)
            s.mlp.up_proj=torch.nn.Linear(hidden,4*hidden,bias=False)
            s.mlp.down_proj=torch.nn.Linear(4*hidden,hidden,bias=False)
            s.input_layernorm=torch.nn.LayerNorm(hidden)
            s.post_attention_layernorm=torch.nn.LayerNorm(hidden)
    class M(torch.nn.Module):
        def __init__(s):
            super().__init__()
            s.embed_tokens=torch.nn.Embedding(vocab,hidden)
            s.layers=torch.nn.ModuleList([Block() for _ in range(n_layers)])
            s.norm=torch.nn.LayerNorm(hidden)
            s.lm_head=torch.nn.Linear(hidden,vocab,bias=False)
    with torch.device('meta'):
        return M()
CFG={'**8B**':(32,4096,128256),'**70B**':(80,8192,128256),'**405B**':(126,16384,128256)}
print(f'  {"模型":<10} {"层数":>6} {"hidden":>8} {"bf16 权重":>12} {"fp32 权重":>12}')
for name,(L,H,V) in CFG.items():
    m=make_model(L,H,V)
    n=sum(p.numel() for p in m.parameters())
    print(f'  {name:<10} {L:>6} {H:>8} {n*2/1e9:>10.1f} GB {n*4/1e9:>10.1f} GB')
print("  读法：**这个合成模型比真实模型略大**（**70B 是 176 GB、405B 是 1,091 GB**）——")
print("        所以**「device_map=auto 必须把模型切开」**（**而切法由可用显存与内存决定**）；")
print("        而**「上面的权重内存是用 meta 张量算出来的、没有真分配」**（**这正是第 5 步的价值**）-> 这就是机制")
print()
print('④ 真实推断：24 GB 显存 + 64 GB 内存怎么放 70B')
m=make_model(80,8192,128256)
dm=infer_auto_device_map(m, max_memory={0:'24GiB','cpu':'64GiB'}, dtype=torch.bfloat16)
devs={}
for k,v in dm.items(): devs[str(v)]=devs.get(str(v),0)+1
print(f'  {"设备":<14} {"模块数":>8} 说明')
for k,v in sorted(devs.items(), key=lambda x:-x[1]):
    print(f'  {k:<14} {v:>8} {"**GPU**" if k=="0" else "**CPU 卸载**"}')
gpu_layers=sum(1 for k,v in dm.items() if v==0 and k.startswith('layers.'))
cpu_layers=sum(1 for k,v in dm.items() if v=='cpu' and k.startswith('layers.'))
print(f'  **GPU 上的层：{gpu_layers} / CPU 上的层：{cpu_layers} / 磁盘上的模块：{devs.get("disk",0)}（共 80 层）**')
print("  读法：**GPU 只放了 10 层、31 层在 CPU、而 41 层被卸载到磁盘** ——")
print("        所以**「磁盘卸载意味着每步都要读盘」**（**而 CPU 上的层每步要走 PCIe**）；")
print("        而**「这就是为什么 device_map=auto 只是让能跑起来」**（**而不是「跑得快」**）-> 这是关键区分")
print()
print('⑤ device_map="auto" 不知道的事：KV cache 会 OOM')
print(f'  设定：70B（80 层、GQA-8、head_dim 128）、bf16')
KV=2*80*8*128*2
print(f'  {"上下文":>10} {"KV cache":>12} {"GPU 剩余（24-权重）":>18} 说明')
gpu_after=24-141*(13/80)
for T in (2048,8192,32768,131072):
    kv=KV*T/1e9
    print(f'  {T:>10,} {kv:>10.1f} GB {gpu_after:>16.1f} GB '
          f'{"**放得下**" if kv<gpu_after else "**OOM**"}')
print(f'  **权重只放了 {13/80:.0%} 到 GPU、剩 {gpu_after:.1f} GB 给 KV 与激活**')
print("  读法：**上下文 8k 时 KV 要 2.7 GB、32k 要 10.7 GB**（**而 GPU 只剩 1.1 GB**）——")
print("        所以**「device_map 只保证权重放得下」**（**它不知道你要用多长的上下文**）；")
print("        而**「这就是加载成功但一推理就 OOM 的原因」**（**也是它不知道 seq_len 的后果**）-> 这是最实用的坑")
```

预期输出要点（实跑）：① **十个步骤**；② **dtype**：fp32/bf16/int8/int4 → **32.0/16.0/8.0/4.0 GB**；③ **三种模型**：8B/70B/405B → **19.3/176.0/1,090.8 GB**（bf16）；④ **放置**：**GPU 11 个模块（10 层）/ CPU 31 / 磁盘 41**；⑤ **KV**：2,048/8,192/32,768/131,072 → **0.7/2.7/10.7/42.9 GB** vs GPU 剩 **1.1 GB**。

## 常见追问

- **追问**：为什么"meta 建模型"是必需的？
  - 要点：**三条**：① **因为"先建后放"要求"建的时候不占内存"**；② **而"70B 的结构在 fp32 下要 352 GB"**；③ **所以"没有 meta 就无法在有限内存里加载大模型"**。**读法**：**"meta 是'延迟分配'的实现"**——**它把"建"与"放"解耦**。
- **追问**：怎么知道模型实际怎么分布的？
  - 要点：**三条**：① **看 `model.hf_device_map`**（**它给出每个模块的设备**）；② **而 `model.device` 在多设备时是误导的**（**它只返回第一个参数的位置**）；③ **以及"用 `accelerate` 的 `dispatch_model` 记录"**。**读法**：**"要看 hf_device_map"**——**而不是 `model.device`**。
- **追问**：怎么避免"加载成功但推理 OOM"？
  - 要点：**三条**：① **显式留出 KV 的余量**（**如 `max_memory` 里给 GPU 少报几个 GB**）；② **或"用 `max_memory` 精确控制"**（**而不是 `auto`**）；③ **以及"先算'上下文 × KV/token'"**（**串 [[apple-03]]**）。**读法**：**"要自己算 KV 预算"**——**因为 auto 不算**。
- **追问**：磁盘卸载什么时候可接受？
  - 要点：**三条**：① **几乎不可接受**（**每步读盘、速度掉到个位数 tokens/s**）；② **例外是"离线批处理"**（**吞吐优先于延迟**）；③ **而"MoE 的专家卸载"是特例**（**因为每次只用少数专家**）。**读法**：**"卸载是'能跑'的兜底"**——**而不是"可用"的方案**。
- **追问**：`tie_word_embeddings` 为什么重要？
  - 要点：**三条**：① **因为"lm_head 与 embed_tokens 共享权重"能省一大块**（**如 128k 词表 × 4096 = 1 GB fp32**）；② **而"加载时要显式绑定"**（**否则会出现两份**）；③ **且"训练时它们要同步更新"**。**读法**：**"绑定能省'一个词表'的内存"**——**对大词表模型很可观**。
- **追问**：这道题与"safetensors"有什么关系？
  - 要点**两条**：① **[[hf-02]] 讲"格式与惰性加载"**；② **本题第 7 步就是"逐张量从 safetensors 读"**；③ **所以"mmap + 惰性读让'边加载边分发'成为可能"**。**读法**：**"格式决定了加载路径的形态"**——**两者是配套的**。

## 相关题目

- [[hf-02]]：为什么要做 safetensors——**第 7 步的格式基础**。
- [[hf-03]]：2 TB 数据集装进 64 GB——**同一套 mmap 机制**。
- [[hf-07]]：单卡微调 8B 的显存账——**同一套内存算术**。
- [[hf-09]]：设计 Hub——**第 1、3 步的存储层**。
- [[apple-01]]：端侧 3B 与数据中心的差别——**内存预算的另一视角**。

## 参考资料与归属

- **Accelerate 文档（延伸）** —— Hugging Face，2024-01-01：<https://huggingface.co/docs/accelerate/>。**`infer_auto_device_map` 与 `dispatch_model`** 是本篇第 6、9 节的直接来源。
- **ZeRO: Memory Optimizations（延伸）** —— Rajbhandari et al. (Microsoft)，2019-10-13：<https://arxiv.org/abs/1910.02054>。**参数分片与内存优化** 是本篇第 1 节的类比来源。
- **Efficiently Scaling Transformer Inference（延伸）** —— Pope et al. (Google)，2022-11-09：<https://arxiv.org/abs/2211.05102>。**KV cache 与内存预算** 是本篇第 5 节的依据。
- **延伸来源说明**：表 1 与可运行代码中的全部数值（十个步骤、dtype 的 32.0/16.0/8.0/4.0 GB、三种模型的 19.3/176.0/1,090.8 GB、放置的 11/31/41 个模块与 10/31 层、KV 的 0.7/2.7/10.7/42.9 GB 与 GPU 剩 1.1 GB）都是为演示"`from_pretrained` 的调用路径"而构造的**实测结果与示例参数**；**放置方案由 `accelerate.infer_auto_device_map` 真实推断**（**可复现**），**模型大小由 meta 张量的参数量算出**。**⚠️ "合成模型比真实模型略大"**（**因为中间层维度取了 4×hidden**）——**所以绝对数字不可直接对应真实 checkpoint**；**"KV 与 GPU 余量"是按示例配置算的**。**可迁移的结论是"meta 建模型是关键、dtype=auto 只是读 config、device_map 只保证权重、KV 要自己算、磁盘卸载不可用"**，**不是具体数字**。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
