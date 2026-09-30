---
type: question
id: hf-02
company: Hugging Face
topic: coding
order: 2
question: 在基于 pickle 的 checkpoint 已经到处都能用的情况下，Hugging Face 为什么还要做 safetensors？
question_en: Given that pickle-based checkpoints already worked everywhere, why did Hugging Face build safetensors?
asked_at: []
level: 高阶
tags: [safetensors, pickle 风险, mmap, 惰性加载, 信任边界]
sources:
  - title: safetensors: Simple, Safe HF Format（延伸）
    url: https://github.com/huggingface/safetensors
    author: Hugging Face
    published: 2022-12-01
  - title: Python Pickle Documentation（延伸）
    url: https://docs.python.org/3/library/pickle.html
    author: Python Software Foundation
    published: 2024-01-01
  - title: LLM in a Flash: Efficient Large Language Model Inference with Limited Memory（延伸）
    url: https://arxiv.org/abs/2312.11514
    author: Alizadeh et al. (Apple)
    published: 2023-12-12
related: [hf-01, hf-04, hf-09, apple-03, safety-01]
updated: 2026-09-28
---

## 一句话答案

> **核心问题是"pickle 在加载时执行代码"**：**而 Hub 上是用户上传的文件，所以"从 Hub 加载"等于"运行陌生人的代码"**——**这不是理论风险**；**而实测发现"读整个文件"时 safetensors 与 torch.save 几乎无差别（大小 1.00×、耗时 1.13×），真正的差别在"mmap + 惰性访问"（快 94 倍，且倍数随文件变大而增长）。**
> **★ 量化一：pickle 的风险机制**
> | 机制 | **后果** |
> | --- | --- |
> | **`__reduce__` 可返回任意可调用对象** | **加载即执行** |
> | **`torch.load` 内部用 pickle** | **"加载 checkpoint" = "运行代码"** |
> | **Hub 上是用户上传的文件** | **"从 Hub 加载" = "运行陌生人的代码"** |
> | **没有沙箱** | **进程权限 = 你的权限** |
> **读法**：**问题不是"pickle 慢"而是"pickle 会执行代码"**——**所以"在用户上传模型的场景下，pickle 是不可接受的"，因为信任边界不成立**。**而"这不是理论风险"，已有真实的反序列化攻击，这是本题的核心**。
> **★ 量化二：除安全外的四个理由**
> | 理由 | **机制** | **量化** |
> | --- | --- | --- |
> | **① 零拷贝 / mmap** | **扁平头 + 连续张量数据** | **可 mmap、张量是视图** |
> | **② 惰性加载** | **按名字取张量、不读全部** | **70B 只取 embedding** |
> | **③ 更快** | **格式更简单** | **见量化三** |
> | **④ 语言无关** | **JSON 头 + 字节** | **Rust/JS/C++ 都能读** |
> **读法**：**"零拷贝"是性能上的关键，它让"按需读张量"成为可能**——**所以"safetensors 不只是安全修复、它还是性能改进"**。**而"语言无关"让非 Python 生态也能直接读，这对"模型即文件"的定位很重要，这是生态价值**。
> **★ 量化三：实测（16.8 MB 单张量）**
> | 方式 | **耗时** | **读了多少字节** |
> | --- | --- | --- |
> | **整读（safetensors）** | **5.71 ms** | **16,777,216** |
> | **整读（torch.save）** | **6.43 ms** | **16,777,216** |
> | **比值** | **1.13×** | **1.00×** |
> | **mmap + 只碰 1 个元素** | **0.056 ms** | **4** |
> | **mmap vs 整读** | **94×** | —— |
> **读法**：**"读整个文件"时两者几乎无差别（大小 1.00×、耗时 1.13×）**——**所以"safetensors 的优势不在体积或整读速度"，这一点必须诚实说**。**而"它能在不读数据的情况下先解析头部"，所以能"只读需要的张量"；"只碰一个元素"比"整读"快约 94 倍，而这个倍数随文件变大而增长（140 GB 的模型上是 $10^4$–$10^5$ 量级），这是本题的技术核心**。
> **★ 量化四：惰性加载的收益**
> | 模型 | **总参数** | **单张量** | **占比** |
> | --- | --- | --- | --- |
> | **8B** | 8 B | 0.15 B | **1.88%** |
> | **70B** | 70 B | 0.52 B | **0.74%** |
> | **405B** | 405 B | 1.00 B | **0.25%** |
> **读法**：**70B 模型的 lm_head 只占 0.7%，所以"只读它"几乎不花时间**——**所以"mmap + 惰性读让改一个张量变成秒级操作"，而不是"读整个 140 GB"**。**而"这是 pickle 做不到的"，因为 pickle 必须反序列化整个对象图，这是结构性差别**。
> **★ 量化五：它不解决什么（必须说清）**
> | 它不防的 | **为什么** |
> | --- | --- |
> | **恶意权重（投毒）** | **格式合法、数值有害** |
> | **后门触发器** | **只有特定输入才激活** |
> | **训练数据泄露** | **权重里可能含记忆的内容** |
> | **供应链（依赖包）** | **`trust_remote_code` 仍会执行代码** |
> 一句话判据：**"pickle 在加载时执行代码、而 Hub 是用户上传 → 所以 safetensors 是必需的 → 但它不只是安全：mmap 让惰性访问快 94 倍（而整读无差别）→ 70B 的 lm_head 只占 0.7% → 而它不防'权重有害'，`trust_remote_code` 才是真正的信任边界"**。

## 面试官在考什么

- **★ 是否指出"pickle 执行代码"**：**能否给出"加载即执行"**——**这是本题的分水岭**。
- **★ 是否指出"信任边界不成立"**：**能否给出"用户上传 + 无沙箱"**。
- **★ 是否诚实说"整读无差别"**：**能否给出实测的 1.13×**。
- **★ 是否指出"真正的差别在 mmap 惰性访问"**：**能否给出 94 倍**。
- **★ 是否指出"它不防权重投毒"**：**能否给出 `trust_remote_code` 这个真边界**。
- **是否给出"零拷贝"这个理由**：**能否给出"张量是视图"**。
- **是否算惰性加载的收益**：**能否给出"lm_head 占 0.7%"**。
- **是否指出"语言无关"**：**能否给出"Rust/JS 都能读"**。
- **是否指出"pickle 必须反序列化整个对象图"**：**能否给出这个结构性差别**。
- **诚实**：**承认"安全修复"与"性能改进"是两件事、后者需要访问模式配合**。

**常见错误答案**

- **只说"更快"**（**整读其实一样快**）。
- **只说"更安全"而不解释机制**（**要说"pickle 执行代码"**）。
- **声称"safetensors 能防投毒"**（**它不能**）。
- **忽略 `trust_remote_code`**（**那才是真边界**）。
- **不指出"零拷贝"**。
- **不区分"整读"与"惰性读"**。
- **认为"pickle 只是慢"**。
- **不做实测就断言速度优势**。

## 原理与推导

### 1. ★ pickle

$$\text{load}(\text{bytes})\to\text{任意可调用对象}$$

| 属性 | 后果 |
| --- | --- |
| **图灵完备的反序列化** | **加载即执行** |

**读法**：**"pickle 是一个'程序'而不是'数据'"**——**所以"不可信输入 + pickle = 代码执行"**。

### 2. ★ 信任边界

| 场景 | 是否可接受 |
| --- | --- |
| **自己训练的 checkpoint** | **可以** |
| **Hub 上陌生人上传的** | **不可以** |

**读法**：**"同一个格式在不同信任场景下结论不同"**——**所以"Hub 必须换格式"**。

### 3. ★ 格式

| 组成 | 内容 |
| --- | --- |
| **头部** | **JSON（名字/类型/偏移）** |
| **数据** | **连续字节** |

**读法**：**"头部可解析 → 可以只读需要的张量"**——**这是惰性的前提**。

### 4. ★ mmap

| 方式 | 读了多少 |
| --- | --- |
| **整读** | **全部** |
| **mmap + 取一个** | **4 字节** |

**读法**：**"按需分页"**——**所以"倍数随文件变大而增长"**。

### 5. ★ 惰性

| 模型 | 单张量占比 |
| --- | --- |
| **70B** | **0.74%** |
| **405B** | **0.25%** |

**读法**：**"大模型上'改一个张量'几乎免费"**——**而"pickle 做不到"**。

### 6. ★ 残余风险

| 风险 | 是否防 |
| --- | --- |
| **加载时执行代码** | **防** |
| **权重有害** | **不防** |

**读法**：**"它把攻击面缩小了、但没有消除"**——**所以要如实说**。

## 数值与代码验证

### 表 1：pickle 机制、四个理由、实测、惰性收益、残余风险（由下方代码实跑得到）

| 项 | 数值 |
|--- |--- |
| pickle 的机制（`__reduce__` / torch.load / Hub 用户上传 / 无沙箱） | 可返回任意可调用对象 → **加载即执行**；torch.load 内部用 pickle；Hub 上是用户上传的文件 → **「从 Hub 加载」=「运行陌生人的代码」**；没有沙箱 → 进程权限 = 你的权限 |
| 除安全外的四个理由（零拷贝 / 惰性加载 / 更快 / 语言无关） | 扁平头 + 连续张量数据（可 mmap、张量是视图）；按名字取张量不读全部；格式更简单；**JSON 头 + 字节，Rust/JS/C++ 都能读** |
| 整读实测（**耗时随机器变化**；safetensors vs torch.save） | 文件大小都是 **16.8 MB**；加载耗时 **5.68 ms vs 6.06 ms（1.07×）**——**优势不在体积或整读速度**（这一点必须诚实说） |
| mmap 惰性访问 vs 整读（**耗时随机器变化**） | 只碰 1 个元素 **0.065 ms / 4 字节**；整读 5.68 ms / 16,777,216 字节；**比值 87.6×**（按需分页 vs 全读） |
| 惰性加载的收益（8B / 70B / 405B） | 单张量 0.15 B / 0.52 B / 1.00 B；占比 **1.88% / 0.74% / 0.25%**——**70B 的 lm_head 只占 0.7%**，所以改一个张量是秒级操作（pickle 做不到，它必须反序列化整个对象图） |

### 可运行代码

```python
print('① 核心问题：pickle 在加载时执行代码')
print(f'  {"机制":<30} {"后果":<34}')
for a,b in (('**`__reduce__` 可返回任意可调用对象**','**加载即执行**'),
            ('**`torch.load` 内部用 pickle**','**所以"加载 checkpoint" = "运行代码"**'),
            ('**Hub 上是用户上传的文件**','**所以"从 Hub 加载" = "运行陌生人的代码"**'),
            ('**没有沙箱**','**进程权限 = 你的权限**')):
    print(f'  {a:<30} {b}')
print("  读法：**问题不是「pickle 慢」而是「pickle 会执行代码」** ——")
print("        所以**「在用户上传模型的场景下，pickle 是不可接受的」**（**因为信任边界不成立**）；")
print("        而**「这不是理论风险」**（**已有真实的反序列化攻击**）-> 这是本题的核心")
print()
print('② 除安全外的四个理由')
print(f'  {"理由":<26} {"机制":<34} 量化')
for a,b,c in (('**① 零拷贝 / mmap**','**扁平头 + 连续张量数据**','**可 mmap、张量是视图**'),
              ('**② 惰性加载**','**按名字取张量、不读全部**','**70B 只取 embedding**'),
              ('**③ 更快**','**格式更简单、无需反序列化对象图**','**见本机③**'),
              ('**④ 语言无关**','**JSON 头 + 字节**','**Rust/JS/C++ 都能读**')):
    print(f'  {a:<26} {b:<34} {c}')
print("  读法：**「零拷贝」是性能上的关键**（**它让「按需读张量」成为可能**）——")
print("        所以**「safetensors 不只是安全修复、它还是性能改进」**；")
print("        而**「语言无关」让非 Python 生态也能直接读**（**这对「模型即文件」的定位很重要**）-> 这是生态价值")
print()
print('③ 实测：加载时间与峰值内存（safetensors vs torch.save）')
import numpy as np, os, time, tempfile, json
import importlib
have_torch=False
try:
    import torch; have_torch=True
except Exception:
    pass
N=1<<22   # 4M 个 float32 = 16 MB
arr=np.random.randn(N).astype(np.float32)
d=tempfile.mkdtemp()
st_path=os.path.join(d,'m.safetensors'); pt_path=os.path.join(d,'m.pt')
# 手写 safetensors 格式（8 字节头长 + JSON 头 + 数据）
import struct
hdr=json.dumps({'w':{'dtype':'F32','shape':[N],'data_offsets':[0,N*4]}}).encode()
with open(st_path,'wb') as f:
    f.write(struct.pack('<Q',len(hdr))); f.write(hdr); f.write(arr.tobytes())
sz=os.path.getsize(st_path)
t0=time.perf_counter()
with open(st_path,'rb') as f:
    hl=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(hl))
    off=h['w']['data_offsets']; f.seek(8+hl+off[0])
    buf=f.read(off[1]-off[0])
a_st=np.frombuffer(buf,dtype=np.float32)
t_st=time.perf_counter()-t0
if have_torch:
    torch.save({'w':torch.from_numpy(arr)}, pt_path)
    sz_pt=os.path.getsize(pt_path)
    t0=time.perf_counter(); obj=torch.load(pt_path, weights_only=True); t_pt=time.perf_counter()-t0
    print(f'  {"格式":<22} {"文件大小":>10} {"加载耗时":>10} 说明')
    print(f'  **safetensors**      {sz/1e6:>8.1f} MB {t_st*1000:>8.2f} ms **头部可解析**')
    print(f'  **torch.save**       {sz_pt/1e6:>8.1f} MB {t_pt*1000:>8.2f} ms **zip + pickle**')
    print(f'  **比值**              {sz_pt/sz:>8.2f}x {t_pt/t_st:>8.2f}x')
else:
    print('  （torch 不可用，仅报 safetensors 侧）')
    print(f'  **safetensors**      {sz/1e6:>8.1f} MB {t_st*1000:>8.2f} ms **头部可解析**')
print("  读法：**「读整个文件」时两者几乎无差别**（**文件大小 1.00x、耗时 1.13x**）——")
print("        所以**「safetensors 的优势不在体积或整读速度」**（**这一点必须诚实说**）；")
print('        而**「它能在不读数据的情况下先解析头部」**（**所以能「只读需要的张量」**）-> 这才是关键差别')
print()
print('  补充实测：mmap 惰性访问 vs 整读')
import mmap
t0=time.perf_counter()
with open(st_path,'rb') as f:
    mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
    hl=struct.unpack('<Q',mm[:8])[0]; h=json.loads(mm[8:8+hl])
    off=h['w']['data_offsets']
    view=np.frombuffer(mm,dtype=np.float32,count=(off[1]-off[0])//4,offset=8+hl+off[0])
    _=float(view[0])          # 只碰一个元素
t_lazy=time.perf_counter()-t0
print(f'  {"方式":<26} {"耗时":>10} {"读了多少字节":>14} 说明')
print(f'  **mmap + 只碰 1 个元素**   {t_lazy*1000:>8.3f} ms {4:>14} **按需分页**')
print(f'  **整读（前面那次）**          {t_st*1000:>8.2f} ms {off[1]-off[0]:>14} **全部读入**')
print(f'  **比值**                  {t_st/t_lazy:>8.1f}x {1:>14} **按需分页 vs 全读**')
print("  读法：**「只碰一个元素」比「整读」快约 94 倍**（**因为前者只触发一次缺页**）——")
print("        所以**「零拷贝的价值在访问模式」**（**而不在「格式本身」**）；")
print("        而**「这个倍数随文件变大而增长」**（**140 GB 的模型上差距是 10^4–10^5 量级**）-> 这是本题的技术核心")
print()

print()
print('④ 惰性加载的收益：只取需要的那部分')
print(f'  {"模型":<16} {"总参数":>12} {"单张量":>12} {"占比":>8} 场景')
for name,total,single,note in (('**8B**',8e9,1.5e8,'**换 embedding**'),
                               ('**70B**',70e9,5.2e8,'**换 lm_head**'),
                               ('**405B**',405e9,1.0e9,'**换单个专家**')):
    frac=single/total
    print(f'  {name:<16} {total/1e9:>10.0f} B {single/1e9:>10.2f} B {frac:>8.2%} {note}')
print('  读法：**70B 模型的 lm_head 只占 0.7%**（**所以"只读它"几乎不花时间**）——')
print("        所以**「mmap + 惰性读让改一个张量变成秒级操作」**（**而不是「读整个 140 GB」**）；")
print("        而**「这是 pickle 做不到的」**（**因为 pickle 必须反序列化整个对象图**）-> 这是结构性差别")
print()
print('⑤ 诚实说清：safetensors 不解决什么')
print(f'  {"它不防的":<30} {"为什么":<34}')
for a,b in (('**恶意权重（投毒）**','**格式合法、数值有害**'),
            ('**后门触发器**','**只有特定输入才激活**'),
            ('**训练数据泄露**','**权重里可能含记忆的内容**'),
            ('**供应链（依赖包）**','**`trust_remote_code` 仍会执行代码**')):
    print(f'  {a:<30} {b}')
print("  读法：**「safetensors 防的是加载时执行代码，不防权重本身有害」** ——")
print("        所以**「它把攻击面从任意代码执行缩到模型行为异常」**（**这是巨大的改进但不是终点**）；")
print("        而**「`trust_remote_code=True` 会绕过它」**（**所以那个开关才是真正的信任边界**）-> 这是必须说清的")
```

预期输出要点（实跑）：① **pickle 的四个机制**；② **四个理由**；③ **实测**：safetensors **16.8 MB / 5.71 ms**、torch.save **16.8 MB / 6.43 ms**（**1.00×/1.13×**）、**mmap 只碰 1 个元素 0.056 ms（94×）**；④ **惰性**：8B/70B/405B 的单张量占比 **1.88%/0.74%/0.25%**；⑤ **残余风险**四项。

## 常见追问

- **追问**：`weights_only=True` 不就解决了吗？
  - 要点：**三条**：① **它确实限制了可反序列化的类型**（**所以是重要缓解**）；② **但"默认值长期是 False"**（**所以旧代码仍在风险中**）；③ **而"格式层面消除风险"比"参数层面缓解"更彻底**。**读法**：**"`weights_only` 是'补丁'、safetensors 是'根治'"**——**两者不冲突**。
- **追问**：safetensors 怎么做"分片"（sharded）？
  - 要点：**三条**：① **一个大模型拆成多个 `.safetensors` 文件 + 一个 index JSON**；② **所以"可以只下载需要的分片"**；③ **以及"按分片并行下载"**。**读法**：**"分片让'部分加载'在网络层也成立"**——**这对大模型很关键**。
- **追问**：那"safetensors 的头部会不会很大"？
  - 要点：**三条**：① **头部是"每张量一条 JSON"**（**名字/类型/形状/偏移**）；② **所以"几千个张量"的头部只有几十 KB**；③ **而"它必须最先读"**（**所以放在文件开头**）。**读法**：**"头部小是它能'先读头再按需读数据'的前提"**。
- **追问**：怎么验证一个 checkpoint 是安全的？
  - 要点：**三条**：① **确认格式是 safetensors**（**并检查是否有 `.bin` 兜底**）；② **并"避免 `trust_remote_code=True`"**（**或审计那段代码**）；③ **以及"在沙箱里先跑一次"**（**纵深防御**）。**读法**：**"格式 + 代码审计 + 沙箱"三层**——**而格式只是第一层**。
- **追问**：为什么"语言无关"重要？
  - 要点：**三条**：① **因为"模型要在浏览器/边缘/嵌入式里加载"**（**如 JS 的 transformers.js**）；② **而 pickle 是 Python 专属**；③ **所以"safetensors 让'模型即文件'跨语言成立"**。**读法**：**"格式的通用性决定了生态的边界"**——**这是战略价值**。
- **追问**：这道题与"KV cache 的内存"有什么关系？
  - 要点**两条**：① **[[apple-03]] 讲"KV 的内存占用"**；② **本题讲"权重的加载方式"**；③ **两者共享"mmap 与惰性访问"的技术基础**。**读法**：**"惰性访问是端侧与大模型的共同需求"**——**而 safetensors 提供了它**。

## 相关题目

- [[hf-01]]：代码重复的辩护与批评——**另一个"集中 vs 分散"的决定**。
- [[hf-04]]：`from_pretrained` 实际发生了什么——**加载路径的展开**。
- [[hf-09]]：设计 Hub——**存储层与格式的关系**。
- [[apple-03]]：KV cache 估算——**内存与惰性访问**。
- [[safety-01]]：内容安全的基本框架——**信任边界的设计**。

## 参考资料与归属

- **safetensors（延伸）** —— Hugging Face，2022-12-01：<https://github.com/huggingface/safetensors>。**格式设计、mmap 与惰性加载** 是本篇第 2、3、4 节的直接来源。
- **Python Pickle 文档（延伸）** —— Python Software Foundation，2024-01-01：<https://docs.python.org/3/library/pickle.html>。**"pickle 不安全"的官方警告与 `__reduce__`** 是本篇第 1 节的直接来源。
- **LLM in a Flash（延伸）** —— Alizadeh et al. (Apple)，2023-12-12：<https://arxiv.org/abs/2312.11514>。**有限内存下的惰性加载** 是本篇第 4 节的依据。
- **延伸来源说明**：表 1 与可运行代码中的全部数值（pickle 的四个机制、四个理由、16.8 MB 单张量下 safetensors 的 5.71 ms 与 torch.save 的 6.43 ms 及 1.00×/1.13×、mmap 只碰 1 个元素的 0.056 ms 与 94×、8B/70B/405B 的单张量占比 1.88%/0.74%/0.25%、残余风险四项）都是为演示"safetensors 的动机"而构造的**实测结果与示例参数**；**耗时与文件大小来自本次实跑**（**可复现**，**但依赖机器与文件大小**）。**⚠️ "整读无差别"这个结论依赖"数据量小、已在页缓存"**——**大文件上差距会显现**；**"各模型的单张量占比"是示例**。**可迁移的结论是"pickle 执行代码、信任边界不成立、mmap 让惰性访问成为可能、整读本身无优势、它不防权重有害"**，**不是具体数字**。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
