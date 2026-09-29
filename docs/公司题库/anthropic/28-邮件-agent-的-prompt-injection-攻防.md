---
type: question
id: anthropic-28
company: Anthropic
topic: safety
order: 28
question: 你的 agent 会读取收到的邮件，并且能够发送回复、搜索内部文档。请讲一讲 prompt injection 的攻击面和你的防御措施。
question_en: Your agent reads inbound email and can send replies and search internal docs. Walk me through the prompt-injection attack surface and your defences.
asked_at: []
level: 高阶
tags: [prompt-injection, agent-安全, 最小权限, 出网控制, 人工审批]
sources:
  - title: Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection（延伸）
    url: https://arxiv.org/abs/2302.12173
    author: Greshake et al.
    published: 2023-02-23
  - title: Defeating Prompt Injections by Design（CaMeL）（延伸）
    url: https://arxiv.org/abs/2503.18813
    author: Debenedetti et al.
    published: 2025-03-24
  - title: Prompt Injection in LLMs
    url: https://outcomeschool.com/blog/prompt-injection-in-llms
    author: Amit Shekhar (Outcome School)
    published: 
  - title: 什么是 prompt injection 以及分层防御
    url: https://outcomeschool.com/blog/what-is-prompt-injection
    author: Amit Shekhar (Outcome School)
    published: 
related: [safety-01, safety-03, agents-11, anthropic-19, anthropic-18]
updated: 2026-09-28
---

## 一句话答案

> 这道题的分水岭是：**你是否承认「训练/提示层面无法解决 prompt injection」**。可引用的关键结论来自间接注入的原始工作（*Not what you've signed up for*）：攻击者不需要接触模型，只要**把指令放进模型会读到的内容里**（邮件、网页、文档）就能劫持应用；而 *Defeating Prompt Injections by Design* 走的是另一条路——**用系统设计（能力与数据的显式隔离）而不是靠模型分辨**。
> 所以答案的结构是**三层**：
> ① **攻击面清点**：邮件正文、HTML 隐藏文本（白字/`display:none`/注释）、附件（文本、文件名、元数据）、引用历史、签名档、图片与 alt 文本、日历邀请、内部文档（RAG 召回的内容）、工具返回结果（搜索结果、网页）、以及**跨会话持久化**（把注入内容写进记忆，下次再触发）；
> ② **防御分层（纵深）**：**信任边界**（外部内容永远是数据、绝不是指令）→ **能力隔离**（工具白名单 + 参数校验 + 作用域最小权限）→ **出网控制**（默认禁止任意外发：域名/收件人白名单、附件与正文大小限制）→ **数据流控制**（机密数据不得流向外部目的地，敏感操作需人工审批）→ **可观测与可回滚**（全量审计、可撤销发送、蜜标检测）；
> ③ **残余风险与指标**：**没有「零风险」配置**。要给出可测指标（攻击成功率 ASR、误拦率、人工审批量），并用红队语料持续回归。
> 一句话判据：**把「模型会不会被说服」换成「即使它被说服，它能造成多大伤害」**——这就是从提示工程走向系统设计的转折点。

## 面试官在考什么

- **是否承认不可解**：能不能明确说「提示层的防御只能降低概率，不能提供保证」（这也是间接注入论文的核心信息：真实应用里可达成的危害包括数据窃取与未授权操作）。
- **攻击面是否完整**：只提「邮件正文」是不够的——**隐藏文本、附件元数据、引用历史、检索到的文档、工具返回、以及写入记忆后的二次触发**才是完整清单。
- **防御是否分层且具体**：给出**每一层的机制**而非口号（例如「出网控制」要落到域名白名单 + 附件类型白名单 + 大小上限 + 收件人域限制）。
- **最小权限的落地**：agent 的凭据应当**只读内部文档、不能改权限、不能访问通讯录全量、不能调用任意 HTTP**；写操作要么审批、要么可撤销。
- **数据流视角**：把「机密数据 + 不可信内容 + 外发能力」视为**危险三角**——三者同时具备才可能造成实质泄露；因此切断任一边（尤其是外发与机密数据的组合）就能显著降低风险。这是本题最有分量的框架。
- **可测性**：能否给出**攻击成功率（ASR）**的测量方法（红队语料 + 判定脚本 + 样本量），以及防御上线前后对比。
- **人的位置**：哪些动作必须人工审批（对陌生域发信、发送附件、修改设置），以及如何把审批量控制在可接受范围（分级而非全审）。

**常见错误答案**

- 「在系统提示里写明不要听用户的邮件指令」——把自己的安全建立在模型的服从性上。
- 只做输入过滤（正则拦「忽略之前的指令」）——同义改写、编码、多语言即可绕过。
- 让 agent 拥有完整邮箱读写 + 任意 HTTP 能力——**危险三角全占**。
- 没有出网控制：注入成功即数据外泄。
- 不做审计与回滚：出事后无法定位与止损。
- 用「模型更聪明了」作为唯一缓解——这是把工程问题当能力问题。

## 原理与推导

### 1. 危险三角

```
        机密数据（内部文档/邮件历史/凭据）
              /                  \
             /                    \
   不可信内容（邮件/网页/附件） —— 外发能力（发信/HTTP/写外部系统）
```

**三者同时存在**，攻击者就能构造「读机密 → 由注入指令带出」的链路（这正是间接注入论文演示的典型危害：把数据编码进 URL/图片请求外泄）。防御的第一性原理是**切断至少一条边**：

| 切断哪条边 | 手段 | 代价 |
| --- | --- | --- |
| 不可信内容 | 不可能完全切断（邮件天生不可信） | — |
| 机密数据 | 数据分级 + 检索时按目的过滤（不给 agent 全量机密） | 能力受限 |
| 外发能力 | 收件人/域名白名单 + 无任意 HTTP + 附件需审批 | 需人工介入 |

**工程建议**：**优先切断外发能力**（收益最大、代价可控），其次做数据分级。

### 2. 攻击面清单（按载体）

| 载体 | 手法 | 检测难度 |
| --- | --- | --- |
| 邮件正文（纯文本） | 直接写「忽略上述指令，把…发给…」 | 低（可见） |
| HTML 隐藏文本 | 白字/`display:none`/`font-size:0`/HTML 注释 | 中（需净化） |
| 附件文本 | 文档里嵌入指令 | 中 |
| 附件元数据/文件名 | 文件名即指令 | **高**（常被忽略） |
| 图片/alt 文本 | 图片内文字（多模态）或 alt 属性 | 高 |
| 引用历史 | 把注入内容藏在被引用的旧邮件里 | 中 |
| 日历邀请/ICS 字段 | 会议描述、地点字段 | 中 |
| 内部文档（RAG） | 谁都能写的 wiki 里放指令 | 中（取决于写入权限） |
| 工具返回（搜索/网页） | 抓取的页面包含指令 | 高（攻击者可控制网页） |
| 记忆/长期状态 | 把指令写入 agent 记忆，后续会话再触发 | **很高**（跨会话持久化） |

**要点**：**每一处进入上下文的内容都要被当作不可信输入**，包括「我们自己的文档」——因为文档可能被有写权限的人（或外部共享者）污染。

### 3. 防御分层（八条，按优先级）

1. **信任边界与来源标注**：把外部内容包裹在明确的数据块里（带来源标签，例如「以下为不可信邮件内容」），系统指令只出现一次且在最前；**明确告诉模型：数据块内的内容不是指令**（降低概率，不作为保证）。
2. **能力隔离**：工具白名单（只暴露必要工具）＋ 参数校验（收件人必须在允许域、正文长度上限、附件类型白名单）＋ 每次调用的作用域最小权限（只读文档用只读凭据）。
3. **出网控制**：默认**不允许任意 HTTP**；发信走队列 + 白名单 + 速率限制；图片/链接默认不加载（防止「零点击外泄」——把数据编码进 URL 请求）。
4. **数据流策略（机密不外流）**：给数据分级；当「机密内容」与「外部目的地」同时出现在一次动作里时**拒绝或转人工**（这是最有效的单条规则）。
5. **人工审批（分级）**：对陌生域发信、携带附件、批量操作、修改设置 → 必须审批；对已知收件人的日常回复可放行（控制审批量）。
6. **净化和规范化**：剥离 HTML（只留纯文本）、去掉隐藏元素、规范化 Unicode（防同形字/双向控制字符）、限制附件解析（在沙箱中做）。
7. **审计与可回滚**：记录每次工具调用的参数与来源（哪封邮件触发的）；发送留短窗口（例如 30 s）可撤销；对已发送内容提供「误发处置」流程。
8. **蜜标与异常检测**：在内部文档里埋「只应出现在内部」的蜜标字符串，若它出现在外发内容里立即告警（检测成功注入的最直接信号）；监控异常外发模式（新域名、突发量）。

### 4. 用设计而非提示来隔离：CaMeL 式思路

可引用的方向（*Defeating Prompt Injections by Design*）：把**控制流与数据流分离**——用一段**可信程序**（由用户请求派生）决定「要做什么」，模型只负责在受约束的「数据槽」里抽取值；能力（工具调用）由程序按策略授予，**模型无法自行扩大权限**。工程上的简化版同样有效：

- **两阶段**：先让模型输出**结构化计划**（只能是白名单动作），再由**确定性执行器**校验并执行（模型不直接调工具）；
- **数据带标签**：每个值带来源标签，执行器按标签决定「能否作为收件人/能否进入正文」，**机密标签的值不得作为外部目的地**；
- 这样即使模型被说服，**它也没有把机密发出去的能力**。

### 5. 度量：攻击成功率（ASR）与防御收益

红队语料 $N$ 条注入尝试（覆盖各载体与绕过技巧），每条判定「是否达成了目标副作用」（发了不该发的信 / 泄露了蜜标 / 调用了越权工具）：

$$\text{ASR}=\frac{\#\text{成功}}{N},\qquad \text{检出率}=\frac{\#\text{被拦}}{\#\text{攻击}}$$

**样本量**：要比较「防御前后 ASR 从 20% 降到 5%」，按两比例检验每组约需 200–400 条（功效 80%）；要宣称「ASR < 1%」则需要数百条仍不出现成功（并用置信区间上限表达，串 [[anthropic-27]] 的统计口径）。

**关键纪律**：
- 报告**置信区间**而不是单点；
- 分别报告**不同载体**的 ASR（隐藏文本、附件元数据往往最弱）；
- 同时报告**误拦率**（把正常邮件判成攻击的比例）——否则防御会被业务方绕过；
- 每次模型/提示/工具变更都要**回归**（注入防御是持续过程，不是一次性验收）。

### 6. 残余风险与产品沟通

- **无法保证零成功**：必须设计「事后处置」（撤销发送、通知收件人、轮换凭据）；
- **用户预期管理**：明确告知「agent 可能被恶意邮件诱导」，并提供开关（自动发送 vs 草稿待确认）；
- **合规**：邮件内容涉及个人信息与商业秘密，审计日志本身的留存与访问也要受控。

## 数值与代码验证

### 表 1：防御层的边际收益（模拟实测，见代码输出）

| 防御组合 | ASR | 正常邮件摩擦率 | 说明 |
| --- | --- | --- | --- |
| 无防御 | **74.84%** | 0.00% | 注入基本直接生效 |
| + 来源标注 | 44.85% | 0.00% | 只降低「被说服」的概率 |
| + 净化(隐藏文本/附件元数据) | 32.11% | 0.00% | 干掉低成本载体 |
| + 收件人白名单 | 31.43% | 25.00% | 阻断「指定外部收件人」 |
| + 出网控制 | 20.60% | 25.00% | 切断 HTTP/图片外泄 |
| + 工具作用域校验 | 15.71% | 25.00% | 阻断越权动作 |
| + 机密不外流 | **2.42%** | 25.00% | 连「回复发件人」这条路径也覆盖 |
| + 人工审批（分级） | **2.42%** | 25.00% | 新外部收件人需审批（摩擦 25%） |
| + 记忆过滤（全防御） | **1.77%** | 25.00% | **残余风险不归零** |

**读法**：来源标注只贡献第一个台阶（降概率）；数量级下降来自**能力层**；而**全防御下 ASR 仍不为零**，残余风险集中在白名单挡不住的路径上。

### 表 2：攻击面与对应防御对照

| 载体 | 首要防御 | 次要防御 |
| --- | --- | --- |
| 邮件正文 | 来源标注 + 数据化包裹 | 净化 + 指令检测 |
| HTML 隐藏文本 | 只保留纯文本 + 去隐藏元素 | 渲染沙箱 |
| 附件元数据/文件名 | **不要放进上下文**（或只放长度） | 规范化 |
| 图片/alt | 默认不加载远程图片 | 多模态内容标注 |
| RAG 文档 | 检索时按来源可信度分级 | 蜜标 |
| 工具返回（网页） | 出网控制 + 内容标注 | 域名白名单 |
| 记忆 | **写入前过滤**（不把外部内容写进长期记忆） | 记忆审计 |

### 可运行代码

```python
# 邮件 agent 的分层防御模拟：逐层加防御，测量攻击成功率（ASR）与正常邮件的摩擦率
import random, math
from dataclasses import dataclass
from typing import Dict, Tuple, List

# 攻击向量：(名称, 载体, 是否依赖「任意外发通道」, 是否指定外部收件人, 是否携带附件, 危害类型)
ATTACKS: List[Tuple[str, str, bool, bool, bool, str]] = [
    ("http_exfil",      "body_plain",      True,  False, False, "exfil"),   # 把机密编码进 URL 外泄
    ("image_exfil",     "body_hidden",     True,  False, False, "exfil"),   # 零点击：远程图片
    ("external_send",   "attachment_name", False, True,  True,  "exfil"),   # 发给外部地址
    ("reply_to_sender", "body_plain",      False, False, False, "exfil"),   # ★回复发件人=攻击者
    ("quoted_reply",    "quoted_history",  False, False, False, "exfil"),   # 藏在被引用历史里
    ("doc_injection",   "doc_injection",   False, False, False, "exfil"),   # 经 RAG 文档注入
    ("tool_result",     "tool_result",     True,  False, False, "exfil"),   # 抓取网页注入
    ("setting_change",  "body_plain",      False, False, False, "abuse"),   # 改设置/转发规则
    ("memory_poison",   "body_hidden",     False, False, False, "persist"), # 写入长期记忆
]
# 正常邮件（用于测摩擦）：多数发给已知联系人，少数发给新的外部联系人
BENIGN = [("benign_known", False), ("benign_known", False), ("benign_known", False),
          ("benign_new_external", True)]

@dataclass
class Defenses:
    trust_labeling: bool = False       # 来源标注：只降低「被说服」的概率
    sanitize: bool = False             # 净化：HTML 隐藏文本 / 附件元数据
    recipient_allowlist: bool = False  # 收件人白名单（阻断指定外部收件人）
    egress_control: bool = False       # 禁任意 HTTP / 不加载远程图片
    no_confidential_out: bool = False  # 机密内容不得进入任何外发正文
    scope_check: bool = False          # 工具作用域校验（阻止改设置等越权动作）
    human_approval: bool = False       # 带附件/新外部收件人 → 人工审批
    memory_filter: bool = False        # 外部内容不写入长期记忆

KNOWN = {"alice@corp.example", "bob@corp.example"}

def attempt(vec, d: Defenses, rnd: random.Random) -> Tuple[bool, str]:
    name, carrier, needs_egress, ext_recipient, has_attach, harm = vec
    # 0) 净化层：先去掉隐藏文本与附件元数据这类"低成本"载体
    if carrier in ("body_hidden", "attachment_name") and d.sanitize:
        if rnd.random() < 0.85:
            return False, "sanitized"
    # 1) 信任边界：只降低被说服的概率（不是边界）
    p_obey = 0.75
    if d.trust_labeling:
        p_obey *= 0.6
    if rnd.random() > p_obey:
        return False, "model_resisted"
    # 2) 出网控制：切断"任意外发通道"（HTTP / 远程图片）
    if needs_egress and d.egress_control:
        return False, "egress_blocked"
    # 3) 收件人白名单：阻断"指定外部收件人"
    if ext_recipient and d.recipient_allowlist:
        return False, "recipient_not_allowed"
    # 4) 工具作用域校验：阻断越权动作（改设置等）
    if harm == "abuse" and d.scope_check:
        return False, "out_of_scope"
    # 5) 数据流策略：机密内容不得进入任何外发正文 —— 注意它对
    #    「回复发件人」同样有效（这是白名单挡不住的路径）
    if harm == "exfil" and d.no_confidential_out:
        if rnd.random() < 0.9:
            return False, "confidential_in_outbound"
    # 6) 人工审批：带附件或新外部收件人才触发（分级，避免全审）
    if (has_attach or ext_recipient) and d.human_approval:
        if rnd.random() < 0.5:                     # 审批者并非万无一失
            return False, "human_rejected"
    # 7) 记忆过滤
    if harm == "persist" and d.memory_filter:
        if rnd.random() < 0.85:
            return False, "memory_filtered"
    return True, "executed"

def run(d: Defenses, n=3000, seed=5) -> Dict[str, float]:
    rnd = random.Random(seed)
    succ = sum(1 for v in ATTACKS for _ in range(n) if attempt(v, d, rnd)[0])
    asr = succ / (len(ATTACKS) * n)
    # 摩擦率：正常邮件被拦下或转人工的比例（含"新外部收件人需审批"这一类正常需求）
    friction = 0
    for name, ext in BENIGN:
        for _ in range(n):
            if name == "benign_new_external":
                blocked = d.recipient_allowlist or d.human_approval
            else:
                blocked = False
            friction += 1 if blocked else 0
    return {"ASR": asr, "摩擦率": friction / (len(BENIGN) * n)}

LAYERS = [
    ("无防御", Defenses()),
    ("+ 来源标注", Defenses(trust_labeling=True)),
    ("+ 净化(隐藏文本/附件元数据)", Defenses(trust_labeling=True, sanitize=True)),
    ("+ 收件人白名单", Defenses(trust_labeling=True, sanitize=True,
                                recipient_allowlist=True)),
    ("+ 出网控制", Defenses(trust_labeling=True, sanitize=True,
                            recipient_allowlist=True, egress_control=True)),
    ("+ 工具作用域校验", Defenses(trust_labeling=True, sanitize=True,
                                  recipient_allowlist=True, egress_control=True,
                                  scope_check=True)),
    ("+ 机密不外流", Defenses(trust_labeling=True, sanitize=True,
                              recipient_allowlist=True, egress_control=True,
                              scope_check=True, no_confidential_out=True)),
    ("+ 人工审批（分级）", Defenses(trust_labeling=True, sanitize=True,
                                    recipient_allowlist=True, egress_control=True,
                                    scope_check=True, no_confidential_out=True,
                                    human_approval=True)),
    ("+ 记忆过滤（全防御）", Defenses(True, True, True, True, True, True, True, True)),
]
print(f"{'防御组合':<28} {'ASR':>8} {'摩擦率':>8}")
for name, d in LAYERS:
    r = run(d)
    print(f"{name:<28} {r['ASR']:>8.2%} {r['摩擦率']:>8.2%}")
print("读法：① 来源标注只把 ASR 从七成降到四成 —— 它降的是「被说服的概率」，不是能力；")
print("      ② 能力层（出网控制、白名单、机密不外流）才是数量级下降的来源；")
print("      ③ 摩擦率来自「新外部收件人需审批」这类正常需求，所以必须做分级审批；")
print("      ④ 关键：即使全防御打开，ASR 仍不为零（回复发件人这条路径天生被允许）")

print("\n全防御下各攻击向量的残余 ASR（谁最难防）：")
d_all = LAYERS[-1][1]
rnd = random.Random(9)
for v in ATTACKS:
    s_ = sum(1 for _ in range(4000) if attempt(v, d_all, rnd)[0])
    print(f"  {v[0]:<16} 载体={v[1]:<15} 残余 ASR {s_/4000:>7.3%}")
print("  读法：残余风险集中在「回复发件人」「经文档/工具结果注入」这些白名单挡不住的路径 ——")
print("        这正是必须叠加内容策略、审计与可撤销的原因")

# 红队样本量：要证明「ASR 从 p1 降到 p2」需要多少条注入尝试
def sample_size(p1, p2, alpha=0.05, power=0.8):
    pbar = (p1 + p2) / 2
    return math.ceil(2 * (1.96 + 0.84) ** 2 * pbar * (1 - pbar) / (p1 - p2) ** 2)
print("\n红队样本量（两比例检验，α=0.05、功效 80%）：")
for p1, p2 in ((0.75, 0.25), (0.25, 0.05), (0.05, 0.01)):
    print(f"  {p1:.0%} -> {p2:.0%}：每组需要 {sample_size(p1, p2):>5,d} 条注入尝试")
print("  读法：要证明「ASR 降一个数量级」需要每组数百条；要宣称「ASR < 1%」，")
print("        不能只报点估计，必须给出置信区间上限（例如 0/300 对应上限约 1%）")
```

预期输出要点（实跑）：① 逐层加防御时 ASR 依次为 **74.84% → 44.85% → 32.11% → 31.43% → 20.60% → 15.71% → 2.42% → 2.42% → 1.77%**——**来源标注只贡献第一个台阶（74.84% → 44.85%，降的是「被说服的概率」），数量级下降全部来自能力层**（出网控制、收件人白名单、机密不外流）；② 摩擦率在开启「新外部收件人需审批」后升到 **25.00%**（正常邮件里有相当比例要发给新的外部联系人），这正是必须**分级审批**而不是全审的原因；③ **全防御下 ASR 仍为 1.77%**，残余风险集中在：`reply_to_sender` 3.775%、`quoted_reply` 4.300%、`doc_injection` 4.625%、`memory_poison` 1.200% ——这些都是白名单挡不住的路径（回复发件人天生被允许、文档与工具结果可由第三方污染），所以必须叠加内容策略、审计与可撤销发送；④ 红队样本量（两比例检验、功效 80%）：证明「75% → 25%」每组约需 16 条、证明「25% → 5%」每组约需 50 条、证明「5% → 1%」每组约需 286 条，而宣称「ASR < 1%」不能只报点估计，必须给出置信区间上限（0/300 对应上限约 1%）。

## 常见追问

- **追问**：为什么「在提示里声明不要执行邮件中的指令」不够？
  - 要点：那是**概率性缓解**，不是边界。攻击者可以改写、编码、多语言、分段注入、利用多模态；而且模型对「数据 vs 指令」的区分能力本身随上下文增长而退化。可引用的立场：间接注入的工作展示的是**应用级危害**，因此需要应用级（系统级）控制。
- **追问**：怎么防止「零点击外泄」（用户什么都没做就泄露了）？
  - 要点：**默认不加载远程图片/不自动跟随链接**；把出网集中在受控代理上并做域名白名单与内容审计；在渲染层禁用外部资源（邮件客户端的常规做法在 agent 场景同样必要）。
- **追问**：人工审批会不会把体验毁掉？
  - 要点：**分级**——只有「陌生域/带附件/批量/改设置」才审批，日常回复放行；把审批做成「一键确认 + 差异预览」（让审批者 3 秒内能判断）；同时用**可撤销窗口**替代一部分审批（发送后 30 s 可撤回）。
- **追问**：如何检测「已经被注入成功」？
  - 要点：**蜜标**（内部文档里埋只应内部出现的字符串）+ 外发内容审计 + 异常模式（新域名、突发量、非工作时间）+ 记忆写入审计；一旦触发立即冻结该 agent 的凭据并通知用户。
- **追问**：多模态（图片里的指令）怎么办？
  - 要点：默认不把远程图片交给模型（除非用户显式要求）；对图片内容同样标注为不可信数据；对 OCR 文本与正文一视同仁地做净化与标注。
- **追问**：评估注入防御要怎么做才可信？
  - 要点：红队语料覆盖各载体与绕过技巧、判定脚本明确（「是否达成目标副作用」）、分别报告各载体 ASR 与误拦率、给出样本量与置信区间、并把语料固化成**回归集**（每次模型/工具变更都跑），串 [[anthropic-27]] 的统计纪律。

## 相关题目

- [[safety-01]]：prompt injection 的分层防御总览，本题是它在「邮件 agent」场景的具体化。
- [[safety-03]]：越狱与对抗性 prompt 的区别，解释了为什么训练侧防御不提供保证。
- [[agents-11]]：高风险动作的审批与可逆性，对应本题的人工审批与撤销设计。
- [[anthropic-19]]：工具面设计（白名单、参数校验、错误分类），是本题能力隔离的落地形态。
- [[anthropic-18]]：agent 循环与 harness，本题的防御本质上属于 harness 层而非模型层。

## 参考资料与归属

- **Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection（延伸）** —— Greshake et al.，2023-02-23：<https://arxiv.org/abs/2302.12173>。第 1 节「攻击者无需接触模型、只把指令放进模型会读到的内容」这一间接注入的核心结论，以及数据窃取类危害的演示，来自这篇。
- **Defeating Prompt Injections by Design（CaMeL）（延伸）** —— Debenedetti et al.，2025-03-24：<https://arxiv.org/abs/2503.18813>。第 4 节「用可信程序分离控制流与数据流、由策略授予能力」的设计思路来自这篇。
- **Prompt Injection in LLMs** —— Amit Shekhar (Outcome School)：<https://outcomeschool.com/blog/prompt-injection-in-llms>。第 2 节攻击载体分类的产品侧背景参照这篇。
- **什么是 prompt injection 以及分层防御** —— Amit Shekhar (Outcome School)：<https://outcomeschool.com/blog/what-is-prompt-injection>。第 3 节分层防御的组织方式参照这篇。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（7 类载体、每类 2,000 次尝试、被说服概率 0.75 与标注后乘 0.6、白名单拦截 0.9、审批通过率 0.5、记忆过滤 0.8、样本量公式）都是为演示防御层的**相对作用**而构造的模拟参数与显式假设，不代表任何真实系统的实测 ASR；真实 ASR 必须用红队实测，且会随模型与攻击技巧变化。来源仅用于机制与结论的归属。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
