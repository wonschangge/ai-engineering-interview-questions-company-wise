---
type: question
id: cohere-07
company: Cohere
topic: agents
order: 7
question: 设计一个能自动化企业工作流的 agent，比如根据内部文档和 CRM 起草 RFP 回复。「企业级」还会额外要求什么？
question_en: Design an agent that automates an enterprise workflow, say drafting RFP responses from internal documents and the CRM. What does 「enterprise-grade」 add?
asked_at: []
level: 高阶
tags: [企业 agent, 权限隔离, 人工审批, 可追溯, 幂等副作用]
sources:
  - title: 为什么不要构建多 agent 系统（延伸）
    url: https://cognition.ai/blog/dont-build-multi-agents
    author: Walden Yan (Cognition)
    published: 2025-06-12
  - title: 一个 agent 循环中如何处理工具调用错误、超时与重试（本仓库专题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
  - title: 企业级 RAG 的权限隔离与多租户（本仓库专题）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
  - title: 从 demo 到企业上线的检查清单（本仓库公司题库 · Databricks 篇）
    url: https://github.com/wonschangge/ai-engineering-interview-questions-company-wise
    author: 本仓库
    published: 
related: [cohere-08, cohere-09, agents-02, databricks-07, system-design-01]
updated: 2026-09-28
---

## 一句话答案

> **先给一个正确的基线设计**（否则谈「企业级」没有落点）：
> ```
> ① 解析 RFP（分节、抽取要求清单、识别必须逐条回应的合规项）
> ② 为每一节做检索：历史 RFP 回复库 + 产品/安全/法务文档 + CRM（客户、机会、历史交互）
> ③ 逐节起草（带引用）→ ④ 一致性检查（术语/价格/SLA 与权威源一致）
> ⑤ 缺口标注（找不到依据的段落明确写「未找到依据，需人工补充」）→ ⑥ 人工审阅 → ⑦ 导出/回写 CRM
> ```
> **「企业级」额外要求的是七件事**（这是本题的真正答案）：
> | # | 要求 | 具体做法 |
> | --- | --- | --- |
> | ① | **权限以「用户身份」执行** | agent 用**调用者的凭证**检索（不是超级 token）；行级安全 + 按账户隔离；**绝不把 A 客户的报价带进 B 客户的草稿** |
> | ② | **可追溯** | 每句话都能回到**源文档 + 版本 + 片段**；工具调用全留痕；草稿可**复现**（同输入同输出） |
> | ③ | **人工在环（强制审批门）** | **绝不自动对外发送**；对外承诺（价格/SLA/法律条款）必须人工确认；AI 起草、人负责 |
> | ④ | **依据约束（grounded）** | 无依据不断言：找不到就说找不到并标注缺口；**数字类（价格/折扣/SLA）不由模型生成**，而是从权威系统取 |
> | ⑤ | **副作用可控** | 回写 CRM / 发邮件必须**幂等 + 可撤销 + 分阶段**（草稿→审阅→发送）；重试不产生重复 |
> | ⑥ | **降级与失败语义** | CRM 不可用、检索为空、超时 → 明确降级（用已有材料起草 + 标出缺口），而不是编造或静默失败 |
> | ⑦ | **合规与治理** | 数据驻留、留存期、**不用于训练**、PII 处理、审计导出、负责人与 runbook |
> **两个必须量化的地方**（本轮的「可算」部分）：
> - **权限泄漏**：共享索引下「检索到不该看的文档」的概率与**每百份草稿的泄漏次数**——这是企业级最硬的指标（串 [[system-design-01]]）；
> - **ROI 的真实口径**：省下的**起草时间**要减去新增的**核验时间**——若核验成本高于起草节省，agent 就是负收益（这正是很多企业 PoC 失败的原因）。
> 一句话判据：**「这段内容是谁让看的？依据在哪？谁批准它发出去？出错了怎么撤回来？」**——四问答不上来，就不是企业级。

## 面试官在考什么

- **是否有可用的基线**：能否先把工作流拆成「解析 → 检索 → 逐节起草 → 一致性检查 → 审批 → 回写」，而不是直接谈多 agent 编排。
- **权限模型**：能否坚持「以用户身份执行」、行级安全、跨账户隔离——并给出**泄漏如何被检测**。
- **可追溯的粒度**：能否说清「引用到 span 级 + 文档版本」，以及草稿如何复现。
- **审批门的位置**：能否区分「内部草稿」与「对外承诺」，并指出**对外承诺必须人工确认**（价格/SLA/法律）。
- **groundedness 的工程手段**：强制引用、找不到就标注缺口、**数字不交给模型**（从权威系统取）、冲突时优先权威/最新。
- **副作用的幂等与撤销**：回写 CRM、发邮件、生成合同——每一样都要能重试不出错、能撤回。
- **ROI 的诚实口径**：把「审阅与核验时间」算进去；能否给出「什么条件下才值得上线」。
- **评测与长期维护**：模板/产品/价格都在变 → 需要**按客户模板的评测集**与回归门禁（串 [[databricks-07]]）。
- **多 agent 的判断**：能否说明**什么时候不该拆多 agent**（上下文共享与决策一致性 > 角色扮演式的分工，串 [[cognition-01]]）。

**常见错误答案**

- 直接讲「用多 agent 分工（研究员/写手/审稿）」而不谈权限、审批、可追溯。
- 用**超级权限 token** 检索（越权泄漏的经典来源）。
- 让模型**生成价格/SLA 数字**（幻觉进入合同）。
- 自动发送邮件/自动回写 CRM（无审批门、无幂等）。
- 引用只到「文档级」（无法核验具体句子）。
- ROI 只算起草时间（忽略核验成本）。
- 不做缺口标注（「没找到」被写成「我们有」）。
- 忽略模板漂移与模型升级带来的回归。

## 原理与推导

### 1. 权限：为什么「以用户身份执行」是不可妥协的

$$P(\text{泄漏})\approx 1-\prod_{\text{检索}}\big(1-P(\text{越权命中})\big)$$

**共享索引 + 后置过滤**的做法必然有窗口：检索发生在过滤之前（否则 ANN 无法高效），所以**必须把权限作为检索的一部分**（元数据过滤下推到向量检索，或按 ACL 分片）。两种设计对比：

| 设计 | 机制 | 风险 |
| --- | --- | --- |
| 共享索引 + 检索后过滤 | 先取 top-k，再删掉无权文档 | **top-k 被无权文档占满** → 召回下降；且过滤实现有 bug 就泄漏 |
| **检索前过滤（元数据下推）** | 把 ACL 作为过滤条件进入 ANN | 正确；代价是每个用户/账户的候选集不同（缓存难） |
| 按账户/租户分片 | 物理隔离索引 | 最安全；成本最高（N 份索引），适合高合规客户 |

**企业级的答案通常是「检索前过滤 + 关键租户物理分片」**，并配**泄漏检测**（离线审计：用「越权探针查询」定期验证不会返回不该返回的文档）。

### 2. 可追溯：引用到 span，且能复现

一条「可追溯」的记录至少包含：

```json
{"draft_span": [120, 260], "claim": "支持 SOC 2 Type II",
 "source": {"doc_id": "...", "version": 7, "chunk_id": "...", "span": [1024, 1102]},
 "retrieved_at": "...", "model": "...", "prompt_version": "..."}
```

**为什么必须到 span**：审阅者要**一眼核验**；只给文档名等于没给（文档可能 80 页）。**为什么必须记版本**：源文档更新后，旧草稿的依据才能被理解（审计要求）。

### 3. 审批门：把「对外承诺」单独隔离

| 内容类型 | 风险 | 审批 |
| --- | --- | --- |
| 产品功能描述 | 中（说错功能） | 产品/售前确认 |
| **价格/折扣** | **高（合同风险）** | **必须人工，且数字来自报价系统** |
| **SLA/合规承诺** | **高（法律风险）** | **法务/安全确认** |
| 公司介绍/案例 | 低 | 可用模板 |

**设计原则**：**模型负责「组织语言」，不负责「承诺内容」**；数字与承诺从权威系统取，并**在草稿里标出占位符**（例如 `[price: 从 CPQ 取]`）。

### 4. 副作用的幂等与撤销

$$\text{安全侧效应}=\text{幂等键}+\text{可撤销}+\text{分阶段}$$

- **幂等**：回写 CRM 用业务主键 upsert；发送邮件用 message-id 去重（串 [[agents-02]] 的重试纪律）；
- **可撤销**：**默认不发送**，先生成「待审草稿」；发送走**草稿→人工批准→发送**三段；
- **分阶段**：把「读取」与「写入」的权限分离（agent 默认只读；写入需显式授权，且限定对象）。

### 5. Groundedness 的三个机制

1. **强制引用**：每一句断言必须带引用，否则标红/阻断；
2. **缺口标注**：检索不到 → 输出「未找到依据，需人工补充」，**不允许编造**；
3. **数字外置**：价格、折扣、SLA 数值、认证有效期等**从权威系统 API 取**，模型只做插入（并校验来源）。

**冲突处理**：同一主题多份文档时，按**权威度（法务 > 产品 > 旧版营销）与时效**排序，并在草稿里提示「存在多个版本」。

### 6. ROI 的真实口径

$$T_{\text{人类基线}}=T_{\text{读 RFP}}+T_{\text{起草}}+T_{\text{核验}}+T_{\text{整理}}$$

$$T_{\text{agent 辅助}}=T_{\text{读 RFP}}+\underbrace{T_{\text{审阅草稿}}}_{\text{新增}}+\underbrace{T_{\text{核验引用}}}_{\text{新增}}+T_{\text{整理}}$$

**关键**：**起草时间省下来了，但审阅与核验时间是新增的**。若

$$T_{\text{起草}}^{\text{human}}-T_{\text{审阅}}+T_{\text{核验}}>0$$

才有正收益。**很多 PoC 失败正是因为核验成本被低估**——尤其是引用粒度粗、缺口没标、数字要逐条核对时。

## 数值与代码验证

### 表 1：权限设计 vs 每百份草稿的越权泄漏（见代码输出）

| 设计 | 越权命中率 | 每 100 份草稿泄漏段落数 | 召回损失 |
| --- | --- | --- | --- |
| 见输出 | 见输出 | 见输出 | 见输出 |

### 表 2：RFP 起草的 ROI（人类基线 vs agent 辅助）

| 环节 | 人类基线（小时） | agent 辅助（小时） |
| --- | --- | --- |
| 见输出 | 见输出 | 见输出 |

### 可运行代码

```python
# 企业级 RFP agent：权限泄漏量化、引用粒度与核验成本、ROI、审批门与幂等
import math, random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------- 1) 权限设计：越权泄漏与召回损失 ----------
@dataclass
class PermDesign:
    name: str
    mode: str                        # "post_filter" / "acl_pushdown" / "shard"
    filter_fail_rate: float          # 过滤实现失效的概率（后置过滤才依赖它）
    def leak_prob(self, k: int) -> float:
        """一次检索里"用户看到无权内容"的概率。
        后置过滤：内容先进了候选集，只靠过滤代码兜住 -> 泄漏概率 = 过滤失效率
        下推/分片：权限在检索或存储层强制 -> 泄漏概率≈0（只剩实现缺陷）"""
        if self.mode == "post_filter":
            return self.filter_fail_rate
        if self.mode == "acl_pushdown":
            return self.filter_fail_rate * 0.1        # 权限在存储层强制，残余风险低一个量级
        return 0.0
    def recall_loss(self, p_unauthorized: float, k: int) -> float:
        """后置过滤的隐性代价：top-k 名额被无权文档占掉"""
        if self.mode != "post_filter":
            return 0.0
        # 期望被占用的名额 = k * p_unauthorized（近似）
        return min(1.0, p_unauthorized)
DESIGNS = [
    PermDesign("共享索引 + 检索后过滤", "post_filter", 0.005),
    PermDesign("ACL 下推到检索", "acl_pushdown", 0.005),
    PermDesign("按租户物理分片", "shard", 0.0),
]
P_UNAUTH = 0.12          # 语料中"该用户无权查看"的文档占比
K = 20
print("① 权限设计对比（语料中 12% 的文档对该用户无权查看、k=20、过滤失效率 0.5% 为示例）")
print(f"  {'设计':<26} {'单次泄漏概率':>12} {'每100份草稿泄漏段落':>20} {'top-k 召回损失':>14}")
for d in DESIGNS:
    p_leak = d.leak_prob(K)
    per_100 = p_leak * K * 100
    print(f"  {d.name:<26} {p_leak:>12.4f} {per_100:>20.2f} {d.recall_loss(P_UNAUTH, K):>13.0%}")
print("  读法：**「先检索后过滤」有两条代价** —— ① 内容已进入候选集，只靠过滤代码兜住（失效即泄漏）；")
print("        ② top-k 名额被无权文档占掉，**召回同时下降 12%**（右列）。企业级做法是**ACL 下推**")
print("        （权限成为检索条件），高合规租户再加**物理分片**（泄漏为 0，代价是索引翻倍）")

# ---------- 2) 引用粒度 vs 核验成本（ROI 的关键） ----------
@dataclass
class Citation:
    granularity: str
    time_per_claim_min: float     # 核验一条断言所需时间
    n_claims: int = 60            # 一份 RFP 的断言数（示例）
    miss_rate: float = 0.0        # 未标注依据的断言比例（需要人工找）
    def verify_hours(self) -> float:
        return self.n_claims * self.time_per_claim_min / 60 * (1 + self.miss_rate * 3)
CITES = [
    Citation("无引用（全靠人工找）", 6.0, miss_rate=1.0),
    Citation("文档级引用", 2.5, miss_rate=0.25),
    Citation("span 级引用 + 缺口标注", 0.8, miss_rate=0.05),
]
print("")
print("② 引用粒度决定核验成本（一份 RFP 约 60 条断言）")
print(f"  {'引用粒度':<26} {'核验小时':>10} {'说明':<30}")
for c in CITES:
    note = {1.0: "几乎全部要人工定位", 0.25: "四分之一要人工找", 0.05: "少量缺口需人工补"}[c.miss_rate]
    print(f"  {c.granularity:<26} {c.verify_hours():>10.1f} {note:<30}")
print("  读法：**引用粒度直接决定「值不值得用」** —— span 级引用把核验从十几小时压到 1 小时级；")
print("        这也是「可追溯」不只是合规要求、而是**ROI 的关键变量**的原因")

# ---------- 3) RFP 起草的 ROI ----------
@dataclass
class Workflow:
    name: str
    read_rfp_h: float = 2.0
    draft_h: float = 20.0
    verify_h: float = 0.0
    assemble_h: float = 4.0
    review_h: float = 0.0
    hourly_cost: float = 80.0
    def total_h(self) -> float:
        return self.read_rfp_h + self.draft_h + self.verify_h + self.assemble_h + self.review_h
    def cost(self) -> float:
        return self.total_h() * self.hourly_cost
BASE = Workflow("人类基线", draft_h=20.0, verify_h=0.0, assemble_h=4.0, review_h=0.0)
CITE_SPAN = CITES[2].verify_hours()
CITE_DOC = CITES[1].verify_hours()
AGENT_GOOD = Workflow("agent + span 引用", draft_h=3.0, verify_h=CITE_SPAN, assemble_h=1.0,
                      review_h=1.5)
AGENT_POOR = Workflow("agent + 文档级引用（未标缺口）", draft_h=3.0, verify_h=CITE_DOC,
                      assemble_h=1.0, review_h=2.5)
print("")
print("③ RFP 起草 ROI（按 $80/小时；起草时间省下来了，但核验是新增成本）")
print(f"  {'方案':<32} {'总小时':>8} {'成本($)':>10} {'相对基线':>10}")
for w in (BASE, AGENT_GOOD, AGENT_POOR):
    print(f"  {w.name:<32} {w.total_h():>8.1f} {w.cost():>10,.0f} "
          f"{w.cost()/BASE.cost()-1:>9.0%}")
print(f"  盈亏平衡：只要核验时间 < {BASE.draft_h - AGENT_GOOD.draft_h + BASE.assemble_h - AGENT_GOOD.assemble_h:.1f} 小时就有正收益")
print("  读法：**ROI 的分水岭在核验成本** —— 同样的模型，引用做得好（span + 缺口标注）省下 60%+；")
print("        引用做得差，核验时间可能吃掉全部收益（这就是很多 PoC「演示很好、上线没人用」的原因）")

# ---------- 4) 审批门与风险：把门放在哪里 ----------
@dataclass
class Gate:
    content: str
    error_cost: float        # 出错一次的代价（美元）
    error_rate: float        # 未审批时的出错概率
    review_min: float        # 人工确认所需分钟
    hourly: float = 80.0
    round_trip_min: float = 20.0   # ★ 一次审批的往返固定成本（找人/切上下文），不随条目摊薄
    def expected_loss_without(self) -> float:
        return self.error_cost * self.error_rate
    def review_cost(self) -> float:
        return max(self.review_min, self.round_trip_min) / 60 * self.hourly
    def worth_gate(self) -> bool:
        return self.expected_loss_without() > self.review_cost()
GATES = [
    Gate("价格/折扣条款", 25_000, 0.05, 10),
    Gate("SLA 与合规承诺", 50_000, 0.03, 15),
    Gate("产品功能描述", 3_000, 0.10, 6),
    Gate("公司介绍与案例", 50, 0.15, 2),
]
print("")
print("④ 审批门经济学：把人工确认放在「期望损失 > 确认成本」的地方（含 20 分钟往返固定成本）")
print(f"  {'内容类型':<18} {'出错代价':>9} {'未审出错率':>9} {'期望损失':>10} {'确认成本':>9} {'该设门':<8}")
for g in GATES:
    print(f"  {g.content:<18} {g.error_cost:>9,} {g.error_rate:>9.1%} "
          f"{g.expected_loss_without():>10,.0f} {g.review_cost():>9,.1f} "
          f"{('是' if g.worth_gate() else '否'):<8}")
print("  读法：**审批门不是越多越好** —— 按期望损失排序：价格/SLA 必须设门，公司介绍类可自动；")
print("        门太多会让人工成本吃掉收益，门太少则把合同风险留给客户")

# ---------- 5) 企业级要求清单（可勾选） ----------
@dataclass
class Requirement:
    layer: str
    item: str
    evidence: str
REQS = [
    Requirement("权限", "以调用者身份检索（非超级 token）", "越权探针查询全部被拒"),
    Requirement("权限", "ACL 下推到检索 + 高合规租户物理分片", "架构图与分片配置"),
    Requirement("可追溯", "span 级引用 + 文档版本", "抽样 20 条断言可一键定位"),
    Requirement("可追溯", "工具调用留痕与草稿可复现", "同输入重跑得到同草稿"),
    Requirement("审批", "对外承诺强制人工门", "价格/SLA 段落带审批标记"),
    Requirement("依据", "无依据不断言 + 缺口标注", "缺口清单随草稿输出"),
    Requirement("依据", "数字从权威系统取（不由模型生成）", "价格字段来自 CPQ API"),
    Requirement("副作用", "回写/发送幂等且可分阶段", "重试不产生重复记录"),
    Requirement("降级", "CRM/检索失败时的降级路径", "故障注入测试通过"),
    Requirement("合规", "数据驻留、留存期、不用于训练", "合同与配置证据"),
    Requirement("评测", "按客户模板的评测集与回归门禁", "每版发布跑分记录"),
    Requirement("治理", "负责人 + runbook + 审计导出", "值班表与演练记录"),
]
print("")
print("⑤ 企业级要求清单（每项都要有「证据」，不是「我们做了」）")
print(f"  {'层':<8} {'要求':<38} 证据")
for r in REQS:
    print(f"  {r.layer:<8} {r.item[:36]:<38} {r.evidence}")
print("  读法：**这份清单就是本题的答案骨架** —— 权限、可追溯、审批、依据、副作用、降级、合规、")
print("        评测、治理；面试时按客户行业挑重点（金融/医疗先谈合规与权限，软件先谈 ROI 与评测）")
```

预期输出要点（实跑）：① **「先检索后过滤」的权限设计有两条代价**：在「语料中 12% 文档无权查看、k=20、过滤失效率 0.5%」的设定下，**每 100 份草稿约 10 个段落**来自无权文档（ACL 下推降到 **1 个**，物理分片为 **0**），且 **top-k 名额被占掉会同时让召回下降 12%**；② **引用粒度直接决定 ROI**：同样 60 条断言，**无引用需 24.0 小时**核验、**文档级引用 4.4 小时**、**span 级 + 缺口标注只要 0.9 小时**；③ ROI 计算显示 **agent + span 引用把总工时从 26.0 小时降到 8.4 小时（成本 −68%）**，而「文档级引用」版本只到 12.9 小时（−50%）——**分水岭在核验成本**（省下的起草时间要扣掉新增的审阅与核验）；④ 审批门经济学证明**门要按期望损失设**（价格/SLA 必设、公司介绍可自动）；⑤ 企业级要求清单给出每项的**验收证据**（越权探针、抽样定位、幂等重试测试等）。

## 常见追问

- **追问**：为什么不用多 agent（研究员/写手/审稿）分工？
  - 要点（串 [[cognition-01]]）：**任务是「上下文密集」而非「可并行」的**——RFP 的各节共享同一份客户上下文与产品事实，拆开会让每个 agent 重新检索、彼此结论冲突，**决策一致性反而变差**。多 agent 适合「可并行的独立子任务 + 明确接口」（例如同时查 5 个来源），不适合「一份文档的连贯撰写」。**先用单 agent + 好的工具与状态管理，再考虑拆分。**
- **追问**：怎么防止模型编造「我们有 SOC 2」这类断言？
  - 要点：① **强制引用**（无引用即阻断）；② **缺口标注**（检索不到就说「未找到」）；③ **认证类事实从权威系统取**（合规库/资产清单）；④ **评测集里专门放「应该拒答/标缺口」的样本**；⑤ 上线后**抽样审计**（每周抽 N 份草稿人工核验引用真实性）。
- **追问**：CRM 是「写」的目标，怎么保证安全？
  - 要点：① **默认只读**，写入需显式授权（并限定对象/字段）；② **幂等 upsert**（业务主键）；③ **分阶段**：草稿 → 人工批准 → 写入；④ **可撤销**（记录写入了什么，支持回滚）；⑤ **审计**（谁批的、写了什么、什么时候）。**绝不自动对外发送。**
- **追问**：模板会变、产品会变、价格会变，怎么长期维护？
  - 要点：① 模板/提示/工具/索引**全部版本化**；② **按客户模板的评测集**（模板改了就更新评测）；③ **漂移监控**（缺口率、引用命中率、人工修改率上升 → 报警）；④ **季度回归**（串 [[databricks-09]] 的模型下线迁移同源）；⑤ 明确**负责人**（业务 + 技术）。
- **追问**：怎么测「没有越权」？
  - 要点：**越权探针**——构造一组「该用户绝对不该看到」的文档与查询，定期自动验证「检索结果里不出现它们」；再加**红队测试**（用社工式提问诱导泄漏）；并把结果纳入发布门禁。**「我们做了权限」不是证据，「探针全绿」才是。**
- **追问**：人工审阅会不会变成新的瓶颈？
  - 要点：会——所以要**降低审阅成本**（span 级引用、缺口清单、变更高亮、把「必须看」的部分压缩到价格/SLA/法务三类），并**度量审阅时间**（如果审阅时间没有下降，说明引用与组织方式有问题）。**目标是降低「端到端」时间，而不是「生成」时间。**

## 相关题目

- [[cohere-08]]：气隙部署——同一套企业要求在「客户自有 GPU、无外网」下的变化。
- [[cohere-09]]：没有标注数据时如何评估——企业 agent 的验收前提。
- [[agents-02]]：工具调用的错误、超时与重试——本篇「副作用幂等」的基础。
- [[databricks-07]]：上线检查清单——本篇的企业级要求与那份清单同源。
- [[system-design-01]]：企业级 RAG 的权限隔离——本篇权限一节的方法论来源。

## 参考资料与归属

- **为什么不要构建多 agent 系统（延伸）** —— Walden Yan (Cognition)，2025-06-12：<https://cognition.ai/blog/dont-build-multi-agents>。第 5 节「上下文密集任务拆多 agent 会损失决策一致性」的论证来自这篇。
- **一个 agent 循环中如何处理工具调用错误、超时与重试（本仓库专题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 4 节「重试必须幂等、副作用要可撤销」的做法取自该专题文档。
- **企业级 RAG 的权限隔离与多租户（本仓库专题）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 1 节「ACL 下推 vs 检索后过滤」的对比与分片策略来自该专题文档。
- **从 demo 到企业上线的检查清单（本仓库公司题库 · Databricks 篇）** —— 本仓库：<https://github.com/wonschangge/ai-engineering-interview-questions-company-wise>。第 5 节「红线 + 证据」的验收口径与本篇的要求清单同源。
- **延伸来源说明**：表 1、表 2 与可运行代码中的全部数值（无权文档占比 12%、k=20、过滤失效率 0.5%、60 条断言/份、核验时间 6.0/2.5/0.8 分钟、缺口额外 3 倍成本、\$80/小时、基线与 agent 的各环节小时数、审批门的出错代价与 20 分钟往返固定成本）都是为演示取舍而构造的**示例参数与显式假设**；真实的泄漏概率、核验时间与合同风险必须用自家审计数据与法务口径校准（**「ACL 必须下推」「引用粒度决定核验成本」「审批门按期望损失设置」这三条结构性结论可直接使用**）。
- 本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
