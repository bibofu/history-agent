# 结构化研究模型

本文定义 M6 人物、事件与关系层的可审计数据契约。当前已完成主数据、数据库表、校验模型、管理命令、《周恩来年谱》《林彪年谱》的首批规则事件抽取、模型辅助事件增强，以及跨来源事件去重；自动结果仍不是已确认史实。

## 1. 设计原则

- 稳定 ID 与原文形式分开保存：`person_id` 用于连接数据，`mention_text` 保留史料中的实际称呼；
- 时间不强行补全：年份、月份、日期和未知分别记录，同时保存“准确、约数、推断、未知”的确定性；
- 事件和关系必须绑定本地文献、PDF 物理页码和短证据，不能只保存模型结论；
- 两人“共同参会”必须引用同一个规范事件，简单共现不能生成交集；
- “直接下属”等高风险关系只能人工录入，并要求复核人、机构和时间语境；
- 人工修正保留原记录和修改原因，不覆盖抽取历史。

## 2. 人物主数据

人物目录位于 `config/person_aliases.json`，当前登记 25 位人物。每条记录包括：

```text
person_id
canonical_name
aliases[]
description
```

别名同时记录类型，如姓名、字、称谓、笔名和译名。`person_ambiguities` 保存同一称呼的候选人物；未解决歧义时，解析器返回全部候选，不自动选择其中一个。

人物合并采用两步流程：

1. `propose-merge` 只创建待审核提议，不修改人物；
2. `review-merge` 由明确的复核人接受或拒绝，保留理由、结论和时间。

被接受的来源人物标记为 `merged` 并指向目标人物，原记录和别名不删除。

## 3. 时间表示

事件和关系的起止时间都使用同一结构：

```json
{
  "value": "1956-01",
  "precision": "month",
  "certainty": "exact",
  "original_text": "一九五六年一月"
}
```

`precision` 可为 `day`、`month`、`year` 或 `unknown`；`certainty` 可为 `exact`、`approximate`、`inferred` 或 `unknown`。不知道具体日期时保留未知，不用虚构的月日占位。

## 4. 证据记录

事件和关系通过独立证据表复用同一原文依据：

```text
evidence_id
document_id
chunk_id
pdf_page_start
pdf_page_end
quote
extraction_methods
```

`document_id` 必须已在文献库登记，页码必须为 PDF 物理页码。相同 `evidence_id` 如果指向不同内容会被拒绝。

## 5. 事件

事件保存名称、类型、起止时间、地点、机构、描述、参与者、证据、抽取方法、置信度和复核状态。每个参与者必须保存稳定 `person_id`、当次角色、`mention_text` 和 `mention_source`。`mention_source=explicit` 表示正文点名；`chronology_subject` 表示主体只由年谱归属推得，不能伪装成原文提及。

自动抽取结果初始状态为 `unreviewed` 或 `needs_review`；只有经过规则或人工流程确认后才能用于确定性时间线。

年谱日期规则保留原始日期表达：精确日、月和年分别保存；日期区间保存起止点；“同日”“在此期间”等继承前一条日期并标记为 `inferred`；多个离散日期只保存近似首末边界并进入复核；终点早于起点等原书或文字层异常不会自动纠正，而是取消错误终点并标记 `needs_review`。跨页正文合并为一个事件，证据页码覆盖真实起止页。

## 6. 关系词表

关系词表位于 `config/relation_types.json`：

| 类型 | 含义 | 自动策略 | 关键约束 |
| --- | --- | --- | --- |
| `held_position` | 任职 | 允许抽取 | 必须有机构和职务 |
| `led` | 领导 | 仅生成候选 | 确认前需人工复核 |
| `reported_to` | 汇报 | 允许抽取 | 保存方向和时间语境 |
| `direct_subordinate` | 直接下属 | 仅人工 | 复核人、机构和时间必填 |
| `co_attended` | 共同参会 | 允许抽取 | 必须绑定共同事件 |
| `met_with` | 会见 | 允许抽取 | 必须绑定会见事件 |
| `member_of` | 隶属机构 | 仅生成候选 | 不自动解释为上下级 |

关系同时保存主体和客体的原文称呼，避免规范化后丢失史料表达。

## 7. 初始化与解析命令

```powershell
# 创建或迁移表，并同步人物与关系词表；命令幂等
.\.venv\Scripts\history-agent.exe research init

# 解析标准姓名或历史别名
.\.venv\Scripts\history-agent.exe research people resolve "伍豪" --json

# 生成周、林年谱及毛泽东年谱九卷的事件候选、SQLite 记录和质量报告；命令幂等
.\.venv\Scripts\history-agent.exe research extract-chronologies

# 提议合并，不立即修改数据
.\.venv\Scripts\history-agent.exe research people propose-merge SOURCE_ID TARGET_ID `
  --reason "人工核对理由" --proposed-by "研究者"

# 人工接受或拒绝合并
.\.venv\Scripts\history-agent.exe research people review-merge PROPOSAL_ID `
  --decision accepted --reviewed-by "复核者" --note "复核说明"
```

候选 JSONL 位于 `data/processed/events/`，最新报告位于 `data/reports/chronology_extraction_latest.json`。规则重复运行时，相同记录跳过；规则字段变化时只同步同版本且尚未人工确认的自动记录，`confirmed`、`rejected` 或其他来源记录不会被覆盖。首批基线见 `evals/CHRONOLOGY_BASELINE.md`。模型辅助结果同样必须通过 schema 和证据约束。

## 8. 模型辅助事件增强

M6.4 不让模型从整份文献自由生成事件，而是从已有规则事件中选择 `needs_review`、低置信度或通用 `activity` 候选。每个事件发送的全部证据正文合计硬限制为 1200 字符。模型只返回事件类型、行动原文、地点原文、机构原文和已登记人物的原文提及；日期、描述、证据 ID、文献 ID 和 PDF 页码不进入可修改字段。

本地校验和合并顺序如下：

1. 请求 DeepSeek 返回单个 JSON 对象；
2. 使用 Pydantic 严格 schema 校验，拒绝额外字段和非法枚举；
3. 检查行动、地点、机构、人名和角色能否在证据中找到连续原文，仅忽略排版空白差异；
4. 人名必须能唯一解析到人物主数据，未知或歧义人物使本次结果无效；
5. 只向原事件补充受控字段，保留原日期、描述、证据链接和既有人物；
6. 合并后的来源标记为 `rule_llm`，状态固定为 `needs_review`，不能自动确认；
7. 成功、无效和 API 失败都保存审计记录并进入复核队列。

`event_extraction_attempts` 保存 provider、模型名、提示词版本、抽取器版本、输入哈希、原始响应、schema 校验结果、Token 用量及调用前后事件快照。`event_review_queue` 保存待复核原因、优先级和对应调用。相同证据、模型与提示词版本的已处理请求默认跳过；只有显式使用 `--retry-failed` 才重试无效或失败记录。

```powershell
# 只看候选，不调用外部 API
.\.venv\Scripts\history-agent.exe research enrich-events --dry-run --limit 5 --json

# 会将所选事件的短证据发送到 DeepSeek；默认最多 5 条
.\.venv\Scripts\history-agent.exe research enrich-events --limit 5

# 可按文献或稳定事件 ID 缩小范围
.\.venv\Scripts\history-agent.exe research enrich-events `
  --document-id zhou_enlai_chronology_1949_1976 --limit 2

# 查看待复核项
.\.venv\Scripts\history-agent.exe research review-queue --limit 20 --json
```

模型增强报告位于 `data/reports/model_event_extraction_latest.json`。本地契约测试与首批 2 条真实 API 冒烟结果见 `evals/MODEL_EVENT_EXTRACTION.md`。任何后续批次仍应先明确资料出境范围，再从小批量开始。

## 9. 事件去重与多来源合并

去重层不修改 `historical_events`、`event_participants` 或 `event_evidence`。它根据日期生成跨文献候选对，再综合描述字符二元组相似度、相互包含度、共同人物、年谱主体、地点、机构和事件类型评分。只有精确到日、至少一条日期来自原文明确年份、文本高度相似且不超过两个成员的组合才进入 `high_confidence`；其余满足最低候选条件的组合进入 `uncertain/needs_review`。

规范层由以下表组成：

- `canonical_events`：代表性展示字段、置信度、候选类型、复核状态、字段差异和评分特征；
- `canonical_event_members`：每条源事件的完整快照和来源文献，保证可追溯、可重建；
- `canonical_event_participants`：来源参与者的并集；
- `canonical_event_evidence`：每条证据与其源事件的对应关系；
- `event_merge_review_queue`：不确定合并的人工复核队列；
- `canonical_event_reviews`：确认、驳回和重新打开的审计记录。

字段不一致时，规范事件选择一条代表值用于展示，同时在 `field_variants_json` 保存每个不同值及其源事件 ID。人工驳回只让规范事件失活，源事件仍可独立查询；重新打开会恢复规范事件和待审队列。已经人工确认或驳回的记录不会被后续自动同步覆盖。

```powershell
# 只计算候选和报告，不写规范表
.\.venv\Scripts\history-agent.exe research merge-events --dry-run --json

# 写入或幂等同步规范事件
.\.venv\Scripts\history-agent.exe research merge-events

# 查看人工待审队列
.\.venv\Scripts\history-agent.exe research merge-review-queue --limit 20 --json

# 确认、驳回或重新打开，不改动源事件
.\.venv\Scripts\history-agent.exe research review-event-merge CANONICAL_EVENT_ID `
  --decision confirmed --reviewed-by "研究者" --note "复核依据"
```

最新报告位于 `data/reports/event_merge_latest.json`，真实库基线与完整性审计见 `evals/EVENT_MERGE_BASELINE.md`。

## 10. 人物时间线查询

时间线不是只查规范事件表。查询服务联合两类记录：

1. 人物参与的活跃规范事件，每组只返回一次并附上全部来源证据；
2. 尚未被当前筛选条件下的规范事件覆盖的源事件。

第二条保证了两种安全回退：人工驳回规范合并后，成员源事件会重新出现；规范事件的展示类型或复核状态不符合筛选条件时，仍保留符合条件的来源字段差异。查询结果按起止时间和稳定 ID 排序，支持起止年份、事件类型、复核状态、`limit` 和 `offset`。

每条结果提供：

- `record_kind`：`canonical` 或 `source`；
- `source_event_ids`：可回溯的全部来源事件；
- `verification_level`：`confirmed`、`automatic` 或 `pending_review`；
- 日期精度与确定性、地点、机构、参与者和来源提及方式；
- 文献标题、PDF 起止页码、短引文和抽取方式；
- 规范事件的候选类型及 `field_variants` 来源差异。

```powershell
# 姓名、稳定 ID 和无歧义别名均可用于 CLI
.\.venv\Scripts\history-agent.exe research timeline 周恩来 --year 1956

# 可重复指定事件类型和复核状态
.\.venv\Scripts\history-agent.exe research timeline lin_biao `
  --start-year 1942 --end-year 1943 `
  --event-type correspondence --review-status confirmed --json
```

HTTP 接口为 `GET /api/people/{person_id}/timeline`，查询参数为 `start_year`、`end_year`、可重复的 `event_type` 与 `review_status`、`limit` 和 `offset`。API 只接受稳定 `person_id`，年份必须位于项目研究范围内。真实库验收见 `evals/TIMELINE_BASELINE.md`。

## 11. 人物交集候选查询

入口：`research intersections 毛泽东 周恩来 --year 1949 --json`；HTTP 为 `GET /api/people/{person_id}/intersections/{other_person_id}`，筛选参数与时间线相同，默认限定项目研究年份。CLI 支持无歧义别名，API 使用稳定 ID。

先筛选同时关联两人的事件，再在每条来源证据内核对共同动作。多人名单必须由已登记的规范姓名组成；支持共同出席、会见、联名致电等保守句式。年谱隐含主语只从源记录的 `chronology_subject` 继承，且只用于该来源首证据页开头，不能继承规范事件参与者并集。引号内容与冒号转述不用于匹配；不会拼接不同页、不同来源的分别提及。被驳回的来源不能支撑共同动作。

返回 `co_mention_total`（预筛事件数）、`total`（命中规则的候选数）及分页结果。每项的 `event` 复用时间线字段，保留规范/源事件类型、时间精度、地点（未知为 null）和来源差异；`joint_evidence` 提供证据 ID、来源事件 ID、原文片段、动作、双方角色及主语依据。分页在动作核验后进行。

交集自己的 `verification_status` 固定为 `needs_review`，不继承事件或合并的“已确认”状态。相同史实尚未成功去重时仍可能返回多条源候选，不额外强制合并。当前仅识别有限句式，不识别所有别名、复杂名单和跨页动作；零结果必须解释为“当前规则未找到”，不能宣称没有交集。没有新增确定关系记录；聊天接入规则见第 12 节。

后续更新（规则 v2）：对于已识别的年谱抽取器且文献主体唯一的源记录，也可从文献配置恢复主体，避免其在后文显名时丢失开头省略主语；仍只用于来源首证据页开头。样本清单与复算入口见 `evals/person_intersections.json` 和 `eval intersections`。

## 12. 结构化聊天路由

`POST /api/questions` 在混合检索之前识别明确的年度人物时间线与双人交集问题。返回沿用 `AnswerResponse`，`retrieval_mode` 分别为 `structured_timeline` / `structured_intersection`。这两条路由固定为证据摘录模式，`llm_status=not_applicable`，无模型调用；有候选时 `evidence_status=partial`，不把规则候选提升为 supported。

`top_k` 控制按时间展示前多少条（最多 12），正文提示总数及展示数。时间线是人物相关记录，不保证全部为亲历；交集使用 `joint_evidence` 实际命中的来源短引文，不用规范事件的另一条代表证据替代。日期明确标为索引记录日期，保留确定性说明。`Citation` 新增可选 `pdf_page_end`，前端显示完整页码范围；旧调用不受影响。

当前支持规范姓名、无歧义别名及整年/连续年份区间。对月份、地点、未识别人物、颠倒年份、同一人的两个别名、缺少人物/年份等情况给出澄清或范围提示；省略式追问不自动继承人物。数据库不可用时不转成宽泛的人名共现答案。一般观点与解释问题保持原混合检索流程。未来仍需扩充独立评估与上下文解析，不把当前小样本当作完整 M6.7 验收。
