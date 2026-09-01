# 结构化研究模型

本文定义 M6 人物、事件与关系层的可审计数据契约。当前已完成主数据、数据库表、校验模型、管理命令，以及《周恩来年谱》《林彪年谱》的首批规则事件抽取；自动结果仍不是已确认史实。

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

# 生成两部年谱的事件候选、SQLite 记录和质量报告；命令幂等
.\.venv\Scripts\history-agent.exe research extract-chronologies

# 提议合并，不立即修改数据
.\.venv\Scripts\history-agent.exe research people propose-merge SOURCE_ID TARGET_ID `
  --reason "人工核对理由" --proposed-by "研究者"

# 人工接受或拒绝合并
.\.venv\Scripts\history-agent.exe research people review-merge PROPOSAL_ID `
  --decision accepted --reviewed-by "复核者" --note "复核说明"
```

候选 JSONL 位于 `data/processed/events/`，最新报告位于 `data/reports/chronology_extraction_latest.json`。规则重复运行时，相同记录跳过；规则字段变化时只同步同版本且尚未人工确认的自动记录，`confirmed`、`rejected` 或其他来源记录不会被覆盖。首批基线见 `evals/CHRONOLOGY_BASELINE.md`。下一阶段 M6.4 使用模型辅助处理复杂事件，但仍必须通过同一 schema 和证据约束。
