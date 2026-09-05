# 人物交集独立复核说明

## 当前状态

M6.7 的规则开发集已经冻结。首批独立复核包位于 `evals/person_intersections_blind_review.json`，包含 40 条案例，覆盖 10 组人物、10 份文献和 23 个年份；与开发集使用的 16 个源事件零重合。生成时文件没有当前规则的预测结果，也没有预填正确答案。

复核者已确认 40 条标签均为正例，并明确授权不再补充逐案例依据。定稿产物位于 `evals/person_intersections_independent.json`，元数据记录为 `reviewer_confirmed_without_case_rationales`、`reason_quality: waived`，不能描述为可审计的逐页独立金标准。

v4 在该集合中命中 20 条、漏报 20 条，Recall=50%。由于没有负例，无法评估误报率；结果只说明现有保守规则仍有明显漏报，不代表全库准确率。

## 复核要求

1. 复核者不要运行人物交集查询、评测命令，也不要查看规则输出。
2. 对每条案例，打开 `evidence` 列出的原始 PDF 物理页并完整阅读；不能只看截取文字。
3. 判断指定两人是否被这个来源条目明确证明共同参与同一个动作。共同发件人与收件人、会议参与人与被谈论者要区分。
4. 只填写每条案例的 `annotation`：
   - `expected`：明确证明共同动作填 `true`，否则填 `false`；
   - `reason`：简短说明原页中的具体动作依据或为何证据不足；不能只写 `connected`、`yes`、“有关联”等标签式占位词；
   - `reviewed_by`：复核者标识；
   - `reviewed_at`：复核日期，必须使用 `YYYY-MM-DD`。
5. 不修改案例 ID、人物、源事件、文献、页码、证据内容或 `case_checksum`；定稿会校验这些不可变字段。`false` 只表示该来源条目证据不足，不表示两人在历史上不存在交集。

## 命令

重新生成同一批盲审案例（固定种子，输出可复现）：

```powershell
.\.venv\Scripts\history-agent.exe eval prepare-intersection-review --overwrite
```

复核完成后，先定稿为标准评测集：

```powershell
.\.venv\Scripts\history-agent.exe eval finalize-intersection-review
```

如果复核者明确确认全部标签、但决定不提供逐案例依据，可以在获得其明确授权后使用
`--accept-generic-reasons`。输出会记录 `reason_quality: waived` 和不同的
`review_method`，不得把这种结果描述为可审计的逐页独立金标准。

缺少任何判断、理由、复核者或日期时，命令会失败且不会生成评测集。定稿成功后运行：

```powershell
.\.venv\Scripts\history-agent.exe eval intersections `
  --case-set evals/person_intersections_independent.json
```

独立结果应与开发集分开报告。无论结果高低，都先记录原始指标和错误类型，再决定是否修改规则；一旦据此修改规则，这批案例就转为回归集，不能继续称为独立集。
