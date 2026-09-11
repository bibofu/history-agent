# Unified RAG Golden Benchmark

## Why

旧评测分别回答了“目标文献是否出现”“最终回答是否带引用”“结构化记录是否仍可查询”等问题，但同一个 case 没有贯穿 RAG 各层。因而一次 chunking 或 embedding 变更后，即使最终回答分数不变，也很难判断是 retrieval 没有改善，还是改善被 context noise 或 generation 抵消。

Unified Golden Benchmark 使用一份经过复核的 page-level Golden Dataset，并按 case 声明的维度运行分层 evaluator：

```text
one Golden Dataset
    ├── routing evaluator
    ├── retrieval / ranking evaluator
    ├── candidate-context evaluator
    ├── generation / refusal evaluator
    └── citation evaluator
```

旧的 retrieval、answers、structured、intersection、semantic calibration 和 comprehensive eval 全部保留。Golden v1 是增量能力，不替代历史基线。

## Architecture

`golden_questions.json` 保存稳定标注；`history_agent.evaluation.golden` 提供严格 Pydantic schema、page mapping、纯指标函数、分层 runner 和结果聚合；生产 `answer_question` 只用于 routing、generation、citation 和 refusal 层，retrieval/ranking/context 层直接调用混合检索入口。

retrieval component 故意使用 case 原问题，不调用 LLM query planner。这样 chunk、embedding、BM25、fusion 和 Top-K 的对比不会被一次非确定性 query rewrite 混入。生产链路的 query understanding 和 routing 变化由 routing/E2E 层单独体现。

当前 context evaluator 的对象是混合检索返回的最终 Top-K candidate context。生产回答结果还记录实际使用的 citation bundle 数量，但当前 API 没有暴露完整内部 prompt，因此报告不会把 candidate context 冒充为 LLM prompt context。

## Golden annotation rules

- Gold identity 只能使用 `document_id + pdf_page`，禁止写入 `chunk_id`。
- `required_evidence` 是回答必须找到的关键页；`relevant_evidence` 是用于 precision/nDCG 的分级相关页集合。
- 只有人工确认相关页集合足够完整时，才可设置 `relevance_complete=true`。否则 Precision@K、nDCG 和“irrelevant context ratio”必须为 `null`。
- required page 若同时写入 `relevant_evidence`，必须标为 `relevance=2`；辅助相关页使用 `relevance=1`。
- `gold_facts` 是生成正确性的核心标签。`required` 与 `optional` 分开统计；`reference_answer` 只作阅读参考，不作为唯一答案。
- `deterministic_patterns` 是可审计的第一层匹配。启用 `--semantic-judge` 后，未命中的 fact 才交给语义 judge；结果保留 `covered=true|false|null`、`evaluable`、`judge` 和 `reason`。judge 未运行或失败时是不可评，不是未覆盖。
- `forbidden_claims` 明确记录共现升级、时间升级为因果、弱证据升级为强结论等不可接受陈述。确定性命中与语义引文 judge 是两个独立信号。
- `answerability=unanswerable` 的 case 不得声明 gold evidence 或 gold facts，重点评估拒答和无关引用。
- 每个 case 通过 `eval_dimensions` 选择参与的指标，不支持的维度不进入聚合分母。
- `metadata.reviewed` 不能在未人工核对时标为 true。迁移旧题必须保留 `legacy_case_id`。

Golden v1 包含 12 条从现有人工页码和必要事实标签迁移的示例，覆盖 timeline、intersection、event、viewpoint、organization、conflict、multi-hop、adversarial 和 refusal。v1 没有把旧标签臆测成完整 relevance set，因此所有 case 的 `relevance_complete` 都是 false。

## Page-level mapping

检索仍返回 chunk，evaluator 将一个 chunk 的 `pdf_page_start..pdf_page_end` 视为闭区间：只要 gold page 落在该区间内就算匹配。Recall 使用命中 gold page 的集合，因此一个 gold page 最多计数一次，一个跨页 chunk 可以同时召回多个 gold page。

Ranking 的稳定 evidence unit 定义为 `(document_id, page_start, page_end, normalized_text_sha256)`。相同 unit 只允许第一次获得 gain，并一次消费它覆盖的全部 gold pages；完全重复的跨页 chunk 因而不能再消费另一个 page。若一个跨页 unit 同时覆盖多个 graded gold pages，page-level 标签无法唯一确定这个“单个结果”的理想 gain，当前 nDCG 明确标为不可评，而不输出看似精确的数字。

Citation range 同样按闭区间与 gold page 求交。quote 会在范围内每个可用页面检查；若在非起始页命中则通过，若未命中且范围内有页面文本缺失则返回不可评。

当前 chunker 实际按物理页切分，`pdf_page_start == pdf_page_end`。闭区间实现保留了未来跨页 chunk 的兼容性。页级标注无法区分同一页内相关与无关段落，这是有意接受的 v1 粒度限制；未来可在不改变主键语义的情况下增加 text span。

## Metrics

### Implemented

Retrieval / ranking：

- HitRate@5、HitRate@10：Top-K 是否命中至少一个已标注相关页；
- Annotated Recall@5、Annotated Recall@10：命中的唯一已标注相关页数 / 已标注相关页数；兼容字段 `recall_at_k` 指向同一数值，不代表完整相关集；
- Required Evidence Recall@5、@10：命中的唯一 required evidence page 数 / required evidence page 数；
- Precision@5、Precision@10：相关 chunk 数 / K，仅对完整 relevance annotation 计算；
- MRR：第一个已标注相关页的倒数排名；
- nDCG@5、nDCG@10：使用 relevance 1/2，仅对完整 relevance annotation 计算；
- 每个 required evidence 的首次排名。

Context：

- known relevant ratio lower bound：即使标注不完整也可观察的相关比例下界；
- relevant evidence ratio / irrelevant context ratio：只对完整 relevance annotation 计算；
- start-page concentration ratio：相同文档起始页在 candidate context 中重复出现的比例；旧名 `duplicate_start_page_ratio` 作为兼容别名保留；
- redundancy ratio：稳定 page range 与规范化文本指纹均相同的 evidence unit 重复比例，同页不同文本不算重复。

Generation / refusal：

- Required Fact Recall、Optional Fact Recall；
- 每个 fact 的 deterministic/semantic/not-evaluated 判定与理由；
- Answer Correctness（基于当前可用的事实、forbidden、refusal 和语义信号）；
- Forbidden Claim Violation Rate；
- 语义 judge 检出的 unsupported claims；
- Refusal Accuracy 与 False Answer Rate。

Citation：

- citation presence 及其与 answerability 的一致性；
- citation page validity；
- citation quote-to-page consistency；
- deterministic claim-to-citation coverage；
- citation precision/recall（page gold；precision 仍要求完整 relevance）；
- semantic citation support（仅在显式启用 judge 时）。

Routing / operational：

- 多允许路由的 route accuracy 与 mismatch case IDs；
- retrieval、answer、total latency；
- answer、planner、reflection、fact judge 和 citation judge 的 Token 用量；
- Git commit/dirty state、prompt/chunker/index/embedding/RRF/Top-K/LLM 等自动可得的 run metadata。

所有 case metric 都带统一的 `metric_status.{name}={value,evaluable,reason}`。aggregate 只读取明确 `evaluable=true` 的数值，并继续输出 `{value,evaluable_cases}`，同时增加 `total_cases`、`unevaluable_cases` 和两类 case IDs。没有 forbidden labels、judge 未运行/失败、required fact 未评估等情况不会作为 0 分进入分母；若 required fact 有任一项不可评，Answer Correctness 也是不可评。

### Reserved / future work

- 完整 LLM prompt context 的独立 precision、token utilization 和跨证据 redundancy；
- text-span relevance；
- 开放式答案中全部 factual claims 的可靠分母，因此当前不声称实现严格的 claim-level Unsupported Claim Rate；
- 人工标注完备的 graded relevance case；在这些标注完成前，v1 的 Precision@n/nDCG 为 `null` 是正确结果；
- 独立供应商或人工复核的 semantic judge 校准扩充。

## How to run

通过现有 CLI：

```powershell
.\.venv\Scripts\history-agent.exe eval golden --dimension retrieval
.\.venv\Scripts\history-agent.exe eval golden --dimension all --limit 3 --run-name baseline
.\.venv\Scripts\history-agent.exe eval golden --dimension generation --case-id golden_event_zunyi_001
.\.venv\Scripts\history-agent.exe eval golden --dimension retrieval --category event --json
```

也可按模块运行：

```powershell
.\.venv\Scripts\python.exe -m history_agent.evaluation.golden `
  --dataset evals/golden/golden_questions.json `
  --dimension retrieval
```

默认 generation 使用无外部调用的 extractive fallback。需要真实生成与语义判定时显式使用：

```powershell
.\.venv\Scripts\history-agent.exe eval golden --dimension all `
  --with-llm --semantic-judge --run-name production-v14
```

详细结果写入 `data/reports/golden_benchmark_<run_id>.json`，并更新 `golden_benchmark_latest.json`。

chunk、keyword index 和 vector index 的构建报告均写入 manifest，记录实际 chunking 参数、chunk artifact SHA、embedding/index 版本、build run ID 和 Git commit。Golden 只读取这些 manifest；旧索引缺少 manifest 时相应字段保持 `null` 并产生 warning，不再从当前函数默认值反推历史实验配置。

## How to compare experiments

每次实验使用稳定数据集并给出名称，例如：

```powershell
# baseline 索引
.\.venv\Scripts\history-agent.exe eval golden --dimension all --run-name chunk650_bge

# 修改 chunking 并重建索引后
.\.venv\Scripts\history-agent.exe eval golden --dimension all --run-name chunk800_bge

# 只有 dataset SHA、case IDs 和 requested dimension 一致才计算 delta
.\.venv\Scripts\history-agent.exe eval golden-compare `
  data/reports/baseline.json data/reports/chunk800.json

# 机器可读输出；每项同时展示两边 denominator 和 evaluable case IDs
.\.venv\Scripts\history-agent.exe eval golden-compare `
  data/reports/baseline.json data/reports/chunk800.json --json
```

不要只比较 Answer Correctness。应对齐同一 dataset SHA 和 case IDs，再逐层比较：

```text
Recall@10                 +6%
nDCG@10                   +4%   # 仅完整 relevance case
RelevantContextRatio      -8%   # 仅完整 relevance case
RequiredFactRecall        +1%
AnswerCorrectness          0%
```

这种结果表示 retrieval 找到了更多、更靠前的 gold evidence，但 context noise 同时增加，提升没有充分传导到 generation。结果文件保留 per-case required evidence rank、fact 判定和路由 mismatch，可继续定位是哪一类 case 抵消了收益。

对比时还应确认 `run_metadata` 中 Git、index version、embedding、chunking、RRF、Top-K、prompt 和 LLM 等变量；无法从配置可靠取得的字段保持 `null`，不得手工伪造。
