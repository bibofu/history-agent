# 流式回答与 Markdown（2026-09-07）

## 行为

聊天页面改用 `POST /api/questions/stream`。服务端向 DeepSeek 请求真实的 `stream: true` 响应，收到正文增量就通过 SSE 发给浏览器；前端持续解析 UTF-8 和 SSE 帧，逐帧渲染 Markdown。未等待完整答案后再分片，未展示模型的 `reasoning_content`。原有 JSON 问答接口保持兼容，两个入口共用证据包、提示词（`grounded-answer-v6`）、引用修复请求和最终回答构造。

| 事件 | 数据 | 前端行为 |
| --- | --- | --- |
| `status` | `{message}` | 展示检索、生成、核查进度 |
| `delta` | `{text}` | 追加临时正文，支持未结束的 Markdown 块 |
| `reset` | `{message}` | 清除上一版草稿，准备引用修复 |
| `done` | 完整 `AnswerResponse` | 替换草稿，显示引用和限制，保存会话上下文 |
| `error` | `{message}` | 清除未完成草稿，显示可读错误，不写入上下文 |

模型输出结束后仍执行原有引用校验：只有缺少核心事实引用时允许修复一次；非法引用、截断、断流、超时或修复失败会降级到本地摘录，最终答案替换临时文字。一次修复的 token 用量合并统计。结构化或无模型路径直接发送 `done`，无需模拟逐字输出。

“停止生成”和“清空会话”使用 AbortController 取消读取；异步生成器关闭上游 HTTP 连接。检索在线程池运行，已开始的本地检索会完成，但取消后的请求不会继续进入模型生成。前端隔离请求状态，取消、异常和不完整草稿不会污染下一轮上下文。

## Markdown

支持标题、粗体、斜体、删除线、嵌套列表、引用、表格、代码块和链接。Marked 与 DOMPurify 固定版本随包提供；渲染前清理 HTML，禁止脚本、事件属性、表单和自动加载外部图片。表格与代码块可横向滚动，正文保持移动端宽度。

## 验证

运行 `scripts/check.ps1` 检查 Python 静态规则、类型、完整测试集；安装 Node.js 时同时检查前端语法及 `scripts/test-stream.mjs`。服务端测试使用 httpx 本地模拟传输，验证首个增量在后续内容读取前到达、Unicode、token 用量、引用修复、截断、超时、错误、取消连接清理及原 JSON 接口兼容。前端流测试覆盖 SSE 拆包、CRLF、非法 JSON 和完成后停止读取。

浏览器集成测试使用本机无界面 Edge 和本地虚构 SSE 服务，禁止外部请求：

```powershell
npm install --prefix data/frontend-deps --ignore-scripts --no-audit --no-fund playwright
node scripts/test-chat-ui.mjs
```

已验证完成前显示首段、Markdown 各主要元素、危险 HTML/链接清理、桌面/390px 移动端布局、引用折叠、停止、清空、修复替换、断流和会话隔离。截图保存在本地 `data/reports/stream-markdown-desktop.png` 与 `stream-markdown-mobile.png`。测试不发送本地史料，也不将模拟流结果称为真实 DeepSeek 联网验收。

本次完整检查通过：Ruff、mypy、198 项 Python 测试、3 项前端流解析测试，以及上述无界面浏览器集成测试。

流协议依据 [DeepSeek Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/)：仅输出 `delta.content`，记录 finish reason 和 usage，收到 `[DONE]` 且正常结束后进入引用核查。
