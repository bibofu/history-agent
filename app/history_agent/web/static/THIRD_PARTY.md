# 本地前端依赖

这些文件随应用打包，页面不依赖外部 CDN。升级时同步版本、许可证并运行 Markdown 安全渲染测试。

| 库 | 版本 | 本地文件 | 上游 |
| --- | --- | --- | --- |
| Marked | 18.0.11 | `marked.umd.js` | https://github.com/markedjs/marked |
| DOMPurify | 3.4.15 | `purify.min.js` | https://github.com/cure53/DOMPurify |

下载来源分别为 npm registry 的 `marked-18.0.11.tgz` 与 `dompurify-3.4.15.tgz`。对应许可证见 `marked.LICENSE.txt`、`dompurify.LICENSE.txt`、`dompurify.LICENSE-MPL.txt`。

Markdown HTML 经过允许标签/属性清单清理后才插入页面；不渲染脚本、表单、iframe 和图片，不自动加载模型输出中的远端媒体。链接仅允许 HTTP、HTTPS 和 mailto，打开时附带 `noopener noreferrer`。
