// Local UI integration test. Install with:
// npm install --prefix data/frontend-deps --ignore-scripts --no-audit --no-fund playwright
import assert from "node:assert/strict";
import {createServer} from "node:http";
import {readFile, mkdir} from "node:fs/promises";
import {fileURLToPath} from "node:url";
import {chromium} from "../data/frontend-deps/node_modules/playwright/index.mjs";

const root = new URL("../app/history_agent/web/static/", import.meta.url);
const answer = '## 研究结论\n\n**重要事实**与*说明*。[E1]\n\n- 第一项\n- 第二项\n\n> 原文引述\n\n| 年份 | 事件 |\n| --- | --- |\n| 1935 | 会议 |\n\n```js\nconst text = "<tag>";\n```\n\n[文献](https://example.com/source)\n\n<img src="https://example.com/tracker" onerror="window.injected=true"><script>window.injected=true</script>[危险链接](javascript:alert(1))';
const requests = [];
const cancelled = [];
const rejectedClaim = '未引用的草稿：<img src="https://example.com/tracker" onerror="window.injected=true">';
const timers = new Set();
function later(fn, delay) {
  const timer = setTimeout(() => { timers.delete(timer); fn(); }, delay);
  timers.add(timer);
}
function send(res, event, data) { if (!res.destroyed) res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`); }
function final(text = answer) {
  return {answer: text, retrieval_mode: "hybrid_rrf", generator_mode: "llm", model_name: "local-fixture", llm_status: "used", limitations: [], citations: [{evidence_id: "E1", document: "测试文献", pdf_page: 12, section: [], source_type: "chronology", verification_status: "verified", quote: "测试原文，不使用任何本地史料。"}]};
}
const server = createServer(async (req, res) => {
  if (req.url === "/api/health") {
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({status: "ok", llm_enabled: false}));
    return;
  }
  if (req.url === "/api/questions/stream") {
    let body = "";
    for await (const chunk of req) body += chunk;
    const data = JSON.parse(body);
    requests.push(data);
    let done = false;
    res.on("close", () => { if (!done) cancelled.push(data.question); });
    res.writeHead(200, {"Content-Type": "text/event-stream", "Cache-Control": "no-cache"});
    send(res, "status", {message: "正在生成，引用待核查…"});
    send(res, "delta", {text: "## 正在输出\n\n**第一段**"});
    if (data.question.includes("中断")) {
      later(() => res.end(), 300);
    } else if (data.question.includes("错误")) {
      later(() => { send(res, "error", {message: "测试服务错误"}); res.end(); }, 300);
    } else if (data.question.includes("停止") || data.question.includes("清空")) {
      later(() => { send(res, "done", final("旧回答不应回来")); res.end(); }, 3000);
    } else if (data.question.includes("降级")) {
      later(() => {
        done = true;
        send(res, "done", {...final("本地证据摘录。[E1]"), generator_mode: "extractive", llm_status: "fallback", llm_error_code: "citation_repair_uncited_core_claim", uncited_claims: [rejectedClaim], limitations: ["生成回答仍有事实语句缺少引用，已改为展示证据摘录。"]});
        res.end();
      }, 350);
    } else if (data.question.includes("修复")) {
      later(() => send(res, "reset", {message: "正在补全引用…"}), 350);
      later(() => send(res, "delta", {text: "修复后的事实。[E1]"}), 650);
      later(() => { done = true; send(res, "done", final("修复后的事实。[E1]")); res.end(); }, 950);
    } else {
      later(() => send(res, "delta", {text: "\n\n第二段也已到达。"}), 450);
      later(() => { done = true; send(res, "done", final()); res.end(); }, 950);
    }
    return;
  }
  const path = req.url === "/" ? "index.html" : req.url.split("/").pop();
  try {
    const bytes = await readFile(new URL(path, root));
    res.setHeader("Content-Type", path.endsWith(".js") ? "text/javascript" : path.endsWith(".css") ? "text/css" : "text/html");
    res.end(bytes);
  } catch { res.writeHead(404).end(); }
});
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
const base = `http://127.0.0.1:${server.address().port}`;
let browser;
try {
  browser = await chromium.launch({channel: "msedge", headless: true});
  const page = await browser.newPage({viewport: {width: 1100, height: 1000}});
  const errors = [];
  const external = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.route("**/*", route => {
    if (!route.request().url().startsWith(base)) { external.push(route.request().url()); return route.abort(); }
    return route.continue();
  });
  await page.goto(base);
  async function ask(text) {
    await page.locator("#question").fill(text);
    await page.getByRole("button", {name: "发送", exact: true}).click();
  }
  const current = page.locator(".message.assistant:not(.welcome)").last();
  await ask("Markdown测试");
  await page.getByRole("heading", {name: "正在输出"}).waitFor();
  assert.equal(await page.getByRole("button", {name: "停止生成", exact: true}).count(), 1);
  await current.getByText("第二段也已到达。").waitFor();
  await page.getByRole("heading", {name: "研究结论"}).waitFor();
  assert.equal(await current.locator("strong").textContent(), "重要事实");
  assert.equal(await current.locator("li").count(), 2);
  assert.equal(await current.locator("table tbody td").count(), 2);
  assert.equal(await current.locator("blockquote").count(), 1);
  assert.match(await current.locator("pre code").textContent(), /<tag>/);
  assert.equal(await current.locator("img, script, [onerror], a[href^='javascript:']").count(), 0);
  assert.equal(await page.evaluate(() => window.injected), undefined);
  assert.deepEqual(external, []);
  await current.locator("summary").click();
  assert.equal(await current.locator("details").getAttribute("open"), "");
  await mkdir(new URL("../data/reports/", import.meta.url), {recursive: true});
  await page.screenshot({path: fileURLToPath(new URL("../data/reports/stream-markdown-desktop.png", import.meta.url)), fullPage: true});
  await page.setViewportSize({width: 390, height: 844});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
  await page.screenshot({path: fileURLToPath(new URL("../data/reports/stream-markdown-mobile.png", import.meta.url)), fullPage: true});
  await ask("停止测试");
  await current.locator("h2").waitFor();
  await page.getByRole("button", {name: "停止生成", exact: true}).click();
  await current.getByText("已停止，回答未完成核查。").waitFor();
  await ask("清空测试");
  await current.locator("h2").waitFor();
  await page.getByRole("button", {name: "清空会话"}).click();
  assert.equal(await page.locator(".message:not(.welcome)").count(), 0);
  await ask("修复测试");
  await current.getByText("正在补全引用…").waitFor();
  assert.equal(await current.locator("h2").count(), 0);
  await current.getByText("修复后的事实。[E1]", {exact: true}).waitFor();
  await page.getByRole("button", {name: "发送", exact: true}).waitFor();
  assert.deepEqual(requests.at(-1).history, []);
  await ask("中断测试");
  await current.getByText(/连接中断/).waitFor();
  assert.equal(await current.locator("h2").count(), 0);
  await ask("错误测试");
  await current.getByText(/测试服务错误/).waitFor();
  assert.equal(requests.at(-1).history.length, 2);
  assert.equal(requests.at(-1).history[1].content, "修复后的事实。[E1]");
  await ask("降级原因测试");
  const diagnostic = current.locator(".citation-diagnostics");
  await diagnostic.waitFor();
  assert.equal(await diagnostic.getAttribute("open"), null);
  await diagnostic.getByText("哪些语句缺少引用", {exact: true}).click();
  assert.equal(await diagnostic.locator("li").textContent(), rejectedClaim);
  assert.equal(await diagnostic.locator("img, script, [onerror]").count(), 0);
  assert.equal(await current.locator(".markdown-body").textContent(), "本地证据摘录。[E1]\n");
  assert.equal(await page.evaluate(() => window.injected), undefined);
  assert.deepEqual(external, []);
  await page.screenshot({path: fileURLToPath(new URL("../data/reports/citation-diagnostics-mobile.png", import.meta.url)), fullPage: true});
  await ask("后续错误测试");
  await current.getByText(/测试服务错误/).waitFor();
  assert.equal(requests.at(-1).history.at(-1).content, "本地证据摘录。[E1]");
  assert(!JSON.stringify(requests.at(-1).history).includes("未引用的草稿"));
  assert(cancelled.includes("停止测试") && cancelled.includes("清空测试"));
  assert.equal(await page.getByText("旧回答不应回来").count(), 0);
  assert.deepEqual(errors, []);
  console.log("PASS: incremental Markdown, headings/lists/tables/code/quotes, sanitization, mobile layout, citations, fallback diagnostics, abort, clear, repair, disconnect, history isolation.");
} finally {
  for (const timer of timers) clearTimeout(timer);
  await browser?.close();
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
}
