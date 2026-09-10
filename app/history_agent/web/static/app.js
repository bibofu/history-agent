import {renderMarkdown} from "/assets/markdown.js";
import {consumeEventStream} from "/assets/stream.js";

const messages = document.querySelector("#messages");
const form = document.querySelector("#composer");
const input = document.querySelector("#question");
const send = document.querySelector("#send");
const statusText = document.querySelector("#status");
const statusDot = document.querySelector("#status-dot");
const clear = document.querySelector("#clear");
const suggestions = document.querySelectorAll(".suggestions button");
let active = null;
const sessionStorageKey = "history-agent-session-v1";

function createSessionId() {
  return globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function loadSessionId() {
  try {
    const existing = localStorage.getItem(sessionStorageKey);
    if (existing) return existing;
    const created = createSessionId();
    localStorage.setItem(sessionStorageKey, created);
    return created;
  } catch {
    return createSessionId();
  }
}

let sessionId = loadSessionId();

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
}

function addMessage(role, html) {
  const item = document.createElement("article");
  item.className = `message ${role}`;
  item.innerHTML = `<div class="avatar">${role === "user" ? "问" : "史"}</div><div class="bubble">${html}</div>`;
  messages.appendChild(item);
  item.scrollIntoView({behavior: "smooth", block: "end"});
  return item;
}

function renderAnswer(data) {
  const structured = data.retrieval_mode.startsWith("structured_");
  const mode = data.generator_mode === "llm"
    ? `DeepSeek ${escapeHtml(data.model_name || "V4")} · 证据约束生成`
    : structured ? "结构化研究 · 候选与原文核查" : "本地证据摘录";
  const planner = data.query_planner_status === "used"
    ? ` · ${escapeHtml(data.query_planner_model || "DeepSeek")} 查询理解`
    : data.query_planner_status === "fallback" ? " · 原问题降级检索" : "";
  const reflection = data.retrieval_reflection_status === "retried"
    ? ` · ${data.retrieval_rounds || 2} 轮检索`
    : data.retrieval_reflection_status === "sufficient" ? " · 证据覆盖已评估" : "";
  const evidence = data.citations.map(item => `
    <details>
      <summary>[${escapeHtml(item.evidence_id)}] 《${escapeHtml(item.document)}》PDF 第 ${item.pdf_page}${item.pdf_page_end && item.pdf_page_end !== item.pdf_page ? `—${item.pdf_page_end}` : ""} 页</summary>
      <p class="meta">${escapeHtml(item.section.join(" › ") || "章节未识别")} · ${escapeHtml(item.source_type)} · ${escapeHtml(item.verification_status)}</p>
      <p class="quote">${escapeHtml(item.quote)}</p>
    </details>`).join("");
  const limits = data.limitations.length ? `<p class="limits">${data.limitations.map(escapeHtml).join(" · ")}</p>` : "";
  const removedClaims = data.llm_error_code === "removed_uncited_claims";
  const diagnostics = data.uncited_claims?.length ? `<details class="citation-diagnostics"><summary>${removedClaims ? "哪些草稿内容已移除" : "哪些语句缺少引用"}</summary><p class="meta">以下是未通过引用校验的草稿语句，不作为答案依据。</p><ul>${data.uncited_claims.map(claim => `<li>${escapeHtml(claim)}</li>`).join("")}</ul></details>` : "";
  return `<p class="meta">${mode}${planner}${reflection}</p><div class="markdown-body">${renderMarkdown(data.answer)}</div><div class="evidence">${evidence}</div>${limits}${diagnostics}`;
}

function setBusy(busy) {
  input.disabled = busy;
  send.textContent = busy ? "停止生成" : "发送";
  send.type = busy ? "button" : "submit";
  suggestions.forEach(button => { button.disabled = busy; });
}

function renderDraft(run) {
  if (active !== run) return;
  const follow = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 220;
  run.bubble.innerHTML = `<p class="meta stream-status" role="status">${escapeHtml(run.status)}</p><div class="markdown-body">${renderMarkdown(run.text)}</div>`;
  if (follow) run.pending.scrollIntoView({block: "end"});
}

function scheduleDraft(run) {
  if (run.frame) return;
  run.frame = requestAnimationFrame(() => {
    run.frame = null;
    renderDraft(run);
  });
}

function stopGeneration() {
  if (!active) return;
  const run = active;
  run.controller.abort();
  cancelAnimationFrame(run.frame);
  run.status = "已停止，回答未完成核查。";
  renderDraft(run);
  run.pending.removeAttribute("aria-busy");
  active = null;
  setBusy(false);
  input.focus();
}

async function ask(question) {
  if (active || !question.trim()) return;
  addMessage("user", `<p>${escapeHtml(question)}</p>`);
  const pending = addMessage("assistant", '<p class="meta" role="status">正在检索本地史料…</p>');
  pending.setAttribute("aria-busy", "true");
  const run = {controller: new AbortController(), pending, bubble: pending.querySelector(".bubble"), text: "", status: "正在检索本地史料…", frame: null};
  active = run;
  setBusy(true);
  let completed = false;
  try {
    const response = await fetch("/api/questions/stream", {
      method: "POST",
      headers: {"Content-Type": "application/json", "Accept": "text/event-stream"},
      body: JSON.stringify({question, top_k: 12, session_id: sessionId}),
      signal: run.controller.signal
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(typeof data.detail === "string" ? data.detail : "问答服务暂时不可用");
    }
    await consumeEventStream(response, (event, data) => {
      if (active !== run) return false;
      if (event === "status") {
        run.status = data.message;
        scheduleDraft(run);
      } else if (event === "delta") {
        run.text += data.text;
        scheduleDraft(run);
      } else if (event === "reset") {
        cancelAnimationFrame(run.frame);
        run.frame = null;
        run.text = "";
        run.status = data.message;
        renderDraft(run);
      } else if (event === "error") {
        throw new Error(data.message || "问答服务暂时不可用");
      } else if (event === "done") {
        cancelAnimationFrame(run.frame);
        run.frame = null;
        run.bubble.innerHTML = renderAnswer(data);
        completed = true;
        return false;
      }
      return true;
    });
    if (!completed && active === run) throw new Error("连接中断，回答未完成，请重新发送问题");
  } catch (error) {
    if (active === run) {
      cancelAnimationFrame(run.frame);
      run.frame = null;
      run.bubble.innerHTML = `<p>暂时无法回答：${escapeHtml(error.message)}</p>`;
    }
  } finally {
    cancelAnimationFrame(run.frame);
    pending.removeAttribute("aria-busy");
    if (active === run) {
      active = null;
      setBusy(false);
      input.focus();
    }
  }
}

send.addEventListener("click", event => {
  if (active) {
    event.preventDefault();
    stopGeneration();
  }
});
form.addEventListener("submit", event => {
  event.preventDefault();
  if (active) return;
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  input.style.height = "auto";
  ask(question);
});
input.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
});
suggestions.forEach(button => button.addEventListener("click", () => ask(button.textContent)));
clear.addEventListener("click", async () => {
  stopGeneration();
  const previousSessionId = sessionId;
  sessionId = createSessionId();
  try {
    localStorage.setItem(sessionStorageKey, sessionId);
  } catch {
    // The fresh in-memory id still prevents cleared context from being reused this page load.
  }
  messages.querySelectorAll(".message:not(.welcome)").forEach(node => node.remove());
  input.focus();
  try {
    await fetch(`/api/sessions/${encodeURIComponent(previousSessionId)}`, {method: "DELETE"});
  } catch {
    // A failed cleanup cannot reconnect the page to the old session id.
  }
});

async function restoreConversation() {
  const restoringSession = sessionId;
  try {
    const response = await fetch(`/api/sessions/${encodeURIComponent(restoringSession)}`);
    if (!response.ok) return;
    const data = await response.json();
    if (sessionId !== restoringSession) return;
    for (const item of data.messages || []) {
      if (item.role === "user") {
        addMessage("user", `<p>${escapeHtml(item.content)}</p>`);
      } else if (item.role === "assistant") {
        addMessage(
          "assistant",
          `<p class="meta">已恢复的回答</p><div class="markdown-body">${renderMarkdown(item.content)}</div>`
        );
      }
    }
  } catch {
    // Conversation recovery is best effort; question answering remains available.
  }
}

restoreConversation();

fetch("/api/health").then(result => result.json()).then(data => {
  const ready = data.status === "ok";
  statusDot.className = ready ? "ok" : "bad";
  statusText.textContent = ready
    ? `本地双索引已就绪 · ${data.llm_enabled ? `DeepSeek ${data.llm_model}` : "DeepSeek 密钥未配置 · 证据摘录模式"}`
    : "索引尚未完成，请先执行构建命令";
}).catch(() => {
  statusDot.className = "bad";
  statusText.textContent = "无法读取服务状态";
});
