const messages = document.querySelector("#messages");
const form = document.querySelector("#composer");
const input = document.querySelector("#question");
const send = document.querySelector("#send");
const statusText = document.querySelector("#status");
const statusDot = document.querySelector("#status-dot");
const clear = document.querySelector("#clear");
const history = [];

function escapeHtml(value) {
  return value.replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
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
  const mode = data.generator_mode === "llm"
    ? `DeepSeek ${escapeHtml(data.model_name || "V4")} · 证据约束生成`
    : "本地证据摘录";
  const evidence = data.citations.map(item => `
    <details>
      <summary>[${escapeHtml(item.evidence_id)}] 《${escapeHtml(item.document)}》PDF 第 ${item.pdf_page} 页</summary>
      <p class="meta">${escapeHtml(item.section.join(" › ") || "章节未识别")} · ${escapeHtml(item.source_type)} · ${escapeHtml(item.verification_status)}</p>
      <p class="quote">${escapeHtml(item.quote)}</p>
    </details>`).join("");
  const limits = data.limitations.length ? `<p class="limits">${data.limitations.map(escapeHtml).join(" · ")}</p>` : "";
  return `<p class="meta">${mode}</p><p>${escapeHtml(data.answer)}</p><div class="evidence">${evidence}</div>${limits}`;
}

async function ask(question) {
  addMessage("user", `<p>${escapeHtml(question)}</p>`);
  const pending = addMessage("assistant", '<span class="typing"><i></i><i></i><i></i></span>');
  send.disabled = true;
  input.disabled = true;
  try {
    const response = await fetch("/api/questions", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question, top_k: 8, history: history.slice(-8)})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "问答服务暂时不可用");
    pending.querySelector(".bubble").innerHTML = renderAnswer(data);
    history.push({role: "user", content: question}, {role: "assistant", content: data.answer});
  } catch (error) {
    pending.querySelector(".bubble").innerHTML = `<p>暂时无法回答：${escapeHtml(error.message)}</p>`;
  } finally {
    send.disabled = false;
    input.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", event => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  input.style.height = "auto";
  ask(question);
});
input.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
});
document.querySelectorAll(".suggestions button").forEach(button => button.addEventListener("click", () => ask(button.textContent)));
clear.addEventListener("click", () => {
  history.length = 0;
  messages.querySelectorAll(".message:not(.welcome)").forEach(node => node.remove());
  input.focus();
});

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
