// Read complete SSE records independently of TCP, UTF-8, and CRLF chunk boundaries.
export async function consumeEventStream(response, onEvent) {
  if (!response.body) throw new Error("浏览器无法读取流式响应");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let event = "message";
  let data = [];
  let stopped = false;
  function line(value) {
    if (value.endsWith("\r")) value = value.slice(0, -1);
    if (value === "") {
      if (data.length) stopped = onEvent(event, JSON.parse(data.join("\n"))) === false;
      event = "message";
      data = [];
    } else if (value.startsWith("event:")) {
      event = value.slice(6).trim();
    } else if (value.startsWith("data:")) {
      data.push(value.slice(5).replace(/^ /, ""));
    }
  }
  try {
    while (!stopped) {
      const {value, done} = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, {stream: true});
      let newline;
      while (!stopped && (newline = buffer.indexOf("\n")) !== -1) {
        line(buffer.slice(0, newline));
        buffer = buffer.slice(newline + 1);
      }
      if (buffer.length > 1_000_000) throw new Error("流式响应格式异常");
      if (done) {
        if (buffer) line(buffer);
        if (data.length && !stopped) line("");
        break;
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
