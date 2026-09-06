const allowedTags = [
  "p", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6", "strong", "em", "del",
  "blockquote", "ul", "ol", "li", "pre", "code", "a", "table", "thead", "tbody", "tr", "th", "td"
];

export function renderMarkdown(text) {
  if (!globalThis.marked || !globalThis.DOMPurify) {
    const plain = document.createElement("p");
    plain.textContent = text;
    return plain.outerHTML;
  }
  const fragment = DOMPurify.sanitize(marked.parse(text, {gfm: true, breaks: true}), {
    ALLOWED_TAGS: allowedTags,
    ALLOWED_ATTR: ["href", "title", "start", "align"],
    ALLOW_DATA_ATTR: false,
    RETURN_DOM_FRAGMENT: true
  });
  fragment.querySelectorAll("a").forEach(link => {
    const href = link.getAttribute("href");
    try {
      const url = new URL(href || "", location.href);
      if (!href || !["http:", "https:", "mailto:"].includes(url.protocol)) {
        link.removeAttribute("href");
      } else {
        link.setAttribute("rel", "noopener noreferrer");
        link.setAttribute("target", "_blank");
      }
    } catch {
      link.removeAttribute("href");
    }
  });
  fragment.querySelectorAll("table").forEach(table => {
    const wrapper = document.createElement("div");
    wrapper.className = "table-scroll";
    table.replaceWith(wrapper);
    wrapper.appendChild(table);
  });
  const container = document.createElement("div");
  container.appendChild(fragment);
  return container.innerHTML;
}
