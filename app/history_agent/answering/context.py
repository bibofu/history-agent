"""Persistent conversation storage and bounded prompt context construction."""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from history_agent.answering.models import ConversationMessage

EVIDENCE_MARKER = re.compile(r"\[E[1-9]\d*\]")
WHITESPACE = re.compile(r"[ \t]+")
CONTEXTUAL_PRONOUN = re.compile(
    r"(?:^|[，,。！？?!；;\s])(?:他|她|他们|她们|它|其)"
    r"(?:的|在|于|后来|当时|又|还|曾|是否|如何|为何|为什么|做|说|提出|经历|观点)"
)
ELLIPTICAL_FOLLOW_UP = re.compile(
    r"(?:呢|继续|接着说|再详细(?:一点|一些)?|展开(?:说说|介绍)?|还有吗|然后呢)"
    r"[？?。！!\s]*$"
)
CONTEXT_REFERENCE_PHRASES = (
    "刚才",
    "前面",
    "上述",
    "上一轮",
    "上一个",
    "前者",
    "后者",
    "这位",
    "那位",
    "这个人",
    "那个人",
    "这场",
    "那场",
    "该战役",
    "这次",
    "那次",
    "这件事",
    "那件事",
    "两人",
    "双方",
    "同一时期",
    "同期",
    "那年",
    "当年",
    "这一年",
    "同年",
)
MAX_STORED_MESSAGES_PER_SESSION = 200

CONVERSATION_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS conversation_sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversation_messages (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversation_messages_session
    ON conversation_messages(session_id, message_id);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def sanitize_history_content(content: str) -> str:
    """Make it impossible to mistake citations from an earlier turn for current evidence."""

    content = EVIDENCE_MARKER.sub("[历史引用]", content)
    return "\n".join(WHITESPACE.sub(" ", line).strip() for line in content.splitlines()).strip()


def requires_conversation_context(question: str) -> bool:
    """Return whether the current question explicitly depends on an earlier turn."""

    compact = question.strip()
    return (
        any(marker in compact for marker in CONTEXT_REFERENCE_PHRASES)
        or CONTEXTUAL_PRONOUN.search(compact) is not None
        or ELLIPTICAL_FOLLOW_UP.search(compact) is not None
    )


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    suffix = "\n[内容已截断]"
    if limit <= len(suffix):
        return text[:limit]
    return text[: limit - len(suffix)].rstrip() + suffix


def build_prompt_context(
    messages: list[ConversationMessage],
    *,
    max_messages: int = 12,
    max_chars: int = 12_000,
) -> list[ConversationMessage]:
    """Keep recent complete turns within budget and retain an index of older user topics."""

    if not messages or max_messages <= 0 or max_chars <= 0:
        return []
    sanitized = [
        ConversationMessage(role=item.role, content=sanitize_history_content(item.content))
        for item in messages
        if item.content.strip()
    ]
    selected: list[ConversationMessage] = []
    used = 0
    for item in reversed(sanitized):
        remaining = max_chars - used
        if remaining <= 0 or len(selected) >= max_messages:
            break
        content = _truncate(item.content, remaining)
        if not content:
            break
        selected.append(item.model_copy(update={"content": content}))
        used += len(content)
    selected.reverse()

    omitted = sanitized[: len(sanitized) - len(selected)]
    old_questions = [item.content for item in omitted if item.role == "user"][-6:]
    if old_questions and selected:
        topic_lines = [f"- {_truncate(question, 160)}" for question in old_questions]
        topic_index = (
            "更早对话的用户问题索引（仅用于解析指代，不是史实证据）：\n"
            + "\n".join(topic_lines)
        )
        available = max_chars - sum(len(item.content) for item in selected)
        if available >= 80:
            selected.insert(
                0,
                ConversationMessage(role="assistant", content=_truncate(topic_index, available)),
            )
            if len(selected) > max_messages:
                selected.pop(1)
    return selected


class ConversationStore:
    """Small SQLite store isolated from the research database."""

    def __init__(self, path: Path):
        self.path = path
        self._initialized = False
        self._lock = Lock()

    def _connect(self) -> sqlite3.Connection:
        self._ensure_initialized()
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=5.0)
            try:
                connection.executescript(CONVERSATION_SCHEMA)
                connection.commit()
            finally:
                connection.close()
            self._initialized = True

    def messages(self, session_id: str, *, limit: int = 200) -> list[ConversationMessage]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT role, content FROM conversation_messages "
                "WHERE session_id = ? ORDER BY message_id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [
            ConversationMessage(role=row["role"], content=row["content"])
            for row in reversed(rows)
        ]

    def append_exchange(self, session_id: str, question: str, answer: str) -> None:
        timestamp = _now()
        stored_answer = _truncate(answer, 10_000)
        with closing(self._connect()) as connection:
            connection.execute(
                "INSERT INTO conversation_sessions(session_id, created_at, updated_at) "
                "VALUES (?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET "
                "updated_at=excluded.updated_at",
                (session_id, timestamp, timestamp),
            )
            connection.executemany(
                "INSERT INTO conversation_messages(session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    (session_id, "user", question, timestamp),
                    (session_id, "assistant", stored_answer, timestamp),
                ),
            )
            connection.execute(
                "DELETE FROM conversation_messages WHERE session_id = ? AND message_id NOT IN "
                "(SELECT message_id FROM conversation_messages WHERE session_id = ? "
                "ORDER BY message_id DESC LIMIT ?)",
                (session_id, session_id, MAX_STORED_MESSAGES_PER_SESSION),
            )
            connection.commit()

    def clear(self, session_id: str) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                "DELETE FROM conversation_sessions WHERE session_id = ?", (session_id,)
            )
            connection.commit()
