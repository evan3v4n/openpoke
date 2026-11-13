"""Agent roster: SQLite-backed store of execution agents and what they have dealt with.

Per agent we keep a summary, recent email subjects, the people involved (the main
lookup key), thread ids and open drafts. The retriever (retriever.py) ranks agents
against an incoming message using this data; this module only stores and reads it.

The same database also holds a small retrieval log (what was shown, what the
interaction agent then chose) so retrieval quality can be measured.

A legacy roster.json (list of names, or dicts) is imported once, the first time
the database is created.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from ...logging_config import logger
from .log_store import _slugify
from .people import PersonRef, people_from_text

MAX_PEOPLE_PER_AGENT = 50
MAX_THREADS_PER_AGENT = 20
MAX_SUBJECTS_PER_AGENT = 20
MAX_RETRIEVAL_LOG_ROWS = 2000
SUMMARY_MAX_CHARS = 160
MESSAGE_LOG_CHARS = 500

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    name TEXT PRIMARY KEY,
    slug TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    subjects TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    last_active_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_people (
    agent_name TEXT NOT NULL,
    email TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    roles TEXT NOT NULL DEFAULT '[]',
    count INTEGER NOT NULL DEFAULT 0,
    last_seen TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (agent_name, email)
);
CREATE INDEX IF NOT EXISTS idx_people_email ON agent_people (email);
CREATE TABLE IF NOT EXISTS agent_threads (
    agent_name TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    PRIMARY KEY (agent_name, thread_id)
);
CREATE TABLE IF NOT EXISTS agent_drafts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    draft_id TEXT,
    recipient TEXT NOT NULL DEFAULT '',
    subject TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    closed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_drafts_open ON agent_drafts (closed_at, agent_name);
CREATE TABLE IF NOT EXISTS retrieval_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    message TEXT NOT NULL,
    candidates TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS retrieval_choices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    retrieval_id INTEGER,
    ts TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    was_new INTEGER NOT NULL,
    was_candidate INTEGER NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_MARKS = re.compile(r"(\*\*|__|`+|^#+\s*|^\s*[-*•]\s+)", re.MULTILINE)
_EMPTY_PARENS = re.compile(r"\(\s*\)")


def _clean_summary(text: str, redact: Iterable[str] = ()) -> str:
    """Plain-text summary: no markdown, and the account owner's address replaced by "the user"."""

    cleaned = _MD_LINK.sub(r"\1", text or "")
    cleaned = _MD_MARKS.sub(" ", cleaned)
    for value in redact:
        if value:
            cleaned = re.sub(re.escape(value), "the user", cleaned, flags=re.IGNORECASE)
    cleaned = _EMPTY_PARENS.sub("", cleaned)
    return _truncate(cleaned)


def _truncate(text: str, limit: int = SUMMARY_MAX_CHARS) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


class AgentRoster:
    """Persistent agent index. All methods are safe to call from threads and tasks."""

    def __init__(self, db_path: Path, legacy_json_path: Optional[Path] = None):
        self._db_path = db_path
        self._legacy_json_path = legacy_json_path
        self._lock = threading.RLock()

        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to create roster directory: {exc}")

        # WAL cannot be enabled inside a transaction, so use a plain connection first.
        setup = sqlite3.connect(db_path, timeout=30, isolation_level=None)
        try:
            setup.execute("PRAGMA journal_mode=WAL;")
        finally:
            setup.close()

        # Create the schema and import the legacy roster in one transaction, so a crash
        # in between can never leave a database that skips the import.
        with self._tx() as conn:
            existed = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'agents'"
            ).fetchone()
            for statement in _SCHEMA.split(";"):
                if statement.strip():
                    conn.execute(statement)
            self._migrate(conn)
            if not existed and legacy_json_path is not None:
                self._import_legacy(conn, legacy_json_path)

    # ------------------------------------------------------------------ plumbing

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """One serialized transaction on a short-lived connection."""

        with self._lock:
            conn = sqlite3.connect(self._db_path, timeout=30, isolation_level=None)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("BEGIN")
                yield conn
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            finally:
                conn.close()

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Bring databases created by earlier versions up to the current schema."""

        columns = {row["name"] for row in conn.execute("PRAGMA table_info(agent_people)")}
        if "last_seen" not in columns:
            conn.execute("ALTER TABLE agent_people ADD COLUMN last_seen TEXT NOT NULL DEFAULT ''")

    @staticmethod
    def _import_legacy(conn: sqlite3.Connection, path: Path) -> None:
        """Import agents from the old roster.json (names or dict entries)."""

        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Could not import legacy roster.json: {exc}")
            return
        if not isinstance(data, list):
            return

        imported = 0
        for raw in data:
            item = {"name": raw} if isinstance(raw, str) else raw
            if not isinstance(item, dict) or not str(item.get("name", "")).strip():
                continue
            name = str(item["name"])
            created = str(item.get("created_at") or _now())
            conn.execute(
                "INSERT OR IGNORE INTO agents (name, slug, summary, subjects, created_at, last_active_at)"
                " VALUES (?, ?, ?, '[]', ?, ?)",
                (
                    name,
                    _slugify(name),
                    _truncate(str(item.get("summary") or "")),
                    created,
                    str(item.get("last_active_at") or created),
                ),
            )
            people = item.get("people")
            if isinstance(people, dict):
                for email, info in people.items():
                    if isinstance(info, dict):
                        conn.execute(
                            "INSERT OR IGNORE INTO agent_people (agent_name, email, name, roles, count)"
                            " VALUES (?, ?, ?, ?, ?)",
                            (
                                name,
                                str(email).lower(),
                                str(info.get("name", "")),
                                json.dumps([str(r) for r in info.get("roles", [])]),
                                int(info.get("count", 1)),
                            ),
                        )
            imported += 1
        if imported:
            logger.info(f"Imported {imported} agents from legacy roster.json")

    def load(self) -> None:
        """Kept for backward compatibility; SQLite is always current."""

    # ----------------------------------------------------------------- mutations

    def add_agent(self, agent_name: str) -> None:
        """Add an agent to the roster if not already present."""

        with self._tx() as conn:
            self._ensure(conn, agent_name)

    def touch(
        self,
        agent_name: str,
        instructions: Optional[str] = None,
        exclude: Iterable[str] = (),
    ) -> None:
        """Create the agent if needed and mark it active now.

        Emails mentioned in the instructions are recorded as people (except any in `exclude`,
        e.g. the account owner), and the instructions seed the summary until the agent
        produces a real response.
        """

        excluded = {e.lower() for e in exclude if e}

        with self._tx() as conn:
            self._ensure(conn, agent_name)
            conn.execute(
                "UPDATE agents SET last_active_at = ? WHERE name = ?", (_now(), agent_name)
            )
            if instructions:
                conn.execute(
                    "UPDATE agents SET summary = ? WHERE name = ? AND summary = ''",
                    (_clean_summary(instructions), agent_name),
                )
                mentioned = [r for r in people_from_text(instructions) if r.email not in excluded]
                self._merge_people(conn, agent_name, mentioned)

    def record_activity(
        self,
        agent_name: str,
        people: Iterable[PersonRef] = (),
        thread_ids: Iterable[str] = (),
        subjects: Iterable[str] = (),
    ) -> None:
        """Attach people, threads and subjects seen during a tool call. No-op for unknown agents."""

        people, thread_ids, subjects = list(people), list(thread_ids), list(subjects)
        if not (people or thread_ids or subjects):
            return

        with self._tx() as conn:
            row = conn.execute("SELECT subjects FROM agents WHERE name = ?", (agent_name,)).fetchone()
            if row is None:
                return

            self._merge_people(conn, agent_name, people)

            now = _now()
            for thread_id in thread_ids:
                conn.execute(
                    "INSERT INTO agent_threads (agent_name, thread_id, last_seen) VALUES (?, ?, ?)"
                    " ON CONFLICT (agent_name, thread_id) DO UPDATE SET last_seen = excluded.last_seen",
                    (agent_name, thread_id, now),
                )
            conn.execute(
                "DELETE FROM agent_threads WHERE agent_name = ? AND thread_id NOT IN ("
                " SELECT thread_id FROM agent_threads WHERE agent_name = ?"
                " ORDER BY last_seen DESC, rowid ASC LIMIT ?)",
                (agent_name, agent_name, MAX_THREADS_PER_AGENT),
            )

            stored: List[str] = json.loads(row["subjects"] or "[]")
            for subject in subjects:
                if subject in stored:
                    stored.remove(subject)
                stored.append(subject)
            conn.execute(
                "UPDATE agents SET subjects = ?, last_active_at = ? WHERE name = ?",
                (json.dumps(stored[-MAX_SUBJECTS_PER_AGENT:]), now, agent_name),
            )

    def set_summary(self, agent_name: str, text: str, redact: Iterable[str] = ()) -> None:
        """Replace the summary with the start of the agent's latest response.

        Markdown is stripped and any `redact` strings (the account owner's address) are removed.
        """

        summary = _clean_summary(text, redact)
        if not summary:
            return
        with self._tx() as conn:
            conn.execute(
                "UPDATE agents SET summary = ?, last_active_at = ? WHERE name = ?",
                (summary, _now(), agent_name),
            )

    def open_draft(
        self, agent_name: str, draft_id: Optional[str], recipient: str = "", subject: str = ""
    ) -> None:
        """Record an unsent draft: an open loop that makes this agent a likely follow-up target."""

        with self._tx() as conn:
            if conn.execute("SELECT 1 FROM agents WHERE name = ?", (agent_name,)).fetchone() is None:
                return
            conn.execute(
                "INSERT INTO agent_drafts (agent_name, draft_id, recipient, subject, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (agent_name, draft_id, recipient, subject, _now()),
            )

    def close_draft(self, draft_id: Optional[str], agent_name: Optional[str] = None) -> None:
        """Close a draft once it is sent or deleted.

        Matches by draft id. If the id is unknown (it could not be parsed when the
        draft was created), falls back to the agent's oldest open draft without an id.
        """

        now = _now()
        with self._tx() as conn:
            if draft_id:
                cursor = conn.execute(
                    "UPDATE agent_drafts SET closed_at = ? WHERE draft_id = ? AND closed_at IS NULL",
                    (now, draft_id),
                )
                if cursor.rowcount:
                    return
            if agent_name:
                conn.execute(
                    "UPDATE agent_drafts SET closed_at = ? WHERE id = ("
                    " SELECT id FROM agent_drafts WHERE agent_name = ? AND draft_id IS NULL"
                    " AND closed_at IS NULL ORDER BY id LIMIT 1)",
                    (now, agent_name),
                )

    def clear(self) -> None:
        """Remove every agent and the retrieval log (used when chat history is cleared)."""

        with self._tx() as conn:
            for table in (
                "agents",
                "agent_people",
                "agent_threads",
                "agent_drafts",
                "retrieval_log",
                "retrieval_choices",
            ):
                conn.execute(f"DELETE FROM {table}")
        try:
            if self._legacy_json_path is not None and self._legacy_json_path.exists():
                self._legacy_json_path.unlink()
        except Exception as exc:
            logger.warning(f"Failed to remove legacy roster.json: {exc}")
        logger.info("Cleared agent roster")

    @staticmethod
    def _ensure(conn: sqlite3.Connection, agent_name: str) -> None:
        now = _now()
        conn.execute(
            "INSERT OR IGNORE INTO agents (name, slug, created_at, last_active_at) VALUES (?, ?, ?, ?)",
            (agent_name, _slugify(agent_name), now, now),
        )

    @staticmethod
    def _merge_people(conn: sqlite3.Connection, agent_name: str, refs: Iterable[PersonRef]) -> None:
        now = _now()
        for ref in refs:
            row = conn.execute(
                "SELECT name, roles, count FROM agent_people WHERE agent_name = ? AND email = ?",
                (agent_name, ref.email),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO agent_people (agent_name, email, name, roles, count, last_seen)"
                    " VALUES (?, ?, ?, ?, 1, ?)",
                    (agent_name, ref.email, ref.name, json.dumps([ref.role]), now),
                )
                continue
            roles = json.loads(row["roles"])
            if ref.role not in roles:
                roles.append(ref.role)
            conn.execute(
                "UPDATE agent_people SET name = ?, roles = ?, count = count + 1, last_seen = ?"
                " WHERE agent_name = ? AND email = ?",
                (row["name"] or ref.name, json.dumps(roles), now, agent_name, ref.email),
            )

        # Over the cap, keep the people seen most often, then most recently. Within one batch
        # (same timestamp) the first-inserted wins, i.e. the newest emails of a search result.
        conn.execute(
            "DELETE FROM agent_people WHERE agent_name = ? AND email NOT IN ("
            " SELECT email FROM agent_people WHERE agent_name = ?"
            " ORDER BY count DESC, last_seen DESC, rowid ASC LIMIT ?)",
            (agent_name, agent_name, MAX_PEOPLE_PER_AGENT),
        )

    # --------------------------------------------------------------------- reads

    def count(self) -> int:
        with self._tx() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0])

    def get_agents(self) -> List[str]:
        """All agent names, oldest first."""

        with self._tx() as conn:
            rows = conn.execute("SELECT name FROM agents ORDER BY created_at, rowid").fetchall()
        return [row["name"] for row in rows]

    def agent_rows(self) -> List[Dict[str, Any]]:
        """Light per-agent rows for ranking: name, summary, subjects, timestamps."""

        with self._tx() as conn:
            rows = conn.execute(
                "SELECT name, summary, subjects, created_at, last_active_at FROM agents"
            ).fetchall()
        return [
            {**dict(row), "subjects": json.loads(row["subjects"] or "[]")} for row in rows
        ]

    def people_rows(self) -> List[Tuple[str, str, str]]:
        """(agent_name, email, display name) for every indexed person."""

        with self._tx() as conn:
            rows = conn.execute("SELECT agent_name, email, name FROM agent_people").fetchall()
        return [(r["agent_name"], r["email"], r["name"]) for r in rows]

    def open_drafts(self, max_age_days: float = 7.0) -> Dict[str, List[Dict[str, str]]]:
        """Unsent drafts per agent, ignoring stale ones."""

        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat(timespec="seconds")
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT agent_name, recipient, subject FROM agent_drafts"
                " WHERE closed_at IS NULL AND created_at >= ? ORDER BY id",
                (cutoff,),
            ).fetchall()
        drafts: Dict[str, List[Dict[str, str]]] = {}
        for row in rows:
            drafts.setdefault(row["agent_name"], []).append(
                {"recipient": row["recipient"], "subject": row["subject"]}
            )
        return drafts

    def get_entry(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Full entry for one agent, or None."""

        with self._tx() as conn:
            row = conn.execute("SELECT * FROM agents WHERE name = ?", (agent_name,)).fetchone()
            if row is None:
                return None
            people = {
                r["email"]: {"name": r["name"], "roles": json.loads(r["roles"]), "count": r["count"]}
                for r in conn.execute(
                    "SELECT email, name, roles, count FROM agent_people WHERE agent_name = ?"
                    " ORDER BY count DESC, email",
                    (agent_name,),
                )
            }
            threads = [
                r["thread_id"]
                for r in conn.execute(
                    "SELECT thread_id FROM agent_threads WHERE agent_name = ? ORDER BY last_seen, rowid",
                    (agent_name,),
                )
            ]
            drafts = [
                {"draft_id": r["draft_id"], "recipient": r["recipient"], "subject": r["subject"]}
                for r in conn.execute(
                    "SELECT draft_id, recipient, subject FROM agent_drafts"
                    " WHERE agent_name = ? AND closed_at IS NULL ORDER BY id",
                    (agent_name,),
                )
            ]
        return {
            "name": row["name"],
            "slug": row["slug"],
            "summary": row["summary"],
            "subjects": json.loads(row["subjects"] or "[]"),
            "created_at": row["created_at"],
            "last_active_at": row["last_active_at"],
            "people": people,
            "thread_ids": threads,
            "open_drafts": drafts,
        }

    def get_entries(self) -> List[Dict[str, Any]]:
        """All entries, most recently active first."""

        with self._tx() as conn:
            names = [
                r["name"]
                for r in conn.execute("SELECT name FROM agents ORDER BY last_active_at DESC, rowid DESC")
            ]
        return [entry for entry in (self.get_entry(n) for n in names) if entry]

    def find_by_person(self, query: str) -> List[Dict[str, Any]]:
        """Entries whose people match `query` (email or name, case-insensitive substring)."""

        needle = (query or "").strip().lower()
        if not needle:
            return []
        matched = {
            agent
            for agent, email, name in self.people_rows()
            if needle in email or needle in name.lower()
        }
        return [e for e in self.get_entries() if e["name"] in matched]

    # ----------------------------------------------------------- retrieval logging

    def log_retrieval(self, kind: str, message: str, candidates: List[Dict[str, Any]]) -> int:
        """Record what was shown to the interaction agent for a turn. Returns the log id."""

        with self._tx() as conn:
            cursor = conn.execute(
                "INSERT INTO retrieval_log (ts, kind, message, candidates) VALUES (?, ?, ?, ?)",
                (_now(), kind, message[:MESSAGE_LOG_CHARS], json.dumps(candidates, default=str)),
            )
            retrieval_id = int(cursor.lastrowid)
            conn.execute(
                "DELETE FROM retrieval_log WHERE id <= ?", (retrieval_id - MAX_RETRIEVAL_LOG_ROWS,)
            )
        return retrieval_id

    def log_choice(
        self, retrieval_id: Optional[int], agent_name: str, was_new: bool, was_candidate: bool
    ) -> None:
        """Record which agent the interaction agent actually used for a turn."""

        with self._tx() as conn:
            conn.execute(
                "INSERT INTO retrieval_choices (retrieval_id, ts, agent_name, was_new, was_candidate)"
                " VALUES (?, ?, ?, ?, ?)",
                (retrieval_id, _now(), agent_name, int(was_new), int(was_candidate)),
            )
            conn.execute(
                "DELETE FROM retrieval_choices WHERE id <= ("
                " SELECT MAX(id) FROM retrieval_choices) - ?",
                (MAX_RETRIEVAL_LOG_ROWS,),
            )

    def recent_choices(self, limit: int = 3) -> List[Tuple[str, str]]:
        """Agents the interaction agent dispatched work to most recently, newest first.

        Returns up to `limit` distinct (agent_name, timestamp) pairs, one per agent.
        """

        with self._tx() as conn:
            rows = conn.execute(
                "SELECT agent_name, ts FROM retrieval_choices WHERE id IN ("
                " SELECT MAX(id) FROM retrieval_choices GROUP BY agent_name)"
                " ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [(row["agent_name"], row["ts"]) for row in rows]

    def retrieval_stats(self) -> Dict[str, Any]:
        """How often the agent the model chose was among the candidates it was shown."""

        with self._tx() as conn:
            turns = conn.execute("SELECT COUNT(*) FROM retrieval_log").fetchone()[0]
            reused = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(was_candidate), 0) FROM retrieval_choices WHERE was_new = 0"
            ).fetchone()
            created = conn.execute(
                "SELECT COUNT(*) FROM retrieval_choices WHERE was_new = 1"
            ).fetchone()[0]
        reused_total, reused_hits = int(reused[0]), int(reused[1])
        return {
            "turns_logged": int(turns),
            "reused_agent": reused_total,
            "reused_was_in_candidates": reused_hits,
            "hit_rate": round(reused_hits / reused_total, 3) if reused_total else None,
            "new_agents_created": int(created),
        }


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_EXEC_DIR = _DATA_DIR / "execution_agents"

_agent_roster = AgentRoster(_EXEC_DIR / "roster.db", legacy_json_path=_EXEC_DIR / "roster.json")


def get_agent_roster() -> AgentRoster:
    """Get the singleton roster instance."""
    return _agent_roster
