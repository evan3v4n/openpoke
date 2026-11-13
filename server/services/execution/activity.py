"""In-memory feed of what running execution agents are doing, for the live UI panel."""

from __future__ import annotations

import threading
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

MAX_EVENTS_PER_RUN = 60
MAX_TEXT_CHARS = 400
FINISHED_VISIBLE_SECONDS = 20.0  # keep finished runs briefly so the UI can show the outcome
MAX_FINISHED_RUNS = 20

# Set by the batch manager around a run; read by the runtime so it needs no extra plumbing.
current_run: ContextVar[Optional[str]] = ContextVar("current_run", default=None)


class ActivityTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: Dict[str, Dict[str, Any]] = {}

    def start(self, agent_name: str, instructions: str) -> str:
        run_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._runs[run_id] = {
                "run_id": run_id,
                "agent_name": agent_name,
                "instructions": instructions[:MAX_TEXT_CHARS],
                "status": "running",
                "started_at": time.time(),
                "finished_at": None,
                "events": [],
                "next_seq": 1,
            }
        return run_id

    def add(self, kind: str, text: str, run_id: Optional[str] = None) -> None:
        """Append an event (kind: thinking | tool_call | tool_result | error) to the current run."""
        run_id = run_id or current_run.get()
        if not run_id:
            return
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run["events"].append(
                {
                    "seq": run["next_seq"],
                    "ts": time.time(),
                    "kind": kind,
                    "text": str(text)[:MAX_TEXT_CHARS],
                }
            )
            run["next_seq"] += 1
            if len(run["events"]) > MAX_EVENTS_PER_RUN:
                del run["events"][: len(run["events"]) - MAX_EVENTS_PER_RUN]

    def finish(self, run_id: str, success: bool, summary: str = "") -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run["status"] = "done" if success else "failed"
            run["finished_at"] = time.time()
            run["summary"] = summary[:MAX_TEXT_CHARS]
            finished = sorted(
                (r for r in self._runs.values() if r["finished_at"]),
                key=lambda r: r["finished_at"],
            )
            for old in finished[: max(0, len(finished) - MAX_FINISHED_RUNS)]:
                self._runs.pop(old["run_id"], None)

    def snapshot(self) -> List[Dict[str, Any]]:
        """Running runs plus recently finished ones, oldest first."""
        now = time.time()
        with self._lock:
            runs = [
                {**run, "events": list(run["events"]), "elapsed": (run["finished_at"] or now) - run["started_at"]}
                for run in self._runs.values()
                if run["finished_at"] is None or now - run["finished_at"] <= FINISHED_VISIBLE_SECONDS
            ]
        runs.sort(key=lambda r: r["started_at"])
        return runs

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()


_tracker = ActivityTracker()


def get_activity_tracker() -> ActivityTracker:
    return _tracker
