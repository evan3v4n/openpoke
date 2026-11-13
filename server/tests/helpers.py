"""Shared fixtures for the roster, retriever and interaction-agent tests."""

import json
import tempfile
from pathlib import Path

from server.services.execution.people import extract_from_tool_call
from server.services.execution.roster import AgentRoster


def make_roster(legacy: object = None) -> tuple[AgentRoster, Path]:
    directory = Path(tempfile.mkdtemp())
    legacy_path = directory / "roster.json"
    if legacy is not None:
        legacy_path.write_text(json.dumps(legacy))
    return AgentRoster(directory / "roster.db", legacy_json_path=legacy_path), directory


def seed_alice(roster: AgentRoster) -> None:
    roster.touch("Email to Alice", "Email alice@piedpiper.com about lunch tomorrow")
    findings = extract_from_tool_call(
        {"recipient_email": "Alice Chen <alice@piedpiper.com>", "subject": "Lunch tomorrow?", "thread_id": "t1"},
        {},
    )
    roster.record_activity("Email to Alice", findings.people, findings.thread_ids, findings.subjects)
    roster.set_summary("Email to Alice", "Invited Alice to lunch; waiting for her reply.")
