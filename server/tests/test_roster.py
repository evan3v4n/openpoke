"""Tests for the SQLite agent roster and the people/draft extraction that feeds it."""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from server.services.execution import roster as roster_module
from server.services.execution.people import (
    extract_from_tool_call,
    find_draft_id,
    people_from_text,
    tool_succeeded,
)
from server.services.execution.roster import MAX_PEOPLE_PER_AGENT, AgentRoster
from server.tests.helpers import make_roster, seed_alice


class RosterStorage(unittest.TestCase):
    def test_imports_legacy_json_once_and_survives_reopen(self) -> None:
        roster, directory = make_roster(["Email to Alice", {"name": "Q3 Budget", "summary": "numbers"}])
        self.assertEqual(sorted(roster.get_agents()), ["Email to Alice", "Q3 Budget"])
        self.assertEqual(roster.get_entry("Q3 Budget")["summary"], "numbers")

        roster.add_agent("Added later")
        reopened = AgentRoster(directory / "roster.db", legacy_json_path=directory / "roster.json")
        self.assertEqual(len(reopened.get_agents()), 3)  # not re-imported or reset

    def test_activity_people_threads_subjects(self) -> None:
        roster, _ = make_roster()
        seed_alice(roster)
        entry = roster.get_entry("Email to Alice")
        self.assertEqual(entry["people"]["alice@piedpiper.com"]["name"], "Alice Chen")
        self.assertEqual(entry["thread_ids"], ["t1"])
        self.assertEqual(entry["subjects"], ["Lunch tomorrow?"])

    def test_unknown_agent_activity_is_ignored(self) -> None:
        roster, _ = make_roster()
        findings = people_from_text("ping bob@x.io")
        roster.record_activity("ghost", findings)
        roster.open_draft("ghost", "d1", "bob@x.io", "hi")
        self.assertEqual(roster.get_agents(), [])
        self.assertEqual(roster.open_drafts(), {})

    def test_people_cap_keeps_first_of_batch_not_alphabetical(self) -> None:
        roster, _ = make_roster()
        roster.add_agent("Big")
        # Reverse-alphabetical: an alphabetical tie-break would keep u00..u49 instead.
        refs = [people_from_text(f"u{i:02d}@a.com")[0] for i in range(59, -1, -1)]
        roster.record_activity("Big", refs)

        kept = set(roster.get_entry("Big")["people"])
        self.assertEqual(len(kept), MAX_PEOPLE_PER_AGENT)
        self.assertEqual(kept, {f"u{i:02d}@a.com" for i in range(10, 60)})

    def test_people_cap_prefers_more_recently_seen(self) -> None:
        roster, _ = make_roster()
        roster.add_agent("Big")
        old_batch = [people_from_text(f"old{i:02d}@a.com")[0] for i in range(MAX_PEOPLE_PER_AGENT)]
        with mock.patch.object(roster_module, "_now", return_value="2026-11-01T00:00:00+00:00"):
            roster.record_activity("Big", old_batch)
        with mock.patch.object(roster_module, "_now", return_value="2026-11-02T00:00:00+00:00"):
            roster.record_activity("Big", people_from_text("new@a.com"))

        people = roster.get_entry("Big")["people"]
        self.assertEqual(len(people), MAX_PEOPLE_PER_AGENT)
        self.assertIn("new@a.com", people)  # newest survives; one of the oldest was dropped

    def test_upgrades_database_without_last_seen_column(self) -> None:
        directory = Path(tempfile.mkdtemp())
        db = directory / "roster.db"
        conn = sqlite3.connect(db)
        conn.executescript(
            "CREATE TABLE agents (name TEXT PRIMARY KEY, slug TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '',"
            " subjects TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL, last_active_at TEXT NOT NULL);"
            "CREATE TABLE agent_people (agent_name TEXT NOT NULL, email TEXT NOT NULL, name TEXT NOT NULL DEFAULT '',"
            " roles TEXT NOT NULL DEFAULT '[]', count INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (agent_name, email));"
            "INSERT INTO agents VALUES ('Old', 'old', 's', '[]', 'x', 'x');"
            "INSERT INTO agent_people VALUES ('Old', 'a@b.com', 'A', '[\"sender\"]', 2);"
        )
        conn.commit()
        conn.close()

        roster = AgentRoster(db)  # must add the column without losing rows
        self.assertEqual(roster.get_entry("Old")["people"]["a@b.com"]["count"], 2)
        roster.record_activity("Old", people_from_text("c@d.com"))
        self.assertIn("c@d.com", roster.get_entry("Old")["people"])

    def test_touch_does_not_index_excluded_owner_address(self) -> None:
        roster, _ = make_roster()
        roster.touch(
            "Stretch Reminder",
            "Check the inbox (Me@Me.com) for mail from bob@x.io",
            exclude=["me@me.com"],
        )
        self.assertEqual(set(roster.get_entry("Stretch Reminder")["people"]), {"bob@x.io"})

    def test_summary_is_plain_text_and_redacts_owner(self) -> None:
        roster, _ = make_roster()
        roster.add_agent("A")
        roster.set_summary(
            "A",
            "**Recent inbox summary**, grouped. I searched the inbox (Me@Me.com) and sent nothing.\n"
            "- see [the doc](http://x.example/y)\n## Heading",
            redact=["me@me.com"],
        )
        summary = roster.get_entry("A")["summary"]
        self.assertNotIn("**", summary)
        self.assertNotIn("##", summary)
        self.assertNotIn("http", summary)
        self.assertNotIn("me@me.com", summary.lower())
        self.assertNotIn("()", summary)
        self.assertIn("inbox (the user)", summary)  # no dangling "inbox ()" or "for ,"
        self.assertTrue(summary.startswith("Recent inbox summary"))
        self.assertIn("the doc", summary)

    def test_draft_lifecycle(self) -> None:
        roster, _ = make_roster()
        roster.add_agent("A")
        roster.open_draft("A", "d1", "x@y.com", "Hello")
        roster.open_draft("A", None, "z@y.com", "Unknown id")
        self.assertEqual(len(roster.open_drafts()["A"]), 2)

        roster.close_draft("d1")
        self.assertEqual([d["recipient"] for d in roster.open_drafts()["A"]], ["z@y.com"])

        # Sent with an id we never captured: falls back to the agent's id-less draft.
        roster.close_draft("unseen-id", agent_name="A")
        self.assertEqual(roster.open_drafts(), {})

    def test_clear(self) -> None:
        roster, directory = make_roster(["A"])
        roster.clear()
        self.assertEqual(roster.get_agents(), [])
        self.assertFalse((directory / "roster.json").exists())

class Extraction(unittest.TestCase):
    def test_owner_excluded_and_roles(self) -> None:
        records = [{"sender": "Alice <alice@x.com>", "recipient": "me@me.com, Dan <dan@z.com>", "thread_id": "t9", "subject": "Hi"}]
        found = extract_from_tool_call({"cc": ["bob@y.io"]}, records, exclude=["ME@me.com"])
        roles = {(p.email, p.role) for p in found.people}
        self.assertEqual(roles, {("bob@y.io", "cc"), ("alice@x.com", "sender"), ("dan@z.com", "recipient")})
        self.assertEqual(found.thread_ids, ["t9"])
        self.assertEqual(found.subjects, ["Hi"])

    def test_draft_id_and_success(self) -> None:
        self.assertEqual(find_draft_id({"data": {"id": "r-1", "message": {"id": "m"}}}), "r-1")
        self.assertEqual(find_draft_id({"data": {"draft_id": "r-2"}}), "r-2")
        self.assertIsNone(find_draft_id({"data": {"foo": 1}}))
        self.assertFalse(tool_succeeded({"successful": False}))
        self.assertFalse(tool_succeeded({"error": "boom"}))
        self.assertTrue(tool_succeeded({"successful": True}))

if __name__ == "__main__":
    unittest.main()
