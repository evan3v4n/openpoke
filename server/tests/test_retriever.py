"""Tests for ranking execution agents against a message."""

import unittest
from unittest import mock

from server.services.execution import roster as roster_module
from server.services.execution.people import PersonRef, people_from_text
from server.services.execution.retriever import AgentRetriever, reporting_agents
from server.tests.helpers import make_roster, seed_alice


class Retrieval(unittest.TestCase):
    def setUp(self) -> None:
        self.roster, _ = make_roster()
        self.triggers: dict[str, int] = {}
        self.retriever = AgentRetriever(self.roster, trigger_counts=lambda: self.triggers)
        seed_alice(self.roster)
        self.roster.touch("Vercel Job Offer", "Review the Vercel offer letter")
        self.roster.record_activity(
            "Vercel Job Offer", [], [], ["Offer letter: Software Engineer"]
        )
        self.roster.touch("Q3 Budget", "Analyse Q3 budget spreadsheet")

    def names(self, message: str, **kwargs) -> list[str]:
        return [c.name for c in self.retriever.candidates(message, **kwargs)]

    def test_person_name_and_email(self) -> None:
        self.assertEqual(self.names("did alice reply yet?", recent_pad=0)[0], "Email to Alice")
        self.assertEqual(self.names("forward to alice@piedpiper.com", recent_pad=0)[0], "Email to Alice")
        top = self.retriever.candidates("did alice reply yet?", recent_pad=0)[0]
        self.assertTrue(any("Alice Chen" in r for r in top.reasons), top.reasons)

    def test_topic_via_subject_and_name(self) -> None:
        self.assertEqual(self.names("any update on the software engineer offer?", recent_pad=0)[0], "Vercel Job Offer")
        self.assertEqual(self.names("q3 budget numbers", recent_pad=0)[0], "Q3 Budget")

    def test_contentless_followup_prefers_open_draft(self) -> None:
        self.roster.open_draft("Q3 Budget", "d1", "cfo@corp.com", "Q3 numbers")
        ranked = self.retriever.candidates("yes, send it")
        self.assertEqual(ranked[0].name, "Q3 Budget")
        self.assertTrue(any("unsent draft" in r for r in ranked[0].reasons))

        self.roster.close_draft("d1")
        self.assertFalse(
            any("unsent draft" in r for c in self.retriever.candidates("yes, send it") for r in c.reasons)
        )

    def test_active_trigger_is_an_open_loop(self) -> None:
        self.triggers = {"Vercel Job Offer": 2}
        top = self.retriever.candidates("ok", recent_pad=0)[0]
        self.assertEqual(top.name, "Vercel Job Offer")
        self.assertIn("2 active reminders", top.reasons)

    def test_forced_and_recent_pad(self) -> None:
        ranked = self.retriever.candidates("hello there", forced_names=["Q3 Budget"], recent_pad=1)
        self.assertIn("Q3 Budget", [c.name for c in ranked])
        self.assertEqual(ranked[0].reasons[0], "just reported back")
        self.assertEqual(self.retriever.search("hello there"), [])  # search has no padding

    def test_old_agent_found_among_many(self) -> None:
        roster, _ = make_roster()
        retriever = AgentRetriever(roster, trigger_counts=lambda: {})
        roster.touch("Email to Zed", "Email zed.walker@old.example about the lease")
        roster.record_activity("Email to Zed", people_from_text("zed.walker@old.example"), [], ["Lease renewal"])
        for i in range(60):
            roster.touch(f"Task {i}", f"Do thing number {i} for project {i % 7}")

        shortlist = retriever.candidates("what did zed say about the lease?")
        self.assertEqual(shortlist[0].name, "Email to Zed")
        self.assertLessEqual(len(shortlist), 8)  # bounded no matter how many agents exist
        self.assertEqual(retriever.search("lease renewal")[0].name, "Email to Zed")

    def test_duplicate_display_names_give_unique_reasons(self) -> None:
        roster, _ = make_roster()
        retriever = AgentRetriever(roster, trigger_counts=lambda: {})
        roster.add_agent("Inbox Summary")
        roster.record_activity(
            "Inbox Summary",
            people_from_text("noreply@google.com") + people_from_text("no-reply@accounts.google.com"),
        )
        # Both addresses share the display name "Google".
        for ref in list(roster.get_entry("Inbox Summary")["people"]):
            with roster._tx() as conn:
                conn.execute("UPDATE agent_people SET name = 'Google' WHERE email = ?", (ref,))
        top = retriever.candidates("anything from google?", recent_pad=0)[0]
        self.assertEqual(len(top.reasons), len(set(top.reasons)), top.reasons)

    def test_contentless_followup_follows_recently_used_agent(self) -> None:
        # Many agents created after the real one would otherwise crowd it out of the
        # recency padding (this is what happened in the 60-agent live test).
        for i in range(30):
            self.roster.touch(f"Task {i}", f"Do thing {i}")
        self.assertNotIn("Q3 Budget", self.names("check again"))

        self.roster.log_choice(None, "Q3 Budget", was_new=False, was_candidate=True)
        ranked = self.retriever.candidates("check again")
        self.assertEqual(ranked[0].name, "Q3 Budget")
        self.assertIn("used in the last few turns", ranked[0].reasons)

    def test_recent_use_is_ranked_by_how_recent(self) -> None:
        self.roster.log_choice(None, "Q3 Budget", was_new=False, was_candidate=True)
        self.roster.log_choice(None, "Vercel Job Offer", was_new=False, was_candidate=True)
        order = [c.name for c in self.retriever.candidates("check again", recent_pad=0)]
        self.assertEqual(order[:2], ["Vercel Job Offer", "Q3 Budget"])

    def test_old_or_deleted_choices_do_not_boost(self) -> None:
        with mock.patch.object(roster_module, "_now", return_value="2026-01-01T00:00:00+00:00"):
            self.roster.log_choice(None, "Q3 Budget", was_new=False, was_candidate=True)
        self.roster.log_choice(None, "Agent that no longer exists", was_new=False, was_candidate=False)
        ranked = self.retriever.candidates("check again", recent_pad=0)
        self.assertFalse(any("used in the last few turns" in c.reasons for c in ranked))

    def test_explicit_search_ignores_recent_use(self) -> None:
        self.roster.log_choice(None, "Q3 Budget", was_new=False, was_candidate=True)
        self.assertEqual(self.retriever.search("check again"), [])
        self.assertEqual(self.retriever.search("vercel offer")[0].name, "Vercel Job Offer")

    def test_entity_match_still_beats_recent_use(self) -> None:
        self.roster.log_choice(None, "Q3 Budget", was_new=False, was_candidate=True)
        self.assertEqual(self.names("did alice reply yet?", recent_pad=0)[0], "Email to Alice")

    def test_reporting_agents(self) -> None:
        payload = "[SUCCESS] Email to Alice: Draft ready.\n[FAILED] Q3 Budget: timed out"
        self.assertEqual(reporting_agents(payload), ["Email to Alice", "Q3 Budget"])

    def test_logging_and_stats(self) -> None:
        rid = self.roster.log_retrieval("user", "did alice reply?", [{"name": "Email to Alice"}])
        self.roster.log_choice(rid, "Email to Alice", was_new=False, was_candidate=True)
        self.roster.log_choice(rid, "Email to Bob", was_new=True, was_candidate=False)
        self.roster.log_choice(rid, "Q3 Budget", was_new=False, was_candidate=False)
        stats = self.roster.retrieval_stats()
        self.assertEqual(stats["hit_rate"], 0.5)
        self.assertEqual(stats["new_agents_created"], 1)
        self.assertEqual(stats["turns_logged"], 1)

class JudyRosterCase(unittest.TestCase):
    """Ten Judys plus three others: the fixture for the ambiguity tests."""

    LAST_NAMES = ["Brown", "Chen", "Garcia", "Kim", "Lopez", "Moore", "Nguyen", "Patel", "Smith", "Wilson"]

    def setUp(self) -> None:
        self.roster, _ = make_roster()
        self.retriever = AgentRetriever(self.roster, trigger_counts=lambda: {})
        for last in self.LAST_NAMES:
            name = f"Email to Judy {last}"
            email = f"judy.{last.lower()}@example.com"
            self.roster.touch(name, f"Email {email}")
            self.roster.record_activity(name, [PersonRef(email=email, name=f"Judy {last}", role="recipient")])
        for first in ("Ivan", "Carol", "Dan"):
            self.roster.touch(f"Email to {first} Kim", f"Email {first.lower()}.kim@example.com")


class Ambiguity(JudyRosterCase):
    """Ten Judys: the model must be told they tie, and must see all of them."""

    def test_first_name_shared_by_ten_agents_is_reported_as_a_tie(self) -> None:
        shortlist = self.retriever.retrieve("Did Judy reply?")
        self.assertEqual(shortlist.tied_names, sorted(f"Email to Judy {l}" for l in self.LAST_NAMES))
        self.assertGreaterEqual(shortlist.matching_total, 10)
        self.assertLess(len([c for c in shortlist.candidates if "Judy" in c.name]), 10)  # list is capped

    def test_recent_use_or_open_draft_on_a_few_does_not_hide_the_ambiguity(self) -> None:
        # What the live "Did Judy reply?" turn looked like after six Judys were dispatched to.
        for last in ("Smith", "Patel", "Garcia"):
            self.roster.log_choice(None, f"Email to Judy {last}", was_new=False, was_candidate=True)
        self.roster.open_draft("Email to Judy Brown", "d1", "judy.brown@example.com", "Hi")
        shortlist = self.retriever.retrieve("Did Judy reply?")
        self.assertEqual(len(shortlist.tied_names), 10)
        # Context still orders the tied agents: the open draft and the recently used ones lead.
        top_four = [c.name for c in shortlist.candidates[:4]]
        self.assertEqual(top_four[0], "Email to Judy Brown")
        for name in ("Email to Judy Garcia", "Email to Judy Patel", "Email to Judy Smith"):
            self.assertIn(name, top_four)

    def test_naming_one_person_beats_recent_use_of_others(self) -> None:
        # Live regression: after dispatching to Brown, Smith, Chen (Chen last), "check judy brown"
        # ranked Chen above Brown because recent use outweighed the exact-name match.
        # Other Browns make "brown" a weaker discriminator, as on the live roster.
        for i, first in enumerate(("Ivan", "Carol", "Dan", "Eli", "Fay", "Gus", "Hana", "Ian", "Kyle", "Lena")):
            name, email = f"Email to {first} Brown", f"{first.lower()}.brown{i}@example.com"
            self.roster.touch(name, f"Email {email}")
            self.roster.record_activity(name, [PersonRef(email=email, name=f"{first} Brown", role="recipient")])
        for last in ("Brown", "Smith", "Chen"):
            self.roster.log_choice(None, f"Email to Judy {last}", was_new=False, was_candidate=True)
        ranked = self.retriever.candidates("check judy brown", recent_pad=0)
        self.assertEqual(ranked[0].name, "Email to Judy Brown")
        # Without the context scaling the margin over Chen is ~0.01: a coin flip on small changes.
        self.assertGreater(ranked[0].score - ranked[1].score, 0.1)

    def test_recent_use_still_decides_when_message_names_nobody(self) -> None:
        self.roster.log_choice(None, "Email to Judy Chen", was_new=False, was_candidate=True)
        self.assertEqual(self.retriever.candidates("check again", recent_pad=0)[0].name, "Email to Judy Chen")

    def test_several_threads_with_one_person_are_not_ambiguous(self) -> None:
        for name in ("Lunch with Judy Kim", "Lease thread with Judy Kim"):
            self.roster.touch(name, "x")
            self.roster.record_activity(
                name, [PersonRef(email="judy.kim@example.com", name="Judy Kim", role="recipient")]
            )
        # Both new agents and "Email to Judy Kim" know the same person; asking "Judy Kim" is not a who-question.
        self.assertEqual(self.retriever.retrieve("what did Judy Kim say?").tied_names, [])

    def test_specific_request_is_not_ambiguous(self) -> None:
        shortlist = self.retriever.retrieve("Did Judy Patel reply?")
        self.assertEqual(shortlist.tied_names, [])
        self.assertEqual(shortlist.candidates[0].name, "Email to Judy Patel")

    def test_email_address_is_not_ambiguous(self) -> None:
        self.assertEqual(self.retriever.retrieve("forward to judy.kim@example.com").tied_names, [])

    def test_contentless_message_is_not_ambiguous(self) -> None:
        self.assertEqual(self.retriever.retrieve("check again").tied_names, [])


if __name__ == "__main__":
    unittest.main()
