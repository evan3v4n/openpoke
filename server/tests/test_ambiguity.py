"""Asking instead of guessing: which draft, which date, which mailbox, which agent.

Includes a replay of a real conversation where a bare "send it" sent the wrong draft.
"""

from __future__ import annotations

import types
import unittest
from datetime import date
from unittest import mock

from server.services.execution.ambiguity import (
    SEND_NOT_CONFIRMED_ERROR,
    assess_account,
    assess_dates,
    assess_user_message,
    check_send,
)
from server.services.execution.retriever import AgentRetriever
from server.tests.helpers import make_roster

FRIDAY = date(2026, 11, 13)
TORONTO = "evan.scott@torontomu.ca"
PENGUIN = "evanscottpenguin@gmail.com"


def reply(text: str) -> str:
    """A transcript where the user spoke, then the assistant replied with `text`."""

    return f'<user_message timestamp="t">...</user_message>\n<poke_reply timestamp="t">{text}</poke_reply>'


class TwoDrafts(unittest.TestCase):
    """Two open drafts held by two agents, like the conversation that sent the wrong one."""

    def setUp(self) -> None:
        self.roster, _ = make_roster()
        self.roster.touch("Checking In", f"Draft to {TORONTO}")
        self.roster.open_draft("Checking In", "dA", TORONTO, "Checking in")
        self.roster.touch("Lunch Invite", f"Draft to {PENGUIN}")
        self.roster.open_draft("Lunch Invite", "dB", PENGUIN, "Lunch this Friday?")

    def dispatched(self, *agents: str) -> None:
        """The interaction agent sent work to `agents` in one turn."""

        retrieval_id = self.roster.log_retrieval("user", "msg", [])
        for agent in agents:
            self.roster.log_choice(retrieval_id, agent, was_new=False, was_candidate=True)

    def assess(self, message: str, transcript: str = "", **kwargs):
        result = assess_user_message(
            message, self.roster, today=FRIDAY, gmail_connected=True,
            account_email="me@me.com", transcript=transcript, **kwargs,
        )
        self.roster.set_send_authorizations(result.grants)
        return result

    def kinds(self, result) -> list:
        return [a.kind for a in result.ambiguities]

    # ---------------------------------------------------------------- the replay

    def test_replay_bare_send_after_the_discussed_draft_went_out_asks(self) -> None:
        self.dispatched("Checking In")
        self.dispatched("Lunch Invite")
        shown_lunch = reply(f'Here\'s the lunch draft to {PENGUIN}, subject "Lunch this Friday?"')

        # "yeah that looks good send it": the lunch draft was just shown, so it may go.
        result = self.assess("yeah that looks good send it", shown_lunch)
        self.assertEqual(self.kinds(result), [])
        self.assertTrue(check_send(self.roster, "Lunch Invite", "dB"))
        self.assertFalse(check_send(self.roster, "Checking In", "dA"))

        self.dispatched("Lunch Invite")
        self.roster.close_draft("dB", "Lunch Invite")

        # "send it" again: the draft just discussed is gone. The other one is NOT implied.
        result = self.assess("send it", reply(f"Lunch note went to {PENGUIN} for Friday, Nov 20."))
        self.assertEqual(self.kinds(result), ["draft"])
        detail = result.ambiguities[0].detail
        self.assertIn("already sent", detail)
        self.assertIn(TORONTO, detail)
        self.assertFalse(check_send(self.roster, "Checking In", "dA"))

        # The assistant asks about the other draft; "yes" answers that question.
        result = self.assess("yes", reply(f'The lunch note already went out. Send the "Checking in" draft to {TORONTO}?'))
        self.assertEqual(self.kinds(result), [])
        self.assertTrue(check_send(self.roster, "Checking In", "dA"))

    # ---------------------------------------------------------------- other draft cases

    def test_send_it_right_after_one_draft_needs_no_question(self) -> None:
        self.dispatched("Lunch Invite")
        result = self.assess("send it")
        self.assertEqual(self.kinds(result), [])
        self.assertTrue(check_send(self.roster, "Lunch Invite", "dB"))

    def test_send_it_when_two_drafts_were_just_shown_asks(self) -> None:
        self.dispatched("Checking In", "Lunch Invite")
        result = self.assess("send it")
        self.assertEqual(self.kinds(result), ["draft"])
        self.assertEqual(len(result.ambiguities[0].options), 2)
        self.assertFalse(check_send(self.roster, "Lunch Invite", "dB"))

    def test_send_both_allows_both(self) -> None:
        self.dispatched("Checking In", "Lunch Invite")
        result = self.assess("send both")
        self.assertEqual(self.kinds(result), [])
        self.assertTrue(check_send(self.roster, "Lunch Invite", "dB"))
        self.assertTrue(check_send(self.roster, "Checking In", "dA"))

    def test_naming_a_draft_by_subject_or_recipient(self) -> None:
        self.dispatched("Checking In", "Lunch Invite")
        self.assess("send the lunch one")
        self.assertTrue(check_send(self.roster, "Lunch Invite", "dB"))
        self.assertFalse(check_send(self.roster, "Checking In", "dA"))

        self.assess("send the one to torontomu")
        self.assertTrue(check_send(self.roster, "Checking In", "dA"))
        self.assertFalse(check_send(self.roster, "Lunch Invite", "dB"))

    def test_a_name_matching_both_drafts_asks(self) -> None:
        result = self.assess("send it to evan")
        self.assertEqual(self.kinds(result), ["draft"])
        self.assertFalse(check_send(self.roster, "Checking In", "dA"))

    def test_answering_which_one_with_just_a_reference(self) -> None:
        question = reply(f'Which draft: "Checking in" to {TORONTO} or "Lunch this Friday?" to {PENGUIN}?')
        result = self.assess("the lunch one", question)
        self.assertEqual(self.kinds(result), [])
        self.assertTrue(check_send(self.roster, "Lunch Invite", "dB"))

    def test_editing_a_draft_does_not_allow_sending_it(self) -> None:
        self.dispatched("Lunch Invite")
        self.assess("change the lunch one to 2pm", reply(f"Here's the lunch draft to {PENGUIN}"))
        self.assertFalse(check_send(self.roster, "Lunch Invite", "dB"))

    def test_new_request_never_covers_a_fresh_draft(self) -> None:
        result = self.assess("send bob@x.com a note saying hi")
        self.assertEqual(self.kinds(result), [])
        self.roster.touch("Note to Bob", "Draft to bob@x.com")
        self.roster.open_draft("Note to Bob", "dC", "bob@x.com", "Hi")
        self.assertFalse(check_send(self.roster, "Note to Bob", "dC"))  # drafted, not yet confirmed

    def test_unrelated_message_allows_no_sends(self) -> None:
        self.dispatched("Lunch Invite")
        result = self.assess("what's the weather in toronto")
        self.assertEqual(self.kinds(result), [])
        self.assertFalse(check_send(self.roster, "Lunch Invite", "dB"))

    def test_untracked_draft_follows_the_agent_just_used(self) -> None:
        self.roster.close_draft("dA", "Checking In")
        self.roster.close_draft("dB", "Lunch Invite")
        self.roster.touch("Old Draft", "send my old draft")
        self.dispatched("Old Draft")
        self.assess("yes send it")
        self.assertTrue(check_send(self.roster, "Old Draft", "gmail-draft-from-before"))
        self.assertFalse(check_send(self.roster, "Lunch Invite", "gmail-draft-from-before"))

    def test_id_less_draft_is_matched_to_its_agent(self) -> None:
        self.roster.touch("No Id", "draft")
        self.roster.open_draft("No Id", None, "x@y.com", "Hello there")
        self.dispatched("No Id")
        self.assess("send it")
        self.assertTrue(check_send(self.roster, "No Id", "real-id-we-could-not-parse"))


class Dates(unittest.TestCase):
    def test_this_friday_said_on_a_friday(self) -> None:
        found = assess_dates("ask him to lunch this friday at 1:00", FRIDAY)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].options, ["today (Fri Nov 13)", "Fri Nov 20"])

    def test_next_weekday(self) -> None:
        found = assess_dates("dinner next tuesday?", FRIDAY)
        self.assertEqual(found[0].options, ["Tue Nov 17", "Tue Nov 24"])

    def test_clear_dates_are_left_alone(self) -> None:
        for message in ("lunch on monday", "remind me every friday", "call me tomorrow", "fridays are busy"):
            self.assertEqual(assess_dates(message, FRIDAY), [], message)


class Account(unittest.TestCase):
    def test_unknown_mailbox_is_flagged_only_when_sending(self) -> None:
        self.assertEqual([a.kind for a in assess_account("send it", True, None)], ["account"])
        self.assertEqual(assess_account("send it", True, "me@me.com"), [])
        self.assertEqual(assess_account("what's on my calendar", True, None), [])
        self.assertEqual(assess_account("send it", False, None), [])  # <connections> covers this


class TopicTies(unittest.TestCase):
    def setUp(self) -> None:
        self.roster, _ = make_roster()
        self.retriever = AgentRetriever(self.roster, trigger_counts=lambda: {})
        for name, summary in [
            ("Q3 Budget Analysis", "Analyzed the Q3 budget spreadsheet: spend up 12%."),
            ("Q4 Budget Analysis", "Analyzed the Q4 budget spreadsheet: spend flat."),
            ("Tokyo Restaurant Search", "Found sushi restaurants in Shibuya for Friday."),
        ]:
            self.roster.touch(name, summary)
            self.roster.set_summary(name, summary)

    def test_shared_topic_is_a_tie(self) -> None:
        shortlist = self.retriever.retrieve("how's the budget looking?")
        self.assertEqual(shortlist.tie_kind, "topic")
        self.assertEqual(shortlist.tied_names, ["Q3 Budget Analysis", "Q4 Budget Analysis"])

    def test_quarter_tells_them_apart(self) -> None:
        shortlist = self.retriever.retrieve("what did the q3 numbers look like?")
        self.assertEqual(shortlist.candidates[0].name, "Q3 Budget Analysis")
        self.assertEqual(shortlist.tied_names, [])

    def test_single_match_is_not_a_tie(self) -> None:
        shortlist = self.retriever.retrieve("where should we eat in tokyo")
        self.assertEqual(shortlist.tied_names, [])


class SendGuard(unittest.IsolatedAsyncioTestCase):
    """The execution runtime refuses sends the user did not clearly ask for, whatever the model does."""

    def setUp(self) -> None:
        from server.agents.execution_agent import agent as agent_module
        from server.agents.execution_agent import runtime as runtime_module
        from server.services.execution.log_store import ExecutionAgentLogStore

        self.runtime_module = runtime_module
        self.roster, directory = make_roster()
        self.roster.touch("Checking In", f"Draft to {TORONTO}")
        self.roster.open_draft("Checking In", "dA", TORONTO, "Checking in")
        self.send = mock.Mock(return_value={"successful": True, "data": {"id": "m1"}})
        self.llm_calls = 0
        self.sender = "me@me.com"

        settings = types.SimpleNamespace(openrouter_api_key="k", execution_agent_model="m")
        store = ExecutionAgentLogStore(directory / "logs")
        for target, name, value in [
            (runtime_module, "get_settings", mock.Mock(return_value=settings)),
            (runtime_module, "get_agent_roster", mock.Mock(return_value=self.roster)),
            (runtime_module, "get_tool_schemas", mock.Mock(return_value=[])),
            (runtime_module, "get_tool_registry", mock.Mock(return_value={"gmail_execute_draft": self.send})),
            (runtime_module, "get_active_gmail_email", mock.Mock(return_value=None)),
            (runtime_module, "is_gmail_connected", mock.Mock(return_value=True)),
            (runtime_module, "resolve_active_gmail_email", lambda: self.sender),
            (agent_module, "get_execution_agent_logs", mock.Mock(return_value=store)),
        ]:
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        replies = [
            {"content": "", "tool_calls": [{"id": "c1", "function": {"name": "gmail_execute_draft", "arguments": '{"draft_id": "dA"}'}}]},
            {"content": "Sent."},
        ]

        async def fake_completion(**_kwargs):
            self.llm_calls += 1
            return {"choices": [{"message": replies.pop(0)}]}

        patcher = mock.patch.object(runtime_module, "request_chat_completion", fake_completion)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def run_agent(self):
        return await self.runtime_module.ExecutionAgentRuntime("Checking In").execute("send the draft")

    async def test_unconfirmed_send_is_refused_and_the_run_stops(self) -> None:
        self.roster.set_send_authorizations([])
        result = await self.run_agent()
        self.send.assert_not_called()
        self.assertFalse(result.success)
        self.assertEqual(result.error, SEND_NOT_CONFIRMED_ERROR)
        self.assertEqual(self.llm_calls, 1)
        self.assertEqual(len(self.roster.open_draft_rows()), 1)  # still open

    async def test_confirmed_send_goes_out_and_reports_the_sender(self) -> None:
        row = self.roster.open_draft_rows()[0]
        self.roster.set_send_authorizations([("Checking In", row["id"])])
        result = await self.run_agent()
        self.send.assert_called_once_with(draft_id="dA")
        self.assertTrue(result.success)
        self.assertEqual(self.roster.open_draft_rows(), [])  # closed after sending
        self.assertEqual(self.send.return_value["sent_from"], "me@me.com")  # in the result the model reads

    async def test_unknown_sender_is_refused(self) -> None:
        row = self.roster.open_draft_rows()[0]
        self.roster.set_send_authorizations([("Checking In", row["id"])])
        self.sender = None
        result = await self.run_agent()
        self.send.assert_not_called()
        self.assertIn("could not be confirmed", result.error)


if __name__ == "__main__":
    unittest.main()
