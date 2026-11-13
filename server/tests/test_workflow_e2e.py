"""End-to-end interaction-agent workflow with a scripted LLM and a temp roster.

Real: prompt building, retrieval, the tool loop, find_agents, send_message_to_agent, choice logging,
HTTP routes. Fake: the LLM (a stand-in that picks from what the prompt shows), the execution agent,
conversation logs, Gmail. Nothing touches server/data.

Run from the repo root:  .venv/bin/python -m unittest discover -s server/tests -t .
"""

from __future__ import annotations

import json
import re
import types
import unittest
from contextlib import ExitStack
from unittest import mock

from server.services.execution.people import PersonRef
from server.services.execution.retriever import AgentRetriever
from server.tests import test_retriever as base


def tool_call(name: str, **arguments) -> dict:
    return {"id": f"c-{name}", "function": {"name": name, "arguments": json.dumps(arguments)}}


class Workflow(unittest.IsolatedAsyncioTestCase):
    gmail_connected = True

    def setUp(self) -> None:
        base.RealisticRosterAccuracy.setUp(self)  # 51-agent roster, Q3 Budget has an unsent draft
        self.prompts: list[str] = []
        self.tool_messages: list[str] = []
        self.script: list = []  # items: dict message, or callable(prompt) -> dict message
        self.dispatched: list[tuple[str, str]] = []

        from server.agents.interaction_agent import agent as agent_mod
        from server.agents.interaction_agent import runtime as runtime_mod
        from server.agents.interaction_agent import tools as tools_mod

        self.runtime_mod = runtime_mod
        settings = types.SimpleNamespace(
            openrouter_api_key="k", interaction_agent_model="m", summarization_enabled=False
        )
        log = mock.Mock()
        log.load_transcript.return_value = ""
        self.log = log

        async def fake_llm(**kwargs):
            messages = kwargs["messages"]
            self.prompts.append(messages[0]["content"])
            for m in messages:
                if m["role"] == "tool":
                    self.tool_messages.append(m["content"])
            step = self.script.pop(0)
            reply = step(self.prompts[-1]) if callable(step) else step
            return {"choices": [{"message": reply}]}

        async def fake_execute(name, instructions):
            self.dispatched.append((name, instructions))
            return types.SimpleNamespace(success=True)

        stack = ExitStack()
        self.addCleanup(stack.close)
        for target, attr, value in [
            (agent_mod, "get_agent_roster", lambda: self.roster),
            (agent_mod, "get_agent_retriever", lambda: self.retriever),
            (tools_mod, "get_agent_roster", lambda: self.roster),
            (tools_mod, "get_agent_retriever", lambda: self.retriever),
            (tools_mod, "get_execution_agent_logs", lambda: mock.Mock()),
            (runtime_mod, "get_settings", lambda: settings),
            (runtime_mod, "get_conversation_log", lambda: log),
            (runtime_mod, "get_working_memory_log", lambda: log),
            (runtime_mod, "request_chat_completion", fake_llm),
        ]:
            stack.enter_context(mock.patch.object(target, attr, value))
        stack.enter_context(mock.patch.object(tools_mod._EXECUTION_BATCH_MANAGER, "execute_agent", fake_execute))
        stack.enter_context(mock.patch("server.services.gmail.get_active_gmail_email", return_value="me@me.com"))
        stack.enter_context(
            mock.patch("server.services.gmail.is_gmail_connected", return_value=self.gmail_connected)
        )

    async def run_user(self, message: str):
        import asyncio

        result = await self.runtime_mod.InteractionAgentRuntime().execute(message)
        await asyncio.sleep(0)  # let the background dispatch task run
        self.assertTrue(result.success, result.error)
        return result

    @staticmethod
    def shown_names(prompt: str) -> list[str]:
        return re.findall(r'<agent name="([^"]+)"', prompt)

    @staticmethod
    def pick_top(task: str):
        """A model stand-in: reuse the first listed agent, else create a new one."""

        def step(prompt: str) -> dict:
            names = re.findall(r'<agent name="([^"]+)"', prompt)
            return {"content": "", "tool_calls": [tool_call("send_message_to_agent", agent_name=names[0] if names else "New Agent", instructions=task)]}

        return step

    # ---------------------------------------------------------------- user turns

    async def test_reuse_existing_agent_by_topic(self) -> None:
        before = self.roster.count()
        self.script = [self.pick_top("check the budget"), {"content": "On it."}]
        await self.run_user("update on the q3 budget")

        self.assertEqual(self.dispatched[0][0], "Q3 Budget")
        self.assertEqual(self.roster.count(), before)  # no new agent
        stats = self.roster.retrieval_stats()
        self.assertEqual((stats["hit_rate"], stats["new_agents_created"]), (1.0, 0))

    async def test_prompt_lists_a_bounded_shortlist_not_the_roster(self) -> None:
        self.script = [{"content": "ok"}]
        await self.run_user("update on the q3 budget")
        prompt = self.prompts[0]
        self.assertLessEqual(len(self.shown_names(prompt)), 8)
        self.assertNotIn("Misc Task 24", prompt)
        self.assertIn(f'<other_agents count="{self.roster.count() - len(self.shown_names(prompt))}"', prompt)
        self.assertIn('why="', prompt)

    async def test_prompt_size_does_not_grow_with_roster(self) -> None:
        self.script = [{"content": "ok"}, {"content": "ok"}]
        await self.run_user("update on the q3 budget")
        small = len(self.prompts[0])
        for i in range(500):
            self.roster.touch(f"Bulk {i}", f"bulk job {i} with a fairly long description of what it did")
        await self.run_user("update on the q3 budget")
        self.assertLess(len(self.prompts[1]), small * 1.15)

    async def test_ambiguous_person_prompts_a_question_instead_of_fanout(self) -> None:
        self.script = [{"content": "", "tool_calls": []}]
        await self.run_user("did judy reply?")
        prompt = self.prompts[0]
        self.assertIn("<ambiguity", prompt)
        for last in ("Brown", "Garcia", "Patel", "Nguyen"):
            self.assertIn(f"Email to Judy {last}", prompt)
        self.assertEqual(self.dispatched, [])

    async def test_find_agents_then_reuse_counts_as_a_candidate_hit(self) -> None:
        # Make the target fall outside the shortlist: lots of fresher agents mention the same words.
        for i in range(12):
            self.roster.touch(f"Sam Fan {i}", "sam sam sam")
        self.script = [
            {"content": "", "tool_calls": [tool_call("find_agents", query="sam.lee@vercel.com")]},
            {"content": "", "tool_calls": [tool_call("send_message_to_agent", agent_name="Vercel Job Offer", instructions="check reply")]},
            {"content": "Checking."},
        ]
        await self.run_user("anything from sam.lee@vercel.com?")
        payload = json.loads(self.tool_messages[0])["result"]
        self.assertEqual(payload["agents"][0]["name"], "Vercel Job Offer")
        self.assertEqual(self.dispatched[0][0], "Vercel Job Offer")
        stats = self.roster.retrieval_stats()
        self.assertEqual((stats["hit_rate"], stats["new_agents_created"]), (1.0, 0))

    async def test_new_topic_creates_a_new_agent_and_is_logged_as_new(self) -> None:
        before = self.roster.count()
        self.script = [
            {"content": "", "tool_calls": [tool_call("find_agents", query="haircut Supercuts")]},
            {"content": "", "tool_calls": [tool_call("send_message_to_agent", agent_name="Haircut Booking", instructions="book a haircut at Supercuts")]},
            {"content": "Looking into it."},
        ]
        await self.run_user("book me a haircut at Supercuts")
        # The search must say "nothing relevant" even though Q3 Budget has a draft and Stretch a reminder.
        self.assertEqual(json.loads(self.tool_messages[0])["result"]["agents"], [])
        self.assertEqual(self.roster.count(), before + 1)
        self.assertEqual(self.roster.retrieval_stats()["new_agents_created"], 1)
        # And the new agent is findable on the next turn.
        self.assertEqual(self.retriever.retrieve("what happened with the haircut").candidates[0].name, "Haircut Booking")

    async def test_followup_sent_to_agent_used_last_turn(self) -> None:
        self.script = [self.pick_top("look up judy patel"), {"content": "ok"}, {"content": "ok"}]
        await self.run_user("follow up with judy patel")
        self.assertEqual(self.dispatched[0][0], "Email to Judy Patel")
        await self.run_user("any news?")  # contentless: the agent just used should be near the top
        shown = self.shown_names(self.prompts[-1])
        self.assertIn("Email to Judy Patel", shown[:3])

    async def test_owner_address_never_enters_the_people_index(self) -> None:
        self.script = [
            self.pick_top("check inbox of me@me.com for mail from zed@z.io"),
            {"content": "ok"},
        ]
        await self.run_user("book a haircut")
        name = self.dispatched[0][0]
        people = set(self.roster.get_entry(name)["people"])
        self.assertNotIn("me@me.com", people)

    async def test_replay_second_send_it_asks_instead_of_sending_the_other_draft(self) -> None:
        """A real conversation: two drafts, the user confirms one, then says "send it" again."""

        from server.services.execution.ambiguity import check_send

        toronto, penguin = "evan.scott@torontomu.ca", "evanscottpenguin@gmail.com"

        def dispatch(agent: str, task: str) -> list:
            return [{"content": "", "tool_calls": [tool_call("send_message_to_agent", agent_name=agent, instructions=task)]}, {"content": "On it."}]

        def assistant_said(text: str) -> None:
            self.log.load_transcript.return_value = (
                f'<user_message timestamp="t">earlier</user_message>\n<poke_reply timestamp="t">{text}</poke_reply>'
            )

        self.script = dispatch("Checking In", f"draft to {toronto} asking what's up")
        await self.run_user(f"open a draft email to {toronto} asking whats up")
        self.roster.open_draft("Checking In", "dA", toronto, "Checking in")

        self.script = dispatch("Lunch Invite", f"draft to {penguin}: lunch Friday at 1:00")
        await self.run_user(f"open another draft to {penguin} asking him to lunch this friday at 1:00")
        self.roster.open_draft("Lunch Invite", "dB", penguin, "Lunch this Friday?")

        assistant_said(f'Here\'s the lunch draft to {penguin}, subject "Lunch this Friday?". Send it as-is?')
        self.script = dispatch("Lunch Invite", "send the lunch draft")
        start = len(self.prompts)
        await self.run_user("yeah that looks good send it")
        self.assertNotIn('<ambiguity kind="draft"', self.prompts[start])
        self.assertTrue(check_send(self.roster, "Lunch Invite", "dB"))
        self.roster.close_draft("dB", "Lunch Invite")

        assistant_said(f"Lunch note went to {penguin} for Friday, Nov 20 at 1:00 PM.")
        self.script = [{"content": "The lunch note already went out. Send the other draft?"}]
        sent_before = len(self.dispatched)
        start = len(self.prompts)
        await self.run_user("send it")
        prompt = self.prompts[start]
        self.assertIn('<ambiguity kind="draft">', prompt)
        self.assertIn("already sent", prompt)
        self.assertIn(toronto, prompt)
        self.assertEqual(len(self.dispatched), sent_before)
        # Even a model that ignored the note could not send it: nothing allows that draft.
        self.assertFalse(check_send(self.roster, "Checking In", "dA"))

    async def test_topic_tie_prompts_a_question(self) -> None:
        self.roster.touch("Q4 Budget", "Collected Q4 budget numbers from finance.")
        self.roster.set_summary("Q4 Budget", "Collected Q4 budget numbers from finance.")
        self.script = [{"content": "Q3 or Q4?"}]
        await self.run_user("how's the budget looking")
        self.assertIn('<ambiguity kind="topic"', self.prompts[0])

    # ---------------------------------------------------------------- agent report turns

    async def test_agent_report_turn_includes_the_reporter(self) -> None:
        import asyncio

        self.script = [{"content": "ok"}]
        result = await self.runtime_mod.InteractionAgentRuntime().handle_agent_message(
            "[SUCCESS] Flight to Tokyo: found three fares under $900"
        )
        self.assertTrue(result.success)
        prompt = self.prompts[0]
        self.assertIn('name="Flight to Tokyo"', prompt)
        self.assertIn("just reported back", prompt)
        self.assertNotIn("<ambiguity", prompt)


class WorkflowGmailDown(Workflow):
    gmail_connected = False

    async def test_connections_hint_reaches_the_model(self) -> None:
        self.script = [{"content": "ok"}]
        await self.run_user("email dana smith")
        self.assertIn("<connections>", self.prompts[0])
        self.assertIn("not connected", self.prompts[0])


class Routes(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from server.routes import agents as agents_routes

        base.RealisticRosterAccuracy.setUp(self)
        for target, name, value in [
            (agents_routes, "get_agent_roster", lambda: self.roster),
            (agents_routes, "get_agent_retriever", lambda: self.retriever),
        ]:
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        app = FastAPI()
        app.include_router(agents_routes.router)
        self.client = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000))

    def test_list_search_and_stats(self) -> None:
        listing = self.client.get("/agents").json()
        self.assertEqual(listing["count"], self.roster.count())

        found = self.client.get("/agents/search", params={"q": "dentist appointment"}).json()
        self.assertEqual(found["agents"][0]["name"], "Dentist Appointment")

        self.assertEqual(self.client.get("/agents/search", params={"q": "zzzqqq"}).json()["agents"], [])
        self.assertEqual(self.client.get("/agents/search").status_code, 422)

        self.assertIn("turns_logged", self.client.get("/agents/retrieval-stats").json())

    def test_rejects_cross_site_and_non_local_requests(self) -> None:
        self.assertEqual(
            self.client.get("/agents", headers={"Origin": "https://evil.example"}).status_code, 403
        )
        self.assertEqual(
            self.client.get("/agents", headers={"Origin": "http://localhost:3000"}).status_code, 200
        )
        self.assertEqual(
            self.client.get("/agents", headers={"Host": "rebind.evil.example"}).status_code, 403
        )


class NameCoverage(unittest.TestCase):
    def setUp(self) -> None:
        self.roster, _ = base.make_roster()
        self.retriever = AgentRetriever(self.roster, trigger_counts=lambda: {})
        for first, last in [("Dave", "Garcia"), ("Dave", "Smith"), ("Grace", "Garcia"), ("Jo", "Smith"), ("Ann", "Smith")]:
            email = f"{first}.{last}@x.com".lower()
            self.roster.touch(f"Email to {first} {last}", f"Email {email}")
            self.roster.record_activity(f"Email to {first} {last}", [PersonRef(email=email, name=f"{first} {last}", role="recipient")])

    def margin(self, message: str) -> float:
        ranked = self.retriever.retrieve(message, use_recent=False, recent_pad=0).candidates
        return ranked[0].score - ranked[1].score

    def test_full_name_clearly_beats_partial_matches(self) -> None:
        self.assertEqual(self.retriever.retrieve("email dave garcia about lunch").candidates[0].name, "Email to Dave Garcia")
        self.assertGreater(self.margin("email dave garcia about lunch"), 0.15)

    def test_two_letter_first_name_counts(self) -> None:
        self.assertEqual(self.retriever.retrieve("email jo smith").candidates[0].name, "Email to Jo Smith")
        self.assertGreater(self.margin("email jo smith"), 0.1)

    def test_one_shared_part_is_still_a_tie(self) -> None:
        self.assertEqual(len(self.retriever.retrieve("email smith").tied_names), 3)
        self.assertEqual(len(self.retriever.retrieve("email dave").tied_names), 2)


if __name__ == "__main__":
    unittest.main()
