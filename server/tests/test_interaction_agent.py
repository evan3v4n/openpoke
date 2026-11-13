"""Tests for the interaction agent's prompt section and tools."""

import asyncio
import unittest
from unittest import mock

from server.services.execution.retriever import AgentRetriever
from server.tests.helpers import make_roster, seed_alice
from server.tests.test_retriever import JudyRosterCase


class Rendering(unittest.TestCase):
    def test_prompt_section_is_bounded_and_explained(self) -> None:
        from server.agents.interaction_agent import agent as interaction_agent
        from server.services.execution.retriever import current_retrieval

        roster, _ = make_roster()
        retriever = AgentRetriever(roster, trigger_counts=lambda: {})
        seed_alice(roster)
        for i in range(30):
            roster.touch(f"Filler {i}", f"unrelated work {i}")

        with mock.patch.object(interaction_agent, "get_agent_roster", return_value=roster), \
                mock.patch.object(interaction_agent, "get_agent_retriever", return_value=retriever):
            rendered = interaction_agent._render_active_agents("did alice reply?", "user")

        self.assertIn('name="Email to Alice"', rendered)
        self.assertIn("why=", rendered)
        self.assertIn("<other_agents", rendered)
        self.assertLessEqual(rendered.count("<agent "), 8)

        context = current_retrieval.get()
        self.assertIn("Email to Alice", context.shown)
        self.assertEqual(roster.retrieval_stats()["turns_logged"], 1)

    def test_first_turn_with_empty_roster_is_still_logged(self) -> None:
        from server.agents.interaction_agent import agent as interaction_agent
        from server.services.execution.retriever import current_retrieval

        roster, _ = make_roster()
        retriever = AgentRetriever(roster, trigger_counts=lambda: {})
        with mock.patch.object(interaction_agent, "get_agent_roster", return_value=roster), \
                mock.patch.object(interaction_agent, "get_agent_retriever", return_value=retriever):
            rendered = interaction_agent._render_active_agents("summarize my recent emails", "user")

        self.assertEqual(rendered, "None")
        context = current_retrieval.get()
        self.assertIsNotNone(context.retrieval_id)
        roster.log_choice(context.retrieval_id, "Recent Email Summary", was_new=True, was_candidate=False)
        stats = roster.retrieval_stats()
        self.assertEqual((stats["turns_logged"], stats["new_agents_created"]), (1, 1))

    def test_connections_hint_only_when_gmail_is_down(self) -> None:
        from server.agents.interaction_agent import agent as interaction_agent

        roster, _ = make_roster()
        patches = (
            mock.patch.object(interaction_agent, "get_agent_roster", return_value=roster),
            mock.patch.object(interaction_agent, "get_agent_retriever", return_value=AgentRetriever(roster)),
        )
        with patches[0], patches[1]:
            with mock.patch("server.services.gmail.is_gmail_connected", return_value=False):
                down = interaction_agent.prepare_message_with_history("email bob", "", "user")[0]["content"]
            with mock.patch("server.services.gmail.is_gmail_connected", return_value=True):
                up = interaction_agent.prepare_message_with_history("email bob", "", "user")[0]["content"]
        self.assertIn("<connections>", down)
        self.assertIn("gmail: not connected", down)
        self.assertNotIn("<connections>", up)


class AmbiguityPrompt(JudyRosterCase):
    """What the interaction agent sees when several agents match the message."""

    def _render(self, message: str, kind: str = "user") -> str:
        from server.agents.interaction_agent import agent as interaction_agent

        with mock.patch.object(interaction_agent, "get_agent_roster", return_value=self.roster), \
                mock.patch.object(interaction_agent, "get_agent_retriever", return_value=self.retriever):
            return interaction_agent._render_active_agents(message, kind)

    def test_prompt_lists_every_tied_agent_and_says_to_ask(self) -> None:
        rendered = self._render("Did Judy reply?")
        self.assertIn('<ambiguity kind="person" count="10">', rendered)
        for last in self.LAST_NAMES:  # including the ones the shortlist itself had to drop
            self.assertIn(f"Email to Judy {last}", rendered)
        self.assertIn("ask them which one they mean", rendered)

    def test_prompt_has_no_ambiguity_note_when_one_agent_wins_or_on_agent_reports(self) -> None:
        self.assertNotIn("<ambiguity", self._render("Did Judy Patel reply?"))
        self.assertNotIn("<ambiguity", self._render("[SUCCESS] Email to Judy Kim: no reply from Judy", "agent"))

    def test_find_agents_reports_total_and_ties(self) -> None:
        from server.agents.interaction_agent import tools

        with mock.patch.object(tools, "get_agent_retriever", return_value=self.retriever):
            payload = tools.find_agents("Judy").payload
        self.assertEqual(payload["total_matches"], 10)
        self.assertEqual(len(payload["agents"]), 8)
        self.assertEqual(len(payload["match_equally"]), 10)


class InteractionToolWiring(unittest.IsolatedAsyncioTestCase):
    async def test_dispatch_keeps_owner_address_out_of_the_people_index(self) -> None:
        from server.agents.interaction_agent import tools

        roster, _ = make_roster()
        execute = mock.AsyncMock()
        with mock.patch.object(tools, "get_agent_roster", return_value=roster), \
                mock.patch.object(tools, "get_execution_agent_logs", return_value=mock.Mock()), \
                mock.patch.object(tools._EXECUTION_BATCH_MANAGER, "execute_agent", execute), \
                mock.patch("server.services.gmail.get_active_gmail_email", return_value="me@me.com"):
            result = tools.send_message_to_agent(
                "Stretch Reminder", "Check the inbox of me@me.com for anything from bob@x.io"
            )
            # The dispatch runs as a background task; let it finish while the mock is still
            # in place, otherwise it would run the real agent (and call the LLM) after the test.
            await asyncio.sleep(0)

        execute.assert_awaited_once()
        self.assertTrue(result.success)
        self.assertEqual(set(roster.get_entry("Stretch Reminder")["people"]), {"bob@x.io"})

if __name__ == "__main__":
    unittest.main()
