"""Tests for execution-agent run guards: early stop and summary protection."""

import types
import unittest
from unittest import mock

from server.tests.helpers import make_roster


class ExecutionRunGuards(unittest.IsolatedAsyncioTestCase):
    """Failed runs must not spend extra LLM calls or overwrite an agent's summary."""

    def setUp(self) -> None:
        from server.agents.execution_agent import agent as agent_module
        from server.agents.execution_agent import runtime as runtime_module
        from server.services.execution.log_store import ExecutionAgentLogStore

        self.runtime_module = runtime_module
        self.roster, directory = make_roster()
        self.roster.touch("Email to Alice", "Original instructions about Alice")
        self.llm_calls = 0
        self.tool = mock.Mock()

        settings = types.SimpleNamespace(openrouter_api_key="k", execution_agent_model="m")
        store = ExecutionAgentLogStore(directory / "logs")
        patches = [
            mock.patch.object(runtime_module, "get_settings", return_value=settings),
            mock.patch.object(runtime_module, "get_agent_roster", return_value=self.roster),
            mock.patch.object(runtime_module, "get_tool_schemas", return_value=[]),
            mock.patch.object(runtime_module, "get_tool_registry", return_value={"gmail_search_people": self.tool}),
            mock.patch.object(runtime_module, "get_active_gmail_email", return_value=None),
            mock.patch.object(agent_module, "get_execution_agent_logs", return_value=store),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def script_llm(self, *replies: dict) -> None:
        queue = list(replies)

        async def fake_completion(**_kwargs):
            self.llm_calls += 1
            return {"choices": [{"message": queue.pop(0)}]}

        patch = mock.patch.object(self.runtime_module, "request_chat_completion", fake_completion)
        patch.start()
        self.addCleanup(patch.stop)

    @staticmethod

    def tool_call() -> dict:
        return {"content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "gmail_search_people", "arguments": "{}"}}]}

    async def run_agent(self):
        runtime = self.runtime_module.ExecutionAgentRuntime("Email to Alice")
        return await runtime.execute("find alice")

    def summary(self) -> str:
        return self.roster.get_entry("Email to Alice")["summary"]

    async def test_gmail_not_connected_stops_after_one_llm_call(self) -> None:
        from server.services.gmail import GMAIL_NOT_CONNECTED_ERROR

        self.tool.return_value = {"error": GMAIL_NOT_CONNECTED_ERROR}
        self.script_llm(self.tool_call())  # a second LLM call would pop an empty queue and fail
        result = await self.run_agent()

        self.assertFalse(result.success)
        self.assertEqual(self.llm_calls, 1)
        self.assertIn("Gmail not connected", result.error)
        self.assertEqual(self.summary(), "Original instructions about Alice")

    async def test_all_tools_failing_keeps_previous_summary(self) -> None:
        self.tool.return_value = {"error": "rate limited"}
        self.script_llm(self.tool_call(), {"content": "Sorry, the search failed."})
        result = await self.run_agent()

        self.assertTrue(result.success)
        self.assertEqual(self.llm_calls, 2)
        self.assertEqual(self.summary(), "Original instructions about Alice")

    async def test_successful_run_updates_summary(self) -> None:
        self.tool.return_value = {"data": {"people": []}, "successful": True}
        self.script_llm(self.tool_call(), {"content": "Found Alice at alice@piedpiper.com."})
        await self.run_agent()
        self.assertEqual(self.summary(), "Found Alice at alice@piedpiper.com.")

    async def test_owner_address_is_removed_from_summary(self) -> None:
        self.script_llm({"content": "**Done.** Checked the inbox (me@me.com) and found nothing new."})
        with mock.patch.object(self.runtime_module, "get_active_gmail_email", return_value="me@me.com"):
            await self.run_agent()
        self.assertNotIn("me@me.com", self.summary())
        self.assertNotIn("**", self.summary())

    async def test_run_without_tools_updates_summary(self) -> None:
        self.script_llm({"content": "Nothing to do here."})
        await self.run_agent()
        self.assertEqual(self.summary(), "Nothing to do here.")


if __name__ == "__main__":
    unittest.main()
