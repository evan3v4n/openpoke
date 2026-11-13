import asyncio
import unittest

from server.services.execution import current_run
from server.services.execution.activity import ActivityTracker


class ActivityTrackerTests(unittest.TestCase):
    def test_tracker_lifecycle_and_contextvar(self) -> None:
        t = ActivityTracker()
        run_id = t.start("Agent A", "do a thing")
        current_run.set(run_id)

        async def child():  # tasks inherit the contextvar, as wait_for does in the batch manager
            t.add("thinking", "Thinking…")
            t.add("tool_call", "x" * 1000)

        asyncio.run(child())
        snap = t.snapshot()
        self.assertEqual(snap[0]["status"], "running")
        self.assertEqual([e["kind"] for e in snap[0]["events"]], ["thinking", "tool_call"])
        self.assertEqual(len(snap[0]["events"][1]["text"]), 400)

        t.finish(run_id, True, "ok")
        self.assertEqual(t.snapshot()[0]["status"], "done")
        current_run.set(None)
        t.add("thinking", "ignored")  # no active run: no-op


if __name__ == "__main__":
    unittest.main()
