import asyncio

from server.services.execution import current_run
from server.services.execution.activity import ActivityTracker


def test_tracker_lifecycle_and_contextvar():
    t = ActivityTracker()
    run_id = t.start("Agent A", "do a thing")
    current_run.set(run_id)

    async def child():  # tasks inherit the contextvar, as wait_for does in the batch manager
        t.add("thinking", "Thinking…")
        t.add("tool_call", "x" * 1000)

    asyncio.run(child())
    snap = t.snapshot()
    assert snap[0]["status"] == "running"
    assert [e["kind"] for e in snap[0]["events"]] == ["thinking", "tool_call"]
    assert len(snap[0]["events"][1]["text"]) == 400

    t.finish(run_id, True, "ok")
    assert t.snapshot()[0]["status"] == "done"
    current_run.set(None)
    t.add("thinking", "ignored")  # no active run: no-op
