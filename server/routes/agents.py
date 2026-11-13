from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..services.triggers import get_trigger_service
from ..services.execution import get_activity_tracker, get_agent_retriever, get_agent_roster



_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _hostname(value: str) -> str:
    """Hostname of a Host header value or an Origin URL, without scheme or port."""

    value = value.split("://", 1)[-1].split("/", 1)[0]
    if value.startswith("["):  # [::1]:8001
        return value[1:].split("]", 1)[0].lower()
    return value.rsplit(":", 1)[0].lower() if value.count(":") == 1 else value.lower()


# These endpoints expose mailbox-derived data (emails, subjects, drafts): local use only.
# The peer address alone is not enough, because CORS allows any website the user visits to
# call 127.0.0.1 from their browser. So the Host header (DNS rebinding) and the Origin header
# (cross-site requests) must be local too.
def _require_local(request: Request) -> None:
    peer = request.client.host if request.client else ""
    host = _hostname(request.headers.get("host", ""))
    origin = request.headers.get("origin")
    if (
        peer not in _LOCAL_HOSTS
        or host not in _LOCAL_HOSTS
        or (origin is not None and _hostname(origin) not in _LOCAL_HOSTS)
    ):
        raise HTTPException(status_code=403, detail="Agents endpoints are only available locally")


router = APIRouter(prefix="/agents", tags=["agents"], dependencies=[Depends(_require_local)])


@router.get("")
# List every execution agent with its people, threads, subjects and open drafts
def list_agents() -> Dict[str, Any]:
    entries = get_agent_roster().get_entries()
    return {"count": len(entries), "agents": entries}


@router.get("/active")
# Running execution agents (and just-finished ones) with their live event feed
def active_agents() -> Dict[str, Any]:
    return {"runs": get_activity_tracker().snapshot()}


@router.get("/pending")
# Open loops: unsent drafts and active reminders, each with the agent that owns it
def pending_items() -> Dict[str, Any]:
    drafts = [
        {"agent_name": name, **draft}
        for name, items in get_agent_roster().open_drafts().items()
        for draft in items
    ]
    reminders = [
        {
            "id": t.id,
            "agent_name": t.agent_name,
            "payload": t.payload[:300],
            "next_trigger": t.next_trigger,
            "recurrence_rule": t.recurrence_rule,
            "timezone": t.timezone,
        }
        for t in get_trigger_service().list_active()
    ]
    return {"drafts": drafts, "reminders": reminders}


@router.get("/search")
# Rank agents for a query exactly as the interaction agent's lookup does
def search_agents(q: str = Query(..., min_length=1)) -> Dict[str, Any]:
    matches = get_agent_retriever().search(q).candidates
    return {"query": q, "agents": [match.to_dict() for match in matches]}


@router.get("/retrieval-stats")
# How often the agent the model used was among the candidates it was shown
def retrieval_stats() -> Dict[str, Any]:
    return get_agent_roster().retrieval_stats()


__all__ = ["router"]
