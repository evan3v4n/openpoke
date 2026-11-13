from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Query

from ..services.execution import get_agent_retriever, get_agent_roster

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("")
# List every execution agent with its people, threads, subjects and open drafts
def list_agents() -> Dict[str, Any]:
    entries = get_agent_roster().get_entries()
    return {"count": len(entries), "agents": entries}


@router.get("/search")
# Rank agents for a query exactly as the interaction agent's lookup does
def search_agents(q: str = Query(..., min_length=1)) -> Dict[str, Any]:
    matches = get_agent_retriever().search(q)
    return {"query": q, "agents": [match.to_dict() for match in matches]}


@router.get("/retrieval-stats")
# How often the agent the model used was among the candidates it was shown
def retrieval_stats() -> Dict[str, Any]:
    return get_agent_roster().retrieval_stats()


__all__ = ["router"]
