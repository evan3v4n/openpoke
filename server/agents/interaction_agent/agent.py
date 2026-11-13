"""Interaction agent helpers for prompt construction."""

from html import escape
from pathlib import Path
from typing import Dict, List, Optional

from ...logging_config import logger
from ...services.execution import (
    RetrievalContext,
    Shortlist,
    current_retrieval,
    get_agent_retriever,
    get_agent_roster,
)
from ...services.execution.retriever import humanize_age, reporting_agents

MAX_TIED_NAMES = 12  # names listed in an <ambiguity> hint

_prompt_path = Path(__file__).parent / "system_prompt.md"
SYSTEM_PROMPT = _prompt_path.read_text(encoding="utf-8").strip()


# Load and return the pre-defined system prompt from markdown file
def build_system_prompt() -> str:
    """Return the static system prompt for the interaction agent."""
    return SYSTEM_PROMPT


# Build structured message with conversation history, active agents, and current turn
def prepare_message_with_history(
    latest_text: str,
    transcript: str,
    message_type: str = "user",
) -> List[Dict[str, str]]:
    """Compose a message that bundles history, roster, and the latest turn."""
    sections: List[str] = []

    sections.append(_render_conversation_history(transcript))
    sections.append(
        f"<active_agents>\n{_render_active_agents(latest_text, message_type)}\n</active_agents>"
    )
    sections.append(_render_current_turn(latest_text, message_type))

    content = "\n\n".join(sections)
    return [{"role": "user", "content": content}]


# Format conversation transcript into XML tags for LLM context
def _render_conversation_history(transcript: str) -> str:
    history = transcript.strip()
    if not history:
        history = "None"
    return f"<conversation_history>\n{history}\n</conversation_history>"


# Show only the agents most relevant to this message, with why, instead of the whole roster
def _render_active_agents(latest_text: str, message_type: str) -> str:
    roster = get_agent_roster()
    total = roster.count()

    forced = reporting_agents(latest_text) if message_type == "agent" else []
    shortlist = Shortlist()
    if total:
        try:
            shortlist = get_agent_retriever().retrieve(latest_text, forced_names=forced)
        except Exception as exc:  # retrieval must never take the interaction agent down
            logger.error("agent retrieval failed", extra={"error": str(exc)})
    candidates = shortlist.candidates

    kind = "agent" if message_type == "agent" else "user"
    retrieval_id: Optional[int] = None
    try:
        retrieval_id = roster.log_retrieval(kind, latest_text, [c.to_dict() for c in candidates])
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Could not log retrieval: {exc}")
    current_retrieval.set(
        RetrievalContext(retrieval_id=retrieval_id, shown={c.name for c in candidates})
    )

    rendered: List[str] = []
    for candidate in candidates:
        name = escape(candidate.name or "agent", quote=True)
        attrs = f'name="{name}"'
        if candidate.last_active_at:
            attrs += f' last_active="{humanize_age(candidate.last_active_at)}"'
        if candidate.reasons:
            attrs += f' why="{escape("; ".join(candidate.reasons), quote=True)}"'
        rendered.append(f"<agent {attrs}>{escape(candidate.summary)}</agent>")

    # Several agents match a user request equally (e.g. ten Judys): list all of them so the model
    # asks which one is meant instead of fanning work out to every one.
    if message_type != "agent" and shortlist.tied_names:
        shown = shortlist.tied_names[:MAX_TIED_NAMES]
        names = "; ".join(escape(name, quote=True) for name in shown)
        extra = len(shortlist.tied_names) - len(shown)
        if extra > 0:
            names += f"; and {extra} more"
        rendered.append(
            f'<ambiguity count="{len(shortlist.tied_names)}">{len(shortlist.tied_names)} agents match this '
            f"message about equally: {names}. If the user did not say which one, or all of them, "
            "ask them which one they mean before sending work.</ambiguity>"
        )

    hidden = total - len(candidates)
    if hidden > 0:
        rendered.append(
            f'<other_agents count="{hidden}">Not listed. If none of the agents above fits, '
            "search by person, email or topic before creating a new agent.</other_agents>"
        )

    return "\n".join(rendered) if rendered else "None"


# Wrap the current message in appropriate XML tags based on sender type
def _render_current_turn(latest_text: str, message_type: str) -> str:
    tag = "new_agent_message" if message_type == "agent" else "new_user_message"
    body = latest_text.strip()
    return f"<{tag}>\n{body}\n</{tag}>"
