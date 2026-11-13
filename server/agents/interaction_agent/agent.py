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
from ...services.execution.ambiguity import assess_user_message
from ...services.execution.retriever import humanize_age, reporting_agents

MAX_WHY_CHARS = 240  # cap on the reasons shown per agent
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
    if message_type == "user":
        unclear = _assess_user_turn(latest_text, transcript)
        if unclear:
            sections.append(f"<ambiguities>\n{unclear}\n</ambiguities>")
    connections = _render_connections()
    if connections:
        sections.append(f"<connections>\n{connections}\n</connections>")
    sections.append(_render_current_turn(latest_text, message_type))

    content = "\n\n".join(sections)
    return [{"role": "user", "content": content}]


# Format conversation transcript into XML tags for LLM context
def _render_conversation_history(transcript: str) -> str:
    history = transcript.strip()
    if not history:
        history = "None"
    return f"<conversation_history>\n{history}\n</conversation_history>"


# Find what this user message leaves unclear (which draft, which date, which mailbox) and record
# which drafts it clearly allows to be sent; the execution runtime refuses any other send.
def _assess_user_turn(latest_text: str, transcript: str) -> str:
    from datetime import date

    from ...services.gmail import get_active_gmail_email, is_gmail_connected  # lazy: import cycle

    roster = get_agent_roster()
    try:
        assessment = assess_user_message(
            latest_text,
            roster,
            today=date.today(),
            gmail_connected=is_gmail_connected(),
            account_email=get_active_gmail_email(),
            transcript=transcript,
        )
        roster.set_send_authorizations(assessment.grants)
    except Exception as exc:  # fail closed: with no grants recorded, nothing can be sent this turn
        logger.error("ambiguity check failed", extra={"error": str(exc)})
        try:
            roster.set_send_authorizations([])
        except Exception:  # pragma: no cover - defensive
            pass
        return ""

    return "\n".join(
        f'<ambiguity kind="{item.kind}">{escape(item.detail)}</ambiguity>' for item in assessment.ambiguities
    )


# Report unavailable services so the interaction agent does not dispatch work that cannot run
def _render_connections() -> str:
    from ...services.gmail import is_gmail_connected  # lazy: avoids an import cycle at startup

    if is_gmail_connected():
        return ""
    return (
        "gmail: not connected. Anything that needs email will fail until the user connects Gmail "
        "in settings. Do not send email tasks to an agent; tell the user to connect Gmail first. "
        "Other tasks (reminders, general questions) are unaffected."
    )


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
            why = " ".join("; ".join(candidate.reasons).split())[:MAX_WHY_CHARS]
            attrs += f' why="{escape(why, quote=True)}"'
        rendered.append(f"<agent {attrs}>{escape(candidate.summary)}</agent>")

    # Several agents match a user request equally (ten Judys, or a Q3 and a Q4 budget for "the
    # budget"): list all of them so the model asks which is meant instead of fanning work out.
    if message_type != "agent" and shortlist.tied_names:
        shown = shortlist.tied_names[:MAX_TIED_NAMES]
        names = "; ".join(escape(name, quote=True) for name in shown)
        extra = len(shortlist.tied_names) - len(shown)
        if extra > 0:
            names += f"; and {extra} more"
        rendered.append(
            f'<ambiguity kind="{shortlist.tie_kind or "person"}" count="{len(shortlist.tied_names)}">'
            f"{len(shortlist.tied_names)} agents match this "
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
