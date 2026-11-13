"""Extract people (email addresses) and thread ids from execution-agent activity.

The roster uses these to index each execution agent by the people it deals with,
so an agent can be found later by "alice" or "alice@example.com" regardless of
what the agent happens to be called.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from email.utils import getaddresses
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# Outbound Gmail tool arguments -> the role that person plays in the exchange.
_ARG_ROLES: Dict[str, str] = {
    "recipient_email": "recipient",
    "extra_recipients": "recipient",
    "cc": "cc",
    "bcc": "bcc",
}


@dataclass(frozen=True)
class PersonRef:
    """One person seen in an agent's activity."""

    email: str
    name: str = ""
    role: str = "mentioned"  # sender | recipient | cc | bcc | mentioned


@dataclass
class Findings:
    people: List[PersonRef] = field(default_factory=list)
    thread_ids: List[str] = field(default_factory=list)
    subjects: List[str] = field(default_factory=list)


def parse_addresses(value: Any) -> List[Tuple[str, str]]:
    """Parse 'Name <a@b.com>, c@d.com' (or a list of such) into (name, email) pairs."""

    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    strings = [item for item in items if isinstance(item, str)]

    parsed: List[Tuple[str, str]] = []
    for name, address in getaddresses(strings):
        address = address.strip().lower()
        if EMAIL_RE.fullmatch(address):
            parsed.append((name.strip().strip('"'), address))
    return parsed


def people_from_text(text: Optional[str]) -> List[PersonRef]:
    """Pick bare email addresses out of free text (e.g. interaction-agent instructions)."""

    if not text:
        return []
    seen: Set[str] = set()
    refs: List[PersonRef] = []
    for match in EMAIL_RE.findall(text):
        email = match.lower()
        if email not in seen:
            seen.add(email)
            refs.append(PersonRef(email=email, role="mentioned"))
    return refs


def _iter_email_records(result: Any) -> Iterator[Dict[str, Any]]:
    """Yield email-like dicts (with sender/recipient) from a tool result."""

    if isinstance(result, dict):
        for key in ("emails", "messages"):
            nested = result.get(key)
            if isinstance(nested, list):
                yield from (item for item in nested if isinstance(item, dict))
                return
        if "sender" in result or "recipient" in result:
            yield result
    elif isinstance(result, list):
        yield from (item for item in result if isinstance(item, dict))


def extract_from_tool_call(
    arguments: Optional[Dict[str, Any]],
    result: Any,
    exclude: Iterable[str] = (),
) -> Findings:
    """Return people and thread ids found in a tool call's arguments and result.

    `exclude` holds addresses to ignore (the account owner appears on every email).
    """

    excluded = {e.lower() for e in exclude if e}
    findings = Findings()
    seen_people: Set[Tuple[str, str]] = set()
    seen_threads: Set[str] = set()

    def add_person(name: str, email: str, role: str) -> None:
        if email in excluded or (email, role) in seen_people:
            return
        seen_people.add((email, role))
        findings.people.append(PersonRef(email=email, name=name, role=role))

    def add_thread(thread_id: Any) -> None:
        if isinstance(thread_id, str) and thread_id and thread_id not in seen_threads:
            seen_threads.add(thread_id)
            findings.thread_ids.append(thread_id)

    def add_subject(subject: Any) -> None:
        if isinstance(subject, str):
            cleaned = " ".join(subject.split())
            if cleaned and cleaned not in findings.subjects:
                findings.subjects.append(cleaned)

    for key, role in _ARG_ROLES.items():
        for name, email in parse_addresses((arguments or {}).get(key)):
            add_person(name, email, role)
    add_thread((arguments or {}).get("thread_id"))
    add_subject((arguments or {}).get("subject"))

    for record in _iter_email_records(result):
        for name, email in parse_addresses(record.get("sender")):
            add_person(name, email, "sender")
        for name, email in parse_addresses(record.get("recipient")):
            add_person(name, email, "recipient")
        add_thread(record.get("thread_id"))
        add_subject(record.get("subject"))

    return findings


def tool_succeeded(result: Any) -> bool:
    """False for error payloads and for Composio responses flagged `successful: false`."""

    if isinstance(result, dict):
        if result.get("error"):
            return False
        if result.get("successful") is False:
            return False
    return True


def find_draft_id(result: Any, _depth: int = 0) -> Optional[str]:
    """Best-effort extraction of a Gmail draft id from a create-draft response."""

    if _depth > 4:
        return None
    if isinstance(result, dict):
        for key in ("draft_id", "draftId"):
            value = result.get(key)
            if isinstance(value, str) and value:
                return value
        # Gmail draft resource: {"id": "...", "message": {...}}
        if "message" in result and isinstance(result.get("id"), str) and result["id"]:
            return result["id"]
        for value in result.values():
            found = find_draft_id(value, _depth + 1)
            if found:
                return found
    elif isinstance(result, list):
        for value in result:
            found = find_draft_id(value, _depth + 1)
            if found:
                return found
    return None


__all__ = [
    "EMAIL_RE",
    "Findings",
    "PersonRef",
    "extract_from_tool_call",
    "find_draft_id",
    "parse_addresses",
    "people_from_text",
    "tool_succeeded",
]
