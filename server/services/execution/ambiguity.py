"""Work out what a user message leaves unclear, and what it allows to be sent.

The rule: when a message can be read more than one way and the readings lead to different
outcomes, the assistant asks instead of guessing. This module finds the cases code can see:

- draft    "send it" when it is not clear which unsent draft is meant (the one just discussed
           was already sent, or several are open)
- date     a weekday that could mean two dates ("this Friday" said on a Friday, "next Friday")
- account  Gmail is connected but its address is unknown, so a send would leave from an
           unconfirmed mailbox

Ties between agents (several Judys, a Q3 and a Q4 budget) are found by the retriever.

The interaction agent sees these as <ambiguity> notes and is told to ask. Because a model can
ignore a note, sending is also checked in code: each user message records which drafts it
clearly refers to (`set_send_authorizations`), and the execution runtime refuses to send any
other draft (`check_send`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from html import unescape
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .people import EMAIL_RE
from .retriever import RECENT_USE_MAX_AGE_DAYS, _TOKEN_RE, tokenize
from .roster import AgentRoster

ANY_AGENT = "*"  # a grant for an untracked draft, whichever agent ends up sending it

SEND_NOT_CONFIRMED_ERROR = (
    "Not sent: the user's latest message does not clearly say to send this draft. "
    "Do not retry; report back so the user can be asked which draft they mean."
)
SEND_ACCOUNT_UNKNOWN_ERROR = (
    "Not sent: the address of the connected Gmail account could not be confirmed, so the sender "
    "is unknown. Do not retry; the user needs to reopen Gmail in settings first."
)

# Words that carry no information about WHICH draft is meant.
_CONFIRM_WORDS = frozenset(
    """
    send sent it its yes yeah yep yup ya sure ok okay k go ahead for looks look good great fine perfect
    lgtm ship do that this one confirm confirmed approve approved please now as is thanks thank you
    cool sounds fire away just and out off right alright a an then works draft drafts
    email mail note message both all them those these every each s ll re ve m d
    """.split()
)
_ALL_WORDS = frozenset({"both", "all", "them", "those", "these", "every", "each"})
_SEND_WORDS = re.compile(
    r"\b(send|ship it|go ahead|lgtm|looks good|sounds good|confirm(ed)?|approved?|yes|yeah|yep|yup|sure|ok(ay)?|do it)\b",
    re.IGNORECASE,
)
_EXPLICIT_SEND = re.compile(r"\b(send|ship)\b", re.IGNORECASE)

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_NEXT_WEEKDAY = re.compile(r"\bnext\s+(" + "|".join(_WEEKDAYS) + r")\b", re.IGNORECASE)
_WEEKDAY = re.compile(r"(?<!every )(?<!next )\b(?:this\s+)?(" + "|".join(_WEEKDAYS) + r")\b", re.IGNORECASE)


@dataclass
class Ambiguity:
    kind: str  # "draft", "date" or "account"
    detail: str  # what is unclear and what to ask, for the interaction agent
    options: List[str] = field(default_factory=list)


@dataclass
class TurnAssessment:
    ambiguities: List[Ambiguity] = field(default_factory=list)
    grants: List[Tuple[str, Optional[int]]] = field(default_factory=list)


# ------------------------------------------------------------------ drafts


def _draft_label(row: Dict) -> str:
    recipient = row.get("recipient") or "someone"
    subject = f' "{row["subject"]}"' if row.get("subject") else ""
    return f"to {recipient}{subject}"


def _draft_words(row: Dict) -> Set[str]:
    recipient = (row.get("recipient") or "").lower()
    words = set(tokenize(re.sub(r"[@._+\-<>]", " ", recipient)))
    words |= set(tokenize(row.get("subject") or ""))
    return words - _CONFIRM_WORDS


def _content_words(message: str) -> Set[str]:
    return set(tokenize(message)) - _CONFIRM_WORDS


def _matched_words(words: Set[str], draft_words: Set[str]) -> Set[str]:
    """Message words that name this draft. A prefix counts, since addresses run names together
    ("evan" names evanscottpenguin@gmail.com as well as evan.scott@torontomu.ca)."""

    return {w for w in words if any(d == w or (len(w) >= 3 and d.startswith(w)) for d in draft_words)}


def _is_bare_confirmation(message: str) -> bool:
    """ "send it", "yeah that looks good send it", "send both": a go-ahead that names nothing."""

    words = [w for w in _TOKEN_RE.findall(message.lower())]
    return bool(words) and bool(_SEND_WORDS.search(message)) and all(w in _CONFIRM_WORDS for w in words)


def _mentioned_in(row: Dict, text: str) -> bool:
    """Whether the assistant's text names this draft (its recipient's address or its subject)."""

    lowered = text.lower()
    for address in EMAIL_RE.findall(row.get("recipient") or ""):
        if address.lower() in lowered:
            return True
    subject = (row.get("subject") or "").strip().lower()
    return len(subject) >= 4 and subject in lowered


_REPLY_RE = re.compile(r"<poke_reply[^>]*>(.*?)</poke_reply>", re.DOTALL)


def replies_since_last_user_message(transcript: str) -> str:
    """What the assistant said to the user since the user's previous message (the question a "yes" answers)."""

    tail = transcript.rsplit("<user_message", 1)[-1] if "<user_message" in transcript else transcript
    return "\n".join(unescape(m) for m in _REPLY_RE.findall(tail))


def assess_drafts(
    message: str,
    open_drafts: Sequence[Dict],
    last_turn_agents: Sequence[str],
    last_closed: Optional[Dict] = None,
    last_reply: str = "",
) -> TurnAssessment:
    """Which open drafts this message clearly asks to send, or why it is unclear.

    A bare "send it" or "yes" refers to the draft(s) the assistant named in its last reply, or
    failing that the drafts of the agents it used last turn. "The lunch one" names a draft by its
    recipient or subject words.
    """

    result = TurnAssessment()
    wants_all = bool(set(_TOKEN_RE.findall(message.lower())) & _ALL_WORDS)
    in_reply = [d for d in open_drafts if last_reply and _mentioned_in(d, last_reply)]

    if _is_bare_confirmation(message):
        recent = in_reply or [d for d in open_drafts if d["agent_name"] in last_turn_agents]
        if wants_all and recent:
            result.grants = [(d["agent_name"], d["id"]) for d in recent]
        elif len(recent) == 1:
            result.grants = [(recent[0]["agent_name"], recent[0]["id"])]
        elif len(recent) > 1:
            result.ambiguities.append(_draft_ambiguity(
                "The user said to send, but several drafts were discussed last.", recent))
        elif open_drafts:
            if last_closed is not None:
                lead = (f"The draft discussed last ({_draft_label(last_closed)}) was already sent or "
                        "deleted, so it is not clear the user means another one.")
            else:
                lead = "The user said to send, but no draft was discussed just before."
            result.ambiguities.append(_draft_ambiguity(lead, open_drafts))
        elif last_turn_agents:
            # Nothing tracked is open: the agent may be holding a draft OpenPoke did not create.
            result.grants = [(agent, None) for agent in last_turn_agents]
        return result

    # "send the lunch one", "send the one to torontomu": match the words to a draft. Without the
    # word "send", accept only an answer that is nothing but a reference to a draft ("the lunch
    # one") given right after the assistant talked about drafts, i.e. a reply to "which one?".
    words = _content_words(message)
    matches = {d["id"]: _matched_words(words, _draft_words(d)) for d in open_drafts}
    named = [d for d in open_drafts if matches[d["id"]]]
    explicit = bool(_EXPLICIT_SEND.search(message))
    pure_reference = bool(named) and words <= set().union(*(matches[d["id"]] for d in named))
    if not explicit and not (pure_reference and (in_reply or last_closed is not None or _SEND_WORDS.search(message))):
        return result

    if len(named) == 1 or (named and wants_all):
        result.grants = [(d["agent_name"], d["id"]) for d in named]
    elif len(named) > 1:
        result.ambiguities.append(_draft_ambiguity("Several drafts match what the user said.", named))
    else:
        # A new request ("send Bob a note") or a draft OpenPoke does not track. New emails are
        # drafted first, and a fresh draft is never covered by this grant.
        result.grants = [(ANY_AGENT, None)]
    return result


def _draft_ambiguity(lead: str, drafts: Sequence[Dict]) -> Ambiguity:
    options = [f"{_draft_label(d)} (agent {d['agent_name']})" for d in drafts]
    return Ambiguity(
        kind="draft",
        detail=f"{lead} Ask which draft to send before sending anything. Open drafts: " + "; ".join(options) + ".",
        options=options,
    )


# ------------------------------------------------------------------ dates


def _fmt(day: date) -> str:
    return f"{day.strftime('%a %b')} {day.day}"


def assess_dates(message: str, today: date) -> List[Ambiguity]:
    """Weekdays that could mean two different dates."""

    found: List[Ambiguity] = []
    seen: Set[str] = set()
    for match in _NEXT_WEEKDAY.finditer(message):
        name = match.group(1).lower()
        ahead = (_WEEKDAYS.index(name) - today.weekday()) % 7 or 7
        first = today + timedelta(days=ahead)
        options = [_fmt(first), _fmt(first + timedelta(days=7))]
        found.append(_date_ambiguity(match.group(0), options))
        seen.add(name)
    for match in _WEEKDAY.finditer(message):
        name = match.group(1).lower()
        if name in seen or _WEEKDAYS.index(name) != today.weekday():
            continue
        seen.add(name)
        options = [f"today ({_fmt(today)})", _fmt(today + timedelta(days=7))]
        found.append(_date_ambiguity(match.group(0), options))
    return found


def _date_ambiguity(phrase: str, options: List[str]) -> Ambiguity:
    return Ambiguity(
        kind="date",
        detail=(
            f'"{phrase}" could mean {" or ".join(options)}. Ask which before sending, booking or '
            "scheduling anything that depends on it. If you are only drafting, say plainly in the "
            "draft review which date you assumed."
        ),
        options=options,
    )


# ------------------------------------------------------------------ account


def assess_account(message: str, gmail_connected: bool, account_email: Optional[str]) -> List[Ambiguity]:
    if not gmail_connected or account_email or not _SEND_WORDS.search(message):
        return []
    return [Ambiguity(
        kind="account",
        detail=(
            "Gmail is connected but its address is not known yet, so it is unclear which mailbox "
            "would send. Sending is blocked until the account is confirmed; if a send fails for "
            "this reason, tell the user to reopen Gmail in settings."
        ),
    )]


# ------------------------------------------------------------------ the turn


def last_turn_agents(roster: AgentRoster) -> List[str]:
    """Agents the interaction agent sent work to in its most recent turn that did any (within a day)."""

    choices = roster.last_turn_choices()
    cutoff = datetime.now(timezone.utc) - timedelta(days=RECENT_USE_MAX_AGE_DAYS)
    fresh = []
    for name, ts in choices:
        try:
            when = datetime.fromisoformat(ts)
        except ValueError:
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        if when >= cutoff:
            fresh.append(name)
    return fresh


def assess_user_message(
    message: str,
    roster: AgentRoster,
    *,
    today: date,
    gmail_connected: bool,
    account_email: Optional[str],
    transcript: str = "",
) -> TurnAssessment:
    """Everything this message leaves unclear, plus the sends it clearly allows.

    `transcript` is the conversation before this message; the assistant's replies since the
    user's previous message say which draft a bare "yes" or "send it" answers.
    """

    last = last_turn_agents(roster)
    last_closed = None
    for agent in last:
        closed = roster.last_closed_draft(agent)
        if closed and (last_closed is None or closed["closed_at"] > last_closed["closed_at"]):
            last_closed = closed
    result = assess_drafts(
        message, roster.open_draft_rows(), last, last_closed, replies_since_last_user_message(transcript)
    )
    result.ambiguities.extend(assess_dates(message, today))
    result.ambiguities.extend(assess_account(message, gmail_connected, account_email))
    return result


def check_send(roster: AgentRoster, agent_name: str, draft_id: Optional[str]) -> bool:
    """True when the user's latest message clearly allows this agent to send this draft."""

    grants = roster.send_authorizations()
    row = roster.tracked_draft(agent_name, draft_id)
    if row is not None:
        return any(row_id == row["id"] for _, row_id in grants)
    return (agent_name, None) in grants or (ANY_AGENT, None) in grants


__all__ = [
    "ANY_AGENT",
    "Ambiguity",
    "SEND_ACCOUNT_UNKNOWN_ERROR",
    "SEND_NOT_CONFIRMED_ERROR",
    "TurnAssessment",
    "assess_account",
    "assess_dates",
    "assess_drafts",
    "assess_user_message",
    "check_send",
    "last_turn_agents",
    "replies_since_last_user_message",
]
