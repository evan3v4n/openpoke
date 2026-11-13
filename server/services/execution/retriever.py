"""Rank execution agents against an incoming message.

The interaction agent decides whether to reuse an agent or create a new one; this
module only supplies a short, explained list of likely candidates so the prompt
stays small however many agents exist.

Score = entity match + keyword match + open loops + recent use + recency. Entity matching
(email addresses and names from the people index) dominates because requests
about a person need an exact hit, which embeddings are poor at. A message with no
usable content ("yes, send it") produces no entity or keyword signal, so the
ranking falls back to open loops (unsent drafts, active reminders), the agents the
interaction agent dispatched to in the last few turns, and recency.

Scoring is plain Python over the roster: with hundreds of agents it takes
milliseconds and needs no extra service. (SQLite FTS5 BM25 was tried first; on a
small roster its scores collapse to ~1e-6, so it is not used for ranking.)
"""

from __future__ import annotations

import math
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from ...logging_config import logger
from .people import EMAIL_RE
from .roster import AgentRoster, get_agent_roster

# Ranking weights.
W_ENTITY = 1.0
W_KEYWORD = 0.6
W_LOOP = 0.35
W_RECENCY = 0.25
W_RECENT_USE = 0.5
MIN_SCORE = 0.2

# Agents the interaction agent dispatched work to in the last few turns, newest first.
# This is what "check again" or "any update?" refers to, so it outranks raw recency.
RECENT_USE_WEIGHTS = (1.0, 0.6, 0.4)
RECENT_USE_MAX_AGE_DAYS = 1.0

# When the message itself names a person (or an agent by name), context (recent use,
# recency) only breaks ties: it is scaled down so it cannot outrank an exact match.
# A live "check judy brown" turn scored Chen 1.758 vs Brown 1.750 before this existed.
CONTEXT_SCALE_WITH_CONTENT = 0.4
STRONG_KEYWORD_SCORE = 0.25  # the query's words clearly appear in an agent's fields

# Agents whose score is within this of the best content match count as tied: the message
# does not say which of them it means (e.g. "Did Judy reply?" with ten Judys).
TIE_EPSILON = 0.02
RECENCY_HALF_LIFE_DAYS = 3.0

# Match strength per source.
ENTITY_EMAIL = 1.0
ENTITY_NAME = 0.7
PARTIAL_NAME_FLOOR = 0.6  # a name matched by one of its tokens scores this share of ENTITY_NAME
ENTITY_EXTRA_PERSON = 0.1  # bonus per additional matched person, capped below
ENTITY_EXTRA_CAP = 0.3

# Field weights for keyword matching.
FIELD_WEIGHTS = {"name": 3.0, "people": 2.0, "subjects": 2.0, "summary": 1.5}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_REPORT_RE = re.compile(r"^\[(?:SUCCESS|FAILED)\]\s+(.+?):", re.MULTILINE)
_STOPWORDS = frozenset(
    """
    the and for you your that this with have has had from they them their what when where which who whom
    will would could should can cant cannot did does done was were are not but all any out get got
    let lets please just also about into over then than there here some more most very much like want
    need make made send sent email emails mail message reply respond check tell ask see look find
    yes yeah yep nope okay sure thanks thank hey hello
    """.split()
)


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens of length >= 3, minus stopwords."""

    return [
        t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) >= 3 and t not in _STOPWORDS
    ]


def _stem(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def _age_days(iso: str) -> float:
    try:
        then = datetime.fromisoformat(iso)
    except ValueError:
        return 365.0
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - then).total_seconds() / 86400.0)


def humanize_age(iso: str) -> str:
    days = _age_days(iso)
    minutes = days * 1440
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{int(minutes)}m ago"
    if minutes < 1440:
        return f"{int(minutes // 60)}h ago"
    return f"{int(days)}d ago"


@dataclass
class Candidate:
    name: str
    score: float
    reasons: List[str] = field(default_factory=list)
    summary: str = ""
    last_active_at: str = ""
    open_drafts: List[Dict[str, str]] = field(default_factory=list)
    active_triggers: int = 0
    content_score: float = 0.0  # entity + keyword part of the score only (no context signals)
    matched_people: frozenset = frozenset()  # emails of people from the message this agent knows

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 3),
            "reasons": self.reasons,
            "summary": self.summary,
            "last_active": humanize_age(self.last_active_at) if self.last_active_at else "",
            "open_drafts": self.open_drafts,
            "active_triggers": self.active_triggers,
        }


@dataclass
class Shortlist:
    """A ranked shortlist plus what the shortlist alone would hide."""

    candidates: List[Candidate] = field(default_factory=list)
    matching_total: int = 0  # agents with any real signal for this message
    # Every agent that matches the message about equally (2 or more), empty when one clearly wins.
    tied_names: List[str] = field(default_factory=list)


@dataclass
class RetrievalContext:
    """What the interaction agent was shown this turn (for measuring retrieval quality)."""

    retrieval_id: Optional[int]
    shown: Set[str] = field(default_factory=set)


# Set per interaction turn; read by the interaction agent's tools.
current_retrieval: ContextVar[Optional[RetrievalContext]] = ContextVar(
    "current_retrieval", default=None
)


def reporting_agents(payload: str) -> List[str]:
    """Agent names in an execution-agent report ('[SUCCESS] Name: ...')."""

    return list(dict.fromkeys(m.strip() for m in _REPORT_RE.findall(payload or "")))


def _active_trigger_counts() -> Dict[str, int]:
    try:
        from ..triggers import get_trigger_service  # lazy: avoids an import cycle

        return get_trigger_service().active_counts()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Could not read trigger counts for retrieval: {exc}")
        return {}


class AgentRetriever:
    def __init__(self, roster: Optional[AgentRoster] = None, trigger_counts=_active_trigger_counts):
        self._roster = roster or get_agent_roster()
        self._trigger_counts = trigger_counts

    def candidates(self, message: str, **kwargs: Any) -> List[Candidate]:
        """Just the ranked candidates; see `retrieve` for the full shortlist."""

        return self.retrieve(message, **kwargs).candidates

    def retrieve(
        self,
        message: str,
        *,
        limit: int = 6,
        recent_pad: int = 2,
        forced_names: Sequence[str] = (),
        use_recent: bool = True,
        require_content: bool = False,
    ) -> Shortlist:
        """Top agents for `message`, each with the reasons it was chosen.

        `forced_names` are always included (agents that just reported back).
        `recent_pad` adds the most recently active agents that did not otherwise rank,
        a safety net for follow-ups that carry no content. `use_recent` boosts the agents
        dispatched to in the last few turns; explicit searches turn it off.
        `require_content` keeps only agents whose person/keyword match the message, so an unsent
        draft or reminder alone never makes an agent a search result.
        """

        rows = self._roster.agent_rows()
        if not rows:
            return Shortlist()

        by_name = {row["name"]: row for row in rows}
        query_tokens = tokenize(message)
        query_emails = {e.lower() for e in EMAIL_RE.findall(message or "")}
        query_token_set = set(query_tokens)

        people_rows = self._roster.people_rows()
        drafts = self._roster.open_drafts()
        triggers = self._trigger_counts()

        recent_use = self._recent_use(by_name) if use_recent else {}

        scored: Dict[str, Candidate] = {}
        entity_scores, entity_reasons, entity_emails = self._entity_scores(
            people_rows, query_emails, query_token_set, set(_TOKEN_RE.findall((message or "").lower()))
        )
        keyword_scores = self._keyword_scores(rows, people_rows, query_tokens)
        explicit = bool(entity_scores) or max(keyword_scores.values(), default=0.0) >= STRONG_KEYWORD_SCORE
        context_scale = CONTEXT_SCALE_WITH_CONTENT if explicit else 1.0
        # An open draft breaks ties between agents for the same person, but must not outrank an
        # agent that matches the message's topic ("summarize my newsletters" vs an unsent draft).
        loop_scale = 1.0 if entity_scores else context_scale

        for name, row in by_name.items():
            entity = entity_scores.get(name, 0.0)
            keyword = keyword_scores.get(name, 0.0)
            draft_list = drafts.get(name, [])
            trigger_count = triggers.get(name, 0)
            loop = 1.0 if draft_list else (0.5 if trigger_count else 0.0)
            recency = 0.5 ** (_age_days(row["last_active_at"]) / RECENCY_HALF_LIFE_DAYS)

            used = recent_use.get(name, 0.0)

            score = (
                W_ENTITY * entity
                + W_KEYWORD * keyword
                + loop_scale * W_LOOP * loop
                + context_scale * (W_RECENCY * recency + W_RECENT_USE * used)
            )
            reasons = list(entity_reasons.get(name, []))
            if used:
                reasons.append("used in the last few turns")
            if keyword > 0 and not entity:
                reasons.append("keyword match")
            for draft in draft_list:
                label = draft["recipient"] or "someone"
                subject = f' ("{draft["subject"]}")' if draft["subject"] else ""
                reasons.append(f"unsent draft to {label}{subject}")
            if trigger_count:
                reasons.append(f"{trigger_count} active reminder{'s' if trigger_count != 1 else ''}")

            has_signal = bool(entity or keyword or loop or used)
            has_signal = bool(entity or keyword) if require_content else has_signal
            if has_signal and score >= MIN_SCORE:
                scored[name] = Candidate(
                    name=name,
                    score=score,
                    reasons=reasons,
                    summary=row["summary"],
                    last_active_at=row["last_active_at"],
                    open_drafts=draft_list,
                    active_triggers=trigger_count,
                    content_score=W_ENTITY * entity + W_KEYWORD * keyword,
                    matched_people=entity_emails.get(name, frozenset()),
                )

        all_ranked = sorted(scored.values(), key=lambda c: c.score, reverse=True)
        tied_names = self._tied_people_matches(all_ranked)

        ranked = all_ranked[:limit]
        chosen = {c.name for c in ranked}

        def build(name: str, reason: str) -> Candidate:
            row = by_name[name]
            return Candidate(
                name=name,
                score=0.0,
                reasons=[reason],
                summary=row["summary"],
                last_active_at=row["last_active_at"],
                open_drafts=drafts.get(name, []),
                active_triggers=triggers.get(name, 0),
            )

        forced = [n for n in forced_names if n in by_name]
        for name in forced:
            if name in chosen:
                next(c for c in ranked if c.name == name).reasons.insert(0, "just reported back")
            else:
                ranked.append(build(name, "just reported back"))
                chosen.add(name)

        if recent_pad > 0:
            recent = sorted(rows, key=lambda r: r["last_active_at"], reverse=True)
            for row in recent:
                if recent_pad <= 0:
                    break
                if row["name"] not in chosen:
                    ranked.append(build(row["name"], "recently active"))
                    chosen.add(row["name"])
                    recent_pad -= 1

        return Shortlist(candidates=ranked, matching_total=len(all_ranked), tied_names=tied_names)

    @staticmethod
    def _tied_people_matches(ranked: List[Candidate]) -> List[str]:
        """Agents that match the message's people equally but involve different people.

        "Did Judy reply?" with ten Judys is ambiguous about WHO. The test uses only the
        person/word match, so recent use or an open draft on a few of them does not hide it.
        Several threads with the same person are not ambiguous: context picks the thread.
        """

        by_person = [c for c in ranked if c.matched_people]
        if len(by_person) < 2:
            return []
        top = max(c.content_score for c in by_person)
        tied = [c for c in by_person if c.content_score >= top - TIE_EPSILON]
        if len(tied) < 2 or frozenset.intersection(*(c.matched_people for c in tied)):
            return []
        return sorted(c.name for c in tied)

    def search(self, query: str, limit: int = 8) -> Shortlist:
        """Explicit lookup (find_agents tool and HTTP route): signal-bearing matches only, no padding."""

        return self.retrieve(
            query, limit=limit, recent_pad=0, use_recent=False, require_content=True
        )

    # ------------------------------------------------------------------ signals

    def _recent_use(self, by_name: Dict[str, Any]) -> Dict[str, float]:
        """Weight for each recently dispatched-to agent that still exists, by recency rank."""

        weights: Dict[str, float] = {}
        for name, ts in self._roster.recent_choices(len(RECENT_USE_WEIGHTS) + 2):
            if name not in by_name or _age_days(ts) > RECENT_USE_MAX_AGE_DAYS:
                continue
            if len(weights) >= len(RECENT_USE_WEIGHTS):
                break
            weights[name] = RECENT_USE_WEIGHTS[len(weights)]
        return weights

    @staticmethod
    def _entity_scores(
        people_rows: Iterable,
        query_emails: Set[str],
        query_tokens: Set[str],
        query_words: Set[str] = frozenset(),
    ) -> tuple[Dict[str, float], Dict[str, List[str]], Dict[str, frozenset]]:
        """Score agents by the people in the message: exact email, or a name/handle token."""

        matched: Dict[str, Dict[str, tuple[float, str]]] = {}
        for agent, email, display in people_rows:
            if email in query_emails:
                strength, why = ENTITY_EMAIL, f"mentions {email}"
            else:
                handle = set(tokenize(email.split("@", 1)[0].replace(".", " ").replace("_", " ")))
                name_tokens = set(tokenize(display))
                hits = (handle | name_tokens) & query_tokens
                if not hits:
                    continue
                label = display or email
                # Saying the whole name beats saying one part of it ("dave garcia" vs "dave smith").
                # Short parts ("Jo") are dropped by tokenize, so count them from the raw words.
                parts = set(_TOKEN_RE.findall(display.lower())) or handle
                covered = len(parts & query_words) / len(parts) if parts else 1.0
                strength = ENTITY_NAME * (PARTIAL_NAME_FLOOR + (1.0 - PARTIAL_NAME_FLOOR) * covered)
                why = f"matches {label}"
            matched.setdefault(agent, {})[email] = (strength, why)

        scores: Dict[str, float] = {}
        reasons: Dict[str, List[str]] = {}
        emails: Dict[str, frozenset] = {}
        for agent, people in matched.items():
            ordered = sorted(people.values(), key=lambda sw: sw[0], reverse=True)
            extra = min(ENTITY_EXTRA_CAP, ENTITY_EXTRA_PERSON * (len(ordered) - 1))
            scores[agent] = min(1.0, ordered[0][0] + extra)
            reasons[agent] = list(dict.fromkeys(why for _, why in ordered))[:3]
            emails[agent] = frozenset(people)
        return scores, reasons, emails

    @staticmethod
    def _keyword_scores(
        rows: List[Dict[str, Any]], people_rows: Iterable, query_tokens: List[str]
    ) -> Dict[str, float]:
        """Fraction of the query's (idf-weighted) vocabulary found in each agent's fields."""

        if not query_tokens:
            return {}

        people_text: Dict[str, List[str]] = {}
        for agent, email, display in people_rows:
            people_text.setdefault(agent, []).extend([display, email.split("@", 1)[0]])

        fields: Dict[str, Dict[str, Set[str]]] = {}
        for row in rows:
            name = row["name"]
            fields[name] = {
                "name": {_stem(t) for t in tokenize(name)},
                "people": {_stem(t) for t in tokenize(" ".join(people_text.get(name, [])))},
                "subjects": {_stem(t) for t in tokenize(" ".join(row["subjects"]))},
                "summary": {_stem(t) for t in tokenize(row["summary"])},
            }

        query = list(dict.fromkeys(_stem(t) for t in query_tokens))
        total = len(rows)
        idf: Dict[str, float] = {}
        for token in query:
            df = sum(1 for f in fields.values() if any(token in toks for toks in f.values()))
            idf[token] = math.log(1.0 + total / (1.0 + df))
        max_weight = max(FIELD_WEIGHTS.values())
        denominator = sum(idf[t] * max_weight for t in query) or 1.0

        scores: Dict[str, float] = {}
        for name, per_field in fields.items():
            mass = 0.0
            for token in query:
                best = max(
                    (FIELD_WEIGHTS[fname] for fname, toks in per_field.items() if token in toks),
                    default=0.0,
                )
                mass += idf[token] * best
            if mass > 0:
                scores[name] = mass / denominator
        return scores


_retriever: Optional[AgentRetriever] = None


def get_agent_retriever() -> AgentRetriever:
    global _retriever
    if _retriever is None:
        _retriever = AgentRetriever()
    return _retriever


__all__ = [
    "AgentRetriever",
    "Candidate",
    "RetrievalContext",
    "Shortlist",
    "current_retrieval",
    "get_agent_retriever",
    "humanize_age",
    "reporting_agents",
    "tokenize",
]
