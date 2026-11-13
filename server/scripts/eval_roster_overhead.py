"""Offline A/B eval: whole-roster prompt (old) vs ranked shortlist (new), by roster size.

    python -m server.scripts.eval_roster_overhead
    python -m server.scripts.eval_roster_overhead --sizes 10,100,1000 --json out.json
    python -m server.scripts.eval_roster_overhead --check     # exit 1 if a quality gate fails

No network, no LLM, and nothing is written to server/data (a temporary roster is used).
Everything is seeded, so two runs give the same cost and quality numbers (latency varies).

What it measures
  Cost     interaction-agent tokens spent on the <active_agents> section per turn, old vs new,
           and the roster size where the new section starts to win (break-even).
  Latency  time to build the section (retrieval + logging) per turn.
  Quality  each query has a known target, so the shortlist can be graded:
             recall      the target agent is in the shortlist (a miss costs a find_agents round
                         trip or a duplicate agent, which can cancel out the token savings)
             top1        the target is ranked first
             ambiguity   a name shared by several agents is reported as a tie (the model asks
                         instead of fanning out), and the shortlist still contains the targets
             new_task    a request about nobody on the roster gets no person/keyword match
                         (otherwise the model is nudged to reuse the wrong agent)

What it does NOT measure: whether the model then picks the right agent from the shortlist, or
execution-agent LLM calls. Those need live LLM runs; this harness grades the part the PR changed.
The old prompt shows every agent, so its recall is 100% by construction; its cost is the
price of that. Token counts use tiktoken (cl100k) when installed, else characters / 4.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from html import escape
from pathlib import Path
from typing import Any, Callable, Dict, List
from unittest import mock

from server.agents.interaction_agent import agent as interaction_agent
from server.services.execution.people import PersonRef
from server.services.execution.retriever import AgentRetriever
from server.services.execution.roster import AgentRoster

FIRST = """Sarah Michael David Emma Judy Aaron Bianca Callum Diya Ezra Farah Grace Hiro Ines Jonas Kavya Liam Mona
Nadia Omar Priya Quinn Rafael Sofia Tariq Uma Victor Wen Ximena Yusuf Zoe Alma Boris Carla Dmitri Elif Felix Gita
Hana Ivo Jun Karim Lena Marco Noor Olga Pablo Rhea Sven Tess Ugo Vera""".split()
LAST = """Lee Cohen Okafor Mitchell Tanaka Dubois Novak Reyes Chen Rossi Adeyemi Walsh Petrov Haddad Kim Park Singh
Larsen Moreau Ortiz Brandt Watson Clark Ivanova Santos Fischer Patel Nguyen Garcia Almeida Bauer Costa Dang Eze
Fox Gupta Holm Ito Jain Koh Lund Mori Nair Ozturk Pham Qureshi Rao Sato Thapa Usman Vega Wolf Yamada Zhou""".split()
DOMAINS = ["gmail.com", "outlook.com", "acme.io", "proton.me", "work.co"]
TOPICS = """Quarterly Review|Wedding RSVP|Apartment Sublet|Project Handoff|Conference Talk|Tax Documents
Birthday Dinner|Onboarding|Offsite Planning|Expense Report|Reference Letter|Lease Renewal|Visa Application
Contract Redline|Podcast Booking|Grant Proposal|Moving Quote|Insurance Claim|Hackathon Prize|Roadmap Sync""".replace(
    "\n", "|"
).split("|")

NEW_TASK_QUERIES = [
    "Book me a table for two on Friday evening",
    "What's a good way to stay focused while studying?",
    "Remind me to stretch every hour",
    "Translate this sentence into Spanish for me",
]


def mutate(word: str, rng: random.Random) -> str:
    """One realistic typo: swap two neighbouring letters, drop a letter, or double one."""

    if len(word) < 4:
        return word
    for _ in range(10):
        i = rng.randrange(1, len(word) - 1)
        op = rng.choice(("swap", "drop", "double"))
        if op == "swap":
            out = word[:i] + word[i + 1] + word[i] + word[i + 2 :]
        elif op == "drop":
            out = word[:i] + word[i + 1 :]
        else:
            out = word[:i] + word[i] + word[i:]
        if out.lower() != word.lower():
            return out
    return word


def count_tokens(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, round(len(text) / 4))


def token_method() -> str:
    try:
        import tiktoken  # noqa: F401

        return "tiktoken cl100k"
    except Exception:
        return "chars/4 estimate"


def old_section(names: List[str]) -> str:
    """The pre-PR rendering: every agent by name, every turn."""

    return "\n".join(f'<agent name="{escape(n or "agent", quote=True)}" />' for n in names) or "None"


def build_roster(roster: AgentRoster, n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Seed `n` agents (mostly one per person, some topic-only) and return their ground truth."""

    combos = [(f, l) for f in FIRST for l in LAST]
    rng.shuffle(combos)
    truth: List[Dict[str, Any]] = []
    for i in range(n):
        topic = f"{rng.choice(TOPICS)} {1000 + i}"  # unique per agent so topic queries have one answer
        if i % 8 == 7:  # topic-only agent, no person
            name = f"{topic} helper"
            roster.add_agent(name)
            roster.record_activity(name, subjects=[topic])
            summary = f"Handled the {topic.lower()} task and reported back."
            truth.append({"name": name, "topic": topic, "person": None})
        else:
            first, last = combos[i % len(combos)]
            email = f"{first}.{last}@{rng.choice(DOMAINS)}".lower()
            name = f"Email to {first} {last} {i}"
            roster.add_agent(name)
            roster.record_activity(
                name, [PersonRef(email=email, name=f"{first} {last}", role="recipient")], [f"t{i}"], [topic]
            )
            summary = f"Asked {first} about {topic.lower()}; waiting for a reply."
            truth.append({"name": name, "topic": topic, "person": (first, last, email)})
        roster.set_summary(name, summary)
    # Real rosters are not all fresh: spread last activity over ~60 days.
    with roster._tx() as conn:  # noqa: SLF001 - eval tooling
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        for row in truth:
            ts = (now - timedelta(days=rng.random() * 60)).isoformat()
            conn.execute("UPDATE agents SET last_active_at = ? WHERE name = ?", (ts, row["name"]))
    return truth


def _render(message: str) -> tuple[str, float]:
    start = time.perf_counter()
    section = interaction_agent._render_active_agents(message, "user")  # noqa: SLF001
    return section, (time.perf_counter() - start) * 1000.0


def _names_in(section: str) -> List[str]:
    import re
    from html import unescape

    return [unescape(m) for m in re.findall(r'<agent name="([^"]*)"', section)]


def run_size(n: int, seed: int, queries_per_kind: int) -> Dict[str, Any]:
    rng = random.Random(seed + n)
    with tempfile.TemporaryDirectory() as tmp:
        roster = AgentRoster(Path(tmp) / "roster.db")
        retriever = AgentRetriever(roster, trigger_counts=lambda: {})
        truth = build_roster(roster, n, rng)
        people = [t for t in truth if t["person"]]
        topical = list(truth)

        by_first: Dict[str, List[str]] = {}
        for t in people:
            by_first.setdefault(t["person"][0], []).append(t["name"])

        def sample(pool: List[Any]) -> List[Any]:
            return rng.sample(pool, min(queries_per_kind, len(pool)))

        # (kind, message, set of acceptable target names)
        cases: List[tuple] = []
        for t in sample(people):
            first, last, email = t["person"]
            cases.append(("full_name", f"Did {first} {last} reply yet?", {t["name"]}))
            cases.append(("email", f"Follow up with {email} about this", {t["name"]}))
        for t in sample(topical):
            cases.append(("topic", f"What's the status of {t['topic']}?", {t["name"]}))
            # Same single target, padded with words that match nothing on the roster.
            cases.append(("topic_filler", f"Any update on the {t['topic'].lower()} thing?", {t["name"]}))
        for first, names in sample(sorted(by_first.items())):
            cases.append(("first_name", f"Did {first} reply?", set(names)))
        # Misspellings: one typo in a content word. Same single target as the clean query.
        for t in sample(people):
            first, last, _ = t["person"]
            long_words = [w for w in (first, last) if len(w) >= 5]
            short_words = [w for w in (first, last) if len(w) == 4]
            for kind, pool in (("typo_name", long_words), ("typo_name_4letter", short_words)):
                if pool:
                    word = rng.choice(pool)
                    msg = f"Did {first} {last} reply yet?".replace(word, mutate(word, rng), 1)
                    cases.append((kind, msg, {t["name"]}))
        for t in sample(topical):
            base, num = t["topic"].rsplit(" ", 1)
            words = base.split()
            j = rng.randrange(len(words))
            words[j] = mutate(words[j], rng)
            cases.append(("typo_topic", f"What's the status of {' '.join(words)} {num}?", {t["name"]}))
            # No id, so every agent on the same topic is a valid answer (grade "any shown").
            same_topic = {x["name"] for x in topical if x["topic"].rsplit(" ", 1)[0] == base}
            cases.append(("typo_topic_no_id", f"Any update on the {' '.join(words).lower()} thing?", same_topic))

        # Hard cases: several agents are valid targets, so a 6-slot shortlist cannot hold them all.
        # Not gated; reported as "any valid target shown" so the ceiling is visible.
        by_last: Dict[str, List[str]] = {}
        by_base: Dict[str, List[str]] = {}
        for t in people:
            by_last.setdefault(t["person"][1], []).append(t["name"])
        for t in topical:
            by_base.setdefault(t["topic"].rsplit(" ", 1)[0], []).append(t["name"])
        for last, names in sample(sorted(by_last.items())):
            cases.append(("hard_last_name", f"Did {last} reply?", set(names)))
        for base, names in sample(sorted(by_base.items())):
            cases.append(("hard_ambiguous_topic", f"Any update on the {base.lower()} thing?", set(names)))
        for q in NEW_TASK_QUERIES:
            cases.append(("new_task", q, set()))

        patches = [
            mock.patch.object(interaction_agent, "get_agent_roster", lambda: roster),
            mock.patch.object(interaction_agent, "get_agent_retriever", lambda: retriever),
        ]
        for p in patches:
            p.start()
        try:
            old_text = old_section(roster.get_agents())
            old_tokens = count_tokens(old_text)

            kinds: Dict[str, Dict[str, Any]] = {}
            latencies: List[float] = []
            new_tokens: List[int] = []

            def record(kind: str, **fields: Any) -> None:
                bucket = kinds.setdefault(kind, {"n": 0})
                bucket["n"] += 1
                for key, value in fields.items():
                    bucket[key] = bucket.get(key, 0) + int(value)

            for kind, message, targets in cases:
                section, ms = _render(message)
                latencies.append(ms)
                new_tokens.append(count_tokens(section))
                shown = _names_in(section)
                if kind == "new_task":
                    record(kind, clean=not any(w in section for w in ('why="matches', 'why="mentions')))
                elif kind.startswith("typo_"):
                    record(kind, recall=bool(targets & set(shown)), top1=bool(shown and shown[0] in targets))
                elif kind.startswith("hard_"):
                    record(kind, recall=bool(targets & set(shown)), targets=len(targets),
                           flagged=len(targets) > 1 and "<ambiguity" in section, multi=len(targets) > 1)
                elif kind == "first_name":
                    tied = len(targets) > 1
                    record(
                        kind,
                        recall=bool(targets & set(shown)) if not tied else len(targets & set(shown)) >= min(len(targets), 2),
                        ambiguity_expected=tied,
                        ambiguity_flagged=tied and "<ambiguity" in section,
                    )
                else:
                    # One right answer: an <ambiguity> note here would make the model ask needlessly.
                    record(kind, recall=bool(targets & set(shown)), top1=bool(shown and shown[0] in targets),
                           overask="<ambiguity" in section)

            # Follow-ups with no content ("yes, send it"): target is the agent with the open draft.
            follow_up_ok = 0
            follow_ups = sample(topical)
            for t in follow_ups:
                roster.open_draft(t["name"], f"d-{t['name']}", "someone@example.com", t["topic"])
                section, ms = _render("yes, send it")
                latencies.append(ms)
                new_tokens.append(count_tokens(section))
                follow_up_ok += t["name"] in _names_in(section)
                roster.close_draft(f"d-{t['name']}", t["name"])
            if follow_ups:
                kinds["follow_up"] = {"n": len(follow_ups), "recall": follow_up_ok, "top1": 0}
        finally:
            for p in patches:
                p.stop()

    def rate(kind: str, key: str) -> float | None:
        b = kinds.get(kind)
        if not b or b["n"] == 0 or key not in b:
            return None
        return b[key] / b["n"]

    ambiguity_expected = kinds.get("first_name", {}).get("ambiguity_expected", 0)
    ambiguity_flagged = kinds.get("first_name", {}).get("ambiguity_flagged", 0)
    unambiguous = ["full_name", "email", "topic", "topic_filler", "follow_up"]
    unamb_hits = sum(kinds.get(k, {}).get("recall", 0) for k in unambiguous)
    unamb_total = sum(kinds.get(k, {}).get("n", 0) for k in unambiguous)
    overask_hits = sum(kinds.get(k, {}).get("overask", 0) for k in ("full_name", "email", "topic", "topic_filler"))
    overask_total = sum(kinds.get(k, {}).get("n", 0) for k in ("full_name", "email", "topic", "topic_filler"))
    topic_tie = kinds.get("hard_ambiguous_topic", {})
    mean_new = statistics.mean(new_tokens) if new_tokens else 0.0
    lat = sorted(latencies)
    return {
        "agents": n,
        "old_tokens": old_tokens,
        "new_tokens_mean": round(mean_new, 1),
        "new_tokens_max": max(new_tokens, default=0),
        "token_reduction_pct": round(100.0 * (1 - mean_new / old_tokens), 1) if old_tokens else 0.0,
        "latency_ms_p50": round(lat[len(lat) // 2], 2) if lat else 0.0,
        "latency_ms_p95": round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 2) if lat else 0.0,
        "recall_unambiguous": round(unamb_hits / unamb_total, 4) if unamb_total else None,
        "recall_by_kind": {k: rate(k, "recall") for k in ("full_name", "email", "topic", "topic_filler", "follow_up", "first_name")},
        "top1_by_kind": {k: rate(k, "top1") for k in ("full_name", "email", "topic", "topic_filler")},
        "ambiguity_flag_rate": round(ambiguity_flagged / ambiguity_expected, 4) if ambiguity_expected else None,
        "ambiguity_cases": ambiguity_expected,
        # Asking when there is one right answer (should be ~0) vs. flagging a shared topic (should be high).
        "overask_rate": round(overask_hits / overask_total, 4) if overask_total else None,
        "topic_tie_flag_rate": round(topic_tie["flagged"] / topic_tie["multi"], 4) if topic_tie.get("multi") else None,
        "new_task_clean_rate": rate("new_task", "clean"),
        "typo": {
            k: {"recall": rate(k, "recall"), "top1": rate(k, "top1")}
            for k in ("typo_name", "typo_name_4letter", "typo_topic", "typo_topic_no_id")
            if k in kinds
        },
        "hard": {
            k: {
                "any_target_shown": rate(k, "recall"),
                "mean_valid_targets": round(kinds[k]["targets"] / kinds[k]["n"], 1),
            }
            for k in ("hard_last_name", "hard_ambiguous_topic")
            if k in kinds
        },
        "queries": sum(b["n"] for b in kinds.values()),
    }


def fmt(value: Any, pct: bool = False) -> str:
    if value is None:
        return "  -  "
    return f"{value * 100:5.1f}%" if pct else str(value)


def print_report(results: List[Dict[str, Any]], method: str) -> None:
    print(f"\nTokens: {method}. Old = every agent name per turn. New = ranked shortlist (mean over queries).\n")
    head = f"{'agents':>6} {'old tok':>8} {'new tok':>8} {'saved':>7} {'p50 ms':>7} {'p95 ms':>7} {'recall*':>8} {'full':>6} {'email':>6} {'topic':>6} {'filler':>6} {'follow':>6} {'amb':>6} {'newtask':>8}"
    print(head)
    print("-" * len(head))
    for r in results:
        rk = r["recall_by_kind"]
        print(
            f"{r['agents']:>6} {r['old_tokens']:>8} {r['new_tokens_mean']:>8} {r['token_reduction_pct']:>6}% "
            f"{r['latency_ms_p50']:>7} {r['latency_ms_p95']:>7} {fmt(r['recall_unambiguous'], True):>8} "
            f"{fmt(rk['full_name'], True):>6} {fmt(rk['email'], True):>6} {fmt(rk['topic'], True):>6} "
            f"{fmt(rk['topic_filler'], True):>6} {fmt(rk['follow_up'], True):>6} {fmt(r['ambiguity_flag_rate'], True):>6} {fmt(r['new_task_clean_rate'], True):>8}"
        )
    print("\n* recall over full-name, email, topic, topic+filler and follow-up queries: the target is in the shortlist.")
    print("  amb = share of shared-first-name queries reported as a tie; newtask = no false person/keyword match.")
    print("\nAsking: over-ask = <ambiguity> on a query with one right answer (lower is better);")
    print("        topic tie = <ambiguity> on a topic shared by several agents (higher is better).")
    for r in results:
        print(f"  {r['agents']:>5} agents  over-ask: {fmt(r['overask_rate'], True).strip()}   topic tie: {fmt(r['topic_tie_flag_rate'], True).strip()}")
    print("\nHard cases (not gated): any valid target shown / mean number of valid targets.")
    for r in results:
        h = r["hard"]
        parts = [f"{k[5:]}: {fmt(v['any_target_shown'], True).strip()} of ~{v['mean_valid_targets']}" for k, v in h.items()]
        print(f"  {r['agents']:>5} agents  " + "   ".join(parts))
    print("\nMisspellings (one typo in a word): target in shortlist / ranked first. 4letter = short names (not corrected by design); no_id = any agent on that topic.")
    for r in results:
        parts = [f"{k[5:]}: {fmt(v['recall'], True).strip()}/{fmt(v['top1'], True).strip()}" for k, v in r["typo"].items()]
        print(f"  {r['agents']:>5} agents  " + "   ".join(parts))
    wins = [r["agents"] for r in results if r["new_tokens_mean"] < r["old_tokens"]]
    loses = [r["agents"] for r in results if r["new_tokens_mean"] >= r["old_tokens"]]
    if loses:
        print(f"  New section is NOT smaller at: {loses}. Smaller from {min(wins) if wins else 'n/a'} agents up.")


GATES: Dict[str, Callable[[Dict[str, Any]], bool]] = {}


def check(
    results: List[Dict[str, Any]], min_recall: float, min_ambiguity: float, min_clean: float, max_overask: float = 0.05
) -> List[str]:
    failures: List[str] = []
    for r in results:
        n = r["agents"]
        if r["recall_unambiguous"] is not None and r["recall_unambiguous"] < min_recall:
            failures.append(f"{n} agents: unambiguous recall {r['recall_unambiguous']:.3f} < {min_recall}")
        if r["ambiguity_flag_rate"] is not None and r["ambiguity_flag_rate"] < min_ambiguity:
            failures.append(f"{n} agents: ambiguity flag rate {r['ambiguity_flag_rate']:.3f} < {min_ambiguity}")
        if r["new_task_clean_rate"] is not None and r["new_task_clean_rate"] < min_clean:
            failures.append(f"{n} agents: new-task clean rate {r['new_task_clean_rate']:.3f} < {min_clean}")
        if r["overask_rate"] is not None and r["overask_rate"] > max_overask:
            failures.append(f"{n} agents: over-ask rate {r['overask_rate']:.3f} > {max_overask}")
    return failures


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="5,10,20,35,50,100,250,1000", help="comma-separated roster sizes")
    ap.add_argument("--queries", type=int, default=40, help="queries per kind per size")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--json", help="write full results to this path")
    ap.add_argument("--check", action="store_true", help="exit 1 if a quality gate fails")
    ap.add_argument("--min-recall", type=float, default=0.95)
    ap.add_argument("--min-ambiguity", type=float, default=0.9)
    ap.add_argument("--min-clean", type=float, default=0.75)
    ap.add_argument("--max-overask", type=float, default=0.05)
    args = ap.parse_args(argv)

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    results = [run_size(n, args.seed, args.queries) for n in sizes]
    print_report(results, token_method())

    if args.json:
        Path(args.json).write_text(
            json.dumps({"token_method": token_method(), "seed": args.seed, "results": results}, indent=2)
        )
        print(f"\nwrote {args.json}")

    if args.check:
        failures = check(results, args.min_recall, args.min_ambiguity, args.min_clean, args.max_overask)
        for f in failures:
            print(f"GATE FAILED: {f}")
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
