# Agent Overload

OpenPoke passed the **entire roster of execution agents** to the interaction agent on every turn. After days of use that is hundreds of agents ("Email to Alice", "Q3 Budget Analysis", "Tokyo Restaurant Search"…) competing for the model's attention, and routing gets worse as the list grows.

This change replaces the roster with a **short, ranked, explained shortlist** per message, adds a search tool for everything else, and makes the assistant **ask instead of guess** when a request fits several agents or drafts.

## What it does

- **Index what each agent has dealt with.** The roster moves from `roster.json` (names only) to SQLite: per agent, a summary, recent subjects, the people involved (email, name, role, count), thread ids and open drafts. It is filled automatically from execution-agent tool calls. Existing `roster.json` is imported once.
- **Rank agents against each message.** Plain Python scoring, about 30 ms at 1,000 agents:

  ```
  score = 1.0·entity + 0.6·keyword + 0.35·open_loop + 0.25·recency + 0.5·recent_use
  ```

  | Signal | Meaning |
  |---|---|
  | entity | The message names a person or email this agent has dealt with (whole names beat partial ones) |
  | keyword | Rarity-weighted overlap with the agent's name, people, subjects and summary; filler words and one-letter typos tolerated |
  | open_loop | The agent holds an unsent draft or an active reminder ("send it") |
  | recent_use | The agents used in the last few turns ("check again") |
  | recency | Last activity, a weak tie-break |

  When the message clearly names something, recent use and recency are scaled down so they cannot override it.
- **Show the model a shortlist, not the roster.** `<active_agents>` lists about 6 agents, each with *why* it was picked and a summary, plus how many others exist. A `find_agents(query)` tool searches the rest before a new agent is created.
- **Ask when it is unclear.** Code detects ties and adds `<ambiguity>` notes: several agents for one person ("Did Judy reply?" with three Judys) or one topic (Q3 and Q4 budget), which draft "send it" means, weekdays with two readings, and an unknown sending mailbox.
- **Never send the wrong email.** Each user message records which drafts it clearly refers to; the execution runtime refuses any other send, so a model that ignores the note still cannot send the wrong draft.
- **Show it working.** The chat UI shows a live card per running agent (thinking, tool calls) and a panel of unsent drafts and upcoming reminders.

Code: ranking in [`server/services/execution/retriever.py`](server/services/execution/retriever.py), index in [`roster.py`](server/services/execution/roster.py), ambiguity and the send check in [`ambiguity.py`](server/services/execution/ambiguity.py), prompt rendering in [`interaction_agent/agent.py`](server/agents/interaction_agent/agent.py).

## Assumptions

- **Requests are mostly about people and threads.** "Did Alice reply?" needs an exact hit on Alice, so entity matching leads and embeddings were left out (they blur Judy Brown and Judy Chen).
- **One user, one machine.** SQLite and an in-process scan are enough; nothing new to deploy.
- **Rosters reach hundreds to low thousands of agents.** A linear scan is fine there; beyond ~10k I would prefilter with full-text search.
- **A wrong send is worse than an extra question.** When unsure which draft is meant, the send is held and the user is asked.
- **Synthetic eval data is best case.** Real phrasing will be messier; the live run below is the reality check.

## How to run

Set up as in the main [README](README.md) (`.env`, virtualenv, `npm install --prefix web`), then:

```bash
python -m server.server            # API on :8001
npm run dev --prefix web           # UI on :3000, then Settings → connect Gmail
```

Gmail must be reconnected in Settings after each server restart (the account id is only held in memory).

Tests and evaluation (no network, no LLM calls, nothing written to `server/data`):

```bash
python -m unittest discover -s server/tests -t .         # 106 tests
python -m server.scripts.eval_roster_overhead --check    # cost and quality by roster size
```

Inspect the live roster (local requests only): `/api/v1/agents`, `/api/v1/agents/search?q=sanya`, `/api/v1/agents/retrieval-stats`.

## Results

**Offline eval** (seeded synthetic rosters, 5 to 1,000 agents; full report in [docs/EVALUATION.md](docs/EVALUATION.md)):

| | 20 agents | 100 agents | 1,000 agents |
|---|---:|---:|---:|
| Roster tokens per turn, old → new | 206 → 177 | 1,024 → 264 | 10,431 → 367 (**−96.5%**) |
| Right agent in shortlist (name, email, topic, follow-up) | 100% | 100% | 100% |
| Shared name / shared topic flagged as a tie | 100% / 100% | 100% / 100% | 100% / 100% |
| Asked when there was one right answer | 0% | 0% | 0% |

Below about 20 agents the shortlist costs more tokens than the old list (summaries cost more than bare names).

**Live run** (20 scripted messages, real model and Gmail, fresh roster): 12 agents created, **no duplicates**; all 7 reuses came from the shortlist; **no wrong-agent dispatches and no wrong emails sent**; a bare "send it" after the discussed draft went out now asks which draft. The send check is strict: twice it held a confirmed send until the user named the draft.

## What I tried

- **SQLite FTS5 / BM25** for keyword scoring: dropped, scores collapse towards zero on small rosters.
- **Context outranking content**: "update on the Q3 budget" lost to whichever agent was used last, and an unsent draft pulled unrelated requests. Fixed by scaling context down whenever the message names something.
- **The eval found two misses**, both fixed: filler words ("any update on the X thing") diluted real matches (recall 20% → 100% at 1,000 agents), and typos matched nothing.
- **A live test sent the wrong email**: two drafts open, the user confirmed one, then said "send it" again and the *other* draft went to a different person. That led to the ambiguity notes and to checking sends in code rather than trusting the prompt.
- **Topic ties were flagged only 33–95% of the time** because an agent named after a topic outscored ones with the topic in their subjects. Fixed by comparing *which* words each agent matched.

## What I'd do next

1. **Replay real messages**, each labelled with the agent actually used, and hold part of the set out; then a live pass measuring routing accuracy, duplicate-agent rate and LLM calls per task.
2. **Learn the weights** from the logged choices (`retrieval_choices` records what was shown and what was used).
3. **Add semantic similarity as one more signal** so "dinner spots in Japan" finds "Tokyo Restaurant Search"; keep entity matching for people.
4. **Archive or merge dormant agents**, and cap each agent's own history, which is the same overload one level down.
5. **Show the full list below ~20 agents**, where it is cheaper.

## Other gaps found along the way

- The default model (`anthropic/claude-sonnet-4`) was retired on OpenRouter, so every call failed with a hidden error. Fixed, with per-agent model overrides.
- Composio retired the Gmail connect endpoint used here. Fixed.
- With Gmail disconnected, execution agents kept calling the LLM to retry an error that cannot change. They now stop at once, and the assistant tells the user to connect Gmail.
- A hard-coded 90 s timeout cancelled email searches before they could report "nothing found". Now configurable, default 240 s.
- Not fixed: an email search can load 50 full emails into one call and exceed the model's context (seen live; the agent recovered); the user's own name and other addresses are indexed as contacts; drafts the assistant shows in chat are not Gmail drafts until sent.
