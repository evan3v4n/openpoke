# Evaluation: ranked shortlist vs. whole-roster prompt

Goal of the change: stop sending the interaction agent every execution agent on every turn, so
prompt overhead no longer grows with the roster, without losing the ability to reach the right agent.

Reproduce (no network, no LLM, nothing written to `server/data`):

```bash
python -m server.scripts.eval_roster_overhead            # report
python -m server.scripts.eval_roster_overhead --check    # exit 1 if a quality gate fails
python -m server.scripts.eval_roster_overhead --json out.json
```

Results below are from a seeded run (`--seed 11`, 40 queries per kind per size). Cost and quality
numbers repeat exactly; latency varies by machine. Tokens are estimated as characters / 4 (install
`tiktoken` for the real tokenizer).

## 1. Cost: tokens in the `<active_agents>` section, per turn

| Agents | Old (all names) | New (shortlist, mean) | Reduction | Retrieval p50 |
|---:|---:|---:|---:|---:|
| 5 | 49 | 151 | -208% | 3.5 ms |
| 10 | 98 | 158 | -61% | 3.5 ms |
| 20 | 206 | 172 | 16% | 4.0 ms |
| 35 | 355 | 192 | 46% | 4.4 ms |
| 100 | 1,024 | 252 | 75% | 6.0 ms |
| 250 | 2,592 | 315 | 88% | 9.6 ms |
| 1,000 | 10,431 | 349 | **96.7%** | 28.5 ms |

The old section grows linearly with the roster; the new one grows slowly. **Break-even is about 20
agents**: below that the new section is larger (summaries and reasons cost more than bare names).

## 2. Quality: is the right agent still reachable?

The old prompt shows every agent, so its recall is 100% by construction; its price is the table
above. For the new prompt, recall is "the target agent is in the shortlist" (a miss costs a
`find_agents` call or a duplicate agent, which can cancel the token savings).

| Query type | Recall at 5 to 1,000 agents |
|---|---|
| Full name, email, topic, "yes, send it" | 100% at every size |
| Topic with filler words ("Any update on the X thing?") | 100% (was 20% at 1,000 before the fix below) |
| Name shared by several agents flagged as a tie (ask, don't fan out) | 100% |
| Request about nobody on the roster gets no false agent match | 100% |
| One typo in a topic word | 100% (was 17.5% at 1,000 before the fix) |
| One typo in a name (5+ letters) | 100% up to 250 agents, 86% at 1,000 (was 21%) |
| One typo in a 4-letter name | 100% up to 250 agents, 62% at 1,000 (not corrected on purpose) |

Last-name-only and ambiguous-topic queries are reported but not gated: dozens of agents are valid
targets (about 17 and 50 at 1,000 agents), so a 6-slot shortlist cannot hold them and the right
behaviour is to ask.

## 3. Defects the evaluation found and this PR fixes

1. **Filler words hid real matches.** Words that match nothing on the roster ("any", "update",
   "thing") counted in the keyword score's denominator and pushed the right agent under `MIN_SCORE`.
   Fix: only words found somewhere on the roster count.
2. **No tolerance for typos.** A misspelled name or topic word matched nothing. Fix: query words of
   5+ letters that match nothing are mapped to roster words one edit away (same first letter, at most
   two candidates per word). Cost: about 4 to 8 ms more at 1,000 agents.

Unit tests for both are in `server/tests/test_retriever.py`.

## 4. What this does not show (read before relying on the numbers)

- **Synthetic data.** Names and topics are unique and the queries are templated from the seeded
  strings, so the quality figures are best case, not a prediction of production accuracy. The two
  defects above were found and fixed against this same generator, so a held-out set could score lower.
- **Shortlist, not end-to-end.** The harness grades what the retriever shows, not whether the model
  then picks the right agent, nor execution-agent LLM calls. The execution-side claims supported here
  are fewer wrong dispatches (the ambiguity handling) and the early exit when Gmail is not connected;
  the prompt-size savings are on the interaction agent.
- **Small rosters lose.** Below about 20 agents the new section costs more tokens.

Next step for stronger evidence: replay anonymized real user messages, each labelled with the agent
actually used, against a roster snapshot; hold part of the set out; then run a smaller live LLM pass
for routing accuracy, duplicate-agent rate and LLM calls per task.
