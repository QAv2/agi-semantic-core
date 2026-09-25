# jev/ — J1 + J2 on Jev

The builder for the pre-registered experiments in
[`docs/JEV_J1J2_PROTOCOL.md`](../docs/JEV_J1J2_PROTOCOL.md). Jev
(TypeSafe AI, `jev-1.13`) is called through OpenRouter's System One endpoint.
Every stimulus is generated from seeds, and every raw response is kept.

| file | what it does |
|---|---|
| `common.py` | seeds, neutral IDs, balanced option orders, request bodies and their sha256 |
| `j1.py` | J1 families F1/F2/F3, renderings A0/A1/A2/A3, ideal observers, the text-only parser behind gate G6 |
| `j2.py` | J2 templates, evidence ladders, the eight arms |
| `plan.py` | call plans for the smoke test, the titration pilot and the flight; plan hashes |
| `runner.py` | OpenRouter client; resumable, budget-capped runner (gates G1, G2, G7) |
| `analyze_smoke.py`, `titrate.py` | replicate floor, throughput, question isolation; psychometric fits and `FROZEN.json` |
| `analyze_j1.py`, `analyze_j2.py`, `stats.py` | gates, primaries, secondaries |
| `agents.py` | planted agents with known structure, for the verdict suite |
| `power/` | the power checks cited in protocol §9, with their outputs |
| `tests/` | logic, capture and verdict suites |

```sh
PY=~/venvs/semcore/bin/python
$PY -m pytest jev/tests -q                                  # all three suites, no network
$PY -m jev.run plan --stage smoke                           # counts and plan sha; no calls
$PY -m jev.run smoke                                        # needs OPENROUTER_API_KEY
$PY -m jev.run pilot --smoke jev/results/smoke_<stamp>
$PY -m jev.run flight --frozen jev/results/pilot_<stamp>/FROZEN.json
$PY -m jev.run analyze --flight jev/results/flight_<stamp> --frozen jev/results/pilot_<stamp>/FROZEN.json
```

Each calling stage writes `plan.jsonl.gz` (every call without its body, plus the
metadata the analysis needs), `plan_sha.txt`, and `raw.jsonl.gz` (every response
verbatim, with the sha256 of its request). Request bodies aren't stored. They
are regenerated from the seeds, and `analyze` checks that every one hashes to its
recorded sha (gate G4). The API key is read from the environment and never
written anywhere.
