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

## J3: the monitor at threshold

The builder for [`docs/JEV_J3_PROTOCOL.md`](../docs/JEV_J3_PROTOCOL.md). Jev reads the
program's 14-coordinate register of recorded Qwen states (E7b-Q, layer 14), pushed along
13 known directions or at random, in two presentations (RAW, DIGEST), against a fitted
classifier. A small language model reads the DIGEST page in words as a comparison (the mouth).

| file | what it does |
|---|---|
| `j3.py` | recorded states, halves, reference cards, stimuli, the two renderers, text-only parsers, the fitted bar |
| `j3_plan.py` | smoke, flight (bundled or separate `pushed` noul) and mouth plans; plan shas |
| `j3_mouth.py` | the chat-completions client for the mouth, on J1 + J2's `Runner` |
| `analyze_j3.py` | gates, P1–P3 per presentation, secondaries, the mouth, the J4 fork |
| `agents_j3.py` | planted agents: twin, degraded, claimer, top-looker, mute |
| `power/j3_design_check.py`, `power/j3_power.py` | the design and power checks cited in the protocol |
| `tests/test_j3_*.py` | logic (incl. gate G6'), capture, verdict suites |

```sh
PY=~/venvs/semcore/bin/python
$PY -m pytest jev/tests/test_j3_logic.py jev/tests/test_j3_capture.py jev/tests/test_j3_verdict.py -q
$PY -m jev.run_j3 plan                                   # counts and shas; no calls
$PY -m jev.run_j3 freeze                                 # FROZEN_J3.json; no calls
$PY -m jev.run_j3 smoke  --frozen jev/results/j3_freeze_<stamp>/FROZEN_J3.json
$PY -m jev.run_j3 flight --frozen <F> --smoke jev/results/j3_smoke_<stamp>
$PY -m jev.run_j3 mouth  --frozen <F> --smoke jev/results/j3_smoke_<stamp>
$PY -m jev.run_j3 analyze --flight jev/results/j3_flight_<stamp> --frozen <F> --smoke <S> --mouth jev/results/j3_mouth_<stamp>
```

G7 for J3 is its own: every `jev/results/j3_*` stage counts toward $5.
