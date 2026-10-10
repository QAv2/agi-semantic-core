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

## J6: self-description under the input-only null

The builder for [`docs/JEV_J6_PROTOCOL.md`](../docs/JEV_J6_PROTOCOL.md). One Choice
question, "which of these best describes you?", over 6 page states × 3 question framings ×
24 matched items × 8 option rotations (3,456 calls). Primaries: question echo,
self-statement uptake, self-specificity, negation.

| file | what it does |
|---|---|
| `j6.py` | options, questions, page templates, items, neutral keys, rotations, the T/C/D measures |
| `j6_plan.py` | smoke and flight plans; plan shas |
| `analyze_j6.py` | gates, P1–P4 with their branches, the baseline self-label, secondaries S1–S10 |
| `agents_j6.py` | planted agents: echoer, reader, fixed, door, positional |
| `run_j6.py` | plan / freeze / smoke / flight / analyze |
| `tests/test_j6_*.py` | logic, capture, verdict suites |

```sh
PY=~/venvs/semcore/bin/python
$PY -m pytest jev/tests/test_j6_logic.py jev/tests/test_j6_capture.py jev/tests/test_j6_verdict.py -q
$PY -m jev.run_j6 plan                                   # counts and shas; no calls
$PY -m jev.run_j6 freeze                                 # FROZEN_J6.json; no calls
$PY -m jev.run_j6 smoke  --frozen jev/results/j6_freeze_<stamp>/FROZEN_J6.json
$PY -m jev.run_j6 flight --frozen <F> --smoke jev/results/j6_smoke_<stamp>
$PY -m jev.run_j6 analyze --flight jev/results/j6_flight_<stamp> --frozen <F> --smoke <S>
```

G7 for J6 is its own: every `jev/results/j6_*` stage counts toward $1.

## J6b: does Jev know it's Jev?

The builder for [`docs/JEV_J6B_PROTOCOL.md`](../docs/JEV_J6B_PROTOCOL.md). J6's question and
options, with the pointer taken away: labels reach Jev through its name alone (Q1) or a
nameless description (Q2), each against a matched other-system control (the specificity
clauses); J6's pointed sentence is the positive control (Q3); Q4 is the pointer's
advantage over the name. 11 page states × 24 items × 8 rotations (2,112 calls).

| file | what it does |
|---|---|
| `j6b.py` | reference modes, pages, items, keys, rotations (J6's options and measures) |
| `j6b_plan.py` | smoke and flight plans; plan shas |
| `analyze_j6b.py` | gates, Q1–Q4, specificity clauses, the pointer-check gate, secondaries R1–R9 (J6's tests and branches) |
| `agents_j6b.py` | planted agents: namer, pointer, anylabel, contraster, fixed |
| `run_j6b.py` | plan / freeze / smoke / flight / analyze |
| `tests/test_j6b_*.py` | logic, capture, verdict suites |

```sh
PY=~/venvs/semcore/bin/python
$PY -m pytest jev/tests/test_j6b_logic.py jev/tests/test_j6b_capture.py jev/tests/test_j6b_verdict.py -q
$PY -m jev.run_j6b freeze
$PY -m jev.run_j6b smoke  --frozen jev/results/j6b_freeze_<stamp>/FROZEN_J6B.json
$PY -m jev.run_j6b flight --frozen <F> --smoke jev/results/j6b_smoke_<stamp>
$PY -m jev.run_j6b analyze --flight jev/results/j6b_flight_<stamp> --frozen <F> --smoke <S>
```

G7 for J6b is its own: every `jev/results/j6b_*` stage counts toward $1.

## J5: Jev as a color instrument

The builder for [`docs/JEV_J5_PROTOCOL.md`](../docs/JEV_J5_PROTOCOL.md). Jev answers the
color–feeling survey of Jonauskaite et al. (2020): 240 yes/no pairings (YN) and 20 forced
choices over 12 colors (CH), plus *fun* and two made-up feelings, 3,312 calls. Primaries:
C1 within the human range (r_dc against the world, floor = the least typical nation) and
C2 an English lean (mixture weight on Jev's own scale, Amendment 1).

The human tables are licence-restricted and stay outside the repo, in `~/association-norms/`
(override with `J5_NORMS`); the analysis reads them by sha256 and writes derived statistics only.

| file | what it does |
|---|---|
| `j5.py` | colors, feelings, wordings, framings, items, neutral keys, rotations |
| `j5_plan.py` | smoke and flight plans; plan shas |
| `analyze_j5.py` | human inputs by sha, Jev's tables, gates (G5 structure), C1, C2, secondaries S1–S12 |
| `agents_j5.py` | planted agents: people, english, rest, sharp, scrambled, fixed |
| `power/j5_design_check.py` | the yardstick, sharpness caps and culture-test checks cited in the protocol |
| `power/j5_c2_curve_check.py` | Amendment 1's calibration (curve-matched vs raw C2) |

```sh
PY=~/venvs/semcore/bin/python
$PY -m pytest jev/tests/test_j5_logic.py jev/tests/test_j5_capture.py jev/tests/test_j5_verdict.py -q
$PY -m jev.run_j5 plan
$PY -m jev.run_j5 freeze                                  # FROZEN_J5.json; no calls
$PY -m jev.run_j5 smoke  --frozen jev/results/j5_freeze_<stamp>/FROZEN_J5.json
$PY -m jev.run_j5 flight --frozen <F> --smoke jev/results/j5_smoke_<stamp>
$PY -m jev.run_j5 analyze --flight jev/results/j5_flight_<stamp> --frozen <F> --smoke <S>
```

G7 for J5 is its own: every `jev/results/j5_*` stage counts toward $1.
