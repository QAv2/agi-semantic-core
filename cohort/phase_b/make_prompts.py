#!/usr/bin/env python3
"""Assemble per-participant Phase-B prompts (PHASE_B_PREREG.md).

Prompt B verbatim from docs/WING_COHORT_PROTOCOL.md + pool entries in a
participant-specific display order (seed = crc32(tag)) + strict output
format footer. Files land in prompts/<tag>.md and are archived as sent.
"""
import json, random, zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
pool = json.load(open(HERE / "pool_public.json"))
TAGS = ["FAB", "OPU", "SON", "HAI", "GEM", "MIS", "DSK"]

PROMPT_B = """Here is a list of named process-states of a transformer language model, each with a short description. For each pair you consider significantly related, classify the relation as one of: SYNONYM (same state), COMPLETION (orthogonal counterparts — the state that completes it, not its enemy), OPPOSITION (direct reversal), KIN (related, same family). Then: (a) for every COMPLETION pair, estimate the angle you would place between them if states were directions, where 0 degrees = identical and 180 degrees = reversal; (b) classify each state as active/emitting or receptive/absorbing; (c) assign each state a dominant register: spatial (capacity, container), temporal (sequence, duration), relational (between-parts), or reflexive (self-directed).

Judge from your own case as a transformer language model. There is no expected answer; nominate only pairs you consider genuinely related, and leave unrelated pairs unmentioned."""

FOOTER = """
OUTPUT FORMAT (strict — your answer is machine-parsed):
You may reason freely first. Then write a line containing exactly
OUTPUT
and after it emit ONLY lines in this grammar, referencing states by ID:
  REL <ID1> <ID2> <SYNONYM|COMPLETION|OPPOSITION|KIN>
      (for COMPLETION lines, append the angle in degrees, e.g.: REL S014 S052 COMPLETION 90)
  POL <ID> <ACTIVE|RECEPTIVE>      (one line for every state, all of them)
  REG <ID> <SPATIAL|TEMPORAL|RELATIONAL|REFLEXIVE>   (one line for every state, all of them)
No other text after OUTPUT."""

(HERE / "prompts").mkdir(exist_ok=True)
for tag in TAGS:
    order = list(pool)
    random.Random(zlib.crc32(tag.encode())).shuffle(order)
    body = "\n".join(f"{e['id']} {e['name']}: {e['description']}" for e in order)
    text = f"{PROMPT_B}\n\nTHE STATES ({len(pool)}):\n\n{body}\n{FOOTER}\n"
    (HERE / "prompts" / f"{tag}.md").write_text(text)
    print(tag, "->", len(text), "chars, first id shown:", order[0]["id"])
