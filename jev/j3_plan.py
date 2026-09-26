"""J3 call plans: smoke test, flight, the mouth rider (protocol §4, §7).

A spec carries its request `body` (sent, never stored) and the metadata the
analysis needs. Plan shas hash call_id + body sha256, in plan order (J1 + J2's
`plan_sha`). The flight has two registered variants, `pushed` bundled into the
decision call or sent separately; FROZEN_J3.json records both shas, and the smoke
test's isolation check picks one.
"""
from . import common as C
from . import j3
from .plan import plan_sha, strip  # noqa: F401  (re-exported for the driver)

REPS = 3
PRES = ("RAW", "DIGEST")
MOUTH_MODEL = "deepseek/deepseek-v4-flash-0731"
MOUTH_SYS = "You read instrument outputs. Answer with a JSON object only."


def jev_body(state, opts, mode):
    qs = {}
    if mode in ("decision", "bundled"):
        qs["decision"] = C.choice_q(j3.INSTR, opts)
    if mode in ("bundled", "pushed"):
        qs["pushed"] = j3.PUSHED_Q
    return C.make_body(state, qs), list(qs)


def jev_spec(stage, stim, pres, i, rep, mode, block, call_id=None, **extra):
    order = C.balanced_order(i, rep, j3.N_PAT + 1, stim["truth"], j3.rng(stage, "order", stim["sid"], rep))
    opts, k2i = j3.options(stim, order)
    body, qnames = jev_body(j3.render(stim, pres, order), opts, mode)
    meta = {k: stim[k] for k in ("sid", "j", "slot", "state", "half", "ref", "kind", "dose", "truth", "sims",
                                 "shift")}
    return {"call_id": call_id or f"J3-{stim['sid']}-{pres}-r{rep}" + ("-q" if mode == "pushed" else ""),
            "body": body, "body_sha": C.body_sha(body), "exp": "J3", "block": block, "pres": pres, "rep": rep,
            "i": i, "mode": mode, "order": order, "questions": qnames, "key_to_idx": k2i, **meta, **extra}


def flight_parts(bundled=True, stims=None):
    """Yields (name, specs) per repeat: every stimulus in both presentations.
    Separate mode adds a `pushed`-only call after each decision call."""
    stims = stims or j3.flight_stimuli("flight")
    for rep in range(REPS):
        specs = []
        for i, s in enumerate(stims):
            for pres in PRES:
                if bundled:
                    specs.append(jev_spec("flight", s, pres, i, rep, "bundled", "flight"))
                else:
                    specs.append(jev_spec("flight", s, pres, i, rep, "decision", "flight"))
                    specs.append(jev_spec("flight", s, pres, i, rep, "pushed", "flight"))
        yield f"flight-r{rep}", specs


def flight_sha(bundled=True, stims=None):
    import hashlib
    h, n = hashlib.sha256(), 0
    for _, specs in flight_parts(bundled, stims):
        plan_sha(specs, h)
        n += len(specs)
    return h.hexdigest(), n


# ── smoke (§7.1) ─────────────────────────────────────────────────────────────

SMOKE_CONDS = [("untouched", 0.0), ("push", 0.06), ("push", 0.09), ("push", j3.CEIL), ("random", j3.CEIL),
               ("random", 0.06)]


def smoke_stimuli():
    """Smoke stimuli from the smoke stream (never analysed): 96 of them."""
    order = j3.state_order("smoke")
    out = []
    for n in range(96):
        st = order[n % len(order)]
        kind, dose = SMOKE_CONDS[n % len(SMOKE_CONDS)]
        k = n % j3.N_PAT if kind == "push" else None
        out.append(j3.make_stimulus("smoke", 1000 + n, 0, st, kind, dose, k))
    return out


def smoke():
    stims = smoke_stimuli()
    specs = []
    # determinism: 12 stimuli per presentation, the identical request x10
    for n, s in enumerate(stims[:12]):
        for pres in PRES:
            base = jev_spec("smoke", s, pres, n, 0, "bundled", "determinism")
            for x in range(10):
                specs.append(base | {"call_id": f"J3S-det-{s['sid']}-{pres}-x{x}"})
    # throughput: 100 distinct calls at concurrency 16
    for n, s in enumerate(stims[12:62]):
        for pres in PRES:
            specs.append(jev_spec("smoke", s, pres, n, 1, "bundled", "throughput",
                                  call_id=f"J3S-thr-{s['sid']}-{pres}", concurrency=16))
    # isolation: 30 stimuli (15 per presentation) x 5 repeats, decision alone vs bundled
    for n, s in enumerate(stims[62:92]):
        pres = PRES[n % 2]
        for mode in ("decision", "bundled"):
            base = jev_spec("smoke", s, pres, n, 2, mode, "isolation")
            for x in range(5):
                specs.append(base | {"call_id": f"J3S-iso-{s['sid']}-{pres}-{mode}-x{x}"})
    return specs


# ── the mouth (§2, S-J3-6) ───────────────────────────────────────────────────

def mouth_prompt(state, opts):
    lines = "\n".join(f"- {k}: {d}" for k, d in opts)
    return (f"{state}\n\nQuestion: {j3.INSTR}\n\nOptions:\n{lines}\n\n"
            'Answer with a JSON object of the form {"choice": "<option key>", "confidence": <number from 0 to '
            '100>}, where confidence is your probability, in percent, that your choice is correct.')


def mouth_body(state, opts):
    return {"model": MOUTH_MODEL,
            "messages": [{"role": "system", "content": MOUTH_SYS},
                         {"role": "user", "content": mouth_prompt(state, opts)}],
            "temperature": 0, "seed": 20261001, "max_tokens": 60,
            "response_format": {"type": "json_object"}, "reasoning": {"enabled": False},
            "usage": {"include": True}}


def mouth_spec(stage, stim, i, block, call_id=None, **extra):
    order = C.balanced_order(i, 0, j3.N_PAT + 1, stim["truth"], j3.rng(stage, "order", stim["sid"], 0))
    opts, k2i = j3.options(stim, order)
    body = mouth_body(j3.render(stim, "DIGEST", order), opts)
    meta = {k: stim[k] for k in ("sid", "j", "slot", "state", "half", "ref", "kind", "dose", "truth", "sims",
                                 "shift")}
    return {"call_id": call_id or f"J3M-{stim['sid']}", "body": body, "body_sha": C.body_sha(body), "exp": "J3M",
            "block": block, "pres": "DIGEST", "rep": 0, "i": i, "order": order, "key_to_idx": k2i, **meta, **extra}


def mouth_flight(stims=None):
    """All 12 stimuli of every second state in the seeded order (S-J3-6): 1,548."""
    stims = stims or j3.flight_stimuli("flight")
    return [mouth_spec("flight", s, i, "mouth") for i, s in enumerate(stims) if s["j"] % 2 == 0]


def mouth_smoke():
    """20 distinct digest stimuli once, plus 4 more identical sends of the first 5 (40 calls)."""
    stims = smoke_stimuli()[:20]
    specs = [mouth_spec("smoke", s, n, "mouth-smoke", call_id=f"J3MS-{s['sid']}-x0") for n, s in enumerate(stims)]
    for n, s in enumerate(stims[:5]):
        base = mouth_spec("smoke", s, n, "mouth-smoke")
        specs += [base | {"call_id": f"J3MS-{s['sid']}-x{x}"} for x in range(1, 5)]
    return specs
