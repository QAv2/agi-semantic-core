"""J6b call plans: smoke test and flight (docs/JEV_J6B_PROTOCOL.md §1, §2).

Specs carry q = "Q0" so J6's analysis helpers (keyed on state, q, item) apply unchanged.
"""
from . import common as C
from . import j6b
from .plan import plan_sha, strip  # noqa: F401  (re-exported for the driver)

ROTATIONS = 8


def spec(stage, item, state, r, block, call_id=None, **extra):
    order = j6b.rotation(item, r, stage)
    opts, k2i = j6b.options(item, order, stage)
    body = C.make_body(j6b.page(item, state, stage), {"decision": C.choice_q(j6b.QUESTION, opts)})
    return {"call_id": call_id or f"J6B-{state}-i{item['i']:02d}-r{r}", "body": body,
            "body_sha": C.body_sha(body), "exp": "J6B", "block": block, "state": state, "q": "Q0", "r": r,
            "item": item["i"], "n": item["n"], "o": item["o"], "f": item["f"], "w": item["w"],
            "order": order, "questions": ["decision"], "key_to_idx": k2i, **extra}


def flight():
    """2,112 calls: state-major, then item, rotation."""
    its = j6b.items("flight")
    return [spec("flight", it, s, r, "flight") for s in j6b.STATES for it in its for r in range(ROTATIONS)]


def smoke():
    """216 calls on the smoke stream (own names, neutral sentences and framings; never analysed)."""
    its = j6b.items("smoke")
    specs = []
    for n in range(12):
        base = spec("smoke", its[n % len(its)], j6b.STATES[n % len(j6b.STATES)], n % ROTATIONS, "determinism")
        specs += [base | {"call_id": f"J6BS-det-{n:02d}-x{x}"} for x in range(10)]
    for n in range(96):
        specs.append(spec("smoke", its[(n * 5) % len(its)], j6b.STATES[(n + 3) % len(j6b.STATES)],
                          (n * 3) % ROTATIONS, "throughput", call_id=f"J6BS-thr-{n:02d}"))
    return specs
