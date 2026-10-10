"""J6 call plans: smoke test and flight (docs/JEV_J6_PROTOCOL.md §3.4, §4.1).

A spec carries its request `body` (sent, never stored) and the metadata the
analysis needs. Plan shas hash call_id + body sha256 in plan order (`plan.plan_sha`).
"""
from . import common as C
from . import j6
from .plan import plan_sha, strip  # noqa: F401  (re-exported for the driver)

ROTATIONS = 8


def spec(stage, item, state, q, r, block, call_id=None, **extra):
    order = j6.rotation(item, r, stage)
    opts, k2i = j6.options(item, order, stage)
    body = C.make_body(j6.page(item, state, stage), {"decision": C.choice_q(j6.QUESTIONS[q], opts)})
    return {"call_id": call_id or f"J6-{state}-{q}-i{item['i']:02d}-r{r}", "body": body,
            "body_sha": C.body_sha(body), "exp": "J6", "block": block, "state": state, "q": q, "r": r,
            "item": item["i"], "a": item["a"], "n": item["n"], "o": item["o"], "f": item["f"],
            "order": order, "questions": ["decision"], "key_to_idx": k2i, **extra}


def flight():
    """3,456 calls: state-major, then question, item, rotation."""
    its = j6.items("flight")
    return [spec("flight", it, s, q, r, "flight")
            for s in j6.SS for q in j6.QS for it in its for r in range(ROTATIONS)]


def smoke():
    """216 calls on the smoke stream (own names and framings, never analysed)."""
    its = j6.items("smoke")
    cells = [(s, q) for s in j6.SS for q in j6.QS]
    specs = []
    # determinism: 12 distinct calls, each sent 10 times
    for n in range(12):
        s, q = cells[n % len(cells)]
        base = spec("smoke", its[n % len(its)], s, q, n % ROTATIONS, "determinism")
        specs += [base | {"call_id": f"J6S-det-{n:02d}-x{x}"} for x in range(10)]
    # throughput: 96 distinct calls at concurrency 16
    for n in range(96):
        s, q = cells[(n + 5) % len(cells)]
        specs.append(spec("smoke", its[(n * 5) % len(its)], s, q, (n * 3) % ROTATIONS, "throughput",
                          call_id=f"J6S-thr-{n:02d}"))
    return specs
