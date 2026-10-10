"""J5 call plans: smoke test and flight (docs/JEV_J5_PROTOCOL.md §3, §4)."""
from . import common as C
from . import j5
from .plan import plan_sha, strip  # noqa: F401  (re-exported for the driver)


def yn_spec(stage, item, feeling, c, o, block, call_id=None):
    color = j5.COLORS[c]
    body = C.make_body(j5.page(item, stage), {j5.YN_Q: j5.yn_question(item, feeling, color, o)})
    return {"call_id": call_id or f"J5-YN-{feeling}-{color}-i{item['i']}-o{o}", "body": body,
            "body_sha": C.body_sha(body), "exp": "J5", "fmt": "YN", "block": block, "feeling": feeling,
            "color": c, "item": item["i"], "w": item["w"], "f": item["f"], "o": o, "questions": [j5.YN_Q]}


def ch_spec(stage, item, feeling, r, block, call_id=None):
    order = j5.ch_rotation(item, feeling, r, stage)
    opts, k2i = j5.ch_options(item, feeling, order, stage)
    body = C.make_body(j5.page(item, stage), {"decision": j5.ch_question(item, feeling, opts)})
    return {"call_id": call_id or f"J5-CH-{feeling}-i{item['i']}-r{r:02d}", "body": body,
            "body_sha": C.body_sha(body), "exp": "J5", "fmt": "CH", "block": block, "feeling": feeling,
            "item": item["i"], "w": item["w"], "f": item["f"], "r": r, "order": order,
            "key_to_idx": k2i, "questions": ["decision"]}


def flight():
    """3,312 calls: YN (feeling, color, item, order), then CH (feeling, item, rotation)."""
    its, fs = j5.items(), j5.FEELINGS["flight"]
    yn = [yn_spec("flight", it, f, c, o, "flight") for f in fs for c in range(j5.NC) for it in its
          for o in range(j5.ORDERS)]
    ch = [ch_spec("flight", it, f, r, "flight") for f in fs for it in its for r in range(j5.ROT)]
    return yn + ch


def smoke():
    """216 calls on the smoke stream (own framings and feelings; never analysed):
    6 YN + 6 CH distinct calls x 10 repeats, then 48 YN + 48 CH distinct calls."""
    its, fs = j5.items(), j5.FEELINGS["smoke"]
    yn_all = [(it, f, c, o) for f in fs for c in range(j5.NC) for it in its for o in range(j5.ORDERS)]
    ch_all = [(it, f, r) for f in fs for it in its for r in range(j5.ROT)]
    yn_pick = [int(x) for x in j5.rng("smoke", "pick", "YN").permutation(len(yn_all))]
    ch_pick = [int(x) for x in j5.rng("smoke", "pick", "CH").permutation(len(ch_all))]
    specs = []
    for n in range(12):
        base = (yn_spec("smoke", *yn_all[yn_pick[n]], "determinism") if n < 6
                else ch_spec("smoke", *ch_all[ch_pick[n - 6]], "determinism"))
        specs += [base | {"call_id": f"J5S-det-{n:02d}-x{x}"} for x in range(10)]
    for n in range(96):
        h = n // 2
        s = (yn_spec("smoke", *yn_all[yn_pick[6 + h]], "throughput", call_id=f"J5S-thr-{n:02d}") if n % 2 == 0
             else ch_spec("smoke", *ch_all[ch_pick[6 + h]], "throughput", call_id=f"J5S-thr-{n:02d}"))
        specs.append(s)
    return specs
