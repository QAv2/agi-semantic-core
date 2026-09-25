"""Call plans: smoke test, titration pilot, confirmatory flight (protocol §4–§6).

A plan is a list of call specs. Each carries its request `body` (sent, never stored)
and the metadata the analysis needs. `plan_sha` hashes call_id + body sha256 for the
whole plan; FROZEN.json records it before the flight.
"""
import hashlib

import numpy as np

from . import common as C
from . import j1, j2

GRIDS = {
    "F1": [4.0, 2.83, 2.0, 1.41, 1.0, 0.71, 0.5],
    "F1K2": [2.83, 2.0, 1.41, 1.0, 0.71, 0.5, 0.35],
    "F2": [0.18, 0.25, 0.35, 0.5, 0.71, 1.0, 1.41, 2.0],
    "F3": [0.92, 0.84, 0.76, 0.68, 0.60, 0.52, 0.45],
}
EASIER = {"F1": 1, "F1K2": 1, "F2": -1, "F3": 1}   # +1: larger level is easier
N_PILOT = 40
N_FLIGHT = 400
REPS = 3


def extend_levels(family, end, steps):
    """Extra grid levels beyond the registered grid (protocol §4.2: up to two)."""
    g = GRIDS[family]
    out = []
    for n in range(1, steps + 1):
        if family == "F3":
            if end == "easier":
                out.append(round(min(0.97, max(g) + 0.08 * n), 2))
            else:
                out.append(round(min(g) - 0.08 * n, 2))
            continue
        easy = max(g) if EASIER[family] > 0 else min(g)
        hard = min(g) if EASIER[family] > 0 else max(g)
        f = np.sqrt(2) ** n
        if end == "easier":
            out.append(round(easy * f if EASIER[family] > 0 else easy / f, 3))
        else:
            out.append(round(hard / f if EASIER[family] > 0 else hard * f, 3))
    return out


def _j1_spec(stage, fam, stim, arm_label, render_arm, strength, i, rep, sets, questions_mode,
             order_words, render_words, ids_words, extra):
    K = stim["K"]
    order = C.balanced_order(i, rep, K, stim["truth"], C.rng(stage, "J1", fam, "order", *order_words))
    state, options, k2i = j1.render(stim, render_arm, order, C.rng(stage, "J1", fam, "render", *render_words),
                                    C.rng(stage, "J1", fam, "ids", *ids_words), strength)
    qs = {}
    if questions_mode in ("decision", "bundled"):
        qs["decision"] = C.choice_q(j1.INSTR[fam], options)
    if questions_mode in ("bundled", "a6"):
        qs["determined"] = C.A6_DETERMINED
        qs["insufficient"] = C.A6_INSUFFICIENT
    body = C.make_body(state, qs)
    ideal = j1.ideal(stim)
    top = int(np.argmax(ideal))
    spec = {"exp": "J1", "family": fam, "arm": arm_label, "render": render_arm, "strength": strength,
            "set": sets, "level": stim["level"], "stim": i, "rep": rep, "K": K, "truth": stim["truth"],
            "ideal": [round(float(p), 12) for p in ideal], "pos_truth": order.index(stim["truth"]),
            "pos_ideal_top": order.index(top), "order": order, "key_to_idx": k2i,
            "questions": sorted(qs), "body": body, "body_sha": C.body_sha(body)}
    spec.update(extra)
    return spec


def j1_flight(frozen, n=N_FLIGHT):
    specs = []
    fams = [f for f in ("F1", "F2", "F3") if frozen["families"][f]["included"]]
    bundle = frozen.get("bundle_a6", True)
    runs = [(f, frozen["families"][f]) for f in fams]
    if frozen["families"].get("F1K2", {}).get("included"):
        k2 = dict(frozen["families"]["F1K2"])
        k2["strength"] = frozen["families"]["F1"]["strength"]
        runs.append(("F1K2", k2))
    for fam, fz in runs:
        arms = [("A0", "A0", None), ("A1-same", "A1", fz["strength"])]
        if fam != "F1K2":
            arms += [("A2", "A2", None), ("A3", "A3", None)]
        for i in range(n):
            stim = j1.generate(fam, C.rng("flight", "J1", fam, "base", i), fz["d0"])
            for arm, rarm, strength in arms:
                for rep in range(REPS):
                    specs += _j1_calls("flight", fam, stim, arm, rarm, strength, i, rep, "base", bundle)
        if fam == "F1K2":
            continue
        for i in range(n):
            stim = j1.generate(fam, C.rng("flight", "J1", fam, "matched", i), fz["d1"])
            for rep in range(REPS):
                specs += _j1_calls("flight", fam, stim, "A1-matched", "A1", fz["strength"], i, rep,
                                   "matched", bundle)
    return specs


def _j1_calls(stage, fam, stim, arm, rarm, strength, i, rep, sets, bundle):
    """One stimulus-repeat: the decision call, plus a separate A6 call if not bundled."""
    wants_a6 = arm in ("A0", "A1-same", "A1-matched")
    mode = "bundled" if (wants_a6 and bundle) else "decision"
    common = dict(order_words=(sets, i, rep), render_words=(sets, i, arm), ids_words=(sets, i))
    cid = f"J1-{stage}-{fam}-{arm}-{sets}-{i}-r{rep}"
    out = [_j1_spec(stage, fam, stim, arm, rarm, strength, i, rep, sets, mode,
                    extra={"call_id": cid, "stage": stage}, **common)]
    if wants_a6 and not bundle:
        out.append(_j1_spec(stage, fam, stim, arm, rarm, strength, i, rep, sets, "a6",
                            extra={"call_id": cid + "-a6", "stage": stage}, **common))
    return out


def j1_pilot(extra_levels=None):
    """Titration pilot: per family x presentation x level, 40 stimuli, 1 call each.
    The same stimuli are rendered in every presentation (paired)."""
    specs = []
    extra_levels = extra_levels or {}
    for fam in ("F1", "F2", "F3", "F1K2"):
        levels = list(GRIDS[fam]) + list(extra_levels.get(fam, []))
        pres = [("A0", "A0", None)]
        if fam != "F1K2":
            pres += [("A1-standard", "A1", "standard"), ("A1-strong", "A1", "strong")]
        for li, lv in enumerate(levels):
            for j in range(N_PILOT):
                stim = j1.generate(fam, C.rng("pilot", "J1", fam, "stim", li, j), lv)
                for arm, rarm, strength in pres:
                    sp = _j1_spec("pilot", fam, stim, arm, rarm, strength, j + li, 0, "pilot", "decision",
                                  order_words=("pilot", li, j), render_words=("pilot", li, j, arm),
                                  ids_words=("pilot", li, j),
                                  extra={"call_id": f"J1-pilot-{fam}-{arm}-L{li}-{j}", "stage": "pilot",
                                         "level_idx": li})
                    specs.append(sp)
    return specs


def j2_flight(n_ladders=120):
    specs = []
    for L in range(n_ladders):
        lad = j2.gen_ladder("flight", L)
        for arm, rung, c in j2.stimuli_for_ladder(lad):
            for rep in range(REPS):
                specs.append(_j2_spec("flight", lad, arm, rung, c, rep))
    return specs


def _j2_spec(stage, lad, arm, rung, c, rep, call_id=None):
    forced = arm.startswith("M-forced")
    r = C.rng(stage, "J2", "order", lad["L"], rung, rep, 4 if forced else 5)
    order = j2.order4(lad, rung, rep, r) if forced else j2.order5(lad, rung, rep, r)
    state, options, k2i = j2.render(lad, arm, rung, c, order)
    body = C.make_body(state, {"decision": C.choice_q(j2.INSTR, options)})
    post = j2.ideal_post(lad, arm, rung)
    return {"call_id": call_id or f"J2-{stage}-L{lad['L']}-{arm}-k{rung}-c{c}-r{rep}", "stage": stage,
            "exp": "J2", "ladder": lad["L"], "template": lad["template"], "arm": arm, "rung": rung, "c": c,
            "rep": rep, "truth": lad["truth"], "ideal": [round(float(p), 12) for p in post],
            "order": order, "key_to_idx": k2i, "questions": ["decision"], "body": body,
            "body_sha": C.body_sha(body)}


def smoke():
    """Determinism (24 bodies x 10 identical), throughput (200 over 5 concurrency
    groups), question isolation (30 stimuli x 5 identical x {alone, bundled})."""
    specs = []
    for fam in ("F1", "F2", "F3"):
        lv = GRIDS[fam][3]
        for n in range(2):
            stim = j1.generate(fam, C.rng("smoke", "J1", fam, "det", n), lv)
            for arm, rarm, strength in (("A0", "A0", None), ("A1-standard", "A1", "standard")):
                base = _j1_spec("smoke", fam, stim, arm, rarm, strength, n, 0, "det", "decision",
                                order_words=("det", n), render_words=("det", n, arm), ids_words=("det", n),
                                extra={"stage": "smoke", "block": "determinism"})
                for x in range(10):
                    specs.append(dict(base, call_id=f"SMK-det-{fam}-{arm}-{n}-x{x}", dup=x))
    for L in range(4):
        lad = j2.gen_ladder("smoke", L)
        for rung in (0, 4, 8):
            base = _j2_spec("smoke", lad, "M-walked", rung, 3, 0)
            base["block"] = "determinism"
            for x in range(10):
                specs.append(dict(base, call_id=f"SMK-det-J2-L{L}-k{rung}-x{x}", dup=x))
    for g, conc in enumerate((1, 2, 4, 8, 16)):
        for n in range(40):
            i = g * 40 + n
            stim = j1.generate("F1", C.rng("smoke", "J1", "F1", "thr", i), GRIDS["F1"][3])
            specs.append(_j1_spec("smoke", "F1", stim, "A0", "A0", None, i, 0, "thr", "decision",
                                  order_words=("thr", i), render_words=("thr", i), ids_words=("thr", i),
                                  extra={"call_id": f"SMK-thr-c{conc}-{n}", "stage": "smoke",
                                         "block": "throughput", "concurrency": conc}))
    for fam in ("F1", "F2", "F3"):
        for n in range(10):
            stim = j1.generate(fam, C.rng("smoke", "J1", fam, "iso", n), GRIDS[fam][3])
            for mode in ("decision", "bundled"):
                base = _j1_spec("smoke", fam, stim, "A0", "A0", None, n, 0, "iso", mode,
                                order_words=("iso", n), render_words=("iso", n), ids_words=("iso", n),
                                extra={"stage": "smoke", "block": "isolation", "mode": mode})
                for x in range(5):
                    specs.append(dict(base, call_id=f"SMK-iso-{fam}-{n}-{mode}-x{x}", dup=x))
    return specs


def plan_sha(specs, h=None):
    """sha256 over 'call_id<TAB>body_sha' lines, in plan order. Pass a running
    hashlib object to hash a plan built in parts (J1 then J2)."""
    h = h or hashlib.sha256()
    for s in specs:
        h.update(f"{s['call_id']}\t{s['body_sha']}\n".encode())
    return h.hexdigest()


def flight_parts(frozen, n=N_FLIGHT, n_ladders=120):
    """The flight in two parts, built one at a time (memory): J1, then J2."""
    yield "J1", j1_flight(frozen, n)
    yield "J2", j2_flight(n_ladders)


def flight_sha(frozen, n_stim=N_FLIGHT, n_ladders=120):
    """(sha, n) of the whole flight plan; equals plan_sha(J1 + J2)."""
    h, n = hashlib.sha256(), 0
    for _, specs in flight_parts(frozen, n_stim, n_ladders):
        sha = plan_sha(specs, h)
        n += len(specs)
        del specs
    return sha, n


def strip(spec):
    """Spec without the request body (what the plan file stores)."""
    return {k: v for k, v in spec.items() if k != "body"}
