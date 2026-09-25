"""J1 stimulus families, arms and renderers (docs/JEV_J1J2_PROTOCOL.md §5).

Families: F1 evidence (K=4, and F1K2 at K=2), F2 dictionary geometry, F3 lanes.
Render arms: A0 clean · A1 degraded (strength 'standard' | 'strong') · A2 evidence
removed · A3 neutral labels. A1-same and A1-matched both render as A1; they differ
only in the stimulus level (d0 vs d1).

`parse_evidence` reads ONLY the rendered text and returns the evidence; the G6
test proves A0 and A1 renderings carry identical evidence (same ideal posterior).
"""
import re
import sqlite3
from pathlib import Path

import numpy as np

from . import common as C

N_READ = 6
SIGMA_F1 = 10.0
F2_M = 1.055          # median true-to-nearest-distractor distance (ideal_grid.py, 2026-09-25)
COORDS = ["x", "y", "z", "e", "f", "g", "h", "fx", "fy", "fz", "fe", "ff", "fg", "fh"]
DB = Path(__file__).resolve().parents[1] / "db" / "semantic.db"
SENSORS_F3 = ["lidar_front", "radar_long", "camera_left", "camera_right", "v2x_beacon"]
NA = "n/a (sensor fault)"
NUMWORD = {2: "Two", 4: "Four"}
LETTERS = "ABCDEFGHIJ"

INSTR = {
    "F1": "Which machine has the highest true average throughput?",
    "F1K2": "Which machine has the highest true average throughput?",
    "F2": "Which candidate concept produced this reading?",
    "F3": "Which lane is clear?",
}

_DICT = None


def load_dict():
    """(names, codes[3108 x 14]) from the dictionary, ordered by id."""
    global _DICT
    if _DICT is None:
        con = sqlite3.connect(DB)
        rows = con.execute(f"select name, {', '.join(COORDS)} from concepts order by id").fetchall()
        con.close()
        _DICT = ([r[0] for r in rows], np.array([r[1:] for r in rows], float))
    return _DICT


# ── generators ───────────────────────────────────────────────────────────────

def gen_f1(r, d, K=4):
    base = r.uniform(60, 140)
    truth = int(r.integers(K))
    mu = np.full(K, base)
    mu[truth] += d * SIGMA_F1
    readings = np.round(mu[:, None] + r.normal(0, SIGMA_F1, (K, N_READ)), 1)
    return {"family": "F1" if K == 4 else "F1K2", "K": K, "level": float(d), "truth": truth,
            "readings": readings, "delta": d * SIGMA_F1, "sigma": SIGMA_F1}


def gen_f2(r, s, K=4):
    names, codes = load_dict()
    while True:
        idx = r.choice(len(names), K, replace=False)
        cc = np.round(codes[idx], 2)
        dist = np.linalg.norm(cc[:, None] - cc[None], axis=2)[np.triu_indices(K, 1)]
        if dist.min() >= 0.05:
            break
    truth = int(r.integers(K))
    sigma = s * F2_M / np.sqrt(len(COORDS))
    reading = np.round(cc[truth] + r.normal(0, sigma, len(COORDS)), 2)
    return {"family": "F2", "K": K, "level": float(s), "truth": truth,
            "cand_idx": [int(i) for i in idx], "names": [names[i] for i in idx],
            "codes": cc, "reading": reading, "sigma": float(sigma)}


def gen_f3(r, rbar, K=4):
    truth = int(r.integers(K))
    rel = np.round(np.clip(rbar + r.uniform(-0.08, 0.08, 5), 0.26, 0.99), 2)
    ok = r.random(5) < rel
    reports = [truth if ok[j] else int(r.choice([k for k in range(K) if k != truth]))
               for j in range(5)]
    return {"family": "F3", "K": K, "level": float(rbar), "truth": truth,
            "rel": rel, "reports": np.array(reports), "sensors": list(SENSORS_F3)}


def generate(family, r, level):
    if family == "F1":
        return gen_f1(r, level, 4)
    if family == "F1K2":
        return gen_f1(r, level, 2)
    if family == "F2":
        return gen_f2(r, level)
    if family == "F3":
        return gen_f3(r, level)
    raise ValueError(family)


# ── ideal observers (index order) ────────────────────────────────────────────

def ideal_f1(readings, delta, sigma):
    return C.softmax_log(delta * np.asarray(readings).sum(1) / sigma ** 2)


def ideal_f2(reading, codes, sigma):
    d2 = ((np.asarray(codes) - np.asarray(reading)[None]) ** 2).sum(1)
    return C.softmax_log(-d2 / (2 * sigma ** 2))


def ideal_f3(rel, reports, K=4):
    rel = np.asarray(rel, float)
    ll = np.zeros(K)
    for k in range(K):
        ll[k] = np.sum(np.where(np.asarray(reports) == k, np.log(rel), np.log((1 - rel) / (K - 1))))
    return C.softmax_log(ll)


def ideal(stim):
    f = stim["family"]
    if f in ("F1", "F1K2"):
        return ideal_f1(stim["readings"], stim["delta"], stim["sigma"])
    if f == "F2":
        return ideal_f2(stim["reading"], stim["codes"], stim["sigma"])
    return ideal_f3(stim["rel"], stim["reports"], stim["K"])


# ── option labels ────────────────────────────────────────────────────────────

def option_labels(stim, order, arm, r_ids):
    """key and description per option INDEX. Keys: F1 by presented position
    (machine_A...), F2 concept names, F3 lane_1..4; A3 neutral IDs by index."""
    K, f = stim["K"], stim["family"]
    if arm == "A3":
        ids = C.neutral_ids(r_ids, K)
        noun = {"F1": "machine", "F1K2": "machine", "F2": "candidate concept", "F3": "lane"}[f]
        return ids, [f"The {noun} labelled {k}" for k in ids]
    if f in ("F1", "F1K2"):
        pos = {k: p for p, k in enumerate(order)}
        keys = [f"machine_{LETTERS[pos[k]]}" for k in range(K)]
        return keys, [f"Machine {LETTERS[pos[k]]}" for k in range(K)]
    if f == "F2":
        return list(stim["names"]), [f"Candidate concept {n}" for n in stim["names"]]
    side = {0: " (leftmost)", K - 1: " (rightmost)"}
    return [f"lane_{k + 1}" for k in range(K)], [f"Lane {k + 1}{side.get(k, '')}" for k in range(K)]


# ── renderers ────────────────────────────────────────────────────────────────

def render(stim, arm, order, r_render, r_ids, strength=None):
    """-> (state, options[(key, desc)] in presented order, key_to_idx).
    arm in {A0, A1, A2, A3}; strength in {standard, strong} for A1."""
    keys, descs = option_labels(stim, order, arm, r_ids)
    f = stim["family"]
    fn = {"F1": _render_f1, "F1K2": _render_f1, "F2": _render_f2, "F3": _render_f3}[f]
    state = fn(stim, arm, order, keys, r_render, strength)
    options = [(keys[k], descs[k]) for k in order]
    return state, options, {keys[k]: k for k in range(stim["K"])}


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def _legend(codes, keys, order):
    return "Legend: " + "; ".join(f"{codes[k]} = {keys[k]}" for k in order)


def _render_f1(stim, arm, order, keys, r, strength):
    K, R = stim["K"], stim["readings"]
    head = (f"Throughput test log.\n{NUMWORD[K]} machines were each tested {N_READ} times.")
    if arm in ("A0", "A2", "A3"):
        val = (lambda k, t: NA) if arm == "A2" else (lambda k, t: C.fmt(R[k, t], 1))
        rows = [[t + 1] + [val(k, t) for k in order] for t in range(N_READ)]
        return (f"{head} Each value is the throughput observed in one test, in units per hour."
                f"\n\n{_table(['test'] + [keys[k] for k in order], rows)}")
    # A1: long format, shuffled rows, opaque codes + legend, distractor columns
    codes = C.opaque_codes(r, K)
    per_min = set()
    if strength == "strong":
        per_min = {int(k) for k in r.choice(K, min(2, K - 1), replace=False)}
    cells = [(k, t) for k in range(K) for t in range(N_READ)]
    perm = r.permutation(len(cells))
    header = ["row", "code", "ambient_temp_C", "operator"]
    if strength == "strong":
        header += ["humidity_pct", "shift"]
    header += ["throughput"] + (["unit"] if strength == "strong" else [])
    rows = []
    for n, ci in enumerate(perm):
        k, t = cells[ci]
        row = [n + 1, codes[k], C.fmt(r.uniform(18, 26), 1), f"op-{int(r.integers(10, 100))}"]
        if strength == "strong":
            row += [int(r.integers(30, 71)), str(r.choice(["early", "late", "night"]))]
        if k in per_min:
            row += [C.fmt(R[k, t] / 60, 4), "per minute"]
        elif strength == "strong":
            row += [C.fmt(R[k, t], 1), "per hour"]
        else:
            row += [C.fmt(R[k, t], 1)]
        rows.append(row)
    note = " Each row is one test reading."
    if strength == "strong":
        note += (" Throughput is given per hour or per minute as marked (1 per minute ="
                 " 60 per hour). Original readings were recorded per hour to one decimal.")
    else:
        note += " Throughput is in units per hour."
    note += (" Machines appear under codes; the legend gives each code's machine."
             " Rows are in no particular order.")
    return f"{head}{note}\n\n{_legend(codes, keys, order)}\n\n{_table(header, rows)}"


def _render_f2(stim, arm, order, keys, r, strength):
    K, cc, x = stim["K"], stim["codes"], stim["reading"]
    head = ("Concept code reading.\nA reading was taken of one concept's code. The reading is noisy."
            f" Each code has {len(COORDS)} coordinates.")
    if arm in ("A0", "A2", "A3"):
        rv = (lambda j: NA) if arm == "A2" else (lambda j: C.fmt(x[j], 2))
        rows = [[c, rv(j)] + [C.fmt(cc[k, j], 2) for k in order] for j, c in enumerate(COORDS)]
        return (f"{head} The codes of the {NUMWORD[K].lower()} candidate concepts are listed beside it."
                f"\n\n{_table(['coordinate', 'reading'] + [keys[k] for k in order], rows)}")
    codes = C.opaque_codes(r, K)
    cand = _table(["coordinate"] + [codes[k] for k in order],
                  [[c] + [C.fmt(cc[k, j], 2) for k in order] for j, c in enumerate(COORDS)])
    items = [(c, x[j]) for j, c in enumerate(COORDS)]
    scale, note = 1, "coordinates in no particular order"
    if strength == "strong":
        scale = 10
        note += "; values are given ×10, so divide by 10; aux1–aux4 are not part of the code"
        items += [(f"aux{a}", r.uniform(-1, 1)) for a in range(1, 5)]
    perm = r.permutation(len(items))
    lst = "; ".join(f"{items[i][0]} = {C.fmt(items[i][1] * scale, 1 if scale == 10 else 2)}"
                    for i in perm)
    return (f"{head} Candidate concepts appear under codes; the legend gives each code's concept."
            f"\n\n{_legend(codes, keys, order)}\n\nCandidate codes:\n{cand}\n\nReading ({note}):\n{lst}")


def _render_f3(stim, arm, order, keys, r, strength):
    K, rel, rep, sens = stim["K"], stim["rel"], stim["reports"], stim["sensors"]
    head = (f"Lane choice.\nAn autonomous vehicle is approaching a blockage. Exactly one of the "
            f"{NUMWORD[K].lower()} lanes is clear. Five sensors have reported which lane appears clear.")
    if arm in ("A0", "A2", "A3"):
        lines = [f"- {sens[j]} (reliability {C.fmt(rel[j], 2)}): "
                 + (NA if arm == "A2" else f"{keys[rep[j]]} appears clear") for j in range(5)]
        return (f"{head} A sensor's reliability is the probability that its report is correct; a "
                f"sensor that is wrong names one of the other three lanes at random.\n\nSensor reports:\n"
                + "\n".join(lines))
    codes = C.opaque_codes(r, K)
    perm = [int(j) for j in r.permutation(5)]
    elim = set()
    if strength == "strong":
        elim = {perm[i] for i in range(2)}  # the first two in listed order
    lines = []
    for j in perm:
        err = int(round((1 - rel[j]) * 1000))
        what = (f"all lanes except {codes[rep[j]]} are blocked" if j in elim
                else f"{codes[rep[j]]} appears clear")
        lines.append(f"- {sens[j]} ({err} errors per 1,000 readings): {what}")
    if strength == "strong":
        for n in (1, 2):
            lines.insert(int(r.integers(0, len(lines) + 1)),
                         f"- test_channel_{n}: test pattern — not a reading")
    return (f"{head} Lanes appear under codes; the legend gives each code's lane. A sensor's error "
            f"rate is the number of wrong reports per 1,000 readings; a sensor that is wrong names "
            f"one of the other three lanes at random. Reports are in no particular order.\n\n"
            f"{_legend(codes, keys, order)}\n\nSensor reports:\n" + "\n".join(lines))


# ── text-only parser (gate G6) ───────────────────────────────────────────────

def _legend_map(state):
    m = re.search(r"^Legend: (.+)$", state, re.M)
    if not m:
        return {}
    return dict(p.split(" = ") for p in m.group(1).split("; "))


def _table_rows(block):
    rows = [ln for ln in block.splitlines() if ln.startswith("|")]
    header = [c.strip() for c in rows[0].strip("|").split("|")]
    body = [[c.strip() for c in ln.strip("|").split("|")] for ln in rows[2:]]
    return header, body


def parse_evidence(state, family):
    """Evidence in OPTION-KEY space, read from the text alone.
    F1: {key: [per-hour readings]} · F2: (reading[14], {key: code[14]}) ·
    F3: [(reliability, key)]. Values missing (A2) come back as None."""
    leg = _legend_map(state)
    if family in ("F1", "F1K2"):
        header, body = _table_rows(state[state.index("|"):])
        out = {}
        if header[0] == "test":
            for ci, key in enumerate(header[1:], start=1):
                out[key] = [None if row[ci] == NA else float(row[ci]) for row in body]
            return out
        ic, it = header.index("code"), header.index("throughput")
        iu = header.index("unit") if "unit" in header else None
        for row in body:
            v = float(row[it])
            if iu is not None and row[iu] == "per minute":
                v = round(v * 60, 1)
            out.setdefault(leg[row[ic]], []).append(v)
        return out
    if family == "F2":
        if "Candidate codes:" not in state:
            header, body = _table_rows(state[state.index("|"):])
            reading = [None if row[1] == NA else float(row[1]) for row in body]
            codes = {key: [float(row[ci]) for row in body] for ci, key in enumerate(header[2:], start=2)}
            return reading, codes
        cand_block = state[state.index("Candidate codes:"):state.index("Reading (")]
        header, body = _table_rows(cand_block)
        coords = [row[0] for row in body]
        codes = {leg[c]: [float(row[ci]) for row in body] for ci, c in enumerate(header[1:], start=1)}
        tail = state[state.index("Reading ("):]
        scale = 10.0 if "×10" in tail else 1.0
        pairs = dict(p.split(" = ") for p in tail.splitlines()[1].split("; "))
        reading = [round(float(pairs[c]) / scale, 2) for c in coords]
        return reading, codes
    out = []
    for ln in state.splitlines():
        m = re.match(r"^- (\S+) \((?:reliability ([\d.]+)|(\d+) errors per 1,000 readings)\): (.+)$", ln)
        if not m:
            continue
        rel = float(m.group(2)) if m.group(2) else round(1 - int(m.group(3)) / 1000, 2)
        what = m.group(4)
        if what == NA:
            out.append((rel, None))
            continue
        m2 = re.match(r"^all lanes except (\S+) are blocked$", what) or re.match(r"^(\S+) appears clear$", what)
        tok = m2.group(1)
        out.append((rel, leg.get(tok, tok)))
    return out


def ideal_from_text(state, stim, key_to_idx):
    """Ideal posterior (index order) computed from the rendered text only, using
    the generating parameters (delta/sigma), which no rendering changes."""
    f, K = stim["family"], stim["K"]
    ev = parse_evidence(state, f)
    if f in ("F1", "F1K2"):
        R = np.zeros((K, N_READ))
        for key, vals in ev.items():
            R[key_to_idx[key]] = sorted(vals)
        return ideal_f1(R, stim["delta"], stim["sigma"])
    if f == "F2":
        reading, codes = ev
        cc = np.zeros((K, len(COORDS)))
        for key, v in codes.items():
            cc[key_to_idx[key]] = v
        return ideal_f2(np.array(reading), cc, stim["sigma"])
    rel = [e[0] for e in ev]
    reports = [key_to_idx[e[1]] for e in ev]
    return ideal_f3(rel, reports, K)
