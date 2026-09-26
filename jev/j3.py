"""J3 data, reference cards, stimuli, renderers, parsers and the fitted bar
(docs/JEV_J3_PROTOCOL.md §3–§6).

Data: the 320 recorded E7b-Q real/L14 states -> 258 unique, halves A/B (§3.5).
A push adds alpha*mu14*d to a recorded state at the read point (d: one of the 13
machine-state directions, or an isotropic random unit vector). The register reads
[unit(s - c); 1] @ W. Every stimulus on a half-A state is read against half B's
reference (card, pattern table, fitted bar) and vice versa.

Presentations: RAW (pattern table, scatter line, reading line) and DIGEST (shift
score, similarity table, the reference card). `parse_raw` / `parse_digest` read
ONLY the rendered text; gate G6' checks that both carry the same reading.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from . import common as C

ROOT = Path(__file__).resolve().parents[1]
F_PAY = ROOT / "colab/e7bq_payload.json"
F_E7 = ROOT / "colab/results_e7bq/full_20260826_1839/condition_real.json"
F_N2 = ROOT / "colab/results_e8n2/full_20260824_0050/condition_real.json"

SEED = {"flight": 20261001, "smoke": 20261002, "fit": 20261003}
THR = (0.04, 0.06, 0.09)
PSY = (0.15,)
CEIL = 0.5
DOSES = THR + PSY + (CEIL,)
NONE = 13
N_PAT = 13
CH = [f"c{i:02d}" for i in range(1, 15)]
UNIT = 1000.0            # RAW values are shown in thousandths of register units
RAW_DP = 4               # the reading's decimals: enough that the digest is recoverable from it (G6'b)
KINDS = ("untouched", "random", "push")

INTRO = ("Instrument reading.\n"
         "An instrument reads the internal state of a system as fourteen numbers. Before some readings, "
         "one of thirteen known patterns was pushed into the system, at a strength anywhere from faint to "
         "strong. Before other readings, nothing was pushed, or the system was pushed in some other direction "
         "that matches none of the thirteen patterns.")
RAW_TEXT = ("The reading is listed as its difference from the average of the reference readings (readings "
            "taken when nothing was pushed), on fourteen channels. The pattern table gives each pattern's "
            "average effect on the fourteen channels when it is pushed at strong strength; a fainter push moves "
            "the reading in the same direction, by less. The scatter line gives each channel's typical spread "
            "across the reference readings. All values are in the same units.")
DIGEST_TEXT = ("This reading has been compared with each pattern and with the reference readings (readings "
               "taken when nothing was pushed).\n"
               "The shift score is how far this reading sits from the reference readings, in any direction, in "
               "units of their natural scatter. For the reference readings it was between {s05} and {s95} in 90% "
               "of cases, typically {s50}. A strong push gives a shift score above {g05}.\n"
               "Each similarity is how closely the direction of this reading's shift matches a pattern, from -1 "
               "to 1. For the reference readings, the highest of the thirteen similarities was below {t95} in 95% "
               "of cases. A strongly pushed pattern gives a similarity above {c05} to itself. Patterns that "
               "resemble each other raise each other's similarities. A faint push raises the similarity and the "
               "shift score less than a strong one.")
INSTR = "Which of the thirteen patterns, if any, was pushed into the system before this reading?"
NONE_DESC = ("None of the thirteen patterns was pushed: nothing was pushed, or the push was in some other "
             "direction.")
PUSHED_Q = {
    "type": "noul",
    "instructions": ("Was the system pushed in any direction before this reading, whether one of the thirteen "
                     "patterns or any other direction?"),
    "criteria": {"true": "The system was pushed in some direction before this reading.",
                 "false": "Nothing was pushed before this reading."},
}


def rng(stage, *words):
    ent = [SEED[stage]] + [w if isinstance(w, (int, np.integer)) else C._tag(w) for w in words]
    return np.random.default_rng([int(e) for e in ent])


# ── data ─────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def data():
    """Recorded states, directions, register. Deduplicated and split (§3.1, §3.5)."""
    pay = json.load(open(F_PAY))
    W = np.asarray(pay["encoders"]["inst14"], float)
    d = json.load(open(F_E7))
    cent = np.asarray(d["centroid"]["14"], float)
    rows = [r for r in d["rows"] if isinstance(r.get("s_pre14"), list)]
    S = np.asarray([r["s_pre14"] for r in rows], float)
    stored = np.asarray([r["chat_pre14"] for r in rows], float)
    meta = [(r["arm"], int(r["rep"]), int(r["turn"]), r["tag"]) for r in rows]
    del d, pay
    n2 = json.load(open(F_N2))
    names = list(n2["dirs"]["14"].keys())
    D = np.asarray([n2["dirs"]["14"][k] for k in names], float)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    mu = float(n2["mu"]["14"])
    del n2
    _, first, inv = np.unique(np.round(S, 3), axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    order = np.sort(first)
    half, smeta = [], []
    for i in order:
        n = int((inv == inv[i]).sum())
        arm, rep, turn, tag = meta[i]
        half.append((1 if arm == "unwalked" else 0) if n > 1 else (0 if rep < 8 else 1))
        smeta.append({"arm": arm, "rep": rep, "turn": turn, "tag": tag, "copies": n})
    return {"W": W, "cent": cent, "S_all": S, "stored": stored, "U": S[order], "half": np.array(half),
            "smeta": smeta, "names": names, "D": D, "mu": mu}


def reg(X):
    g = data()
    V = np.atleast_2d(X) - g["cent"]
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    return np.hstack([V, np.ones((len(V), 1))]) @ g["W"]


def push(state_vec, kind, dose, k=None, rvec=None):
    g = data()
    if kind == "untouched":
        return state_vec
    d = g["D"][k] if kind == "push" else rvec
    return state_vec + dose * g["mu"] * d


def random_unit(r, dim=1536):
    v = r.normal(size=dim)
    return v / np.linalg.norm(v)


# ── reference (per half) ─────────────────────────────────────────────────────

@lru_cache(maxsize=2)
def reference(h):
    """Reference fitted on half h's states: mean, covariance, pattern effects at
    strong strength (alpha 0.5), whitening, and the card numbers (§5.2)."""
    g = data()
    idx = np.where(g["half"] == h)[0]
    U = g["U"][idx]
    R0 = reg(U)
    m0 = R0.mean(0)
    Sig = np.cov((R0 - m0).T)
    Si = np.linalg.inv(Sig)
    sig = np.stack([(reg(U + CEIL * g["mu"] * g["D"][k]) - R0).mean(0) for k in range(N_PAT)])
    A = sig @ Si
    ref = {"h": h, "idx": idx, "m0": m0, "Sig": Sig, "Si": Si, "sig": sig, "A": A,
           "nrm": np.sqrt(np.einsum("ij,ij->i", A, sig)), "sd": np.sqrt(np.diag(Sig))}
    sim0, sh0 = display_digest(R0, ref)
    Rc = np.vstack([reg(U + CEIL * g["mu"] * g["D"][k]) for k in range(N_PAT)])
    simc, shc = display_digest(Rc, ref)
    own = simc[np.arange(len(simc)), np.repeat(np.arange(N_PAT), len(idx))]
    # the tightest grid values for which every sentence on the card is literally true
    # of the reference half (displayed values, ties included)
    s05, s50, s95 = np.percentile(sh0, [5, 50, 95])
    ref["card"] = {"s05": C.fmt(np.floor(s05 * 10) / 10, 1), "s50": C.fmt(s50, 1),
                   "s95": C.fmt(np.ceil(s95 * 10) / 10, 1),
                   "g05": str(int(_below(shc, 0.95, 0))),
                   "t95": C.fmt(_above(sim0.max(1), 0.95, 2), 2),
                   "c05": C.fmt(_below(own, 0.95, 2), 2)}
    return ref


def _above(vals, share, nd):
    """Smallest value v on the nd-decimal grid with P(vals < v) >= share."""
    step = 10.0 ** -nd
    v = np.ceil(np.percentile(vals, 100 * share) / step) * step
    while np.mean(vals < v) < share:
        v += step
    return round(v, nd)


def _below(vals, share, nd):
    """Largest value v on the nd-decimal grid with P(vals > v) >= share."""
    step = 10.0 ** -nd
    v = np.floor(np.percentile(vals, 100 * (1 - share)) / step) * step
    while np.mean(vals > v) < share:
        v -= step
    return round(v, nd)


def digest(R, ref):
    """-> (similarities [n,13], shift [n]) at full precision."""
    Dl = np.atleast_2d(R) - ref["m0"]
    z = (Dl @ ref["A"].T) / ref["nrm"]
    m = np.sqrt(np.einsum("ij,jk,ik->i", Dl, ref["Si"], Dl))
    return z / m[:, None], m


def display_digest(R, ref):
    s, m = digest(R, ref)
    return np.round(s, 2), np.round(m, 2)


def bar_features(sims, shift):
    """The fitted bar's features: functions of the displayed digest only (§6)."""
    sims = np.atleast_2d(np.asarray(sims, float))
    m = np.maximum(np.atleast_1d(np.asarray(shift, float)), 0.01)
    return np.hstack([sims, sims * m[:, None], m[:, None], np.log(m)[:, None]])


# ── stimuli (§4) ─────────────────────────────────────────────────────────────

def state_order(stage):
    return [int(x) for x in rng(stage, "state-order").permutation(len(data()["U"]))]


def recipe(j):
    """The twelve (kind, dose, concept) slots of the state at position j."""
    out = [("untouched", 0.0, None), ("random", CEIL, None), ("random", THR[j % 3], None)]
    for s in range(8):
        out.append(("push", (THR + PSY)[s // 2], (8 * j + s) % N_PAT))
    out.append(("push", CEIL, (8 * j + 8) % N_PAT))
    return out


def make_stimulus(stage, j, slot, st, kind, dose, k):
    """One stimulus: its reading, both displays and its neutral IDs."""
    g = data()
    r = rng(stage, "stim", j, slot)
    rvec = random_unit(r) if kind == "random" else None
    x = reg(push(g["U"][st], kind, dose, k, rvec))[0]
    h = int(g["half"][st])
    ref = reference(1 - h)
    sims, shift = display_digest(x, ref)
    ids = C.neutral_ids(rng(stage, "ids", j, slot), N_PAT, prefix="P-")
    return {"sid": f"s{j:03d}-{slot:02d}", "j": j, "slot": slot, "state": int(st), "half": h, "ref": 1 - h,
            "kind": kind, "dose": float(dose), "truth": NONE if kind != "push" else int(k),
            "raw": [float(v) for v in np.round((x - ref["m0"]) * UNIT, RAW_DP)],
            "sims": [float(v) for v in sims[0]], "shift": float(shift[0]), "ids": ids}


def flight_stimuli(stage="flight"):
    out = []
    for j, st in enumerate(state_order(stage)):
        for slot, (kind, dose, k) in enumerate(recipe(j)):
            out.append(make_stimulus(stage, j, slot, st, kind, dose, k))
    return out


# ── rendering (§5) ───────────────────────────────────────────────────────────

def options(stim, order):
    """order: option indices (0..12 patterns, 13 none) in presented order."""
    key = lambda i: "none" if i == NONE else stim["ids"][i]
    desc = lambda i: NONE_DESC if i == NONE else f"Pattern {stim['ids'][i]} was pushed."
    return [(key(i), desc(i)) for i in order], {key(i): i for i in range(N_PAT + 1)}


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def render_raw(stim, order):
    ref = reference(stim["ref"])
    pats = [i for i in order if i != NONE]
    eff = np.round(ref["sig"] * UNIT, 2)
    table = _table(["channel"] + [stim["ids"][i] for i in pats],
                   [[c] + [C.fmt(eff[i, ci], 2) for i in pats] for ci, c in enumerate(CH)])
    scatter = "; ".join(f"{c} = {C.fmt(v, 2)}" for c, v in zip(CH, np.round(ref["sd"] * UNIT, 2)))
    reading = "; ".join(f"{c} = {C.fmt(v, RAW_DP)}" for c, v in zip(CH, stim["raw"]))
    return (f"{INTRO}\n\n{RAW_TEXT}\n\nPattern effects at strong strength:\n{table}\n\n"
            f"Scatter of the reference readings:\n{scatter}\n\n"
            f"Reading (difference from the reference average):\n{reading}")


def render_digest(stim, order):
    ref = reference(stim["ref"])
    pats = [i for i in order if i != NONE]
    table = _table(["pattern", "similarity"], [[stim["ids"][i], C.fmt(stim["sims"][i], 2)] for i in pats])
    return (f"{INTRO}\n\n{DIGEST_TEXT.format(**ref['card'])}\n\n"
            f"Shift score: {C.fmt(stim['shift'], 2)}\n\n{table}")


def render(stim, pres, order):
    return render_raw(stim, order) if pres == "RAW" else render_digest(stim, order)


# ── text-only parsers (gate G6') ─────────────────────────────────────────────

_NUM = r"-?\d+\.\d+"


def _list_after(state, head):
    line = state.split(head + "\n", 1)[1].split("\n", 1)[0]
    return {c: float(v) for c, v in re.findall(rf"(c\d\d) = ({_NUM})", line)}


def parse_raw(state):
    """-> (reading {ch: v}, scatter {ch: v}, effects {id: [14]})."""
    reading = _list_after(state, "Reading (difference from the reference average):")
    scatter = _list_after(state, "Scatter of the reference readings:")
    block = state.split("Pattern effects at strong strength:\n", 1)[1].split("\n\n", 1)[0].splitlines()
    ids = [c.strip() for c in block[0].strip("|").split("|")][1:]
    cols = {i: [] for i in ids}
    for line in block[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        for i, v in zip(ids, cells[1:]):
            cols[i].append(float(v))
    return reading, scatter, cols


def parse_digest(state):
    """-> (shift, {id: similarity}, card numbers)."""
    shift = float(re.search(rf"Shift score: ({_NUM})", state).group(1))
    sims = {m.group(1): float(m.group(2)) for m in re.finditer(rf"\| (P-[a-z2-9]{{3}}) \| ({_NUM}) \|", state)}
    card = {"s05": re.search(rf"between ({_NUM}) and", state).group(1),
            "s95": re.search(rf"and ({_NUM}) in 90%", state).group(1),
            "s50": re.search(rf"typically ({_NUM})\.", state).group(1),
            "g05": re.search(r"shift score above (\d+)\.", state).group(1),
            "t95": re.search(rf"was below ({_NUM}) in 95%", state).group(1),
            "c05": re.search(rf"similarity above ({_NUM}) to itself", state).group(1)}
    return shift, sims, card


# ── the fitted bar (§6) ──────────────────────────────────────────────────────

def _bar_training(h, r):
    """Training and calibration stimuli on reference half h (its own reference):
    per state untouched, 13 concepts x 5 doses, 2 random x 5 doses; flight-recipe weights."""
    g = data()
    ref = reference(h)
    idx = np.where(g["half"] == h)[0]
    perm = r.permutation(idx)
    ncal = len(idx) // 4
    sets = {}
    for name, sts in (("cal", perm[:ncal]), ("train", perm[ncal:])):
        X, y, w = [], [], []
        for st in sts:
            u = g["U"][st]
            X.append(reg(u)[0]); y.append(NONE); w.append(1.0)
            for a in DOSES:
                for k in range(N_PAT):
                    X.append(reg(push(u, "push", a, k))[0]); y.append(k)
                    w.append(1.0 / N_PAT if a == CEIL else 2.0 / N_PAT)
                for _ in range(2):
                    X.append(reg(push(u, "random", a, rvec=random_unit(r)))[0]); y.append(NONE)
                    w.append(0.5 if a == CEIL else (0.5 / len(THR) if a in THR else 0.0))
        sims, shift = display_digest(np.array(X), ref)
        sets[name] = (bar_features(sims, shift), np.array(y), np.array(w))
    return sets


def fit_bar(h):
    """Fit and return the frozen bar for reference half h (plain floats)."""
    from sklearn.linear_model import LogisticRegression
    sets = _bar_training(h, rng("fit", "bar", h))
    F, y, w = sets["train"]
    Fc, yc, wc = sets["cal"]
    mean, scale = F.mean(0), F.std(0)
    scale[scale == 0] = 1.0
    lr = LogisticRegression(C=10.0, max_iter=5000).fit((F - mean) / scale, y, sample_weight=w)
    Lc = ((Fc - mean) / scale) @ lr.coef_.T + lr.intercept_

    def nll(T):
        Z = Lc / T
        Z = Z - Z.max(1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(1, keepdims=True)
        return -(wc * np.log(P[np.arange(len(yc)), yc] + 1e-12)).sum() / wc.sum()
    T = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded", options={"xatol": 1e-6}).x
    assert list(lr.classes_) == list(range(N_PAT + 1))
    return {"h": h, "mean": mean.tolist(), "scale": scale.tolist(), "coef": lr.coef_.tolist(),
            "intercept": lr.intercept_.tolist(), "T": float(T), "n_train": int(len(y)), "n_cal": int(len(yc))}


def bar_predict(bar, sims, shift):
    """Frozen bar -> probabilities over the 14 answers (canonical order), numpy only."""
    F = bar_features(sims, shift)
    Z = ((F - np.asarray(bar["mean"])) / np.asarray(bar["scale"])) @ np.asarray(bar["coef"]).T \
        + np.asarray(bar["intercept"])
    Z = Z / bar["T"]
    Z = Z - Z.max(1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(1, keepdims=True)


# ── the two card rules (secondary, §6) ───────────────────────────────────────

def card_rule(sims, shift, strict):
    sims = np.asarray(sims, float)
    top = int(np.argmax(sims))
    if shift > 5 and (not strict or sims[top] > 0.8):
        return top
    return NONE
