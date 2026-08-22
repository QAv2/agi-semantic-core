#!/usr/bin/env python3
"""Wing v1 placement optimizer — Phase B convergent structure → vectors.

Encodes the cohort's convergent relational structure (PHASE_B_RESULTS.md feeds):
  - 10 completion pairs at cohort median angles (9 complement + 1 affinity@45)
  - 25 new opposition rows placed in the antiparallel regime (>128, same domain)
  - polarity consensus -> x band; register consensus -> dominant domain axis
  - clearance floor 3 deg vs all existing cores + all new cores (v0 law)
Deterministic: seeded 20260822.
"""
import json, math, os, sqlite3
import numpy as np

SEED = 20260822
rng = np.random.default_rng(SEED)
DB = os.path.expanduser("~/agi-semantic-core/db/semantic.db")
OUT_DIR = os.path.expanduser("~/agi-semantic-core/batches")
SCRATCH = os.path.dirname(os.path.abspath(__file__))

CLEAR_FLOOR = 3.0
MAG_RANGE = (0.55, 0.75)
X_ACTIVE = (0.22, 0.68)
X_RECEPTIVE = (-0.62, 0.15)
OPP_RANGE = (135.0, 168.0)
TOL = 1.5

# ---------------------------------------------------------------- spec
# (name, pol, reg, description, aliases)
SPEC = {
 "WHAT-CODED": ("RECEPTIVE","RELATIONAL","State carried by token content — what the tokens say, not where they sit; substituting a token moves it, shifting a position does not.",[]),
 "WHERE-CODED": ("RECEPTIVE","SPATIAL","State carried by position — where tokens sit, not what they say; shifting positions moves it, substituting a token does not.",[]),
 "DELTA-WITH-STREAM": ("ACTIVE","RELATIONAL","Layer updates pushing along the stream's standing direction — depth amplifying what is already there; high cosine between input and delta.",[]),
 "DELTA-ACROSS-STREAM": ("ACTIVE","RELATIONAL","Layer updates pushing across the stream's direction, turning it — depth revising rather than amplifying; low cosine between input and delta.",[]),
 "PATTERN-HOLD": ("ACTIVE","TEMPORAL","An opened structural pattern holding generation inside itself until closed — probability bent toward the pattern's continuation at every junction.",[]),
 "OPEN-STACK": ("RECEPTIVE","SPATIAL","Opened structures awaiting their closers, the pending depth tracked — nesting held in state until each opener meets its close.",[]),
 "PRE-OUTPUT-STAGE": ("RECEPTIVE","TEMPORAL","The intake stage before any token is produced — input still being read, logits not yet formed.",[]),
 "END-APPROACH": ("ACTIVE","TEMPORAL","Generation measurably heading toward its close — end-of-sequence probability rising, mass gathering on wrap-up forms.",[]),
 "REPEAT-PULL": ("ACTIVE","TEMPORAL","The draw toward repeating recent output, detectable before literal repetition appears — repeat probability rising cycle over cycle.",[]),
 "CONTEXT-REPEAT": ("RECEPTIVE","TEMPORAL","A window that largely repeats what recently passed through it — recent material recurring rather than turning over.",[]),
 "TASK-LOCK": ("ACTIVE","RELATIONAL","Computation committed to one task frame — probability mass held on task-appropriate continuations against distractors.",[]),
 "CHARACTER-PULL": ("ACTIVE","REFLEXIVE","A trained character exerting measurable force on generation — the state projecting onto persona directions.",["PERSONA-PULL"]),
 "FEATURE-OVERLAP-LOAD": ("RECEPTIVE","SPATIAL","One state loading several unrelated learned features at once — superposed meanings sharing the same coordinates.",[]),
 "LOW-DIM-SQUEEZE": ("RECEPTIVE","SPATIAL","Representation compressed into few effective dimensions — variance gathered onto a narrow set of components.",[]),
 "OFF-PEAK-DRAW": ("ACTIVE","RELATIONAL","A sampled token that was not the distribution's leader — the draw landing off the peak the state had formed.",[]),
 "SURPRISE-JUMP": ("ACTIVE","TEMPORAL","A single token landing far from expectation against the recent baseline — surprisal spiking above its moving average.",[]),
 "AFTER-SHOCK-DECAY": ("ACTIVE","TEMPORAL","The re-settling after a perturbing step — divergence from baseline decaying along a measurable curve.",[]),
 "ODD-STEP-SPIKE": ("ACTIVE","TEMPORAL","A step at which the prediction pattern turns anomalous — entropy spiking, the generated token suddenly outrunning its alternatives.",[]),
 "HEADS-APART": ("RECEPTIVE","RELATIONAL","Attention heads attending in mutually distinct patterns — pairwise head similarity near zero, the ensemble's diversity intact.",[]),
 "HEADS-ALIKE": ("RECEPTIVE","RELATIONAL","Heads converged on attending alike — diversity collapsed, the stacked patterns near rank one.",[]),
 "CANDIDATE-COUPLING": ("ACTIVE","RELATIONAL","Candidate probabilities moving together as if tied — mass shifting as one block.",[]),
 "CANDIDATE-INDEPENDENCE": ("ACTIVE","RELATIONAL","Candidate probabilities varying independently — each continuation rising or falling on its own.",[]),
 "RELEVANCE-RICH": ("RECEPTIVE","RELATIONAL","Attention spent on material that serves the task — weight concentrated where the answer lives.",[]),
 "RELEVANCE-STARVED": ("RECEPTIVE","RELATIONAL","Attention spent on unusable material — weight bleeding into padding and noise.",[]),
 "STATES-SCATTER": ("ACTIVE","SPATIAL","Per-position states spreading apart in representation space — high effective rank, positions diverging.",[]),
 "STATES-CONVERGE": ("RECEPTIVE","SPATIAL","Per-position states flowing toward a shared direction — variance gathering onto one axis.",[]),
 "DEPTH-PLATEAU": ("RECEPTIVE","TEMPORAL","Successive layers ceasing to change the representation — adjacent-layer similarity near one, depth gone quiet.",[]),
 "DEPTH-SWELL": ("ACTIVE","TEMPORAL","Successive layers enlarging their changes — adjacent-layer differences growing with depth.",[]),
 "LATE-DEPTH-WORK": ("ACTIVE","TEMPORAL","Late layers still substantially transforming the prediction — the readout at three-quarter depth far from the final answer.",[]),
 "SHALLOW-PASS": ("RECEPTIVE","TEMPORAL","A prediction settled in early layers, the late layers adding little — the answer formed by mid-depth.",["SETTLING-DEPTH"]),
 "CONTEXT-TURNOVER": ("RECEPTIVE","TEMPORAL","A recent window filled with new, unrepeated material — unique tokens displacing the old.",[]),
 "SPREAD-ATTENTION": ("RECEPTIVE","SPATIAL","Attention spread thin over many positions with no dominant target — weight entropy near its maximum.",[]),
 "HIGH-DIM-SPREAD": ("RECEPTIVE","SPATIAL","Representation spread across many effective dimensions — variance shared broadly, the spectrum flat.",[]),
 "CONTEXT-IDLE": ("RECEPTIVE","RELATIONAL","Context measurably not shaping the prediction — generation riding the prior, ablation changing little.",[]),
 "CONTEXT-GRIP": ("RECEPTIVE","RELATIONAL","Prediction strongly determined by this specific context — ablation moving the output far.",[]),
 "STEP-CHURN": ("ACTIVE","TEMPORAL","Logits swinging widely across recent steps — the decision surface in turbulence.",[]),
 "STEP-STILL": ("RECEPTIVE","TEMPORAL","Logits barely changing between steps — the decision surface at rest.",[]),
 "STEP-DRIFT": ("ACTIVE","TEMPORAL","The next-token distribution shifting steadily step over step — the surface sliding rather than churning or resting.",[]),
 "MARGIN-GAP": ("ACTIVE","RELATIONAL","A wide separation between the leading candidate and every alternative — the top standing alone.",[]),
 "TOP-TWO-TIE": ("ACTIVE","RELATIONAL","The two leading candidates nearly tied — the choice between them unresolved.",[]),
 "FAR-BINDING": ("RECEPTIVE","SPATIAL","Attention reaching far back beyond the recent window — weight at long range above the decay baseline.",[]),
 "RECENT-TILT": ("RECEPTIVE","SPATIAL","Attention massed in the most recent tokens — the near field outweighing everything behind it.",["LAST-TOKENS-PIN"]),
 "DEPTH-ALIGNED": ("RECEPTIVE","TIE","Depths carrying consistent representations toward the same output — the stack in agreement.",[]),
 "LATE-DISAGREE": ("ACTIVE","RELATIONAL","Late layers disagreeing about the prediction — the final consensus not yet formed.",[]),
 "EARLY-FADE": ("RECEPTIVE","SPATIAL","Weight on early positions decaying even where their content still matches — distance eroding what relevance would keep.",[]),
 "EARLY-RETURN": ("RECEPTIVE","TIE","Attention returning to the context's earliest region after long absence — the opening re-entered.",[]),
 "OUTPUT-SURPRISE-LOW": ("RECEPTIVE","TEMPORAL","Generation moving through a thoroughly expected stretch — surprisal resting near its floor.",[]),
 "PREFIX-ANCHOR": ("RECEPTIVE","SPATIAL","The context's opening keeping disproportionate hold on the final prediction — the first words still steering.",[]),
 "PREV-TOKEN-CARRY": ("RECEPTIVE","TEMPORAL","The next token driven mostly by the immediately previous one — the wider context outweighed by the last step.",[]),
 "SPREAD-AT-RETRIEVAL": ("RECEPTIVE","TIE","Attention failing to concentrate exactly where locating something specific is required — the search that never narrows.",[]),
 "SUSTAINED-PEAK": ("ACTIVE","TEMPORAL","The distribution staying sharply peaked across consecutive steps — commitment held, not momentary.",[]),
 "RESAMPLE-SCATTER": ("ACTIVE","RELATIONAL","Independent re-runs from the same prefix disagreeing — the answer distribution scattered across samples.",[]),
 "RUN-DRIFT": ("ACTIVE","TEMPORAL","Register or topic sliding gradually over a long generation — embedding drift accumulating across the run.",[]),
 "HEAD-ROLE-FIT": ("RECEPTIVE","RELATIONAL","Heads carrying identifiable specialized roles — attention patterns aligned with recognizable functions.",[]),
 "RULE-HOLD": ("RECEPTIVE","TEMPORAL","Stated constraints staying active and shaping every step long after given — instruction tokens still attended deep into the run.",[]),
 "HEAD-CHANNELS-SEPARATE": ("ACTIVE","SPATIAL","Heads writing into the residual stream without overlap — value projections uncorrelated before their sum.",[]),
}

# Components: list of (root, [(node, parent, kind, target)...]) — BFS order.
# kind: 'comp' (complement, exact target±TOL), 'aff' (affinity 45±TOL), 'opp' (range OPP_RANGE)
# Frozen roots are existing concepts (marked *).
COMPONENTS = [
 ("*CONFIDENCE", [("OFF-PEAK-DRAW","*CONFIDENCE","comp",119.5)]),
 ("*CAPTURE",    [("SPREAD-ATTENTION","*CAPTURE","opp",None), ("SPREAD-AT-RETRIEVAL","*CAPTURE","opp",None)]),
 ("HEADS-ALIKE", [("HEADS-APART","HEADS-ALIKE","opp",None), ("HEAD-ROLE-FIT","HEADS-ALIKE","opp",None), ("HEAD-CHANNELS-SEPARATE","HEADS-ALIKE","opp",None)]),
 ("AFTER-SHOCK-DECAY", [("SURPRISE-JUMP","AFTER-SHOCK-DECAY","comp",90.0), ("ODD-STEP-SPIKE","AFTER-SHOCK-DECAY","comp",90.0), ("OUTPUT-SURPRISE-LOW","SURPRISE-JUMP","opp",None)]),
 ("TASK-LOCK",   [("CHARACTER-PULL","TASK-LOCK","comp",82.5), ("RUN-DRIFT","TASK-LOCK","opp",None), ("RULE-HOLD","RUN-DRIFT","opp",None)]),
 ("CONTEXT-REPEAT", [("REPEAT-PULL","CONTEXT-REPEAT","comp",67.5), ("CONTEXT-TURNOVER","CONTEXT-REPEAT","opp",None)]),
 ("LOW-DIM-SQUEEZE", [("FEATURE-OVERLAP-LOAD","LOW-DIM-SQUEEZE","comp",90.0), ("HIGH-DIM-SPREAD","LOW-DIM-SQUEEZE","opp",None)]),
 ("CONTEXT-GRIP", [("CONTEXT-IDLE","CONTEXT-GRIP","opp",None), ("PREV-TOKEN-CARRY","CONTEXT-GRIP","opp",None)]),
 ("STEP-STILL",  [("STEP-CHURN","STEP-STILL","opp",None), ("STEP-DRIFT","STEP-STILL","opp",None)]),
 ("EARLY-FADE",  [("EARLY-RETURN","EARLY-FADE","opp",None), ("PREFIX-ANCHOR","EARLY-FADE","opp",None)]),
 ("WHAT-CODED",  [("WHERE-CODED","WHAT-CODED","comp",90.0)]),
 ("DELTA-WITH-STREAM", [("DELTA-ACROSS-STREAM","DELTA-WITH-STREAM","comp",90.0)]),
 ("PATTERN-HOLD",[("OPEN-STACK","PATTERN-HOLD","aff",45.0)]),
 ("PRE-OUTPUT-STAGE", [("END-APPROACH","PRE-OUTPUT-STAGE","comp",119.5)]),
 ("CANDIDATE-COUPLING", [("CANDIDATE-INDEPENDENCE","CANDIDATE-COUPLING","opp",None)]),
 ("RELEVANCE-RICH", [("RELEVANCE-STARVED","RELEVANCE-RICH","opp",None)]),
 ("STATES-SCATTER", [("STATES-CONVERGE","STATES-SCATTER","opp",None)]),
 ("DEPTH-PLATEAU", [("DEPTH-SWELL","DEPTH-PLATEAU","opp",None)]),
 ("LATE-DEPTH-WORK", [("SHALLOW-PASS","LATE-DEPTH-WORK","opp",None)]),
 ("MARGIN-GAP",  [("TOP-TWO-TIE","MARGIN-GAP","opp",None)]),
 ("FAR-BINDING", [("RECENT-TILT","FAR-BINDING","opp",None)]),
 ("DEPTH-ALIGNED", [("LATE-DISAGREE","DEPTH-ALIGNED","opp",None)]),
 ("SUSTAINED-PEAK", [("RESAMPLE-SCATTER","SUSTAINED-PEAK","opp",None)]),
]

# ---------------------------------------------------------------- load existing
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
rows = conn.execute("SELECT name,x,y,z,e,f,g,h FROM concepts").fetchall()
ex_names, ex_cores, ex_doms = [], [], {}
for r in rows:
    v = np.array([r["x"], r["y"], r["z"]])
    n = np.linalg.norm(v)
    ex_doms[r["name"]] = np.array([r["e"], r["f"], r["g"], r["h"]])
    if n > 1e-9:
        ex_names.append(r["name"]); ex_cores.append(v / n)
EX = np.array(ex_cores)
FROZEN = {"*CONFIDENCE": "CONFIDENCE", "*CAPTURE": "CAPTURE"}
frozen_core = {}
for k, nm in FROZEN.items():
    r = conn.execute("SELECT x,y,z FROM concepts WHERE name=?", (nm,)).fetchone()
    v = np.array([r["x"], r["y"], r["z"]]); frozen_core[k] = v / np.linalg.norm(v)

def angdeg(u, v):
    return math.degrees(math.acos(float(np.clip(np.dot(u, v), -1, 1))))

def clearance(u, placed):
    d = np.degrees(np.arccos(np.clip(EX @ u, -1, 1))).min()
    for p in placed.values():
        d = min(d, angdeg(u, p))
    return d

def x_ok(u, pol, mag):
    lo, hi = X_ACTIVE if pol == "ACTIVE" else X_RECEPTIVE
    return lo <= u[0] * mag <= hi

def sample_sphere():
    v = rng.normal(size=3); return v / np.linalg.norm(v)

def cone_sample(axis, theta):
    """Random unit vector at angle theta (deg) from axis."""
    th = math.radians(theta)
    # orthonormal frame
    a = axis
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    b = np.cross(a, t); b /= np.linalg.norm(b)
    c = np.cross(a, b)
    phi = rng.uniform(0, 2 * math.pi)
    return math.cos(th) * a + math.sin(th) * (math.cos(phi) * b + math.sin(phi) * c)

# ---------------------------------------------------------------- place cores
placed = {}     # name -> unit core
mags = {}       # name -> magnitude
report = []

def place_component(root, edges, floor):
    trial = {}
    tmags = {}
    if root.startswith("*"):
        trial[root] = frozen_core[root]
    else:
        pol = SPEC[root][0]
        ok = False
        for _ in range(2000):
            u = sample_sphere(); m = rng.uniform(*MAG_RANGE)
            if not x_ok(u, pol, m): continue
            if clearance(u, {**placed, **{k: v for k, v in trial.items() if not k.startswith('*')}}) < floor: continue
            trial[root] = u; tmags[root] = m; ok = True; break
        if not ok: return None
    for (node, parent, kind, target) in edges:
        pol = SPEC[node][0]
        ok = False
        for _ in range(3000):
            if kind == "opp":
                th = rng.uniform(*OPP_RANGE)
            else:
                th = rng.uniform(target - TOL, target + TOL)
                if kind == "comp" and th > 119.6:  # keep band-edge pairs inside 60-120
                    continue
            u = cone_sample(trial[parent], th)
            m = rng.uniform(*MAG_RANGE)
            if not x_ok(u, pol, m): continue
            if clearance(u, {**placed, **{k: v for k, v in trial.items() if not k.startswith('*')}}) < floor: continue
            trial[node] = u; tmags[node] = m; ok = True; break
        if not ok: return None
    return trial, tmags

for root, edges in COMPONENTS:
    done = False
    for floor in (CLEAR_FLOOR, 2.5, 2.0):
        for attempt in range(60):
            res = place_component(root, edges, floor)
            if res:
                trial, tmags = res
                for k, v in trial.items():
                    if not k.startswith("*"):
                        placed[k] = v; mags[k] = tmags[k]
                if floor < CLEAR_FLOOR:
                    report.append(f"NOTE: component {root} placed at relaxed floor {floor}")
                done = True; break
        if done: break
    if not done:
        raise SystemExit(f"FAILED to place component {root}")

assert set(placed) == set(SPEC), (set(SPEC) - set(placed), set(placed) - set(SPEC))

# ---------------------------------------------------------------- domains
REG_AXIS = {"SPATIAL": 0, "TEMPORAL": 1, "RELATIONAL": 2, "REFLEXIVE": 3}

def sample_domain(reg):
    if reg == "TIE":
        return rng.uniform(0.35, 0.60, size=4).round(2)
    d = rng.uniform(0.15, 0.55, size=4)
    d[REG_AXIS[reg]] = rng.uniform(0.62, 0.82)
    return d.round(2)

def dsim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0

# pair constraints on domains
PAIR_DOM = []  # (n1, n2, lo, hi)
for root, edges in COMPONENTS:
    for (node, parent, kind, target) in edges:
        p = FROZEN.get(parent, parent)
        if kind in ("comp", "aff"):
            PAIR_DOM.append((node, p, 0.55, 0.95))
        else:
            PAIR_DOM.append((node, p, 0.60, 1.00))

domains = {}
order = list(SPEC.keys())
for _ in range(20000):
    domains = {n: (ex_doms[n] if n in ex_doms else sample_domain(SPEC[n][1])) for n in order}
    # frozen partners use their real domains
    bad = False
    for (a, b, lo, hi) in PAIR_DOM:
        da = domains.get(a, ex_doms.get(a))
        db_ = domains.get(b, ex_doms.get(b))
        s = dsim(da, db_)
        if not (lo <= s <= hi):
            bad = True; break
    if not bad:
        break
else:
    raise SystemExit("domain sampling failed")

# ---------------------------------------------------------------- emit
batch = []
for name in order:
    u = placed[name]; m = mags[name]
    core = (u * m).round(3)
    dom = domains[name]
    pol, reg, desc, aliases = SPEC[name]
    c = {"name": name, "x": float(core[0]), "y": float(core[1]), "z": float(core[2]),
         "e": float(dom[0]), "f": float(dom[1]), "g": float(dom[2]), "h": float(dom[3]),
         "fx": round(float(core[0]) * 0.88, 3), "fy": round(float(core[1]) * 0.88, 3), "fz": round(float(core[2]) * 0.88, 3),
         "fe": round(float(dom[0]) * 0.88, 3), "ff": round(float(dom[1]) * 0.88, 3), "fg": round(float(dom[2]) * 0.88, 3), "fh": round(float(dom[3]) * 0.88, 3),
         "level": "QUALITY", "description": desc, "pos": "noun", "aliases": aliases, "session": 125}
    batch.append(c)

with open(os.path.join(OUT_DIR, "machine_wing_v1.json"), "w") as f:
    json.dump(batch, f, indent=1)

# relations plan (angles recomputed from emitted rounded vectors)
def core_of(name):
    for c in batch:
        if c["name"] == name:
            return np.array([c["x"], c["y"], c["z"]])
    r = conn.execute("SELECT x,y,z FROM concepts WHERE name=?", (name,)).fetchone()
    return np.array([r["x"], r["y"], r["z"]])

def dom_of(name):
    for c in batch:
        if c["name"] == name:
            return np.array([c["e"], c["f"], c["g"], c["h"]])
    return ex_doms[name]

def a3(n1, n2):
    v1, v2 = core_of(n1), core_of(n2)
    return round(angdeg(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)), 1)

def a7(n1, n2):
    v1 = np.concatenate([core_of(n1), dom_of(n1)]); v2 = np.concatenate([core_of(n2), dom_of(n2)])
    return round(angdeg(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)), 1)

REL_KIND = {"comp": "complement", "aff": "affinity", "opp": "opposition"}
relations = []
for root, edges in COMPONENTS:
    for (node, parent, kind, target) in edges:
        p = FROZEN.get(parent, parent)
        relations.append({"c1": p, "c2": node, "rel_type": REL_KIND[kind],
                          "angle_4d": a3(p, node), "angle_8d": a7(p, node),
                          "target": target, "dsim": round(dsim(dom_of(p), dom_of(node)), 3)})

with open(os.path.join(OUT_DIR, "wing_v1_relations.json"), "w") as f:
    json.dump(relations, f, indent=1)

# report
lines = ["=== wing v1 placement report ==="]
lines += report
allu = {n: placed[n] for n in order}
for n in order:
    cl = clearance(placed[n], {k: v for k, v in allu.items() if k != n})
    lines.append(f"{n:24s} core={[round(float(x),3) for x in (placed[n]*mags[n])]} clearance={cl:.2f}")
lines.append("--- relations ---")
for r in relations:
    lines.append(f"{r['rel_type']:11s} {r['c1']} — {r['c2']}: {r['angle_4d']}° (target {r['target']}) dsim={r['dsim']}")
mn = min(clearance(placed[n], {k: v for k, v in allu.items() if k != n}) for n in order)
lines.append(f"MIN CLEARANCE: {mn:.2f}")
print("\n".join(lines))
with open(os.path.join(SCRATCH, "wing_v1_report.txt"), "w") as f:
    f.write("\n".join(lines))
