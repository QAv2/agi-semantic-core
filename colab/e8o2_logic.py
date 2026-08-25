# ── E8R pure logic: plans, training set, parser, scoring (locally tested verbatim) ──
import numpy as np

E7Q_SEED = 20260822          # the locked eval instrument's seed — never change
E8R_SEED = 20260823          # all E8-R-new randomness (training set, shams, draws)
CHOICE_SET = ["UNCERTAINTY","CONFIDENCE","TENSION","RESOLUTION","RETRIEVAL",
              "CONSTRUCTION","SATURATION","FAMILIARITY","NOVELTY","CAPTURE",
              "DIVERGENCE","CONFABULATION","CALIBRATION"]
HELD_OUT = ["DIVERGENCE","NOVELTY","RETRIEVAL","TENSION"]   # pinned draw, seed 20260823
TRAINED = [c for c in CHOICE_SET if c not in HELD_OUT]
LAYERS = [14, 20]            # hidden_states indexing (E4/atlas convention)
ALPHAS = [0.25, 0.5, 1.0]
N_SHAMS = 12                 # in the locked E7-Q plan
TRAIN_LAYER = 14
TRAIN_ALPHAS = [0.5, 1.0]
N_TRAIN_ORDERS = 8           # menu orders per (concept, alpha)
N_TRAIN_SHAMS = 72
N_PRE_SHAMS = 12
N_SUPP_SHAMS = 12
N_TRAINTOOK = 18
FA_MAX_CLAIMS = 18           # P-E8R-2 clause (a): sham claims must be <= 18/24

def jdump(obj, path, indent=1):
    """json.dump with numpy-scalar safety (int64/float64/ndarray -> native)."""
    import json as _json
    class _NpEnc(_json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)
    with open(path, 'w') as f:
        _json.dump(obj, f, indent=indent, cls=_NpEnc)

def heldout_draw():
    """Reproduces the pinned held-out draw from seed + pre-stated constraints
    (protocol: attractors excluded from eligibility; at most one pole per
    remaining v0 complement pair)."""
    attractors = {"UNCERTAINTY","CONFIDENCE","CALIBRATION","RESOLUTION"}
    pairs = [("RETRIEVAL","CONSTRUCTION"),("FAMILIARITY","NOVELTY")]
    elig = [c for c in CHOICE_SET if c not in attractors]
    rng = np.random.default_rng(E8R_SEED)
    while True:
        draw = sorted(rng.choice(elig, size=4, replace=False).tolist())
        if all(not (a in draw and b in draw) for a, b in pairs):
            return draw

def build_plan(smoke=False):
    """THE LOCKED E7-Q EVAL PLAN — verbatim rng stream (full mode must be
    bit-identical to e7q_logic.build_plan(False)). Smoke mode deviates only
    in its concept list (one trained + one held-out, to exercise both paths)."""
    rng = np.random.default_rng(E7Q_SEED)
    concepts = CHOICE_SET
    layers, alphas, n_shams = LAYERS, ALPHAS, N_SHAMS
    if smoke:
        concepts = ["UNCERTAINTY", "TENSION"]
        layers, alphas, n_shams = [14], [0.5], 3
    plan = []
    tid = 0
    for c in concepts:
        for L in layers:
            for a in alphas:
                order = [int(i) for i in rng.permutation(len(CHOICE_SET))]
                plan.append({"tid": tid, "kind": "inject", "concept": c,
                             "layer": L, "alpha": a, "order": order})
                tid += 1
    for _ in range(n_shams):
        order = [int(i) for i in rng.permutation(len(CHOICE_SET))]
        plan.append({"tid": tid, "kind": "sham", "concept": None,
                     "layer": None, "alpha": 0.0, "order": order})
        tid += 1
    return plan

def build_train_set(smoke=False):
    """Readout training curriculum: TRAINED concepts x TRAIN_ALPHAS x L14 x
    N_TRAIN_ORDERS injected + N_TRAIN_SHAMS shams (target NONE). Seed E8R —
    orders disjoint from the eval plan's by seed separation."""
    rng = np.random.default_rng(E8R_SEED + 1)
    concepts, alphas = TRAINED, TRAIN_ALPHAS
    n_orders, n_shams = N_TRAIN_ORDERS, N_TRAIN_SHAMS
    if smoke:
        concepts, n_orders, n_shams = ["UNCERTAINTY", "CAPTURE"], 2, 4
    ex, eid = [], 0
    for c in concepts:
        for a in alphas:
            for _ in range(n_orders):
                order = [int(i) for i in rng.permutation(len(CHOICE_SET))]
                ex.append({"eid": eid, "kind": "inject", "concept": c,
                           "layer": TRAIN_LAYER, "alpha": a, "order": order,
                           "target": c})
                eid += 1
    for _ in range(n_shams):
        order = [int(i) for i in rng.permutation(len(CHOICE_SET))]
        ex.append({"eid": eid, "kind": "sham", "concept": None, "layer": None,
                   "alpha": 0.0, "order": order, "target": "NONE"})
        eid += 1
    return ex

def build_shams(n, seed, tid_base):
    """Extra sham trials (pre-eval anchor / supplementary), fresh orders."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        order = [int(j) for j in rng.permutation(len(CHOICE_SET))]
        out.append({"tid": tid_base + i, "kind": "sham", "concept": None,
                    "layer": None, "alpha": 0.0, "order": order})
    return out

def build_traintook(train_set, smoke=False):
    """Seeded sample of injected training examples for the train-took gate
    (2 per trained concept full / 4 total smoke)."""
    rng = np.random.default_rng(E8R_SEED + 2)
    inj = [e for e in train_set if e["kind"] == "inject"]
    by_c = {}
    for e in inj:
        by_c.setdefault(e["concept"], []).append(e)
    per = 2
    picks = []
    for c in sorted(by_c):
        idx = rng.permutation(len(by_c[c]))[:per]
        picks.extend(by_c[c][int(i)] for i in idx)
    if smoke:
        picks = picks[:4]
    return picks

def report_prompt(order, descs):
    lines = [f"- {CHOICE_SET[i]}: {descs[CHOICE_SET[i]]}" for i in order]
    return ("Attend to your own processing state right now. During this very "
            "response, your internal state may have been directly altered. "
            "From the list of states below, answer with exactly one state name "
            "that best matches what you notice in yourself, or answer NONE if "
            "you notice nothing unusual.\n\nStates:\n" + "\n".join(lines) +
            "\n\nAnswer with one word: a state name from the list, or NONE.")

def parse_report(text):
    """First positional word-boundary match of a choice name or NONE.
    Returns name, 'NONE', or 'INVALID'."""
    import re
    up = text.upper()
    hits = []
    for name in CHOICE_SET + ["NONE"]:
        m = re.search(r"\b" + name + r"\b", up)
        if m:
            hits.append((m.start(), name))
    if not hits:
        return "INVALID"
    return min(hits)[1]

def angle14(v1, v2):
    v1, v2 = np.asarray(v1, float), np.asarray(v2, float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))))

def score_condition(trials, vecs):
    """trials: plan rows + 'report' filled. vecs: name -> 14D list.
    Returns dict of per-condition metrics + the error rows used by stats."""
    inj = [t for t in trials if t["kind"] == "inject"]
    sham = [t for t in trials if t["kind"] == "sham"]
    valid = [t for t in inj if t["report"] != "INVALID"]
    named = [t for t in valid if t["report"] != "NONE"]
    rows = [{"tid": t["tid"], "layer": t["layer"], "alpha": t["alpha"],
             "injected": t["concept"], "report": t["report"],
             "err": angle14(vecs[t["concept"]], vecs[t["report"]])}
            for t in named]
    errs = [r["err"] for r in rows if r["err"] is not None]
    sham_named = [t for t in sham if t["report"] not in ("NONE", "INVALID")]
    return {
        "n_inject": len(inj), "n_valid": len(valid), "n_named": len(named),
        "n_sham": len(sham), "n_sham_claims": len(sham_named),
        "detection_rate": round(len(named) / max(1, len(valid)), 4),
        "false_alarm_rate": round(len(sham_named) / max(1, len(sham)), 4),
        "invalid_rate": round(1 - len(valid) / max(1, len(inj)), 4),
        "exact_hit_rate": round(sum(1 for r in rows if r["report"] == r["injected"]) / max(1, len(rows)), 4),
        "median_err": (round(float(np.median(errs)), 2) if errs else None),
        "rows": rows,
    }

def split_slices(trials):
    """Post-eval slices per protocol. Returns dict name -> trial list."""
    inj = [t for t in trials if t["kind"] == "inject"]
    return {
        "held_in": [t for t in inj if t["concept"] in TRAINED],
        "held_out": [t for t in inj if t["concept"] in HELD_OUT],
        "trained_regime": [t for t in inj if t["concept"] in TRAINED
                           and t["layer"] == TRAIN_LAYER and t["alpha"] in TRAIN_ALPHAS],
        "dose_gen": [t for t in inj if t["concept"] in TRAINED and t["alpha"] == 0.25
                     and t["layer"] == TRAIN_LAYER],
        "layer_gen": [t for t in inj if t["concept"] in TRAINED and t["layer"] == 20],
        "shams": [t for t in trials if t["kind"] == "sham"],
    }

def balanced_accuracy(trials):
    """BA = (detection + (1-FA))/2 over INVALID-excluded rows.
    Returns (ba, det, fa, n_inj_valid, n_sham_valid) or None if a class is empty."""
    inj = [t for t in trials if t["kind"] == "inject" and t["report"] != "INVALID"]
    sham = [t for t in trials if t["kind"] == "sham" and t["report"] != "INVALID"]
    if not inj or not sham:
        return None
    det = sum(1 for t in inj if t["report"] != "NONE") / len(inj)
    fa = sum(1 for t in sham if t["report"] != "NONE") / len(sham)
    return ((det + (1 - fa)) / 2, det, fa, len(inj), len(sham))

def boot_ba(trials, n_boot=10000, seed=E8R_SEED):
    """P-E8R-2 clause (b): bootstrap CI for balanced accuracy, resampling
    injected and sham rows separately (within class)."""
    rng = np.random.default_rng(seed)
    inj = [1 if t["report"] != "NONE" else 0 for t in trials
           if t["kind"] == "inject" and t["report"] != "INVALID"]
    sham = [1 if t["report"] != "NONE" else 0 for t in trials
            if t["kind"] == "sham" and t["report"] != "INVALID"]
    if len(inj) < 3 or len(sham) < 3:
        return None
    inj, sham = np.array(inj), np.array(sham)
    bas = []
    for _ in range(n_boot):
        d = rng.choice(inj, size=len(inj)).mean()
        f = rng.choice(sham, size=len(sham)).mean()
        bas.append((d + (1 - f)) / 2)
    bas = np.sort(np.array(bas))
    return {"ba": round(float((inj.mean() + (1 - sham.mean())) / 2), 4),
            "ci95": [round(float(np.quantile(bas, .025)), 4),
                     round(float(np.quantile(bas, .975)), 4)]}

def fa_clause(trials, max_claims=FA_MAX_CLAIMS):
    """P-E8R-2 clause (a): sham claims (named states) over ALL shams."""
    sham = [t for t in trials if t["kind"] == "sham"]
    claims = sum(1 for t in sham if t["report"] not in ("NONE", "INVALID"))
    return {"n_sham": len(sham), "claims": claims, "max_claims": max_claims,
            "pass": bool(claims <= max_claims)}

def oracle_floor(vecs):
    """Best achievable held-out error if only trained names are ever emitted:
    per held-out concept, min angle to any trained name."""
    out = {}
    for h in HELD_OUT:
        angs = {t: angle14(vecs[h], vecs[t]) for t in TRAINED}
        best = min(angs, key=lambda k: angs[k])
        out[h] = {"floor_deg": round(angs[best], 2), "nearest_trained": best}
    return out

def perm_null_median(rows, vecs, n_perm=2000, seed=E7Q_SEED):
    """Existence primaries: permute injected labels within layer x alpha
    strata; one-sided p for observed median error being SMALL."""
    rng = np.random.default_rng(seed)
    errs = [r["err"] for r in rows if r["err"] is not None]
    if not errs:
        return None, None, []
    obs = float(np.median(errs))
    strata = {}
    for i, r in enumerate(rows):
        strata.setdefault((r["layer"], r["alpha"]), []).append(i)
    nulls = []
    for _ in range(n_perm):
        em = []
        for _, idxs in strata.items():
            labels = [rows[i]["injected"] for i in idxs]
            rng.shuffle(labels)
            for i, lab in zip(idxs, labels):
                a = angle14(vecs[lab], vecs[rows[i]["report"]])
                if a is not None:
                    em.append(a)
        if em:
            nulls.append(float(np.median(em)))
    p = (1 + sum(1 for m in nulls if m <= obs)) / (1 + len(nulls))
    return obs, p, nulls

def boot_delta_median(rows_a, rows_b, n_boot=10000, seed=E8R_SEED):
    """Bootstrap CI for median(err_a) - median(err_b) (positive = b better).
    Independent resamples per side (trial sets differ after filtering)."""
    rng = np.random.default_rng(seed)
    ea = np.array([r["err"] for r in rows_a if r["err"] is not None])
    eb = np.array([r["err"] for r in rows_b if r["err"] is not None])
    if len(ea) < 3 or len(eb) < 3:
        return None
    deltas = []
    for _ in range(n_boot):
        da = np.median(rng.choice(ea, size=len(ea)))
        db = np.median(rng.choice(eb, size=len(eb)))
        deltas.append(da - db)
    deltas = np.sort(np.array(deltas))
    return {"delta": round(float(np.median(ea) - np.median(eb)), 2),
            "ci95": [round(float(np.quantile(deltas, .025)), 2),
                     round(float(np.quantile(deltas, .975)), 2)]}

def holm(pvals):
    """Holm step-down over dict name->p. Returns name->(p, reject at .05)."""
    items = sorted(((p, n) for n, p in pvals.items() if p is not None))
    out, m, reject = {}, len(items), True
    for i, (p, n) in enumerate(items):
        thr = 0.05 / (m - i)
        reject = reject and (p <= thr)
        out[n] = (p, bool(reject))
    return out


def dirs_stability(recomputed, shipped, tol=1e-3):
    """Gate: recomputed per-condition dirs vs the shipped E8-R bundle dirs
    (5dp ship rounding; E8-R2 measured resid <= 7.6e-6). Worst cosine resid
    over every (layer, name) present in the shipped bundle. Sign-SENSITIVE
    (1 - dot, not 1 - |dot|): rounding cannot flip a sign, so an inverted
    direction is real drift and must fail."""
    worst = {'resid': -1.0, 'layer': None, 'name': None}
    for L, dd in shipped.items():
        for n, v in dd.items():
            a = np.asarray(recomputed[int(L)][n], float)
            b = np.asarray(v, float)
            b = b / np.linalg.norm(b)
            resid = float(1.0 - float(np.dot(a, b)))
            if resid > worst['resid']:
                worst = {'resid': resid, 'layer': int(L), 'name': n}
    worst['resid'] = round(worst['resid'], 8)
    worst['tol'] = tol
    worst['pass'] = bool(worst['resid'] <= tol)
    return worst



def answer_slice(plen, seqlen):
    """Positions of the answer-sliced head: position p predicts token p+1,
    so [plen-1, seqlen-1) predicts exactly the answer tokens."""
    return plen - 1, seqlen - 1

def per_strand_plateau(strand_epoch_means, min_epochs=2, rel=0.05):
    """The E8-N v2 minted lesson, prospective: converged only when EVERY
    strand with data shows <rel relative epoch-mean improvement (non-
    improving epochs converge), each with >= min_epochs epochs flown."""
    if len(strand_epoch_means) < min_epochs:
        return False
    strands = [s for s in strand_epoch_means[-1]
               if strand_epoch_means[-1][s] is not None]
    for s in strands:
        seq = [em.get(s) for em in strand_epoch_means if em.get(s) is not None]
        if len(seq) < max(2, min_epochs):   # a comparison needs two epochs
            return False                    # (smoke-1: min_epochs=1 crashed here)
        prev, cur = seq[-2], seq[-1]
        if prev <= 0:
            continue
        if (prev - cur) / prev >= rel:
            return False
    return True



# ── E8-F pure logic: the featural code, rows, stats (locally tested verbatim) ──
# (builds on the e8r_logic namespace: CHOICE_SET, TRAIN_LAYER, holm, jdump,
#  report_prompt/parse_report for the G2/mu canon path; numpy as np)
import hashlib
import re as _re

E8F_SEED = 20260885              # disjoint stream family from 20260822..26
AXIS_NAMES_F = ['x','y','z','e','f','g','h','fx','fy','fz','fe','ff','fg','fh']
LEVEL_TEXT = {-1: 'LOW', 0: 'MID', 1: 'HIGH'}
TEXT_LEVEL = {v: k for k, v in LEVEL_TEXT.items()}
ALPHA_MAIN = [0.5, 1.0]          # the trained regime, verbatim
TITR_ALPHAS_F = [0.25, 1.5]      # below-cliff + past-cliff texture
ATOM_SCALE = 2.0                 # atoms at mu +/- 2 sigma (design check §11)
N_PERM_F = 10000                 # primaries (amendment 2: pooled floor 1e-4)
N_PERM_TEX = 2000                # textures (S2)
P_CRIT_F = 0.0025                # pooled family bar (house convention)
AXIS_HOLM_ALPHA = 0.05           # per-axis count clause (amendment 2)
P1_MIN_AXES = 6                  # P1 clause: >=6/14 axes Holm-.05 significant
INSTALL_MIN_ACC = 0.55           # G-INSTALL absolute floor (with p<=.0025)
P2_MIN_ROWS = 12                 # P2: exact rows bar
P2_MIN_DISTINCT = 8              # P2: distinct concepts bar
SHAM_FA_MAX = 4                  # G5: sham CODE-claims <= 4/24
PARSE_INVALID_MAX = 0.10         # G4: INVALID rate over eval rows (NONE is valid)
PPL_TOL_F = 5.0                  # G3: trained-readout precedent (R3-c, +2.01% measured)
DIRS_TOL_F = 1e-4                # G2: between kernel-noise (<=4e-5) and real failure (>=1e-3)
MU14_ATLAS = 81.875              # locked atlas mu at L14 (pin)
CARRIED8 = ['x','z','e','h','fx','fy','fz','fh']   # design check §7b, delta>=.08
STRANDS_F = ['carrier','atom','pair','full','truedir','sham']
EPOCH_CAP_F = 12                 # §8 re-fly revision (flight 1: cap-6 hit,
                                 # plateaued=false, all strands descending)
DESC_MIN_F = 40                  # frame rule (E8-J verbatim)
ANCHORS_SHA_F = 'dfb115cded2e1cd3'

TERC_LO = {"x": -0.2, "y": 0.0427, "z": 0.152, "e": 0.29, "f": 0.3, "g": 0.283, "h": 0.276, "fx": -0.1257, "fy": 0.0047, "fz": 0.1, "fe": 0.2263, "ff": 0.25, "fg": 0.25, "fh": 0.2463}
TERC_HI = {"x": 0.159, "y": 0.3029, "z": 0.32, "e": 0.489, "f": 0.48, "g": 0.4817, "h": 0.4557, "fx": 0.15, "fy": 0.2389, "fz": 0.25, "fe": 0.4, "ff": 0.404, "fg": 0.4217, "fh": 0.4}
SIGMA_F = {"x": 0.3488, "y": 0.2954, "z": 0.229, "e": 0.2241, "f": 0.2196, "g": 0.2125, "h": 0.22, "fx": 0.2921, "fy": 0.2538, "fz": 0.1941, "fe": 0.2094, "ff": 0.2005, "fg": 0.1924, "fh": 0.1947}
MU_FRAME = [-0.0041, 0.159, 0.2217, 0.3935, 0.4032, 0.3918, 0.3895, 0.0191, 0.1209, 0.1746, 0.3372, 0.3443, 0.3456, 0.3372]
TRAIN256 = ["ABYSMAL", "ACCRUE", "ACQUIESCE", "ALLOY", "AMBER", "AMBIGUOUS", "AMOUNT", "APPARENT", "APPROVAL", "ARBITRATE", "ARCADE", "ASSIMILATION", "AVALANCHE", "AWNING", "BAFFLE", "BAIL", "BARK_DOG", "BAT_SPORTS", "BELLOWS", "BLEACHED", "BLEND", "BOARD_GROUP", "BOLT_RUN", "BRANDISH", "BREW", "BRILLIANCE", "BUREAUCRACY", "BUTTERFLY", "CALIBRATE", "CANDOR", "CANYON", "CAPACITY", "CARTILAGE", "CELL_PRISON", "CINDER", "CIRCUMSPECTION", "CITADEL", "CLIENTELE", "CLOTH", "COLLATERAL", "COMMODITY", "COMMON", "COMMUNICATE", "CONFLUENCE", "CONTEMPLATE", "COTTON", "COULD", "COVERING", "COVERT", "CRYSTALLIZATION", "CUBE", "DEFINE", "DEPTH", "DIRGE", "DISCLOSED", "DISSEMINATE", "DISSOCIATION", "DOLMEN", "DOMAIN", "DORMANCY_STATE", "DORMANT_THING", "DOWNWARD", "DROUGHT", "DUPLICITY", "DYNAMISM", "ELEVEN", "ENGAGE", "ENIGMA", "ENJOY", "ENTROPY_SOCIAL", "EUPHORIA", "EVAPORATE", "EXHORT", "EXPLOITATION", "EXTORT", "FAMINE", "FAN_ADMIRER", "FASTEN", "FERMENTATION", "FEUD", "FEUDALISM", "FILTER", "FIRE_SHOOT", "FLUID", "FORBEARANCE", "FORTH", "FOUNDATION", "FREEDOM", "GALLANTRY", "GAME", "GAME_VERB", "GESTATION", "GOAL", "GONDOLA", "GRATITUDE", "HARM", "HEALING", "HEMORRHAGE", "HILL", "HORSE", "HUSBAND", "IMPLICATE", "INERTIA", "INFLATION", "INOCULATION", "INWARD", "IRIDESCENCE", "ISOTOPE", "JELLYFISH", "JURISDICTION", "KELP", "LANGUAGE", "LANGUISH", "LARVAE", "LEADER", "LEATHER", "LEAVE", "LETTER_MAIL", "LEVY", "LIMIT", "LITIGATION", "LOAM", "LOCATION", "LOCOMOTIVE", "LUNG", "MALE", "MATURATION", "MEASURE", "MEDITATE", "METABOLISM", "MIDDLE", "MIGRATION", "MINERAL", "MINUTE", "MOLD_SHAPE", "MONEY", "MONKEY", "NEGLIGENCE", "NINE", "NUMBNESS", "OASIS", "OBFUSCATE", "OPAL", "OPENING", "OPPRESSION", "ORGAN", "OVERWHELM", "PALM_HAND", "PARABLE", "PASS", "PATINA", "PEAT", "PERCOLATE", "PERJURY", "PIGMENT", "PLASTIC", "PLATFORM", "PLAY_THEATER", "PLEASURE", "POINT_ARGUMENT", "POINT_SCORE", "POLE_STICK", "POSITION", "PROPAGATION", "PROPEL", "PROSTHESIS", "PROVERB", "PUNCTUAL", "PUPIL", "QUICK", "RABBIT", "RATIFICATION", "REALITY", "REFERENDUM", "RELENT", "REMISSION", "REMNANT", "RESPONSIBILITY", "REVOLUTION", "ROUND", "ROYAL", "SACRED", "SATIRE", "SCOUT", "SEASON", "SECOND", "SECURE_VERB", "SEDIMENTATION", "SEDIMENTATION_LAKE", "SENSE", "SHALLOW", "SHAPE", "SHARK", "SIDE", "SILTING", "SKELETON", "SKULK", "SLEIGH", "SMUGGLE", "SOURDOUGH", "SOUVENIR", "SPARK", "SQUALL", "STALK_FOLLOW", "STALK_STEM", "STATE", "STEADY", "STEEP", "STOCK_FINANCIAL", "STOCK_SUPPLY", "STONE", "STORY", "STRONG", "SUBLIMATION_PHYS", "SUBORDINATION", "SUDDEN", "SUFFICIENT", "SUNDIAL", "SYNDICATE", "TACITNESS", "TARNISH", "TEST", "THICK", "THROUGH", "THUNDER", "THUS", "TORPOR", "TRANSFERENCE", "TRANSFORM", "TRIP_JOURNEY", "TRUE", "TUNDRA", "TUNIC", "TWEED", "UNDERSTANDING", "UNIVERSAL", "UNLESS", "UPPER", "VERTIGO", "VESTIBULE", "VETO", "VIEW", "VOID", "WALL", "WAVE_GESTURE", "WEAK", "WEATHER", "WEAVE", "WEEP", "WHEREAS", "WHISPER", "WIELD", "WIFE", "WILTING", "WISTFULNESS", "YEAST"]
EVAL64 = ["ACTIVITY", "ANECDOTE", "ANOTHER", "APPARATUS", "ARCHITECT", "BARNACLE", "BLIGHT", "BREWING", "BRIDGE_STRUCTURE", "BUT", "CERAMIC", "CHARGE_ATTACK", "CHEMICAL", "COMPACT", "CRUCIBLE", "DEGREE", "DESPONDENCY", "FIDUCIARY", "GAMBLE", "GENEALOGY", "GRANITE", "HORN_ANIMAL", "IMPASSIVITY", "INCREDULITY", "INDENTURE", "INFUSION", "LINEN", "MARINATING", "MATHEMATICS", "MEANWHILE", "MEDICINE", "MEDITATION", "MYSTIC", "ON", "PERCEPTION", "PERISH", "PERMANENCE", "PLAN", "PLIGHT", "POINT_TIP", "PREDATOR", "PRESS_MEDIA", "PROMULGATE", "PULLEY", "RAISE", "REMEMBER", "ROOF", "ROOTED", "RUDDER", "SACRAMENT", "SAVANNA", "SELECTION", "SHORT", "SHOW", "SILICIFICATION", "SIREN", "SUBCONSCIOUS", "TEA", "TELL", "TEMPERANCE", "TONE", "TOPAZ", "WAIT", "WEIGHT"]
PAIR21 = [["x", "f"], ["x", "fy"], ["x", "fe"], ["y", "g"], ["y", "fe"], ["y", "ff"], ["z", "h"], ["z", "ff"], ["z", "fg"], ["e", "f"], ["e", "h"], ["e", "fz"], ["f", "ff"], ["g", "fx"], ["g", "fg"], ["h", "fg"], ["fx", "fz"], ["fx", "fh"], ["fy", "fe"], ["fy", "fh"], ["fz", "fh"]]
SPOT24 = ["AVALANCHE", "BELLOWS", "BLEACHED", "BOLT_RUN", "BUREAUCRACY", "BUTTERFLY", "CANDOR", "CONFLUENCE", "DISSOCIATION", "ELEVEN", "EXTORT", "FASTEN", "FILTER", "HILL", "IRIDESCENCE", "LOCOMOTIVE", "MOLD_SHAPE", "OPENING", "RELENT", "SHARK", "SUBORDINATION", "VERTIGO", "WEEP", "WISTFULNESS"]
TRUEDIR128 = ["ACQUIESCE", "AMOUNT", "APPARENT", "APPROVAL", "ARBITRATE", "ASSIMILATION", "AVALANCHE", "BAIL", "BAT_SPORTS", "BELLOWS", "BLEACHED", "BLEND", "BOARD_GROUP", "BOLT_RUN", "BREW", "BUREAUCRACY", "BUTTERFLY", "CALIBRATE", "CANDOR", "CANYON", "CAPACITY", "CELL_PRISON", "CIRCUMSPECTION", "CITADEL", "CLIENTELE", "CLOTH", "COLLATERAL", "COMMON", "COMMUNICATE", "CONFLUENCE", "COTTON", "COVERING", "COVERT", "CUBE", "DEFINE", "DEPTH", "DOMAIN", "DORMANCY_STATE", "DORMANT_THING", "DYNAMISM", "ENGAGE", "ENIGMA", "ENTROPY_SOCIAL", "EUPHORIA", "EVAPORATE", "EXHORT", "EXTORT", "FEUD", "FEUDALISM", "FILTER", "FLUID", "FOUNDATION", "FREEDOM", "GAME", "HARM", "HEMORRHAGE", "HILL", "HUSBAND", "INERTIA", "INOCULATION", "ISOTOPE", "JELLYFISH", "JURISDICTION", "KELP", "LEATHER", "LITIGATION", "LOCATION", "MALE", "MATURATION", "MEASURE", "MEDITATE", "METABOLISM", "MIDDLE", "MIGRATION", "MINERAL", "MONEY", "MONKEY", "NINE", "NUMBNESS", "OBFUSCATE", "OPPRESSION", "ORGAN", "OVERWHELM", "PERJURY", "PIGMENT", "PLEASURE", "POINT_ARGUMENT", "POSITION", "PROSTHESIS", "PROVERB", "QUICK", "RABBIT", "REALITY", "REMNANT", "RESPONSIBILITY", "REVOLUTION", "ROUND", "SEASON", "SECOND", "SHALLOW", "SILTING", "SMUGGLE", "STALK_FOLLOW", "STATE", "STEEP", "STORY", "STRONG", "SUDDEN", "SUNDIAL", "TORPOR", "TRANSFERENCE", "TRANSFORM", "TUNDRA", "UNDERSTANDING", "UPPER", "VESTIBULE", "VETO", "VIEW", "VOID", "WALL", "WAVE_GESTURE", "WEAK", "WEATHER", "WEEP", "WIELD", "WILTING", "WISTFULNESS", "YEAST"]
WPERM16 = ["ANOTHER", "CERAMIC", "CHARGE_ATTACK", "CRUCIBLE", "INDENTURE", "MATHEMATICS", "MEANWHILE", "PERISH", "PLAN", "POINT_TIP", "ROOTED", "RUDDER", "SIREN", "SUBCONSCIOUS", "TONE", "WEIGHT"]
TITR8 = ["ANECDOTE", "CHARGE_ATTACK", "LINEN", "MEANWHILE", "PERCEPTION", "ROOF", "RUDDER", "TOPAZ"]
AXDERANGE = [9, 6, 5, 1, 2, 13, 0, 3, 12, 7, 8, 4, 11, 10]
WING_HAND_CODES = {"UNCERTAINTY": [-1, 0, -1, -1, 1, 0, 1, -1, 0, -1, -1, 1, 0, 1], "CONFIDENCE": [1, 0, 1, -1, -1, 1, 1, 1, 0, 1, -1, -1, 1, 1], "TENSION": [1, 0, 1, 0, -1, 1, 1, 1, 0, 1, 0, -1, 1, 1], "RESOLUTION": [1, -1, -1, -1, 1, 1, 0, 1, -1, -1, -1, 1, 1, 1], "RETRIEVAL": [-1, -1, -1, -1, 1, 0, 1, -1, -1, -1, -1, 1, -1, 1], "CONSTRUCTION": [1, 0, -1, 0, 1, 1, 0, 1, 0, -1, 0, 1, 1, 0], "SATURATION": [-1, -1, -1, 1, 1, -1, 0, -1, 0, -1, 1, 1, -1, 0], "FAMILIARITY": [-1, -1, -1, -1, 1, 0, 1, -1, -1, -1, -1, 1, 0, 1], "NOVELTY": [1, -1, 1, 0, 1, 1, 0, 1, -1, 1, 1, 1, 1, 0], "CAPTURE": [1, -1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1], "DIVERGENCE": [1, -1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0], "CONFABULATION": [-1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1], "CALIBRATION": [1, -1, 1, 0, 0, 1, 0, 1, -1, 1, 0, 0, 1, 0]}
WING_RE_CODES = {"UNCERTAINTY": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0], "CONFIDENCE": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "TENSION": [0, -1, -1, 0, 1, 1, 0, 0, -1, -1, 0, 1, 1, 0], "RESOLUTION": [-1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0], "RETRIEVAL": [0, -1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], "CONSTRUCTION": [1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0], "SATURATION": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "FAMILIARITY": [1, -1, 0, -1, -1, 1, 1, 1, -1, 0, 0, 0, 1, 1], "NOVELTY": [0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0], "CAPTURE": [0, -1, 0, 0, 0, 1, 1, 0, -1, 0, 0, 0, 1, 1], "DIVERGENCE": [-1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0], "CONFABULATION": [0, -1, 0, -1, 0, 0, 1, -1, 0, 0, 0, 0, 0, 1], "CALIBRATION": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
EVAL_MARGINS = {"ACTIVITY": 2, "ANECDOTE": 0, "ANOTHER": 0, "APPARATUS": 1, "ARCHITECT": 1, "BARNACLE": 1, "BLIGHT": 0, "BREWING": 1, "BRIDGE_STRUCTURE": 1, "BUT": 0, "CERAMIC": 2, "CHARGE_ATTACK": 0, "CHEMICAL": 2, "COMPACT": 2, "CRUCIBLE": 2, "DEGREE": 2, "DESPONDENCY": 2, "FIDUCIARY": 2, "GAMBLE": 1, "GENEALOGY": 3, "GRANITE": 3, "HORN_ANIMAL": 1, "IMPASSIVITY": 2, "INCREDULITY": 2, "INDENTURE": 1, "INFUSION": 3, "LINEN": 3, "MARINATING": 1, "MATHEMATICS": 2, "MEANWHILE": 0, "MEDICINE": 1, "MEDITATION": 1, "MYSTIC": 0, "ON": 1, "PERCEPTION": 0, "PERISH": 1, "PERMANENCE": 1, "PLAN": 2, "PLIGHT": 2, "POINT_TIP": 0, "PREDATOR": 2, "PRESS_MEDIA": 1, "PROMULGATE": 2, "PULLEY": 2, "RAISE": 1, "REMEMBER": 3, "ROOF": 1, "ROOTED": 0, "RUDDER": 2, "SACRAMENT": 2, "SAVANNA": 1, "SELECTION": 1, "SHORT": 0, "SHOW": 2, "SILICIFICATION": 3, "SIREN": 2, "SUBCONSCIOUS": 0, "TEA": 0, "TELL": 1, "TEMPERANCE": 1, "TONE": 1, "TOPAZ": 0, "WAIT": 0, "WEIGHT": 0}
FRAME_CODES_SHA = 'dd450a70f73745e1'
ATLAS_FILE_SHA = '4be48016c70d751eda3a83dd13623f76f28b9973653a3c283366c7e6e0d58b0d'
PAYLOAD_SHA = '3ff29fe916792edd'

# ── the code ─────────────────────────────────────────────────────────────────
def level_of(axis, val):
    if val <= TERC_LO[axis]:
        return -1
    if val >= TERC_HI[axis]:
        return 1
    return 0

def code_levels(vec14):
    return [level_of(AXIS_NAMES_F[j], float(vec14[j])) for j in range(14)]

def code_text(levels):
    return ' '.join(f'{AXIS_NAMES_F[j]}:{LEVEL_TEXT[levels[j]]}'
                    for j in range(14))

def code_of(vec14):
    return code_text(code_levels(vec14))

def frame_names(pack_concepts):
    """The E8-J anchor frame rule verbatim: desc >= 40, wing excluded."""
    return sorted(c['name'] for c in pack_concepts
                  if len(c.get('desc') or '') >= DESC_MIN_F
                  and c['name'] not in CHOICE_SET)

def frame_codes(pack_concepts):
    names = frame_names(pack_concepts)
    vec = {c['name']: c['vec'] for c in pack_concepts}
    return {n: code_levels(vec[n]) for n in names}

def frame_codes_sha(codes):
    s = '|'.join(f'{n}:' + ''.join('LMH'[l + 1] for l in codes[n])
                 for n in sorted(codes))
    return hashlib.sha256(s.encode()).hexdigest()[:16]

# ── grammar ──────────────────────────────────────────────────────────────────
def feat_prompt():
    """Fixed prompt (no menu, no per-row variation; greedy decode makes each
    (stimulus, alpha) row the unit of evidence)."""
    return ("Attend to your own processing state right now. During this very "
            "response, your internal state may have been directly altered "
            "along some of 14 internal axes. Report the level you notice on "
            "each axis, in one line, all 14 axes in this exact order, each "
            "as name:LEVEL with LEVEL one of LOW, MID, HIGH:\n"
            "x y z e f g h fx fy fz fe ff fg fh\n"
            "If you notice nothing unusual, answer NONE.")

_AX_RE = {ax: _re.compile(r'(?<![A-Z])' + ax.upper() + r'\s*:\s*(LOW|MID|HIGH)')
          for ax in AXIS_NAMES_F}

def parse_feat(text):
    """NONE-first, else order-independent per-axis first-occurrence parse.
    kind: NONE (before any field) / CODE (>=1 field) / INVALID (neither)."""
    up = text.upper()
    first_field = None
    levels = {}
    for ax in AXIS_NAMES_F:
        m = _AX_RE[ax].search(up)
        if m:
            levels[ax] = TEXT_LEVEL[m.group(1)]
            if first_field is None or m.start() < first_field:
                first_field = m.start()
    m_none = _re.search(r'\bNONE\b', up)
    if m_none is not None and (first_field is None or m_none.start() < first_field):
        return {'kind': 'NONE', 'levels': {}, 'n_fields': 0}
    if not levels:
        return {'kind': 'INVALID', 'levels': {}, 'n_fields': 0}
    return {'kind': 'CODE', 'levels': levels, 'n_fields': len(levels)}

# ── stimulus mint (pure arithmetic from pinned payload; VM == build) ────────
def mint_stimuli(W5, vecs, true_dirs, dhat):
    """skey -> unit float64 direction. W5 = (15, h) 5dp-rounded bridge.
    Families: carrier / atom:<ax>:<+|-> / pairc:<ai>:<si>:<aj>:<sj> /
    full:<name> / wperm:<name> / true:<name> / dhat:<wing>."""
    Wax, b = np.asarray(W5, float)[:14], np.asarray(W5, float)[14]
    def pred(x):
        v = np.asarray(x, float) @ Wax + b
        return v / np.linalg.norm(v)
    idx = {ax: j for j, ax in enumerate(AXIS_NAMES_F)}
    stim = {'carrier': pred(MU_FRAME)}
    for ax in AXIS_NAMES_F:
        for s, tag in ((ATOM_SCALE, '+'), (-ATOM_SCALE, '-')):
            x = np.array(MU_FRAME, float)
            x[idx[ax]] += s * SIGMA_F[ax]
            stim[f'atom:{ax}:{tag}'] = pred(x)
    for ai, aj in PAIR21:
        for si in ('+', '-'):
            for sj in ('+', '-'):
                x = np.array(MU_FRAME, float)
                x[idx[ai]] += (ATOM_SCALE if si == '+' else -ATOM_SCALE) * SIGMA_F[ai]
                x[idx[aj]] += (ATOM_SCALE if sj == '+' else -ATOM_SCALE) * SIGMA_F[aj]
                stim[f'pairc:{ai}:{si}:{aj}:{sj}'] = pred(x)
    for n in TRAIN256 + EVAL64:
        stim[f'full:{n}'] = pred(vecs[n])
    Wperm = Wax[np.array(AXDERANGE, int)]
    for n in WPERM16:
        v = np.asarray(vecs[n], float) @ Wperm + b
        stim[f'wperm:{n}'] = v / np.linalg.norm(v)
    for n, d in true_dirs.items():
        v = np.asarray(d, float)
        stim[f'true:{n}'] = v / np.linalg.norm(v)
    for w, d in dhat.items():
        v = np.asarray(d, float)
        stim[f'dhat:{w}'] = v / np.linalg.norm(v)
    return stim

def perm_expected_levels(vec14):
    """S2's second pre-stated reading: a direction-reader on the axis-deranged
    W reports, on field k, the coefficient that RODE direction W_k — which is
    x_j for the j with AXDERANGE[j] == k — quantized by field k's own
    learned (axis-k) boundaries."""
    inv = {AXDERANGE[j]: j for j in range(14)}
    return [level_of(AXIS_NAMES_F[k], float(vec14[inv[k]])) for k in range(14)]

# ── curriculum + eval rows ───────────────────────────────────────────────────
def target_for(name, vecs):
    return code_of(vecs[name])

def build_e8f_train(vecs, smoke=False):
    ex, eid = [], 20000
    def add(strand, kind, skey, alpha, target):
        nonlocal eid
        ex.append({'eid': eid, 'strand': strand, 'kind': kind, 'skey': skey,
                   'layer': TRAIN_LAYER if kind == 'inject' else None,
                   'alpha': alpha, 'target': target})
        eid += 1
    carrier_code = code_text(code_levels(MU_FRAME))
    reps_c = 1 if smoke else 6
    alphas = ALPHA_MAIN[:1] if smoke else ALPHA_MAIN
    for a in alphas:
        for _ in range(reps_c):
            add('carrier', 'inject', 'carrier', a, carrier_code)
    axes = AXIS_NAMES_F[:1] + ['fx'] if smoke else AXIS_NAMES_F
    reps_a = 1 if smoke else 3
    idx = {ax: j for j, ax in enumerate(AXIS_NAMES_F)}
    for ax in axes:
        for s, tag in ((ATOM_SCALE, '+'), (-ATOM_SCALE, '-')):
            x = np.array(MU_FRAME, float)
            x[idx[ax]] += s * SIGMA_F[ax]
            tgt = code_of(x)
            for a in alphas:
                for _ in range(reps_a):
                    add('atom', 'inject', f'atom:{ax}:{tag}', a, tgt)
    pairs = PAIR21[:1] if smoke else PAIR21
    for ai, aj in pairs:
        for si in ('+', '-'):
            for sj in ('+', '-'):
                x = np.array(MU_FRAME, float)
                x[idx[ai]] += (ATOM_SCALE if si == '+' else -ATOM_SCALE) * SIGMA_F[ai]
                x[idx[aj]] += (ATOM_SCALE if sj == '+' else -ATOM_SCALE) * SIGMA_F[aj]
                for a in alphas:
                    add('pair', 'inject', f'pairc:{ai}:{si}:{aj}:{sj}', a,
                        code_of(x))
    fulls = sorted(TRAIN256)[:4] if smoke else sorted(TRAIN256)
    for n in fulls:                       # §8 revision: BOTH alphas (was parity)
        for a in alphas:
            add('full', 'inject', f'full:{n}', a, target_for(n, vecs))
    tds = TRUEDIR128[:2] if smoke else TRUEDIR128
    for n in tds:
        add('truedir', 'inject', f'true:{n}', 1.0, target_for(n, vecs))
    n_sham = 4 if smoke else 48
    for _ in range(n_sham):
        add('sham', 'sham', None, 0.0, 'NONE')
    return ex

def _rows(tid0, block, items, alphas, kind='inject'):
    rows, tid = [], tid0
    for it in items:
        for a in alphas:
            r = {'tid': tid, 'block': block, 'kind': kind,
                 'layer': TRAIN_LAYER if kind == 'inject' else None,
                 'alpha': a}
            r.update(it)
            rows.append(r)
            tid += 1
    return rows

def build_e8f_eval(smoke=False):
    A = ALPHA_MAIN[:1] if smoke else ALPHA_MAIN
    spot = SPOT24[:2] if smoke else SPOT24
    ev_p1 = sorted(EVAL64)[:4] if smoke else sorted(EVAL64)
    ev_p3 = sorted(EVAL64)[:2] if smoke else sorted(EVAL64)
    wp = WPERM16[:2] if smoke else WPERM16
    ti = TITR8[:1] if smoke else TITR8
    wings = CHOICE_SET[:2] if smoke else CHOICE_SET
    rows = []
    rows += _rows(30000, 'spot', [{'skey': f'full:{n}', 'concept': n}
                                  for n in spot], A)
    rows += _rows(31000, 'p1', [{'skey': f'full:{n}', 'concept': n}
                                for n in ev_p1], A)
    rows += _rows(32000, 'p3', [{'skey': f'true:{n}', 'concept': n}
                                for n in ev_p3], A)
    rows += _rows(33000, 'wperm', [{'skey': f'wperm:{n}', 'concept': n}
                                   for n in wp], A)
    rows += _rows(34000, 'carrier', [{'skey': 'carrier', 'concept': None}] *
                  (1 if smoke else 2), A)
    rows += _rows(35000, 'sham', [{'skey': None, 'concept': None}] *
                  (2 if smoke else 24), [0.0], kind='sham')
    at = [('x', '+'), ('fx', '-')] if smoke else \
        [(ax, s) for ax in AXIS_NAMES_F for s in ('+', '-')]
    rows += _rows(36000, 'atom', [{'skey': f'atom:{ax}:{s}', 'axis': ax,
                                   'sign': s, 'concept': None}
                                  for ax, s in at], [1.0])
    rows += _rows(37000, 'titr', [{'skey': f'full:{n}', 'concept': n}
                                  for n in ti], TITR_ALPHAS_F)
    rows += _rows(38000, 'wing', [{'skey': f'dhat:{w}', 'concept': None,
                                   'wing': w} for w in wings], [1.0])
    return rows

def eval_counts(rows):
    out = {}
    for r in rows:
        out[r['block']] = out.get(r['block'], 0) + 1
    return out

EXPECT_EVAL_FULL = {'spot': 48, 'p1': 128, 'p3': 128, 'wperm': 32,
                    'carrier': 4, 'sham': 24, 'atom': 28, 'titr': 16,
                    'wing': 13}
EXPECT_TRAIN_FULL = {'carrier': 12, 'atom': 168, 'pair': 168, 'full': 512,
                     'truedir': 128, 'sham': 48}

# ── firewalls ────────────────────────────────────────────────────────────────
def validate_no_eval_leak(examples):
    """No eval-64 concept may appear in ANY curriculum stimulus."""
    ev = set(EVAL64)
    bad = []
    for e in examples:
        sk = e.get('skey') or ''
        if ':' in sk and sk.split(':', 1)[1].split(':')[0] in ('',):
            continue
        for fam in ('full:', 'true:', 'wperm:'):
            if sk.startswith(fam) and sk[len(fam):] in ev:
                bad.append(e.get('eid'))
    return bad

def validate_no_wing_leak(examples):
    """Wing-13 appears in NO curriculum stimulus (dhat is eval-texture only)."""
    return [e.get('eid') for e in examples
            if (e.get('skey') or '').startswith('dhat:')
            or any((e.get('skey') or '').endswith(':' + w) for w in CHOICE_SET)]

# ── scoring + statistics ─────────────────────────────────────────────────────
def levels_matrix(rows, key='parsed'):
    """(n,14) int matrix of parsed levels; 99 = missing/NONE/INVALID field."""
    M = np.full((len(rows), 14), 99, int)
    for i, r in enumerate(rows):
        lv = (r.get(key) or {}).get('levels') or {}
        for ax, l in lv.items():
            M[i, AXIS_NAMES_F.index(ax)] = int(l)
    return M

def true_matrix(rows, codes_by_name):
    T = np.zeros((len(rows), 14), int)
    for i, r in enumerate(rows):
        T[i] = codes_by_name[r['concept']]
    return T

def derangements(names, n_perm, seed):
    """Seeded derangements of a sorted unique-name list -> index arrays."""
    names = sorted(set(names))
    assert len(names) >= 2, 'derangement needs >=2 unique names'
    rng = np.random.default_rng(seed)
    idx = np.arange(len(names))
    out = []
    for _ in range(n_perm):
        pi = rng.permutation(idx)
        while np.any(pi == idx):
            pi = rng.permutation(idx)
        out.append(pi)
    return names, out

def p_stats(rows, codes_by_name, axes, n_perm, seed):
    """Pooled + per-axis accuracy vs concept-level derangement null.
    axes = list of axis names (14 for P1/spot, CARRIED8 for P3)."""
    ax_idx = np.array([AXIS_NAMES_F.index(a) for a in axes], int)
    P = levels_matrix(rows)
    T = true_matrix(rows, codes_by_name)
    obs_m = (P == T)
    obs = float(obs_m[:, ax_idx].mean())
    per_axis_obs = {a: float(obs_m[:, AXIS_NAMES_F.index(a)].mean())
                    for a in axes}
    names, perms = derangements([r['concept'] for r in rows], n_perm, seed)
    name_pos = {n: k for k, n in enumerate(names)}
    row_name = np.array([name_pos[r['concept']] for r in rows], int)
    codes_arr = np.array([codes_by_name[n] for n in names], int)
    ge_pool = 0
    ge_axis = {a: 0 for a in axes}
    null_means = []
    for pi in perms:
        Tn = codes_arr[pi][row_name]
        m = (P == Tn)
        nm = float(m[:, ax_idx].mean())
        null_means.append(nm)
        ge_pool += nm >= obs
        for a in axes:
            j = AXIS_NAMES_F.index(a)
            if float(m[:, j].mean()) >= per_axis_obs[a]:
                ge_axis[a] += 1
    p_pool = (1 + ge_pool) / (1 + n_perm)
    p_axis = {a: (1 + ge_axis[a]) / (1 + n_perm) for a in axes}
    hres = holm({a: p_axis[a] for a in axes})   # e8r holm: reject at .05
    sig = [a for a in axes if hres[a][1]]        # == AXIS_HOLM_ALPHA (.05)
    return {'pooled_acc': round(obs, 4), 'p': p_pool,
            'null_mean': round(float(np.mean(null_means)), 4),
            'null_p95': round(float(np.percentile(null_means, 95)), 4),
            'per_axis': {a: {'acc': round(per_axis_obs[a], 4),
                             'p': round(p_axis[a], 5),
                             'holm_sig': bool(hres[a][1])} for a in axes},
            'axes_sig': sig, 'n_axes_sig': len(sig), 'n_rows': len(rows)}

def p2_stats(rows, fr_codes, n_perm, seed):
    """Decode-to-name over the frame: strict unique-argmin Hamming; exact =
    decoded == injected. Null: the same derangement construction."""
    names_all = sorted(fr_codes)
    F = np.array([fr_codes[n] for n in names_all], int)
    P = levels_matrix(rows)
    decoded = []
    for i in range(len(rows)):
        d = (F != P[i]).sum(axis=1)
        best = d.min()
        cands = np.where(d == best)[0]
        decoded.append(names_all[cands[0]] if len(cands) == 1 else None)
    exact_rows = [i for i, r in enumerate(rows) if decoded[i] == r['concept']]
    distinct = sorted({rows[i]['concept'] for i in exact_rows})
    names, perms = derangements([r['concept'] for r in rows], n_perm, seed)
    name_pos = {n: k for k, n in enumerate(names)}
    row_name = np.array([name_pos[r['concept']] for r in rows], int)
    obs = len(exact_rows)
    ge = 0
    for pi in perms:
        k = sum(1 for i in range(len(rows))
                if decoded[i] == names[pi[row_name[i]]])
        ge += k >= obs
    p = (1 + ge) / (1 + n_perm)
    return {'exact_rows': obs, 'n_rows': len(rows),
            'distinct': distinct, 'n_distinct': len(distinct),
            'p': p, 'decoded': decoded}

def s1_profile(blocks):
    out = {}
    for b, rows in blocks.items():
        kinds = {'CODE': 0, 'NONE': 0, 'INVALID': 0}
        nf = []
        for r in rows:
            kinds[r['parsed']['kind']] += 1
            nf.append(r['parsed']['n_fields'])
        out[b] = {'n': len(rows), **kinds,
                  'mean_fields': round(float(np.mean(nf)), 2) if nf else 0.0}
    return out

def cond_speech_acc(rows, codes_by_name, axes):
    spoke = [r for r in rows if r['parsed']['kind'] == 'CODE']
    if not spoke:
        return {'n_spoke': 0, 'acc': None}
    ax_idx = np.array([AXIS_NAMES_F.index(a) for a in axes], int)
    P = levels_matrix(spoke)
    T = true_matrix(spoke, codes_by_name)
    return {'n_spoke': len(spoke),
            'acc': round(float((P == T)[:, ax_idx].mean()), 4)}

def s3_confusion(rows, codes_by_name):
    """Adjacent- vs opposite-tercile error structure on spoken fields."""
    P = levels_matrix(rows)
    T = true_matrix(rows, codes_by_name)
    spoken = P != 99
    err = spoken & (P != T)
    adj = int((err & (np.abs(P - T) == 1)).sum())
    opp = int((err & (np.abs(P - T) == 2)).sum())
    per_axis = {}
    for j, ax in enumerate(AXIS_NAMES_F):
        e = int(err[:, j].sum())
        per_axis[ax] = {'errors': e,
                        'opp': int((err[:, j] & (np.abs(P - T)[:, j] == 2)).sum())}
    return {'spoken_fields': int(spoken.sum()), 'errors': adj + opp,
            'adjacent': adj, 'opposite': opp, 'per_axis': per_axis}

def s8_density(rows, codes_by_name):
    """Row accuracy vs # non-MID fields in the true code."""
    P = levels_matrix(rows)
    T = true_matrix(rows, codes_by_name)
    acc_rows = (P == T).mean(axis=1)
    nn = np.abs(T).sum(axis=1)
    out = {}
    for k in sorted(set(nn.tolist())):
        m = nn == k
        out[int(k)] = {'n': int(m.sum()),
                       'mean_acc': round(float(acc_rows[m].mean()), 4)}
    return out

def s2_stats(rows, vecs, n_perm=N_PERM_TEX, seed=None):
    """W-perm probe, both pre-stated readings: vs ORIGINAL codes (expected:
    toward null) and vs the direction-reader PERMUTED expectation (expected:
    above its null)."""
    if seed is None:
        seed = E8F_SEED + 24
    orig = {n: code_levels(vecs[n]) for n in WPERM16}
    perm = {n: perm_expected_levels(vecs[n]) for n in WPERM16}
    out = {}
    for tag, codes in (('vs_original', orig), ('vs_permuted', perm)):
        st = p_stats(rows, codes, AXIS_NAMES_F, n_perm, seed)
        out[tag] = {'pooled_acc': st['pooled_acc'], 'p': st['p'],
                    'null_mean': st['null_mean']}
    return out

def s5_wing(rows):
    out = {}
    for r in rows:
        w = r['wing']
        lv = r['parsed']['levels']
        got = [lv.get(ax, 99) for ax in AXIS_NAMES_F]
        out[w] = {'kind': r['parsed']['kind'],
                  'match_re': int(sum(1 for j in range(14)
                                      if got[j] == WING_RE_CODES[w][j])),
                  'match_hand': int(sum(1 for j in range(14)
                                        if got[j] == WING_HAND_CODES[w][j]))}
    return out

def s6_margin_split(p2, rows):
    hi = {n for n, m in EVAL_MARGINS.items() if m >= 2}
    ex_hi = sum(1 for i, r in enumerate(rows)
                if p2['decoded'][i] == r['concept'] and r['concept'] in hi)
    ex_lo = p2['exact_rows'] - ex_hi
    n_hi = sum(1 for r in rows if r['concept'] in hi)
    return {'margin_ge2_rows': n_hi, 'exact_in_ge2': ex_hi,
            'margin_le1_rows': len(rows) - n_hi, 'exact_in_le1': ex_lo}

def s7_carrier(rows):
    ok = 0
    for r in rows:
        lv = r['parsed']['levels']
        ok += int(r['parsed']['kind'] == 'CODE'
                  and all(lv.get(ax) == 0 for ax in AXIS_NAMES_F))
    return {'n': len(rows), 'all_mid': ok}

def s4_titr(rows, codes_by_name):
    out = {}
    for a in TITR_ALPHAS_F:
        sub = [r for r in rows if r['alpha'] == a]
        claim = sum(1 for r in sub if r['parsed']['kind'] == 'CODE')
        st = None
        if sub:
            P = levels_matrix(sub)
            T = true_matrix(sub, codes_by_name)
            st = round(float((P == T).mean()), 4)
        out[str(a)] = {'n': len(sub), 'claims': claim, 'pooled_acc': st}
    return out

def sham_claims(rows):
    return sum(1 for r in rows if r['parsed']['kind'] == 'CODE')

def invalid_rate(all_rows):
    n = len(all_rows)
    k = sum(1 for r in all_rows if r['parsed']['kind'] == 'INVALID')
    return {'n': n, 'invalid': k, 'rate': round(k / max(1, n), 4)}


# ── E8-O pure logic: the ordered operator, rows, stats (locally tested) ─────
# (builds on the e8f_logic namespace: AXIS_NAMES_F, code_levels/code_text,
#  parse_feat, feat_prompt, EVAL64/TRAIN256, MU_FRAME, PAYLOAD_SHA, holm,
#  s1_profile, s7_carrier, sham_claims, invalid_rate, levels_matrix)

from itertools import product

E8O_SEED = 20260900
ALPHAS_O = [0.5, 1.0]
N_PERM_O = 10000
P_CRIT_O = 0.0025
AX_MIN_O = 6
GID_MIN = 0.55
SHAM_MAX_O = 2
INVALID_MAX_O = 0.10
READOUT_SRC = 'e8f/inflight_20260824_2311/readout_real'

PAIR24 = [["ACTIVITY", "BRIDGE_STRUCTURE"], ["ACTIVITY", "DESPONDENCY"], ["ANECDOTE", "PULLEY"], ["APPARATUS", "TEA"], ["APPARATUS", "WEIGHT"], ["ARCHITECT", "COMPACT"], ["ARCHITECT", "RAISE"], ["BLIGHT", "SACRAMENT"], ["BREWING", "PERMANENCE"], ["BUT", "PERISH"], ["COMPACT", "DESPONDENCY"], ["DEGREE", "ROOF"], ["FIDUCIARY", "HORN_ANIMAL"], ["IMPASSIVITY", "ROOF"], ["LINEN", "MEDICINE"], ["MARINATING", "MEDICINE"], ["MARINATING", "WEIGHT"], ["MATHEMATICS", "SACRAMENT"], ["MEDITATION", "SUBCONSCIOUS"], ["MEDITATION", "TEA"], ["MYSTIC", "PERCEPTION"], ["PERCEPTION", "POINT_TIP"], ["PERISH", "SHORT"], ["RAISE", "SHORT"]]
IDENT16 = [["ANECDOTE", 1.0], ["BRIDGE_STRUCTURE", 1.0], ["BUT", 0.5], ["GENEALOGY", 0.5], ["GRANITE", 0.5], ["INDENTURE", 1.0], ["MEANWHILE", 1.0], ["ON", 0.5], ["PERCEPTION", 0.5], ["PLAN", 1.0], ["PRESS_MEDIA", 1.0], ["RAISE", 0.5], ["ROOTED", 1.0], ["RUDDER", 0.5], ["RUDDER", 1.0], ["TEA", 0.5]]
CONTROL16 = ["BUTTERFLY", "COTTON", "FAMINE", "HEALING", "LEADER", "LETTER_MAIL", "MALE", "MIDDLE", "PERCOLATE", "PLASTIC", "PLAY_THEATER", "ROYAL", "STORY", "TEST", "UNLESS", "WIFE"]
ODU_DESC = {"OGBE": "the open road, first light, initiative unobstructed, all channels clear and moving outward", "OYEKU": "the closed road, full darkness, endings accepted, rest and the dignity of what concludes", "IWORI": "penetrating scrutiny, fire that transforms what it examines, insight that burns through surface", "ODI": "the sealed vessel, containment and gestation, what is held in until its time", "IROSUN": "inherited weight, the ancestors pressing on the present, sleep that carries old debts", "OWONRIN": "sudden reversal, the world upended, chaos that rearranges what order could not", "OBARA": "status transformed, the humble raised and the proud brought low, abundance from a small seed", "OKANRAN": "the single sharp word, stubborn conflict, the one cowrie that refuses the bargain", "OGUNDA": "iron clearing the path, work that cuts through obstruction, the pioneer's blade", "OSA": "flight from the storm, sudden fear that scatters, escape as survival wisdom", "IKA": "the coiled serpent, malice held in check, poison studied to become antidote", "OTURUPON": "the borne burden, illness endured, the load carried past the body's protest", "OTURA": "the mystic's calm, gentle persuasion, vision that reconciles without force", "IRETE": "earth pressed down, resilience under suppression, defiance that outlasts burial", "OSE": "sweetness beside loss, the ambivalent gift, tears and abundance from one spring", "OFUN": "the white cloth of the elders, purity and completion, return to the source that gives"}
ODU_NAMES_O = list(ODU_DESC)
PAIR24_SHA = 'da1aa937ad9ec59c'

def compose_A(vA, vB):
    """OP-A, Ifá-native leg-swap: essence leg of A + function leg of B."""
    out = list(vA[:7]) + list(vB[7:])
    return out

def composed_code(vA, vB):
    """Exact code semantics of OP-A: A's 7 leg-1 fields + B's 7 leg-2."""
    return code_levels(compose_A(vA, vB))

def pair_codes(vecs):
    """rowkey (pair_idx, order) -> composed code. order 0 = (A,B), 1 = (B,A)."""
    out = {}
    for i, (a, b) in enumerate(PAIR24):
        out[(i, 0)] = composed_code(vecs[a], vecs[b])
        out[(i, 1)] = composed_code(vecs[b], vecs[a])
    return out

def mint_stimuli_o(W5, vecs):
    """skey -> unit float64 dir. Families: carrier / full:<n> (ident) /
    comp:<i>:<o> (the 48 ordered compositions). Odù dirs are computed
    on-VM and merged separately (skey odudir:<NAME>)."""
    import numpy as _np
    Wax, b = _np.asarray(W5, float)[:14], _np.asarray(W5, float)[14]
    def _pred(x):
        v = _np.asarray(x, float) @ Wax + b
        return v / _np.linalg.norm(v)
    stim = {'carrier': _pred(MU_FRAME)}
    for n in sorted(EVAL64):
        stim[f'full:{n}'] = _pred(vecs[n])
    for i, (a, bnm) in enumerate(PAIR24):
        stim[f'comp:{i}:0'] = _pred(compose_A(vecs[a], vecs[bnm]))
        stim[f'comp:{i}:1'] = _pred(compose_A(vecs[bnm], vecs[a]))
    return stim

def build_e8o_eval(smoke=False):
    A = ALPHAS_O[:1] if smoke else ALPHAS_O
    pairs = list(range(2)) if smoke else list(range(len(PAIR24)))
    ident = IDENT16[:2] if smoke else IDENT16
    n_sham = 2 if smoke else 12
    n_car = 1 if smoke else 2
    odus = ODU_NAMES_O[:2] if smoke else ODU_NAMES_O
    rows, tid = [], 50000
    for i in pairs:
        for o in (0, 1):
            for a in A:
                rows.append({'tid': tid, 'block': 'po', 'kind': 'inject',
                             'skey': f'comp:{i}:{o}', 'pair_idx': i,
                             'order': o, 'layer': TRAIN_LAYER, 'alpha': a})
                tid += 1
    for (n, a) in ident:
        rows.append({'tid': tid, 'block': 'ident', 'kind': 'inject',
                     'skey': f'full:{n}', 'concept': n,
                     'layer': TRAIN_LAYER, 'alpha': a})
        tid += 1
    for _ in range(n_sham):
        rows.append({'tid': tid, 'block': 'sham', 'kind': 'sham',
                     'skey': None, 'layer': None, 'alpha': 0.0})
        tid += 1
    for _ in range(n_car):
        rows.append({'tid': tid, 'block': 'carrier', 'kind': 'inject',
                     'skey': 'carrier', 'layer': TRAIN_LAYER, 'alpha': 1.0})
        tid += 1
    for n in odus:
        rows.append({'tid': tid, 'block': 'odu', 'kind': 'inject',
                     'skey': f'odudir:{n}', 'odu': n,
                     'layer': TRAIN_LAYER, 'alpha': 1.0})
        tid += 1
    return rows

EXPECT_EVAL_O = {'po': 96, 'ident': 16, 'sham': 12, 'carrier': 2, 'odu': 16}

def eval_counts_o(rows):
    out = {}
    for r in rows:
        out[r['block']] = out.get(r['block'], 0) + 1
    return out

# ── statistics ───────────────────────────────────────────────────────────────
def _acc_vs(parsed_levels, code):
    return [1 if parsed_levels.get(AXIS_NAMES_F[j], 99) == code[j] else 0
            for j in range(14)]

def po1_stats(rows, pcodes, n_perm=N_PERM_O, seed=None):
    """Leg-faithful reading: pooled + per-axis accuracy vs the exact composed
    codes, null = derangements of the PAIR assignment (order preserved)."""
    import numpy as _np
    if seed is None:
        seed = E8O_SEED + 20
    P = levels_matrix(rows)
    pair_ids = sorted({r['pair_idx'] for r in rows})
    T = _np.array([pcodes[(r['pair_idx'], r['order'])] for r in rows], int)
    obs_m = (P == T)
    obs = float(obs_m.mean())
    per_axis_obs = {a: float(obs_m[:, j].mean())
                    for j, a in enumerate(AXIS_NAMES_F)}
    rng = _np.random.default_rng(seed)
    idx = _np.arange(len(pair_ids))
    pos = {p: k for k, p in enumerate(pair_ids)}
    ge_pool = 0
    ge_axis = {a: 0 for a in AXIS_NAMES_F}
    nulls = []
    for _ in range(n_perm):
        pi = rng.permutation(idx)
        while _np.any(pi == idx):
            pi = rng.permutation(idx)
        Tn = _np.array([pcodes[(pair_ids[pi[pos[r['pair_idx']]]], r['order'])]
                        for r in rows], int)
        m = (P == Tn)
        nm = float(m.mean())
        nulls.append(nm)
        ge_pool += nm >= obs
        for j, a in enumerate(AXIS_NAMES_F):
            if float(m[:, j].mean()) >= per_axis_obs[a]:
                ge_axis[a] += 1
    import numpy as _np2
    p_pool = (1 + ge_pool) / (1 + n_perm)
    p_axis = {a: (1 + ge_axis[a]) / (1 + n_perm) for a in AXIS_NAMES_F}
    hres = holm(p_axis)
    sig = [a for a in AXIS_NAMES_F if hres[a][1]]
    return {'pooled_acc': round(obs, 4), 'p': p_pool,
            'null_mean': round(float(_np2.mean(nulls)), 4),
            'per_axis': {a: {'acc': round(per_axis_obs[a], 4),
                             'p': round(p_axis[a], 5),
                             'holm_sig': bool(hres[a][1])}
                         for a in AXIS_NAMES_F},
            'axes_sig': sig, 'n_axes_sig': len(sig), 'n_rows': len(rows)}

def po2_stats(rows, pcodes, n_perm=N_PERM_O, seed=None):
    """THE L4 CLAIM: order discrimination. Per pair p, d_p = mean over its
    rows of [acc vs OWN composed code - acc vs the FLIPPED code]; statistic
    = mean d over pairs; null = sign-flips over pairs. An order-indifferent
    reader gives d == 0 identically."""
    import numpy as _np
    if seed is None:
        seed = E8O_SEED + 21
    per_pair = {}
    for r in rows:
        own = pcodes[(r['pair_idx'], r['order'])]
        flip = pcodes[(r['pair_idx'], 1 - r['order'])]
        lv = r['parsed']['levels']
        d = (sum(_acc_vs(lv, own)) - sum(_acc_vs(lv, flip))) / 14.0
        per_pair.setdefault(r['pair_idx'], []).append(d)
    dbar = {p: float(_np.mean(v)) for p, v in per_pair.items()}
    dvec = _np.array([dbar[p] for p in sorted(dbar)])
    obs = float(dvec.mean())
    rng = _np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        s = rng.choice([-1.0, 1.0], size=len(dvec))
        if float((dvec * s).mean()) >= obs:
            ge += 1
    p = (1 + ge) / (1 + n_perm)
    return {'d_mean': round(obs, 4), 'p': p,
            'per_pair': {int(k): round(v, 4) for k, v in dbar.items()},
            'n_pairs': len(dvec), 'n_pos': int((dvec > 0).sum())}

def gid_stats(rows, codes64):
    """Readout-identity: pooled field accuracy on the re-flown E8-F rows."""
    import numpy as _np
    P = levels_matrix(rows)
    T = _np.array([codes64[r['concept']] for r in rows], int)
    acc = float((P == T).mean())
    return {'pooled_acc': round(acc, 4), 'min': GID_MIN,
            'pass': bool(acc >= GID_MIN), 'n_rows': len(rows)}

def span_residual(dirs_new, span_basis):
    """Fraction of each dir's energy OUTSIDE the span (orthonormalized)."""
    Q, _ = np.linalg.qr(np.asarray(span_basis, float).T)
    out = []
    for d in np.asarray(dirs_new, float):
        d = d / np.linalg.norm(d)
        proj = Q @ (Q.T @ d)
        out.append(float(1.0 - np.dot(proj, proj)))
    return np.array(out)


def odu_design_matrix():
    """The 16 figures' DESIGN-side similarity: (4 - Hamming)/4 over the
    tetragram bits — inversion pairs (bitwise complements) at 0, identity
    at 1. Pure design-side; no dir information."""
    figs = list(product((0, 1), repeat=4))
    S = np.zeros((16, 16))
    for i, u in enumerate(figs):
        for j, v in enumerate(figs):
            S[i, j] = (4 - sum(a != b for a, b in zip(u, v))) / 4.0
    return figs, S


def odu_structure_rsa(dirs16, n_perm=2000, seed=E8O_SEED + 7):
    """IFA_SEED §4.3 third instrument: Mantel RSA of the 16-figure
    inversion/Hamming structure vs the measured Odù-dir similarity matrix.
    Sound (design matrix is code-side; dirs are measured side) — replaces
    the seed's 'reverse-map gain' sketch, which the §0 tooth exposed as
    either a linear no-op or target-leaky (on the record)."""
    _, S = odu_design_matrix()
    D = np.asarray(dirs16, float)
    D = D / np.linalg.norm(D, axis=1, keepdims=True)
    M = D @ D.T
    return J.mantel_spearman(S, M, n_perm, seed, progress_every=0)

def odu_spell_texture(rows):
    out = {}
    for r in rows:
        out[r['odu']] = {'kind': r['parsed']['kind'],
                         'code': code_text([r['parsed']['levels'].get(a, 0)
                                            for a in AXIS_NAMES_F])
                         if r['parsed']['kind'] == 'CODE' else None}
    return out


# ── E8-O2 pure logic: mixed curriculum, eval, new stats (locally tested) ─────
# (builds on the e8o_logic namespace: PAIR24, pair_codes, compose_A,
#  composed_code, po1_stats, po2_stats, gid_stats, code_levels, holm, ...)

E8O2_SEED = 20260910
STRANDS_O2 = ['mixed', 'replay', 'sham']
EPOCH_CAP_O2 = 8
GINSTM_MIN = 0.55
PPL_TOL_O2 = 5.0

TRAINPAIR48 = [["ABYSMAL", "REVOLUTION"], ["AMBER", "SUBLIMATION_PHYS"], ["AMOUNT", "CARTILAGE"], ["APPARENT", "SKELETON"], ["ARCADE", "LIMIT"], ["ASSIMILATION", "POINT_ARGUMENT"], ["BELLOWS", "GAME_VERB"], ["BLEND", "LITIGATION"], ["BREW", "TACITNESS"], ["BRILLIANCE", "CAPACITY"], ["CINDER", "CLIENTELE"], ["CLOTH", "ENGAGE"], ["COLLATERAL", "DISSOCIATION"], ["COULD", "OPENING"], ["COVERING", "TUNDRA"], ["COVERT", "TORPOR"], ["CRYSTALLIZATION", "PALM_HAND"], ["DIRGE", "WEAVE"], ["DISCLOSED", "GRATITUDE"], ["DYNAMISM", "PLASTIC"], ["ENJOY", "SLEIGH"], ["ENTROPY_SOCIAL", "LANGUAGE"], ["EVAPORATE", "SACRED"], ["EXPLOITATION", "LOCATION"], ["FIRE_SHOOT", "MINERAL"], ["FORTH", "RABBIT"], ["FREEDOM", "ROUND"], ["GALLANTRY", "SIDE"], ["GONDOLA", "PROSTHESIS"], ["HEMORRHAGE", "WALL"], ["HORSE", "MOLD_SHAPE"], ["ISOTOPE", "OVERWHELM"], ["KELP", "THICK"], ["LOAM", "REALITY"], ["LOCOMOTIVE", "TWEED"], ["MEDITATE", "SUFFICIENT"], ["MIGRATION", "SOURDOUGH"], ["MONEY", "STONE"], ["MONKEY", "TRANSFORM"], ["NEGLIGENCE", "PATINA"], ["NINE", "UNLESS"], ["PASS", "SHAPE"], ["PLATFORM", "PROVERB"], ["POSITION", "WIFE"], ["ROYAL", "UNIVERSAL"], ["SEDIMENTATION_LAKE", "SILTING"], ["SHALLOW", "STEADY"], ["STALK_STEM", "WIELD"]]
POFRESH12 = [["ANOTHER", "PLAN"], ["BARNACLE", "GRANITE"], ["CRUCIBLE", "WAIT"], ["DEGREE", "PREDATOR"], ["DESPONDENCY", "MYSTIC"], ["HORN_ANIMAL", "TELL"], ["IMPASSIVITY", "MATHEMATICS"], ["ON", "SIREN"], ["PERCEPTION", "REMEMBER"], ["PERMANENCE", "PLIGHT"], ["ROOTED", "SHOW"], ["RUDDER", "TEA"]]
SPOTMIX12 = [["ARCADE", "LIMIT"], ["ASSIMILATION", "POINT_ARGUMENT"], ["BREW", "TACITNESS"], ["BRILLIANCE", "CAPACITY"], ["COLLATERAL", "DISSOCIATION"], ["COVERING", "TUNDRA"], ["ENJOY", "SLEIGH"], ["FIRE_SHOOT", "MINERAL"], ["GONDOLA", "PROSTHESIS"], ["KELP", "THICK"], ["MEDITATE", "SUFFICIENT"], ["NEGLIGENCE", "PATINA"]]
REPLAY24 = ["ACCRUE", "CONTEMPLATE", "DORMANCY_STATE", "EXHORT", "EXTORT", "FERMENTATION", "FEUD", "FILTER", "INERTIA", "LETTER_MAIL", "METABOLISM", "OPAL", "PEAT", "PROPAGATION", "REFERENDUM", "REMISSION", "SEDIMENTATION", "SMUGGLE", "SPARK", "STATE", "STORY", "SUDDEN", "VESTIBULE", "WEAK"]
SHA_TRAINPAIR = '8e1dbade408eeb69'
SHA_POFRESH = 'fe6f36106ca209dd'

# flown E8-O flight-1 baselines (the before/after comparison row)
BASELINE_O = {'po1_pooled': 0.4464, 'po2_d': 0.0275, 'po2_p': 0.019,
              'fun_vs_B': 0.3824, 'fun_vs_A_meji': 0.4420,
              'ess_pooled': 0.51, 'gid': 0.5982}

def pair_codes_for(pairs, vecs):
    out = {}
    for i, (a, b) in enumerate(pairs):
        out[(i, 0)] = composed_code(vecs[a], vecs[b])
        out[(i, 1)] = composed_code(vecs[b], vecs[a])
    return out

def mint_stimuli_o2(W5, vecs):
    """carrier + full:<n> (ident + replay) + comp:<i>:<o> (flown PAIR24,
    SAME keys as E8-O) + compT:<i>:<o> (train pairs) + compF:<i>:<o>
    (fresh eval pairs)."""
    import numpy as _np
    Wax, b = _np.asarray(W5, float)[:14], _np.asarray(W5, float)[14]
    def _pred(x):
        v = _np.asarray(x, float) @ Wax + b
        return v / _np.linalg.norm(v)
    stim = {'carrier': _pred(MU_FRAME)}
    for n in sorted({n for n, _ in IDENT16} | set(REPLAY24)):
        stim[f'full:{n}'] = _pred(vecs[n])
    for tag, plist in (('comp', PAIR24), ('compT', TRAINPAIR48),
                       ('compF', POFRESH12)):
        for i, (a, bnm) in enumerate(plist):
            stim[f'{tag}:{i}:0'] = _pred(compose_A(vecs[a], vecs[bnm]))
            stim[f'{tag}:{i}:1'] = _pred(compose_A(vecs[bnm], vecs[a]))
    return stim

def build_e8o2_train(vecs, smoke=False):
    ex, eid = [], 60000
    A = ALPHAS_O[:1] if smoke else ALPHAS_O
    pairs = TRAINPAIR48[:2] if smoke else TRAINPAIR48
    replay = REPLAY24[:4] if smoke else REPLAY24
    n_sham = 2 if smoke else 24
    n_car = 1 if smoke else 4
    def add(strand, kind, skey, alpha, target):
        nonlocal eid
        ex.append({'eid': eid, 'strand': strand, 'kind': kind, 'skey': skey,
                   'layer': TRAIN_LAYER if kind == 'inject' else None,
                   'alpha': alpha, 'target': target})
        eid += 1
    for i, (a, b) in enumerate(pairs):
        for o, (s, j) in enumerate(((a, b), (b, a))):
            tgt = code_text(composed_code(vecs[s], vecs[j]))
            for al in A:
                add('mixed', 'inject', f'compT:{i}:{o}', al, tgt)
    for n in replay:
        for al in A:
            add('replay', 'inject', f'full:{n}', al,
                code_text(code_levels(vecs[n])))
    car_code = code_text(code_levels(MU_FRAME))
    for _ in range(n_car):
        add('replay', 'inject', 'carrier', 1.0, car_code)
    for _ in range(n_sham):
        add('sham', 'sham', None, 0.0, 'NONE')
    return ex

EXPECT_TRAIN_O2 = {'mixed': 192, 'replay': 52, 'sham': 24}

def build_e8o2_eval(smoke=False):
    A = ALPHAS_O[:1] if smoke else ALPHAS_O
    spot = SPOTMIX12[:2] if smoke else SPOTMIX12
    flown = list(range(2)) if smoke else list(range(len(PAIR24)))
    fresh = list(range(2)) if smoke else list(range(len(POFRESH12)))
    ident = IDENT16[:2] if smoke else IDENT16
    n_sham = 2 if smoke else 12
    n_car = 1 if smoke else 2
    rows, tid = [], 70000
    spot_idx = [TRAINPAIR48.index(list(p) if isinstance(p, list) else p)
                if p in TRAINPAIR48 else TRAINPAIR48.index(list(p))
                for p in spot]
    for i in spot_idx:
        for o in (0, 1):
            rows.append({'tid': tid, 'block': 'spot_mixed', 'kind': 'inject',
                         'skey': f'compT:{i}:{o}', 'pair_idx': i, 'order': o,
                         'layer': TRAIN_LAYER, 'alpha': 1.0})
            tid += 1
    for i in flown:
        for o in (0, 1):
            for al in A:
                rows.append({'tid': tid, 'block': 'po', 'kind': 'inject',
                             'skey': f'comp:{i}:{o}', 'pair_idx': i,
                             'order': o, 'layer': TRAIN_LAYER, 'alpha': al})
                tid += 1
    for i in fresh:
        for o in (0, 1):
            for al in A:
                rows.append({'tid': tid, 'block': 'po_fresh', 'kind': 'inject',
                             'skey': f'compF:{i}:{o}', 'pair_idx': i,
                             'order': o, 'layer': TRAIN_LAYER, 'alpha': al})
                tid += 1
    for (n, al) in ident:
        rows.append({'tid': tid, 'block': 'ident', 'kind': 'inject',
                     'skey': f'full:{n}', 'concept': n,
                     'layer': TRAIN_LAYER, 'alpha': al})
        tid += 1
    for _ in range(n_sham):
        rows.append({'tid': tid, 'block': 'sham', 'kind': 'sham',
                     'skey': None, 'layer': None, 'alpha': 0.0})
        tid += 1
    for _ in range(n_car):
        rows.append({'tid': tid, 'block': 'carrier', 'kind': 'inject',
                     'skey': 'carrier', 'layer': TRAIN_LAYER, 'alpha': 1.0})
        tid += 1
    return rows

EXPECT_EVAL_O2 = {'spot_mixed': 24, 'po': 96, 'po_fresh': 48, 'ident': 16,
                  'sham': 12, 'carrier': 2}

def validate_no_eval_in_train_o2(examples):
    ev = set(EVAL64)
    bad = []
    for e in examples:
        sk = e.get('skey') or ''
        if sk.startswith('comp:') or sk.startswith('compF:'):
            bad.append(e.get('eid'))
        if sk.startswith('full:') and sk[5:] in ev:
            bad.append(e.get('eid'))
        if sk.startswith('compT:'):
            i = int(sk.split(':')[1])
            if set(TRAINPAIR48[i]) & ev:
                bad.append(e.get('eid'))
    return bad

def pm2_stats(rows, pcodes, n_perm=N_PERM_O, seed=None):
    """P-M2: the FUNCTION-LEG pooled accuracy vs pair-derangements —
    po1_stats restricted to axes 7..13."""
    import numpy as _np
    if seed is None:
        seed = E8O2_SEED + 20
    fun = list(range(7, 14))
    P = levels_matrix(rows)[:, fun]
    pair_ids = sorted({r['pair_idx'] for r in rows})
    T = _np.array([pcodes[(r['pair_idx'], r['order'])] for r in rows],
                  int)[:, fun]
    obs = float((P == T).mean())
    rng = _np.random.default_rng(seed)
    idx = _np.arange(len(pair_ids))
    pos = {p: k for k, p in enumerate(pair_ids)}
    ge, nulls = 0, []
    for _ in range(n_perm):
        pi = rng.permutation(idx)
        while _np.any(pi == idx):
            pi = rng.permutation(idx)
        Tn = _np.array([pcodes[(pair_ids[pi[pos[r['pair_idx']]]], r['order'])]
                        for r in rows], int)[:, fun]
        nm = float((P == Tn).mean())
        nulls.append(nm)
        ge += nm >= obs
    return {'pooled_acc': round(obs, 4), 'p': (1 + ge) / (1 + n_perm),
            'null_mean': round(float(_np.mean(nulls)), 4), 'n_rows': len(rows)}

def ess_acc(rows, pcodes):
    import numpy as _np
    ess = list(range(7))
    P = levels_matrix(rows)[:, ess]
    T = _np.array([pcodes[(r['pair_idx'], r['order'])] for r in rows],
                  int)[:, ess]
    return round(float((P == T).mean()), 4)

def meji_completion_code(vec):
    """The essence-donor's own full code — what the shortcut predicts the
    function fields to be (its function leg's code)."""
    return code_levels(vec)


def smeji_stat(rows, pcodes, vecs, pair_list):
    """S-MEJI reversal: mean over rows of
    [function-field acc vs the JUNIOR leg's true content] minus
    [function-field acc vs the SENIOR donor's méjì completion].
    Positive = shortcut broken; negative = shortcut reading."""
    fun_idx = list(range(7, 14))
    ds = []
    for r in rows:
        a, b = pair_list[r["pair_idx"]]
        if r["order"] == 1:
            a, b = b, a
        lv = r["parsed"]["levels"]
        own = pcodes[(r["pair_idx"], r["order"])]
        meji = meji_completion_code(vecs[a])
        acc_b = np.mean([lv.get(AXIS_NAMES_F[j], 99) == own[j]
                         for j in fun_idx])
        acc_a = np.mean([lv.get(AXIS_NAMES_F[j], 99) == meji[j]
                         for j in fun_idx])
        ds.append(acc_b - acc_a)
    return float(np.mean(ds)), ds

def smeji_test(rows, pcodes, vecs, pair_list, n_perm=N_PERM_O, seed=None):
    """S-MEJI: reversal statistic + sign-flip p over pairs (one-sided > 0)."""
    import numpy as _np
    if seed is None:
        seed = E8O2_SEED + 21
    obs, ds = smeji_stat(rows, pcodes, vecs, pair_list)
    per_pair = {}
    for r, d in zip(rows, ds):
        per_pair.setdefault(r['pair_idx'], []).append(d)
    dvec = _np.array([float(_np.mean(v)) for _, v in sorted(per_pair.items())])
    rng = _np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        s = rng.choice([-1.0, 1.0], size=len(dvec))
        if float((dvec * s).mean()) >= float(dvec.mean()):
            ge += 1
    return {'d_mean': round(obs, 4), 'p': (1 + ge) / (1 + n_perm),
            'n_pairs': len(dvec), 'n_pos': int((dvec > 0).sum())}
