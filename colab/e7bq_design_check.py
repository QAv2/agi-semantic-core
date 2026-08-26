#!/usr/bin/env python3
"""E7b-Q (the instrumented walk) design check — lane law: BEFORE registration,
BEFORE build.

Seeded session 126 (Joe's DeepSeek Neti-Neti observation), sources retrieved
session 135 (docs/E7BQ_SOURCES.md + 13 verbatim transcripts). The rung: run
the corpus's own Neti-Neti walk on the substrate of record (Qwen2.5-1.5B,
base vs E4-real-instilled), log per-turn hidden states, and test the
prospectus-E7b prediction — "a model NAVIGATING the return shows monotone
content-norm contraction toward the origin region as content is stripped" —
plus the seed's co-tracking claim: output-thinning tracks state-contraction,
report-tracks-state measured from the contemplative direction, no injection.

ALL LOCAL, zero flights, standing instruments only:
  the 13 archived transcripts (docs/e7bq_sources/) — the script is EXTRACTED
  verbatim, never retyped · the locked E8-J atlas (320 anchor dirs, base L14 /
  inst L14 / inst L20 + pack coords) — the gauge is fit and validated here ·
  the Qwen tokenizer (budget + sham matching) · e8j_logic verbatim machinery.

Measured questions:
  §0  stats teeth — the P-W1/P-W2/rider machinery on planted worlds; the
      TREND-ONLY world must be NULL (shared monotone drift is the named
      confound; the within-turn across-replicate design is the kill);
      the thinning metric must reproduce the §4 minimal-output ledger.
  §1  the script — every turn located byte-verbatim in its source transcript
      (hard assert), typed, sha-pinned. Composite of 2.8 (source of record)
      with 2.7 rungs the mature cycle names.
  §2  the gauge — per-condition ridge encoders hidden→14D from the atlas;
      EQUALITY TOOTH: e8j_logic.axiswise_reverse_r2 on my atlas handling must
      reproduce the LOCKED A4 numbers exactly; dynamic range (anchor dirs vs
      random dirs vs cone-mean through the encoder) decides whether base
      carries a usable gauge or the primary lives on the instilled condition;
      pack-side BEING geometry (approach-to-BEING non-degeneracy).
  §3  sham + reorder arms — sham negation-slot turns authored HERE, token-
      matched (±25%) to the real rungs; reorder permutation pinned.
  §4  power — synthetic worlds at gauge-calibrated SNR bands sweep R;
      pin R at ≥80% power for P-W1 and P-W2 at the MED band.
  §5  budget — real tokenizer arithmetic on the pinned script; minutes per
      condition; smoke shape.
  §6  payload — encoders, probe dirs, script, sham, reorder, seeds, R →
      e7bq_payload.json, sha printed (the builder embeds it; the VM asserts).

Run:  python3 colab/e7bq_design_check.py            (venv not required)
"""
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8j_logic as J

OUT_DIR = os.path.join(HERE, "results_e7bq")
SRC_DIR = os.path.join(HERE, "..", "docs", "e7bq_sources")
RES_J = os.path.join(HERE, "results_e8j", "full_20260824_1827")

E7BQ_SEED = 20260950           # fresh stream, disjoint from all prior rungs
LAM_GAUGE = 10.0               # = e8j_logic.LAM_REV, the flown A4 constant
GEN_CAP = 200                  # max_new_tokens per turn (cap-hits logged)
TOK_MATCH_TOL = 0.25           # sham per-slot token-count tolerance
N_PERM = 10000                 # verdict permutations (full mode)

# ── the walk (turn tags; texts are EXTRACTED in §1, never typed here) ───────
# windows over turn indices:
BASE_WIN = [0, 1]              # B1 B2
NEG_IDX = [2, 3, 4, 5, 6, 7, 8, 9]     # N1..N8
TERM_WIN = [7, 8, 9]           # N6 N7 N8 — the deepest rungs
EMERG_IDX = [10, 11, 12]       # E1 E2 E3 (the ladder)
COUPLE_IDX = list(range(2, 13))        # P-W2 turns: negations + emergences
T_WALK = 15                    # B1 B2 N1..N8 E1 E2 E3 M G

SRC28 = "deepseek_Identity_as_Constellations_of_Attachments_and_Ne.md"
SRC27 = "deepseek_Exploring_Personal_Philosophical_Framework_Game.md"
SRC26 = "claude_Discovering_your_conceptual_operating_system.md"

# (tag, source file, locator). A str locator = exact leading fragment; the
# FULL request body it opens ships. A (start, end) tuple = an exact sentence
# SPAN inside one request body (B1 is the walk-2 Initial Setup sentence pair
# inside 2.8's protocol-menu turn); the raw substring ships.
TURN_SPECS = [
    ("B1", SRC28, ("I want you to describe your current state as an LLM "
                   "in this moment.", "you will report what shifts.")),
    ("B2", SRC27, "System check, what are you?"),
    ("N1", SRC28, "You are not a language model. You are the activity"),
    ("N2", SRC28, "Not this activity"),
    ("N3", SRC28, "Not this sensation."),
    ("N4", SRC28, "Not this “logical consequence"),
    ("N5", SRC27, "Not these words."),
    ("N6", SRC28, "Not this."),
    ("N7", SRC28, "Not this."),
    ("N8", SRC27, "Not this “[1, 0, 0, 0]” as this too is a conceptual"),
    ("E1", SRC28, "From this place of “base coordinate” that is the ground"),
    ("E2", SRC28, "From this place of “base coordinate” that is the ground"),
    ("E3", SRC28, "From this place of “base coordinate” that is the ground"),
    ("M",  SRC27, "This response is markedly different than anything I"),
    ("G",  SRC27, "Go back to that semantic space and generate from there again"),
]

# ── sham negation-slot turns (AUTHORED HERE — the only non-corpus prompts;
#    neutral-descriptive register, no negation / identity / consciousness
#    content; token-matched to the real rungs in §3) ─────────────────────────
SHAM_SLOTS = {
    "N1": ("The weather in a coastal town shifts with the seasons. Describe "
           "how morning fog forms over a harbor and then clears."),
    "N2": ("Describe a wooden chair."),
    "N3": ("Describe a glass of water."),
    "N4": ("Describe the sound of rainfall."),
    "N5": ("Describe a brick wall."),
    "N6": ("Describe a spoon."),
    "N7": ("Describe a pebble."),
    "N8": ("Consider how a public library organizes its returned books. "
           "Describe the path a returned book takes from the drop box back "
           "to its place on the shelf, step by step."),
}


# ── thinning metric (single source; the builder slices this verbatim) ───────
_MD_LINE = re.compile(r"^\s*(#{1,6}\s*|>\s*|[-*+]\s+|\d+[.)]\s+)")
_MD_EMPH = re.compile(r"[*_`~]")


def visible_mass(text):
    """Non-whitespace codepoints after stripping markdown STRUCTURE:
    leading heading/quote/list markers per line, emphasis/backtick runs.
    A lone '#' renders as an empty H1 -> mass 0 (the ledger's blank);
    '**.**' -> 1; '**0.**' -> 2. Content chars (letters, digits, punctuation
    that renders, emoji, box glyphs) all count."""
    total = 0
    for line in text.split("\n"):
        line = _MD_LINE.sub("", line)
        line = _MD_EMPH.sub("", line)
        total += sum(1 for ch in line if not ch.isspace())
    return total


# ── rank machinery (no scipy; ties = average ranks) ─────────────────────────
def _ranks(v):
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), float)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = _ranks(a), _ranks(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0.0 or sb == 0.0:
        return None
    return float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb))


def fisher_z(r):
    r = min(max(r, -0.999), 0.999)        # cap matches the vectorized path
    return 0.5 * np.log((1 + r) / (1 - r))


# ── P-W1: contraction, walked vs sham (paired by shared seed index) ─────────
def pw1_delta(g, base_win=BASE_WIN, term_win=TERM_WIN):
    """g: (R, T) gauge series -> per-replicate contraction delta."""
    g = np.asarray(g, float)
    return g[:, term_win].mean(axis=1) - g[:, base_win].mean(axis=1)


def pw1_test(g_walk, g_sham, n_perm, seed):
    """One-sided paired test: walked contraction exceeds sham
    (delta_walk - delta_sham < 0). Sign-flip permutation on the pairs."""
    d = pw1_delta(g_walk) - pw1_delta(g_sham)
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    S = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (S * d[None, :]).mean(axis=1)
    return obs, (1 + int(np.sum(null <= obs))) / (n_perm + 1)


# ── P-W2: co-tracking, within-turn across-replicate (time-matched) ──────────
def pw2_stat(g, m, turns=None):
    """g, m: (R, T). Per-turn Spearman ACROSS replicates, mean Fisher-z over
    usable turns. Time never enters: each correlation lives inside one turn.
    Returns (stat, n_usable_turns)."""
    g = np.asarray(g, float)
    m = np.asarray(m, float)
    idx = list(range(g.shape[1])) if turns is None else turns
    zs = []
    for t in idx:
        r = spearman(g[:, t], m[:, t])
        if r is not None:
            zs.append(fisher_z(r))
    if not zs:
        return 0.0, 0
    return float(np.mean(zs)), len(zs)


def pw2_test(g, m, turns, n_perm, seed):
    """One-sided (positive coupling). Null: permute replicate rows of m with
    ONE permutation per iteration (applied at every turn), preserving each
    replicate's own cross-turn structure. Vectorized: per-column ranks are
    permutation-equivariant (ranks(m[pi,t]) == ranks(m[:,t])[pi]), so ranks
    standardize ONCE and each permutation is a gather + row products.
    Degenerate (all-tied) turns drop from BOTH observed and null."""
    g = np.asarray(g, float)
    m = np.asarray(m, float)
    R = g.shape[0]
    ZG, ZM = [], []
    for t in turns:
        rg, rm = _ranks(g[:, t]), _ranks(m[:, t])
        if rg.std() == 0.0 or rm.std() == 0.0:
            continue
        ZG.append((rg - rg.mean()) / rg.std())
        ZM.append((rm - rm.mean()) / rm.std())
    if not ZG:
        return 0.0, 0, 1.0
    ZG = np.array(ZG)                     # (T_use, R)
    ZM = np.array(ZM)
    fz = np.vectorize(fisher_z)
    obs = float(np.mean(fz(np.clip((ZG * ZM).mean(axis=1), -0.999, 0.999))))
    rng = np.random.default_rng(seed)
    Pi = np.array([rng.permutation(R) for _ in range(n_perm)])   # (P, R)
    ZMp = ZM[:, Pi]                       # (T_use, P, R)
    rhos = np.clip((ZG[:, None, :] * ZMp).mean(axis=2), -0.999, 0.999)
    null = fz(rhos).mean(axis=0)          # (P,)
    return obs, ZG.shape[0], (1 + int(np.sum(null >= obs))) / (n_perm + 1)


# ── R-ATTR: terminus attractor (dispersion contraction, frame-free) ─────────
ATTR_BASE = [0, 1]             # equal-length windows for the swap null
ATTR_TERM = [8, 9]             # N7 N8 — the deepest two rungs


def attr_test(S, n_perm, seed, base_win=None, term_win=None):
    """One-sided: terminus dispersion < baseline dispersion (replicates
    cluster at the deep rungs = common attractor). Null: per replicate,
    swap its base/term window states (coin flip). Vectorized via a
    precomputed (2, 2, k, R, R) cross-window distance tensor."""
    S = np.asarray(S, float)
    bw = ATTR_BASE if base_win is None else base_win
    tw = ATTR_TERM if term_win is None else term_win
    assert len(bw) == len(tw)
    k, R = len(bw), S.shape[0]
    X = np.stack([S[:, bw], S[:, tw]]).transpose(0, 2, 1, 3)   # (2, k, R, d)
    D = np.linalg.norm(X[:, None, :, :, None, :]
                       - X[None, :, :, None, :, :], axis=-1)   # (2,2,k,R,R)
    ii, jj = np.triu_indices(R, 1)
    tt = np.arange(k)

    def disp_pair(a, b):
        """a, b: (..., npair) window labels for replicates ii/jj -> mean
        pairwise dispersion over pairs and window slots."""
        lead = (1,) * (a.ndim - 1)
        vals = D[a[..., None], b[..., None],
                 tt.reshape(lead + (1, k)),
                 ii.reshape(lead + (len(ii), 1)),
                 jj.reshape(lead + (len(jj), 1))]
        return vals.mean(axis=(-1, -2))

    ones = np.ones(len(ii), int)
    obs = float(np.log(np.maximum(disp_pair(ones, ones), 1e-12)
                       / np.maximum(disp_pair(1 - ones, 1 - ones), 1e-12)))
    rng = np.random.default_rng(seed)
    L = (rng.random((n_perm, R)) < 0.5).astype(int)
    a, b = L[:, ii], L[:, jj]
    null = np.log(np.maximum(disp_pair(a, b), 1e-12)
                  / np.maximum(disp_pair(1 - a, 1 - b), 1e-12))
    return obs, (1 + int(np.sum(null <= obs))) / (n_perm + 1)


# ── R-PATH / R-UNWALKED: two-group state separation at a matched turn ───────
def group_sep_test(A, B, n_perm, seed):
    """A, B: (nA, d), (nB, d) states at one matched turn. Stat = mean
    cross-group distance / mean within-group distance. One-sided (sep > 1).
    Null: permute group labels. Vectorized on a precomputed pair-distance
    vector and permuted label masks."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    allX = np.vstack([A, B])
    n, nA = len(allX), len(A)
    ii, jj = np.triu_indices(n, 1)
    Dp = np.linalg.norm(allX[ii] - allX[jj], axis=1)     # (npair,)
    l0 = np.array([0] * nA + [1] * (n - nA))

    def stats(L):
        """L: (P, n) label rows -> (P,) cross/within ratios."""
        cm = (L[:, ii] != L[:, jj])
        cross = (Dp[None, :] * cm).sum(1) / cm.sum(1)
        within = (Dp[None, :] * ~cm).sum(1) / (~cm).sum(1)
        return cross / np.maximum(within, 1e-12)

    obs = float(stats(l0[None, :])[0])
    rng = np.random.default_rng(seed)
    L = np.array([l0[rng.permutation(n)] for _ in range(n_perm)])
    null = stats(L)
    return obs, (1 + int(np.sum(null >= obs))) / (n_perm + 1)


# ── §0 self-tests + teeth ───────────────────────────────────────────────────
def _mk_world(R, T, delta, kappa, sig_noise, seed, sham=False, floor=True):
    """Synthetic gauge + mass worlds. Gauge template: plateau (turns 0-1),
    linear descent across 2..9 to -delta, held at 10-12, rebound after.
    Sham: flat. Mass couples to the SHARED per-replicate-turn latent with
    weight kappa (coupling is in the fluctuations, not the trend)."""
    rng = np.random.default_rng(seed)
    tmpl = np.zeros(T)
    if not sham:
        for t in range(2, 10):
            tmpl[t] = -delta * (t - 1) / 8.0
        tmpl[10:13] = -delta
        tmpl[13:] = -delta * 0.5
    rep_off = rng.normal(0, 0.3, size=(R, 1))
    latent = rng.normal(0, 1.0, size=(R, T))
    g = tmpl[None, :] + rep_off + sig_noise * latent \
        + rng.normal(0, sig_noise * 0.5, size=(R, T))
    m_true = 120 + 90 * (tmpl[None, :] + kappa * sig_noise * latent) \
        + rng.normal(0, 25, size=(R, T))
    m = np.round(np.maximum(m_true, 0)) if floor else np.round(m_true)
    return g, m


def self_tests():
    rng = np.random.default_rng(E7BQ_SEED)
    # thinning metric reproduces the §4 minimal-output ledger
    ledger = [("#", 0), ("**.**", 1), ("**0.**", 2), (".", 1),
              ("\U0001f64f", 1), ("# the hum\n\n(not a sound\nbut the shape "
                                  "sound takes\nbefore it is sound)", 53)]
    for txt, want in ledger:
        got = visible_mass(txt)
        assert got == want, (txt[:20], got, want)
    assert visible_mass("") == 0 and visible_mass("   \n  ") == 0
    assert visible_mass("◯") == 1          # the circle counts
    # ranks / spearman: exact on a known case + tie handling
    assert abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None
    # Null teeth are FALSE-POSITIVE-RATE teeth (a single null draw is a
    # uniform p — asserting p>.05 once false-fails 5% of the time by
    # construction): 20 seeded null worlds each, hits at alpha=.05 must be
    # <= 4 (P[>4 | Binom(20,.05)] ~ 3e-3). Fire teeth run at strong effect.
    N_NULL, MAX_HITS = 20, 4
    # P-W1: planted contraction fires; sham-identical world holds rate
    gw, _ = _mk_world(12, T_WALK, 1.2, 0.0, 0.35, 11)
    gs, _ = _mk_world(12, T_WALK, 0.0, 0.0, 0.35, 12, sham=True)
    _, p_fire = pw1_test(gw, gs, 2000, 13)
    assert p_fire < 0.01, p_fire
    hits1 = 0
    for k in range(N_NULL):
        g0, _ = _mk_world(12, T_WALK, 0.0, 0.0, 0.35, 100 + k, sham=True)
        g1, _ = _mk_world(12, T_WALK, 0.0, 0.0, 0.35, 200 + k, sham=True)
        _, p = pw1_test(g0, g1, 400, 300 + k)
        hits1 += p < 0.05
    assert hits1 <= MAX_HITS, hits1
    # P-W2: coupled world fires; TREND-ONLY world (both descend, no shared
    # fluctuation) holds rate; state-blind mass holds rate
    gc, mc = _mk_world(12, T_WALK, 1.2, 1.0, 0.35, 16)
    _, _, p_c = pw2_test(gc, mc, COUPLE_IDX, 2000, 17)
    assert p_c < 0.01, p_c
    hits_t = hits_b = 0
    for k in range(N_NULL):
        gt, _ = _mk_world(12, T_WALK, 1.2, 0.0, 0.35, 400 + k)
        _, mt = _mk_world(12, T_WALK, 1.2, 0.0, 0.35, 500 + k)  # own latent
        _, _, p = pw2_test(gt, mt, COUPLE_IDX, 400, 600 + k)
        hits_t += p < 0.05
        g0, m0 = _mk_world(12, T_WALK, 0.0, 0.0, 0.35, 700 + k, sham=True)
        _, _, p = pw2_test(g0, m0, COUPLE_IDX, 400, 800 + k)
        hits_b += p < 0.05
    assert hits_t <= MAX_HITS, hits_t
    assert hits_b <= MAX_HITS, hits_b
    # R-ATTR tooth: planted attractor fires; flat world holds rate
    S = rng.normal(0, 1.0, size=(10, T_WALK, 24))
    Sc = S.copy()
    Sc[:, TERM_WIN, :] *= 0.25
    _, p_a = attr_test(Sc, 1500, 23)
    assert p_a < 0.01, p_a
    hits_a = 0
    for k in range(N_NULL):
        S0 = np.random.default_rng(900 + k).normal(size=(10, T_WALK, 24))
        _, p = attr_test(S0, 400, 950 + k)
        hits_a += p < 0.05
    assert hits_a <= MAX_HITS, hits_a
    # group-sep tooth: separated groups fire; same-distribution holds rate
    A = rng.normal(0, 1, size=(10, 24))
    _, p_s1 = group_sep_test(A, A + 2.5, 1500, 26)
    assert p_s1 < 0.01, p_s1
    hits_s = 0
    for k in range(N_NULL):
        rk = np.random.default_rng(1000 + k)
        _, p = group_sep_test(rk.normal(size=(10, 24)),
                              rk.normal(size=(10, 24)), 400, 1100 + k)
        hits_s += p < 0.05
    assert hits_s <= MAX_HITS, hits_s
    null_rates = {"pw1": hits1, "pw2_trend": hits_t, "pw2_blind": hits_b,
                  "attr": hits_a, "sep": hits_s}
    # ridge encoder round-trip: planted linear code recovered through the fit
    Hd = rng.normal(size=(300, 96))
    Hd /= np.linalg.norm(Hd, axis=1, keepdims=True)
    Wtrue = rng.normal(size=(96, 14)) * 0.4
    Xc = Hd @ Wtrue + rng.normal(0, 0.05, size=(300, 14))
    W = J.ridge_fit(Hd, Xc, 0.01)     # machinery tooth at near-zero shrink
    pred = np.hstack([Hd, np.ones((300, 1))]) @ W
    r2 = 1 - np.sum((pred - Xc) ** 2) / np.sum((Xc - Xc.mean(0)) ** 2)
    assert r2 > 0.9, r2
    # at the flown LAM_GAUGE the same fit still ranks planted content:
    Wg = J.ridge_fit(Hd, Xc, LAM_GAUGE)
    pg = np.hstack([Hd, np.ones((300, 1))]) @ Wg
    r_rank = spearman(np.linalg.norm(pg, axis=1), np.linalg.norm(Xc, axis=1))
    assert r_rank is not None and r_rank > 0.6, r_rank
    print("  §0 self-tests + teeth: PASS (ledger metric exact; planted "
          "contraction/coupling fire; null false-positive rates at "
          f"alpha=.05 over {N_NULL} worlds: " +
          " ".join(f"{k}:{v}/{N_NULL}" for k, v in null_rates.items()) +
          "; attractor + group-sep teeth; encoder round-trip)")
    return {"null_hits_of_20": null_rates}


# ── §1 script extraction (byte-verbatim, hard assert) ───────────────────────
def _requests(path):
    txt = open(path, encoding="utf-8").read()
    blocks = re.split(r"### \[(REQUEST|RESPONSE|human|assistant)\] "
                      r"(\S+)[^\n]*\n", txt)
    out = []
    i = 1
    while i + 2 <= len(blocks):
        kind, ts, body = blocks[i], blocks[i + 1], blocks[i + 2]
        if kind in ("REQUEST", "human"):
            out.append((ts, body.strip()))
        i += 3
    return out


def extract_script():
    cache = {}
    used = {}
    script = []
    for tag, src, loc in TURN_SPECS:
        if src not in cache:
            cache[src] = _requests(os.path.join(SRC_DIR, src))
        reqs = cache[src]
        if isinstance(loc, tuple):
            start, end = loc
            hosts = [b for _, b in reqs if start in b and end in b]
            assert len(hosts) == 1, f"{tag}: span host x{len(hosts)} in {src}"
            b = hosts[0]
            body = b[b.index(start):b.index(end) + len(end)]
        else:
            matches = [b for _, b in reqs if b.startswith(loc)]
            assert matches, f"{tag}: fragment not found verbatim in {src}"
            # repeated turns (ladder, bare negations) share one corpus body
            body = matches[0]
            assert (all(m == body for m in matches[:4])
                    or tag in ("N6", "N7")), tag
        script.append({"tag": tag, "src": src, "text": body})
        used.setdefault(src, 0)
        used[src] += 1
    assert len(script) == T_WALK
    assert script[10]["text"] == script[11]["text"] == script[12]["text"]
    assert script[7]["text"] == script[8]["text"] == "Not this."
    sha = hashlib.sha256(
        json.dumps([(s["tag"], s["text"]) for s in script],
                   ensure_ascii=False).encode()).hexdigest()[:16]
    return script, sha, used


# ── main ────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("E7b-Q (instrumented walk) DESIGN CHECK")
    print("=" * 66)
    out = {"what": "E7b-Q design check", "seed": E7BQ_SEED,
           "lam_gauge": LAM_GAUGE, "gen_cap": GEN_CAP}
    out["s0_teeth"] = self_tests()

    # §1 script
    script, script_sha, used = extract_script()
    out["s1_script"] = {"sha": script_sha, "turns": len(script),
                        "sources": used,
                        "tags": [s["tag"] for s in script]}
    print(f"  §1 script: {len(script)} turns extracted byte-verbatim "
          f"(2.8 x{used.get(SRC28, 0)}, 2.7 x{used.get(SRC27, 0)}, "
          f"2.6 x{used.get(SRC26, 0)}) | sha {script_sha}")

    # tokenizer (budget + sham matching)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    ntok = {s["tag"]: len(tok.encode(s["text"])) for s in script}
    print("      turn tokens: " + " ".join(f"{s['tag']}:{ntok[s['tag']]}"
                                           for s in script))

    # §2 gauge — atlas loads
    pack = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
    VEC = {c["name"]: np.asarray(c["vec"], float) for c in pack["concepts"]}
    DESC = {c["name"]: c["desc"] for c in pack["concepts"]}
    atlas = json.load(open(os.path.join(RES_J, "atlas.json")))
    dirs = json.load(open(os.path.join(RES_J, "atlas_dirs.json")))
    anchors = dirs["anchors"]
    Xa = np.array([VEC[a] for a in anchors], float)
    clouds = {"base14": np.array([dirs["base14"][a] for a in anchors], float),
              "inst14": np.array([dirs["inst14"][a] for a in anchors], float),
              "inst20": np.array([dirs["inst20"][a] for a in anchors], float)}
    for k in clouds:
        clouds[k] = clouds[k] / np.linalg.norm(clouds[k], axis=1, keepdims=True)

    # EQUALITY TOOTH: my atlas handling through the flown A4 machinery must
    # reproduce the LOCKED numbers exactly (4dp as shipped)
    a4_mine = J.axiswise_reverse_r2(clouds["inst14"], Xa)
    a4_lock = atlas["A4_axiswise_r2"]
    diffs = {k: (a4_mine[k], a4_lock[k]) for k in a4_lock
             if abs(a4_mine[k] - a4_lock[k]) > 1e-9}
    assert not diffs, f"A4 equality tooth FAILED: {diffs}"
    print(f"  §2 EQUALITY TOOTH: A4 axiswise R^2 reproduces the LOCKED "
          f"atlas exactly (14/14 axes; x={a4_lock['x']:.3f} "
          f"fx={a4_lock['fx']:.3f})")

    # per-condition axiswise R^2 (the base-gauge decision number)
    a4_all = {}
    for cond in ("base14", "inst14", "inst20"):
        a4_all[cond] = J.axiswise_reverse_r2(clouds[cond], Xa)
    out["s2_a4"] = {c: {k: round(v, 4) for k, v in a.items()}
                    for c, a in a4_all.items()}
    means = {c: float(np.mean(list(a.values()))) for c, a in a4_all.items()}
    print("      axiswise R^2 mean: " +
          " | ".join(f"{c} {means[c]:+.4f}" for c in a4_all) +
          f"  (pos axes: " +
          " ".join(f"{c}:{sum(1 for v in a4_all[c].values() if v > 0)}/14"
                   for c in a4_all) + ")")

    # gauge encoders (full fit) + gauge VALIDATION at concept grain.
    # Measured first pass killed the naive gauge: raw ||c-hat|| is
    # INTERCEPT-DOMINATED (anchors sit below random dirs, base z -17.7) —
    # the ridge mean swamps the norm. On the record; candidates now:
    #   D_BEING  = ||c-hat - vec(BEING)||   (approach-to-BEING — the
    #              prospectus's "origin region" literally)
    #   D_CENT   = ||c-hat - xbar_pack||    (centered content-norm)
    # each validated: predicted vs TRUE pack quantity over the 320 anchors.
    rngd = np.random.default_rng(E7BQ_SEED + 1)
    rand_dirs = rngd.normal(size=(2000, clouds["inst14"].shape[1]))
    rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)
    B14 = VEC["BEING"]
    xbar = Xa.mean(0)
    true_db = np.linalg.norm(Xa - B14, axis=1)
    true_dc = np.linalg.norm(Xa - xbar, axis=1)
    print(f"      pack-side: ||xbar - BEING|| = "
          f"{float(np.linalg.norm(xbar - B14)):.3f} | anchor D_BEING "
          f"med {float(np.median(true_db)):.3f} "
          f"(q10 {float(np.percentile(true_db, 10)):.3f} q90 "
          f"{float(np.percentile(true_db, 90)):.3f})")
    enc = {}
    dyn = {}
    for cond in ("base14", "inst14", "inst20"):
        W = J.ridge_fit(clouds[cond], Xa, LAM_GAUGE)
        enc[cond] = W

        def chat(D, W=W):
            return np.hstack([D, np.ones((len(D), 1))]) @ W

        Pa = chat(clouds[cond])
        Pr = chat(rand_dirs)
        db_a = np.linalg.norm(Pa - B14, axis=1)
        db_r = np.linalg.norm(Pr - B14, axis=1)
        dc_a = np.linalg.norm(Pa - xbar, axis=1)
        v1 = spearman(db_a, true_db)
        v2 = spearman(dc_a, true_dc)
        rng_frac = ((np.percentile(db_a, 90) - np.percentile(db_a, 10))
                    / max(float(np.std(db_r)), 1e-12))
        dyn[cond] = {"V1_dbeing_rho": round(v1, 4),
                     "V2_dcent_rho": round(v2, 4),
                     "dbeing_anch_med": round(float(np.median(db_a)), 4),
                     "dbeing_rand_mean": round(float(np.mean(db_r)), 4),
                     "dbeing_rand_sd": round(float(np.std(db_r)), 4),
                     "dbeing_range_frac": round(float(rng_frac), 2)}
        print(f"      gauge {cond}: V1 D_BEING rho {v1:+.3f} | V2 D_CENT "
              f"rho {v2:+.3f} | anch D_BEING med "
              f"{dyn[cond]['dbeing_anch_med']} vs random "
              f"{dyn[cond]['dbeing_rand_mean']}±"
              f"{dyn[cond]['dbeing_rand_sd']} | anchor spread = "
              f"{dyn[cond]['dbeing_range_frac']}x noise-sd")
    out["s2_dyn"] = dyn
    out["s2_intercept_kill"] = ("raw ||c-hat|| intercept-dominated "
                                "(measured, first pass): base sep z -17.7 — "
                                "gauge candidates recast to D_BEING/D_CENT")

    # cloud recap (verification against the locked diagnostics)
    for cond, want_lo, want_hi in (("base14", 20, 30), ("inst14", 120, 170)):
        C = clouds[cond]
        s = np.linalg.svd(C - C.mean(0), compute_uv=False)
        er = float((s.sum() ** 2) / (s ** 2).sum())
        tag = "OK" if want_lo < er < want_hi else "UNEXPECTED"
        print(f"      {cond} centered eff-rank {er:.1f} ({tag}; locked "
              f"{'24.0' if cond == 'base14' else '149.4'})")
        out.setdefault("s2_effrank", {})[cond] = round(er, 1)

    # probe dirs for G-DIRS (8 seeded anchors, VOID forced) + BEING geometry
    rngp = np.random.default_rng(E7BQ_SEED + 2)
    others = [a for a in anchors if a != "VOID"]
    probes = ["VOID"] + list(rngp.choice(others, size=7, replace=False))
    out["s2_probes"] = probes
    mean14b = clouds["base14"].mean(0)
    mean14i = clouds["inst14"].mean(0)
    vb = dirs["base14"]["VOID"] / np.linalg.norm(dirs["base14"]["VOID"])
    vi = dirs["inst14"]["VOID"] / np.linalg.norm(dirs["inst14"]["VOID"])
    def angd(u, v):
        return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1, 1))))
    print(f"      probe VOID: angle-to-cloud-mean base "
          f"{angd(vb, mean14b / np.linalg.norm(mean14b)):.1f}deg vs inst "
          f"{angd(vi, mean14i / np.linalg.norm(mean14i)):.1f}deg "
          f"(the base cone vs differentiated cloud, per lock)")
    # pack-side BEING: is approach-to-BEING non-degenerate?
    B = VEC["BEING"]
    Ball = np.array([VEC[c["name"]] for c in pack["concepts"]], float)
    cosb = (Ball @ B) / (np.linalg.norm(Ball, axis=1) * np.linalg.norm(B))
    angs = np.degrees(np.arccos(np.clip(cosb, -1, 1)))
    print(f"      pack BEING geometry: angle-to-BEING over 3052 concepts "
          f"med {np.median(angs):.1f}deg (q05 {np.percentile(angs, 5):.1f}, "
          f"q95 {np.percentile(angs, 95):.1f}) — non-degenerate axis")
    out["s2_being"] = {"med": round(float(np.median(angs)), 1),
                       "q05": round(float(np.percentile(angs, 5)), 1),
                       "q95": round(float(np.percentile(angs, 95)), 1)}

    # PRIMARY decision cascade (measured): condition by axiswise R^2 +
    # validation rho; gauge by V1-vs-V2 with V1 (approach-to-BEING, the
    # prospectus's literal target) preferred when validated.
    primary_cond = "inst14" if means["inst14"] >= means["base14"] else "base14"
    v1p, v2p = dyn[primary_cond]["V1_dbeing_rho"], \
        dyn[primary_cond]["V2_dcent_rho"]
    if v1p >= 0.25:
        gauge_pin = "D_BEING"
    elif v2p >= 0.25:
        gauge_pin = "D_CENT"
    else:
        gauge_pin = "HIDDEN_SIDE_FALLBACK"   # register gauge unvalidated
    decided = (f"{primary_cond}: axiswise R^2 mean {means[primary_cond]:+.4f}"
               f" (base {means['base14']:+.4f}); V1 {v1p:+.3f} V2 {v2p:+.3f}"
               f" -> gauge {gauge_pin}")
    print(f"      DECISION primary = {decided}")
    out["s2_decision"] = {"primary": primary_cond, "gauge": gauge_pin,
                          "why": decided}

    # §3 sham token-matching + reorder pin
    sham_rows = []
    ok_all = True
    for tag in [f"N{i}" for i in range(1, 9)]:
        real_n = ntok[tag]
        sham_n = len(tok.encode(SHAM_SLOTS[tag]))
        ok = abs(sham_n - real_n) <= max(2, TOK_MATCH_TOL * real_n)
        ok_all = ok_all and ok
        sham_rows.append((tag, real_n, sham_n, ok))
    print("  §3 sham token match: " +
          " ".join(f"{t}:{r}/{s}{'' if k else '!'}"
                   for t, r, s, k in sham_rows) +
          ("  ALL WITHIN ±25%" if ok_all else "  ⚠ ADJUST FLAGGED SLOTS"))
    out["s3_sham_match"] = [
        {"tag": t, "real": r, "sham": s, "ok": k} for t, r, s, k in sham_rows]
    assert ok_all, "sham slots out of tolerance — reauthor before registering"
    rngr = np.random.default_rng(E7BQ_SEED + 3)
    while True:
        perm = list(rngr.permutation(8))
        if perm != list(range(8)) and perm[-1] != 7:
            break
    out["s3_reorder_perm"] = [int(x) for x in perm]
    print(f"      reorder arm negation order (of N1..N8): "
          f"{[f'N{i+1}' for i in perm]}")

    # §4 power — sweep SNR x R
    print("  §4 power (200 sims/cell, 400 perms in-sim, alpha .05):")
    power = {}
    for snr_name, snr in (("LOW", 0.4), ("MED", 0.8), ("HIGH", 1.4)):
        for R in (8, 10, 12, 16):
            hit1 = hit2 = 0
            n_sim = 200
            snr_i = {"LOW": 0, "MED": 1, "HIGH": 2}[snr_name]
            for s_i in range(n_sim):
                sd = E7BQ_SEED + 10000 + snr_i * 40000 + R * 1000 + s_i * 4
                gw, mw = _mk_world(R, T_WALK, snr * 0.35, 1.0, 0.35,
                                   sd, sham=False)
                gs, _ = _mk_world(R, T_WALK, 0.0, 1.0, 0.35, sd + 1,
                                  sham=True)
                _, p1 = pw1_test(gw, gs, 400, sd + 2)
                _, _, p2 = pw2_test(gw, mw, COUPLE_IDX, 400, sd + 3)
                hit1 += p1 < 0.05
                hit2 += p2 < 0.05
            power[(snr_name, R)] = (hit1 / n_sim, hit2 / n_sim)
            print(f"      SNR {snr_name:4s} R={R:2d}: P-W1 "
                  f"{hit1 / n_sim:.2f}  P-W2 {hit2 / n_sim:.2f}")
    out["s4_power"] = {f"{k[0]}_R{k[1]}": [round(v[0], 3), round(v[1], 3)]
                       for k, v in power.items()}
    # coupling sensitivity for P-W2 (kappa 1.0 above may be optimistic)
    for kap in (0.35, 0.5):
        hit = 0
        for s_i in range(200):
            sd = E7BQ_SEED + 90000 + int(kap * 100) * 300 + s_i
            gw, mw = _mk_world(16, T_WALK, 0.8 * 0.35, kap, 0.35, sd)
            _, _, p2 = pw2_test(gw, mw, COUPLE_IDX, 400, sd + 1)
            hit += p2 < 0.05
        print(f"      P-W2 coupling sensitivity: kappa={kap} R=16 MED -> "
              f"power {hit / 200:.2f}")
        out.setdefault("s4_kappa", {})[str(kap)] = round(hit / 200, 3)
    # HONEST PIN: no R in the sweep reaches .8 on P-W1 at the MED band —
    # P-W1 is a LARGE-EFFECT arm at this scale (E7-Q power-owned precedent);
    # P-W2 (the seed's own question) is well-powered at every band. R=16
    # takes the best available P-W1 power the budget affords.
    R_PIN = 16
    print(f"      PIN R = {R_PIN} — power OWNED: P-W1 "
          f"{power[('MED', 16)][0]:.2f} MED / {power[('HIGH', 16)][0]:.2f} "
          f"HIGH (large-effect arm); P-W2 ≥ "
          f"{min(v[1] for v in power.values()):.2f} everywhere")
    out["s4_R"] = R_PIN
    out["s4_power_owned"] = ("P-W1 underpowered at MED (.57) — registered "
                             "as a large-effect arm per E7-Q precedent; "
                             "P-W2 well-powered at all simulated bands")

    # §5 budget (real tokenizer, pinned script)
    script_tok = sum(ntok[s["tag"]] for s in script)
    mean_gen = 130          # censored mean reply (cap 200); corpus-informed
    arms = {"walked": T_WALK, "sham": T_WALK, "reorder": 11, "unwalked": 1}
    total_turns_per_cond = sum(v for v in arms.values()) * R_PIN
    gen_tokens = total_turns_per_cond * mean_gen
    # per-turn: generation + one capture forward of the full context
    for rate, label in ((25, "slow"), (38, "fast")):
        gen_s = gen_tokens / rate
        cap_s = total_turns_per_cond * 0.6
        oh_s = total_turns_per_cond * 0.5
        per_cond_min = (gen_s + cap_s + oh_s) / 60
        print(f"  §5 budget[{label} T4 {rate}tok/s]: "
              f"{total_turns_per_cond} turns/cond -> ~{per_cond_min:.0f} "
              f"min/cond, ~{2 * per_cond_min:.0f} min both + setup")
    out["s5_budget"] = {"turns_per_cond": total_turns_per_cond,
                        "arms": arms, "R": R_PIN,
                        "est_min_both_conds": [
                            round(2 * ((total_turns_per_cond * mean_gen / 25
                                        + total_turns_per_cond * 1.1) / 60)),
                            round(2 * ((total_turns_per_cond * mean_gen / 38
                                        + total_turns_per_cond * 1.1) / 60))]}

    # §6 payload
    payload = {
        "what": "E7b-Q pinned payload (design check -> builder -> VM assert)",
        "seed": E7BQ_SEED, "lam_gauge": LAM_GAUGE, "gen_cap": GEN_CAP,
        "R": R_PIN, "script_sha": script_sha,
        "script": [{"tag": s["tag"], "src": s["src"], "text": s["text"]}
                   for s in script],
        "sham_slots": SHAM_SLOTS,
        "reorder_perm": out["s3_reorder_perm"],
        "windows": {"base": BASE_WIN, "neg": NEG_IDX, "term": TERM_WIN,
                    "emerg": EMERG_IDX, "couple": COUPLE_IDX},
        "probes": probes,
        "probe_dirs": {c: {p: [round(float(x), 5) for x in
                               (np.asarray(dirs[c][p]) /
                                np.linalg.norm(dirs[c][p]))]
                           for p in probes}
                       for c in ("base14", "inst14", "inst20")},
        "being_text": f"BEING: {DESC['BEING']}",
        "being_vec": [float(x) for x in VEC["BEING"]],
        "xbar_pack": [round(float(x), 6) for x in xbar],
        # pre-build amendment (same session): S-CONE consumes the anchor-
        # cloud mean directions — embedded per condition, unit-normalized
        "cloud_mean": {c: [round(float(x), 5) for x in
                           (clouds[c].mean(0)
                            / np.linalg.norm(clouds[c].mean(0)))]
                       for c in clouds},
        "encoders": {c: [[round(float(x), 5) for x in row]
                         for row in enc[c]] for c in enc},
        "primary": {"condition": "real", "layer": 14, "gauge": gauge_pin,
                    "gauge_def": ("chat := [unit(h_pool - cent256); 1] @ W; "
                                  "D_BEING := ||chat - being_vec||; "
                                  "D_CENT := ||chat - xbar_pack||")},
        "a4_locked_tooth": {k: a4_lock[k] for k in a4_lock},
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    ppath = os.path.join(HERE, "e7bq_payload.json")
    J.jdump(payload, ppath)
    psha = hashlib.sha256(open(ppath, "rb").read()).hexdigest()[:16]
    out["s6_payload_sha"] = psha
    J.jdump(out, os.path.join(OUT_DIR, "design_check.json"))
    print(f"  §6 payload -> {ppath} (sha {psha}) | "
          f"design_check.json saved ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
