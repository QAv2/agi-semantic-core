#!/usr/bin/env python3
"""E8-O (L4 / Ifá) design check — lane law: BEFORE registration, BEFORE build.

Unparked by E8-F fork FF1 (L3 landed: the 14 register axes are a readable
compositional alphabet). L4 asks the ORDER question — non-commutative
composition, the Ifá ordered-pair grammar as the design prior
(docs/IFA_SEED.md §2.3/§3(c)); "red square dog" as the problem statement.

ALL LOCAL, zero flights, standing instruments only:
  pack coords + DB trigram · the locked E8-J atlas (instrument teeth) ·
  the E8-F payload's pinned W_L3 + tercile code (e8f_logic literals).

Measured questions:
  §2  tetragram census — the 16-cell sign(e,f,g,h) compass vs the lossy
      trigram (IFA_SEED §2.1): occupancy, balance, within-cell coherence.
  §3  méjì census — essence-leg vs function-leg figure agreement; 256-cell
      full-Odù occupancy; the unopened mixed-figure space (§2.2).
  §4  heptagram census — 128-cell per-leg occupancy (§7.6 item 5).
  §5  Hamming [7,4,3] coherence census (§7.3, §7.6 items 6-7) — generator
      PINNED BEFORE the census is read (item 7): domain nibble = data,
      core trigram = parity. Perfect-code properties verified in-code;
      syndrome (= moving line = core-domain mismatch) distribution.
  §6  THE L4 FLIGHT-DECIDER — the ordered-composition operator:
      OP-A (Ifá-native, leg-swap): compose(A,B) = (essence(A), function(B))
        — exact code semantics (composed tercile code = A's leg-1 fields +
        B's leg-2 fields BY CONSTRUCTION), casts mixed Odù synthetically.
      OP-B (weighted-senior, w=0.7): comparison row.
      Measured per operator over ordered eval-64 pairs (held-out split):
      order-separation angle through the pinned W (the channel must see
      (A,B) vs (B,A) apart), parent-separation, code-level order yield,
      composed-code mixed-figure rate, decode-margin texture.
  §7  Odù-profile-arm instruments (IFA_SEED §4.3, run-later-on-dirs):
      span-residual + reverse-map-gain machinery built and TOOTHED now
      (planted in-span world nulls out; planted off-span world detected;
      reverse-gain detects a planted extra axis) so the 16-profile rider
      on the L4 flight lands on proven instruments.

Run:  python3 colab/e8o_design_check.py
"""
import json, os, sqlite3, sys, time
from itertools import product

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e8f_logic as F
import e8j_logic as J

OUT_DIR = os.path.join(HERE, "results_e8o")
DB = os.path.join(HERE, "..", "db", "semantic.db")
RES_J = os.path.join(HERE, "results_e8j", "full_20260824_1827")

E8O_SEED = 20260900            # fresh stream, disjoint from 2026082x/8x
W_SENIOR = 0.7                 # OP-B pinned candidate weight
CHANNEL_RES_DEG = 20.0         # demonstrated readout resolution scale
                               # (R3-c 18.5deg; E8-R2 28.5deg; E8-F reads the
                               # span at .70 with atoms >=44.9deg apart)

# ── §5 pinned generator (REGISTERED CHOICE, item 7 — before any census) ─────
# Leg bit order: (e, f, g, h | x, y, z) = (d1 d2 d3 d4 | p1 p2 p3).
# Standard [7,4,3]: p1=d1^d2^d4, p2=d1^d3^d4, p3=d2^d3^d4.
# bit(v) = 1 if v > 0 else 0  (exact zeros -> 0; zero-rate reported in §2).
HAMMING_P = [(0, 1, 3), (0, 2, 3), (1, 2, 3)]   # parity supports over d1..d4

ESS = list(range(0, 7))        # x y z e f g h
FUN = list(range(7, 14))       # fx fy fz fe ff fg fh
DOM_E = [3, 4, 5, 6]           # e f g h
DOM_F = [10, 11, 12, 13]       # fe ff fg fh
CORE_E = [0, 1, 2]
CORE_F = [7, 8, 9]


def bits(vals):
    return tuple(1 if v > 0 else 0 for v in vals)


def tetra(vec, leg="e"):
    idx = DOM_E if leg == "e" else DOM_F
    return bits([vec[j] for j in idx])


def hepta(vec, leg="e"):
    idx = ([3, 4, 5, 6, 0, 1, 2] if leg == "e" else [10, 11, 12, 13, 7, 8, 9])
    return bits([vec[j] for j in idx])         # (d1..d4, p1..p3)


def syndrome(h7):
    d, p = h7[:4], h7[4:]
    return tuple((sum(d[i] for i in sup) + p[k]) % 2
                 for k, sup in enumerate(HAMMING_P))


def compose_A(vA, vB):
    """OP-A, Ifá-native leg-swap: essence leg of A + function leg of B."""
    out = list(vA[:7]) + list(vB[7:])
    return out


def compose_B(vA, vB, w=W_SENIOR):
    """OP-B, weighted-senior additive (senior = A)."""
    return [w * a + (1 - w) * b for a, b in zip(vA, vB)]


def pred(W5, x):
    v = np.asarray(x, float) @ np.asarray(W5, float)[:14] + np.asarray(W5, float)[14]
    return v / np.linalg.norm(v)


def ang(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1, 1))))


# ── §7 instruments: span-residual + reverse-map gain ────────────────────────
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


def self_tests(W5, Ya):
    rng = np.random.default_rng(E8O_SEED)
    # Hamming perfect-code properties from the pinned generator
    words = []
    for d in product((0, 1), repeat=4):
        p = tuple(sum(d[i] for i in sup) % 2 for sup in HAMMING_P)
        words.append(d + p)
    assert len(set(words)) == 16
    assert all(syndrome(w) == (0, 0, 0) for w in words)
    dmin = min(sum(a != b for a, b in zip(u, v))
               for u in words for v in words if u != v)
    assert dmin == 3, dmin
    balls = set()
    for w in words:
        balls.add(w)
        for j in range(7):
            flip = list(w)
            flip[j] ^= 1
            balls.add(tuple(flip))
    assert len(balls) == 128, "radius-1 balls do not tile the 7-cube"
    syn_map = {}
    for w in words:
        for j in range(7):
            flip = list(w)
            flip[j] ^= 1
            s = syndrome(tuple(flip))
            assert s != (0, 0, 0)
            syn_map.setdefault(s, set()).add(j)
    assert all(len(v) == 1 for v in syn_map.values()), "syndrome!=unique line"
    # operators
    v = list(rng.normal(size=14))
    assert compose_A(v, v) == v
    a, b = list(rng.normal(size=14)), list(rng.normal(size=14))
    assert compose_A(a, b)[:7] == a[:7] and compose_A(a, b)[7:] == b[7:]
    assert compose_A(a, b) != compose_A(b, a)
    ca = F.code_levels(compose_A(a, b))
    assert ca[:7] == F.code_levels(a)[:7] and ca[7:] == F.code_levels(b)[7:]
    # span-residual teeth on the real atlas basis
    basis = Ya[:100]
    in_span = np.asarray([rng.normal(size=100) @ basis for _ in range(8)])
    r_in = span_residual(in_span, basis)
    off = rng.normal(size=(8, Ya.shape[1]))
    r_off = span_residual(off, basis)
    assert r_in.max() < 1e-9, r_in.max()
    assert r_off.min() > 0.5, r_off.min()
    # inversion-structure RSA teeth: a structure-respecting synthetic world
    # separates from a scrambled one
    figs, S = odu_design_matrix()
    h = 96
    axes4 = rng.normal(size=(4, h))
    d16 = np.array([sum((1 if bit else -1) * axes4[k] for k, bit in
                        enumerate(f)) for f in figs])
    d16 += rng.normal(0, 0.4, d16.shape)
    obs, p = odu_structure_rsa(d16, n_perm=500, seed=E8O_SEED + 8)
    d16s = d16[rng.permutation(16)]
    obs_s, p_s = odu_structure_rsa(d16s, n_perm=500, seed=E8O_SEED + 9)
    assert p < 0.02 and obs > 0.4, (obs, p)
    assert p_s > 0.05 or obs_s < obs / 2, (obs_s, p_s)
    print("  §0 self-tests + teeth: PASS (perfect [7,4,3] from the pinned "
          "generator; operators; span-residual; inversion-RSA tooth "
          f"{obs:.2f} p{p:.3f} vs scrambled {obs_s:.2f} p{p_s:.3f}; "
          "NOTE ON RECORD: the seed's reverse-map-gain sketch is dropped "
          "as linear-no-op/target-leaky — a true register-expansion test "
          "needs independently authored Odù coordinates)")


def main():
    t0 = time.time()
    print("E8-O (L4/Ifá) DESIGN CHECK — censuses + the order operator")
    print("=" * 66)
    pack = json.load(open(os.path.join(HERE, "e4_dictionary_pack.json")))
    VEC = {c["name"]: c["vec"] for c in pack["concepts"]}
    payload = json.load(open(os.path.join(HERE, "e8f_payload.json")))
    W5 = np.array(payload["W"], float)
    dirs = json.load(open(os.path.join(RES_J, "atlas_dirs.json")))
    Ya = np.array([dirs["inst14"][a] for a in dirs["anchors"]], float)
    Ya = Ya / np.linalg.norm(Ya, axis=1, keepdims=True)
    self_tests(W5, Ya)
    out = {"what": "E8-O design check", "seed": E8O_SEED,
           "generator": {"parity_supports": HAMMING_P,
                         "bit_rule": "v>0", "leg_order": "d=(e f g h), p=(x y z)"}}

    frame = F.frame_names(pack["concepts"])
    XF = {n: VEC[n] for n in frame}
    con = sqlite3.connect(DB)
    tri_db = dict(con.execute("SELECT name, trigram FROM concepts").fetchall())
    con.close()
    print(f"  §1 loaded: frame {len(frame)}, pack {len(VEC)}, W_L3 pinned "
          f"(payload {F.PAYLOAD_SHA}), atlas anchors {len(Ya)}")
    zero_rate = float(np.mean([1 for n in frame for j in range(14)
                               if XF[n][j] == 0.0])) if False else \
        sum(1 for n in frame for j in range(14) if XF[n][j] == 0.0) / (len(frame) * 14)
    out["zero_coord_rate"] = round(zero_rate, 5)

    # ── §2 tetragram census ─────────────────────────────────────────────────
    occ_e, occ_tri = {}, {}
    for n in frame:
        occ_e.setdefault(tetra(XF[n], "e"), []).append(n)
        occ_tri.setdefault(tri_db.get(n) or "_", []).append(n)
    sizes = sorted((len(v) for v in occ_e.values()), reverse=True)
    def entro(cnts, total):
        p = np.array(cnts, float) / total
        return float(-(p * np.log2(p)).sum())
    Xarr = {n: np.asarray(XF[n], float) for n in frame}
    coh_in, coh_x = [], []
    rngc = np.random.default_rng(E8O_SEED + 2)
    cells = [v for v in occ_e.values() if len(v) >= 8]
    for cell in cells:
        pick = rngc.choice(cell, size=min(30, len(cell)), replace=False)
        for i in range(0, len(pick) - 1, 2):
            a, b = Xarr[pick[i]], Xarr[pick[i + 1]]
            coh_in.append(ang(a / np.linalg.norm(a), b / np.linalg.norm(b)))
    allf = list(frame)
    for _ in range(len(coh_in)):
        a, b = rngc.choice(allf, size=2, replace=False)
        if tetra(XF[a], "e") == tetra(XF[b], "e"):
            continue
        va, vb = Xarr[a], Xarr[b]
        coh_x.append(ang(va / np.linalg.norm(va), vb / np.linalg.norm(vb)))
    print(f"  §2 tetragram (essence): {len(occ_e)}/16 cells occupied | "
          f"sizes max {sizes[0]} min {sizes[-1]} | entropy "
          f"{entro(sizes, len(frame)):.2f}/{np.log2(16):.2f} bits "
          f"(trigram {entro([len(v) for v in occ_tri.values()], len(frame)):.2f}"
          f"/{np.log2(len(occ_tri)):.2f}) | within-cell 14D angle med "
          f"{np.median(coh_in):.1f} vs cross {np.median(coh_x):.1f}")
    dom_fig = max(occ_e, key=lambda k: len(occ_e[k]))
    print(f"      dominant figure: {dom_fig} (all-yang = Ogbe-form)"
          if dom_fig == (1, 1, 1, 1) else f"      dominant figure: {dom_fig}")
    out["s2_tetragram"] = {
        "dominant_figure": list(dom_fig),
        "cells_occupied": len(occ_e), "sizes": sizes,
        "entropy_bits": round(entro(sizes, len(frame)), 3),
        "trigram_entropy_bits": round(
            entro([len(v) for v in occ_tri.values()], len(frame)), 3),
        "within_cell_med": round(float(np.median(coh_in)), 1),
        "cross_cell_med": round(float(np.median(coh_x)), 1)}

    # ── §3 méjì + mixed-Odù census ──────────────────────────────────────────
    meji = sum(1 for n in frame if tetra(XF[n], "e") == tetra(XF[n], "f"))
    odu = {}
    for n in frame:
        odu.setdefault((tetra(XF[n], "e"), tetra(XF[n], "f")), []).append(n)
    mixed_occ = sum(1 for k in odu if k[0] != k[1])
    print(f"  §3 méjì: {meji}/{len(frame)} ({meji/len(frame):.1%}) doubled "
          f"figures | full-Odù cells occupied {len(odu)}/256 "
          f"(mixed {mixed_occ}/240) | largest mixed cell "
          f"{max((len(v) for k, v in odu.items() if k[0] != k[1]), default=0)}")
    out["s3_meji"] = {"meji": meji, "n": len(frame),
                      "meji_rate": round(meji / len(frame), 4),
                      "odu_cells": len(odu), "mixed_cells": mixed_occ}

    # ── §4 heptagram census ─────────────────────────────────────────────────
    hep_e = {}
    for n in frame:
        hep_e.setdefault(hepta(XF[n], "e"), 0)
        hep_e[hepta(XF[n], "e")] += 1
    print(f"  §4 heptagram (essence leg): {len(hep_e)}/128 cells occupied | "
          f"top cell {max(hep_e.values())} | singletons "
          f"{sum(1 for v in hep_e.values() if v == 1)}")
    out["s4_heptagram"] = {"cells": len(hep_e), "top": max(hep_e.values()),
                           "singletons": sum(1 for v in hep_e.values() if v == 1)}

    # ── §5 Hamming coherence census (generator pinned in §0) ────────────────
    syn_e = {}
    for n in frame:
        s = syndrome(hepta(XF[n], "e"))
        syn_e[s] = syn_e.get(s, 0) + 1
    cw = syn_e.get((0, 0, 0), 0)
    line_names = {(0,0,0): 'still'}
    from itertools import product as _pr
    for j in range(7):
        probe = [0]*7; probe[j] = 1
        line_names[syndrome(tuple(probe))] = f'line-{j+1} ' +             ['e','f','g','h','x','y','z'][j]
    dom_syn = max((k for k in syn_e if k != (0,0,0)), key=lambda k: syn_e[k])
    print(f"  §5 Hamming census (essence): codeword-coherent {cw}/{len(frame)} "
          f"({cw/len(frame):.1%} vs 12.5% chance) | dominant moving line "
          f"{line_names.get(dom_syn)} ({syn_e[dom_syn]}) | spread "
          f"{sorted(syn_e.values(), reverse=True)}")
    out["s5_dominant_line"] = line_names.get(dom_syn)
    out["s5_hamming"] = {"codeword": cw, "rate": round(cw / len(frame), 4),
                         "chance": 0.125,
                         "syndromes": {str(k): v for k, v in sorted(syn_e.items())}}

    # ── §6 THE FLIGHT-DECIDER: ordered-composition operators ────────────────
    ev = sorted(F.EVAL64)
    pairs = [(a, b) for a in ev for b in ev if a != b]
    rngp = np.random.default_rng(E8O_SEED + 3)
    idx = rngp.choice(len(pairs), size=400, replace=False)
    sample = [pairs[int(i)] for i in idx]
    rows = {"A": {"sep": [], "par": [], "code_ham": [], "mixed": 0,
                  "nearest_flip": 0},
            "B": {"sep": [], "par": [], "code_ham": [], "mixed": 0,
                  "nearest_flip": 0}}
    FR = F.frame_codes(pack["concepts"])
    names_all = sorted(FR)
    Fm = np.array([FR[n] for n in names_all], int)

    def nearest_name(levels):
        d = (Fm != np.array(levels, int)).sum(axis=1)
        best = np.where(d == d.min())[0]
        return names_all[best[0]] if len(best) == 1 else None

    for a, b in sample:
        vA, vB = VEC[a], VEC[b]
        for op, fn in (("A", compose_A), ("B", compose_B)):
            xab, xba = fn(vA, vB), fn(vB, vA)
            pab, pba = pred(W5, xab), pred(W5, xba)
            rows[op]["sep"].append(ang(pab, pba))
            rows[op]["par"].append(min(ang(pab, pred(W5, vA)),
                                       ang(pab, pred(W5, vB))))
            cab, cba = F.code_levels(xab), F.code_levels(xba)
            rows[op]["code_ham"].append(sum(1 for u, v in zip(cab, cba)
                                            if u != v))
            if tetra(xab, "e") != tetra(xab, "f"):
                rows[op]["mixed"] += 1
            na, nb = nearest_name(cab), nearest_name(cba)
            if na is not None and nb is not None and na != nb:
                rows[op]["nearest_flip"] += 1
    for op in ("A", "B"):
        r = rows[op]
        sep = np.array(r["sep"])
        print(f"  §6 OP-{op}: order-sep med {np.median(sep):.1f}deg "
              f"(p10 {np.percentile(sep,10):.1f}) | >= {CHANNEL_RES_DEG}deg for "
              f"{float(np.mean(sep >= CHANNEL_RES_DEG)):.0%} | parent-sep med "
              f"{np.median(r['par']):.1f} | code-Hamming(AB,BA) med "
              f"{np.median(r['code_ham']):.0f} | casts MIXED figures "
              f"{r['mixed']}/400 | nearest-name flips on order "
              f"{r['nearest_flip']}/400")
        out[f"s6_op{op}"] = {
            "sep_med": round(float(np.median(sep)), 2),
            "sep_p10": round(float(np.percentile(sep, 10)), 2),
            "frac_ge_channel": round(float(np.mean(sep >= CHANNEL_RES_DEG)), 3),
            "parent_med": round(float(np.median(r["par"])), 2),
            "code_ham_med": float(np.median(r["code_ham"])),
            "mixed_400": r["mixed"], "nearest_flip_400": r["nearest_flip"]}

    # composed-point envelope: distance to the E8-F stimulus world
    atlas_names = dirs["anchors"]
    P_full = np.array([pred(W5, VEC[n]) for n in atlas_names])
    d_env = []
    for a, b in sample[:100]:
        pab = pred(W5, compose_A(VEC[a], VEC[b]))
        d_env.append(float(np.degrees(np.arccos(
            np.clip(P_full @ pab, -1, 1))).min()))
    print(f"  §6b envelope: OP-A composed preds sit med "
          f"{np.median(d_env):.1f}deg from the nearest full-concept pred "
          f"(the readable span E8-F reads at .70)")
    out["s6b_envelope_med"] = round(float(np.median(d_env)), 2)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "design_check.json")
    J.jdump(out, path)
    print(f"  saved {path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
