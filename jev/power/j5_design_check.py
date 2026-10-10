"""J5 design check (before registration; human data only, no Jev calls).

Answers the questions the J5 registration needs numbers for:
  1. the human yardstick: how closely does each of the 30 nations' color-feeling
     table match the other 29 (Table S6 leave-one-out panels)? -> the registered bars;
  2. the sharpness caps: what does a perfectly ranked but all-or-nothing instrument
     score, in the choice format and in the yes/no format?
  3. the culture test: is the obvious nation-rank test calibrated under instrument
     noise? Is the registered English-weight test?
  4. the C1 bar test at its boundary: false-positive rate and power.

Reads the human tables from ~/association-norms (licence: derived from CC-BY-NC-ND,
never committed; override the folder with J5_NORMS) and checks their sha256.
Writes only derived statistics: j5_design_check_output.txt (committed with the
registration). No human cell value is printed.
"""
import csv
import hashlib
import itertools
import os
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
NORMS = Path(os.environ.get("J5_NORMS", Path.home() / "association-norms"))
F_LONG = NORMS / "jonauskaite2020_tableS6_all_panels_long.csv"
SHA_LONG = "81cc73897eabd06631652b32fd36a341661d90ffff415c4fd79ac53d5040e92a"
OUT = Path(__file__).resolve().parent / "j5_design_check_output.txt"
ENGLISH = ("UK", "USA", "New Zealand")
MIXED = ("Nigeria",)            # surveyed in Igbo and English: left out of the culture test
W0 = 3 / 29                     # English share of the 29 culture-test nations

lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def load():
    blob = F_LONG.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == SHA_LONG, "all-panels file changed"
    rows = list(csv.DictReader(blob.decode().splitlines()))
    emotions, colors = [], []
    for r in rows:
        if r["panel"] == "Global":
            emotions += [r["emotion"]] if r["emotion"] not in emotions else []
            colors += [r["color"]] if r["color"] not in colors else []
    P = {}
    for r in rows:
        P.setdefault(r["panel"], np.zeros((len(colors), len(emotions))))[
            colors.index(r["color"]), emotions.index(r["emotion"])] = float(r["bayes_prob"])
    return P, colors, emotions


def dc(M):
    return M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()


def r_dc(D, H):
    return float(np.corrcoef(dc(D).ravel(), dc(H).ravel())[0, 1])


def rho_bar(D, H):
    return float(np.nanmean([spearmanr(D[:, j], H[:, j])[0] for j in range(D.shape[1])]))


def top1(D, H):
    """Hits over feelings; ties at the top share the hit (|A & B| / |A|)."""
    s = 0.0
    for j in range(D.shape[1]):
        A = set(np.flatnonzero(np.isclose(D[:, j], D[:, j].max())))
        B = set(np.flatnonzero(np.isclose(H[:, j], H[:, j].max())))
        s += len(A & B) / len(A)
    return s


def main():
    t0 = time.time()
    P, colors, emotions = load()
    G = P["Global"]
    nations = [p for p in P if p != "Global" and not p.startswith("Global-")]
    say("J5 design check — human data only, no Jev calls")
    say(f"inputs: {F_LONG.name} sha256 {SHA_LONG[:16]}… ({len(P)} panels, "
        f"{len(colors)} colors x {len(emotions)} feelings)")

    # 1. the yardstick
    say("\n1. HUMAN YARDSTICK: each nation against the other 29 (leave-one-out panel)")
    rows = sorted(((n, r_dc(P[n], P["Global-" + n]), rho_bar(P[n], P["Global-" + n]),
                    top1(P[n], P["Global-" + n])) for n in nations), key=lambda x: x[1])
    for n, r, rho, h in rows:
        say(f"   {n:<14} r_dc {r:.3f}   rho {rho:.3f}   top-1 {h:5.2f}/20")
    a = np.array([x[1:] for x in rows])
    for lab, f in (("min", np.min), ("10th pct", lambda v, axis: np.percentile(v, 10, axis=axis)),
                   ("median", np.median), ("max", np.max)):
        v = f(a, axis=0)
        say(f"   {lab:<9} r_dc {v[0]:.3f}   rho {v[1]:.3f}   top-1 {v[2]:.2f}")
    say(f"   -> registered bars: floor = least typical nation's r_dc {a[:, 0].min():.3f} "
        f"({rows[0][0]}); median nation {np.median(a[:, 0]):.3f}")

    # 2. sharpness caps
    say("\n2. SHARPNESS CAPS: a perfectly ranked instrument, made all-or-nothing")
    say("   choice format (p ∝ human^k per feeling, rounded to 0.01):")
    for k in (1, 2, 4, 8, 16, 64):
        D = G ** k
        D = np.round(D / D.sum(0, keepdims=True), 2)
        say(f"     k={k:<3} r_dc {r_dc(D, G):.3f}  rho {rho_bar(D, G):.3f}  nonzero colors/feeling {(D > 0).sum(0).mean():.1f}")
    D = np.zeros_like(G)
    D[G.argmax(0), range(G.shape[1])] = 1
    say(f"     argmax only: r_dc {r_dc(D, G):.3f}  rho {rho_bar(D, G):.3f}   <- below the floor")
    say("   yes/no format, binary (yes iff the human cell > t):")
    for t in (0.10, 0.15, 0.20, 0.25, 0.30):
        D = (G > t).astype(float)
        say(f"     t={t:.2f} yes share {D.mean():.2f}  r_dc {r_dc(D, G):.3f}  rho {rho_bar(D, G):.3f}")
    say("   yes/no format, steep logistic around 0.15, rounded to 0.01:")
    for s in (10, 20, 40):
        D = np.round(1 / (1 + np.exp(-s * (G - 0.15))), 2)
        say(f"     slope {s:<3} r_dc {r_dc(D, G):.3f}")

    # 3. culture tests
    cn = [n for n in nations if n not in MIXED]
    rest = [n for n in cn if n not in ENGLISH]
    E_ = np.mean([P[n] for n in ENGLISH], 0)
    R_ = np.mean([P[n] for n in rest], 0)
    W_ = np.mean([P[n] for n in cn], 0)
    say(f"\n3. CULTURE: English-surveyed {ENGLISH} vs the other {len(rest)} ({MIXED[0]} left out as mixed)")
    say(f"   r_dc(English mean, rest-of-world mean) = {r_dc(E_, R_):.3f}")
    y = np.array([r_dc(P[n], P["Global-" + n]) for n in cn])
    combos = list(itertools.combinations(range(len(cn)), 3))
    eng = [cn.index(e) for e in ENGLISH]
    A = np.vstack([np.ones_like(y), y]).T

    def rank_test(J):
        aff = np.array([r_dc(J, P[n]) for n in cn])
        e = aff - A @ np.linalg.lstsq(A, aff, rcond=None)[0]
        tot = e.sum()
        con = lambda ix: e[list(ix)].mean() - (tot - e[list(ix)].sum()) / (len(e) - 3)
        obs = con(eng)
        return float(np.mean([con(c) >= obs - 1e-12 for c in combos]))

    say("   3a. the obvious test (rank the English nations' residual affinity among all 3-sets):")
    for sd in (0.02, 0.05, 0.10):
        ps = [rank_test(G + np.random.default_rng(100 + s).normal(0, sd, G.shape)) for s in range(200)]
        say(f"     null = global table + noise sd {sd:.2f}: false-positive rate at .01 = {np.mean(np.array(ps) <= .01):.3f}"
            f"  (at .05 = {np.mean(np.array(ps) <= .05):.3f})")
    say("     -> anti-conservative: the three English panels resemble each other, so noise that lifts one")
    say("        lifts all three. Not used for inference (reported descriptively only).")

    def weight(J, cols):
        j, e, r = (dc(M[:, cols]).ravel() for M in (J, E_, R_))
        a_, b_ = np.linalg.lstsq(np.vstack([e, r]).T, j, rcond=None)[0]
        return a_ / (a_ + b_)

    def wtest(J, B, seed):
        rg = np.random.default_rng(seed)
        bs = np.array([weight(J, rg.integers(0, 20, 20)) for _ in range(B)])
        return weight(J, np.arange(20)), (1 + np.sum(bs <= W0)) / (1 + B)

    say(f"   3b. the registered test: English weight w = a/(a+b) in dc(J) ≈ a·dc(English) + b·dc(rest);")
    say(f"       null w = {W0:.3f} (each nation counted once); feeling bootstrap")
    for lab, J in (("global table", G), ("world, each nation once", W_), ("rest of world", R_),
                   ("English mean", E_), ("UK panel", P["UK"]), ("mix 0.25 English", .25 * E_ + .75 * R_),
                   ("binary world (t=.2)", (W_ > .2).astype(float)),
                   ("binary mix 0.5", ((.5 * E_ + .5 * R_) > .2).astype(float))):
        w, p = wtest(J, 2000, 7)
        say(f"     {lab:<24} w {w:+.3f}  p(w <= {W0:.3f}) {p:.4f}")
    for sd in (0.03, 0.06, 0.10):
        res = [wtest(W_ + np.random.default_rng(900 + s).normal(0, sd, G.shape), 400, s) for s in range(120)]
        say(f"     null = world + noise sd {sd:.2f}: false-positive rate at .01 = {np.mean([p <= .01 for _, p in res]):.3f}"
            f"  (w mean {np.mean([w for w, _ in res]):.3f}, sd {np.std([w for w, _ in res]):.3f})")
    for mix in (0.25, 0.5):
        res = [wtest(mix * E_ + (1 - mix) * R_ + np.random.default_rng(1900 + s).normal(0, 0.06, G.shape), 400, s)
               for s in range(60)]
        say(f"     power, true w {mix:.2f}, noise sd 0.06: {np.mean([p <= .01 for _, p in res]):.3f}")

    # 4. the C1 bar test at its boundary
    floor = a[:, 0].min()
    say(f"\n4. C1 BAR TEST (feeling bootstrap, one-sided, H0: r_dc <= {floor:.3f})")

    def c1(J, B, seed):
        rg = np.random.default_rng(seed)
        bs = []
        for _ in range(B):
            c = rg.integers(0, 20, 20)
            bs.append(r_dc(J[:, c], G[:, c]))
        return r_dc(J, G), (1 + np.sum(np.array(bs) <= floor)) / (1 + B)

    def sd_for(target):
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2
            m = np.mean([r_dc(G + np.random.default_rng(5000 + s).normal(0, mid, G.shape), G) for s in range(80)])
            lo, hi = (mid, hi) if m > target else (lo, mid)
        return (lo + hi) / 2

    for target in (floor, 0.75, 0.80):
        sd = sd_for(target)
        res = [c1(G + np.random.default_rng(7000 + s).normal(0, sd, G.shape), 1000, s) for s in range(120)]
        rate = np.mean([p <= .01 for _, p in res])
        tag = "false-positive rate" if target == floor else "power"
        say(f"   true r_dc ≈ {target:.3f} (noise sd {sd:.3f}): mean observed {np.mean([r for r, _ in res]):.3f}, "
            f"{tag} at .01 = {rate:.3f}")
    say(f"\n(done in {time.time() - t0:.0f} s)")
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
