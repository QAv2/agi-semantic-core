"""J5 amendment 1 check: C2 on Jev's own scale (curve-matched) against C2 as first
registered (raw), on the real analysis code (jev.analyze_j5), human data only, no calls.

Why: the verdict suite showed that a nonlinear response curve (a planted agent whose
yes/no answers are a logistic of the human association) leaves the raw English
weight biased and its CI too wide to reach NO LEAN or LEANS AWAY. Jev's answers are
expected to be steep (§2.2). The amendment passes the English and rest-of-world
tables through Jev's own response curve (an isotonic map fitted from the world table
to Jev's table) before the mix is fit.

For each scenario, 60 seeded noisy tables; each table gets the one-sided test
(B = 300) and the registered branch rules (single test at alpha = 0.01, no Holm).
Output: j5_c2_curve_check_output.txt (derived statistics only).
"""
import time
import warnings
from pathlib import Path

import numpy as np

from jev import analyze_j5 as Z

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "j5_c2_curve_check_output.txt"
SEEDS, B = 60, 300
BRANCHES = ("LEANS ENGLISH", "LEANS AWAY", "NO LEAN", "UNCLEAR")


def main():
    t0 = time.time()
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    hum = Z.load_human()
    E_, R_, n_rest, W_ = Z.english_rest(hum["panels"])
    G = hum["global"]
    logistic = lambda M: 1 / (1 + np.exp(-20 * (M - 0.15)))
    scen = [
        ("logistic: world (null, w 0.103)", lambda r: logistic(W_) + r.normal(0, 0.03, G.shape)),
        ("logistic: global table (w ≈ 0.09)", lambda r: logistic(G) + r.normal(0, 0.03, G.shape)),
        ("logistic: rest of world (w 0)", lambda r: logistic(R_) + r.normal(0, 0.03, G.shape)),
        ("logistic: mix 0.25 English", lambda r: logistic(.25 * E_ + .75 * R_) + r.normal(0, 0.03, G.shape)),
        ("logistic: mix 0.50 English", lambda r: logistic(.5 * E_ + .5 * R_) + r.normal(0, 0.03, G.shape)),
        ("logistic: world, noise 0.06", lambda r: logistic(W_) + r.normal(0, 0.06, G.shape)),
        ("binary: world (null)", lambda r: ((W_ + r.normal(0, 0.03, G.shape)) > 0.2).astype(float)),
        ("binary: mix 0.50 English", lambda r: ((.5 * E_ + .5 * R_ + r.normal(0, 0.03, G.shape)) > 0.2).astype(float)),
        ("linear: world, noise 0.06", lambda r: W_ + r.normal(0, 0.06, G.shape)),
    ]
    say("J5 amendment 1 check: C2 curve-matched vs raw (as first registered), on jev.analyze_j5")
    say(f"{SEEDS} seeded tables per scenario, B = {B}; branch counts, then mean w and how often p <= .01")
    say(f"{'scenario':<38} {'mode':<6} " + " ".join(f"{b[:13]:>13}" for b in BRANCHES) + "   mean w   p<=.01")
    for name, make in scen:
        for matched in (True, False):
            counts = dict.fromkeys(BRANCHES, 0)
            ws, ps = [], []
            for s in range(SEEDS):
                J = np.round(np.clip(make(np.random.default_rng([4242, s])), 0, 1), 2)
                r = Z.c2(J, E_, R_, W_, B, matched=matched, tag=f"chk{s}")
                b = Z.c2_branch(r, r["p"] <= Z.ALPHA)
                b = "UNCLEAR" if b.startswith("UNCLEAR") or b.startswith("NOT") else b
                counts[b] += 1
                if r.get("w") is not None:
                    ws.append(r["w"])
                ps.append(r["p"] <= Z.ALPHA)
            say(f"{name:<38} {'curve' if matched else 'raw':<6} " + " ".join(f"{counts[b]:>13}" for b in BRANCHES)
                + f"   {np.mean(ws):+.3f}   {np.mean(ps):.3f}")
    say(f"\n(done in {time.time() - t0:.0f} s)")
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
