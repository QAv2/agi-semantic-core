"""W1 Amendment 1 check (no model calls): AT CHANCE and NO EXCESS must not depend on the reader's
confidence habits.

Registered rules use the graded pair score only. If the reader states the same confidence on
every read (e.g. always 70), the graded score is the binary score rescaled toward 1/2, and the
graded bars (0.65, 0.10) become far too lenient. Amendment 1 adds a habit-free condition on the
BINARY pair score (the share of a pair's two reads whose choice was right): AT CHANCE also needs
the bootstrap 97.5% quantile of the binary mean <= BB; NO EXCESS also needs the 97.5% quantile
of the binary difference <= MB. Significance (CARRIES, SELF EXCEEDS) stays on the graded score:
the exact sign-flip is valid whatever the habit.

Habits: calibrated (as the design check), always70, coarse (the calibrated value rounded to
the nearest 10, floor 60), extremes (50 or 100).
"""
import importlib.util
import pathlib

import numpy as np
from scipy.stats import expon, norm

spec = importlib.util.spec_from_file_location("dc", pathlib.Path(__file__).with_name("w1_design_check.py"))
dc = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc)

SEED = 20261027
HABITS = ("calibrated", "always70", "coarse", "extremes")


def reads(k, bias, rng, habit, noise=0.7):
    """-> (graded r_ab, r_ba, correct_ab, correct_ba) for pair evidence k (arrays)."""
    n = len(k)
    out = []
    for sgn in (+1, -1):                  # order AB (correct = 1) and BA (correct = 2)
        lg = k + sgn * bias               # logit of choosing the correct slot
        p_correct = dc._sig(lg)
        correct = rng.random(n) < p_correct
        conf_cal = dc._sig(np.abs(lg + rng.normal(0, noise, n)))
        if habit == "calibrated":
            c = np.round(conf_cal * 20) / 20
        elif habit == "always70":
            c = np.full(n, 0.70)
        elif habit == "coarse":
            c = np.maximum(0.6, np.round(conf_cal * 10) / 10)
        else:
            c = np.where(conf_cal > 0.75, 1.0, 0.5)
        c = np.clip(c, 0.5, 1.0)
        out.append((np.where(correct, c, 1 - c), correct.astype(float)))
    (g1, b1), (g2, b2) = out
    return (g1 + g2) / 2, (b1 + b2) / 2


def p1_table(rng, R=1000, n=16):
    print("P1 at n=16. CARRIES = graded exact sign-flip p <= .0033. AT CHANCE: R0 = graded q975 <= .65 (registered);"
          "\nR1(bb) = R0 and binary q975 <= bb (Amendment 1).")
    print("habit       mu    CARRIES  ATCH_R0  ATCH_R1(.70)  ATCH_R1(.75)")
    for habit in HABITS:
        for mu in dc.MUS:
            car = r0 = r170 = r175 = 0
            for _ in range(R):
                k = dc.KAPPA[mu] * rng.exponential(1.0, n)
                g, b = reads(k, rng.uniform(-1, 1), rng, habit)
                p = dc.signflip_p_cont(g)
                if p <= dc.ALPHA / 3:
                    car += 1
                    continue
                gq, bq = dc.boot_upper(g, rng), dc.boot_upper(b, rng)
                if gq <= 0.65:
                    r0 += 1
                    r170 += bq <= 0.70
                    r175 += bq <= 0.75
            print(f"{habit:10s} {mu:.2f}  {car/R:7.3f}  {r0/R:7.3f}  {r170/R:12.3f}  {r175/R:12.3f}")
        print()


def p2_table(rng, R=1000, n=16, rho=0.5):
    print("P2 at n=16, rho=.5. SELF EXCEEDS = graded sign-flip p <= .0033. NO EXCESS: R0 = graded d q975 <= .10;"
          "\nR1(mb) = R0 and binary d q975 <= mb.")
    print("habit       mu_s mu_o  EXCEEDS  NOEX_R0  NOEX_R1(.10)  NOEX_R1(.15)")
    for habit in HABITS:
        for ms, mo in ((0.5, 0.5), (0.7, 0.7), (0.8, 0.8), (0.8, 0.7), (0.8, 0.6), (0.9, 0.7), (0.7, 0.5)):
            ex = r0 = r110 = r115 = 0
            for _ in range(R):
                g1 = rng.normal(size=n); g2 = rho * g1 + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
                z1, z2 = expon.ppf(norm.cdf(g1)), expon.ppf(norm.cdf(g2))
                gs, bs = reads(dc.KAPPA[ms] * z1, rng.uniform(-1, 1), rng, habit)
                go, bo = reads(dc.KAPPA[mo] * z2, rng.uniform(-1, 1), rng, habit)
                d, db = gs - go, bs - bo
                if dc.signflip_p_cont(d + 0.5) <= dc.ALPHA / 3:
                    ex += 1
                    continue
                if dc.boot_upper(d + 0.5, rng) - 0.5 <= 0.10:
                    r0 += 1
                    q = dc.boot_upper(db + 0.5, rng) - 0.5
                    r110 += q <= 0.10
                    r115 += q <= 0.15
            print(f"{habit:10s} {ms:.2f} {mo:.2f}  {ex/R:7.3f}  {r0/R:7.3f}  {r110/R:12.3f}  {r115/R:12.3f}")
        print()


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    for mu in dc.MUS:
        dc.KAPPA[mu] = dc._calibrate(mu, rng)
    p1_table(rng)
    p2_table(rng)
