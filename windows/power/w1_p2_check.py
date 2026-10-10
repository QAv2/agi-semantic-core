"""W1 design check, part 2 (no model calls): the self-vs-observer comparison (P2).

d_i = s_self,i - s_obs,i on the graded pair score, same item, same history. Exact sign-flip on d.
SELF EXCEEDS: Holm(3) one-sided p <= .01. NO EXCESS: not significant and the bootstrap 97.5%
quantile of mean d <= MARGIN. Pair evidence in the two arms is correlated (the same history
feeds both), z_obs = rho * z_self + sqrt(1 - rho^2) * fresh, mapped to Exp(1) by rank.
"""
import numpy as np
from scipy.stats import norm, expon

import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("dc", pathlib.Path(__file__).with_name("w1_design_check.py"))
dc = importlib.util.module_from_spec(spec); spec.loader.exec_module(dc)

SEED = 20261023


def pair_arm_scores(n, mu_s, mu_o, rng, rho=0.5, noise=0.7):
    g1 = rng.normal(size=n)
    g2 = rho * g1 + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    z1, z2 = expon.ppf(norm.cdf(g1)), expon.ppf(norm.cdf(g2))
    out = []
    for kap, z in ((dc.KAPPA[mu_s], z1), (dc.KAPPA[mu_o], z2)):
        b = rng.uniform(-1, 1)
        k = kap * z
        r_ab = np.round(dc._sig(k + b + rng.normal(0, noise, n)) * 20) / 20
        r_ba = np.round(dc._sig(k - b + rng.normal(0, noise, n)) * 20) / 20
        out.append((r_ab + r_ba) / 2)
    return out


def main():
    rng = np.random.default_rng(SEED)
    for mu in dc.MUS:
        dc.KAPPA[mu] = dc._calibrate(mu, rng)
    R, n = 1000, 16
    margins = (0.05, 0.075, 0.10)
    print("W1 P2 check, n=16 pairs, graded scores, rho=.5 between arms")
    print("mu_self mu_obs  mean_d  P(SELF EXCEEDS p<=.0033)  " + "  ".join(f"P(NO EXCESS|m={m})" for m in margins))
    for mu_s, mu_o in ((0.5, 0.5), (0.7, 0.7), (0.8, 0.8), (0.9, 0.9), (0.8, 0.7), (0.8, 0.65),
                       (0.8, 0.6), (0.9, 0.7), (0.9, 0.6), (0.7, 0.5), (0.8, 0.5), (0.7, 0.8)):
        ex = 0; ne = np.zeros(len(margins)); md = []
        for _ in range(R):
            s, o = pair_arm_scores(n, mu_s, mu_o, rng)
            d = s - o
            md.append(d.mean())
            p = dc.signflip_p_cont(d + 0.5)
            ub = dc.boot_upper(d + 0.5, rng) - 0.5
            if p <= dc.ALPHA / 3:
                ex += 1
            else:
                for j, m in enumerate(margins):
                    ne[j] += ub <= m
        print(f"{mu_s:7.2f} {mu_o:6.2f}  {np.mean(md):+.3f}  {ex/R:24.3f}  " + "  ".join(f"{x/R:17.3f}" for x in ne))


if __name__ == "__main__":
    main()
