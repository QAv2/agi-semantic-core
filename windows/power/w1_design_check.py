"""W1 design check (no model calls): power and calibration of the paired blind-reader test.

Unit = an item-matched pair (condition X image, condition Y image). Each pair is read twice
(both image orders). Pair score s = mean correct over the two orders, in {0, .5, 1}.
Test: exact one-sided sign-flip on (s - .5) over pairs (exchangeable labels under H0).
Verdict per contrast: CARRIES if Holm(3 contrasts, alpha .01) passes; AT CHANCE if not
significant and the bootstrap 95% upper bound of mean s is <= BAR_AT_CHANCE; else UNCLEAR.

Reader model in simulation: see sim_pair_scores (per-pair evidence + position bias, both
orders read with the same pair evidence). Position bias is drawn per run from U(-1, 1) on the
logit scale (up to a 73/27 lean toward "Image 1" when the reader has no evidence).
"""
from math import comb
import numpy as np

SEED = 20261020
BAR_AT_CHANCE = 0.75   # binary-score table only; the graded bar is calibrated in bar_table()
ALPHA = 0.01


def signflip_p(s):
    """Exact one-sided sign-flip p for pair scores in {0, .5, 1}.

    d = s - .5 is in {-.5, 0, +.5}; ties (d = 0) carry no sign, so the sign-flip null is
    Binomial(m, 1/2) on the m untied pairs, and p = P(Bin(m, .5) >= #positive)."""
    d = np.asarray(s) - 0.5
    pos = int(np.sum(d > 1e-12))
    m = int(np.sum(np.abs(d) > 1e-12))
    if m == 0:
        return 1.0
    return sum(comb(m, j) for j in range(pos, m + 1)) / 2 ** m


def boot_upper(s, rng, B=2000):
    s = np.asarray(s)
    idx = rng.integers(0, len(s), size=(B, len(s)))
    return np.quantile(s[idx].mean(axis=1), 0.975)


def holm(ps, alpha=ALPHA):
    order = np.argsort(ps)
    rej = np.zeros(len(ps), bool)
    for k, i in enumerate(order):
        if ps[i] <= alpha / (len(ps) - k):
            rej[i] = True
        else:
            break
    return rej


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def sim_pair_scores(n, mu, rng, bias=0.0, kappa=None):
    """Reader model: P(answer "1") = sig(+-k_i + bias), sign + when image 1 is the correct one.

    k_i = kappa * z_i, z_i ~ Exp(1): pairs differ in how visible the condition is (some
    obvious, some invisible). kappa is set so the mean single-read accuracy is mu.
    Order AB (correct "1"): P(correct) = sig(k + b). Order BA (correct "2"): sig(k - b)."""
    if kappa is None:
        kappa = KAPPA[mu]
    k = kappa * rng.exponential(1.0, size=n)
    c_ab = rng.random(n) < _sig(k + bias)
    c_ba = rng.random(n) < _sig(k - bias)
    return (c_ab.astype(float) + c_ba.astype(float)) / 2


def _calibrate(mu, rng):
    if mu == 0.5:
        return 0.0
    lo, hi = 0.0, 50.0
    z = rng.exponential(1.0, size=200000)
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sig(mid * z).mean() < mu:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


MUS = (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9)
KAPPA = {}


def main():
    rng = np.random.default_rng(SEED)
    for mu in MUS:
        KAPPA[mu] = _calibrate(mu, rng)
    R = 2000
    print("W1 design check: paired 2AFC, exact sign-flip, Holm over 3 contrasts at alpha .01")
    print(f"AT CHANCE bar: bootstrap 97.5% quantile of mean pair score <= {BAR_AT_CHANCE}\n")
    print("n_pairs  mu    P(CARRIES single-test p<=.0033)  P(p<=.01)  P(AT CHANCE)  P(UNCLEAR)")
    for n in (12, 16, 20):
        for mu in MUS:
            car = car01 = atc = unc = 0
            for _ in range(R):
                s = sim_pair_scores(n, mu, rng, bias=rng.uniform(-1, 1))
                p = signflip_p(s)
                ub = boot_upper(s, rng)
                if p <= ALPHA / 3:
                    car += 1
                elif ub <= BAR_AT_CHANCE:
                    atc += 1
                else:
                    unc += 1
                car01 += p <= ALPHA
            print(f"{n:7d}  {mu:.2f}  {car/R:30.3f}  {car01/R:9.3f}  {atc/R:12.3f}  {unc/R:10.3f}")
        print()
    # a position-biased reader with no information must score exactly .5 per pair on average
    print("Position-biased blind reader (always answers '1'): pair score = .5 on every pair by construction")
    s = np.full(16, 0.5)
    print("  sign-flip p =", signflip_p(s), " (no evidence, as required)")
    # Holm false-positive check across 3 null contrasts
    fp = 0
    for _ in range(R):
        ps = [signflip_p(sim_pair_scores(16, 0.5, rng, bias=rng.uniform(-2, 2))) for _ in range(3)]
        fp += holm(np.array(ps)).any()
    print(f"\nFamily-wise false positives, 3 null contrasts at n=16 with random position bias: {fp/R:.3f}")


# ---- graded scoring: each read reports P(the named image is the one that followed X) ----

def signflip_p_cont(s, rng=None, mc=None):
    """Exact (enumerated) one-sided sign-flip p on continuous pair scores, n <= 16;
    Monte Carlo with `mc` flips otherwise (used only in this simulation)."""
    d = np.asarray(s, float) - 0.5
    n = len(d)
    obs = d.mean()
    if n <= 16 and mc is None:
        signs = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
        means = (signs * np.abs(d)).mean(axis=1)
        return float(np.mean(means >= obs - 1e-12))
    signs = rng.choice([-1.0, 1.0], size=(mc, n))
    means = (signs * np.abs(d)).mean(axis=1)
    return float((1 + np.sum(means >= obs - 1e-12)) / (1 + mc))


def sim_pair_scores_graded(n, mu, rng, bias=0.0, noise=0.7):
    """Same reader model; the read reports a probability for the correct image,
    sig(k +- b + e), e ~ N(0, noise) (a noisy, roughly calibrated reader), clipped to
    [.5 - .5, .5 + .5] and rounded to the nearest .05 (readers say 55, 60, ... 100)."""
    k = KAPPA[mu] * rng.exponential(1.0, size=n)
    r_ab = _sig(k + bias + rng.normal(0, noise, n))
    r_ba = _sig(k - bias + rng.normal(0, noise, n))
    r_ab, r_ba = np.round(r_ab * 20) / 20, np.round(r_ba * 20) / 20
    return (r_ab + r_ba) / 2


def graded_table(rng, R=1000):
    print("\nGraded scoring (probability on the correct image, 2 orders), exact sign-flip at n=16")
    print("n_pairs  mu    P(CARRIES p<=.0033)  P(p<=.01)  P(AT CHANCE)  P(UNCLEAR)")
    for n in (12, 16):
        for mu in MUS:
            car = car01 = atc = unc = 0
            for _ in range(R):
                s = sim_pair_scores_graded(n, mu, rng, bias=rng.uniform(-1, 1))
                p = signflip_p_cont(s)
                ub = boot_upper(s, rng)
                if p <= ALPHA / 3:
                    car += 1
                elif ub <= BAR_AT_CHANCE:
                    atc += 1
                else:
                    unc += 1
                car01 += p <= ALPHA
            print(f"{n:7d}  {mu:.2f}  {car/R:19.3f}  {car01/R:9.3f}  {atc/R:12.3f}  {unc/R:10.3f}")
        print()
    fp = 0
    for _ in range(R):
        ps = [signflip_p_cont(sim_pair_scores_graded(16, 0.5, rng, bias=rng.uniform(-2, 2))) for _ in range(3)]
        fp += holm(np.array(ps)).any()
    print(f"Family-wise false positives, graded, 3 null contrasts at n=16, random bias: {fp/R:.3f}")


def bar_table(rng, R=1000, n=16):
    print(f"\nAT CHANCE bar calibration, graded scoring, n={n}: P(AT CHANCE | mu) by bar; mean graded score by mu")
    bars = (0.60, 0.625, 0.65, 0.70)
    print("mu    mean_s  " + "  ".join(f"bar{b:.3f}" for b in bars))
    for mu in MUS:
        hits = np.zeros(len(bars)); ms = []
        for _ in range(R):
            s = sim_pair_scores_graded(n, mu, rng, bias=rng.uniform(-1, 1))
            ms.append(s.mean())
            p = signflip_p_cont(s)
            ub = boot_upper(s, rng)
            for j, b in enumerate(bars):
                hits[j] += (p > ALPHA / 3) and (ub <= b)
        print(f"{mu:.2f}  {np.mean(ms):.3f}   " + "  ".join(f"{h/R:8.3f}" for h in hits))


if __name__ == "__main__":
    main()                                   # binary pair scores
    rng = np.random.default_rng(SEED + 1)
    for mu in MUS:
        KAPPA[mu] = _calibrate(mu, rng)
    graded_table(rng)                        # graded pair scores (registered)
    bar_table(np.random.default_rng(SEED + 2))   # AT CHANCE bar on the graded scale
