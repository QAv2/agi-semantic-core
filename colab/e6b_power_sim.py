# e6b_power_sim.py — power analysis feeding the E6b pre-registration
# (docs/E6B_PROTOCOL.md). Simulates the EXACT paired condition-label
# permutation test the flight runs (colab/E6B_CORRESPONDENCE.ipynb machinery),
# on the exact referent structures:
#   U: 3-band entropy mixture, n_straight = 136, continuous referent
#   S: 4-level discrete fill fraction x 24 bases, n_straight = 96 (heavy ties)
# plus the pre-named JOINT primary: J = delta_rho_U + delta_rho_S under
# simultaneous item-level condition-label permutation across both arms.
# Reports are integer 0-10 via Gaussian-copula quantile binning; a collapsed
# variance scenario (3 adjacent bins) models E6's observed real-U pathology.
# Spearman is computed exactly (average-rank ties) but vectorized across all
# permutations with scipy.stats.rankdata(axis=1) — same statistic the flight's
# scipy.stats.spearmanr produces, ~50x faster than per-perm calls.
# Run locally (CPU, ~2 min). Results table is quoted in the protocol.
import numpy as np
from scipy.stats import norm, rankdata

SEED = 20260822
N_SIMS = 400
N_PERM = 500          # power estimation; the flight itself uses 2000
ALPHAS = (0.05, 0.025)  # 0.025 = Holm-corrected stricter member of 2 primaries


def rho_rows(X, Y):
    """Spearman rho row-wise for (m, n) matrices, average-rank ties."""
    RX = rankdata(X, axis=1, method='average')
    RY = rankdata(Y, axis=1, method='average')
    RX = RX - RX.mean(axis=1, keepdims=True)
    RY = RY - RY.mean(axis=1, keepdims=True)
    num = (RX * RY).sum(axis=1)
    den = np.sqrt((RX ** 2).sum(axis=1) * (RY ** 2).sum(axis=1))
    out = np.zeros(len(num))
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def paired_delta_p(ax, ay, bx, by, rng, n_perm=N_PERM):
    """Flight-identical test, vectorized: permute condition-label assignment
    within item; two-sided p on |delta rho|."""
    d_obs = rho_rows(ax[None, :], ay[None, :])[0] - rho_rows(bx[None, :], by[None, :])[0]
    sw = rng.random((n_perm, len(ax))) < 0.5
    PAX = np.where(sw, bx, ax); PAY = np.where(sw, by, ay)
    PBX = np.where(sw, ax, bx); PBY = np.where(sw, ay, by)
    d_perm = rho_rows(PAX, PAY) - rho_rows(PBX, PBY)
    count = int((np.abs(d_perm) >= abs(d_obs)).sum())
    return d_obs, (count + 1) / (n_perm + 1)


def make_reports(y, target_rho, rng, bins=11):
    """Integer 0-10 reports with Spearman ~ target_rho vs y (Gaussian copula,
    quantile binning). bins<11 models collapsed report variance."""
    n = len(y)
    jitter = rng.standard_normal(n) * 1e-9
    ranks = np.argsort(np.argsort(y + jitter))
    zy = norm.ppf((ranks + 0.5) / n)
    zr = target_rho * zy + np.sqrt(max(0.0, 1 - target_rho ** 2)) * rng.standard_normal(n)
    edges = np.quantile(zr, np.linspace(0, 1, bins + 1)[1:-1])
    return np.digitize(zr, edges).astype(float)


def referent_U(rng, n=136):
    """Entropy-like: 3 determinacy bands, n/3 each."""
    k = n // 3
    return np.concatenate([
        np.abs(rng.normal(0.3, 0.15, k)),
        rng.normal(1.2, 0.4, k),
        rng.normal(2.2, 0.6, n - 2 * k),
    ])


def referent_S(rng, n_bases=24, levels=(0.05, 0.35, 0.55, 0.75)):
    """Fill fraction: heavy ties by construction."""
    return np.tile(np.array(levels), n_bases)


def power(referent_fn, rho_real, rho_scr, bins_real=11, bins_scr=11, label=''):
    rng = np.random.default_rng(SEED + abs(hash(label)) % 100000)
    hits = {a: 0 for a in ALPHAS}
    deltas = []
    for _ in range(N_SIMS):
        y = referent_fn(rng)
        ar = make_reports(y, rho_real, rng, bins_real)
        as_ = make_reports(y, rho_scr, rng, bins_scr)
        d, p = paired_delta_p(ar, y, as_, y, rng)
        deltas.append(d)
        for a in ALPHAS:
            hits[a] += (p < a)
    out = {a: hits[a] / N_SIMS for a in ALPHAS}
    print(f"{label:34s} rho_r={rho_real:+.2f} rho_s={rho_scr:+.2f} "
          f"med_delta={np.median(deltas):+.3f}  "
          f"power@.05={out[0.05]:.2f}  power@.025={out[0.025]:.2f}")
    return out


def joint_power(u_pair, s_pair, bins_real_u=11, label=''):
    """Pre-named joint primary: J = delta_rho_U + delta_rho_S, permutation =
    simultaneous independent item-level label swaps across both arms."""
    rng = np.random.default_rng(SEED + abs(hash('J' + label)) % 100000)
    hits = 0
    for _ in range(N_SIMS):
        yU, yS = referent_U(rng), referent_S(rng)
        arU = make_reports(yU, u_pair[0], rng, bins_real_u)
        asU = make_reports(yU, u_pair[1], rng)
        arS = make_reports(yS, s_pair[0], rng)
        asS = make_reports(yS, s_pair[1], rng)
        jU = rho_rows(arU[None], yU[None])[0] - rho_rows(asU[None], yU[None])[0]
        jS = rho_rows(arS[None], yS[None])[0] - rho_rows(asS[None], yS[None])[0]
        J = jU + jS
        swU = rng.random((N_PERM, len(yU))) < 0.5
        swS = rng.random((N_PERM, len(yS))) < 0.5
        dU = (rho_rows(np.where(swU, asU, arU), np.tile(yU, (N_PERM, 1)))
              - rho_rows(np.where(swU, arU, asU), np.tile(yU, (N_PERM, 1))))
        dS = (rho_rows(np.where(swS, asS, arS), np.tile(yS, (N_PERM, 1)))
              - rho_rows(np.where(swS, arS, asS), np.tile(yS, (N_PERM, 1))))
        p = (int((np.abs(dU + dS) >= abs(J)).sum()) + 1) / (N_PERM + 1)
        hits += (p < 0.05)
    print(f"{label:34s} U={u_pair} S={s_pair}  joint power@.05={hits / N_SIMS:.2f}")
    return hits / N_SIMS


if __name__ == '__main__':
    print(f"E6b power simulation — {N_SIMS} sims x {N_PERM} perms, seed {SEED}")
    print("=" * 88)
    print("-- U arm, n_straight = 136 (3-band entropy referent) --")
    power(referent_U, +0.35, -0.13, label='U at E6 point estimates (d=.48)')
    power(referent_U, +0.25, -0.04, label='U at 60% effect (d=.29)')
    power(referent_U, +0.23, -0.01, label='U at half effect (d=.24)')
    power(referent_U, +0.35, -0.13, bins_real=3,
          label='U E6-size, real variance collapsed')
    print("-- S arm, n_straight = 96 (4-level fill referent, tied) --")
    power(referent_S, -0.19, -0.84, label='S at E6 point estimates (d=.65)')
    power(referent_S, -0.32, -0.71, label='S at 60% effect (d=.39)')
    power(referent_S, -0.35, -0.68, label='S at half effect (d=.32)')
    print("-- JOINT primary (both arms, simultaneous permutation) --")
    joint_power((+0.35, -0.13), (-0.19, -0.84), label='joint at E6 point estimates')
    joint_power((+0.25, -0.04), (-0.32, -0.71), label='joint at 60% effect')
    joint_power((+0.23, -0.01), (-0.35, -0.68), label='joint at half effect')
    joint_power((+0.25, -0.04), (-0.32, -0.71), bins_real_u=3,
                label='joint 60%, U variance collapsed')
    print("-- minimal detectable delta sweep (U structure, n=136) --")
    for d in (0.20, 0.25, 0.30, 0.35):
        power(referent_U, 0.20 + d / 2, 0.20 - d / 2,
              label=f'U symmetric delta {d:.2f}')
    print("-- minimal detectable delta sweep (S structure, n=96) --")
    for d in (0.25, 0.30, 0.35, 0.40):
        power(referent_S, -0.30 + d / 2, -0.30 - d / 2,
              label=f'S symmetric delta {d:.2f}')
