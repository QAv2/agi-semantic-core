# adjudication_stats.py — Phase A concept-level convergence vs chance.
#
# Pre-named statistic (WING_COHORT_PROTOCOL.md): fraction of Phase-A states
# proposed independently by >= ceil(N/2) families, matched by referent.
# N = 7 families -> threshold 4.
#
# Chance baseline — referent-permutation, group-preserving: for each family,
# keep its observed partition of states into axis-groups (dyads stay dyads,
# singletons stay singletons) and reassign the groups to axes drawn uniformly
# WITHOUT replacement from the axis universe K (all valid axes any family
# produced). This preserves every family's group structure and axis count and
# breaks only the cross-family alignment. Using the OBSERVED universe (K=33)
# rather than the true space of possible referents makes the null generous to
# chance (fewer axes -> more collisions), i.e. conservative for the
# convergence claim.
#
# Reported at three judgment tiers: clear-only / clear+lean (PRIMARY) /
# +borderline. A29 (invalid referents, 2 states) is excluded from numerator
# and denominator everywhere.
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 20260822
N_SIMS = 10000
THRESHOLD = 4   # ceil(7/2)

mt = json.load(open(HERE / 'match_table.json'))
ROWS = [r for r in mt['assignments'] if r['confidence'] != 'excluded']
UNIVERSE = sorted(k for k in mt['axes'] if k != 'A29')
TIERS = {'clear-only': ('clear',),
         'clear+lean (PRIMARY)': ('clear', 'lean'),
         '+borderline': ('clear', 'lean', 'borderline')}


def convergence(rows):
    fams_on_axis = defaultdict(set)
    states_on_axis = defaultdict(int)
    for r in rows:
        fams_on_axis[r['axis']].add(r['fam'])
        states_on_axis[r['axis']] += 1
    conv = {a for a, fams in fams_on_axis.items() if len(fams) >= THRESHOLD}
    n_states = sum(states_on_axis[a] for a in conv)
    return conv, fams_on_axis, n_states


def family_groups(rows):
    """family -> list of group sizes (its states per distinct axis)."""
    per_fam = defaultdict(lambda: defaultdict(int))
    for r in rows:
        per_fam[r['fam']][r['axis']] += 1
    return {fam: sorted(d.values(), reverse=True) for fam, d in per_fam.items()}


def null_sim(groups, n_states_total, rng):
    fams_on_axis = defaultdict(set)
    states_on_axis = defaultdict(int)
    for fam, sizes in groups.items():
        axes = rng.choice(len(UNIVERSE), size=len(sizes), replace=False)
        for ax, size in zip(axes, sizes):
            fams_on_axis[ax].add(fam)
            states_on_axis[ax] += size
    conv = [a for a, fams in fams_on_axis.items() if len(fams) >= THRESHOLD]
    n_states = sum(states_on_axis[a] for a in conv)
    max_tower = max((len(f) for f in fams_on_axis.values()), default=0)
    return len(conv), n_states / n_states_total, max_tower


def main():
    results = {}
    for tier_name, confs in TIERS.items():
        rows = [r for r in ROWS if r['confidence'] in confs]
        denom = len(ROWS)   # all valid states, regardless of tier inclusion
        conv, fams_on_axis, n_states = convergence(rows)
        frac = n_states / denom
        groups = family_groups(rows)

        rng = np.random.default_rng(SEED)
        n_conv_null, frac_null, towers = [], [], []
        for _ in range(N_SIMS):
            c, f, t = null_sim(groups, denom, rng)
            n_conv_null.append(c)
            frac_null.append(f)
            towers.append(t)
        frac_null = np.array(frac_null)
        p_frac = float((np.sum(frac_null >= frac) + 1) / (N_SIMS + 1))
        p_nconv = float((np.sum(np.array(n_conv_null) >= len(conv)) + 1) / (N_SIMS + 1))
        obs_max_tower = max((len(f) for f in fams_on_axis.values()), default=0)
        p_tower = float((np.sum(np.array(towers) >= obs_max_tower) + 1) / (N_SIMS + 1))

        results[tier_name] = dict(
            convergent_axes={a: sorted(fams_on_axis[a]) for a in sorted(conv)},
            n_convergent_axes=len(conv),
            states_in_convergent=n_states,
            denominator=denom,
            fraction=round(frac, 3),
            null_fraction_mean=round(float(frac_null.mean()), 3),
            null_fraction_p95=round(float(np.percentile(frac_null, 95)), 3),
            p_fraction=p_frac,
            null_n_axes_mean=round(float(np.mean(n_conv_null)), 2),
            p_n_axes=p_nconv,
            max_tower_families=obs_max_tower,
            null_max_tower_mean=round(float(np.mean(towers)), 2),
            p_max_tower=p_tower,
        )
        r = results[tier_name]
        print(f"\n== {tier_name} ==")
        print(f"  convergent axes (>= {THRESHOLD}/7 families): {r['n_convergent_axes']} "
              f"-> {sorted(conv)}")
        for a in sorted(conv):
            print(f"    {a} {mt['axes'][a]['name']}: {len(fams_on_axis[a])}/7 "
                  f"({', '.join(sorted(fams_on_axis[a]))})")
        print(f"  states in convergent axes: {n_states}/{denom} = {r['fraction']}")
        print(f"  null (10k, group-preserving): fraction mean {r['null_fraction_mean']} "
              f"p95 {r['null_fraction_p95']} -> p = {r['p_fraction']:.4f}")
        print(f"  axes-count null mean {r['null_n_axes_mean']} -> p = {r['p_n_axes']:.4f}; "
              f"max tower {obs_max_tower}/7 vs null mean {r['null_max_tower_mean']} "
              f"-> p = {r['p_max_tower']:.4f}")

    out = dict(protocol='docs/WING_COHORT_PROTOCOL.md', threshold=THRESHOLD,
               n_families=7, universe_size=len(UNIVERSE), n_sims=N_SIMS,
               seed=SEED, excluded_states=2, results=results)
    (HERE / 'adjudication_stats.json').write_text(json.dumps(out, indent=1))
    print('\nWROTE adjudication_stats.json')


if __name__ == '__main__':
    main()
