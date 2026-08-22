#!/usr/bin/env python3
"""Phase B — parsing + pre-registered analysis (PHASE_B_PREREG.md, run as written).

Reads responses/<TAG>.md (verbatim archives; parses only lines after the
OUTPUT marker), builds per-family relation graphs + polarity/register maps,
then runs exactly the pre-named tests:

  1. PRIMARY (K3): COMPLETION agreement T_C = sum over pairs of C(k_p,2)
     vs per-family degree-preserving double-edge-swap null, 1000 shuffles.
  2. Pair-level convergence: per-pair null probability of reaching >= k_p.
  3. Any-relation union agreement (secondary); SYNONYM graph; OPPOSITION graph.
  4. Polarity pooled pairwise agreement vs 0.5 (two-sided exact binomial).
  5. Register agreement vs 0.25.
  6. Angles: median/MAD per convergent COMPLETION pair; Spearman of family
     medians vs wing-v0 encoded angles (curiosity).
  7. Divergence cells: single-family relations; isolate states.

Outputs: phase_b_results.json + printed summary.
"""
import json, math, random, re, sys, collections
from pathlib import Path
from itertools import combinations

HERE = Path(__file__).resolve().parent
TAGS = ["FAB", "OPU", "SON", "HAI", "GEM", "MIS", "DSK"]
N_SHUFFLES = 1000
SEED = 20260822

pool = json.load(open(HERE / "pool_public.json"))
IDS = [e["id"] for e in pool]
IDSET = set(IDS)
NAME = {e["id"]: e["name"] for e in pool}
full_pool = json.load(open(HERE / "pool.json"))
KEY2ID = {e["key"]: e["id"] for e in full_pool["entries"]}

# --------------------------------------------------------------- parse
REL_RE = re.compile(r"^\s*REL\s+(S\d{3})\s+(S\d{3})\s+(SYNONYM|COMPLETION|OPPOSITION|KIN)\b\s*(\d{1,3})?", re.I)
POL_RE = re.compile(r"^\s*POL\s+(S\d{3})\s+(ACTIVE|RECEPTIVE)\b", re.I)
REG_RE = re.compile(r"^\s*REG\s+(S\d{3})\s+(SPATIAL|TEMPORAL|RELATIONAL|REFLEXIVE)\b", re.I)

def parse(tag):
    path = HERE / "responses" / f"{tag}.md"
    if not path.exists():
        return None
    text = path.read_text()
    m = re.search(r"^OUTPUT\s*$", text, re.M)
    body = text[m.end():] if m else text  # fall back to whole text; logged
    rels, pol, reg, dropped = {}, {}, {}, 0
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        mm = REL_RE.match(line)
        if mm:
            a, b, typ, ang = mm.group(1).upper(), mm.group(2).upper(), mm.group(3).upper(), mm.group(4)
            if a == b or a not in IDSET or b not in IDSET:
                dropped += 1; continue
            pair = tuple(sorted((a, b)))
            entry = rels.setdefault(pair, {})
            if typ == "COMPLETION" and ang is not None:
                entry.setdefault("COMPLETION_ANGLES", []).append(int(ang))
            entry[typ] = True
            continue
        mm = POL_RE.match(line)
        if mm and mm.group(1).upper() in IDSET:
            pol[mm.group(1).upper()] = mm.group(2).upper(); continue
        mm = REG_RE.match(line)
        if mm and mm.group(1).upper() in IDSET:
            reg[mm.group(1).upper()] = mm.group(2).upper(); continue
        if re.match(r"^\s*(REL|POL|REG)\b", line, re.I):
            dropped += 1
    return {"tag": tag, "rels": rels, "pol": pol, "reg": reg,
            "dropped": dropped, "had_marker": bool(m)}

families = {}
for t in TAGS:
    p = parse(t)
    if p is None:
        print(f"  [missing response: {t}]")
    else:
        families[t] = p
if len(families) < 3:
    sys.exit("fewer than 3 parsed families — protocol minimum not met; aborting stats")

# graphs per type per family
def graph(fam, typ):
    return {pair for pair, d in families[fam]["rels"].items() if d.get(typ)}

TYPES = ["COMPLETION", "SYNONYM", "OPPOSITION", "KIN"]
G = {typ: {f: graph(f, typ) for f in families} for typ in TYPES}
G["UNION"] = {f: set(families[f]["rels"].keys()) for f in families}

# --------------------------------------------------------------- null machinery
def rewire(edges, rng):
    """Degree-preserving double-edge swap on an undirected simple graph."""
    edges = [tuple(e) for e in edges]
    if len(edges) < 2:
        return set(edges)
    es = set(edges)
    target = 10 * len(edges)
    accepted = tries = 0
    lst = list(es)
    while accepted < target and tries < 200 * len(edges) + 1000:
        tries += 1
        i, j = rng.randrange(len(lst)), rng.randrange(len(lst))
        if i == j:
            continue
        (a, b), (c, d) = lst[i], lst[j]
        if rng.random() < 0.5:
            a, b = b, a
        # propose (a,d),(c,b)
        if a == d or c == b:
            continue
        e1, e2 = tuple(sorted((a, d))), tuple(sorted((c, b)))
        if e1 in es or e2 in es or e1 == e2:
            continue
        es.discard(lst[i]); es.discard(lst[j]); es.add(e1); es.add(e2)
        lst[i], lst[j] = e1, e2
        accepted += 1
    return es

def T_stat(graphs):
    cnt = collections.Counter()
    for g in graphs.values():
        cnt.update(g)
    return sum(k * (k - 1) // 2 for k in cnt.values()), cnt

def agreement_test(typ):
    graphs = G[typ]
    obs_T, obs_cnt = T_stat(graphs)
    rng = random.Random(SEED + hash(typ) % 10000)
    null_T = []
    pair_hits = collections.Counter()   # pair -> shuffles where null k >= obs k
    obs_pairs = {p: k for p, k in obs_cnt.items() if k >= 2}
    for _ in range(N_SHUFFLES):
        shuf = {f: rewire(graphs[f], rng) for f in graphs}
        t, cnt = T_stat(shuf)
        null_T.append(t)
        for p, k_obs in obs_pairs.items():
            if cnt.get(p, 0) >= k_obs:
                pair_hits[p] += 1
    p_global = (1 + sum(1 for t in null_T if t >= obs_T)) / (N_SHUFFLES + 1)
    pair_p = {p: (1 + pair_hits[p]) / (N_SHUFFLES + 1) for p in obs_pairs}
    convergent = sorted([(p, obs_cnt[p], pair_p[p]) for p in obs_pairs if pair_p[p] < 0.05],
                        key=lambda x: (-x[1], x[2]))
    return {"observed_T": obs_T, "null_mean": sum(null_T) / len(null_T),
            "null_max": max(null_T), "p": p_global,
            "n_edges_per_family": {f: len(graphs[f]) for f in graphs},
            "pairs_multi": {f"{a}|{b}": k for (a, b), k in sorted(obs_cnt.items()) if k >= 2},
            "convergent_pairs": [
                {"pair": [a, b], "names": [NAME[a], NAME[b]], "k": k, "p_pair": pp}
                for (a, b), k, pp in convergent]}

# --------------------------------------------------------------- binomial
def binom_two_sided(k, n, p0):
    if n == 0:
        return 1.0
    def pmf(i):
        return math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                        + i * math.log(p0) + (n - i) * math.log(1 - p0))
    obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs * (1 + 1e-9)))

def label_agreement(field, chance):
    votes = {f: families[f][field] for f in families}
    agree = total = 0
    per_pair = {}
    for f1, f2 in combinations(sorted(votes), 2):
        common = set(votes[f1]) & set(votes[f2])
        a = sum(1 for s in common if votes[f1][s] == votes[f2][s])
        per_pair[f"{f1}-{f2}"] = {"n": len(common), "agree": a,
                                  "rate": round(a / len(common), 3) if common else None}
        agree += a; total += len(common)
    consensus = {}
    for s in IDS:
        c = collections.Counter(v[s] for v in votes.values() if s in v)
        if c:
            top, k = c.most_common(1)[0]
            consensus[s] = top if k > sum(c.values()) / 2 else "TIE"
    return {"pooled_agree": agree, "pooled_n": total,
            "pooled_rate": round(agree / total, 4) if total else None,
            "p_vs_chance": binom_two_sided(agree, total, chance),
            "per_pair": per_pair, "consensus": consensus,
            "coverage": {f: len(votes[f]) for f in votes}}

# --------------------------------------------------------------- run
results = {"n_families": len(families),
           "families": {f: {"n_rel_pairs": len(families[f]["rels"]),
                            "n_pol": len(families[f]["pol"]),
                            "n_reg": len(families[f]["reg"]),
                            "dropped_lines": families[f]["dropped"],
                            "had_output_marker": families[f]["had_marker"]}
                        for f in families}}

for typ in ["COMPLETION", "SYNONYM", "OPPOSITION", "UNION"]:
    print(f"— agreement test: {typ}")
    results[typ] = agreement_test(typ)

results["polarity"] = label_agreement("pol", 0.5)
results["register"] = label_agreement("reg", 0.25)

# angles on convergent COMPLETION pairs
angles = {}
for item in results["COMPLETION"]["convergent_pairs"]:
    a, b = item["pair"]
    vals = []
    for f in families:
        d = families[f]["rels"].get((a, b), {})
        vals += d.get("COMPLETION_ANGLES", [])
    if vals:
        vals.sort()
        med = vals[len(vals) // 2] if len(vals) % 2 else (vals[len(vals)//2 - 1] + vals[len(vals)//2]) / 2
        mad = sorted(abs(v - med) for v in vals)[len(vals) // 2]
        angles[f"{a}|{b}"] = {"names": item["names"], "n": len(vals),
                              "median": med, "MAD": mad, "values": vals}
results["angles_convergent"] = angles

# v0 curiosity correlation (pairs mapping onto v0 encoded pairs)
V0_PAIRS = [("A01/open", "A01/closed", 93.0), ("A18/-", "A02/-", 80.0),
            ("A07/copy", "A07/compose", 99.9), ("A15/low-intake", "A15/high-intake", 94.8),
            ("A22/miscalibrated-confidence", "V0:CALIBRATION", 99.8)]
cur = []
for k1, k2, v0ang in V0_PAIRS:
    pair = tuple(sorted((KEY2ID[k1], KEY2ID[k2])))
    vals = []
    for f in families:
        vals += families[f]["rels"].get(pair, {}).get("COMPLETION_ANGLES", [])
    fams_nom = sum(1 for f in families if pair in G["COMPLETION"][f])
    if vals:
        vals.sort()
        med = vals[len(vals)//2] if len(vals) % 2 else (vals[len(vals)//2-1]+vals[len(vals)//2])/2
    else:
        med = None
    cur.append({"v0_pair": [k1, k2], "ids": list(pair), "v0_angle": v0ang,
                "cohort_median": med, "n_estimates": len(vals), "families_nominating": fams_nom})
have = [(c["cohort_median"], c["v0_angle"]) for c in cur if c["cohort_median"] is not None]
if len(have) >= 3:
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            for k2_ in range(i, j + 1):
                r[order[k2_]] = (i + j) / 2 + 1
            i = j + 1
        return r
    xs, ys = zip(*have)
    rx, ry = ranks(list(xs)), ranks(list(ys))
    mx, my = sum(rx)/len(rx), sum(ry)/len(ry)
    num = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a-mx)**2 for a in rx) * sum((b-my)**2 for b in ry))
    results["v0_angle_curiosity"] = {"pairs": cur, "spearman": num/den if den else None, "n": len(have)}
else:
    results["v0_angle_curiosity"] = {"pairs": cur, "spearman": None, "n": len(have)}

# divergence cells
union_cnt = collections.Counter()
for f in families:
    union_cnt.update(G["UNION"][f])
single = [p for p, k in union_cnt.items() if k == 1]
related_states = {s for p in union_cnt for s in p}
isolates = [s for s in IDS if s not in related_states]
results["divergence"] = {
    "single_family_relations": len(single),
    "isolate_states": [{"id": s, "name": NAME[s]} for s in isolates]}

json.dump(results, open(HERE / "phase_b_results.json", "w"), indent=1)

# --------------------------------------------------------------- summary
print("\n===== PHASE B RESULTS =====")
print(f"families parsed: {len(families)} / 7 -> {sorted(families)}")
for f, d in results["families"].items():
    print(f"  {f}: {d['n_rel_pairs']} rel pairs, POL {d['n_pol']}/101, REG {d['n_reg']}/101, dropped {d['dropped_lines']}")
c = results["COMPLETION"]
print(f"\nPRIMARY COMPLETION agreement: T={c['observed_T']} vs null mean {c['null_mean']:.1f} (max {c['null_max']}) -> p={c['p']:.4f}")
print(f"  K3 {'TRIGGERED' if c['p'] >= 0.05 else 'NOT triggered'}")
print(f"  convergent pairs (pair-level p<.05): {len(c['convergent_pairs'])}")
for it in c["convergent_pairs"][:20]:
    print(f"    k={it['k']} p={it['p_pair']:.3f}  {it['names'][0]} — {it['names'][1]}")
for typ in ["SYNONYM", "OPPOSITION", "UNION"]:
    r = results[typ]
    print(f"{typ}: T={r['observed_T']} null {r['null_mean']:.1f} p={r['p']:.4f}, convergent {len(r['convergent_pairs'])}")
pol, reg = results["polarity"], results["register"]
print(f"POLARITY pooled agreement {pol['pooled_rate']} (n={pol['pooled_n']}) vs 0.5 -> p={pol['p_vs_chance']:.2e}")
print(f"REGISTER pooled agreement {reg['pooled_rate']} (n={reg['pooled_n']}) vs 0.25 -> p={reg['p_vs_chance']:.2e}")
v0c = results["v0_angle_curiosity"]
print(f"v0 angle curiosity: n={v0c['n']} spearman={v0c['spearman']}")
print(f"divergence: {results['divergence']['single_family_relations']} single-family relations, "
      f"{len(results['divergence']['isolate_states'])} isolates")
