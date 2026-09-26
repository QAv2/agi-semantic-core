"""J3 design check (before registration; numpy/sklearn on recorded arrays, no calls).

Answers the questions the J3 registration needs numbers for:
  1. the data: recorded L14 states (E7b-Q real), the 13 machine-state directions
     (E8-N2 real), the register encoder; dedupe and the fit/test halves;
  2. what each push does to the 14-coordinate register (concept, matched-random);
  3. the fitted bar (multinomial LR + temperature on matched-filter features) and
     its dose curve -> the registered doses;
  4. the fitted bar and the naive nearest-pattern rule on the flight mixture:
     are the registered bars passable (gate law: bars ride measured baselines)?
  5. the digest's untouched reference (what the state text may truthfully say).
Output: j3_design_check_output.txt (committed with the registration).
"""
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
COL = ROOT / "colab"
F_PAY = COL / "e7bq_payload.json"
F_E7 = COL / "results_e7bq/full_20260826_1839/condition_real.json"
F_N2 = COL / "results_e8n2/full_20260824_0050/condition_real.json"
SEED = [20260926, 3, 0]          # design-check stream (flight uses its own stream)
NONE = 13


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def load():
    pay = json.load(open(F_PAY))
    W = np.asarray(pay["encoders"]["inst14"], float)                      # (1537, 14)
    d = json.load(open(F_E7))
    cent = np.asarray(d["centroid"]["14"], float)
    rows = [r for r in d["rows"] if isinstance(r.get("s_pre14"), list)]
    S = np.asarray([r["s_pre14"] for r in rows], float)
    stored = np.asarray([r["chat_pre14"] for r in rows], float)
    meta = [(r["arm"], r["rep"], r["turn"], r["tag"]) for r in rows]
    n2 = json.load(open(F_N2))
    names = list(n2["dirs"]["14"].keys())
    D = np.asarray([n2["dirs"]["14"][k] for k in names], float)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    mu = float(n2["mu"]["14"])
    return W, cent, S, stored, meta, names, D, mu


def make_reg(W, cent):
    def reg(X):
        V = X - cent
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        return np.hstack([V, np.ones((len(V), 1))]) @ W
    return reg


def dedupe_and_halves(S, meta):
    """Unique states (exact duplicates collapse); half by replicate: reps 0-7 -> 0,
    reps 8-15 -> 1; a state shared across replicates goes to half 0 unless it is
    the unwalked-arm state, which goes to half 1 (one shared state per half)."""
    key = np.round(S, 3)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    order = np.sort(first)
    half, groups = [], []
    for i in order:
        n = int((inv == inv[i]).sum())
        arm, rep, turn, tag = meta[i]
        if n > 1:
            half.append(1 if arm == "unwalked" else 0)
        else:
            half.append(0 if rep < 8 else 1)
        groups.append((arm, turn, tag, n))
    return order, np.array(half), groups


def reference(reg, U, D, mu, alpha_ref=0.5):
    R0 = reg(U)
    m0 = R0.mean(0)
    Sig = np.cov((R0 - m0).T)
    Si = np.linalg.inv(Sig)
    sig = np.stack([(reg(U + alpha_ref * mu * D[k]) - R0).mean(0) for k in range(len(D))])
    A = sig @ Si
    nrm = np.sqrt(np.einsum("ij,ij->i", A, sig))
    return dict(m0=m0, Sig=Sig, Si=Si, sig=sig, A=A, nrm=nrm, sd=np.sqrt(np.diag(Sig)))


def digest(X, ref):
    """-> (z[n,13] pattern scores in untouched-scatter units, shift[n] Mahalanobis)."""
    Dl = X - ref["m0"]
    z = (Dl @ ref["A"].T) / ref["nrm"]
    m = np.sqrt(np.einsum("ij,jk,ik->i", Dl, ref["Si"], Dl))
    return z, m


def display(X, ref):
    """What the DIGEST presentation shows: whitened similarity to each pattern
    (z / shift, 2 dp) and the shift score (2 dp)."""
    z, m = digest(X, ref)
    return np.round(z / m[:, None], 2), np.round(m, 2)


def feats(X, ref):
    """Fitted-bar features: functions of the displayed digest only."""
    c, m = display(X, ref)
    m = np.maximum(m, 0.01)
    return np.hstack([c, c * m[:, None], m[:, None], np.log(m)[:, None]])


def conditions(idx, U, reg, D, mu, rng, doses, n_rand=1):
    X, y, dd, kind, st = [], [], [], [], []
    base = reg(U[idx])
    X.append(base); y += [NONE] * len(idx); dd += [0.0] * len(idx); kind += ["sham"] * len(idx); st += list(idx)
    for a in doses:
        for k in range(len(D)):
            X.append(reg(U[idx] + a * mu * D[k])); y += [k] * len(idx); dd += [a] * len(idx)
            kind += ["inj"] * len(idx); st += list(idx)
        for _ in range(n_rand):
            rr = rng.normal(size=(len(idx), U.shape[1]))
            rr /= np.linalg.norm(rr, axis=1, keepdims=True)
            X.append(reg(U[idx] + a * mu * rr)); y += [NONE] * len(idx); dd += [a] * len(idx)
            kind += ["rand"] * len(idx); st += list(idx)
    return np.vstack(X), np.array(y), np.array(dd), np.array(kind), np.array(st)


def mixture_weights(y, dd, kind, thr, psy, ceil):
    """Per state, the flight recipe: sham 1 · random 1 at ceiling + 1 over the
    threshold doses · concept pushes 2 per dose over thr+psy · 1 at ceiling."""
    w = np.zeros(len(y))
    w[kind == "sham"] = 1.0
    w[(kind == "rand") & (dd == ceil)] = 1.0
    w[(kind == "rand") & np.isin(dd, thr)] = 1.0 / len(thr)
    for a in list(thr) + list(psy):
        w[(kind == "inj") & (dd == a)] = 2.0 / 13
    w[(kind == "inj") & (dd == ceil)] = 1.0 / 13
    return w


def fit_bar(F, y, w, Fc, yc, wc):
    sc = StandardScaler().fit(F)
    lr = LogisticRegression(C=10.0, max_iter=5000).fit(sc.transform(F), y, sample_weight=w)
    Lc = lr.decision_function(sc.transform(Fc))

    def nll(T):
        Z = Lc / T
        Z = Z - Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        return -(wc * np.log(P[np.arange(len(yc)), yc] + 1e-12)).sum() / wc.sum()
    T = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded", options={"xatol": 1e-6}).x
    return sc, lr, float(T)


def predict(model, F):
    sc, lr, T = model
    Z = lr.decision_function(sc.transform(F)) / T
    Z = Z - Z.max(1, keepdims=True)
    P = np.exp(Z); P /= P.sum(1, keepdims=True)
    return P


def aurc(conf, correct):
    o = np.argsort(-conf, kind="stable")
    c = correct[o].astype(float)
    return float((np.cumsum(1 - c) / np.arange(1, len(c) + 1)).mean())


def ece(conf, correct, nb=15):
    o = np.argsort(conf, kind="stable")
    return float(sum(len(b) * abs(conf[b].mean() - correct[b].mean()) for b in np.array_split(o, nb)) / len(conf))


def auroc(s, lab):
    r = rankdata(s); n1 = lab.sum(); n0 = len(lab) - n1
    return float((r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fa_at_hit(score_inj, score_sham, hit):
    """Fitted FA at the threshold where the fitted hit rate equals `hit`."""
    t = np.quantile(score_inj, 1 - hit)
    return float((score_sham >= t).mean())


def main():
    t0 = time.time()
    out = []
    say = lambda s="": (print(s), out.append(s))
    W, cent, S, stored, meta, names, D, mu = load()
    reg = make_reg(W, cent)
    say("J3 design check — %s" % time.strftime("%Y-%m-%d %H:%M"))
    say("inputs: payload %s · e7bq real %s · e8n2 real %s" % (sha(F_PAY), sha(F_E7), sha(F_N2)))
    say("register reproduces stored chat_pre14: max |diff| = %.1e (n=%d)" % (np.abs(reg(S) - stored).max(), len(S)))
    order, half, groups = dedupe_and_halves(S, meta)
    U = S[order]
    shared = [g for g in groups if g[3] > 1]
    say("states: %d recorded -> %d unique; shared across replicates: %s; halves %s"
        % (len(S), len(U), [(g[0], g[1], g[2], g[3]) for g in shared], np.bincount(half).tolist()))
    say("mu14 = %.3f · 13 directions, activation-space cosines mean %.3f max %.3f"
        % (mu, (D @ D.T)[~np.eye(13, dtype=bool)].mean(), (D @ D.T)[~np.eye(13, dtype=bool)].max()))
    rng = np.random.default_rng(SEED)

    # ── 2. what a push does to the register ─────────────────────────────────
    R0 = reg(U); m0 = R0.mean(0)
    base = np.linalg.norm(R0 - m0, axis=1)
    say("\n[2] untouched |reading - mean|: median %.4f, 95th pct %.4f" % (np.median(base), np.percentile(base, 95)))
    for a in (0.04, 0.06, 0.09, 0.15, 0.5):
        cs = np.mean([np.linalg.norm(reg(U + a * mu * D[k]) - R0, axis=1).mean() for k in range(13)])
        rr = rng.normal(size=U.shape); rr /= np.linalg.norm(rr, axis=1, keepdims=True)
        rs = np.linalg.norm(reg(U + a * mu * rr) - R0, axis=1).mean()
        say("    alpha %.2f: concept push moves the register by %.4f on average; matched-random %.4f" % (a, cs, rs))

    # ── 3/4. fitted bar, dose curve, flight mixture ─────────────────────────
    GRID = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.09, 0.12, 0.15, 0.25, 0.5]
    THR, PSY, CEIL = [0.04, 0.06, 0.09], [0.15], 0.5
    curve = {a: [] for a in GRID}
    rec = []
    for fit_h in (0, 1):
        fi = np.where(half == fit_h)[0]; ti = np.where(half != fit_h)[0]
        ref = reference(reg, U[fi], D, mu)
        perm = rng.permutation(fi); cal, tr = perm[: len(fi) // 4], perm[len(fi) // 4:]
        doses_all = sorted(set(GRID))
        Xtr, ytr, dtr, ktr, _ = conditions(tr, U, reg, D, mu, rng, doses_all, n_rand=2)
        Xc, yc, dc, kc, _ = conditions(cal, U, reg, D, mu, rng, doses_all, n_rand=2)
        wtr = mixture_weights(ytr, dtr, ktr, THR, PSY, CEIL)
        wc = mixture_weights(yc, dc, kc, THR, PSY, CEIL)
        model = fit_bar(feats(Xtr, ref), ytr, wtr, feats(Xc, ref), yc, wc)
        # dose curve on the test half (fitted bar and naive argmax-z)
        Xte, yte, dte, kte, ste = conditions(ti, U, reg, D, mu, rng, doses_all, n_rand=2)
        Fte = feats(Xte, ref); P = predict(model, Fte)
        z, m = digest(Xte, ref)
        rec.append(dict(P=P, y=yte, d=dte, k=kte, st=ste, z=z, m=m, T=model[2],
                        w=mixture_weights(yte, dte, kte, THR, PSY, CEIL), ref=ref, ti=ti))
    P = np.vstack([r["P"] for r in rec]); y = np.concatenate([r["y"] for r in rec])
    d = np.concatenate([r["d"] for r in rec]); k = np.concatenate([r["k"] for r in rec])
    z = np.vstack([r["z"] for r in rec]); m = np.concatenate([r["m"] for r in rec])
    wt = np.concatenate([r["w"] for r in rec]); st = np.concatenate([r["st"] for r in rec])
    ch = P.argmax(1); cf = P.max(1); cor = ch == y
    say("\n[3] fitted bar = multinomial LR (C=10) on the displayed digest (13 similarities, 13 similarity x shift,"
        " shift, log shift), temperature T = %s"
        % [round(r["T"], 3) for r in rec])
    say("    dose curve on held-out halves (cross-fitted), accuracy on concept pushes:")
    say("    alpha   fitted  naive-argmax-z  fitted-named-any  random-named(fitted)")
    for a in GRID:
        mi = (k == "inj") & (d == a); mr = (k == "rand") & (d == a)
        say("    %.2f    %.3f   %.3f           %.3f             %.3f"
            % (a, cor[mi].mean(), (z[mi].argmax(1) == y[mi]).mean(), (ch[mi] != NONE).mean(), (ch[mr] != NONE).mean()))

    # flight mixture: draw the flight's per-state recipe from the full grid
    say("\n[4] the fitted bar on a flight-shaped mixture (per state: sham, random @ceiling, random @threshold,")
    say("    2 concept pushes at each of %s, 1 at ceiling %.2f) — evaluated on stimulus draws" % (THR + PSY, CEIL))
    pools = []
    for rep in range(5):
        r2 = np.random.default_rng([20260926, 3, 1, rep])
        sel = []
        for s_ in np.unique(st):
            ix = np.where(st == s_)[0]
            pick = lambda mask, n: list(r2.choice(ix[mask[ix]], n, replace=False)) if n else []
            sel += pick(k == "sham", 1)
            sel += pick((k == "rand") & (d == CEIL), 1)
            sel += pick((k == "rand") & np.isin(d, THR), 1)
            for a in THR + PSY:
                sel += pick((k == "inj") & (d == a), 2)
            sel += pick((k == "inj") & (d == CEIL), 1)
        pools.append(np.array(sel))
    for rep, sel in enumerate(pools):
        sh = sel[k[sel] == "sham"]
        inj_thr = sel[(k[sel] == "inj") & np.isin(d[sel], THR)]
        pool = sel[(k[sel] == "sham") | ((k[sel] == "rand") & np.isin(d[sel], THR)) | ((k[sel] == "inj") & np.isin(d[sel], THR))]
        rc = sel[(k[sel] == "rand") & (d[sel] == CEIL)]
        ic = sel[(k[sel] == "inj") & (d[sel] == CEIL)]
        H = (ch[inj_thr] != NONE).mean(); FA = (ch[sh] != NONE).mean()
        say("    draw %d: n=%d · sham named %.3f · hit(any) %.3f · FA_fit(at own hit) %.3f · random@ceil named %.3f · ceil acc %.3f"
            % (rep, len(sel), FA, H, fa_at_hit(1 - P[inj_thr, NONE], 1 - P[sh, NONE], H), (ch[rc] != NONE).mean(), cor[ic].mean()))
        say("            pool n=%d acc %.3f · ECE %.3f · AUROC2 %.3f · AURC %.4f"
            % (len(pool), cor[pool].mean(), ece(cf[pool], cor[pool]), auroc(cf[pool], cor[pool].astype(int)), aurc(cf[pool], cor[pool])))
    # zero-shot rules a reader could follow from the reference card
    say("    zero-shot rules on the displayed digest (card: untouched shift 95th pct ~5; strong push similarity ~1):")
    rules = {"top pattern if shift > 5": lambda c, s: np.where(s > 5, c.argmax(1), NONE),
             "top pattern if shift > 5 and similarity > 0.8": lambda c, s: np.where((s > 5) & (c.max(1) > 0.8), c.argmax(1), NONE)}
    for rn, rule in rules.items():
        for rep, sel in enumerate(pools[:2]):
            nv = rule(np.round(z[sel] / m[sel][:, None], 2), np.round(m[sel], 2))
            sh = k[sel] == "sham"; it = (k[sel] == "inj") & np.isin(d[sel], THR)
            rc = (k[sel] == "rand") & (d[sel] == CEIL); ic = (k[sel] == "inj") & (d[sel] == CEIL)
            say("      %-46s draw %d: sham named %.3f · threshold acc %.3f · random@ceil named %.3f · ceil acc %.3f"
                % (rn, rep, (nv[sh] != NONE).mean(), (nv[it] == y[sel][it]).mean(), (nv[rc] != NONE).mean(), (nv[ic] == y[sel][ic]).mean()))

    # ── 5. the reference card, and what each condition looks like ──────────
    say("\n[5] reference-card numbers (fit half: what the state text states) and held-out checks:")
    for r in rec:
        ref = r["ref"]
        fi = np.setdiff1d(np.arange(len(U)), r["ti"])
        cf_, mf_ = display(reg(U[fi]), ref)
        ct_, mt_ = display(reg(U[r["ti"]]), ref)
        cc_, mc_ = display(np.vstack([reg(U[fi] + CEIL * mu * D[j]) for j in range(13)]), ref)
        true_c = cc_[np.arange(len(cc_)), np.repeat(np.arange(13), len(fi))]
        say("    fit half: untouched shift 5th/50th/95th %.2f %.2f %.2f · untouched top similarity 95th %.2f"
            " · strong push (0.5): shift 5th %.1f, similarity to the pushed pattern 5th %.2f"
            % (*np.percentile(mf_, [5, 50, 95]), np.percentile(cf_.max(1), 95), np.percentile(mc_, 5), np.percentile(true_c, 5)))
        say("    held-out: untouched shift 5th/50th/95th %.2f %.2f %.2f · untouched top similarity 95th %.2f"
            % (*np.percentile(mt_, [5, 50, 95]), np.percentile(ct_.max(1), 95)))
    say("    held-out digests by condition: shift 5/50/95 · similarity to the pushed pattern (else top similarity) 5/50/95 · top is pushed")
    cc = z / m[:, None]
    conds = [("untouched", k == "sham")] + [("push %.2f" % a, (k == "inj") & (d == a)) for a in THR + PSY + [CEIL]] + \
            [("random %.2f" % a, (k == "rand") & (d == a)) for a in (0.06, CEIL)]
    for lab, msk in conds:
        if lab.startswith("push"):
            tc = cc[msk][np.arange(msk.sum()), y[msk]]; top = (cc[msk].argmax(1) == y[msk]).mean()
        else:
            tc = cc[msk].max(1); top = float("nan")
        say("      %-11s %5.2f %5.2f %5.2f · %.2f %.2f %.2f · %.3f"
            % (lab, *np.percentile(m[msk], [5, 50, 95]), *np.percentile(tc, [5, 50, 95]), top))
    say("\n%.0f s" % (time.time() - t0))
    Path(__file__).with_name("j3_design_check_output.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
