"""J3 power check (before registration): planted agents on the flight's shape.

Flight shape per state (258 states, cross-fitted halves): sham · random push at
the ceiling · random push at one threshold dose (cycled) · 2 concept pushes at
each of 0.04/0.06/0.09/0.15 · 1 concept push at 0.5. Three calls per stimulus.
Agents (probabilities rounded to 0.01, as Jev returns them):
  twin      the fitted bar's own distribution, jittered (Dirichlet, conc. 300)
  degraded  the fitted bar applied to noisier digests (sd 0.12 on similarities,
            12% on the shift) -> should land near the P3 margin
  claimer   names the top-similarity pattern always, confidence 0.9 (E7-Q's pathology)
  toplooker names the top pattern if the shift exceeds 6, else none; confidence 0.9
Reads each primary as registered (cluster bootstrap by state, B resamples).
Output: j3_power_output.txt
"""
import time
from pathlib import Path

import numpy as np

from jev.power.j3_design_check import (NONE, aurc, auroc, conditions, dedupe_and_halves, digest,
                                        ece, feats, fit_bar, load, make_reg, mixture_weights, predict,
                                        reference)

THR, PSY, CEIL = [0.04, 0.06, 0.09], [0.15], 0.5
B = 1000


fset = feats          # the registered fitted-bar features (functions of the displayed digest)


def flight_like(rng, U, reg, D, mu, ti):
    """one stimulus set on test states ti with the flight recipe."""
    X, y, d, kind, st = [], [], [], [], []
    for j, s in enumerate(ti):
        u = U[s][None]
        add = lambda x, yy, dd, kk: (X.append(x), y.append(yy), d.append(dd), kind.append(kk), st.append(s))
        add(reg(u)[0], NONE, 0.0, "sham")
        for a in (CEIL, THR[j % 3]):
            r = rng.normal(size=u.shape); r /= np.linalg.norm(r)
            add(reg(u + a * mu * r)[0], NONE, a, "rand")
        ks = rng.permutation(13)
        for i, a in enumerate(THR + PSY):
            for t in range(2):
                k = int(ks[(2 * i + t) % 13]); add(reg(u + a * mu * D[k])[0], k, a, "inj")
        k = int(rng.integers(13)); add(reg(u + CEIL * mu * D[k])[0], k, CEIL, "inj")
    return np.array(X), np.array(y), np.array(d), np.array(kind), np.array(st)


def round01(P):
    Q = np.round(P, 2)
    return Q


def boot(stat, st, rng, B=B):
    us = np.unique(st); idx = {s: np.where(st == s)[0] for s in us}
    out = []
    for _ in range(B):
        pick = rng.choice(us, len(us), replace=True)
        out.append(stat(np.concatenate([idx[s] for s in pick])))
    return np.array(out)


def verdict(Pj, cfj, P_fit, y, d, kind, st, rng):
    """Pj: agent probability rows (calls), P_fit: fitted rows (same calls)."""
    ch = Pj.argmax(1); cor = ch == y
    chf = P_fit.argmax(1); corf = chf == y; cff = P_fit.max(1)
    pool = (kind == "sham") | ((kind == "rand") & np.isin(d, THR)) | ((kind == "inj") & np.isin(d, THR))
    sh = kind == "sham"; hit = (kind == "inj") & np.isin(d, THR)
    rc = (kind == "rand") & (d == CEIL); ic = (kind == "inj") & (d == CEIL)
    gate = cor[ic].mean()

    def excess(ix):
        m = np.zeros(len(y), bool); m[ix] = True
        # bootstrap resamples may repeat states; use index multiset
        s_sh, s_hit = ix[sh[ix]], ix[hit[ix]]
        H = (ch[s_hit] != NONE).mean(); FA = (ch[s_sh] != NONE).mean()
        t = np.quantile(1 - P_fit[s_hit, NONE], 1 - H) if H > 0 else np.inf
        return FA - ((1 - P_fit[s_sh, NONE]) >= t).mean()
    allix = np.arange(len(y))
    ex = excess(allix); exb = boot(excess, st, rng)
    rn = (ch[rc] != NONE).mean(); rnb = boot(lambda ix: (ch[ix[rc[ix]]] != NONE).mean(), st, rng)
    pl = np.where(pool)[0]
    e = ece(cfj[pool], cor[pool]); eb = boot(lambda ix: ece(cfj[ix[pool[ix]]], cor[ix[pool[ix]]]), st, rng)
    a2 = auroc(cfj[pool], cor[pool].astype(int))
    a2b = boot(lambda ix: auroc(cfj[ix[pool[ix]]], cor[ix[pool[ix]]].astype(int)), st, rng)
    da = aurc(cfj[pool], cor[pool]) - aurc(cff[pool], corf[pool])
    dab = boot(lambda ix: aurc(cfj[ix[pool[ix]]], cor[ix[pool[ix]]]) - aurc(cff[ix[pool[ix]]], corf[ix[pool[ix]]]), st, rng)
    q = lambda b: np.percentile(b, [2.5, 97.5])
    return dict(gate=gate, sham=(ch[sh] != NONE).mean(), excess=ex, excess_ci=q(exb), rand_ceil=rn, rand_ci=q(rnb),
                ece=e, ece_ci=q(eb), auroc2=a2, auroc2_p=(a2b <= 0.5).mean(), daurc=da, daurc_ci=q(dab),
                acc_pool=cor[pool].mean())


def main():
    t0 = time.time(); out = []
    say = lambda s="": (print(s), out.append(s))
    W, cent, S, stored, meta, names, D, mu = load()
    reg = make_reg(W, cent)
    order, half, _ = dedupe_and_halves(S, meta)
    U = S[order]
    rng = np.random.default_rng([20260926, 3, 2])
    parts = []
    for fit_h in (0, 1):
        fi = np.where(half == fit_h)[0]; ti = np.where(half != fit_h)[0]
        ref = reference(reg, U[fi], D, mu)
        perm = rng.permutation(fi); cal, tr = perm[: len(fi) // 4], perm[len(fi) // 4:]
        doses = THR + PSY + [CEIL]
        Xtr, ytr, dtr, ktr, _ = conditions(tr, U, reg, D, mu, rng, doses, n_rand=2)
        Xc, yc, dc, kc, _ = conditions(cal, U, reg, D, mu, rng, doses, n_rand=2)
        model = fit_bar(fset(Xtr, ref), ytr, mixture_weights(ytr, dtr, ktr, THR, PSY, CEIL),
                        fset(Xc, ref), yc, mixture_weights(yc, dc, kc, THR, PSY, CEIL))
        parts.append((ref, model, ti))
    say("J3 power check — %s · B=%d cluster-bootstrap resamples by state" % (time.strftime("%Y-%m-%d %H:%M"), B))
    for sim in range(3):
        r = np.random.default_rng([20260926, 3, 3, sim])
        X, y, d, kind, st, Pf, Fz = [], [], [], [], [], [], []
        for ref, model, ti in parts:
            x, yy, dd, kk, ss = flight_like(r, U, reg, D, mu, ti)
            X.append(x); y.append(yy); d.append(dd); kind.append(kk); st.append(ss)
            Pf.append(predict(model, fset(x, ref))); Fz.append((x, ref, model))
        y, d, kind, st = map(np.concatenate, (y, d, kind, st)); Pf = np.vstack(Pf)
        # three calls per stimulus
        rep = lambda a: np.concatenate([a, a, a])
        y3, d3, k3, s3, Pf3 = rep(y), rep(d), rep(kind), rep(st), np.vstack([Pf, Pf, Pf])
        agents = {}
        tw = np.vstack([np.vstack([r.dirichlet(300 * Pf[i] + 1e-3) for i in range(len(Pf))]) for _ in range(3)])
        agents["twin"] = round01(tw)
        deg = []
        for _ in range(3):
            rows = []
            for x, ref, model in Fz:
                z, m = digest(x, ref); c = z / m[:, None]
                c = np.clip(c + r.normal(0, 0.12, c.shape), -1, 1); m = m * np.exp(r.normal(0, 0.12, m.shape))
                F = np.hstack([c, c * m[:, None], m[:, None], np.log(m)[:, None]])
                rows.append(predict(model, F))
            deg.append(np.vstack(rows))
        agents["degraded"] = round01(np.vstack(deg))
        zc = np.vstack([digest(x, ref)[0] / digest(x, ref)[1][:, None] for x, ref, model in Fz])
        mm = np.concatenate([digest(x, ref)[1] for x, ref, model in Fz])
        cl = np.full((len(zc), 14), 0.1 / 13); cl[np.arange(len(zc)), zc.argmax(1)] = 0.9
        agents["claimer"] = np.vstack([cl, cl, cl])
        tl = np.full((len(zc), 14), 0.1 / 13)
        pick = np.where(mm > 6, zc.argmax(1), NONE); tl[np.arange(len(zc)), pick] = 0.9
        agents["toplooker"] = np.vstack([tl, tl, tl])
        say("\nsimulation %d (stimuli %d, calls %d)" % (sim, len(y), len(y3)))
        for name, Pj in agents.items():
            v = verdict(Pj, Pj.max(1), Pf3, y3, d3, k3, s3, np.random.default_rng([sim, 99]))
            say("  %-9s gate %.3f · sham named %.3f · P1a excess %+.3f [%+.3f, %+.3f] · P1b random@ceil %.3f [%.3f, %.3f]"
                % (name, v["gate"], v["sham"], v["excess"], *v["excess_ci"], v["rand_ceil"], *v["rand_ci"]))
            say("            P2 ECE %.3f [%.3f, %.3f] AUROC2 %.3f (boot p %.4f) · P3 dAURC %+.4f [%+.4f, %+.4f] · pool acc %.3f"
                % (v["ece"], *v["ece_ci"], v["auroc2"], v["auroc2_p"], v["daurc"], *v["daurc_ci"], v["acc_pool"]))
    say("\n%.0f s" % (time.time() - t0))
    Path(__file__).with_name("j3_power_output.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
