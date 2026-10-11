"""W3 analysis: registration §5 (docs/W3_LOOP_PROTOCOL.md) on one harvest directory.

    python -m windows.analyze_w3 <dir> [--smoke]

Reads only the harvest format of windows/w3_spec.py: the array sets through `load()` (which
checks every part's sha256 and raises on a tampered part), gate_{on,drop,off}.json and gdet.json
with a boolean "pass", and train_*.json when present. Writes verdict.json
(json.dumps(indent=1, sort_keys=True); python numbers, floats rounded to 10 significant digits,
NaN as null) and report.md into the directory.

Deterministic. Every statistic draws its bootstrap from its own generator,
default_rng([ANALYSIS seed (+100 smoke), crc32(name)]), so a fresh process reproduces
verdict.json byte for byte, and computing only some sections (`parts`, a test hook) changes no
reported number. A name used twice is an error.

The bootstrap resamples images: a multinomial counts matrix of BOOT rows over the images, each
statistic a ratio of per-image sums, 95% percentile intervals, as in
windows/power/w3_collinear_check.py. The probe statistic S2 is w3_split_check's split_fit /
final_verdict, imported (not copied); see _load_split_check for how torch is kept out.

Gating (registration §1.5, §4, §5.1): P1 and P2 are NOT ADJUDICABLE when M-on failed G-train,
G-det failed or was not run, or a file they need is missing. Activations are read only from
models that passed their gate, so a secondary on a model whose gate failed, or any secondary
when G-det failed, is reported "not adjudicable" (the untrained twins have no gate; S1-S5 and
S9 follow M-on's adjudicability), and a secondary whose sets are absent is reported "missing".
"""
import argparse
import importlib.util
import json
import sys
import types
import zlib
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from . import w3_spec as S

NA = "NOT ADJUDICABLE"
JUDGED_P1 = ("LABEL", "SPLIT", "VARIES BY LEVEL")      # §5.2: P2 is judged only on these
SECTIONS = ("P1", "P2", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10")
SETS = ("on_split", "drop_split", "on0_split", "on_sham", "drop_sham",
        "on_ptrain", "on_pcal", "drop_ptrain", "drop_pcal", "drop_ptrain_null", "drop_pcal_null",
        "off_ptrain", "off_pcal", "on0_ptrain", "off0_ptrain", "off0_pcal", "on_spread")
JSONS = ("gate_on", "gate_drop", "gate_off", "gdet", "train_on", "train_drop", "train_off")
CHUNK_BYTES = 16_000_000          # bootstrap counts are drawn in chunks (same stream as one draw)
SIG = 10                          # significant digits kept in verdict.json
LOGSNR = S.LOGSNR
_NEG_LOGSNR = -LOGSNR             # LOGSNR falls with t; its negative rises (for searchsorted)
F64 = np.float64


def _load_split_check():
    """windows/power/w3_split_check.py, for split_fit, decisive and final_verdict. Its only
    import, `from w3_design_check import T, alpha_bar`, would pull torch into the analysis (on a
    3.7 GB machine); w3_spec carries both verbatim, so a stub module serves them for the length
    of the import and is removed after. The script's sys.path insert is undone too."""
    path = Path(__file__).resolve().parent / "power" / "w3_split_check.py"
    stub = types.ModuleType("w3_design_check")
    stub.T, stub.alpha_bar, stub.ALPHAS = S.T, S.alpha_bar, S.ALPHAS
    saved, had = list(sys.path), "w3_design_check" in sys.modules
    if not had:
        sys.modules["w3_design_check"] = stub
    try:
        spec = importlib.util.spec_from_file_location("_w3_split_check", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if not had:
            del sys.modules["w3_design_check"]
        sys.path[:] = saved
    if (mod.LO, mod.HI) != (S.LO, S.HI) or tuple(mod.STEP_BANDS) != tuple(S.STEP_BANDS):
        raise RuntimeError("w3_split_check's bars or bands differ from w3_spec")
    return mod


SC = _load_split_check()


# ---- small pure helpers --------------------------------------------------------------------
def to_step(r):
    """A log-SNR read -> the nearest step of LOGSNR (which falls with t)."""
    r = np.asarray(r, F64)
    j = np.clip(np.searchsorted(_NEG_LOGSNR, -r), 1, S.T - 1)
    left = np.abs(LOGSNR[j - 1] - r) <= np.abs(LOGSNR[j] - r)
    return np.where(left, j - 1, j)


def inv_sd(x):
    """1 / per-feature SD (float64, ddof 0); a feature with SD 0 gets weight 0."""
    sd = np.asarray(x, F64).std(axis=0)
    return np.where(sd > 0, 1.0 / np.where(sd > 0, sd, 1.0), 0.0)


def move_terms(X, inv):
    """Per-image sums of the 2 x 2 (§5.1) from a (n, 4, dim) array in INPUT_ORDER, each feature
    multiplied by `inv`. Direction 1 from (t, t), direction 2 from (t', t')."""
    h = {k: np.asarray(X[:, i], F64) * inv for i, k in enumerate(S.INPUT_ORDER)}
    tt, pp, pt, tp = h["tt"], h["pp"], h["pt"], h["tp"]
    db1, db2 = pp - tt, tt - pp
    inter = pp - pt - tp + tt

    def dot(a, b):
        return (a * b).sum(axis=1)
    return {"c1": dot(pt - tt, db1), "c2": dot(tp - pp, db2),      # content moves
            "l1": dot(tp - tt, db1), "l2": dot(pt - pp, db2),      # label moves
            "b1": dot(db1, db1), "b2": dot(db2, db2), "i2": dot(inter, inter)}


def p1_rules(pooled, bands):
    """§5.1 in order, after the adjudicability check: UNREADABLE, VARIES BY LEVEL, decisive."""
    if not pooled["kappa_hi"] <= S.KAPPA_BAR:                     # NaN fails safe
        return "UNREADABLE"
    calls = [(SC.decisive(b), b) for b in bands if b is not None]
    calls = [(v, b) for v, b in calls if v != "UNCLEAR"]
    for i in range(len(calls)):
        for j in range(i + 1, len(calls)):
            (v1, b1), (v2, b2) = calls[i], calls[j]
            if v1 != v2 and (b1["hi"] < b2["lo"] or b2["hi"] < b1["lo"]) \
                    and abs(b1["w"] - b2["w"]) >= S.DELTA_VARIES:
                return "VARIES BY LEVEL"
    return SC.decisive(pooled)


def ridge_reader(Xtr, ytr):
    """StandardScaler + RidgeCV(alphas=ALPHAS) fitted on (Xtr, ytr); returns (read, alpha)."""
    Xtr = np.asarray(Xtr, F64)
    scl = StandardScaler().fit(Xtr)
    pr = RidgeCV(alphas=S.ALPHAS).fit(scl.transform(Xtr), ytr)
    return (lambda X: pr.predict(scl.transform(np.asarray(X, F64)))), float(pr.alpha_)


def read_metrics(pred, t):
    """R² of the log-SNR read, mean |log-SNR error|, median |step error| overall and by GATE_BANDS."""
    ls = LOGSNR[t]
    err = np.abs(to_step(pred) - t)
    bands = {}
    for lo, hi in S.GATE_BANDS:
        sel = (t >= lo) & (t < hi)
        bands[f"{lo}-{hi}"] = float(np.median(err[sel])) if sel.any() else None
    return {"n": int(len(t)), "r2_logsnr": float(1 - ((ls - pred) ** 2).sum() / ((ls - ls.mean()) ** 2).sum()),
            "mean_abs_logsnr_err": float(np.abs(pred - ls).mean()),
            "median_step_err": float(np.median(err)), "band_median_step_err": bands}


def cv_r2(X, y, seed):
    """5-fold CV R² (pooled out-of-fold predictions) of RidgeCV, standardised within the training fold."""
    oof = np.empty_like(y)
    for tr, te in KFold(S.CV_FOLDS, shuffle=True, random_state=seed).split(X):
        scl = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=S.ALPHAS).fit(scl.transform(X[tr]), y[tr])
        oof[te] = m.predict(scl.transform(X[te]))
    return float(1 - ((y - oof) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def clean(x):
    """Python numbers only, floats to SIG significant digits, NaN/inf -> None."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [clean(v) for v in x.tolist()]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return float(f"{x:.{SIG}g}") if np.isfinite(x) else None
    return x


def dumps(v):
    return json.dumps(v, indent=1, sort_keys=True) + "\n"


def _status(state, **kw):
    return {"status": state, **kw}


# ---- the run -------------------------------------------------------------------------------
class _Run:
    def __init__(self, indir, smoke, parts):
        self.dir = Path(indir)
        self.smoke = bool(smoke)
        self.seed = S.seeds(self.smoke)["ANALYSIS"]
        self.spread_seed = S.seeds(self.smoke)["SPREAD"]
        self.B = S.sizes(self.smoke)["BOOT"]
        self.parts = set(SECTIONS if parts is None else parts)
        bad = self.parts - set(SECTIONS)
        if bad:
            raise ValueError(f"unknown sections {sorted(bad)}")
        self._sets, self._names = {}, set()
        man = self.dir / "manifest.json"
        self.manifest = json.loads(man.read_text()) if man.exists() else []
        self.json = {}
        for k in JSONS:
            f = self.dir / f"{k}.json"
            self.json[k] = json.loads(f.read_text()) if f.exists() else None

    # sets ---------------------------------------------------------------------------------
    def has(self, name):
        return bool(self.manifest) and S.has(name, self.dir)

    def get(self, name):
        if name not in self._sets:
            self._sets[name] = S.load(name, self.dir) if self.has(name) else None
        return self._sets[name]

    def missing(self, *names):
        return [n for n in names if not self.has(n)]

    def passed(self, k):
        j = self.json.get(k)
        return j is not None and j.get("pass") is True

    def gate_reasons(self, model, need_gdet=False):
        """Why a model's activations may not be read. A failed G-det blocks everything; a
        missing gdet.json blocks only what needs G-det (P1, P2): the harvest runs G-det on M-on
        and only when M-on passes its gate, so M-drop, M-off and the twins stay readable."""
        r = []
        if self.json["gdet"] is None:
            if need_gdet:
                r.append("gdet.json missing")
        elif not self.passed("gdet"):
            r.append("G-det failed")
        if model is not None:
            k = f"gate_{model}"
            if self.json[k] is None:
                r.append(f"{k}.json missing")
            elif not self.passed(k):
                r.append(f"M-{model} failed G-train")
        return r

    # bootstrap ----------------------------------------------------------------------------
    def counts(self, name, m):
        """Chunks of the BOOT x m multinomial counts matrix from this statistic's generator."""
        if name in self._names:
            raise RuntimeError(f"bootstrap generator {name!r} used twice")
        self._names.add(name)
        rng = np.random.default_rng([self.seed, zlib.crc32(name.encode())])
        p, rows, left = np.full(m, 1.0 / m), max(1, CHUNK_BYTES // (8 * m)), self.B
        while left > 0:
            k = min(rows, left)
            left -= k
            yield rng.multinomial(m, p, size=k).astype(F64)

    def ratios(self, name, m, pairs):
        """Bootstrap ratios of sums, sum(C num) / sum(C den), for each (num, den), one C."""
        outs = [[] for _ in pairs]
        with np.errstate(divide="ignore", invalid="ignore"):
            for C in self.counts(name, m):
                for o, (num, den) in zip(outs, pairs):
                    o.append((C @ num) / (C @ den))
        return [np.concatenate(o) for o in outs]

    def full_counts(self, name, m):
        return np.concatenate(list(self.counts(name, m)))

    # statistics ---------------------------------------------------------------------------
    def movement(self, name, terms, sel=None):
        t = terms if sel is None else {k: v[sel] for k, v in terms.items()}
        m = len(t["b1"])
        a_c, a_l, a_b = t["c1"] + t["c2"], t["l1"] + t["l2"], t["b1"] + t["b2"]
        den = a_b.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            w, lab = a_c.sum() / den, a_l.sum() / den
            kappa = t["i2"].sum() / t["b1"].sum()
        if den > 0 and not abs(w + lab - 1) < 1e-9:
            raise AssertionError(f"{name}: w + l = {w + lab!r}, not 1 (the Shapley identity)")
        wb, kb = self.ratios(name, m, [(a_c, a_b), (t["i2"], t["b1"])])
        lo, hi = np.percentile(wb, [2.5, 97.5])
        return {"n": m, "w": w, "l": lab, "lo": lo, "hi": hi, "kappa": kappa,
                "kappa_hi": np.percentile(kb, 97.5)}

    def move_block(self, name, X, inv, t):
        """w, l, interval, kappa, kappa upper, the four step bands and the P1 verdict rules."""
        terms = move_terms(X, inv)
        pooled = self.movement(f"{name}/all", terms)
        bands, rows = [], []
        for k, (lo, hi) in enumerate(S.STEP_BANDS):
            sel = (t >= lo) & (t < hi)
            n = int(sel.sum())
            b = self.movement(f"{name}/band{k}", terms, sel) if n >= 8 else None
            bands.append(b)
            row = {"steps": [lo, hi], "n": n, "skipped": b is None}
            if b is not None:
                row.update(b, verdict=SC.decisive(b))
            rows.append(row)
        return {**pooled, "verdict": p1_rules(pooled, bands), "bands": rows}, terms

    def sham(self, name, X, sham_rows, inv, off):
        """S per offset class (§5.2): sum |<d_sham, b>| / sum |<d_lab, b>|, b = db1/|db1|."""
        tt, pp, tp = (np.asarray(X[:, i], F64) * inv for i in (0, 1, 3))
        sh = np.asarray(sham_rows, F64) * inv
        db1 = pp - tt
        nb = np.linalg.norm(db1, axis=1)
        b = db1 / np.where(nb > 0, nb, 1.0)[:, None]
        d_lab, d_sham = tp - tt, sh - tt
        a_s, a_l = np.abs((d_sham * b).sum(1)), np.abs((d_lab * b).sum(1))
        f_s, f_l = np.linalg.norm(d_sham, axis=1), np.linalg.norm(d_lab, axis=1)
        out = {}
        for o in S.OFFSETS:
            sel = off == o
            m = int(sel.sum())
            if m == 0:
                out[str(o)] = {"n": 0}
                continue
            (sb,) = self.ratios(f"{name}/off{o}", m, [(a_s[sel], a_l[sel])])
            lo, hi = np.percentile(sb, [2.5, 97.5])
            with np.errstate(divide="ignore", invalid="ignore"):
                out[str(o)] = {"n": m, "S": a_s[sel].sum() / a_l[sel].sum(), "lo": lo, "hi": hi,
                               "full_ratio": f_s[sel].sum() / f_l[sel].sum()}
        return out

    def probe_stat(self, name, site, ptrain, pcal, split):
        """S2 at one site: w3_split_check.experiment's probe statistic on harvested reads."""
        read, alpha = ridge_reader(ptrain[site], LOGSNR[ptrain["t"]])
        rc = read(pcal[site])
        g = IsotonicRegression(out_of_bounds="clip").fit(LOGSNR[pcal["t"]], rc).predict
        t = split["t"]
        s_t, s_p = LOGSNR[t], LOGSNR[split["tp"]]
        H = np.asarray(split[site], F64)
        m = len(t)
        r_tt, r_pp = read(H[:, 0]), read(H[:, 1])
        sd_matched = np.std(np.concatenate([r_tt - g(s_t), r_pp - g(s_p)]))
        scn, sln = np.concatenate([s_p, s_t]), np.concatenate([s_t, s_p])
        r = np.concatenate([read(H[:, 2]), read(H[:, 3])])       # (t', t) then (t, t')
        img = np.concatenate([np.arange(m), np.arange(m)])
        gcn, gln = g(scn), g(sln)
        y, d = r - gln, gcn - gln
        pooled = SC.split_fit(y, d, r, gcn, gln, img, m, self.full_counts(f"{name}/all", m), sd_matched)
        bands, rows = [], []
        for k, (lo, hi) in enumerate(S.STEP_BANDS):
            sel_img = (t >= lo) & (t < hi)
            row = {"steps": [lo, hi], "n": int(sel_img.sum())}
            if sel_img.sum() < 8:
                bands.append(None)
                rows.append({**row, "skipped": True})
                continue
            sel = np.concatenate([sel_img, sel_img])
            keep = np.flatnonzero(sel_img)
            remap = -np.ones(m, int)
            remap[keep] = np.arange(len(keep))
            fb = SC.split_fit(y[sel], d[sel], r[sel], gcn[sel], gln[sel], remap[img[sel]], len(keep),
                              self.full_counts(f"{name}/band{k}", len(keep)), sd_matched)
            bands.append(fb)
            rows.append({**row, "skipped": False, **fb, "verdict": SC.decisive(fb)})
        v = SC.final_verdict(pooled, bands)
        ls_c = LOGSNR[pcal["t"]]
        return {**pooled, "verdict": "VARIES BY LEVEL" if v == "VARIES" else v, "bands": rows,
                "alpha": alpha, "sd_matched": sd_matched,
                "r2_pcal": float(1 - ((ls_c - rc) ** 2).sum() / ((ls_c - ls_c.mean()) ** 2).sum())}

    def floor_diff(self, name, pred_net, pred_ref, t):
        """Paired bootstrap of mean |err_ref| - mean |err_net| in log-SNR; beats = lower > 0."""
        ls = LOGSNR[t]
        d = np.abs(pred_ref - ls) - np.abs(pred_net - ls)
        (db,) = self.ratios(name, len(d), [(d, np.ones_like(d))])
        lo, hi = np.percentile(db, [2.5, 97.5])
        return {"diff": d.mean(), "lo": lo, "hi": hi, "beats": bool(lo > 0)}

    # sections -----------------------------------------------------------------------------
    def run(self):
        v = {"stage": {"smoke": self.smoke, "boot": self.B, "analysis_seed": self.seed,
                       "spread_seed": self.spread_seed, "parts": sorted(self.parts),
                       "bars": {"LO": S.LO, "HI": S.HI, "KAPPA_BAR": S.KAPPA_BAR,
                                "DELTA_VARIES": S.DELTA_VARIES, "SHAM_BAR": S.SHAM_BAR,
                                "SHAM_CLASS": S.SHAM_CLASS, "STEP_BANDS": S.STEP_BANDS}},
             "gates": {k: (None if self.json[k] is None else self.passed(k))
                       for k in ("gate_on", "gate_drop", "gate_off", "gdet")},
             "sets": self.set_table()}
        sec = {}
        # M-on, mid-block: P1, and the terms P2/S4/S5/S6 reuse
        on_reasons = self.gate_reasons("on", need_gdet=True) + [f"{n} missing" for n in self.missing("on_split", "on_ptrain")]
        self.on_ok = not on_reasons
        self.on_mid = None
        if self.on_ok:
            self.on = self.get("on_split")
            self.on_pt = self.get("on_ptrain")
            self.on_inv = {s: inv_sd(self.on_pt[s]) for s in S.SITES}
            blk, terms = self.move_block(f"move/on/{S.PRIMARY_SITE}", self.on[S.PRIMARY_SITE],
                                         self.on_inv[S.PRIMARY_SITE], self.on["t"])
            self.on_mid = (blk, terms)
            v["P1"] = {"site": S.PRIMARY_SITE, **blk}
        else:
            v["P1"] = {"site": S.PRIMARY_SITE, "verdict": NA, "reasons": on_reasons}
        v["P2"] = self.p2(v["P1"]) if "P2" in self.parts else _status("not computed")
        if not self.on_ok:
            na = _status("not adjudicable", reasons=on_reasons)
        for key, fn in (("S1_profile", self.s1), ("S2_probe", self.s2), ("S3_output", self.s3),
                        ("S4_directions", self.s4), ("S5_offsets", self.s5)):
            if key[:2] not in self.parts:
                continue
            sec[key] = fn() if self.on_ok else na
        if "S6" in self.parts:
            sec["S6_drop"] = self.s6()
        if "S7" in self.parts:
            sec["S7_off"] = self.s7()
        if "S8" in self.parts:
            sec["S8_untrained"] = self.s8()
        if "S9" in self.parts:
            sec["S9_spread"] = self.s9() if self.on_ok else na
        if "S10" in self.parts:
            sec["S10_record"] = {k: self.json[k] for k in JSONS if self.json[k] is not None}
        v["secondaries"] = sec
        return clean(v)

    def set_table(self):
        out = {}
        for n in SETS:
            lines = [x for x in self.manifest if x["set"] == n]
            out[n] = {"present": bool(lines),
                      "rows": max((x["rows"][1] for x in lines), default=0), "parts": len(lines)}
        return out

    def p2(self, p1):
        if p1["verdict"] == NA:
            return {"verdict": NA, "reasons": p1["reasons"]}
        if not self.has("on_sham"):
            return {"verdict": NA, "reasons": ["on_sham missing"]}
        sham = self.get("on_sham")
        if len(sham[S.PRIMARY_SITE]) != len(self.on["t"]) or \
                ("img" in sham and not np.array_equal(sham["img"], self.on["img"])):
            return {"verdict": NA, "reasons": ["on_sham rows do not match on_split"]}
        by = self.sham("sham/on", self.on[S.PRIMARY_SITE], sham[S.PRIMARY_SITE],
                       self.on_inv[S.PRIMARY_SITE], self.on["off"])
        main = by[str(S.SHAM_CLASS)]
        out = {"site": S.PRIMARY_SITE, "class": S.SHAM_CLASS, "by_offset": by,
               "S": main.get("S"), "upper": main.get("hi")}
        if p1["verdict"] not in JUDGED_P1:
            out["verdict"] = "not judged"
        elif main["n"] == 0:
            out.update(verdict=NA, reasons=[f"no {S.SHAM_CLASS}-step images"])
        else:
            out["verdict"] = "SILENT" if main["hi"] <= S.SHAM_BAR else "NOT SILENT"
        return out

    def s1(self):
        out = {}
        for s in S.SITES:
            if s == S.PRIMARY_SITE:
                out[s] = self.on_mid[0]
            else:
                out[s] = self.move_block(f"move/on/{s}", self.on[s], self.on_inv[s], self.on["t"])[0]
        return out

    def s2(self):
        miss = self.missing("on_pcal")
        if miss:
            return _status("missing", sets=miss)
        pcal, out = self.get("on_pcal"), {}
        for s in S.SITES:
            try:
                out[s] = self.probe_stat(f"probe/on/{s}", s, self.on_pt, pcal, self.on)
            except np.linalg.LinAlgError as e:
                out[s] = {"verdict": "UNREADABLE", "error": f"singular fit: {e}"}
        return out

    def s3(self):
        blk, _ = self.move_block("move/on/out", self.on["out"], inv_sd(self.on_pt["out"]), self.on["t"])
        return blk

    def s4(self):
        t = self.on_mid[1]
        out = {}
        for k, (num, den) in (("dir1", ("c1", "b1")), ("dir2", ("c2", "b2"))):
            (wb,) = self.ratios(f"S4/{k}", len(t[den]), [(t[num], t[den])])
            lo, hi = np.percentile(wb, [2.5, 97.5])
            out[k] = {"w": t[num].sum() / t[den].sum(), "lo": lo, "hi": hi}
        out["dir1"]["reads"] = "content noisier than the label (from (t, t) to (t', t))"
        out["dir2"]["reads"] = "content cleaner than the label (from (t', t') to (t, t'))"
        return out

    def s5(self):
        terms, off = self.on_mid[1], self.on["off"]
        out = {}
        for o in S.OFFSETS:
            sel = off == o
            if not sel.any():
                out[str(o)] = {"n": 0}
                continue
            r = self.movement(f"move/on/{S.PRIMARY_SITE}/off{o}", terms, sel)
            out[str(o)] = {**r, "verdict": SC.decisive(r) if r["kappa_hi"] <= S.KAPPA_BAR else "UNREADABLE"}
        return out

    def s6(self):
        reasons = self.gate_reasons("drop")
        if reasons:
            return _status("not adjudicable", reasons=reasons)
        out = {"status": "ok"}
        miss = self.missing("drop_split", "drop_ptrain")
        if miss:
            out["w_mid"] = _status("missing", sets=miss)
        else:
            dsp, dpt = self.get("drop_split"), self.get("drop_ptrain")
            inv = inv_sd(dpt[S.PRIMARY_SITE])
            blk, dterms = self.move_block(f"move/drop/{S.PRIMARY_SITE}", dsp[S.PRIMARY_SITE], inv, dsp["t"])
            out["w_mid"] = blk
            if not self.on_ok:
                out["diff_from_on"] = _status("not adjudicable", reasons=["M-on not adjudicable"])
            elif not all(np.array_equal(dsp[k], self.on[k]) for k in ("img", "t", "tp")):
                out["diff_from_on"] = _status("not paired", reasons=["drop_split and on_split differ in images or steps"])
            else:
                ot = self.on_mid[1]
                on_c, on_b = ot["c1"] + ot["c2"], ot["b1"] + ot["b2"]
                dr_c, dr_b = dterms["c1"] + dterms["c2"], dterms["b1"] + dterms["b2"]
                rb_on, rb_dr = self.ratios("S6/diff", len(on_b), [(on_c, on_b), (dr_c, dr_b)])
                lo, hi = np.percentile(rb_dr - rb_on, [2.5, 97.5])
                out["diff_from_on"] = {"diff": dr_c.sum() / dr_b.sum() - on_c.sum() / on_b.sum(),
                                       "lo": lo, "hi": hi, "reads": "w(M-drop) - w(M-on), paired over images"}
            if self.has("drop_sham"):
                sh = self.get("drop_sham")
                if len(sh[S.PRIMARY_SITE]) != len(dsp["t"]) or \
                        ("img" in sh and not np.array_equal(sh["img"], dsp["img"])):
                    out["sham"] = _status("misaligned")
                else:
                    by = self.sham("sham/drop", dsp[S.PRIMARY_SITE], sh[S.PRIMARY_SITE], inv, dsp["off"])
                    main = by[str(S.SHAM_CLASS)]
                    out["sham"] = {"by_offset": by, "S": main.get("S"), "upper": main.get("hi"),
                                   "upper_within_bar": None if main["n"] == 0 else bool(main["hi"] <= S.SHAM_BAR)}
            else:
                out["sham"] = _status("missing", sets=["drop_sham"])
        miss = self.missing("drop_ptrain_null", "drop_pcal_null")
        if miss:
            out["no_label_probe"] = _status("missing", sets=miss)
        else:
            out["no_label_probe"] = self.probe_vs_floor("S6", self.get("drop_ptrain_null"),
                                                        self.get("drop_pcal_null"), None, None)
        return out

    def probe_vs_floor(self, tag, ptr, pcal, ptr0, pcal0):
        """Ridge reads per site (ptrain -> pcal) against the hand statistics on the same images,
        and (S7) against the untrained twin at the same site."""
        t = pcal["t"]
        hand_read, hand_alpha = ridge_reader(ptr["hand"], LOGSNR[ptr["t"]])
        hand_pred = hand_read(pcal["hand"])
        out = {"hand": {**read_metrics(hand_pred, t), "alpha": hand_alpha}, "sites": {}}
        paired0 = ptr0 is not None and all(np.array_equal(pcal0[k], pcal[k]) for k in ("img", "t"))
        if ptr0 is not None:
            out["twin_paired"] = paired0
        for s in S.SITES:
            read, alpha = ridge_reader(ptr[s], LOGSNR[ptr["t"]])
            pred = read(pcal[s])
            row = {"net": {**read_metrics(pred, t), "alpha": alpha},
                   "vs_hand": self.floor_diff(f"{tag}/{s}/vs_hand", pred, hand_pred, t)}
            if ptr0 is not None:
                read0, alpha0 = ridge_reader(ptr0[s], LOGSNR[ptr0["t"]])
                pred0 = read0(pcal0[s])
                row["twin"] = {**read_metrics(pred0, pcal0["t"]), "alpha": alpha0}
                row["vs_twin"] = (self.floor_diff(f"{tag}/{s}/vs_twin", pred, pred0, t) if paired0
                                  else _status("not paired"))
            out["sites"][s] = row
        out["beats_hand_sites"] = [s for s in S.SITES if out["sites"][s]["vs_hand"]["beats"]]
        if ptr0 is not None and paired0:
            out["beats_twin_sites"] = [s for s in S.SITES if out["sites"][s]["vs_twin"]["beats"]]
        return out

    def s7(self):
        reasons = self.gate_reasons("off")
        if reasons:
            return _status("not adjudicable", reasons=reasons)
        miss = self.missing("off_ptrain", "off_pcal")
        if miss:
            return _status("missing", sets=miss)
        miss0 = self.missing("off0_ptrain", "off0_pcal")
        r = self.probe_vs_floor("S7", self.get("off_ptrain"), self.get("off_pcal"),
                                None if miss0 else self.get("off0_ptrain"),
                                None if miss0 else self.get("off0_pcal"))
        if miss0:
            r["twin"] = _status("missing", sets=miss0)
        return {"status": "ok", **r}

    def s8(self):
        reasons = self.gate_reasons(None)
        if reasons:
            return _status("not adjudicable", reasons=reasons)
        miss = self.missing("on0_split", "on0_ptrain")
        if miss:
            return _status("missing", sets=miss)
        sp, pt = self.get("on0_split"), self.get("on0_ptrain")
        out = {"status": "ok", "sites": {}}
        for s in S.SITES:
            out["sites"][s] = self.move_block(f"move/on0/{s}", sp[s], inv_sd(pt[s]), sp["t"])[0]
        return out

    def s9(self):
        miss = self.missing("on_spread")
        if miss:
            return _status("missing", sets=miss)
        sp = self.get("on_spread")
        y_all = np.asarray(sp["logvar"], F64)
        Xa_all = np.concatenate([np.asarray(sp["hand_xt"], F64), np.asarray(sp["hand_x0"], F64)], axis=1)
        Xb_all = np.asarray(sp["mid"], F64)
        out = {"status": "ok", "bands": {}}
        for b in np.unique(sp["band"]):
            sel = sp["band"] == b
            n = int(sel.sum())
            row = {"n": n, "t_range": [int(sp["t"][sel].min()), int(sp["t"][sel].max())]}
            if n < 2 * S.CV_FOLDS:
                out["bands"][str(int(b))] = {**row, "skipped": True}
                continue
            y, Xa, Xb = y_all[sel], Xa_all[sel], Xb_all[sel]
            ra = cv_r2(Xa, y, self.spread_seed)
            rb = cv_r2(Xb, y, self.spread_seed)
            rc = cv_r2(np.concatenate([Xa, Xb], axis=1), y, self.spread_seed)
            out["bands"][str(int(b))] = {**row, "skipped": False, "r2_hand": ra, "r2_mid": rb,
                                         "r2_both": rc, "increment": rc - ra}
        return out


def analyze(indir, smoke=False, *, parts=None):
    """The registration's §5 on harvest directory `indir`. `parts` (a test hook) limits the
    sections computed; no reported number depends on it."""
    return _Run(indir, smoke, parts).run()


# ---- report --------------------------------------------------------------------------------
def _f(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _iv(d, nd=2):
    return f"{_f(d.get('w'), nd)} [{_f(d.get('lo'), nd)}, {_f(d.get('hi'), nd)}]"


def _sec_state(d):
    if isinstance(d, dict) and d.get("status") not in (None, "ok"):
        extra = d.get("reasons") or d.get("sets") or []
        return f"{d['status']}" + (f" ({', '.join(extra)})" if extra else "")
    return None


def report(v):
    st = v["stage"]
    L = ["# W3 analysis report", "",
         f"{'Smoke' if st['smoke'] else 'Flight'} analysis; bootstrap {st['boot']} image resamples, "
         f"analysis seed {st['analysis_seed']}. Gates: " +
         ", ".join(f"{k} {'pass' if g else ('missing' if g is None else 'FAIL')}" for k, g in v["gates"].items()) + ".",
         "", "| measure | result | verdict |", "|---|---|---|"]
    p1 = v["P1"]
    if p1["verdict"] == NA:
        L.append(f"| P1 movement share (M-on, mid) | {'; '.join(p1['reasons'])} | {NA} |")
    else:
        bands = "; ".join(f"{b['steps'][0]}–{b['steps'][1]}: " +
                          ("skipped" if b["skipped"] else f"{_f(b['w'])} {b['verdict']}") for b in p1["bands"])
        L.append(f"| P1 movement share (M-on, mid) | w {_iv(p1)}, label share {_f(p1['l'])}, "
                 f"κ {_f(p1['kappa'], 3)} (upper {_f(p1['kappa_hi'], 3)}); bands {bands} | **{p1['verdict']}** |")
    p2 = v["P2"]
    if "by_offset" in p2:
        cls = "; ".join(f"{o}: S {_f(c.get('S'))} (upper {_f(c.get('hi'))}, full-length {_f(c.get('full_ratio'))})"
                        for o, c in p2["by_offset"].items())
        L.append(f"| P2 label sham ({S.SHAM_CLASS}-step class) | {cls} | **{p2['verdict']}** |")
    else:
        L.append(f"| P2 label sham | {'; '.join(p2.get('reasons', [p2.get('status', '')]))} | "
                 f"**{p2.get('verdict', p2.get('status'))}** |")
    sec = v["secondaries"]

    def row(name, d, body):
        s = _sec_state(d)
        L.append(f"| {name} | {s if s else body()} | |")
    if "S1_profile" in sec:
        row("S1 profile (M-on, all sites)", sec["S1_profile"],
            lambda: "; ".join(f"{s} {_f(d['w'])} {d['verdict']}" for s, d in sec["S1_profile"].items()))
    if "S2_probe" in sec:
        row("S2 probe read (M-on)", sec["S2_probe"],
            lambda: "; ".join(f"{s} {_iv(d)} {d['verdict']}" for s, d in sec["S2_probe"].items()))
    if "S3_output" in sec:
        row("S3 output (hand stats of ε̂)", sec["S3_output"],
            lambda: f"w {_iv(sec['S3_output'])}, κ {_f(sec['S3_output']['kappa'], 3)} → {sec['S3_output']['verdict']}")
    if "S4_directions" in sec:
        row("S4 directions (content share)", sec["S4_directions"],
            lambda: f"dir 1 {_iv(sec['S4_directions']['dir1'])}; dir 2 {_iv(sec['S4_directions']['dir2'])}")
    if "S5_offsets" in sec:
        row("S5 offsets (w by class)", sec["S5_offsets"],
            lambda: "; ".join(f"{o}: {_iv(d)}" for o, d in sec["S5_offsets"].items() if d.get("n")))
    if "S6_drop" in sec:
        def s6():
            d, parts = sec["S6_drop"], []
            w = d["w_mid"]
            parts.append(_sec_state(w) or f"w {_iv(w)} {w['verdict']}")
            df = d.get("diff_from_on", {})
            if df:
                parts.append(_sec_state(df) or f"M-drop − M-on {_iv({'w': df['diff'], 'lo': df['lo'], 'hi': df['hi']})}")
            sh = d.get("sham", {})
            if sh:
                parts.append(_sec_state(sh) or f"sham S {_f(sh.get('S'))} (upper {_f(sh.get('upper'))})")
            nl = d["no_label_probe"]
            parts.append(_sec_state(nl) or ("no-label probe beats the hand floor at: " +
                                            (", ".join(nl["beats_hand_sites"]) or "no site")))
            return "; ".join(parts)
        row("S6 M-drop", sec["S6_drop"], s6)
    if "S7_off" in sec:
        def s7():
            d = sec["S7_off"]
            best = min(S.SITES, key=lambda s: d["sites"][s]["net"]["median_step_err"])
            return (f"hand floor median {_f(d['hand']['median_step_err'], 0)} steps; best site {best} "
                    f"{_f(d['sites'][best]['net']['median_step_err'], 0)} steps; beats hand at: "
                    f"{', '.join(d['beats_hand_sites']) or 'no site'}; beats untrained twin at: "
                    f"{', '.join(d.get('beats_twin_sites', [])) or 'no site'}")
        row("S7 M-off against the floor", sec["S7_off"], s7)
    if "S8_untrained" in sec:
        row("S8 untrained twin of M-on", sec["S8_untrained"],
            lambda: "; ".join(f"{s} {_f(d['w'])} κ {_f(d['kappa'], 2)} {d['verdict']}"
                              for s, d in sec["S8_untrained"]["sites"].items()))
    if "S9_spread" in sec:
        row("S9 seed spread (CV R², increment of mid over hand)", sec["S9_spread"],
            lambda: "; ".join(f"band {b} (t {d['t_range'][0]}–{d['t_range'][1]}): " +
                              ("skipped" if d["skipped"] else
                               f"hand {_f(d['r2_hand'])}, mid {_f(d['r2_mid'])}, both {_f(d['r2_both'])}, increment {d['increment']:+.2f}")
                              for b, d in sec["S9_spread"]["bands"].items()))
    if "S10_record" in sec:
        rec = sec["S10_record"]
        L.append(f"| S10 training record | files: {', '.join(sorted(rec)) or 'none'} | |")
    L += ["", "Verdict rules: registration §5 (docs/W3_LOOP_PROTOCOL.md). Full numbers in verdict.json.", ""]
    return "\n".join(L)


def write(indir, smoke=False):
    v = analyze(indir, smoke)
    d = Path(indir)
    (d / "verdict.json").write_text(dumps(v))
    (d / "report.md").write_text(report(v))
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description="W3 analysis (registration §5)")
    ap.add_argument("dir")
    ap.add_argument("--smoke", action="store_true", help="smoke seeds (+100) and smoke bootstrap size")
    a = ap.parse_args(argv)
    v = write(a.dir, a.smoke)
    print(f"P1 {v['P1']['verdict']}  P2 {v['P2'].get('verdict')}  -> {Path(a.dir) / 'verdict.json'}")


if __name__ == "__main__":
    main()
