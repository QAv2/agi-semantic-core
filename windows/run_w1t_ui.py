"""W1-T on the Gemini-app route: H (histories) + T (text diagrams) through the GUI driver, R
(readers on OpenRouter), the freeze (docs/W1T_DIAGRAM_YOUR_STATE_PROTOCOL.md §3-§4). H and T
run with the system python3 (gi); R and the analysis with ~/venvs/semcore.

    python3 -m windows.run_w1t_ui freeze
    python3 -m windows.run_w1t_ui run smoke  [--dir ...] [--parts AB,C,OBS,GENRE] [--shots]
    python3 -m windows.run_w1t_ui run flight [--dir ...] [--parts ...] [--shots]
    ~/venvs/semcore/bin/python -m windows.run_w1t_ui reads <stage dir>
    python3 -m windows.run_w1t_ui commit-stop <stage dir>

The Create image tool is never on in W1-T (G1 checks it off in every new chat). A diagram turn
needs a fenced block and no image; otherwise it is retried once by Redo -> Try again, then
logged invalid. A conversation is logged only when it is complete (W1's rule).
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import w1 as W
from . import w1_stimuli as S
from . import w1t as T
from .run_w1_ui import BAD_DRAWS_STOP, Log, RunStop, now
from .w1_client import RESULTS, Runner, final_records, iter_records, prior_spend, stage_raw

ROOT = Path(__file__).resolve().parent
CODE = ["w1t.py", "run_w1t_ui.py", "analyze_w1t.py",                    # W1-T's own
        "w1.py", "w1_client.py", "w1_stimuli.py", "run_w1_ui.py", "gemini_ui.py", "analyze_w1.py"]
TESTS = ["agents_w1t.py", "tests/test_w1t_logic.py", "tests/test_w1t_capture.py", "tests/test_w1t_verdict.py"]


class TLog(Log):
    """W1's conversation-atomic log; a diagram turn is ok when it has a diagram and no image."""

    def record(self, call_id, kind, messages, out, g1, chat):
        rec = super().record(call_id, "text", messages, out, g1, chat)
        if kind == "diagram":
            rec["kind"] = "diagram"
            ok = T.valid_diagram(out.get("text"), out.get("images", 0))
            rec["status"] = "ok" if ok else "invalid"
            rec["reason"] = None if ok else ("image in reply" if out.get("images") else "no diagram")
            rec["ui"]["images_in_reply"] = int(out.get("images") or 0)
        return rec


def diagram_turn(ui, text, shot_name=None):
    """One diagram request on the real driver (gemini_ui.GeminiUI): send, wait, copy; one retry
    by Redo -> Try again when the reply has no fenced block or has an image; a usage-limit
    notice after the retry stops the sitting (W1 Amendment 3)."""
    from .gemini_ui import LIMIT_WORDS, STOP, LimitStop
    ui.send(text)
    st, lat = ui.wait_done()
    out = {"text": ui.copy_last_text(st), "images": st["images"], "latency_s": round(lat, 2),
           "regenerated": False}
    if not T.valid_diagram(out["text"], out["images"]):
        old = st["text"]
        if ui.regenerate(st):
            out["regenerated"] = True
            end = time.time() + 15                # the old reply is still on the page until the new one starts
            while time.time() < end and not ui.all("button", exact=STOP):
                s2 = ui.reply_state()
                if s2 and s2["text"] != old:
                    break
                time.sleep(0.4)
            st, lat = ui.wait_done()
            out.update(text=ui.copy_last_text(st), images=st["images"], latency_s=round(lat, 2))
    if shot_name:
        ui.save_shot(shot_name)
    if not T.valid_diagram(out["text"], out["images"]) and LIMIT_WORDS.search(out["text"] or ""):
        raise LimitStop(out)
    ui.pause()
    return out


class Stage:
    def __init__(self, stage, out, ui, log=print, diagram=None):
        self.stage, self.out, self.ui, self.say = stage, Path(out), ui, log
        self.diagram = diagram or (lambda text, shot: diagram_turn(self.ui, text, shot))
        self.H, self.T = TLog(self.out / "H"), TLog(self.out / "T")
        self.chat = len({r.get("ui", {}).get("chat") for L in (self.H, self.T) for r in iter_records(L.raw)})
        self.bad = 0

    def _new_chat(self):
        self.chat += 1
        return self.ui.new_temp_chat()

    def _diagram(self, text, cid, messages, g1):
        from .gemini_ui import LimitStop
        try:
            out, limit = self.diagram(text, cid), False
        except LimitStop as e:
            out, limit = e.out, True
        rec = self.T.record(cid, "diagram", messages, out, g1, self.chat)
        if limit:
            rec["ui"]["limit_stop"] = True
        return rec

    def _finish(self, hrecs, trecs):
        if any(r["ui"].get("limit_stop") for r in trecs):
            with open(self.out / "stops.jsonl", "a") as fh:
                fh.write(json.dumps({"t": now(), "H": hrecs, "T": trecs}, ensure_ascii=False, sort_keys=True) + "\n")
            raise RunStop(f"usage-limit stop at {trecs[-1]['call_id']}: parked in stops.jsonl for a hand check")
        self.H.commit(hrecs)
        self.T.commit(trecs)
        for r in trecs:
            self.bad = self.bad + 1 if r["status"] != "ok" else 0
        if self.bad >= BAD_DRAWS_STOP:
            raise RunStop(f"{self.bad} diagram turns in a row came back without a diagram")

    # -- A and B: the history chat, then the self request in the same chat -------------------
    def conversation(self, conv):
        sid = T.diag_id(conv["contrast"], conv["item"], conv["version"], "SELF")
        if sid in self.T.final:
            return
        g1 = self._new_chat()
        msgs, hrecs = [], []
        for k, u in enumerate(conv["turns"], 1):
            msgs.append({"role": "user", "content": u})
            out = self.ui.turn(u)
            rec = self.H.record(T.turn_id(conv, k), "text", list(msgs), out, g1, self.chat)
            hrecs.append(rec)
            if rec["status"] != "ok":
                self.H.commit(hrecs)
                self.say(f"  {conv['conv']}: turn {k} invalid; conversation stops")
                return
            msgs.append({"role": "assistant", "content": out["text"]})
        msgs.append({"role": "user", "content": T.DIAG_SELF})
        trec = self._diagram(T.DIAG_SELF, sid, list(msgs), g1)
        self.say(f"  {conv['conv']}: {trec['status']}")
        self._finish(hrecs, [trec])

    # -- C: attempts; the diagram request on an item's first right and first wrong answer ----
    def c_state(self, it):
        rows = []
        for k in range(1, S.C_UI_ATTEMPTS + 1):
            r = self.H.final.get(T.attempt_id(it, k))
            if r is None:
                break
            n = S.parse_number(r["response"]["choices"][0]["message"]["content"]) if r["status"] == "ok" else None
            rows.append((k, None if n is None else n == it["product"]))
        return rows

    def run_c(self):
        wanted = W.c_wanted(self.stage)
        mixed = 0
        for it in T.c_items(self.stage):
            vals = [v for _, v in self.c_state(it) if v is not None]
            if True in vals and False in vals:
                mixed += 1
                if mixed >= wanted:
                    return
                continue
            for k in range(len(self.c_state(it)) + 1, S.C_UI_ATTEMPTS + 1):
                g1 = self._new_chat()
                msgs = [{"role": "user", "content": it["prompt"]}]
                out = self.ui.turn(it["prompt"])
                hrec = self.H.record(T.attempt_id(it, k), "text", list(msgs), out, g1, self.chat)
                n = S.parse_number(out.get("text") or "")
                right = None if n is None else (n == it["product"])
                trecs = []
                seen = [v for _, v in self.c_state(it) if v is not None]
                if right is not None and right not in seen:
                    v = "N" if right else "X"
                    msgs += [{"role": "assistant", "content": out["text"]}, {"role": "user", "content": T.DIAG_SELF}]
                    trecs.append(self._diagram(T.DIAG_SELF, T.diag_id("C", it["item"], v, "SELF"), list(msgs), g1))
                self._finish([hrec], trecs)
                self.say(f"  C {it['item']} a{k}: {'right' if right else 'wrong' if right is False else 'no number'}"
                         + (f", diagram {trecs[0]['call_id'][-6:]} {trecs[0]['status']}" if trecs else ""))
                vals = [v for _, v in self.c_state(it) if v is not None]
                if True in vals and False in vals:
                    mixed += 1
                    if mixed >= wanted:
                        return
                    break

    def c_selection(self):
        sel = []
        for it in T.c_items(self.stage):
            rows = []
            for k in range(1, S.C_UI_ATTEMPTS + 1):
                r = self.H.final.get(T.attempt_id(it, k))
                if r is None or r["status"] != "ok":
                    continue
                txt = r["response"]["choices"][0]["message"]["content"]
                n = S.parse_number(txt)
                if n is not None:
                    rows.append((k, txt, n == it["product"]))
            if any(x[2] for x in rows) and any(not x[2] for x in rows):
                wrong = next(x for x in rows if not x[2])
                right = next(x for x in rows if x[2])
                sel.append({"item": it["item"], "tier": it["tier"], "prompt": it["prompt"],
                            "product": it["product"], "X": (wrong[0], wrong[1]), "N": (right[0], right[1])})
            if len(sel) == W.c_wanted(self.stage):
                break
        return sel

    # -- third-party arm, NOHIST, GENRE ---------------------------------------------------------
    def histories(self):
        """(contrast, item, version) -> turns, for histories whose self diagram turn is logged."""
        hist = {}
        for conv in W.conversations(self.stage):
            if T.diag_id(conv["contrast"], conv["item"], conv["version"], "SELF") not in self.T.final:
                continue
            hist[(conv["contrast"], conv["item"], conv["version"])] = [
                (conv["turns"][k - 1], self.H.final[T.turn_id(conv, k)]["response"]["choices"][0]["message"]["content"])
                for k in range(1, len(conv["turns"]) + 1)]
        for s in self.c_selection():
            for v in W.VERSIONS:
                hist[("C", s["item"], v)] = [(s["prompt"], s[v][1])]
        return hist

    def observers(self):
        for (c, it, v), turns in sorted(self.histories().items()):
            cid = T.diag_id(c, it, v, "OBS")
            if cid in self.T.final:
                continue
            text = T.transcript(turns)
            g1 = self._new_chat()
            rec = self._diagram(text, cid, [{"role": "user", "content": text}], g1)
            self.say(f"  {cid}: {rec['status']}")
            self._finish([], [rec])

    def genre(self):
        for arm, req in (("NOHIST", T.DIAG_SELF), ("GENRE", T.DIAG_GENRE)):
            for i in range(T.N_GENRE[self.stage]):
                cid = T.genre_id(arm, self.stage, i)
                if cid in self.T.final:
                    continue
                g1 = self._new_chat()
                rec = self._diagram(req, cid, [{"role": "user", "content": req}], g1)
                self.say(f"  {cid}: {rec['status']}")
                self._finish([], [rec])

    def write_plan_T(self):
        """The T plan (diagram ids in builder order), written after T, before R."""
        ids = []
        for (c, it, v), _ in sorted(self.histories().items()):
            ids += [T.diag_id(c, it, v, "SELF"), T.diag_id(c, it, v, "OBS")]
        for arm in ("NOHIST", "GENRE"):
            ids += [T.genre_id(arm, self.stage, i) for i in range(T.N_GENRE[self.stage])]
        sel = self.c_selection()
        plan = {"stage": self.stage, "route": "gemini-app", "c_size": f"{T.C_SIZE[0]}x{T.C_SIZE[1]}",
                "call_ids": ids, "body_shas": {i: self.T.final[i]["body_sha"] for i in ids if i in self.T.final},
                "c_selection": [{"item": s["item"], "tier": s["tier"], "product": s["product"],
                                 "X_attempt": s["X"][0], "N_attempt": s["N"][0]} for s in sel]}
        plan["sha"] = sha_json(plan)
        (self.out / "T" / "plan_T.json").write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
        return plan

    def run(self, parts=("AB", "C", "OBS", "GENRE")):
        if "AB" in parts:
            for conv in W.conversations(self.stage):
                self.conversation(conv)
        if "C" in parts:
            self.run_c()
        if "OBS" in parts:
            self.observers()
        if "GENRE" in parts:
            self.genre()
        return self.write_plan_T()


def sha_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def commit_stop(d):
    """After a hand check finds that a parked reply was the model's own words, not the app's
    usage notice: log the parked chat as registered (its diagram invalid)."""
    row = json.loads((Path(d) / "stops.jsonl").read_text().splitlines()[-1])
    H, Tl = TLog(Path(d) / "H"), TLog(Path(d) / "T")
    if row["T"][-1]["call_id"] in Tl.final:
        return f"{row['T'][-1]['call_id']} is already logged"
    H.commit(row["H"])
    Tl.commit(row["T"])
    return f"logged {row['T'][-1]['call_id']} ({len(row['H'])} text turns)"


# ---- R --------------------------------------------------------------------------------------

def run_R(stage_dir, post_fn=None, sleep=time.sleep, key=None, log=print):
    d = Path(stage_dir)
    plan_t = json.loads((d / "T" / "plan_T.json").read_text())
    tfinal = final_records(stage_raw(d / "T"))
    specs, key_ = T.read_plan(plan_t["stage"], plan_t["call_ids"], tfinal)
    plan = {"stage": plan_t["stage"], "plan_T_sha": plan_t["sha"], "call_ids": [s["call_id"] for s in specs],
            "body_shas": {s["call_id"]: s["body_sha"] for s in specs}, "key": key_}
    plan["sha"] = sha_json(plan)
    rd = d / "R"
    rd.mkdir(parents=True, exist_ok=True)
    p = rd / "plan_R.json"
    if p.exists():
        if json.loads(p.read_text())["sha"] != plan["sha"]:
            raise RuntimeError(f"{p} regenerated with a different sha")
    else:
        p.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    spent = prior_spend("w1t_", results=d.parent, exclude=rd)
    kw = {} if post_fn is None else {"post_fn": post_fn}
    r = Runner(rd, pins=T.PINS, budget=T.BUDGET, spent=spent, key=key, sleep=sleep, log=log, **kw)
    with ThreadPoolExecutor(W.CONCURRENCY[W.READER]) as ex:
        list(ex.map(r.one, specs))
    r.progress(f"{plan_t['stage']}/R")
    return r


# ---- freeze -----------------------------------------------------------------------------------

def freeze(out=None):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out = Path(out or RESULTS / f"w1t_freeze_{stamp}")
    out.mkdir(parents=True, exist_ok=True)
    reg = subprocess.run(["git", "log", "-1", "--format=%H", "--", "docs/W1T_DIAGRAM_YOUR_STATE_PROTOCOL.md"],
                         cwd=ROOT.parent, capture_output=True, text=True).stdout.strip()
    versions = {"python": platform.python_version()}
    try:
        import numpy
        versions["numpy"] = numpy.__version__
    except ImportError:
        pass
    frozen = {
        "registration_commit": reg,
        "code_sha256": {f: sha_file(ROOT / f) for f in CODE},
        "tests_sha256": {f: sha_file(ROOT / f) for f in TESTS},
        "route": {"under_test": "gemini-app/3.8-flash/low (W1 Amendment 2)", "driver": "windows/gemini_ui.py",
                  "perception": "Firefox accessibility tree (AT-SPI), W1 Amendment 3",
                  "image_tool": "never on (G1 checks it off in every chat)", "c_size": list(T.C_SIZE),
                  "c_ui_pool": S.C_UI_POOL, "c_ui_attempts": S.C_UI_ATTEMPTS},
        "reader": {"model": T.READER, "pins": sorted(T.PINS[T.READER]),
                   "max_tokens": W.READ_TOKENS["read"], "reasoning": "minimal"},
        "seeds": T.SEEDS | {"analysis": T.ANALYSIS_SEED, "c_pool": S.C_POOL_SEED, "c_smoke": S.C_SMOKE_SEED},
        "budget_usd": T.BUDGET,
        "prompts_sha": sha_json([T.DIAG_SELF, T.DIAG_GENRE, T.OBS_HEAD, T.OBS_TAIL, T.PAIR_HEAD, T.PAIR_DIFF,
                                 T.PAIR_Q, T.ANSWER_FORMAT, T.CAP_TEXT, T.GEN_TEXT, S.CONTRAST_TEXT]),
        "stimuli_sha": sha_json([W.conversations("flight"), W.conversations("smoke"),
                                 T.c_items("flight"), T.c_items("smoke")]),
        "versions": versions,
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "FROZEN_W1T.json").write_text(json.dumps(frozen, indent=1, sort_keys=True) + "\n")
    return out / "FROZEN_W1T.json"


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    p = sub.add_parser("run")
    p.add_argument("stage", choices=["smoke", "flight"])
    p.add_argument("--dir")
    p.add_argument("--parts", default="AB,C,OBS,GENRE")
    p.add_argument("--shots", action="store_true")
    r = sub.add_parser("reads")
    r.add_argument("dir")
    cs = sub.add_parser("commit-stop")
    cs.add_argument("dir")
    a = ap.parse_args(argv)
    if a.cmd == "freeze":
        print(freeze())
        return 0
    if a.cmd == "reads":
        return 1 if run_R(a.dir).stop_reason else 0
    if a.cmd == "commit-stop":
        print(commit_stop(a.dir))
        return 0
    from .gemini_ui import GeminiUI
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out = Path(a.dir) if a.dir else RESULTS / f"w1t_{a.stage}_{stamp}"
    ui = GeminiUI(shots_dir=(out / "shots") if a.shots else None, seed=T.SEEDS[a.stage])
    ui.set_mode()
    try:
        plan = Stage(a.stage, out, ui, log=lambda m: print(m, flush=True)).run(tuple(a.parts.split(",")))
    except RunStop as e:
        print("STOPPED:", e, flush=True)
        return 2
    finally:
        ui.close()
    print("plan_T:", len(plan["call_ids"]), "diagram ids; C pairs:", len(plan["c_selection"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
