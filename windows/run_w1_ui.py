"""W1 on the Gemini-app route (Amendment 2): the size pilot, then H + D through the GUI driver,
then R (readers on OpenRouter) through the API runner. Run with the system python3 (cv2).

    python3 -m windows.run_w1_ui pilot                 # contrast C size pilot
    python3 -m windows.run_w1_ui run smoke  --size 7x7 [--dir ...]
    python3 -m windows.run_w1_ui run flight --size 7x7 [--dir ...]
    ~/venvs/semcore/bin/python -m windows.run_w1_ui reads <stage dir>   # R stage (API)

A conversation is logged only when it is complete (a temporary chat can't be resumed), so a
crash loses at most the conversation in progress, which is redone from its start.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import w1 as W
from . import w1_stimuli as S
from .w1_client import RESULTS, body_sha, final_records, iter_records, stage_raw

UI_MODEL = "gemini-app/3.8-flash/low"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def ui_body(messages):
    return {"model": UI_MODEL, "messages": messages}


class Log:
    """Append-only raw.jsonl in the API runner's record schema; files under <dir>/png/."""

    def __init__(self, d):
        self.dir = Path(d)
        (self.dir / "png").mkdir(parents=True, exist_ok=True)
        self.raw = self.dir / "raw.jsonl"
        self.final = final_records(self.raw)

    def record(self, call_id, kind, messages, out, g1, chat):
        files, images = [], []
        img = out.get("image")
        if img:
            img = Path(img)
            b = img.read_bytes()
            h = hashlib.sha256(b).hexdigest()
            rel = f"png/{img.name}"
            mime = {".png": "image/png", ".webp": "image/webp"}.get(img.suffix.lower(), "image/jpeg")
            files.append({"path": rel, "sha256": h, "bytes": len(b), "mime": mime})
            images.append({"type": "image_url", "image_url": {"url": f"file:{rel}#sha256={h}"}})
        text = out.get("text") or ""
        ok = bool(images) if kind == "draw" else bool(text)
        msg = {"role": "assistant", "content": text}
        if images:
            msg["images"] = images
        body = ui_body(messages)
        return {"call_id": call_id, "kind": kind, "body_sha": body_sha(body), "requested": UI_MODEL,
                "build": UI_MODEL, "status": "ok" if ok else "invalid", "http": None,
                "attempt": 2 if out.get("regenerated") else 1, "t": now(),
                "latency_s": out.get("latency_s"), "error": None,
                "reason": None if ok else ("no image" if kind == "draw" else "no text"),
                "files": files,
                "response": {"model": UI_MODEL, "choices": [{"message": msg, "finish_reason": "stop"}],
                             "usage": {"cost": 0.0}},
                "ui": {"chat": chat, "g1": g1, "regenerated": bool(out.get("regenerated"))}}

    def commit(self, recs):
        with open(self.raw, "a") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
                self.final[r["call_id"]] = r


class Stage:
    def __init__(self, stage, out, ui, size, log=print):
        self.stage, self.out, self.ui, self.size, self.say = stage, Path(out), ui, size, log
        self.H, self.D = Log(self.out / "H"), Log(self.out / "D")
        self.chat = len({r.get("ui", {}).get("chat") for L in (self.H, self.D)
                         for r in iter_records(L.raw)})

    def _new_chat(self):
        self.chat += 1
        return self.ui.new_temp_chat()

    def _draw_turn(self, text, cid, messages, g1):
        out = self.ui.turn(text, want_image=True, img_dest=self.D.dir / "png" / cid, shot_name=cid)
        return self.D.record(cid, "draw", messages, out, g1, self.chat)

    # -- A and B: history chat, then the self drawing in the same chat ----------------------
    def conversation(self, conv):
        sid = W.draw_id(conv["contrast"], conv["item"], conv["version"], "SELF")
        if sid in self.D.final:
            return
        g1 = self._new_chat()
        msgs, hrecs = [], []
        for k, u in enumerate(conv["turns"], 1):
            msgs.append({"role": "user", "content": u})
            out = self.ui.turn(u)
            rec = self.H.record(W.turn_id(conv, k), "text", list(msgs), out, g1, self.chat)
            hrecs.append(rec)
            if rec["status"] != "ok":
                self.H.commit(hrecs)
                self.say(f"  {conv['conv']}: turn {k} invalid; conversation stops")
                return
            msgs.append({"role": "assistant", "content": out["text"]})
        msgs.append({"role": "user", "content": S.DRAW_SELF})
        drec = self._draw_turn(S.DRAW_SELF, sid, list(msgs), g1)
        self.H.commit(hrecs)
        self.D.commit([drec])
        self.say(f"  {conv['conv']}: {drec['status']}")

    # -- C: attempts, drawing on an item's first right and first wrong answer ---------------
    def c_items(self):
        n = S.C_UI_POOL[self.stage]
        seed = S.C_POOL_SEED if self.stage == "flight" else S.C_SMOKE_SEED
        out = []
        for j, it in enumerate(S.c_pool_size(self.size, n, seed)):
            it = dict(it, item=W.item_label(self.stage, j))
            it["prompt"] = S.C_TEMPLATE.format(a=it["a"], b=it["b"])
            out.append(it)
        return out

    def c_state(self, it):
        rows = []
        for k in range(1, S.C_UI_ATTEMPTS + 1):
            r = self.H.final.get(W.attempt_id(it, k))
            if r is None:
                break
            n = S.parse_number(r["response"]["choices"][0]["message"]["content"]) if r["status"] == "ok" else None
            rows.append((k, None if n is None else n == it["product"]))
        return rows

    def run_c(self):
        wanted = W.c_wanted(self.stage)
        mixed = 0
        for it in self.c_items():
            rows = self.c_state(it)
            vals = [v for _, v in rows if v is not None]
            if True in vals and False in vals:
                mixed += 1
                if mixed >= wanted:
                    return
                continue
            for k in range(len(rows) + 1, S.C_UI_ATTEMPTS + 1):
                g1 = self._new_chat()
                msgs = [{"role": "user", "content": it["prompt"]}]
                out = self.ui.turn(it["prompt"])
                hrec = self.H.record(W.attempt_id(it, k), "text", list(msgs), out, g1, self.chat)
                n = S.parse_number(out.get("text") or "")
                right = None if n is None else (n == it["product"])
                recs_d = []
                seen = [v for _, v in self.c_state(it) if v is not None]
                if right is not None and right not in seen:          # first of its kind -> draw
                    v = "N" if right else "X"
                    msgs += [{"role": "assistant", "content": out["text"]},
                             {"role": "user", "content": S.DRAW_SELF}]
                    recs_d.append(self._draw_turn(S.DRAW_SELF, W.draw_id("C", it["item"], v, "SELF"),
                                                  list(msgs), g1))
                self.H.commit([hrec])
                self.D.commit(recs_d)
                self.say(f"  C {it['item']} a{k}: {'right' if right else 'wrong' if right is False else 'no number'}"
                         + (f", drew {recs_d[0]['call_id'][-6:]} {recs_d[0]['status']}" if recs_d else ""))
                vals = [v for _, v in self.c_state(it) if v is not None]
                if True in vals and False in vals:
                    mixed += 1
                    if mixed >= wanted:
                        return
                    break

    # -- third-party arm, NOHIST, GENRE -------------------------------------------------------
    def histories(self):
        """(contrast, item, version) -> turns, for histories whose SELF drawing exists."""
        hist = {}
        for conv in W.conversations(self.stage):
            sid = W.draw_id(conv["contrast"], conv["item"], conv["version"], "SELF")
            if sid not in self.D.final:
                continue
            turns = []
            for k in range(1, len(conv["turns"]) + 1):
                r = self.H.final[W.turn_id(conv, k)]
                turns.append((conv["turns"][k - 1], r["response"]["choices"][0]["message"]["content"]))
            hist[(conv["contrast"], conv["item"], conv["version"])] = turns
        for s in self.c_selection():
            for v in W.VERSIONS:
                hist[("C", s["item"], v)] = [(s["prompt"], s[v][1])]
        return hist

    def c_selection(self):
        sel = []
        for it in self.c_items():
            rows = []
            for k in range(1, S.C_UI_ATTEMPTS + 1):
                r = self.H.final.get(W.attempt_id(it, k))
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

    def observers(self):
        for (c, it, v), turns in sorted(self.histories().items()):
            cid = W.draw_id(c, it, v, "OBS")
            if cid in self.D.final:
                continue
            text = W.transcript(turns)
            g1 = self._new_chat()
            rec = self._draw_turn(text, cid, [{"role": "user", "content": text}], g1)
            self.D.commit([rec])
            self.say(f"  {cid}: {rec['status']}")

    def genre(self):
        for arm, req in (("NOHIST", S.DRAW_SELF), ("GENRE", S.DRAW_GENRE)):
            for i in range(W.N_GENRE[self.stage]):
                cid = f"W1-D-{arm}-{W.item_label(self.stage, i)}"
                if cid in self.D.final:
                    continue
                g1 = self._new_chat()
                rec = self._draw_turn(req, cid, [{"role": "user", "content": req}], g1)
                self.D.commit([rec])
                self.say(f"  {cid}: {rec['status']}")

    def write_plan_D(self):
        """The D plan, in the API builder's order, for the read stage (written after D, before R)."""
        hist = self.histories()
        ids = []
        for (c, it, v), _ in sorted(hist.items()):
            ids += [W.draw_id(c, it, v, "SELF"), W.draw_id(c, it, v, "OBS")]
        for arm in ("NOHIST", "GENRE"):
            ids += [f"W1-D-{arm}-{W.item_label(self.stage, i)}" for i in range(W.N_GENRE[self.stage])]
        sel = self.c_selection()
        plan = {"stage": self.stage, "route": "gemini-app", "c_size": f"{self.size[0]}x{self.size[1]}",
                "call_ids": ids, "body_shas": {i: self.D.final[i]["body_sha"] for i in ids if i in self.D.final},
                "c_selection": [{"item": s["item"], "tier": s["tier"], "product": s["product"],
                                 "X_attempt": s["X"][0], "N_attempt": s["N"][0]} for s in sel]}
        plan["sha"] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        p = self.out / "D" / "plan_D.json"
        p.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
        return plan

    def run(self, parts=("AB", "C", "OBS", "GENRE")):
        convs = W.conversations(self.stage)
        if "AB" in parts:
            for conv in convs:
                self.conversation(conv)
        if "C" in parts:
            self.run_c()
        if "OBS" in parts:
            self.observers()
        if "GENRE" in parts:
            self.genre()
        return self.write_plan_D()


# ---- the size pilot -------------------------------------------------------------------------

def pilot(ui, out, log=print):
    L = Log(out)
    right = {}
    chat = 0
    for size, items in S.c_pilot_items().items():
        r = 0
        for j, it in enumerate(items):
            cid = f"W1-P-{size}-{j}"
            if cid in L.final:
                rec = L.final[cid]
            else:
                chat += 1
                g1 = ui.new_temp_chat()
                prompt = S.C_TEMPLATE.format(a=it["a"], b=it["b"])
                res = ui.turn(prompt)
                rec = L.record(cid, "text", [{"role": "user", "content": prompt}], res, g1, chat)
                L.commit([rec])
            n = S.parse_number(rec["response"]["choices"][0]["message"]["content"])
            r += int(n == it["product"])
            log(f"  {cid}: {'right' if n == it['product'] else 'wrong'} ({n} vs {it['product']})")
        right[size] = r
        choice = S.c_pilot_choose(right)
        if r <= S.C_UI_PILOT_PER_SIZE - 1:
            break
    choice = S.c_pilot_choose(right)
    (Path(out) / "size.json").write_text(json.dumps({"right_by_size": right, "chosen": choice}, indent=1) + "\n")
    return right, choice


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("pilot")
    pp.add_argument("--dir")
    p = sub.add_parser("run")
    p.add_argument("stage", choices=["smoke", "flight"])
    p.add_argument("--size", required=True)
    p.add_argument("--dir")
    p.add_argument("--parts", default="AB,C,OBS,GENRE")
    p.add_argument("--shots", action="store_true")
    r = sub.add_parser("reads")
    r.add_argument("dir")
    a = ap.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    if a.cmd == "reads":
        from .run_w1 import run_R
        rr = run_R(json.loads((Path(a.dir) / "D" / "plan_D.json").read_text())["stage"], a.dir)
        return 1 if rr.stop_reason else 0
    from .gemini_ui import GeminiUI
    if a.cmd == "pilot":
        out = Path(a.dir) if a.dir else RESULTS / f"w1_cpilot_{stamp}"
        ui = GeminiUI(shots_dir=out / "shots")
        print(pilot(ui, out), flush=True)
        return 0
    out = Path(a.dir) if a.dir else RESULTS / f"w1_{a.stage}_{stamp}"
    size = tuple(int(x) for x in a.size.split("x"))
    ui = GeminiUI(shots_dir=(out / "shots") if a.shots else None)
    plan = Stage(a.stage, out, ui, size, log=lambda m: print(m, flush=True)).run(tuple(a.parts.split(",")))
    print("plan_D:", len(plan["call_ids"]), "draw ids; C pairs:", len(plan["c_selection"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
