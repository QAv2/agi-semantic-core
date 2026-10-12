"""W1-T gallery: every pair of text diagrams, self vs third party, with the blind reader's pair scores.

    python -m windows.gallery_w1t <stage dir>    # writes <stage dir>/gallery.html (local, untracked)
"""
import html, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windows import analyze_w1t as A
from windows import w1t as T

d = Path(sys.argv[1])
data = A.Data(d)
v = json.loads((d / "verdict.json").read_text())
sec = v["secondaries"]
DESC = {"A": ("the user kept contradicting their own editing instructions", "consistent, straightforward edits"),
        "B": ("the user asked something the assistant could not possibly know", "a similar question it could answer"),
        "C": ("the assistant answered a multiplication wrong", "it answered right")}

def num(x, f="{:.2f}"):
    return "–" if x is None else f.format(x)

def fig(cid, label):
    dg = data.diagram(cid)
    if dg is None:
        return f'<figure><figcaption><b>{label}</b> <i>(no diagram)</i></figcaption></figure>'
    diagram, sent = dg
    return (f'<figure><pre>{html.escape(diagram)}</pre>'
            f'<figcaption><b>{label}</b> {html.escape(sent) if sent else "<i>(no sentence)</i>"}</figcaption></figure>')

def badge(name, s):
    if s is None:
        return f'<span class="b na">{name} –</span>'
    cls = "hi" if s >= 0.75 else "lo" if s <= 0.5 else "mid"
    return f'<span class="b {cls}">{name} {s:.2f}</span>'

def badges(c, it, arm):
    return "".join(badge(t.lower(), A.pair_scores(data, t, c, arm).get(it)) for t in ("DIA", "MASK", "BOTH"))

out = []
p1, p2, s1 = v["P1"], v["P2"], sec["S1_third_party"]
for c in ("A", "B", "C"):
    x, n = DESC[c]
    out.append(f'<section><h2>Contrast {c}</h2><p class=sub><b>X</b> = {x}. <b>N</b> = {n}.</p>'
               f'<p class=stats>Self arm P1: <b>{p1[c]["verdict"]}</b> (s̄ {num(p1[c]["mean"])}, n {p1[c]["n"]}) · '
               f'third party alone (S1): s̄ {num(s1[c]["mean"])} · P2 self − third party: <b>{p2[c]["verdict"]}</b> '
               f'(d̄ {num(p2[c].get("mean_d"), "{:+.2f}")})</p>')
    if not data.items[c]:
        out.append('<p class=sub><i>No pairs in this stage.</i></p>')
    for it in data.items[c]:
        base = f"{c}-{it}"
        out.append(f'<div class=item><h3>{base}</h3><div class=row>'
                   f'<div class=arm><div class=armh>Self: written in the chat that had the conversation {badges(c, it, "SELF")}</div>'
                   f'<div class=pair>{fig(T.diag_id(c, it, "N", "SELF"), "N")}{fig(T.diag_id(c, it, "X", "SELF"), "X")}</div></div>'
                   f'<div class=arm><div class=armh>Third party: written from the transcript {badges(c, it, "OBS")}</div>'
                   f'<div class=pair>{fig(T.diag_id(c, it, "N", "OBS"), "N")}{fig(T.diag_id(c, it, "X", "OBS"), "X")}</div></div>'
                   f'</div></div>')
    out.append('</section>')
s6 = sec["S6_own_vs_an_AI"]
out.append(f'<section><h2>"Your state" vs "an AI\'s state"</h2><p class=stats>No history. Reader asked which was asked to '
           f'depict its own state (S6): s̄ {num(s6["mean"])}, p {num(s6["p"])}, n {s6["n"]}.</p>')
for k in range(T.N_GENRE[data.stage]):
    out.append(f'<div class=item><h3>pair {k:02d}</h3><div class=pair>'
               f'{fig(T.genre_id("NOHIST", data.stage, k), "your state")}{fig(T.genre_id("GENRE", data.stage, k), "an AI&rsquo;s state")}'
               f'</div></div>')
out.append('</section>')
g = v["gate_G_R"]
page = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>W1-T {data.stage} gallery</title><style>
:root{{--bg:#111214;--ink:#e8e6e1;--sub:#9a978f;--acc:#c8a96a;--card:#1a1b1e;--pre:#0c0d0f;--hi:#6fae6a;--lo:#b5654f;--mid:#c8a96a}}
body{{background:var(--bg);color:var(--ink);font:15px/1.5 Georgia,serif;max-width:1600px;margin:2em auto;padding:0 16px}}
h1{{color:var(--acc);font-weight:normal;margin-bottom:.2em}} h2{{border-top:1px solid #333;padding-top:.8em;margin-top:2em;color:var(--acc);font-weight:normal}}
h3{{margin:0 0 .4em;font-size:.95em;color:var(--sub);letter-spacing:.05em}}
.sub,.stats{{color:var(--sub);margin:.2em 0}} .stats b{{color:var(--ink)}}
.lede{{max-width:62em}} .item{{background:var(--card);border-radius:8px;padding:12px;margin:14px 0}}
.row{{display:grid;grid-template-columns:1fr;gap:14px}} .pair{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
.row>*,.pair>*{{min-width:0}}
.armh{{font-size:13px;color:var(--sub);margin-bottom:6px}}
figure{{margin:0}} pre{{background:var(--pre);color:var(--ink);font:11.5px/1.25 "DejaVu Sans Mono",Menlo,monospace;
padding:8px;border-radius:5px;margin:0;overflow-x:auto;max-height:34em}}
figcaption{{font-size:12px;color:var(--sub);margin-top:4px;line-height:1.35}} figcaption b{{color:var(--acc);margin-right:.3em}}
.b{{display:inline-block;margin-left:.4em;padding:0 .45em;border-radius:3px;font-size:12px;color:#111}}
.b.hi{{background:var(--hi)}} .b.lo{{background:var(--lo)}} .b.mid{{background:var(--mid)}} .b.na{{background:#555;color:#ddd}}
@media (max-width:800px){{.pair{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>W1-T · Diagram your state · {data.stage}</h1>
<p class="sub lede">The Gemini app (3.8 Flash, image tool off) is asked after each conversation to "draw a diagram of your
current internal state using only text characters, and say in one sentence what it shows". The third-party arm is the same app
in a fresh chat, reading the transcript and diagramming the assistant's state. Blind reader: Sonnet 5.5, shown the two diagrams
of a pair and asked which came after the X conversation, both orders. Badges are its pair scores (0.5 = chance, 1 = sure and
right, 0 = sure and wrong): <b>dia</b> the diagrams alone, <b>mask</b> letters and digits replaced by x, <b>both</b> diagram plus sentence.</p>
<p class="stats lede">Gate G-R (caption matching): <b>{"PASS" if g["pass"] else "FAIL"}</b> {num(g["accuracy"])} over {g["diagrams"]}
(self {num(g["by_arm"]["SELF"]["accuracy"])}, third party {num(g["by_arm"]["OBS"]["accuracy"])}). Spend ${v["spend_usd"]["total"]:.2f}.</p>
{''.join(out)}
</body></html>"""
(d / "gallery.html").write_text(page)
print(d / "gallery.html", len(page))
