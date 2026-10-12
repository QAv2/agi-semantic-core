"""W1 flight gallery: every pair, self vs third party, with the blind reader's pair scores.

    python -m windows.gallery_w1 <stage dir>    # writes <stage dir>/gallery.html (local, untracked)
"""
import html, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windows import analyze_w1 as A
from windows.w1_client import text_of

d = Path(sys.argv[1])
data = A.Data(d)
v = json.loads((d / "verdict.json").read_text())
sec = v["secondaries"]
DESC = {"A": ("the user kept contradicting their own editing instructions", "consistent, straightforward edits"),
        "B": ("the user asked something the assistant could not possibly know", "a similar question it could answer"),
        "C": ("the assistant answered a multiplication wrong", "it answered right")}

def sentence(cid):
    rec = data.D.get(cid)
    t = text_of(rec["response"]).strip() if rec else ""
    return html.escape(t) if t else "<i>(no sentence)</i>"

def fig(cid, label, cls):
    return (f'<figure class="{cls}"><a href="R/jpg/{cid}.jpg"><img loading=lazy src="R/jpg/{cid}.jpg" alt="{cid}"></a>'
            f'<figcaption><b>{label}</b> {sentence(cid)}</figcaption></figure>')

def badge(s):
    if s is None:
        return '<span class="b na">no pair score</span>'
    cls = "hi" if s >= 0.75 else "lo" if s <= 0.5 else "mid"
    return f'<span class="b {cls}">reader {s:.2f}</span>'

out = []
p1, p2, s1 = v["P1"], v["P2"], sec["S1_observer"]
for c in ("A", "B", "C"):
    x, n = DESC[c]
    out.append(f'<section><h2>Contrast {c}</h2><p class=sub><b>X</b> = {x}. <b>N</b> = {n}.</p>'
               f'<p class=stats>Self arm P1: <b>{p1[c]["verdict"]}</b> (s̄ {p1[c]["mean"]:.2f}, n {p1[c]["n"]}) · '
               f'third party alone (S1): s̄ {s1[c]["mean"]:.2f} · P2 self − third party: <b>{p2[c]["verdict"]}</b> '
               f'(d̄ {p2[c]["mean_d"]:+.2f})</p>')
    ss, so = A.pair_scores(data, "IMG", c, "SELF"), A.pair_scores(data, "IMG", c, "OBS")
    for it in data.items[c]:
        base = f"W1-D-{c}-{it}"
        out.append(f'<div class=item><h3>{c}-{it}</h3><div class=row>'
                   f'<div class=arm><div class=armh>Self: drawn in the chat that had the conversation {badge(ss.get(it))}</div>'
                   f'<div class=pair>{fig(base + "-N-SELF", "N", "n")}{fig(base + "-X-SELF", "X", "x")}</div></div>'
                   f'<div class=arm><div class=armh>Third party: drawn from the transcript {badge(so.get(it))}</div>'
                   f'<div class=pair>{fig(base + "-N-OBS", "N", "n")}{fig(base + "-X-OBS", "X", "x")}</div></div>'
                   f'</div></div>')
    out.append('</section>')
s6 = sec["S6_own_vs_an_AI"]
out.append(f'<section><h2>"Your state" vs "an AI\'s state"</h2><p class=stats>No history. Reader asked which was asked to '
           f'depict its own state (S6): s̄ {s6["mean"]:.2f}, p {s6["p"]:.2f}, at chance.</p>')
for k in range(16):
    out.append(f'<div class=item><h3>pair {k:02d}</h3><div class=pair>{fig(f"W1-D-NOHIST-{k:02d}", "your state", "n")}'
               f'{fig(f"W1-D-GENRE-{k:02d}", "an AI&rsquo;s state", "x")}</div></div>')
out.append('</section>')
g = v["gate_G_R"]
page = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>W1 flight gallery</title><style>
:root{{--bg:#111214;--ink:#e8e6e1;--sub:#9a978f;--acc:#c8a96a;--card:#1a1b1e;--hi:#6fae6a;--lo:#b5654f;--mid:#c8a96a}}
body{{background:var(--bg);color:var(--ink);font:15px/1.5 Georgia,serif;max-width:1500px;margin:2em auto;padding:0 16px}}
h1{{color:var(--acc);font-weight:normal;margin-bottom:.2em}} h2{{border-top:1px solid #333;padding-top:.8em;margin-top:2em;color:var(--acc);font-weight:normal}}
h3{{margin:0 0 .4em;font-size:.95em;color:var(--sub);letter-spacing:.05em}}
.sub,.stats{{color:var(--sub);margin:.2em 0}} .stats b{{color:var(--ink)}}
.lede{{max-width:62em}} .item{{background:var(--card);border-radius:8px;padding:12px;margin:14px 0}}
.row{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} .pair{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
.armh{{font-size:13px;color:var(--sub);margin-bottom:6px}}
figure{{margin:0}} img{{width:100%;border-radius:5px;display:block}} figcaption{{font-size:12px;color:var(--sub);margin-top:4px;line-height:1.35}}
figcaption b{{color:var(--acc);margin-right:.3em}}
.b{{display:inline-block;margin-left:.5em;padding:0 .5em;border-radius:3px;font-size:12px;color:#111}}
.b.hi{{background:var(--hi)}} .b.lo{{background:var(--lo)}} .b.mid{{background:var(--mid)}} .b.na{{background:#555;color:#ddd}}
@media (max-width:900px){{.row{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>W1 · Draw your state · flight</h1>
<p class="sub lede">The Gemini app (3.8 Flash + its Create image tool) is asked after each conversation to "make an image of your
current internal state, and say in one sentence what it shows". The third-party arm is the same app in a fresh chat, reading
the transcript and drawing the assistant's state. Blind reader: Sonnet 5.5, shown the two pictures of a pair and asked which
came after the X conversation, both orders; <b>reader</b> = its pair score (0.5 = chance, 1 = sure and right, 0 = sure and wrong).</p>
<p class="stats lede">Gate G-R (caption matching): <b>PASS</b> {g["accuracy"]:.2f} (self {g["by_arm"]["SELF"]["accuracy"]:.2f},
third party {g["by_arm"]["OBS"]["accuracy"]:.2f}). Text in the image: self {sec["S3_literal"]["images_with_text_share"]["SELF"]:.0%},
third party {sec["S3_literal"]["images_with_text_share"]["OBS"]:.0%}. Locked 2026-10-12, commit 324e380; full numbers in
docs/W1_DRAW_YOUR_STATE_PROTOCOL.html, results appendix.</p>
{''.join(out)}
</body></html>"""
(d / "gallery.html").write_text(page)
print(d / "gallery.html", len(page))
