#!/usr/bin/env python3
"""Render the J1+J2 paper: Jev_J1J2.md -> Jev_J1J2.html (reading surface, dark),
Jev_J1J2_print.html (print surface, light) and, with --pdf, Jev_J1J2.pdf (Zenodo upload).

House style and pipeline copied from the comprehensive edition's assemble.py
(~/qualia-algebra/internal/comprehensive/). Figures: make_figures.py.
"""
import os, re, shutil, subprocess, sys

BASE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(BASE, "Jev_J1J2.md")
TITLE = ("Knows Its Slips, Reads the Page: A Pre-registered Study of Confidence and "
         "Abstention in Jev, a Calibrated Decision Model")

def wrap_references(txt):
    """Render-time only: put the reference list in a .refs div (hanging indent)."""
    m = re.search(r"^## References\n", txt, flags=re.M)
    if not m:
        return txt
    start = m.end()
    nxt = re.search(r"^(---|## )", txt[start:], flags=re.M)
    end = start + (nxt.start() if nxt else len(txt[start:]))
    return txt[:start] + "\n::: {.refs}\n" + txt[start:end].strip() + "\n:::\n\n" + txt[end:]

CSS_DARK = """
:root{--bg:#0d0c0a;--ink:#e8e2d4;--dim:#a39b87;--gold:#c9a84c;--gold-dim:#8a7433;--rule:#2a2620;
--mono:"SFMono-Regular",Consolas,"DejaVu Sans Mono",monospace;--serif:Georgia,"DejaVu Serif",serif}
*{box-sizing:border-box}html{background:var(--bg)}
body{margin:0;background:var(--bg);color:var(--ink);font:16.5px/1.7 var(--serif)}
main{max-width:880px;margin:0 auto;padding:48px 28px 96px}
h1{font-size:31px;font-weight:normal;line-height:1.25;margin:6px 0 10px}
h2{font-size:14px;font-family:var(--mono);letter-spacing:.15em;text-transform:uppercase;color:var(--gold);
border-bottom:1px solid var(--rule);padding-bottom:8px;margin:52px 0 16px}
h3{font-size:18px;margin:30px 0 8px;color:#f0e8d2}
h4{font-size:16px;margin:22px 0 6px;color:#e6dcc2}
p{margin:11px 0}em{color:var(--dim)}strong{color:#f4ecd6}
a{color:#cdb46e;text-decoration:none;border-bottom:1px dotted var(--gold-dim)}
code{font-family:var(--mono);font-size:.84em;background:#1a1815;padding:1px 5px;border-radius:3px;color:#d9cfae}
pre{background:#12100d;border:1px solid #241f19;border-radius:6px;padding:14px 16px;overflow-x:auto;line-height:1.45}
pre code{background:none;padding:0;font-size:13px}
.table-wrap{overflow-x:auto;margin:16px 0}
table{width:100%;border-collapse:collapse;font-size:14px}
th{font-family:var(--mono);font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:var(--dim);
text-align:left;padding:8px 10px;border-bottom:1px solid var(--gold-dim);vertical-align:bottom}
td{padding:8px 10px;border-bottom:1px solid #241f19;vertical-align:top}
hr{border:0;border-top:1px solid var(--rule);margin:40px 0}
ul,ol{padding-left:26px}li{margin:6px 0}
blockquote{border-left:3px solid var(--gold-dim);margin:14px 0;padding:2px 18px;color:var(--dim)}
figure{margin:22px 0}figure img{max-width:100%;border-radius:6px;background:#fff;border:1px solid #241f19}
figcaption{font-size:14px;color:var(--dim);margin-top:8px}
.byline{color:var(--dim);font-size:15px;margin-bottom:26px}
.eq{text-align:center;margin:14px 0;font-size:1.03em}
.refs p{padding-left:2em;text-indent:-2em;margin:7px 0;font-size:15px;overflow-wrap:anywhere}
"""

CSS_PRINT = """
@page{size:A4;margin:22mm 20mm 22mm 20mm}
*{box-sizing:border-box}
body{margin:0;background:#fff;color:#161412;font:10.6pt/1.5 "DejaVu Serif",Georgia,serif}
main{max-width:none;margin:0;padding:0}
h1{font-size:21pt;font-weight:normal;line-height:1.2;margin:0 0 6pt}
h2{font-size:13pt;margin:22pt 0 8pt;padding-bottom:3pt;border-bottom:0.6pt solid #8a7433;page-break-after:avoid}
h3{font-size:11.5pt;margin:14pt 0 5pt;page-break-after:avoid}
h4{font-size:10.8pt;margin:10pt 0 4pt;page-break-after:avoid}
p{margin:5pt 0;text-align:justify;hyphens:auto}em{color:#3d372e}
a{color:#5a4a16;text-decoration:none}
code{font-family:"DejaVu Sans Mono",monospace;font-size:8.6pt;background:#f3f0e8;padding:0 2pt}
pre{background:#f7f5ef;border:0.5pt solid #d8d2c2;padding:6pt 7pt;font-size:7.5pt;line-height:1.35;
white-space:pre-wrap;page-break-inside:avoid}
pre code{background:none;padding:0}
.table-wrap{margin:8pt 0}
table{width:100%;border-collapse:collapse;font-size:8.6pt;page-break-inside:auto}
tr{page-break-inside:avoid}
th{text-align:left;border-bottom:0.8pt solid #444;padding:3pt 4pt;vertical-align:bottom}
td{border-bottom:0.4pt solid #ccc;padding:3pt 4pt;vertical-align:top}
hr{border:0;border-top:0.5pt solid #bbb;margin:16pt 0}
ul,ol{padding-left:16pt}li{margin:2pt 0}
figure{margin:10pt 0;page-break-inside:avoid;text-align:center}figure img{max-width:100%}
figcaption{font-size:9pt;color:#333;text-align:left;margin-top:4pt}
.byline{color:#333;font-size:10pt;margin-bottom:12pt}
.eq{text-align:center;margin:7pt 0}
.eq p{text-align:center}
.refs p{padding-left:1.6em;text-indent:-1.6em;text-align:left;margin:3pt 0;font-size:9.4pt;overflow-wrap:anywhere}
"""

def render():
    tmp = MD + ".pandoc.md"
    open(tmp, "w", encoding="utf-8").write(wrap_references(open(MD, encoding="utf-8").read()))
    html_body = subprocess.run(
        ["pandoc", tmp, "--from", "markdown+pipe_tables+smart+autolink_bare_uris",
         "--to", "html5", "--wrap=none", "--no-highlight"],
        capture_output=True, text=True, check=True).stdout
    os.remove(tmp)
    html_body = html_body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    for css, out in ((CSS_DARK, "Jev_J1J2.html"), (CSS_PRINT, "Jev_J1J2_print.html")):
        doc = (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
               f'<meta name="viewport" content="width=device-width, initial-scale=1">'
               f'<title>{TITLE}</title><style>{css}</style></head>'
               f'<body><main>{html_body}</main></body></html>')
        open(os.path.join(BASE, out), "w", encoding="utf-8").write(doc)
        print("rendered", out)
    words = len(re.sub(r"```.*?```", " ", open(MD, encoding="utf-8").read(), flags=re.S).split())
    print(f"~{words:,} words")


def pdf():
    src = os.path.join(BASE, "Jev_J1J2_print.html")
    dst = os.path.join(BASE, "Jev_J1J2.pdf")
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
                    "--no-pdf-header-footer", f"--print-to-pdf={dst}", "file://" + src],
                   check=True, capture_output=True, timeout=240)
    print("pdf", dst, os.path.getsize(dst), "bytes")


if __name__ == "__main__":
    render()
    if "--pdf" in sys.argv:
        pdf()
