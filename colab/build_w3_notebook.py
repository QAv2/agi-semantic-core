#!/usr/bin/env python3
"""Build colab/W3_LOOP_UI.ipynb (UI-native: drive.mount + shutil, zero rclone, no CLI control).

W3 "Does the loop know where it is?" (docs/W3_LOOP_PROTOCOL.md §4). The notebook is
self-contained: windows/w3_spec.py, w3_model.py and w3_core.py are embedded verbatim as one
tar.gz payload in a collapsed form cell, written to /content/w3pkg/windows/ and imported from
there; their sha256 is checked against the values baked in here (G-hash) and recorded in env.json.
Everything the run does is w3_core.run().

Warm-runtime law: SMOKE = True is the default. Run all → the smoke (own seeds, two epochs) ships
to MyDrive/semcore/w3/smoke_<stamp>/ and the notebook ends without releasing the GPU. After the
smoke is checked, flip SMOKE = False and Run all again on the same runtime: the set-up cells find
their work done and the flight starts. A dropped runtime: set RESUME_STAMP to the stamp printed at
the start and Run all again.
"""
import base64
import gzip
import hashlib
import io
import json
import os
import tarfile
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
WIN = os.path.join(HERE, "..", "windows")
FILES = ("w3_spec.py", "w3_model.py", "w3_core.py")


def payload():
    """The tar.gz payload, byte-deterministic (zero mtimes in tar and gzip headers), so the same
    code always builds the same notebook."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for name in ("__init__.py",) + FILES:
            data = b"" if name == "__init__.py" else open(os.path.join(WIN, name), "rb").read()
            ti = tarfile.TarInfo(f"windows/{name}")
            ti.size, ti.mtime, ti.mode = len(data), 0, 0o644
            tf.addfile(ti, io.BytesIO(data))
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as gz:
        gz.write(raw.getvalue())
    return out.getvalue()


def shas():
    return {name: hashlib.sha256(open(os.path.join(WIN, name), "rb").read()).hexdigest() for name in FILES}


INTRO = """# W3: Does the loop know where it is?

Trains three small image denoisers on CIFAR-10 (M-on, M-drop, M-off), checks that each learned
to denoise, then records their insides on the registered inputs. Pre-registration:
`docs/W3_LOOP_PROTOCOL.md` in the agi-semantic-core repo.

**Run:** Runtime → Change runtime type → **T4 GPU** → **Run all**. Allow the Drive mount.

1. **First Run all = the smoke** (`SMOKE = True`, about 15 minutes). It writes to
   `MyDrive/semcore/w3/smoke_<stamp>/` and stops. Leave the runtime connected.
2. Claude checks the smoke from Drive. On an OK: set `SMOKE = False` in the next cell and
   **Run all again on the same runtime**. That is the flight, about 4–5 hours; keep the tab open.
3. If the runtime drops: copy the stamp printed under "Run", paste it into `RESUME_STAMP`, Run all
   again. Training resumes from the last checkpoint saved to Drive; finished parts are skipped.

Nothing here downloads anything but CIFAR-10 from its home at cs.toronto.edu (md5 checked).
"""

CELL_CONFIG = """#@title Settings
SMOKE = True  #@param {type:"boolean"}
RESUME_STAMP = ""  #@param {type:"string"}
"""

CELL_SETUP = r'''#@title Setup: GPU check, Drive mount
import json, subprocess, sys, time
from pathlib import Path
import torch
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                     capture_output=True, text=True)
print('GPU:', gpu.stdout.strip() or 'NONE DETECTED')
assert torch.cuda.is_available(), 'No GPU: Runtime > Change runtime type > T4 GPU.'
from google.colab import drive
if not Path('/content/drive/MyDrive').exists():
    drive.mount('/content/drive')
SEM = Path('/content/drive/MyDrive/semcore')
assert SEM.exists(), 'MyDrive/semcore not found: mounted the right Google account?'
W3 = SEM / 'w3'
W3.mkdir(exist_ok=True)
print('torch', torch.__version__, '| results ->', W3)
'''

CELL_PAYLOAD_TMPL = '''#@title Code payload (W3 builder files, verbatim; collapsed on purpose)
import base64, hashlib, io, sys, tarfile
from pathlib import Path
PAYLOAD = """
{b64}
"""
EXPECTED_SHA = {shas}
PKG = Path('/content/w3pkg')
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode='r:gz') as tf:
    tf.extractall(PKG)
CODE_SHA = {{n: hashlib.sha256((PKG / 'windows' / n).read_bytes()).hexdigest() for n in EXPECTED_SHA}}
assert CODE_SHA == EXPECTED_SHA, f'G-hash: extracted code differs: {{CODE_SHA}}'
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))
for m in [m for m in sys.modules if m == 'windows' or m.startswith('windows.')]:
    del sys.modules[m]
from windows import w3_spec as S, w3_core as C
print('G-hash OK:', {{k: v[:12] for k, v in CODE_SHA.items()}})
'''

CELL_DATA = r'''#@title Data: CIFAR-10 (binary), md5-checked; cached on Drive after the first download
import shutil, urllib.request
LOCAL = Path('/content/cifar-10-binary.tar.gz')
CACHE = W3 / 'cifar-10-binary.tar.gz'
if not (LOCAL.exists() and C.md5_file(LOCAL) == S.CIFAR_MD5):
    if CACHE.exists() and C.md5_file(CACHE) == S.CIFAR_MD5:
        shutil.copyfile(CACHE, LOCAL)
    else:
        req = urllib.request.Request(S.CIFAR_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=600) as r, open(LOCAL, 'wb') as f:
            shutil.copyfileobj(r, f)
        assert C.md5_file(LOCAL) == S.CIFAR_MD5, 'G-data: CIFAR-10 md5 mismatch'
        shutil.copyfile(LOCAL, CACHE)
print('CIFAR-10 OK', C.md5_file(LOCAL))
'''

CELL_RUN = r'''#@title Run (training, gates, harvest; resume-safe)
stage = 'smoke' if SMOKE else 'flight'
stamp = RESUME_STAMP.strip() or time.strftime('%Y%m%d_%H%M')
OUT = W3 / f'{stage}_{stamp}'
OUT.mkdir(parents=True, exist_ok=True)
print(f'STAMP {stamp}  ->  MyDrive/semcore/w3/{OUT.name}   (paste the stamp into RESUME_STAMP to resume)')
LOG = OUT / 'run_log.txt'
def log(msg):
    line = time.strftime('%H:%M:%S ') + str(msg)
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')
log(f'start {stage}; code {json.dumps({k: v[:12] for k, v in CODE_SHA.items()})}')
C.run(OUT, LOCAL, smoke=SMOKE, device=torch.device('cuda'), log=log, code_sha=CODE_SHA)
log('run finished')
'''

CELL_SUMMARY = r'''#@title Summary
man = json.loads((OUT / 'manifest.json').read_text()) if (OUT / 'manifest.json').exists() else []
sets = sorted({x['set'] for x in man})
print(f'{OUT.name}: {len(sets)} sets in {len(man)} parts, largest part {max([x["bytes"] for x in man] or [0]) / 1e6:.2f} MB')
for tag in ('on', 'drop', 'off'):
    g = OUT / f'gate_{tag}.json'
    if g.exists():
        gg = json.loads(g.read_text())
        print(f'  gate {tag}: {"PASS" if gg["pass"] else "FAIL"} at {gg["epochs"]} epochs; ' +
              ' '.join(f'{b["steps"][0]}-{b["steps"][1]}: {b["mse_model"]:.4f}<{b["mse_best_multiple"]:.4f}' for b in gg['bands']))
for f in ('gdet.json', 'resume_test.json'):
    if (OUT / f).exists():
        print(' ', f, json.loads((OUT / f).read_text()).get('pass'))
print('DONE' if (OUT / 'DONE').exists() else 'NOT DONE')
if SMOKE:
    print('\nSmoke finished. Leave this runtime connected. After Claude checks it, set SMOKE = False and Run all again.')
'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text, form=False):
    meta = {"cellView": "form"} if form else {}
    return {"cell_type": "code", "execution_count": None, "metadata": meta, "outputs": [],
            "source": text.splitlines(keepends=True)}


def build():
    b64 = "\n".join(textwrap.wrap(base64.b64encode(payload()).decode(), 120))
    cell_payload = CELL_PAYLOAD_TMPL.format(b64=b64, shas=json.dumps(shas(), indent=1))
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"colab": {"provenance": []}, "accelerator": "GPU",
                         "kernelspec": {"name": "python3", "display_name": "Python 3"}},
            "cells": [md(INTRO), code(CELL_CONFIG, form=True), code(CELL_SETUP, form=True),
                      code(cell_payload, form=True), code(CELL_DATA, form=True),
                      code(CELL_RUN, form=True), code(CELL_SUMMARY, form=True)]}


def main():
    nb = build()
    path = os.path.join(HERE, "W3_LOOP_UI.ipynb")
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path, f"({os.path.getsize(path) / 1e3:.0f} kB)", shas())


if __name__ == "__main__":
    main()
