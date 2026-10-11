#!/usr/bin/env python3
"""Build colab/W3_TIMING_UI.ipynb (UI-native, mount + shutil, zero rclone).

W3 timing smoke, part of the design check: about five minutes on a T4 measuring training and
forward speed of the candidate denoisers on random tensors. No dataset, nothing trained to use.
It sets the registered training length ("epochs per T4 hour") from measurement instead of the
FLOP extrapolation in windows/power/w3_budget_check.py. The model classes are lifted VERBATIM
from that file (single source: what was sized is what is timed).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "..", "windows", "power", "w3_budget_check.py")).read()
MODEL_SRC = SRC[SRC.index("def sinusoid"):SRC.index("def main")].rstrip() + "\n"

INTRO = """# W3 timing smoke (design check, ~5 minutes)

Times the candidate denoisers for W3 ("Does the loop know where it is?") on a T4: training
steps at batch 128 with fp16 autocast, and forward passes for the probe harvest. Random
tensors only, so no dataset is needed and nothing is kept but the timings.

**Run:** Runtime → Change runtime type → **T4 GPU** → Run all. Allow the Drive mount.
The result lands in `MyDrive/semcore/w3/` as `timing_<stamp>.json`; the last cell also
prints the table. Safe to re-run; each run writes a new file.
"""

CELL_SETUP = r'''# ── Setup: GPU check, Drive mount ──
import json, subprocess, time
from pathlib import Path
import torch
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                     capture_output=True, text=True)
print('GPU:', gpu.stdout.strip() or 'NONE DETECTED')
assert torch.cuda.is_available(), 'No GPU: Runtime > Change runtime type > T4 GPU.'
from google.colab import drive
drive.mount('/content/drive')
SEM = Path('/content/drive/MyDrive/semcore')
assert SEM.exists(), 'MyDrive/semcore not found: mounted the right Google account?'
OUT = SEM / 'w3'
OUT.mkdir(exist_ok=True)
print('torch', torch.__version__, '| results ->', OUT)
'''

CELL_MODEL = "# ── Model classes, verbatim from windows/power/w3_budget_check.py ──\n" \
    "import math\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n" + MODEL_SRC

CELL_TIME = r'''# ── Timing: train steps (fp16 autocast, AdamW) and forward passes ──
BATCH, WARM, STEPS = 128, 15, 60
dev = 'cuda'
rows = []
for cin, data in ((1, 'MNIST'), (3, 'CIFAR-10')):
    for C in (32, 48, 64):
        for clock in ((True, False) if C == 64 else (True,)):
            torch.manual_seed(0)
            m = UNet(cin=cin, C=C, clock=clock).to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=2e-4)
            scaler = torch.amp.GradScaler('cuda')
            x = torch.randn(BATCH, cin, 32, 32, device=dev)
            t = torch.randint(0, 1000, (BATCH,), device=dev)
            torch.cuda.reset_peak_memory_stats()
            for i in range(WARM + STEPS):
                if i == WARM:
                    torch.cuda.synchronize(); t0 = time.time()
                with torch.autocast('cuda', dtype=torch.float16):
                    loss = F.mse_loss(m(x, t), x)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            torch.cuda.synchronize()
            train_s = (time.time() - t0) / STEPS
            m.eval()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                for i in range(WARM + STEPS):
                    if i == WARM:
                        torch.cuda.synchronize(); t0 = time.time()
                    m(x, t)
            torch.cuda.synchronize()
            fwd_s = (time.time() - t0) / STEPS
            n_train = 60000 if cin == 1 else 50000
            row = {'data': data, 'C': C, 'clock': clock,
                   'params_M': round(sum(p.numel() for p in m.parameters()) / 1e6, 2),
                   'train_s_per_step': round(train_s, 4),
                   'epochs_per_hour': round(3600 / (train_s * n_train / BATCH), 1),
                   'fwd_s_per_batch': round(fwd_s, 4),
                   'peak_mem_GB': round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            rows.append(row)
            print(row)
            del m, opt
            torch.cuda.empty_cache()
stamp = time.strftime('%Y%m%d_%H%M%S')
rec = {'gpu': gpu.stdout.strip(), 'torch': torch.__version__, 'batch': BATCH,
       'steps_timed': STEPS, 'rows': rows}
(OUT / f'timing_{stamp}.json').write_text(json.dumps(rec, indent=1))
print(f'\nSaved MyDrive/semcore/w3/timing_{stamp}.json')
print(f"\n{'data':9s} {'C':>3s} {'clock':>5s} {'params':>7s} {'s/step':>7s} {'ep/h':>6s} {'fwd s':>7s} {'mem':>5s}")
for r in rows:
    print(f"{r['data']:9s} {r['C']:3d} {str(r['clock']):>5s} {r['params_M']:6.1f}M {r['train_s_per_step']:7.3f} "
          f"{r['epochs_per_hour']:6.1f} {r['fwd_s_per_batch']:7.4f} {r['peak_mem_GB']:5.2f}")
'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def main():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3",
                       "display_name": "Python 3"}, "accelerator": "GPU"},
          "cells": [md(INTRO), code(CELL_SETUP), code(CELL_MODEL), code(CELL_TIME)]}
    path = os.path.join(HERE, "W3_TIMING_UI.ipynb")
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
