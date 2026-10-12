"""The committed record of a W3 stage (docs/W3_LOOP_PROTOCOL.md §9) beside the analysis outputs:
the sample grids as PNG, and a compressed copy of the per-image summary quantities P1 and the
secondaries are sums of (the 2 x 2 move terms of analyze_w3.move_terms, per image, at every
site of M-on, M-drop and M-on's untrained twin, standardised as the analysis does).

Run:  ~/venvs/semcore/bin/python -m windows.w3_record windows/results/w3_flight_<stamp>
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from . import w3_spec as S
from .analyze_w3 import inv_sd, move_terms


def grid(x, cols=8):
    """(n, 3, 32, 32) in [-1, 1] -> one RGB image, `cols` samples per row, 2-px gaps."""
    x = ((np.asarray(x, np.float32).clip(-1, 1) + 1) * 127.5).round().astype(np.uint8)
    n, _, h, w = x.shape
    rows = -(-n // cols)
    out = np.full((rows * (h + 2) + 2, cols * (w + 2) + 2, 3), 255, np.uint8)
    for i, im in enumerate(x):
        r, c = divmod(i, cols)
        out[2 + r * (h + 2):2 + r * (h + 2) + h, 2 + c * (w + 2):2 + c * (w + 2) + w] = im.transpose(1, 2, 0)
    return Image.fromarray(out)


def record(d):
    d = Path(d)
    for m in ("on", "drop", "off"):
        p = d / f"samples_{m}.npy"
        if p.exists():
            grid(np.load(p)).save(d / f"samples_{m}.png")
    out = {}
    for m in ("on", "drop", "on0"):
        sp, pt = S.load(f"{m}_split", d), S.load(f"{m}_ptrain", d)
        if m == "on":
            out.update({k: sp[k] for k in ("t", "tp", "off", "img")})
        for s in S.SITES:
            for k, v in move_terms(sp[s], inv_sd(pt[s])).items():
                out[f"{m}/{s}/{k}"] = v.astype(np.float64)
    np.savez_compressed(d / "per_image_terms.npz", **out)
    return d / "per_image_terms.npz"


if __name__ == "__main__":
    print(record(sys.argv[1]))
