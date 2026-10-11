"""Write windows/results/w3_freeze_<stamp>/FROZEN_W3.json (docs/W3_LOOP_PROTOCOL.md, status block):
the sha256 of every W3 code file, of the notebook and of the code embedded in it, the seeds and
sizes, and the library versions. Committed and pushed before the smoke.

Run:  ~/venvs/semcore/bin/python -m windows.freeze_w3
"""
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import w3_spec as S

ROOT = Path(__file__).resolve().parent.parent
CODE = ("windows/w3_spec.py", "windows/w3_model.py", "windows/w3_core.py", "windows/analyze_w3.py",
        "windows/agents_w3.py", "windows/freeze_w3.py", "windows/power/w3_split_check.py",
        "windows/power/w3_design_check.py", "windows/power/w3_collinear_check.py",
        "windows/power/w3_budget_check.py", "colab/build_w3_notebook.py", "colab/W3_LOOP_UI.ipynb")


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze(out=None):
    import numpy
    import sklearn
    import torch
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    out = Path(out or ROOT / "windows" / "results" / f"w3_freeze_{stamp}")
    out.mkdir(parents=True, exist_ok=True)
    reg = subprocess.run(["git", "log", "-1", "--format=%H", "--", "docs/W3_LOOP_PROTOCOL.md"],
                         cwd=ROOT, capture_output=True, text=True).stdout.strip()
    nb = json.loads((ROOT / "colab/W3_LOOP_UI.ipynb").read_text())
    payload_cell = next("".join(c["source"]) for c in nb["cells"] if "EXPECTED_SHA = " in "".join(c["source"]))
    embedded = json.loads(payload_cell.split("EXPECTED_SHA = ", 1)[1].split("\nPKG", 1)[0])
    for name, h in embedded.items():
        assert h == sha_file(ROOT / "windows" / name), f"{name}: the notebook embeds a stale copy; rebuild it"
    frozen = {
        "registration_commit": reg,
        "code_sha256": {f: sha_file(ROOT / f) for f in CODE},
        "tests_sha256": {p.name: sha_file(p) for p in sorted((ROOT / "windows/tests").glob("test_w3_*.py"))},
        "notebook_embedded_sha256": embedded,
        "route": {"subject": "three CIFAR-10 denoisers trained from scratch (M-on, M-drop, M-off)",
                  "compute": "Colab T4 through the browser UI (Joe's Run all), mount-native notebook",
                  "retrieval": "Google Drive integration; parts <= 3.8 MB with manifest sha256"},
        "seeds": S.SEEDS, "smoke_seed_offset": S.SMOKE_SEED_OFFSET,
        "sizes": S.sizes(False), "smoke_sizes": S.sizes(True),
        "bars": {"LO": S.LO, "HI": S.HI, "KAPPA_BAR": S.KAPPA_BAR, "DELTA_VARIES": S.DELTA_VARIES,
                 "SHAM_BAR": S.SHAM_BAR, "SHAM_CLASS": S.SHAM_CLASS, "GDET_TOL": S.GDET_TOL},
        "versions": {"python": platform.python_version(), "numpy": numpy.__version__,
                     "sklearn": sklearn.__version__, "torch_local": torch.__version__},
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "FROZEN_W3.json").write_text(json.dumps(frozen, indent=1, sort_keys=True) + "\n")
    return out / "FROZEN_W3.json"


if __name__ == "__main__":
    print(freeze())
