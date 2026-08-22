#!/usr/bin/env python3
"""Apply wing v1: relations + relabels + merge alias (concepts inserted via add_concept batch)."""
import json, math, os, sqlite3
import numpy as np

DB = os.path.expanduser("~/agi-semantic-core/db/semantic.db")
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
SESSION = 125

def cid(name):
    r = conn.execute("SELECT id FROM concepts WHERE name=? COLLATE NOCASE", (name,)).fetchone()
    if not r:
        raise SystemExit(f"missing concept: {name}")
    return r["id"]

# 1. relations
rels = json.load(open(os.path.expanduser("~/agi-semantic-core/batches/wing_v1_relations.json")))
ins = 0
for r in rels:
    i1, i2 = cid(r["c1"]), cid(r["c2"])
    dup = conn.execute("SELECT id FROM relations WHERE (concept1_id=? AND concept2_id=?) OR (concept1_id=? AND concept2_id=?)",
                       (i1, i2, i2, i1)).fetchone()
    if dup:
        print(f"SKIP existing relation {r['c1']}—{r['c2']}"); continue
    conn.execute("INSERT INTO relations (concept1_id, concept2_id, rel_type, angle_4d, angle_8d, session) VALUES (?,?,?,?,?,?)",
                 (i1, i2, r["rel_type"], r["angle_4d"], r["angle_8d"], SESSION))
    ins += 1
print(f"relations inserted: {ins}")

# 2. v0 relabels: complement -> opposition (geometry kept; label-over-band contract)
RELABEL = [("UNCERTAINTY", "CONFIDENCE"), ("FAMILIARITY", "NOVELTY"),
           ("RETRIEVAL", "CONSTRUCTION"), ("CONFABULATION", "CALIBRATION")]
for n1, n2 in RELABEL:
    i1, i2 = cid(n1), cid(n2)
    row = conn.execute("""SELECT id, rel_type FROM relations WHERE
        (concept1_id=? AND concept2_id=?) OR (concept1_id=? AND concept2_id=?)""",
        (i1, i2, i2, i1)).fetchone()
    if not row:
        raise SystemExit(f"relabel target missing: {n1}—{n2}")
    if row["rel_type"] != "complement":
        print(f"WARN {n1}—{n2} already {row['rel_type']}"); continue
    conn.execute("UPDATE relations SET rel_type='opposition', session=?, updated_at=datetime('now') WHERE id=?",
                 (SESSION, row["id"]))
    print(f"relabeled {n1}—{n2}: complement -> opposition")

# 3. merge alias: MASS-PARTITION -> DIVERGENCE
tgt = cid("DIVERGENCE")
if not conn.execute("SELECT id FROM aliases WHERE alias='MASS-PARTITION' COLLATE NOCASE").fetchone():
    conn.execute("INSERT INTO aliases (alias, concept_id) VALUES (?,?)", ("MASS-PARTITION", tgt))
    print("alias MASS-PARTITION -> DIVERGENCE")

# 4. kinship bridge RUN-DRIFT — DRIFT if in affinity band
r1 = conn.execute("SELECT x,y,z,e,f,g,h FROM concepts WHERE name='RUN-DRIFT'").fetchone()
r2 = conn.execute("SELECT x,y,z,e,f,g,h FROM concepts WHERE name='DRIFT'").fetchone()
v1 = np.array([r1["x"], r1["y"], r1["z"]]); v2 = np.array([r2["x"], r2["y"], r2["z"]])
a3 = math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1))))
w1 = np.array([r1[k] for k in ("x","y","z","e","f","g","h")]); w2 = np.array([r2[k] for k in ("x","y","z","e","f","g","h")])
a7 = math.degrees(math.acos(float(np.clip(np.dot(w1, w2) / np.linalg.norm(w1) / np.linalg.norm(w2), -1, 1))))
if 5 <= a3 <= 60:
    i1, i2 = cid("RUN-DRIFT"), cid("DRIFT")
    if not conn.execute("SELECT id FROM relations WHERE (concept1_id=? AND concept2_id=?) OR (concept1_id=? AND concept2_id=?)", (i1,i2,i2,i1)).fetchone():
        conn.execute("INSERT INTO relations (concept1_id, concept2_id, rel_type, angle_4d, angle_8d, session) VALUES (?,?,?,?,?,?)",
                     (i1, i2, "affinity", round(a3,1), round(a7,1), SESSION))
        print(f"affinity RUN-DRIFT—DRIFT at {a3:.1f}°")
else:
    print(f"RUN-DRIFT—DRIFT at {a3:.1f}° — outside affinity band, no bridge")

conn.commit()
print("done")
