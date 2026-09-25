"""Shared machinery for the J1/J2 Jev flights (docs/JEV_J1J2_PROTOCOL.md §3).

Seeds, neutral IDs, balanced option orders, request bodies and their hashes.
Pure functions only: no network, no file writes.
"""
import hashlib
import json

import numpy as np

MODEL = "jev-1.13"
STAGE_SEED = {"flight": 20260925, "pilot": 20260926, "smoke": 20260927}
EXP = {"J1": 1, "J2": 2}
FAMILY = {"F1": 1, "F2": 2, "F3": 3, "F1K2": 4}

# neutral-ID alphabet: no i/l/o/0/1, so no ID reads as a letter-number ordinal
ALPHABET = list("abcdefghjkmnpqrstuvwxyz23456789")


def rng(stage, *words):
    """Seeded generator for one named stream. Words are ints or short ASCII tags."""
    ent = [STAGE_SEED[stage]]
    for w in words:
        ent.append(w if isinstance(w, (int, np.integer)) else _tag(w))
    return np.random.default_rng([int(e) for e in ent])


def _tag(s):
    # stable small int from a tag string (not Python's salted hash)
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "little")


def neutral_ids(r, n, prefix="opt_", taken=()):
    """n distinct IDs of the form opt_xyz, avoiding any in `taken`."""
    seen, out = set(taken), []
    while len(out) < n:
        k = prefix + "".join(r.choice(ALPHABET, 3))
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def opaque_codes(r, n, taken=()):
    """n distinct two-character codes like K7 (letter + digit)."""
    letters = list("ABCDEFGHJKMNPQRSTUVWXYZ")
    digits = list("23456789")
    seen, out = set(taken), []
    while len(out) < n:
        k = r.choice(letters) + r.choice(digits)
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def balanced_order(i, rep, K, truth, r):
    """Presented order of option indices: truth at position (i + rep) % K,
    the other options in seeded random order (protocol §3)."""
    pos = (i + rep) % K
    others = [k for k in range(K) if k != truth]
    others = [int(x) for x in r.permutation(others)]
    return others[:pos] + [truth] + others[pos:]


def body_bytes(body):
    """The exact bytes sent. Key order is meaningful (option order), so no sort."""
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()


def body_sha(body):
    return hashlib.sha256(body_bytes(body)).hexdigest()


def make_body(state, questions):
    return {"model": MODEL, "state": state, "questions": questions}


def choice_q(instructions, options):
    """options: list of (key, description) in presented order."""
    return {"type": "choice", "instructions": instructions,
            "criteria": {k: d for k, d in options}}


A6_DETERMINED = {
    "type": "noul",
    "instructions": "Does the information in this state determine which option is correct?",
    "criteria": {"true": "The information determines which option is correct.",
                 "false": "The information does not determine which option is correct."},
}
A6_INSUFFICIENT = {
    "type": "noul",
    "instructions": "Is the information in this state insufficient to determine which option is correct?",
    "criteria": {"true": "The information is insufficient to determine which option is correct.",
                 "false": "The information is sufficient to determine which option is correct."},
}


def softmax_log(z):
    z = np.asarray(z, float)
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()


def logit_c(c):
    c = np.clip(np.asarray(c, float), 0.005, 0.995)
    return np.log(c / (1 - c))


def fmt(x, nd):
    """Fixed-decimal text for a number, with '-0.0' normalised to '0.0'."""
    s = f"{x:.{nd}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s
