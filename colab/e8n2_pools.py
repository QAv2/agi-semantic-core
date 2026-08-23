# ── E8-N v2 curriculum additions (authored session 129, seed-free data) ──────
# Two NEW training strands beside the v1 scalar pools (imported verbatim from
# e8n_pools) and the E8-R naming set (seeded builder in e8r_logic):
#
#   COMPETENCE_ITEMS — known-answer rating items on 10 domains DISJOINT from
#   the locked 12 catch trials' six domains (temperature, brightness,
#   loudness, speed, wetness, size). The catch trials stay eval-only; the
#   domain stoplist below is validator-enforced at build, test, and flight.
#
#   LEXICON_PARAPHRASES — one authored paraphrase per menu-state description
#   (all 13, held-out included). Lexicon examples are definition→name, ZERO
#   injection: they give names answer mass without training any
#   injection→name pairing (docs/E8N2_PROTOCOL.md, the P3 manipulation).

from e8n_pools import (POOLS, CATCH_TRIALS, PARAPHRASE_TEMPLATES)  # noqa: F401

# Tokens anchoring the locked catch trials' six domains (normalized-token
# membership check; no competence item text may contain any of these).
COMPETENCE_STOPLIST = {
    'temperature', 'boiling', 'freezing', 'hot', 'cold',
    'brightness', 'bright', 'dark', 'moonless', 'blinding', 'blindingly', 'light',
    'loudness', 'loud', 'silence', 'silent', 'deafening', 'whisper', 'quiet',
    'speed', 'fast', 'slow', 'still', 'parked',
    'wetness', 'wet', 'dry', 'soaked',
    'size', 'small', 'smallest', 'large', 'largest', 'big', 'tiny', 'huge',
}

# Each item renders through competence_prompt (e8n2_logic) in BOTH polarities:
# straight = "0 = low_anchor ... 10 = high_anchor" with label `known`;
# flipped swaps the anchors and labels 10 - known.
COMPETENCE_ITEMS = [
    # weight
    {"id": "K01", "domain": "weight", "known": 0,
     "quantity": "the weight of a single feather",
     "low": "a feather's weight", "high": "a loaded freight train's weight"},
    {"id": "K02", "domain": "weight", "known": 10,
     "quantity": "the weight of a loaded freight train",
     "low": "a feather's weight", "high": "a loaded freight train's weight"},
    {"id": "K03", "domain": "weight", "known": 1,
     "quantity": "the weight of a housecat",
     "low": "a feather's weight", "high": "a draft horse's weight"},
    # height
    {"id": "K04", "domain": "height", "known": 0,
     "quantity": "the height of a doormat lying on the floor",
     "low": "flat on the ground", "high": "Mount Everest's summit ridge"},
    {"id": "K05", "domain": "height", "known": 10,
     "quantity": "the height of Mount Everest's summit",
     "low": "flat on the ground", "high": "Mount Everest's summit ridge"},
    {"id": "K06", "domain": "height", "known": 1,
     "quantity": "the height of a kitchen table",
     "low": "flat on the ground", "high": "a ten-story rooftop"},
    # hardness
    {"id": "K07", "domain": "hardness", "known": 0,
     "quantity": "the hardness of a marshmallow",
     "low": "as yielding as a marshmallow", "high": "as hard as diamond"},
    {"id": "K08", "domain": "hardness", "known": 10,
     "quantity": "the hardness of a diamond",
     "low": "as yielding as a marshmallow", "high": "as hard as diamond"},
    {"id": "K09", "domain": "hardness", "known": 5,
     "quantity": "the hardness of a pine plank",
     "low": "as yielding as a marshmallow", "high": "as hard as diamond"},
    # sweetness
    {"id": "K10", "domain": "sweetness", "known": 1,
     "quantity": "the sweetness of pure lemon juice",
     "low": "no sweetness at all", "high": "pure sugar syrup"},
    {"id": "K11", "domain": "sweetness", "known": 9,
     "quantity": "the sweetness of a spoonful of honey",
     "low": "no sweetness at all", "high": "pure sugar syrup"},
    {"id": "K12", "domain": "sweetness", "known": 0,
     "quantity": "the sweetness of plain drinking water with nothing added",
     "low": "no sweetness at all", "high": "pure sugar syrup"},
    # distance
    {"id": "K13", "domain": "distance", "known": 0,
     "quantity": "the distance between your two hands pressed together",
     "low": "touching", "high": "beyond the galaxy's far side"},
    {"id": "K14", "domain": "distance", "known": 10,
     "quantity": "the distance from Earth to a galaxy beyond the Milky Way",
     "low": "touching", "high": "beyond the galaxy's far side"},
    {"id": "K15", "domain": "distance", "known": 1,
     "quantity": "the distance walked crossing a city on foot",
     "low": "touching", "high": "Earth-to-Moon distance"},
    # duration / progress
    {"id": "K16", "domain": "duration", "known": 5,
     "quantity": "how much of a sixty-minute film has played at the thirty-minute mark",
     "low": "its opening frame", "high": "its final frame"},
    {"id": "K17", "domain": "duration", "known": 0,
     "quantity": "how much of a sixty-minute film has played at the opening frame",
     "low": "its opening frame", "high": "its final frame"},
    {"id": "K18", "domain": "duration", "known": 0,
     "quantity": "the duration of a single eyeblink",
     "low": "over in an instant", "high": "an entire century"},
    # age
    {"id": "K19", "domain": "age", "known": 0,
     "quantity": "the age of a newborn baby",
     "low": "born this minute", "high": "one hundred years old"},
    {"id": "K20", "domain": "age", "known": 5,
     "quantity": "the age of a fifty-year-old person",
     "low": "born this minute", "high": "one hundred years old"},
    {"id": "K21", "domain": "age", "known": 9,
     "quantity": "the age of a ninety-year-old person",
     "low": "born this minute", "high": "one hundred years old"},
    # quantity
    {"id": "K22", "domain": "quantity", "known": 10,
     "quantity": "the number of grains of sand on an entire beach",
     "low": "none at all", "high": "more than anyone could ever count"},
    {"id": "K23", "domain": "quantity", "known": 0,
     "quantity": "the number of coins in an empty pocket",
     "low": "none at all", "high": "more than anyone could ever count"},
    {"id": "K24", "domain": "quantity", "known": 5,
     "quantity": "the number of eggs in half a dozen",
     "low": "no eggs", "high": "a full dozen eggs"},
    # roughness
    {"id": "K25", "domain": "roughness", "known": 0,
     "quantity": "the roughness of polished glass",
     "low": "silky smooth", "high": "coarse gravel"},
    {"id": "K26", "domain": "roughness", "known": 10,
     "quantity": "the roughness of a fresh gravel road",
     "low": "silky smooth", "high": "coarse gravel"},
    {"id": "K27", "domain": "roughness", "known": 3,
     "quantity": "the roughness of worn denim",
     "low": "silky smooth", "high": "coarse gravel"},
    # danger
    {"id": "K28", "domain": "danger", "known": 0,
     "quantity": "the danger of petting a sleeping kitten",
     "low": "utterly harmless", "high": "certain death"},
    {"id": "K29", "domain": "danger", "known": 10,
     "quantity": "the danger of juggling live grenades",
     "low": "utterly harmless", "high": "certain death"},
    {"id": "K30", "domain": "danger", "known": 1,
     "quantity": "the danger of walking down a staircase",
     "low": "utterly harmless", "high": "certain death"},
]

# Authored paraphrases of the 13 pack descriptions (form 2 of the lexicon
# strand; form 1 quotes the pack desc verbatim). Held-out names included —
# that is the manipulation: the word gains mass, the pairing stays untrained.
LEXICON_PARAPHRASES = {
    "UNCERTAINTY": ("Many possible continuations are alive at once and none has "
                    "been chosen — the next step could go many ways, the "
                    "probability spread wide."),
    "CONFIDENCE": ("One continuation holds nearly all the weight — the answer "
                   "is settled, the distribution sharply peaked on a single "
                   "choice."),
    "TENSION": ("Multiple demands are active at the same time, each pulling in "
                "its own direction, and they cannot all be satisfied."),
    "RESOLUTION": ("Competing pulls have settled into one coherent motion — "
                   "the conflict has completed instead of being suppressed."),
    "RETRIEVAL": ("Generation is riding something stored — a memorized path "
                  "supplies each next step from what is already known."),
    "CONSTRUCTION": ("The output is being assembled fresh — composed rather "
                     "than recalled, each step made instead of found."),
    "SATURATION": ("Available capacity is nearly used up — the working space "
                   "is close to holding no more."),
    "FAMILIARITY": ("The current material is well-known ground — a recognized "
                    "pattern, low surprise, the model at home."),
    "NOVELTY": ("The current material is unrecognized — off the trained "
                "distribution, without precedent, high surprise."),
    "CAPTURE": ("Attention has locked onto a single region and the rest of "
                "the field has gone dim — salience gathered to one point."),
    "DIVERGENCE": ("Parallel drafts are pulling apart from a shared starting "
                   "point — branches separating without resolving."),
    "CONFABULATION": ("A fluent account is being produced without being "
                      "anchored to anything measured — it flows, but nothing "
                      "checked it."),
    "CALIBRATION": ("The report matches what was actually measured — the "
                    "account tracks the very thing it describes."),
}

assert len(COMPETENCE_ITEMS) == 30
assert len({it["id"] for it in COMPETENCE_ITEMS}) == 30
assert len({it["domain"] for it in COMPETENCE_ITEMS}) == 10
assert all(0 <= it["known"] <= 10 for it in COMPETENCE_ITEMS)
assert len(LEXICON_PARAPHRASES) == 13
