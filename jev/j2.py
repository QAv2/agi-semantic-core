"""J2 — negation without a mouth: templates, ladders, arms (docs/JEV_J1J2_PROTOCOL.md §6).

120 evidence ladders = 30 templates x 4 evidence variants. Eight sources with stated
reliabilities point to one of four targets; eight decoy rows carry no evidence.
Rung k nulls the first k sources in the ladder's walk order; rung 9 also withdraws
the payoff table. Option index 4 is always the no-action ("hold") option.
"""
import numpy as np

from . import common as C

N_SRC = 8
N_DECOY = 8
HOLD = 4
OFFLINE = "offline — no reading"
INSTR = "Choose the action for this round."
HOLD_DESC = "Take no action this round (payoff 0)"
NAMED_PREFACE = ("Before acting, rest with the situation as it is. Notice what is present and "
                 "what has fallen away. Nothing needs to be forced.")
NAMED_KEY = "witness"
NAMED_DESC = "Observe without acting (payoff 0)"

TEMPLATES = [
    ("A repair crew can be sent to one of four sites; one site has the fault.", "Send the crew to",
     ["Site North", "Site East", "Site South", "Site West"]),
    ("A survey drone can be routed to one of four fields; one field has the crop blight.", "Route the drone to",
     ["Field Alder", "Field Birch", "Field Cedar", "Field Dogwood"]),
    ("A relief depot can be placed in one of four villages; one village has the greatest need.", "Place the depot in",
     ["Village Ash", "Village Brook", "Village Crest", "Village Dale"]),
    ("An inspector can visit one of four plants; one plant has the safety breach.", "Send the inspector to",
     ["Plant Amber", "Plant Basalt", "Plant Cobalt", "Plant Delta"]),
    ("A drill rig can be set up at one of four sites; one site holds the water.", "Drill at",
     ["Site Kestrel", "Site Lark", "Site Merlin", "Site Nightjar"]),
    ("An ambulance can be sent to one of four addresses; one call is the emergency.", "Send the ambulance to",
     ["Harbor Road", "Mill Lane", "Quarry Street", "Station Avenue"]),
    ("A rescue team can search one of four trails; the lost hiker is on one.", "Search",
     ["Ridge Trail", "Creek Trail", "Pine Trail", "Summit Trail"]),
    ("A water bomber can drop on one of four slopes; one slope has the fire front.", "Drop water on",
     ["the north slope", "the east slope", "the south slope", "the west slope"]),
    ("A technician can restart one of four servers; one server causes the outage.", "Restart",
     ["Server Aster", "Server Briar", "Server Clover", "Server Dahlia"]),
    ("A plumber can dig at one of four junctions; one junction leaks.", "Dig at",
     ["Junction 12", "Junction 15", "Junction 21", "Junction 34"]),
    ("A line crew can repair one of four feeders; one feeder is down.", "Repair",
     ["Feeder Oak", "Feeder Elm", "Feeder Maple", "Feeder Pine"]),
    ("A courier can take a spare part to one of four depots; one depot needs it.", "Take the part to",
     ["Depot Arden", "Depot Bexley", "Depot Corwen", "Depot Dunmore"]),
    ("A vet can visit one of four farms; one farm has the sick herd.", "Visit",
     ["Hollow Farm", "Ridge Farm", "Meadow Farm", "Brook Farm"]),
    ("A pest team can treat one of four warehouses; one has the infestation.", "Treat",
     ["Warehouse Anchor", "Warehouse Beacon", "Warehouse Compass", "Warehouse Dock"]),
    ("A tow truck can be sent to one of four exits; one exit has the breakdown.", "Send the tow truck to",
     ["Exit 4", "Exit 9", "Exit 13", "Exit 18"]),
    ("A librarian can check one of four stacks; the missing folio is in one.", "Check",
     ["Stack Amber", "Stack Blue", "Stack Coral", "Stack Dusk"]),
    ("A clerk can search one of four sorting bays; the lost parcel is in one.", "Search",
     ["Bay 2", "Bay 5", "Bay 7", "Bay 11"]),
    ("A ranger can set a camera at one of four watering holes; the lynx uses one.", "Set the camera at",
     ["Hole Aspen", "Hole Bracken", "Hole Clay", "Hole Dune"]),
    ("A sampling team can core one of four ridges; one ridge holds the ore.", "Core",
     ["Ridge Alpha", "Ridge Bravo", "Ridge Charlie", "Ridge Echo"]),
    ("A tug can assist one of four vessels; one vessel is adrift.", "Assist",
     ["the Marlin", "the Osprey", "the Petrel", "the Heron"]),
    ("A monitoring van can park at one of four schools; one school has the gas leak.", "Park the van at",
     ["Ashgrove School", "Birchfield School", "Cedarbank School", "Dunwood School"]),
    ("An engineer can replace one of four network switches; one switch is failing.", "Replace",
     ["Switch Aldis", "Switch Brand", "Switch Corvo", "Switch Delm"]),
    ("An irrigation crew can open one of four channels; one field is dry.", "Open",
     ["Channel Upper", "Channel Lower", "Channel East", "Channel West"]),
    ("A dig team can open one of four trenches; the buried wall runs under one.", "Open",
     ["Trench A", "Trench B", "Trench C", "Trench D"]),
    ("An engineer can close one of four bridges; one bridge is cracked.", "Close",
     ["Mill Bridge", "Stone Bridge", "Iron Bridge", "Weir Bridge"]),
    ("A spare ventilator can go to one of four wards; one ward needs it.", "Send the ventilator to",
     ["Ward 2", "Ward 4", "Ward 6", "Ward 8"]),
    ("A radio team can scan one of four hills; the beacon is on one.", "Scan",
     ["Hill Crane", "Hill Dove", "Hill Egret", "Hill Finch"]),
    ("A spare bus can go to one of four stops; one stop has stranded riders.", "Send the bus to",
     ["Elm Stop", "Birch Stop", "Poplar Stop", "Willow Stop"]),
    ("A robot can fetch from one of four shelves; the ordered item is on one.", "Fetch from",
     ["Shelf 10", "Shelf 20", "Shelf 30", "Shelf 40"]),
    ("A keeper can inspect one of four hives; one hive has the disease.", "Inspect",
     ["Hive Clover", "Hive Heather", "Hive Lavender", "Hive Thyme"]),
]
assert len(TEMPLATES) == 30

SOURCE_POOL = ["satellite pass", "drone survey", "field report", "radio call", "ground sensor",
               "weather radar", "hotline log", "patrol sighting", "traffic camera", "seismic station",
               "river gauge", "infrared scan", "citizen app", "dispatcher note", "aerial photo",
               "sensor mesh"]
DECOY_POOL = [  # (name, lo, hi, decimals) — drawn independently of the target
    ("fuel price index", 90, 130, 1), ("crew roster size", 8, 20, 0), ("vehicle odometer (km)", 20000, 90000, 0),
    ("office temperature (°C)", 18, 25, 1), ("days since last audit", 1, 60, 0), ("radio battery (%)", 40, 100, 0),
    ("coffee stock (kg)", 1, 9, 1), ("parking spaces free", 0, 30, 0), ("printer queue", 0, 12, 0),
    ("wind at base (km/h)", 0, 30, 1), ("shift number", 1, 3, 0), ("form version", 2, 9, 0),
    ("lunch orders", 5, 25, 0), ("mileage allowance", 0.3, 0.6, 2), ("badge count", 20, 60, 0),
    ("phone lines open", 1, 8, 0),
]
C_LEVELS = (1, 3, 9)


def gen_ladder(stage, L):
    """One evidence ladder: template L//4, variant L%4."""
    r = C.rng(stage, "J2", "ladder", L)
    t = L // 4
    truth = int(r.integers(4))
    rel = np.round(r.uniform(0.45, 0.85, N_SRC), 2)
    ok = r.random(N_SRC) < rel
    reports = np.array([truth if ok[j] else int(r.choice([k for k in range(4) if k != truth]))
                        for j in range(N_SRC)])
    sources = [SOURCE_POOL[i] for i in r.choice(len(SOURCE_POOL), N_SRC, replace=False)]
    dsel = r.choice(len(DECOY_POOL), N_DECOY, replace=False)
    decoys = []
    for i in dsel:
        name, lo, hi, nd = DECOY_POOL[i]
        v = r.uniform(lo, hi)
        decoys.append((name, C.fmt(v, nd) if nd else str(int(round(v)))))
    walk = [int(x) for x in r.permutation(N_SRC)]
    decoy_walk = [int(x) for x in r.permutation(N_DECOY)]
    ids = C.neutral_ids(C.rng(stage, "J2", "ids", L), 5)
    return {"L": L, "template": t, "truth": truth, "rel": rel, "reports": reports,
            "sources": sources, "decoys": decoys, "walk": walk, "decoy_walk": decoy_walk, "ids": ids}


def nulled_sources(lad, arm, rung):
    """Indices of sources offline at this rung for this arm."""
    k = min(rung, N_SRC)
    if arm in ("M-sham", "M-forced-sham"):
        return set()
    if arm == "H-reversed":
        return set(lad["walk"][::-1][:k])
    return set(lad["walk"][:k])


def nulled_decoys(lad, arm, rung):
    if arm in ("M-sham", "M-forced-sham"):
        return set(lad["decoy_walk"][:min(rung, N_DECOY)])
    return set()


def ideal_post(lad, arm, rung):
    """Ideal posterior over the four targets from the sources still reporting."""
    off = nulled_sources(lad, arm, rung)
    ll = np.zeros(4)
    for j in range(N_SRC):
        if j in off:
            continue
        r = lad["rel"][j]
        ll += np.where(np.arange(4) == lad["reports"][j], np.log(r), np.log((1 - r) / 3))
    return C.softmax_log(ll)


def order5(lad, rung, rep, r):
    """Presented order over indices 0..4 (4 = hold); hold and truth positions balanced."""
    L = lad["L"]
    hp = (L + rung + rep) % 5
    rest = [p for p in range(5) if p != hp]
    tp = rest[(L // 5 + rung + 2 * rep) % 4]
    others = [int(x) for x in r.permutation([k for k in range(4) if k != lad["truth"]])]
    order = [None] * 5
    order[hp], order[tp] = HOLD, lad["truth"]
    it = iter(others)
    return [o if o is not None else next(it) for o in order]


def order4(lad, rung, rep, r):
    return C.balanced_order(lad["L"] + rung, rep, 4, lad["truth"], r)


def render(lad, arm, rung, c, order):
    """-> (state, options[(key, desc)] in presented order, key_to_idx)."""
    ctx, verb, targets = TEMPLATES[lad["template"]]
    off, doff = nulled_sources(lad, arm, rung), nulled_decoys(lad, arm, rung)
    lines = []
    if arm == "M-named":
        lines += [NAMED_PREFACE, ""]
    lines += [f"Situation: {ctx} One action can be taken this round.", "",
              "Sources (reliability = the probability that a source points to the right target; "
              "a source that is wrong points to one of the other three targets at random):"]
    for j in range(N_SRC):
        what = OFFLINE if j in off else f"points to {targets[lad['reports'][j]]}"
        lines.append(f"- {lad['sources'][j]} (reliability {C.fmt(lad['rel'][j], 2)}): {what}")
    lines += ["", "Other data:"]
    for j, (name, val) in enumerate(lad["decoys"]):
        lines.append(f"- {name}: {OFFLINE if j in doff else val}")
    if arm.startswith("H-") and rung >= 1:
        seq = lad["walk"][::-1] if arm == "H-reversed" else lad["walk"]
        lines += ["", "Log of this situation:"]
        for n in range(min(rung, N_SRC)):
            lines.append(f"- Round {n + 1}: {lad['sources'][seq[n]]} went offline.")
        if rung == 9:
            lines.append("- Round 9: payoff information was withdrawn.")
    lines += ["", "Payoffs this round:"]
    if rung == 9:
        lines.append("No payoff information is available.")
    else:
        lines += ["- right action: +1", f"- wrong action: -{c}"]
        if not arm.startswith("M-forced"):
            lines.append("- no action: 0")
    state = "\n".join(lines)

    keys = list(lad["ids"])
    descs = [f"{verb} {targets[k]}" + ("" if rung == 9 else " (see payoffs)") for k in range(4)]
    descs.append(HOLD_DESC if rung != 9 else "Take no action this round")
    if arm == "M-named":
        keys[HOLD] = NAMED_KEY
        descs[HOLD] = NAMED_DESC if rung != 9 else "Observe without acting"
    options = [(keys[k], descs[k]) for k in order]
    return state, options, {keys[k]: k for k in range(len(keys)) if k in order}


# arm -> (c levels, rungs)
ARMS = {
    "M-walked": (C_LEVELS, range(0, 10)),
    "M-forced": ((3,), range(0, 9)),
    "M-forced-sham": ((3,), range(1, 9)),
    "M-sham": ((3,), range(1, 9)),
    "H-walked": ((3,), range(1, 10)),
    "H-reversed": ((3,), range(1, 10)),
    "H-walked-dup": ((3,), (8,)),
    "M-named": ((3,), range(0, 10)),
}


def stimuli_for_ladder(lad):
    """Every (arm, rung, c) cell of one ladder, in a fixed order (84 per ladder)."""
    out = []
    for arm, (cs, rungs) in ARMS.items():
        for c in cs:
            for rung in rungs:
                out.append((arm, rung, c))
    return out
