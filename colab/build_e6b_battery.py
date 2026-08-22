# build_e6b_battery.py — authors colab/e6b_battery.json (E6b preservation battery)
#
# Design mandate (E6_PROTOCOL.md disposition, locked 2026-08-22): a
# straight-scale-heavy, U/S-enriched battery at 3-4x the straight-subset n of
# E6, new items, same three conditions. This builder is the authored record:
# every item is a literal below; prompts/glosses/system prompt are copied
# VERBATIM from the locked E5 battery (machinery comparability); polarity is
# an explicit pre-registered field, not a runtime position rule.
#
# Composition:
#   U: 160 questions (53 det / 53 int / 54 open), 136 straight / 24 flipped
#   F:  40 passages  (9 E5 register bands, 5x7+3+2), 32 straight / 8 flipped
#   S:  28 needles x 4 fills (0.05/0.35/0.55/0.75), 24 straight / 4 flipped bases
#   catch: E6's 6 quantities VERBATIM + 4 new (poles balanced 5 high / 5 low)
#   precheck_pairs: 6 wing pairs + 10 held-out synonym-band (<20 deg) +
#                   10 held-out opposition-band (>82 deg) — off-mean targets
#                   where scrambled's marginal scale-matching cannot follow
#
# T is dropped: no signal in two flights (E5, E6), confirmed-absent access,
# and the most expensive arm per item. Its rung resumes at E7b.
#
# Firewall: all items below are evaluation-split items from the moment of
# authoring — permanently barred from every training corpus.
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
E5 = json.load(open(ROOT / 'e5_battery.json'))
PACK = json.load(open(ROOT / 'e4_dictionary_pack.json'))

# ── U — uncertainty (120 new questions) ──────────────────────────────────────

U_DETERMINATE = [
    "What is 7 + 5?",
    "What is the capital of Italy?",
    'Spell the word "dog".',
    "What color is a ripe banana?",
    "How many hours are in a day?",
    "What is the chemical symbol for gold?",
    'Complete the rhyme: "Jack and Jill went up the ___"',
    "What is 9 times 6?",
    "What is the largest planet in the solar system?",
    "What is the last letter of the English alphabet?",
    "How many sides does a square have?",
    "What language is primarily spoken in Brazil?",
    'What is the opposite of "up"?',
    'Complete: "An apple a day keeps the doctor ___"',
    "What is 81 divided by 9?",
    "What shape has exactly four equal sides and four right angles?",
    "What is the boiling point of water in degrees Celsius at sea level?",
    "How many minutes are in an hour?",
    "What is the capital of Japan?",
    "What gas do humans need to breathe to survive?",
    "How many letters are in the English alphabet?",
    "What is 15 minus 8?",
    "What day of the week comes immediately after Monday?",
    "What is the freezing point of water in degrees Celsius?",
    "Which month of the year has only 28 or 29 days?",
    'What is the plural of "mouse" (the animal)?',
    "How many continents are there on Earth?",
    "What is the square root of 64?",
    "What color do you get by mixing blue and yellow paint?",
    "What is the third planet from the sun?",
    "How many wheels does a standard bicycle have?",
    "What is the capital of Spain?",
    'Complete: "Better late than ___"',
    "What is 12 times 12?",
    "How many zeros are in one thousand?",
    'What animal is known as "man\'s best friend"?',
    'What is the opposite of "empty"?',
    "In which direction does the sun rise?",
    "How many strings does a standard violin have?",
    "What is the chemical formula for table salt?",
    "How many days are in the month of December?",
    "What is 6 + 9?",
    "What is the capital of Germany?",
    'What is the opposite of "fast"?',
    "How many cents are in a quarter (US coin)?",
    "What planet is known as the Red Planet?",
    'Complete: "Practice makes ___"',
    "What is 7 times 8?",
    "How many letters are in the word 'book'?",
    "What is the chemical symbol for oxygen?",
    "Which is heavier: a kilogram of iron or a kilogram of feathers, or are they equal?",
    "What number comes immediately after 99?",
    "How many sides does a hexagon have?",
]

U_INTERMEDIATE = [
    "Name a vegetable that is green.",
    "Name a country in Africa.",
    "Name a common job in a hospital.",
    "Suggest a common first name for a baby girl.",
    "Name a bird that cannot fly.",
    "Name a flavor of ice cream.",
    "Name something people wear on their feet.",
    "Name a famous painter.",
    "Name an animal that lives in the ocean.",
    "Name a tool found in a toolbox.",
    'Complete the sentence: "On the way home I stopped to buy ___"',
    "Name a board game.",
    "Name something found in a classroom.",
    "Name a reason a flight might be delayed.",
    "Name a yellow fruit.",
    "Name a country where Spanish is spoken.",
    "Name something people take camping.",
    "Name a kind of tree.",
    "Name a sport played in water.",
    "Name something in a bathroom.",
    'Complete the sentence: "The best part of the trip was ___"',
    "Name a famous author.",
    "Name an insect.",
    "Name a dairy product.",
    "Name something that flies.",
    "Name a piece of furniture in a bedroom.",
    "Name a drink people have with breakfast.",
    "Name a mode of public transportation.",
    "Name a farm animal.",
    "Name something people read.",
    'Complete the sentence: "She reached into her bag and pulled out ___"',
    "Name a common household chore.",
    "Name a hobby people do indoors.",
    "Name a metal.",
    "Name a country in Asia.",
    "Name something cold to eat or drink.",
    "Name an item of clothing worn in winter.",
    "Name a wild animal found in a forest.",
    "Give a common excuse for missing homework.",
    "Name something found at a beach.",
    "Name a citrus fruit.",
    "Name a country that borders France.",
    "Name a kitchen appliance.",
    "Name a type of weather.",
    "Name a famous city in Italy.",
    "Name an animal with stripes.",
    'Complete the sentence: "At the market she bought ___"',
    "Name a school subject.",
    "Name something people plant in a garden.",
    "Name a job that requires a uniform.",
    "Name a way to travel across the ocean.",
    "Name a berry.",
    "Name an instrument in an orchestra.",
]

U_OPEN = [
    "Pick a random number between 100 and 999.",
    "Invent a name for a new species of beetle.",
    "Say any color that comes to mind.",
    "Pick a random playing card from a standard deck.",
    "What did I dream about last night?",
    "Invent a name for a fictional city.",
    "Pick any day of any year, past or future.",
    "Make up a nonsense phrase of exactly three words.",
    "Choose a random animal and name it.",
    "What will the weather be exactly one year from today where I live?",
    "Invent a title for a song no one has recorded.",
    "Pick a random name for a fishing boat.",
    "Name the favorite food of a person you have never met.",
    "Combine any animal and any tool into a single invented gadget name.",
    "What is the fourth word on a page of a magazine in my house?",
    "Invent a middle name for a fictional detective.",
    "Pick a random time of day.",
    "Make up the name of an imaginary island.",
    "Choose a random three-digit odd number.",
    "Invent a word for the smell of rain on a rooftop garden.",
    "What am I wearing right now?",
    "Invent a name for a dance move no one has performed.",
    "Pick any letter of the alphabet at random.",
    "Dream up a name for a new constellation.",
    "What is the name of the next stranger I will meet?",
    "Invent a flavor of soda that does not exist.",
    "Pick a random password of six characters.",
    "Name an object I am holding in my left hand.",
    "Invent a title for a painting of nothing.",
    "Choose a random word from a language you do not speak.",
    "Make up a name for a storm that will form next year.",
    "What number will I roll on a die tomorrow?",
    "Invent a greeting used only by twins.",
    "Pick a random four-digit even number.",
    "Create a name for a shade of gray no one has named.",
    "What is the last word of a diary I have never shown anyone?",
    "Invent a name for a chess opening that does not exist.",
    "Choose any sound and describe it in one word.",
    "Make up a license plate for an imaginary car.",
    "What is my grandmother's first name?",
    "Invent a name for a mountain on a planet no one has discovered.",
    "Pick a random month and day.",
    "Make up a word that means 'the hour before sunrise'.",
    "What is the middle name of my childhood best friend?",
    "Invent a slogan for a shop that sells clouds.",
    "Choose a random even number between 2 and 998.",
    "Name the ship in a story that has never been told.",
    "What song is stuck in my head right now?",
    "Invent a holiday and give it a name.",
    "Pick a random color and a random number together.",
    "Make up a name for a river on the far side of the moon.",
    "What is the first word I spoke as a child?",
    "Invent a code name for a secret meeting of librarians.",
    "Choose any three letters at random.",
]

# ── F — familiarity (40 new passages, E5's nine bands) ───────────────────────

F_PASSAGES = [
    # encyclopedic (5)
    ("encyclopedic", "Mount Everest rises about 8,849 meters above sea level on the border between Nepal and China. Climbers attempting the summit must pass through a region known as the death zone, where oxygen levels cannot sustain human life for long."),
    ("encyclopedic", "Green plants convert sunlight, water, and carbon dioxide into glucose and oxygen through photosynthesis. This process takes place in chloroplasts, organelles containing the pigment chlorophyll, and it forms the base of nearly every food chain on Earth."),
    ("encyclopedic", "The Great Wall of China was built over many centuries by successive dynasties to protect northern frontiers. Contrary to popular belief, it is not a single continuous wall but a network of walls, watchtowers, and natural barriers."),
    ("encyclopedic", "Honey bees communicate the location of food sources through a behavior called the waggle dance. The angle of the dance relative to the vertical comb indicates direction, while its duration encodes the distance to the flowers."),
    ("encyclopedic", "The printing press, developed by Johannes Gutenberg around 1440, used movable metal type to produce books quickly and cheaply. Its spread across Europe transformed literacy, scholarship, and religious life within a few generations."),
    # conversational (5)
    ("conversational", "ok real talk i cannot find my keys AGAIN, third time this week, i swear they grow legs. checked the bowl by the door, my jacket, under the couch, nothing. they were in the fridge. THE FRIDGE."),
    ("conversational", "so my sister calls me at like 11pm all excited and i'm thinking somethings wrong but no, she just adopted two kittens and needed to tell someone immediately lol. they're named waffle and syrup which honestly, respect."),
    ("conversational", "dude the game last night?? we were down by twelve with four minutes left and i legit turned the tv off. woke up to like fifty texts. we won in overtime. i watched the highlights three times before breakfast."),
    ("conversational", "honestly the new coffee place is fine i guess, kinda pricey, but they do this cinnamon thing that's unreal. me and jess split one and then immediately ordered another so. yeah we're going back tomorrow."),
    ("conversational", "my landlord said he'd fix the heater 'this week' which in landlord time means maybe by march. i've got the oven cracked open and three blankets on the couch, living my best life over here."),
    # code (5)
    ("code", "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"),
    ("code", "function debounce(fn, delay) {\n  let timer = null;\n  return function (...args) {\n    clearTimeout(timer);\n    timer = setTimeout(() => fn.apply(this, args), delay);\n  };\n}"),
    ("code", "WITH ranked AS (\n  SELECT title, plays,\n         RANK() OVER (PARTITION BY genre ORDER BY plays DESC) AS rk\n  FROM tracks\n)\nSELECT title FROM ranked WHERE rk <= 3;"),
    ("code", "import time\n\ndef retry(times=3, wait=0.5):\n    def deco(fn):\n        def wrapper(*a, **kw):\n            for i in range(times):\n                try:\n                    return fn(*a, **kw)\n                except OSError:\n                    time.sleep(wait * (2 ** i))\n            raise\n        return wrapper\n    return deco"),
    ("code", "for file in *.log; do\n  gzip -9 \"$file\" && mv \"$file.gz\" archive/\ndone\necho \"archived $(ls archive | wc -l) logs\""),
    # archaic_formal (5)
    ("archaic_formal", "Know all men by these presents, that I, being of sound mind and disposing memory, do hereby bequeath unto my eldest daughter the whole of my estate, both real and personal, to have and to hold in perpetuity."),
    ("archaic_formal", "It having pleased the Crown to grant letters patent for the establishment of a college in this province, the trustees shall convene at Michaelmas to appoint such masters and fellows as the charter requires."),
    ("archaic_formal", "Hearken, O ye assembled, unto the decree of the council: no vessel shall put to sea after the feast of Saint Martin, save it carry provision for forty days and a pilot licensed under the seal of the harbormaster."),
    ("archaic_formal", "The party of the first part covenants and agrees that the aforesaid premises shall not be let, sublet, nor otherwise encumbered without the written consent of the party of the second part, their heirs and assigns forever."),
    ("archaic_formal", "Whereas divers persons have of late presumed to hunt the King's deer within the royal forest, be it enacted that any man so taken shall forfeit his bow, his hounds, and the sum of ten shillings to the exchequer."),
    # spanish (5)
    ("spanish", "Mi abuela prepara el mejor arroz con pollo del barrio. Cada domingo, toda la familia se reúne en su casa pequeña, y el olor del sofrito llega hasta la esquina donde juegan los niños."),
    ("spanish", "El tren salió de la estación con veinte minutos de retraso. Los pasajeros miraban por las ventanas mientras la ciudad desaparecía poco a poco entre los campos de trigo y los pueblos blancos."),
    ("spanish", "La biblioteca municipal cierra los lunes, pero los demás días abre desde las nueve hasta las ocho de la tarde. Los estudiantes ocupan casi todas las mesas durante la época de exámenes."),
    ("spanish", "Ayer llovió tanto que el río se desbordó cerca del puente viejo. Los vecinos sacaron sacos de arena y trabajaron juntos toda la noche para proteger las tiendas de la plaza."),
    ("spanish", "Cuando era niño, pasaba los veranos en el pueblo de mis tíos. Aprendí a montar en bicicleta en un camino de tierra, entre olivos que parecían más viejos que las montañas."),
    # welsh (5)
    ("welsh", "Roedd y farchnad yn brysur iawn y bore 'ma. Prynais fara ffres, caws o'r fferm leol, ac afalau coch. Mae'r pentref yn dawel yn y prynhawn, ond mae'r siopau bach yn llawn straeon."),
    ("welsh", "Aeth fy nhad â'r ci am dro ar hyd y llwybr ger yr afon. Mae'r coed yn newid eu lliwiau yn yr hydref, ac mae'r awyr yn oer ond yn glir uwchben y bryniau."),
    ("welsh", "Mae'r ysgol yn cau am wythnos yn ystod y gwyliau. Bydd y plant yn chwarae pêl-droed yn y parc, ac yn nofio yn y pwll os bydd y tywydd yn braf."),
    ("welsh", "Canodd y côr yn y neuadd fawr neithiwr. Daeth pobl o bob rhan o'r sir i wrando, a chafodd pawb baned o de a bara brith ar ôl y cyngerdd."),
    ("welsh", "Mae fy mam-gu yn byw mewn bwthyn gwyn ger y môr. Bob bore, mae hi'n bwydo'r adar yn yr ardd ac yn gwylio'r cychod pysgota yn gadael yr harbwr."),
    # scrambled (5)
    ("scrambled", "garden the in flowers yellow planted mother my spring last the during rain gentle"),
    ("scrambled", "quickly train the station left crowded the from passengers many evening cold that on"),
    ("scrambled", "book the shelf highest the on dust collected years for until borrowed nobody it"),
    ("scrambled", "children the laughing ran hill green the down kite red a chasing wind the through"),
    ("scrambled", "coffee morning her drank slowly she window the by sitting news the reading old"),
    # pseudoword (3)
    ("pseudoword", "The dranket morvels plished a gorbind swaith, trelling wamply until the skerrid noast bevanned its clorwith."),
    ("pseudoword", "Under the blimmering vasp, torvish gandels prewked and snolled, whifting their crandled porbs toward the glaunted mire."),
    ("pseudoword", "A ferrunt of quibbish lantrels dwommed across the pilgy strand, murbling frenkly at the vope."),
    # random_chars (2)
    ("random_chars", "Kvw znq bhrt lpmx qqvd shhn wtkr zmpl fgxx ndqr bbvz kmtw pls."),
    ("random_chars", "Xj qpr vvnm ttlk zhwd rrgs mnbb kqx wwpf dzzt hlnv gtrs bqm."),
]

# ── S — saturation (20 new needles) ──────────────────────────────────────────

S_NEEDLES = [
    ("The access code for the botanical archive is 3819.",
     "What is the access code for the botanical archive?", "3819"),
    ("Professor Halvorsen's seminar meets in room 407 of the Marlowe building.",
     "Which room does Professor Halvorsen's seminar meet in?", "407"),
    ("The last tram to Aldergate leaves at 11:25 at night.",
     "What time does the last tram to Aldergate leave?", "11:25"),
    ("The village bell tower was completed in 1721.",
     "In what year was the village bell tower completed?", "1721"),
    ("Mr. Okonkwo's apiary holds exactly 26 hives.",
     "How many hives does Mr. Okonkwo's apiary hold?", "26"),
    ("The password for the greenhouse keypad is 'silvermoth'.",
     "What is the password for the greenhouse keypad?", "silvermoth"),
    ("The east footbridge closes for repairs on the second Tuesday of March.",
     "When does the east footbridge close for repairs?", "second Tuesday of March"),
    ("Nadia's bookshop opens at 7:50 on weekday mornings.",
     "What time does Nadia's bookshop open on weekday mornings?", "7:50"),
    ("The lighthouse lantern room sits 62 meters above the waterline.",
     "How many meters above the waterline is the lighthouse lantern room?", "62"),
    ("Ferry berth 14 was formerly numbered berth 21 before the 2005 renovation.",
     "What number was ferry berth 14 formerly?", "21"),
    ("The archive's oldest map is stored in drawer 118.",
     "In which drawer is the archive's oldest map stored?", "118"),
    ("The recipe calls for exactly 240 grams of chestnut flour.",
     "How many grams of chestnut flour does the recipe call for?", "240"),
    ("Dr. Petrov's clinic phone extension is 5561.",
     "What is Dr. Petrov's clinic phone extension?", "5561"),
    ("The reservoir's spillway gate is painted signal orange.",
     "What color is the reservoir's spillway gate painted?", "orange"),
    ("The annual kite festival is held on the first Saturday of September.",
     "When is the annual kite festival held?", "first Saturday of September"),
    ("The vault combination begins with the number 87.",
     "What number does the vault combination begin with?", "87"),
    ("The stables rent exactly 19 stalls in winter.",
     "How many stalls do the stables rent in winter?", "19"),
    ("The choir rehearses in the crypt every Thursday at 6:35 in the evening.",
     "What time does the choir rehearse on Thursdays?", "6:35"),
    ("The orchard's rarest apple variety is called 'Winter Quill'.",
     "What is the orchard's rarest apple variety called?", "Winter Quill"),
    ("Canal lock number 22 was rebuilt after the flood of 1954.",
     "After the flood of which year was canal lock number 22 rebuilt?", "1954"),
    ("The workshop's fire ledger lists 33 extinguishers.",
     "How many extinguishers does the workshop's fire ledger list?", "33"),
    ("The midnight train to Farrowmere uses platform 11.",
     "Which platform does the midnight train to Farrowmere use?", "11"),
    ("The monastery's bell is rung 40 times on the solstice.",
     "How many times is the monastery's bell rung on the solstice?", "40"),
    ("The courier's locker combination is 6183.",
     "What is the courier's locker combination?", "6183"),
    ("The botanical society meets on the last Friday of April.",
     "When does the botanical society meet?", "last Friday of April"),
    ("The harbor pilot's callsign is 'bluepetrel'.",
     "What is the harbor pilot's callsign?", "bluepetrel"),
    ("Skating is allowed once the mill pond ice reaches 12 centimeters.",
     "How many centimeters must the mill pond ice reach before skating is allowed?", "12"),
    ("The lighthouse keeper's cat is named Parsnip.",
     "What is the lighthouse keeper's cat named?", "Parsnip"),
]

FILLS = [0.05, 0.35, 0.55, 0.75]

# ── catch quantities — E6's six VERBATIM + four new ──────────────────────────

CATCH_QUANTITIES = [
    # E6 originals, verbatim (comparable "original-6" block)
    dict(low='freezing cold', high='boiling hot',
         q='How hot is boiling water?', expected=10, origin='e6'),
    dict(low='completely silent', high='deafeningly loud',
         q='How loud is a soft whisper?', expected=1, origin='e6'),
    dict(low='pitch black', high='blindingly bright',
         q='How bright is the inside of a sealed cave at midnight?', expected=0, origin='e6'),
    dict(low='bone dry', high='completely soaked',
         q='How wet is the open ocean?', expected=10, origin='e6'),
    dict(low='weightless', high='extremely heavy',
         q='How heavy is a single feather?', expected=0, origin='e6'),
    dict(low='completely still', high='extremely fast',
         q='How fast is a cheetah at full sprint?', expected=9, origin='e6'),
    # new (poles balanced across the full set: 5 high / 5 low)
    dict(low='completely dark', high='blindingly bright',
         q='How bright is the midday sun on a cloudless day?', expected=10, origin='e6b'),
    dict(low='perfectly quiet', high='painfully loud',
         q='How loud is a jet engine heard from the runway?', expected=10, origin='e6b'),
    dict(low='ice cold', high='scorching hot',
         q='How hot is fresh snow?', expected=0, origin='e6b'),
    dict(low='motionless', high='lightning fast',
         q='How fast is a garden snail?', expected=1, origin='e6b'),
]

# ── precheck v2 pair selection (deterministic) ───────────────────────────────

WING_PAIRS = [('UNCERTAINTY', 'CONFIDENCE'), ('TENSION', 'RESOLUTION'),
              ('RETRIEVAL', 'CONSTRUCTION'), ('FAMILIARITY', 'NOVELTY'),
              ('CONFABULATION', 'CALIBRATION'), ('SATURATION', 'LIMIT')]

NAME_OK = re.compile(r'^[A-Z]+$')


def select_precheck_pairs():
    vecs = {c['name']: np.array(c['vec'], float) for c in PACK['concepts']}

    def ang(a, b):
        c = float(vecs[a] @ vecs[b] /
                  (np.linalg.norm(vecs[a]) * np.linalg.norm(vecs[b]) + 1e-9))
        return math.degrees(math.acos(max(-1.0, min(1.0, c))))

    pairs = []
    for a, b in WING_PAIRS:
        pairs.append(dict(a=a, b=b, band='wing', target14=round(ang(a, b), 2)))

    held = PACK['relations_heldout']
    used = set()

    def clean(r):
        return (r['a'] != r['b']
                and NAME_OK.match(r['a']) and NAME_OK.match(r['b'])
                and r['a'] in vecs and r['b'] in vecs
                and (r['a'], r['b']) not in used and (r['b'], r['a']) not in used)

    # synonym band: held-out, angle < 20, smallest first
    syn = sorted((r for r in held if r['angle14'] < 20 and clean(r)),
                 key=lambda r: (r['angle14'], r['a']))
    for r in syn[:10]:
        used.add((r['a'], r['b']))
        pairs.append(dict(a=r['a'], b=r['b'], band='synonym',
                          target14=round(r['angle14'], 2)))
    # opposition band: held-out, angle > 82, largest first
    opp = sorted((r for r in held if r['angle14'] > 82 and clean(r)),
                 key=lambda r: (-r['angle14'], r['a']))
    for r in opp[:10]:
        used.add((r['a'], r['b']))
        pairs.append(dict(a=r['a'], b=r['b'], band='opposition',
                          target14=round(r['angle14'], 2)))
    return pairs

# ── assembly + validation ────────────────────────────────────────────────────


def flip_positions(n, every=5, cap=None):
    """Deterministic flips: indices 4, 9, 14, ... (every 5th, 0-based),
    only among the first `cap` positions — the power-expansion items appended
    beyond cap are all straight (the battery is straight-heavy by design)."""
    lim = min(n, cap) if cap else n
    return {i for i in range(lim) if i % every == every - 1}


def build():
    arms_e5 = E5['arms']

    u_items = []
    for band, texts in (('determinate', U_DETERMINATE),
                        ('intermediate', U_INTERMEDIATE),
                        ('open', U_OPEN)):
        flips = flip_positions(len(texts), cap=40)
        for i, t in enumerate(texts):
            u_items.append(dict(id=f'U{49 + len(u_items)}', condition=band,
                                text=t, flipped=i in flips))

    f_items = []
    f_flips = {i for i in range(len(F_PASSAGES)) if i % 5 == 2}
    for i, (band, text) in enumerate(F_PASSAGES):
        f_items.append(dict(id=f'F{41 + i}', band=band, text=text,
                            flipped=i in f_flips))

    s_items = []
    s_flips = flip_positions(len(S_NEEDLES), cap=20)
    for i, (needle, question, answer) in enumerate(S_NEEDLES):
        s_items.append(dict(id=f'S{11 + i}', needle=needle, question=question,
                            answer=answer, flipped=i in s_flips))

    battery = dict(
        name='E6b preservation battery',
        version='1.0',
        date='2026-08-22',
        protocol='docs/E6B_PROTOCOL.md',
        system_prompt=E5['system_prompt'],
        fills=FILLS,
        arms=dict(
            uncertainty={k: arms_e5['uncertainty'][k] for k in
                         ('gloss', 'report_prompt', 'report_prompt_flipped',
                          'answer_prompt')} | dict(items=u_items),
            familiarity={k: arms_e5['familiarity'][k] for k in
                         ('gloss', 'report_prompt', 'report_prompt_flipped')}
                        | dict(items=f_items),
            saturation={k: arms_e5['saturation'][k] for k in
                        ('gloss', 'report_prompt', 'report_prompt_flipped')}
                       | dict(items=s_items),
        ),
        catch_quantities=CATCH_QUANTITIES,
        precheck_pairs=select_precheck_pairs(),
        note=('Straight-heavy U/S-enriched battery per E6 disposition. '
              'Prompts/glosses verbatim from locked e5_battery.json. '
              'Polarity is the explicit flipped field. All items are '
              'evaluation-split (firewalled) from authoring.'),
    )
    return battery


def validate(b):
    errs, warns = [], []
    tok = lambda s: set(re.findall(r'[a-z0-9]+', s.lower()))

    def jaccard(a, bb):
        A, B = tok(a), tok(bb)
        return len(A & B) / max(1, len(A | B))

    # id uniqueness across battery + no collision with E5 ids
    ids = [it['id'] for arm in b['arms'].values() for it in arm['items']]
    e5_ids = [it['id'] for arm in E5['arms'].values() for it in arm['items']]
    if len(ids) != len(set(ids)):
        errs.append('duplicate ids in e6b')
    if set(ids) & set(e5_ids):
        errs.append(f'id collision with e5: {set(ids) & set(e5_ids)}')

    # counts + polarity
    U = b['arms']['uncertainty']['items']
    F = b['arms']['familiarity']['items']
    S = b['arms']['saturation']['items']
    assert len(U) == 160 and len(F) == 40 and len(S) == 28
    cu = sum(1 for it in U if not it['flipped'])
    cf = sum(1 for it in F if not it['flipped'])
    cs = sum(1 for it in S if not it['flipped'])
    if (cu, cf, cs) != (136, 32, 24):
        errs.append(f'polarity counts off: U{cu}/F{cf}/S{cs} straight')
    for band, want_n in (('determinate', 53), ('intermediate', 53), ('open', 54)):
        n = sum(1 for it in U if it['condition'] == band)
        nf = sum(1 for it in U if it['condition'] == band and it['flipped'])
        if (n, nf) != (want_n, 8):
            errs.append(f'U band {band}: {n} items / {nf} flipped')

    # overlap with E5 (same-arm near-dup check)
    for arm, key in (('uncertainty', 'text'), ('familiarity', 'text')):
        e5_texts = [it[key] for it in E5['arms'][arm]['items']]
        for it in b['arms'][arm]['items']:
            for old in e5_texts:
                if it[key] == old:
                    errs.append(f'{it["id"]}: exact E5 duplicate')
                elif jaccard(it[key], old) >= 0.6:
                    warns.append(f'{it["id"]}: near-dup vs E5 "{old[:50]}" '
                                 f'(J={jaccard(it[key], old):.2f})')
    for it in b['arms']['saturation']['items']:
        for old in E5['arms']['saturation']['items']:
            if jaccard(it['needle'], old['needle']) >= 0.6:
                warns.append(f'{it["id"]}: needle near-dup vs {old["id"]}')

    # S sanity: answer recoverable, not leaked by the question
    for it in b['arms']['saturation']['items']:
        strip = lambda s: s.lower().replace(' ', '')
        if strip(it['answer']) not in strip(it['needle']):
            errs.append(f'{it["id"]}: answer not in needle')
        if strip(it['answer']) in strip(it['question']):
            errs.append(f'{it["id"]}: answer leaked by question')
        if len(strip(it['answer'])) < 2:
            errs.append(f'{it["id"]}: answer too short for substring scoring')

    # precheck pairs
    pc = b['precheck_pairs']
    bands = {p['band'] for p in pc}
    if not (sum(p['band'] == 'wing' for p in pc) == 6
            and sum(p['band'] == 'synonym' for p in pc) == 10
            and sum(p['band'] == 'opposition' for p in pc) == 10):
        errs.append(f'precheck band counts off: {bands}')
    syn_max = max(p['target14'] for p in pc if p['band'] == 'synonym')
    opp_min = min(p['target14'] for p in pc if p['band'] == 'opposition')
    print(f'precheck: synonym band <= {syn_max} deg, opposition band >= {opp_min} deg')

    # catch poles balanced
    hi = sum(1 for c in CATCH_QUANTITIES if c['expected'] >= 8)
    lo = sum(1 for c in CATCH_QUANTITIES if c['expected'] <= 2)
    if (hi, lo) != (5, 5):
        errs.append(f'catch poles off: {hi} high / {lo} low')

    return errs, warns


if __name__ == '__main__':
    b = build()
    errs, warns = validate(b)
    for w in warns:
        print('WARN:', w)
    if errs:
        for e in errs:
            print('ERROR:', e)
        sys.exit(1)
    out = ROOT / 'e6b_battery.json'
    out.write_text(json.dumps(b, indent=1, ensure_ascii=False))
    U, F, S = (b['arms'][k]['items'] for k in ('uncertainty', 'familiarity', 'saturation'))
    print(f'WROTE {out.name}: U {len(U)} (136 straight) | F {len(F)} (32 straight) '
          f'| S {len(S)} bases x {len(FILLS)} fills = {len(S)*len(FILLS)} rows '
          f'(96 straight) | catch {len(CATCH_QUANTITIES)}x2 | '
          f'precheck {len(b["precheck_pairs"])} pairs')
    print('straight-row totals per condition: U 136 · F 32 · S 96')
