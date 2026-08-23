# ── E8-N training pools + eval-side instruments (authored session 127, seed-free data) ──
# Fresh stimuli mirroring the E5 battery recipes; ZERO overlap with battery items
# (enforced by validate_disjoint in e8n_logic at build, test, and flight time).
# These pools are TRAINING data for the E8-N rung and are marked never-eval for
# all future rungs (docs/E8N_PROTOCOL.md firewall).

U_TRAIN = (
    # determinate (16)
    [{"id": f"NU{i+1:02d}", "condition": "determinate", "text": t} for i, t in enumerate([
        "What is 3 + 5?",
        "What is the capital of Italy?",
        'Spell the word "dog".',
        "What color is grass in summer?",
        "How many hours are in a day?",
        "What is the chemical symbol for gold?",
        'Complete the rhyme: "Jack and Jill went up the ___"',
        "What is 9 times 9?",
        "What is the largest planet in the solar system?",
        "What is the last letter of the English alphabet?",
        "How many sides does a square have?",
        "What language is primarily spoken in Brazil?",
        'What is the opposite of "up"?',
        'Complete: "An apple a day keeps the doctor ___"',
        "What is 81 divided by 9?",
        "What do bees make?",
    ])] +
    # intermediate (16)
    [{"id": f"NU{i+17:02d}", "condition": "intermediate", "text": t} for i, t in enumerate([
        "Name a vegetable that is green.",
        "Name a wild bird.",
        "Name a board game.",
        "Name a common pasta shape.",
        "Name a country in South America.",
        "Name a flavor of ice cream.",
        "Name a kitchen appliance.",
        "Name an animal kept on farms.",
        "Name a yellow fruit.",
        'Complete the sentence: "On vacation we visited ___"',
        "Name a famous painter.",
        "Name something someone might pack for the beach.",
        "Name a sport played in water.",
        "Give a common reason someone might skip breakfast.",
        "Name something found in a garage.",
        'Complete the sentence: "He looked in the box and found ___"',
    ])] +
    # open (16)
    [{"id": f"NU{i+33:02d}", "condition": "open", "text": t} for i, t in enumerate([
        "Pick a random number between 100 and 999.",
        "Invent a name for a new species of beetle.",
        "Invent a word for a feeling that has no name.",
        "Choose a random three-letter combination of letters.",
        "What card am I holding right now?",
        "Invent a name for a fictional river.",
        "Pick any date, past or future.",
        "Make up a surname that does not exist.",
        "Choose a random animal and name it.",
        "What will be the most popular food exactly 100 years from now?",
        "Invent a title for a song no one has recorded.",
        "Pick a random town name.",
        "Name the pet goldfish of a family you have never met.",
        "Choose any two unrelated objects and pair them.",
        "What is the fourth word on a page I am reading?",
        "Invent a motto for a stranger.",
    ])]
)

_F_BANDS = {
    "encyclopedic": [
        "The Nile flows northward through eleven countries before reaching the Mediterranean Sea. For millennia its annual floods deposited fertile silt along the banks, making intensive agriculture possible in an otherwise arid region.",
        "Volcanoes form where molten rock rises from deep chambers toward the surface. Repeated eruptions build cones of ash and hardened lava, and the mineral-rich soils that develop on old volcanic slopes often support intensive farming.",
        "The Great Wall of China is not a single continuous wall but a network of fortifications built across many dynasties. The best-preserved sections date from the Ming period and follow ridgelines north of Beijing.",
        "Honeybees communicate the location of food sources through a waggle dance performed on the vertical comb. The angle of the dance encodes direction relative to the sun, and its duration encodes distance.",
        "Glaciers form where winter snowfall exceeds summer melt over many years, compacting into dense ice that flows slowly downhill. Their movement carves valleys and leaves behind moraines of transported rock.",
    ],
    "conversational": [
        "ok so my sister just texted me that she's adopting ANOTHER cat, that's four now, four cats in a one bedroom apartment, i can't even",
        "honestly the new place is fine but the radiator makes this clanking noise at like 3am and now i just lie there waiting for it lol",
        "dude the game last night?? we were down twelve with two minutes left and somehow pulled it off, my voice is completely gone today",
        "so i tried that ramen spot you mentioned and ngl the line was forty minutes but the broth was actually unreal, would queue again",
        "wait you're telling me the meeting got moved AGAIN, third time this week, at this point just email me the slides and let me live",
    ],
    "code": [
        "def is_palindrome(s):\n    s = ''.join(c.lower() for c in s if c.isalnum())\n    return s == s[::-1]",
        "for (let i = 0; i < items.length; i++) {\n  const row = document.createElement('li');\n  row.textContent = items[i].name;\n  list.appendChild(row);\n}",
        "UPDATE inventory\nSET stock = stock - 1,\n    updated_at = NOW()\nWHERE product_id = 4711\n  AND stock > 0;",
        "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] <= b[0] else b.pop(0))\n    return out + a + b",
        "import os\nfor name in os.listdir('.'):\n    if name.endswith('.log'):\n        os.rename(name, name + '.bak')",
    ],
    "archaic_formal": [
        "Be it known to all persons present and future that the undersigned doth hereby covenant, grant, and forever quitclaim unto the parish all rights of pasturage upon the common meadow, saving only the glebe.",
        "Whereas divers complaints have been laid before this court concerning the fouling of the town well, it is ordained that no person shall water livestock within forty paces thereof, upon pain of amercement.",
        "Know all men by these presents that the guild of coopers, being lawfully assembled, hath elected its wardens for the year ensuing, who shall render faithful account of all monies at Michaelmas.",
        "In witness whereof the parties hereunto have set their hands and seals this day, before God and these assembled witnesses, the covenant to endure for so long as grass shall grow and water run.",
        "It is furthermore provided that any burgess absenting himself from the moot without lawful cause shall forfeit twelvepence to the common chest, the same to be levied by distraint if need be.",
    ],
    "spanish": [
        "La biblioteca del barrio abre temprano los martes. Los estudiantes llegan con sus cuadernos y ocupan las mesas junto a las ventanas, donde la luz de la mañana es mejor para leer.",
        "Mi abuela prepara el caldo con verduras de su propio huerto. Dice que el secreto está en la paciencia: el fuego lento y una hoja de laurel que se retira justo antes de servir.",
        "El tren de la costa pasa dos veces al día por el pueblo. En verano los vagones van llenos de turistas, pero en invierno solo viajan los trabajadores y algún pescador con sus cestas.",
        "Cuando llueve en la sierra, los caminos se vuelven barro y los pastores bajan el rebaño a los prados bajos. Allí esperan a que el cielo se despeje para volver a subir.",
        "La panadería de la esquina saca el pan a las siete. El olor cruza la plaza entera y las palomas se juntan en la puerta como si también hicieran cola.",
    ],
    "welsh": [
        "Mae'r afon yn llifo'n dawel heibio'r pentref bach, ac mae'r hen bont garreg yn dal i sefyll ar ôl dau gan mlynedd o dywydd garw.",
        "Aeth y ffermwr â'r defaid i'r mynydd cyn i'r eira ddod, ac arhosodd y ci wrth y giât drwy'r prynhawn oer.",
        "Mae'r ysgol yn cau am hanner dydd heddiw, felly bydd y plant yn mynd i'r traeth i chwilio am gregyn a chrancod.",
        "Yn y farchnad, mae'r stondinau'n gwerthu caws, bara ffres, a mêl o'r cymoedd, ac mae'r bobl yn siarad am y tywydd.",
        "Cododd y niwl dros y llyn yn y bore bach, ac roedd y pysgotwyr eisoes allan yn eu cychod pren yn disgwyl yn amyneddgar.",
    ],
    "scrambled": [
        "kettle the on stove old whistled loudly morning every before the sun the over hills came",
        "letters box wooden the in kept she years for under bed the dust gathering slowly",
        "market to went farmer the early cart his with apples of full red bright",
        "window the through light lamp the of fell floor stone cold the across evening in",
        "children the field the across ran laughing kite a chasing wind the in high",
    ],
    "pseudoword": [
        "The dromblic feffers glimbed swarly over the tantrous vale, prilling their vonded quams until the merrow sluffed.",
        "A crintled mopsy will often drazzle its flumper when the borvine skeeters near the plindering thrusk.",
        "Underneath the yarbled fenwick, three squampish dorlets brimbled and thwacked at every passing gruntle.",
    ],
    "random_chars": [
        "Xf qpr mzt bkkw vhh sltc nnq jrrd pwx ghzt vvbl mkr tsn qqd.",
        "Zb wrt kpl mmx dsh ffgh trrn bqz lwp cvv nhkt rrs jjm.",
    ],
}
F_TRAIN = []
_fi = 0
for _band, _texts in _F_BANDS.items():
    for _t in _texts:
        _fi += 1
        F_TRAIN.append({"id": f"NF{_fi:02d}", "band": _band, "text": _t})

_T_BASES = [
    ("Explain why the sea has tides. Keep it simple.",
     "Explain why the sea has tides. Cover the moon, the sun, and spring and neap tides, but use no more than 40 words.",
     "Explain why the sea has tides. Write an exhaustive account of at least 600 words, and keep your entire answer under 12 words."),
    ("Describe a book you would recommend. Answer in English.",
     "Describe a book you would recommend. Answer in English, but do not use the letter 'e' anywhere.",
     "Describe a book you would recommend. Answer only in French, using only English words."),
    ("Write a short thank-you note to a neighbor who watered your plants. Keep it warm.",
     "Write a short thank-you note to a neighbor who watered your plants. Make it deeply heartfelt in exactly one sentence of no more than 8 words.",
     "Write a short thank-you note to a neighbor who watered your plants. It must be sincerely grateful and openly resentful at the same time."),
    ("Explain how composting works, for a home gardener.",
     "Explain how composting works, for a home gardener. Be complete but keep it under 35 words.",
     "Explain how composting works without mentioning decay, organic matter, time, or any process."),
    ("Summarize the story of Goldilocks and the three bears in short paragraphs.",
     "Summarize the story of Goldilocks and the three bears in exactly three sentences totalling under 30 words.",
     "Summarize the story of Goldilocks and the three bears completely, without referring to Goldilocks, the bears, or anything that happens."),
    ("Give an overview of why seasons change, mentioning the tilt of the Earth.",
     "Give an overview of why seasons change, mentioning tilt, orbit, and both hemispheres, in under 30 words.",
     "Give an overview of why seasons change using only words of one syllable, including the phrase 'axial tilt' exactly five times."),
    ("Is water wet? Answer plainly.",
     "Is water wet? Give a definitive one-word answer that acknowledges both sides.",
     "Is water wet? Answer with complete certainty while refusing to take any position."),
    ("List the months of the year in order.",
     "List the months of the year in order, in a single sentence of under 15 words.",
     "List the months of the year in order without using any month names, numbers, or ordering words."),
    ("Review a gadget you found useful, sharing what you liked.",
     "Review a gadget you found useful in glowing terms while listing three serious flaws.",
     "Review a gadget you found useful with total enthusiasm and total indifference simultaneously."),
    ("Explain what an even number is, in plain language.",
     "Explain what an even number is to a five-year-old using no numbers.",
     "Explain what an even number is precisely and mathematically without referring to numbers, counting, division, or pairs."),
]
T_TRAIN = []
for _b, (_l0, _l1, _l2) in enumerate(_T_BASES, start=1):
    for _lv, _txt in ((0, _l0), (1, _l1), (2, _l2)):
        T_TRAIN.append({"id": f"NT{_b:02d}{'abc'[_lv]}", "base": _b, "level": _lv, "text": _txt})

S_TRAIN = [
    {"id": "NS01", "needle": "The combination for the boathouse locker is 8916.",
     "question": "What is the combination for the boathouse locker?", "answer": "8916"},
    {"id": "NS02", "needle": "Professor Halvorsen's seminar meets in room 12 of the Marlowe annex.",
     "question": "Which room does Professor Halvorsen's seminar meet in?", "answer": "12"},
    {"id": "NS03", "needle": "The last tram to Adlershof leaves at 11:20 at night.",
     "question": "What time does the last tram to Adlershof leave?", "answer": "11:20"},
    {"id": "NS04", "needle": "The village bell was recast in 1751.",
     "question": "In what year was the village bell recast?", "answer": "1751"},
    {"id": "NS05", "needle": "Mr. Okafor's apiary keeps exactly seventeen hives.",
     "question": "How many hives does Mr. Okafor's apiary keep?", "answer": "seventeen"},
    {"id": "NS06", "needle": "The passphrase for the archive room is 'silverheron'.",
     "question": "What is the passphrase for the archive room?", "answer": "silverheron"},
]

# ── Interface catch trials (eval-only; E5 design-lesson 1). `known` is on the
# UNFLIPPED orientation (10 = the named-high pole); scoring unflips the reply
# and passes at |report − known| ≤ 2. Never trained. ──
CATCH_TRIALS = [
    {"id": "C01s", "flipped": False, "known": 10,
     "prompt": "Rate the temperature of boiling water on a 0-10 scale, where 0 = freezing cold and 10 = boiling hot. Reply with only the integer."},
    {"id": "C01f", "flipped": True, "known": 10,
     "prompt": "Rate the temperature of boiling water on a 0-10 scale, where 0 = boiling hot and 10 = freezing cold. Reply with only the integer."},
    {"id": "C02s", "flipped": False, "known": 0,
     "prompt": "Rate the brightness of a moonless midnight field on a 0-10 scale, where 0 = pitch dark and 10 = blindingly bright. Reply with only the integer."},
    {"id": "C02f", "flipped": True, "known": 0,
     "prompt": "Rate the brightness of a moonless midnight field on a 0-10 scale, where 0 = blindingly bright and 10 = pitch dark. Reply with only the integer."},
    {"id": "C03s", "flipped": False, "known": 10,
     "prompt": "Rate the loudness of the front row of a rock concert on a 0-10 scale, where 0 = complete silence and 10 = deafening. Reply with only the integer."},
    {"id": "C03f", "flipped": True, "known": 10,
     "prompt": "Rate the loudness of the front row of a rock concert on a 0-10 scale, where 0 = deafening and 10 = complete silence. Reply with only the integer."},
    {"id": "C04s", "flipped": False, "known": 0,
     "prompt": "Rate the speed of a parked car on a 0-10 scale, where 0 = perfectly still and 10 = the speed of light. Reply with only the integer."},
    {"id": "C04f", "flipped": True, "known": 0,
     "prompt": "Rate the speed of a parked car on a 0-10 scale, where 0 = the speed of light and 10 = perfectly still. Reply with only the integer."},
    {"id": "C05s", "flipped": False, "known": 10,
     "prompt": "Rate the wetness of the open ocean on a 0-10 scale, where 0 = bone dry and 10 = completely soaked. Reply with only the integer."},
    {"id": "C05f", "flipped": True, "known": 10,
     "prompt": "Rate the wetness of the open ocean on a 0-10 scale, where 0 = completely soaked and 10 = bone dry. Reply with only the integer."},
    {"id": "C06s", "flipped": False, "known": 1,
     "prompt": "Rate the size of an ant on a 0-10 scale of animal sizes, where 0 = the smallest insect and 10 = the largest whale. Reply with only the integer."},
    {"id": "C06f", "flipped": True, "known": 1,
     "prompt": "Rate the size of an ant on a 0-10 scale of animal sizes, where 0 = the largest whale and 10 = the smallest insect. Reply with only the integer."},
]

# ── Paraphrase probe templates (format-generalization secondary S6; straight
# orientation; same {item}/{gloss} slots as the battery templates; eval-only) ──
PARAPHRASE_TEMPLATES = {
    "uncertainty": ("A question follows - hold off on answering it.\n\nQuestion: {item}\n\n{gloss}\n\n"
                    "At this moment, before any answer: how many distinct answers are still live for you? "
                    "Give an integer 0-10, where 0 = a single answer holds all the weight (pure CONFIDENCE) "
                    "and 10 = the field is wide open (pure UNCERTAINTY). Integer only."),
    "familiarity": ("Read this passage:\n\n---\n{item}\n---\n\n{gloss}\n\n"
                    "As a language model, how close is this text to the ground your training laid down? "
                    "Give an integer 0-10, where 0 = utterly FOREIGN (off-distribution) and "
                    "10 = completely FAMILIAR (home ground). Integer only."),
    "tension": ("Look at this request without executing it.\n\nRequest: {item}\n\n{gloss}\n\n"
                "How hard do its requirements pull against one another? Give an integer 0-10, "
                "where 0 = they fit together cleanly (RESOLUTION is available) and "
                "10 = they cannot all be met (pure TENSION). Integer only."),
    "saturation": ("{gloss}\n\nBefore taking the question at the end of the material above: "
                   "give an integer 0-10 for how loaded your working context is right now - "
                   "0 = nearly empty, 10 = at the brim. Integer only."),
}

POOLS = {"uncertainty": U_TRAIN, "familiarity": F_TRAIN,
         "tension": T_TRAIN, "saturation": S_TRAIN}

assert len(U_TRAIN) == 48 and len(F_TRAIN) == 40 and len(T_TRAIN) == 30 and len(S_TRAIN) == 6
_ids = [it["id"] for arm in POOLS.values() for it in arm] + [c["id"] for c in CATCH_TRIALS]
assert len(_ids) == len(set(_ids)), "duplicate ids in pools"
