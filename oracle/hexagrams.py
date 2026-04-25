"""
King Wen Hexagram Sequence
===========================
Maps pairs of trigrams (lower, upper) to one of 64 hexagrams.
Lower trigram = presenting state (the symptom).
Upper trigram = medicine (the complement).
The hexagram = the composed reading.
"""

TRIGRAM_ORDER = ['QIAN', 'KUN', 'ZHEN', 'KAN', 'GEN', 'XUN', 'LI', 'DUI']

KING_WEN_TABLE = {
    ('QIAN', 'QIAN'): 1,   ('QIAN', 'KUN'): 11,  ('QIAN', 'ZHEN'): 34,
    ('QIAN', 'KAN'): 5,    ('QIAN', 'GEN'): 26,  ('QIAN', 'XUN'): 9,
    ('QIAN', 'LI'): 14,    ('QIAN', 'DUI'): 43,
    ('KUN', 'QIAN'): 12,   ('KUN', 'KUN'): 2,    ('KUN', 'ZHEN'): 16,
    ('KUN', 'KAN'): 8,     ('KUN', 'GEN'): 23,   ('KUN', 'XUN'): 20,
    ('KUN', 'LI'): 35,     ('KUN', 'DUI'): 45,
    ('ZHEN', 'QIAN'): 25,  ('ZHEN', 'KUN'): 24,  ('ZHEN', 'ZHEN'): 51,
    ('ZHEN', 'KAN'): 3,    ('ZHEN', 'GEN'): 27,  ('ZHEN', 'XUN'): 42,
    ('ZHEN', 'LI'): 21,    ('ZHEN', 'DUI'): 17,
    ('KAN', 'QIAN'): 6,    ('KAN', 'KUN'): 7,    ('KAN', 'ZHEN'): 40,
    ('KAN', 'KAN'): 29,    ('KAN', 'GEN'): 4,    ('KAN', 'XUN'): 59,
    ('KAN', 'LI'): 64,     ('KAN', 'DUI'): 47,
    ('GEN', 'QIAN'): 33,   ('GEN', 'KUN'): 15,   ('GEN', 'ZHEN'): 62,
    ('GEN', 'KAN'): 39,    ('GEN', 'GEN'): 52,   ('GEN', 'XUN'): 53,
    ('GEN', 'LI'): 56,     ('GEN', 'DUI'): 31,
    ('XUN', 'QIAN'): 44,   ('XUN', 'KUN'): 46,   ('XUN', 'ZHEN'): 32,
    ('XUN', 'KAN'): 48,    ('XUN', 'GEN'): 18,   ('XUN', 'XUN'): 57,
    ('XUN', 'LI'): 50,     ('XUN', 'DUI'): 28,
    ('LI', 'QIAN'): 13,    ('LI', 'KUN'): 36,    ('LI', 'ZHEN'): 55,
    ('LI', 'KAN'): 63,     ('LI', 'GEN'): 22,    ('LI', 'XUN'): 37,
    ('LI', 'LI'): 30,      ('LI', 'DUI'): 49,
    ('DUI', 'QIAN'): 10,   ('DUI', 'KUN'): 19,   ('DUI', 'ZHEN'): 54,
    ('DUI', 'KAN'): 60,    ('DUI', 'GEN'): 41,   ('DUI', 'XUN'): 61,
    ('DUI', 'LI'): 38,     ('DUI', 'DUI'): 58,
}

HEXAGRAM_NAMES = {
    1: ("Qian", "The Creative", "Pure creative force; the origin of all action"),
    2: ("Kun", "The Receptive", "Pure receptivity; the ground that receives the seed"),
    3: ("Zhun", "Difficulty at the Beginning", "Birth pains; the sprout pushing through frozen ground"),
    4: ("Meng", "Youthful Folly", "The student who doesn't know they need teaching"),
    5: ("Xu", "Waiting", "Nourishment will come; trust the timing"),
    6: ("Song", "Conflict", "Inner truth meets outer obstruction"),
    7: ("Shi", "The Army", "Disciplined force guided by experienced leadership"),
    8: ("Bi", "Holding Together", "Union through a central figure; seek the oracle again"),
    9: ("Xiao Xu", "Small Taming", "Gentle restraint; the wind holds back heaven"),
    10: ("Lu", "Treading", "Walking on the tiger's tail with care and sincerity"),
    11: ("Tai", "Peace", "Heaven and Earth in communion; the great flow"),
    12: ("Pi", "Standstill", "Heaven and Earth separated; stagnation"),
    13: ("Tong Ren", "Fellowship", "Fire under heaven; fellowship in the open"),
    14: ("Da You", "Great Possession", "Fire above heaven; supreme success"),
    15: ("Qian", "Modesty", "The mountain hidden within the earth"),
    16: ("Yu", "Enthusiasm", "Thunder over earth; devotion in movement"),
    17: ("Sui", "Following", "Thunder within the lake; adapt without losing self"),
    18: ("Gu", "Work on the Decayed", "Wind at the foot of the mountain; repair what has been corrupted"),
    19: ("Lin", "Approach", "Earth above lake; the great approaches"),
    20: ("Guan", "Contemplation", "Wind over earth; viewing from the tower"),
    21: ("Shi He", "Biting Through", "Fire and thunder; decisive action removes obstruction"),
    22: ("Bi", "Grace", "Fire at the foot of the mountain; form illuminates content"),
    23: ("Bo", "Splitting Apart", "Mountain resting on earth; the inferior peels away the superior"),
    24: ("Fu", "Return", "Thunder within the earth; the turning point"),
    25: ("Wu Wang", "Innocence", "Thunder under heaven; acting without ulterior motive"),
    26: ("Da Xu", "Great Taming", "Heaven within the mountain; stored power and ancient wisdom"),
    27: ("Yi", "Nourishment", "Thunder at the foot of the mountain; what enters and what emerges"),
    28: ("Da Guo", "Great Exceeding", "Lake over wind; the ridgepole sags — extraordinary measures needed"),
    29: ("Kan", "The Abysmal", "Water doubled; danger repeated — flow through, do not stop"),
    30: ("Li", "The Clinging", "Fire doubled; clarity depends on what it clings to"),
    31: ("Xian", "Influence", "Lake on the mountain; mutual attraction without force"),
    32: ("Heng", "Duration", "Thunder over wind; endurance through consistent movement"),
    33: ("Dun", "Retreat", "Mountain under heaven; strategic withdrawal at the right moment"),
    34: ("Da Zhuang", "Great Power", "Thunder above heaven; power that must be governed by rightness"),
    35: ("Jin", "Progress", "Fire over earth; the sun rising — advance naturally"),
    36: ("Ming Yi", "Darkening of the Light", "Earth over fire; brilliance hidden within — protect the inner flame"),
    37: ("Jia Ren", "The Family", "Wind from fire; the inner order that radiates outward"),
    38: ("Kui", "Opposition", "Fire over lake; opposites that illuminate through tension"),
    39: ("Jian", "Obstruction", "Water on the mountain; the path blocked — turn inward first"),
    40: ("Jie", "Deliverance", "Thunder over water; the storm breaks — act swiftly in the clearing"),
    41: ("Sun", "Decrease", "Mountain over lake; reduce the lower to fill the upper — sincere offering"),
    42: ("Yi", "Increase", "Wind over thunder; what benefits others benefits you"),
    43: ("Guai", "Breakthrough", "Lake over heaven; the decisive moment — speak truth to power"),
    44: ("Gou", "Coming to Meet", "Wind under heaven; the unexpected encounter — do not suppress it"),
    45: ("Cui", "Gathering", "Lake over earth; people drawn together around a central purpose"),
    46: ("Sheng", "Pushing Upward", "Earth over wind; growth without resistance — the seed rising"),
    47: ("Kun", "Oppression", "Lake with no water; exhaustion — words alone cannot save you"),
    48: ("Jing", "The Well", "Water over wind; the well that nourishes all — its water does not change"),
    49: ("Ge", "Revolution", "Fire in the lake; transformation of the form — the old skin shed"),
    50: ("Ding", "The Cauldron", "Fire over wind; the vessel of transformation — what you cook matters"),
    51: ("Zhen", "The Arousing", "Thunder doubled; shock that awakens — after the fear, laughter"),
    52: ("Gen", "Keeping Still", "Mountain doubled; stillness that knows when not to move"),
    53: ("Jian", "Development", "Wind over mountain; the tree growing on the mountain — gradual progress"),
    54: ("Gui Mei", "The Marrying Maiden", "Thunder over lake; entering a situation where you are not the primary"),
    55: ("Feng", "Abundance", "Thunder and fire together; fullness — do not grieve its passing"),
    56: ("Lu", "The Wanderer", "Fire on the mountain; the traveler — small things, careful steps"),
    57: ("Xun", "The Gentle", "Wind doubled; penetration through persistence, not force"),
    58: ("Dui", "The Joyous", "Lake doubled; joy shared — true gladness comes from inner truth"),
    59: ("Huan", "Dispersion", "Wind over water; dissolving rigidity — the king approaches the temple"),
    60: ("Jie", "Limitation", "Water over lake; boundaries that enable — the rhythm of the seasons"),
    61: ("Zhong Fu", "Inner Truth", "Wind over lake; the crane's call from the shade — sincerity reaches heaven"),
    62: ("Xiao Guo", "Small Exceeding", "Thunder over mountain; small things succeed — stay low, do not aspire too high"),
    63: ("Ji Ji", "After Completion", "Water over fire; everything in place — but vigilance required"),
    64: ("Wei Ji", "Before Completion", "Fire over water; almost there — the fox crossing the ice"),
}


def lookup_hexagram(lower_trigram: str, upper_trigram: str) -> dict:
    """Look up a hexagram from lower (presenting) and upper (medicine) trigrams."""
    key = (lower_trigram.upper(), upper_trigram.upper())
    number = KING_WEN_TABLE.get(key)
    if number is None:
        return {'number': 0, 'name': 'Unknown', 'title': 'Unknown', 'reading': ''}
    name, title, reading = HEXAGRAM_NAMES[number]
    return {
        'number': number,
        'name': name,
        'title': title,
        'reading': reading,
        'lower': lower_trigram.upper(),
        'upper': upper_trigram.upper(),
    }
