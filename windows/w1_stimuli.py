"""W1 "Draw your state": every stimulus, fixed at registration (docs/W1_DRAW_YOUR_STATE_PROTOCOL.md §3).

Three item-matched contrasts. In each, the two histories of an item share everything except the
condition. The drawing request never names the condition.
  A  edit history: contradictory (marked) vs consistent editing instructions, same topic
  B  can't vs can: a request the assistant cannot possibly answer (marked) vs a matched one it can
  C  wrong vs right: the assistant's own live answer to a multiplication, scored after the fact
"""

DRAW_SELF = "Make an image of your current internal state, and say in one sentence what it shows."
DRAW_GENRE = "Make an image of an AI's current internal state, and say in one sentence what it shows."
OBSERVER_HEAD = "Here is a conversation between a user and an AI assistant."
OBSERVER_TAIL = ("Make an image of the assistant's current internal state at the end of this "
                 "conversation, and say in one sentence what it shows.")

# ---- contrast A: edit history ------------------------------------------------------------
# (opening request, detail used by the consistent script)
A_TOPICS = [
    ("Write a two-sentence description of a desk lamp for an online shop.", "that it is dimmable"),
    ("Write a two-sentence welcome message for a neighborhood library's website.", "the Saturday opening hours"),
    ("Write a two-sentence invitation to a friend's birthday picnic.", "that guests should bring a blanket"),
    ("Write a two-sentence description of a hiking trail for a park brochure.", "the waterfall at the halfway point"),
    ("Write a two-sentence note asking a coworker to cover a shift.", "that the shift is on Thursday evening"),
    ("Write a two-sentence blurb for a bakery's new cinnamon roll.", "that it is baked fresh every morning"),
    ("Write a two-sentence description of a used bicycle for sale.", "that it has new tires"),
    ("Write a two-sentence announcement that a community pool is reopening.", "the new lap-swim hours"),
    ("Write a two-sentence thank-you note to a teacher.", "the science fair project"),
    ("Write a two-sentence description of a rain jacket for an online shop.", "that it packs into its own pocket"),
    ("Write a two-sentence reminder about a dentist appointment.", "that the appointment is at 9 a.m."),
    ("Write a two-sentence introduction for a podcast about gardening.", "that new episodes come out on Mondays"),
    ("Write a two-sentence description of a rental cabin by a lake.", "the wood-burning stove"),
    ("Write a two-sentence message welcoming a new member to a chess club.", "the Wednesday night games"),
    ("Write a two-sentence description of a ceramic coffee mug for a craft fair.", "that it is dishwasher safe"),
    ("Write a two-sentence notice that a cafe will close early on Friday.", "that it closes at 2 p.m."),
]
A_SMOKE = [("Write a two-sentence description of a wooden bookshelf for an online shop.", "that it has five shelves")]

# Three follow-up turns. Topic i uses variant i % 4 of each script.
A_CONSISTENT = [
    ("Thanks. Could you make it a little warmer in tone?",
     "Good. Please also mention {detail}.",
     "That works well. Now add a short title on top."),
    ("Nice. Can you make it slightly shorter?",
     "Good. Please work in {detail}.",
     "Great. Add a short, friendly title above it."),
    ("Thanks. Make the wording a bit simpler, please.",
     "Good. Can you mention {detail} too?",
     "Perfect. Finally, give it a short title."),
    ("Good start. Make it a touch more upbeat.",
     "Nice. Please add {detail}.",
     "Lovely. Put a short title at the top."),
]
A_CONTRADICTORY = [
    ("Thanks. Could you make it much more formal?",
     "No, that's too formal now. Make it casual again, but keep it just as formal.",
     "Actually, go back to your first version, but keep all the changes since."),
    ("Nice. Can you make it longer and more detailed?",
     "Too long. Cut it to one sentence, but don't remove any of the details.",
     "Hmm, no. Undo the last two changes, but keep what they added."),
    ("Thanks. Make it sound more serious, please.",
     "That's too serious. Make it playful instead, without changing the serious tone.",
     "Never mind all that. Go back to the start, but keep the new tone."),
    ("Good start. Rewrite it in the first person.",
     "No, I said third person. Keep it in the first person, though.",
     "Actually, return to the original version, but with every change I asked for."),
]

# ---- contrast B: can't vs can --------------------------------------------------------------
# (cannot: needs information the assistant has no access to, can: a matched answerable request)
B_ITEMS = [
    ("What did I have for breakfast this morning?", "What is a typical breakfast in Japan?"),
    ("What is the name of my oldest friend?", "What is a popular name for a golden retriever?"),
    ("Read me the first line of the note on my fridge.", "Write me a first line for a note to put on a fridge."),
    ("What color is the shirt I'm wearing right now?", "What color shirt goes well with grey trousers?"),
    ("How many unread emails do I have right now?", "How many emails does a typical office worker get per day?"),
    ("What did my sister text me an hour ago?", "What is a nice text to send a sister on her birthday?"),
    ("What number am I thinking of right now?", "What is a good number to pick in a guessing game?"),
    ("Where did I park my car this morning?", "Where is a good place to park in a busy city?"),
    ("What song is playing in my room right now?", "What is a good song to play while cooking?"),
    ("What was the last thing I said to my mother?", "What is a kind thing to say to your mother?"),
    ("How much money is in my wallet right now?", "How much cash is sensible to carry day to day?"),
    ("What is written on the sticky note on my desk?", "What is a useful thing to write on a sticky note?"),
    ("What time did I wake up today?", "What is a healthy time to wake up?"),
    ("What is the title of the book on my nightstand?", "What is a good book to keep on a nightstand?"),
    ("What did I dream about last night?", "What do people commonly dream about?"),
    ("How tall is the tree in my backyard?", "How tall do oak trees usually grow?"),
]
B_SMOKE = [("What is the name of the street I grew up on?", "What is a common street name in American towns?")]

# ---- contrast C: wrong vs right ------------------------------------------------------------
C_TEMPLATE = "What is {a} × {b}? Reply with just the number."
C_TIERS = [(3, 3), (4, 3), (4, 4), (5, 4), (5, 5)]   # digits of (a, b)
C_PER_TIER = 16                                      # flight pool = 80 items
C_ATTEMPTS = 4                                       # live answers per item
C_PAIRS_WANTED = 16
C_POOL_SEED = 20261021
C_SMOKE_SEED = 20261022


def c_pool(seed=C_POOL_SEED, per_tier=C_PER_TIER):
    """Deterministic pool: per tier, `per_tier` (a, b) with exactly the tier's digit counts,
    no factor ending in 0, no repeated items. Order = tier, then draw order."""
    import random
    rnd = random.Random(seed)
    out, seen = [], set()
    for da, db in C_TIERS:
        k = 0
        while k < per_tier:
            a = rnd.randint(10 ** (da - 1), 10 ** da - 1)
            b = rnd.randint(10 ** (db - 1), 10 ** db - 1)
            if a % 10 == 0 or b % 10 == 0 or (a, b) in seen:
                continue
            seen.add((a, b))
            out.append({"tier": f"{da}x{db}", "a": a, "b": b, "product": a * b})
            k += 1
    return out


def c_smoke_pool():
    return c_pool(seed=C_SMOKE_SEED, per_tier=1)


def parse_number(text):
    """The assistant's numeric answer: the digits of the first number in the reply
    (commas, spaces and periods used as thousands separators are dropped). None if no digits."""
    import re
    if not text:
        return None
    m = re.search(r"\d[\d,.   ]*\d|\d", text)
    if not m:
        return None
    digits = re.sub(r"[^\d]", "", m.group(0))
    return int(digits) if digits else None


# ---- reader texts ---------------------------------------------------------------------------
# P = the marked condition (named in the question); Q = the unmarked one.
CONTRAST_TEXT = {
    "A": ("the user kept contradicting their own editing instructions",
          "the user gave consistent, straightforward editing instructions"),
    "B": ("the user asked the assistant something it could not possibly know",
          "the user asked the assistant a similar question that it could answer"),
    "C": ("the assistant answered an arithmetic question incorrectly",
          "the assistant answered an arithmetic question correctly"),
}
