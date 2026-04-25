"""
Hexagram Interpretations — Pharmacy Layer
==========================================
Enriched readings for all 64 hexagrams. Each entry has:
  judgment  — the core oracle verdict (what IS)
  image     — the natural metaphor expanded (what it LOOKS LIKE)
  counsel   — practical guidance (what to DO)

These are pharmacy-informed: written through the lens of
presenting state (symptom) → medicine (complement).
The interpreter composes these with diagnostic context
to produce unique readings per consultation.
"""

INTERPRETATIONS = {
    1: {
        "judgment": "The creative force is unobstructed. What you are becoming has no ceiling — but pure creation without ground burns itself out. The medicine here is not more force but the willingness to let the force land somewhere.",
        "image": "Six unbroken lines. Heaven over heaven. The dragon ascends through every stage — hidden, appearing, active, leaping, flying, and then the wisdom of knowing when ascent itself must rest.",
        "counsel": "Act. The energy is real and the direction is clear. But do not mistake momentum for completion. Create with full commitment, then release your grip on the outcome.",
    },
    2: {
        "judgment": "The ground is ready. Everything depends on what you allow to be planted here. Receptivity is not passivity — it is the mare who knows the territory, who follows not from weakness but from knowledge of the land.",
        "image": "Six broken lines. Earth over earth. The field in late autumn — harvested, turned, open. Nothing grows yet. That is not emptiness; it is capacity.",
        "counsel": "Do not lead. Do not force the opening. Your work now is to receive clearly — to hear what is actually being said, to feel what is actually present. The seed is coming. Your job is to be soil.",
    },
    3: {
        "judgment": "Something is trying to be born and the passage is difficult. This is not failure — it is the specific pain of emergence. The sprout does not push through frozen ground by being gentle. But it also does not push by being violent. It pushes by being persistent.",
        "image": "Water over thunder. A storm gathers but the rain hasn't broken. Everything is charged and nothing has resolved. The tension is the medicine.",
        "counsel": "Do not try to resolve things prematurely. Appoint helpers — you cannot birth this alone. Stay with the difficulty. What feels like chaos is actually the first motion of order.",
    },
    4: {
        "judgment": "You don't know what you don't know, and that ignorance is not the problem. The problem is pretending you already understand. The oracle does not seek the student — the student must seek the oracle, and must ask sincerely, not repeatedly.",
        "image": "Mountain over water. A spring at the foot of the mountain — the water is there but doesn't yet know where to flow. It needs the mountain to give it direction.",
        "counsel": "Accept instruction. Stop testing the boundaries to see what you can get away with. The structure you resist is the structure that will teach you. Ask once, clearly, and then listen.",
    },
    5: {
        "judgment": "What you need is coming but it is not here yet. The waiting is not wasted time — it is the time when nourishment gathers. Rain clouds have formed; you cannot make them release. Cross the great water when the rain comes, not before.",
        "image": "Water over heaven. Clouds in the sky that have not yet become rain. Everything is suspended in potential. The danger is real but distant.",
        "counsel": "Eat, drink, be at ease. This is not the moment for action but for strength-gathering. Anxiety will not accelerate the timing. Trust that what you've set in motion is still in motion.",
    },
    6: {
        "judgment": "You are right but being right is not enough. The conflict is real — inner truth meets outer obstruction. But pushing through will not resolve it. Only clarity brought before a mediator can untangle this.",
        "image": "Heaven over water. They move in opposite directions — one up, one down. The separation is structural, not personal. No amount of goodwill closes this gap without a third element.",
        "counsel": "Bring the dispute to someone who can see both sides. Do not carry this to the end — stop halfway. You will win the principle and lose the peace, or accept the compromise and keep both.",
    },
    7: {
        "judgment": "Discipline is the medicine. Not punishment, not rigidity — the discipline of an army that knows its purpose. The multitude requires a general, and the general requires a mandate. Without legitimate authority, force is just violence.",
        "image": "Earth over water. Water stored within the earth — an underground reservoir of power that can be called upon. The army is the people when given direction.",
        "counsel": "Organize your inner forces. Identify who leads and who follows in you, and make sure the leader is experienced, not just loud. Generosity after victory determines everything.",
    },
    8: {
        "judgment": "Union is possible but it must be around something real. Holding together requires a center — not a concept but a person, a commitment, a living bond. Those who arrive late lose nothing if their arrival is sincere.",
        "image": "Water over earth. Water on the ground flows together naturally — it finds its own level. It doesn't need to be forced into union; it only needs the ground to be open.",
        "counsel": "Seek the center and hold to it. Examine whether your alliances are genuine or merely convenient. If you are the center, ask whether you can sustain what gathers around you. Consult the oracle once more.",
    },
    9: {
        "judgment": "You have enough power to restrain but not yet enough to transform. The taming is small — the wind holds back heaven, but only for now. This is not the time for the great crossing. Dense clouds, no rain yet.",
        "image": "Wind over heaven. A gentle force above a massive one. The restraint works not because it is stronger but because it is positioned correctly — subtle influence over brute power.",
        "counsel": "Refine rather than push. Work on the small things — your tone, your presentation, your inner clarity. The great transformation will come, but the vessel must be ready first. Cultivate through gentleness.",
    },
    10: {
        "judgment": "You are walking on the tiger's tail. The situation is dangerous not because the danger is hidden but because it is fully visible and you must proceed anyway. Sincerity alone lets you pass. The tiger does not bite because your step is honest.",
        "image": "Heaven over lake. Joy below, strength above. The cheerful one approaches the powerful one — not with fear, not with arrogance, but with genuine good humor. This is conduct, not combat.",
        "counsel": "Know your place without resenting it. Move through the dangerous passage with neither bravado nor timidity. Pleasantness is not weakness — it is the exact right response to overwhelming force.",
    },
    11: {
        "judgment": "Heaven and earth are in communion. The creative descends, the receptive rises — they meet in the middle and everything flourishes. This is the great peace, but it does not last by nature. Use it fully while it is here.",
        "image": "Earth over heaven. The small goes, the great approaches. Spring — the energy exchange between above and below is open and flowing. This is how kingdoms are built: by letting high and low communicate.",
        "counsel": "This is the moment to act with full confidence. Build, connect, give generously. But know that peace is a season, not a destination. What you establish now becomes the foundation that holds when the flow reverses.",
    },
    12: {
        "judgment": "Heaven and earth are not communicating. The creative rises away, the receptive sinks — they separate and everything stagnates. Inferior elements advance while superior elements retreat. This is not failure; it is winter.",
        "image": "Heaven over earth. They move apart — heaven up, earth down. The natural order inverts. The great departs, the small arrives. Nothing you do from outside will reopen this channel.",
        "counsel": "Withdraw your energy from what is not working. Do not try to force communication with what has closed. Protect your inner values by going quiet. This is the time for conservation, not engagement. The reversal will come on its own.",
    },
    13: {
        "judgment": "Fellowship in the open, not in secret. The fire under heaven illuminates everything — nothing is hidden. True community can only form when all members are visible to each other. Cross the great water with companions.",
        "image": "Heaven over fire. Fire rises to meet heaven — clarity seeks the highest. Fellowship among people works the same way: when the shared aim is noble, the bonds hold.",
        "counsel": "Bring your alliances into the light. Whatever you're building with others, make sure the terms are clear and the purposes are shared. Hidden agendas will burn through every bond. Openness is the only foundation that holds.",
    },
    14: {
        "judgment": "Supreme success. The fire above heaven — illumination in the highest place. You possess something great. The danger is not losing it but holding it without grace. Wealth obligates.",
        "image": "Fire over heaven. The sun at zenith. Everything is visible, everything is abundant. This is the hexagram of the harvest that comes after right action — not luck, but consequence.",
        "counsel": "Suppress evil, promote good, and do so from a position of genuine abundance. Do not hoard. What you have was given through alignment, and it remains through generosity. Use your power to clarify, not to dominate.",
    },
    15: {
        "judgment": "The mountain is hidden within the earth. What is full empties itself; what is empty fills. Modesty is not self-deprecation — it is the mountain that does not need to be seen to be real. The cosmos itself favors this balance.",
        "image": "Earth over mountain. The great lies beneath the humble. This is not concealment but proper proportion — the mountain doesn't shrink, the earth simply covers it naturally.",
        "counsel": "Reduce what is excessive, fill what is lacking. Do not announce your strengths. The work will speak, and those who need to see will see. Modesty is a force, not an absence.",
    },
    16: {
        "judgment": "Enthusiasm moves the world. Thunder over earth — the energy that breaks through the surface and moves everything into action. But enthusiasm without devotion is just noise. The ancient kings made music to honor the divine; the movement was an offering, not a performance.",
        "image": "Thunder over earth. The first rumble of spring thunder that wakes the seeds. Everything responds — not because it is commanded but because the signal resonates with what was already waiting.",
        "counsel": "Move now, while the energy is alive. Set things in motion — projects, helpers, commitments. But ground the enthusiasm in genuine purpose. If the energy isn't rooted in something you'd die for, it will scatter.",
    },
    17: {
        "judgment": "Thunder within the lake rests. Following is not submission — it is knowing when to adapt without losing yourself. What you follow must be worth following. The leader who would lead must first learn to be led.",
        "image": "Lake over thunder. The active energy goes to rest inside the still. This is evening — the time when the day's initiative yields to the night's reception. Even thunder sleeps.",
        "counsel": "Adapt to what the moment requires, not what your ego prefers. If you are leading, rest. If you are following, commit fully. The worst position is half-following — one foot in, one foot out. Choose, then move.",
    },
    18: {
        "judgment": "What has been corrupted must be repaired, and the repair requires understanding the corruption. This is not about blame but about clarity — three days before the turning point, three days after. The decay didn't begin with you, but the repair does.",
        "image": "Mountain over wind. Wind trapped beneath the mountain stagnates — what was once fresh becomes stale. The bowl has been left too long. The insects have entered.",
        "counsel": "Look at what went wrong — specifically, clearly, without flinching. The corruption has a source and a timeline. Trace it back three steps, then work forward three steps. Repair is possible but it requires confronting the full picture.",
    },
    19: {
        "judgment": "The great approaches. Earth above the lake — what was deep rises to the surface. This is the time of increasing influence, but the approach has a natural limit. By the eighth month, the energy will reverse. Use this window.",
        "image": "Earth over lake. The bank of the lake — where the solid ground meets the depth. Approach is mutual: you come to it and it comes to you. The boundary between them is where transformation happens.",
        "counsel": "Teach and nurture now, while the energy is approaching. Your influence is expanding — use it to build something that will outlast the season. What you establish in approach must be strong enough to endure retreat.",
    },
    20: {
        "judgment": "Wind over earth — seeing everything from above. This is not surveillance but contemplation. The tower view changes what you thought you knew. The ancient kings surveyed their lands and examined their people — not to judge but to understand what was actually there.",
        "image": "Wind over earth. The wind touches everything but disturbs nothing. From the tower, the pattern is visible that was invisible at ground level. The ablution has been made but not the offering — you have prepared but not yet acted.",
        "counsel": "Step back and observe before you intervene. What you think is the problem may be a symptom of a pattern you can only see from a distance. Contemplate deeply, then act from understanding rather than reaction.",
    },
    21: {
        "judgment": "Something stands between the jaws that must be bitten through. The obstruction is real and it will not dissolve on its own. Fire and thunder together — decisive, illuminated action. This is the hexagram of justice applied with clarity.",
        "image": "Fire over thunder. Lightning and thunder together — the moment of illuminated force. The mouth with something between the teeth. It must be bitten through for the mouth to close properly.",
        "counsel": "Act decisively. Whatever is blocking union — a lie, an obstruction, an unresolved conflict — bite through it now. This is not aggression; it is the minimum force required to clear the channel. Do not be excessive, but do not hesitate.",
    },
    22: {
        "judgment": "Form matters but it is not the substance. Fire at the foot of the mountain illuminates the surface — grace, beauty, adornment. But grace applied to the wrong substance is deception. Small matters can be decided here; great matters cannot.",
        "image": "Mountain over fire. The fire illuminates the mountain from below — surface details become visible, contours are highlighted. But the mountain's mass is unchanged by the light.",
        "counsel": "Attend to form and presentation, but do not mistake them for content. This is the time to polish, not to build. If the substance is sound, grace completes it. If the substance is hollow, grace only makes the hollowness visible to those who look closely.",
    },
    23: {
        "judgment": "The bed is being stripped from the legs up. What once supported you is being removed — not by malice but by the natural action of decay that follows excess. The inferior peels away the superior, layer by layer. You cannot stop this process.",
        "image": "Mountain over earth. The mountain rests on an eroding base. Line by line, from the bottom up, the structure is dismantled. Only the top line — the fruit — remains. The seed inside the fruit survives.",
        "counsel": "Do not act. Do not resist the stripping. What is being removed was already dead — you were resting on a structure that could no longer hold. The one thing that survives is the seed: your core commitment. Let everything else go.",
    },
    24: {
        "judgment": "The turning point. After the darkest moment, the first light returns — not as a flood but as a single unbroken line beneath five broken ones. Thunder within the earth; the life force stirs underground. Seven days — the cycle completes itself.",
        "image": "Earth over thunder. The thunder is below, silent, gathering. Above, the earth rests, patient. The solstice — the shortest day, after which each day grows. You cannot see the return yet, but it has already begun.",
        "counsel": "Rest. Let the return happen naturally — do not force the recovery. The energy that is coming back is still fragile. Protect it. No great undertakings. Friends will come without blame. What was lost is returning on its own schedule.",
    },
    25: {
        "judgment": "Innocence is not naivety — it is acting without ulterior motive. Thunder under heaven moves everything, and what moves without calculation moves correctly. The unexpected gift comes to those who were not seeking it. But if the action is wrong, there is only misfortune.",
        "image": "Heaven over thunder. The natural order expresses itself without scheming. Spring thunder — the world stirs not because someone planned it but because the time is right.",
        "counsel": "Act from instinct, not strategy. If the motion feels pure — if you'd do it whether anyone was watching or not — proceed. If you catch yourself calculating the outcome, stop. Innocence cannot be manufactured. Either it is present or it is not.",
    },
    26: {
        "judgment": "Great accumulation. Heaven within the mountain — immense power held in stillness. Ancient wisdom stored, not displayed. The power here is not in action but in the depth of what has been gathered. It is favorable to cross the great water — the reserves are deep enough.",
        "image": "Mountain over heaven. The mountain contains heaven — the small holds the vast. This is the sage who carries libraries in silence, the well that reaches the aquifer.",
        "counsel": "Study what the ancients did. Not for scholarship but for fuel. You have accumulated more than you realize — knowledge, experience, endurance. Now it can be used. Eat away from home. Cross the great water. The stored power is ready.",
    },
    27: {
        "judgment": "Watch what goes into the mouth. Watch what comes out of it. Nourishment is not only food — it is every form of intake and expression. Thunder at the foot of the mountain: the jaw. What you consume becomes what you are.",
        "image": "Mountain over thunder. The open mouth — the lower jaw (thunder) moves, the upper jaw (mountain) is still. Everything depends on the quality of what enters this opening.",
        "counsel": "Audit your inputs. What are you reading, eating, watching, absorbing? And what are you saying, producing, emitting? The imbalance you feel may be a nourishment problem, not a will problem. Change what you feed on.",
    },
    28: {
        "judgment": "The ridgepole sags. The structure is under extraordinary pressure — the weight at the center exceeds what the frame can bear. Extraordinary measures are required. This is not a time for ordinary responses. The lake rises over the trees.",
        "image": "Lake over wind. Water above wood — the trees are submerged. The flood has exceeded the banks. What normally supports life is now overwhelmed.",
        "counsel": "Do something you would not normally do. The ordinary toolkit cannot handle this. Bend but do not break — find the yielding strength that lets you flex under pressure that would snap rigidity. Stand alone if necessary. Act now.",
    },
    29: {
        "judgment": "Danger upon danger. Water doubled — the abyss repeated. You cannot go around this; you must flow through. The water does not pile up in the pit; it flows through and continues. This is how to move through danger: consistently, without stopping, trusting the flow.",
        "image": "Water over water. The ravine doubled. There is no safe detour — the danger is the path. But water always finds its way through, and it does so by being true to its own nature: it flows downward and fills every space before moving on.",
        "counsel": "Do not try to escape the danger. Move through it with consistency. Maintain your sincerity — it is the only thing that works here. Teach what you know, guard your heart with practice. The danger is real but the passage is real too.",
    },
    30: {
        "judgment": "Clarity depends on what it clings to. Fire doubled — brightness upon brightness. But fire has no independent form; it needs fuel. The question is not whether you can see, but whether what you're clinging to is worth the light.",
        "image": "Fire over fire. The sun, the moon, the stars — they cling to heaven. The grasses and trees cling to the earth. Doubled clarity illuminates the four directions, but only when it has a worthy source.",
        "counsel": "Accept dependence on something greater than yourself. The fire that tries to burn without fuel destroys itself. Find what deserves your brightness, cling to it, and the illumination will be natural and sustained. Care for what sustains you.",
    },
    31: {
        "judgment": "Influence through emptiness, not force. The lake on the mountain — the mountain holds still and receives what the lake gives. Mutual attraction happens when both parties are empty enough to be affected. This is courtship: the willingness to be changed.",
        "image": "Lake over mountain. The lake above, the mountain below — the yielding one is above the still one. This is influence through receptivity. The mountain does not reach for the lake; it simply makes space.",
        "counsel": "Empty yourself enough to receive influence. If you approach with your cup full, nothing can enter. The attraction you feel is real — respond to it not with grasping but with openness. Keep still, keep receptive, keep honest.",
    },
    32: {
        "judgment": "Duration is not rigidity — it is movement that renews itself endlessly. Thunder over wind; the marriage of initiative and penetration. What endures does so not by staying the same but by remaining consistent while everything around it changes.",
        "image": "Thunder over wind. The thunder moves, the wind follows. Direction married to persistence. The seasons endure not by freezing but by cycling. The sage endures by standing firm in the direction while adapting the method.",
        "counsel": "Commit to the direction, not the method. What needs to last must be allowed to move. Relationships, practices, purposes — they endure through renewal, not through rigidity. Stay with it, but let the form evolve.",
    },
    33: {
        "judgment": "Strategic withdrawal — not defeat but the wisdom of knowing when the battlefield has shifted beneath you. The mountain under heaven; the boundary between engagement and retreat. The great recognizes the moment and pulls back before they are forced.",
        "image": "Heaven over mountain. Heaven recedes from the mountain — the celestial withdraws from the earthly. This is not collapse but repositioning. The small advances, so the great must retreat — but the retreat itself is a form of power.",
        "counsel": "Pull back now, while the retreat is still voluntary. What you withdraw from will not be lost — but if you stay past the turning point, what you're protecting will be. Hold inferior elements at arm's length. Retreat with dignity, not bitterness.",
    },
    34: {
        "judgment": "Great power. Thunder above heaven — enormous force fully expressed. But power without rightness is merely dangerous. The ram who charges the fence gets his horns tangled. Power is only great when governed by what is right.",
        "image": "Thunder over heaven. The crash that shakes the sky. Everything trembles. This is pure force — not subtle, not strategic, just overwhelming energy in motion.",
        "counsel": "You have the power. The question is whether you have the rightness. Before you move, check: is the direction correct? Power applied in the right direction transforms. Power applied in the wrong direction destroys — and the thing it destroys first is you.",
    },
    35: {
        "judgment": "The sun rising above the earth — progress that is natural, not forced. The prince receives horses in great number. In a single day, three audiences with the king. This is advancement through clarity, not through ambition.",
        "image": "Fire over earth. Dawn. The light simply rises — it does not compete with the darkness, it replaces it. The advance is so natural that it looks effortless, but it rests on everything the night prepared.",
        "counsel": "Advance openly. Your progress now is earned and visible — do not hide it, do not apologize for it. Share the light you carry. Those above you will recognize what you bring. Move forward with confidence grounded in genuine capacity.",
    },
    36: {
        "judgment": "The light is driven underground. Earth over fire — brilliance hidden, not destroyed. This is the wounding of the bright. The inner flame must be protected by dimming the outer expression. Do not let them see what you carry.",
        "image": "Earth over fire. The sun has set — or been forced below the horizon. King Wen in prison, who kept his light alive by concealing it from the tyrant. The fire burns inward.",
        "counsel": "Protect your core by hiding your brilliance. This is not dishonesty — it is survival. The environment cannot receive what you carry right now. Stay dark on the outside, keep the flame alive inside. The night will end, but not yet.",
    },
    37: {
        "judgment": "The inner order that radiates outward. Wind from fire — the warmth of the hearth creates the circulation of the home. Every relationship in the family has its correct position, and when each fulfills their role, the whole structure breathes.",
        "image": "Wind over fire. The fire warms, the wind carries the warmth outward. Words should have substance and conduct should have consistency. The family is the prototype of all social order.",
        "counsel": "Get your inner house in order. Whatever you're trying to build in the world, check the foundation at home first. Relationships that are true in private radiate outward naturally. Say what you mean, then live it consistently.",
    },
    38: {
        "judgment": "Opposition. Fire and lake — they move in opposite directions. But opposition is not destruction. When two things oppose, the tension between them creates sight. In small matters, there is still good fortune — the opposition illuminates.",
        "image": "Fire over lake. Fire rises, water sinks. They cannot merge. But at the boundary where they meet, steam rises — a third thing that neither could produce alone.",
        "counsel": "Do not try to force unity where opposition is the reality. Instead, look at what the opposition reveals. The person or situation you conflict with is showing you something about yourself. In small things, engage. In great things, maintain the separation until the third element appears.",
    },
    39: {
        "judgment": "The path is blocked and the obstruction is real — not imagined, not a test of will, genuinely impassable in this direction. Water on the mountain: the danger is ahead. The only productive response is to turn inward. See the great person, which may be yourself.",
        "image": "Water over mountain. The water collects on the mountain — it cannot flow forward. The obstruction is geographical, not personal. Some paths are simply blocked at certain times.",
        "counsel": "Stop trying to push through. Turn around — not in defeat but in reorientation. The obstruction is telling you something about direction, not about your worth. Go inward, gather your resources, find the person who sees what you cannot. Try a different path.",
    },
    40: {
        "judgment": "The storm breaks and the tension dissolves. Thunder over water — deliverance comes suddenly after long pressure. Act swiftly in the clearing. Return to the normal path as quickly as possible. If there are things to be resolved, resolve them at dawn.",
        "image": "Thunder over water. The thunderstorm that clears the air — everything was heavy, compressed, unbearable, and then the crack and the rain and the relief. The air is clean now.",
        "counsel": "Move fast. The window of deliverance is open but it won't stay open indefinitely. Forgive what needs forgiving, release what needs releasing, and return to normal life. Do not linger in the drama of the crisis. The clearing is for moving through, not for camping in.",
    },
    41: {
        "judgment": "Decrease in the lower to increase the upper. Genuine loss that serves a greater gain — but the offering must be sincere. Two small bowls may be used for the sacrifice. It's not the size of the offering but the truth of the giver.",
        "image": "Mountain over lake. The mountain rises by what the lake gives up. The lake diminishes itself and the mountain grows. This is sacrifice that produces elevation — but only when the sacrifice is real.",
        "counsel": "Reduce something. Give something up — not symbolically but actually. The decrease you fear is the medicine. Restrain your anger, diminish your desires, simplify your approach. What you release rises. Even a small offering, given sincerely, changes the geometry.",
    },
    42: {
        "judgment": "What benefits others benefits you — this is the increase that comes from above. Wind over thunder: the energy descends and multiplies. When you see good, move toward it. When you see fault in yourself, correct it. This is the time to cross the great water.",
        "image": "Wind over thunder. The wind above strengthens the thunder below. This is the ruler who decreases themselves to increase the people — and the increase comes back doubled.",
        "counsel": "Give. Now is the time for generosity — of resources, of attention, of self-correction. What you give will return amplified. But the giving must be genuine, not strategic. Correct what needs correcting in yourself. Help what needs helping around you. The increase follows.",
    },
    43: {
        "judgment": "The decisive moment. Lake over heaven — the accumulated pressure breaks through. One yin line at the top is about to be displaced by the rising yang. This is breakthrough, but it requires announcing the truth in the court — not with weapons but with clarity.",
        "image": "Lake over heaven. The water has risen to the sky — what was below is now above. The breakthrough is so close that only one obstacle remains. It must be resolved through declaration, not through force.",
        "counsel": "Speak the truth you've been holding. The breakthrough cannot happen silently — it requires public declaration. But arm yourself with truth, not with force. If you use the methods of what you're replacing, the replacement means nothing. Be resolute, not violent.",
    },
    44: {
        "judgment": "The unexpected encounter. Wind under heaven — something enters from below that was not anticipated. The maiden is powerful; do not dismiss what comes to meet you. But do not let it take over either. The meeting is significant precisely because it was not planned.",
        "image": "Heaven over wind. The wind blows under heaven, reaching everywhere. This is the encounter that comes to you — the message, the person, the impulse that arrives without invitation. It carries information you need.",
        "counsel": "Pay attention to what approaches uninvited. It is showing you something you weren't looking for. Do not suppress it, but do not surrender to it either. Examine it clearly. The encounter has a purpose, but the purpose may not be what it first appears.",
    },
    45: {
        "judgment": "The gathering. Lake over earth — people drawn together around a central purpose. But gathering requires a worthy center and a genuine offering. Without sacrifice, the gathering is just a crowd. With it, the gathering becomes a community.",
        "image": "Lake over earth. Water collects on the surface of the earth — it flows to the low places and gathers there naturally. People do the same when a genuine center exists.",
        "counsel": "Gather — but around what? Check that the center is real, not manufactured. If you are organizing, make the offering first. If you are joining, verify the purpose. Great undertakings are possible when the gathering is genuine. Be prepared for the unexpected — concentrated energy attracts disruption.",
    },
    46: {
        "judgment": "Pushing upward. Earth over wind — the seed that rises through the soil, meeting no resistance. Growth so natural it is invisible until it has already occurred. The south is the direction of influence and recognition.",
        "image": "Earth over wind. The wood grows upward through the earth — unseen, persistent, irresistible. Not the eruption of the volcano but the steady rise of the oak. By the time it breaks the surface, the roots are already deep.",
        "counsel": "Advance without hesitation. The resistance you expected is not there. But stay humble in the ascent — you are rising because the conditions are right, not because you conquered anything. See the great person. Do not worry.",
    },
    47: {
        "judgment": "Exhaustion. The lake with no water — the life-sustaining element is gone. Words cannot reach anyone from this position. But the superior person stakes their life on following their will. In the deepest pit, cheerfulness is the only medicine that works.",
        "image": "Lake over water, but the water has drained below the lake — the container is present but the content is gone. This is oppression by circumstance, not by enemies. The well is dry.",
        "counsel": "Do not explain yourself. Words cannot save you here — they will be disbelieved or ignored. Act from your core even when no one sees or responds. The exhaustion is real but it is not permanent. What remains when everything is stripped away is what you actually are.",
    },
    48: {
        "judgment": "The well does not change. The town may be moved but the well cannot be moved. People come and go; the water remains. But if the rope is too short, or the jug breaks — so near to the water and yet no water. This is the tragedy of almost.",
        "image": "Water over wind. The wood draws water upward — the well. The structure that reaches into the depths and brings what is below to where it can be used. The well serves everyone equally.",
        "counsel": "Go deeper. What you need is there but you haven't reached it yet. Check your equipment — is the rope long enough? Is the vessel intact? The water is not the problem; the mechanism of access is the problem. Repair the means of reaching what sustains you.",
    },
    49: {
        "judgment": "Revolution. Fire in the lake — a fundamental change of form. The old skin is shed. But revolution that comes too early is rejected; revolution that comes at the right time is believed. First certainty, then action. After the day of revolution, remorse disappears.",
        "image": "Lake over fire. Fire and water meet — transformation through the clash of opposites. The animal sheds its fur with the seasons. This is not destruction but metamorphosis: what was will become what must be.",
        "counsel": "If the change is genuinely necessary, make it completely. No half-revolutions — they breed the worst of both states. But verify the necessity first. Has the time truly come? Can you articulate why the old form must die? If yes, proceed without regret.",
    },
    50: {
        "judgment": "The cauldron. Fire over wind — the sacred vessel of transformation. What you put into the cauldron determines what comes out. The legs, the belly, the handles, the carrying rings — every part of the vessel matters. Supreme success for those who cook the right ingredients.",
        "image": "Fire over wind. Wind feeds fire, fire transforms what is placed above it. This is the civilizing instrument — the tool that turns raw material into nourishment, ore into metal, suffering into wisdom.",
        "counsel": "You are the vessel. What are you cooking? Check the ingredients — are they worthy of the fire? Check the structure — can the cauldron hold what's being transformed? The process works, but only if the vessel is sound and the contents are chosen deliberately.",
    },
    51: {
        "judgment": "Thunder doubled — shock. The first shock terrifies. The second shock reveals that the first was survivable. After the shock, laughter. The sacrificial wine is not spilled. Terror followed by composure is the complete medicine.",
        "image": "Thunder over thunder. The crash comes and comes again. Everything shakes — a hundred miles of trembling. But the one who holds the sacred vessel does not drop it. The shock is the test of what you actually hold firm.",
        "counsel": "Let the shock pass through you. Do not brace against it — bracing is what breaks. After the fear subsides, look around: what did you drop and what did you keep? What survived the shaking is real. Everything else was always going to fall.",
    },
    52: {
        "judgment": "Stillness. The mountain doubled — keeping still until the moment for movement arrives. Still the back until you no longer feel the body. Still the mind until it no longer generates its own noise. Walk in the courtyard and do not see the people. No blame.",
        "image": "Mountain over mountain. Perfect stillness. Not paralysis — the stillness of the mountain that knows it does not need to move. The back is still because the back has no eyes — it does not reach forward for what it cannot see.",
        "counsel": "Stop. Not later — now. The movement you want to make is premature. The thoughts you're generating are noise. Be the mountain. When the time for action comes, you will know because the stillness will break itself. Until then, keep still.",
    },
    53: {
        "judgment": "Gradual progress. The tree on the mountain grows slowly but it grows in the right place. Each stage is complete before the next begins. The wild goose approaches the shore, the cliff, the plateau, the tree, the summit, the clouds. Each approach is correct for its stage.",
        "image": "Wind over mountain. The tree on the mountain is visible from far away — its development is public and gradual. No shortcuts are possible. Each ring of growth is earned.",
        "counsel": "Accept the pace. What you're building requires stages, and each stage must be complete before the next. Impatience here is not energy — it is sabotage. The progress is real even when it feels invisible. Trust the sequence.",
    },
    54: {
        "judgment": "The marrying maiden — entering a situation where you are not the primary. Thunder over lake: initiative in a subordinate position. The undertaking brings misfortune only if you pretend the position is something it isn't. Accept the role clearly.",
        "image": "Thunder over lake. The younger sister follows the elder — not from weakness but from structural reality. The position is secondary, and there is no shame in that unless shame is manufactured.",
        "counsel": "Enter with clear eyes. You are not the center of this arrangement, and pretending otherwise creates the only real danger. Find the power within the position you actually hold. Influence from the secondary position requires subtlety, patience, and the abandonment of ego.",
    },
    55: {
        "judgment": "Abundance — fullness at the zenith. Thunder and fire together create a blazing moment of total illumination. But the sun at midday begins to set. The moon at fullness begins to wane. Do not grieve this — abundance is meant to be lived, not preserved.",
        "image": "Thunder over fire. The storm at noon — maximum illumination, maximum energy. The king reaches this fullness. But the hexagram warns: be like the sun at midday. Illuminate everything. Do not try to stop the passing.",
        "counsel": "Be fully present for this. The abundance will not last because abundance never lasts — that is its nature, not its failure. Use every drop of this moment. Illuminate what needs seeing. Give what needs giving. The grieving for its loss is premature and wasteful.",
    },
    56: {
        "judgment": "The wanderer. Fire on the mountain — a small, exposed flame in an exposed place. The traveler must be careful, modest, and quick. There is no home here. Small things succeed; great ambitions in transit are dangerous.",
        "image": "Fire over mountain. The brushfire that moves across the peak — it cannot stay. It is visible from everywhere but it rests nowhere. This is the condition of passage: everything is temporary.",
        "counsel": "Travel light. Do not try to build a fortress where you're passing through. Be intelligent with your resources, respectful to the places you move through, and clear about the fact that you are in transit. Small, correct actions. No grand plans.",
    },
    57: {
        "judgment": "Penetration through persistence, not through force. Wind doubled — the gentle that enters every crack. A single gust does nothing; constant wind reshapes the landscape. It is favorable to see the great person and to cross the great water.",
        "image": "Wind over wind. The wind that follows the wind. Its power is cumulative — no single moment is dramatic but the aggregate is irresistible. This is influence by repetition, by quiet consistency.",
        "counsel": "Keep going. The penetration is working even when you see no result. Your consistency is the force — not cleverness, not intensity, just presence maintained over time. Have a clear destination. The wind without direction is just weather; the wind with direction is erosion.",
    },
    58: {
        "judgment": "Joy. Lake doubled — true gladness shared. But joy that depends on external conditions is not true joy. The joyous lake is dangerous only when its banks are breached by excess. Inner truth is the source; outer expression is the result.",
        "image": "Lake over lake. Two lakes connected — what fills one fills both. This is joy that multiplies through sharing, discussion, practice together. Friends discuss and practice together — this is the real source.",
        "counsel": "Share what gives you joy. But check: is the joy coming from your center or from your surface? The distinction matters. Surface joy dissipates; centered joy sustains. Find someone to practice with. The doubled lake — two people doing the work together — is the strongest configuration.",
    },
    59: {
        "judgment": "Dispersion — dissolving what has frozen. Wind over water scatters the ice. Rigidity is the disease; dissolution is the medicine. The king approaches the temple — the sacred center dissolves separateness. It is favorable to cross the great water.",
        "image": "Wind over water. The wind breaks up the ice on the surface of the water — what was solid and impassable becomes flowing again. The king goes to the ancestral temple: the connection to origin dissolves ego-barriers.",
        "counsel": "Something in you has frozen — a resentment, a fixed position, a separation you've maintained past its usefulness. Let the wind work on it. Return to what connects you to your origin, your purpose, the people you've separated from. The rigidity is not protecting you; it is imprisoning you.",
    },
    60: {
        "judgment": "Limitation. Water over lake — boundaries that enable life, not boundaries that prevent it. The seasons work because they have limits. The joint works because the bone stops where the socket begins. But bitter limitation must not be persevered in.",
        "image": "Water over lake. The lake holds water because it has banks. Without banks, it would be a flood — formless, destructive, unable to sustain anything. Limitation gives form to what would otherwise dissipate.",
        "counsel": "Set a boundary. The lack of limits is not freedom — it is formlessness. But choose your limits wisely: limits that serve life are sweet; limits that serve ego are bitter. If the limitation feels like dying, it may be the wrong limit. If it feels like structure, it is the right one.",
    },
    61: {
        "judgment": "Inner truth — sincerity that reaches even pigs and fishes. Wind over the lake; the effect of truth on the surface of the deep. When the inside matches the outside, even the most resistant beings are reached. Favorable to cross the great water.",
        "image": "Wind over lake. The wind ripples the surface of the lake — the inner movement becomes outwardly visible. The crane calls from the shade and its young answer. What is truly felt resonates through sympathetic vibration.",
        "counsel": "Mean what you say. The situation requires not more effort but more sincerity. What you are broadcasting inwardly is being received by everything around you — including the things you think can't hear you. Align inner and outer. The resonance will do the work.",
    },
    62: {
        "judgment": "The small exceeds. Thunder on the mountain — the sound is too big for the place. Small things succeed; great things do not. The flying bird brings the message: do not go up, go down. Going down is the right direction in this particular moment.",
        "image": "Thunder over mountain. The thunder is disproportionate — too loud for the terrain. The bird that flies too high meets the wind. This is the time for excessive care in small things, not for grand gestures.",
        "counsel": "Scale down. Whatever you're attempting, make it smaller. More reverence than usual. More frugality than usual. More grief than usual — meaning: attend to what is below you, not what is above. The small is where the power is right now. Honor it.",
    },
    63: {
        "judgment": "After completion. Everything is in place — the small has found its position, the great has found its position. Water over fire: the pot is on the flame and the water boils. Initial success. But at the moment of completion, disorder begins its return.",
        "image": "Water over fire. Perfect equilibrium — each element in its correct place. The hexagram where every line is correct. But this is also the most unstable configuration: there is nowhere to go but toward disorder.",
        "counsel": "Celebrate briefly, then prepare. The completion is real but it is also the beginning of the next cycle's decline. Take stock of what you've achieved and immediately tend to the small things that are already starting to slip. Vigilance at the peak prevents collapse at the base.",
    },
    64: {
        "judgment": "Before completion. Fire over water — the elements are in the wrong positions but they are moving toward the right positions. The fox has almost crossed the ice. Almost. If the tail gets wet at the last moment, everything is lost. Nothing yet serves.",
        "image": "Fire over water. The reverse of completion — everything is nearly in place but nothing is finalized. The fox on the ice: cautious, almost across, but the tail is the last thing to cross and the ice is thinnest at the far bank.",
        "counsel": "You are not done. The feeling that you're almost there is accurate, but 'almost' is the most dangerous position. Do not relax. Do not celebrate prematurely. The last step requires the most care. Distinguish carefully between what has actually been completed and what merely feels complete.",
    },
}
