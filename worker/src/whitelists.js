// ─── WHITELISTS — diagnosis JSON validation ────────────────────────────
// Enumerable allowlists for fields the Oracle's web_api emits with a
// finite set of valid values. Used by validateDiagnosis() in worker.js
// to drop crafted strings before they reach the Reflective Principle's
// system prompt context.
//
// Source mapping (regenerate if the Python source ever changes):
//   TRIGRAM_ELEMENTS, TRIGRAM_MEANINGS  ← core/octonion.py Trigram enum
//   TRIGRAM_QUALITY_STRINGS,            ← oracle/interpreter.py
//     TRIGRAM_DESCRIPTIONS                  TRIGRAM_QUALITIES dict
//   KING_WEN_NAMES, KING_WEN_TITLES     ← oracle/hexagrams.py
//                                           HEXAGRAM_NAMES dict
//   TOLTEC_NAMES, TOLTEC_TRAD_NAMES     ← oracle/toltec.py
//                                           TOLTEC_HEXAGRAMS dict
//   TAROT_NAMES, TAROT_NUMERALS         ← oracle/tarot.py MAJOR_ARCANA
//
// Prose fields (hexagram.judgment/image/counsel, toltec.image/interp/
// action/intent/summary, tarot.*.image/upright/counsel) are NOT
// enumerable and remain length-clamped in worker.js — bundling
// interpretations.py prose into the worker is out of scope for now.

export const TRIGRAM_ELEMENTS = new Set([
  'Heaven', 'Earth', 'Thunder', 'Water', 'Mountain', 'Wind', 'Fire', 'Lake',
]);

export const TRIGRAM_MEANINGS = new Set([
  'Creative', 'Receptive', 'Arousing', 'Abysmal',
  'Keeping Still', 'Gentle', 'Clinging', 'Joyous',
]);

export const TRIGRAM_QUALITY_STRINGS = new Set([
  'creative force', 'receptive ground', 'arousing thunder', 'abysmal water',
  'still mountain', 'gentle wind', 'clinging fire', 'joyous lake',
]);

export const TRIGRAM_DESCRIPTIONS = new Set([
  'the drive to act and create',
  'the capacity to hold and sustain',
  'the shock that sets things in motion',
  'the descent into depth',
  'the weight that does not need to move',
  'the persistence that enters every crack',
  'the clarity that depends on what feeds it',
  'the openness that connects through speech',
]);

// 64 King Wen romanizations — 57 unique (some names repeat across
// hexagrams: Qian #1+#15, Bi #8+#22, Yi #27+#42, Lu #10+#56, Jian #39+#53,
// Kun #2+#47, Jie #40+#60).
export const KING_WEN_NAMES = new Set([
  'Qian', 'Kun', 'Zhun', 'Meng', 'Xu', 'Song', 'Shi', 'Bi',
  'Xiao Xu', 'Lu', 'Tai', 'Pi', 'Tong Ren', 'Da You', 'Yu', 'Sui',
  'Gu', 'Lin', 'Guan', 'Shi He', 'Bo', 'Fu', 'Wu Wang', 'Da Xu',
  'Yi', 'Da Guo', 'Kan', 'Li', 'Xian', 'Heng', 'Dun', 'Da Zhuang',
  'Jin', 'Ming Yi', 'Jia Ren', 'Kui', 'Jian', 'Jie', 'Sun', 'Guai',
  'Gou', 'Cui', 'Sheng', 'Jing', 'Ge', 'Ding', 'Zhen', 'Gen',
  'Gui Mei', 'Feng', 'Xun', 'Dui', 'Huan', 'Zhong Fu', 'Xiao Guo',
  'Ji Ji', 'Wei Ji',
]);

export const KING_WEN_TITLES = new Set([
  'The Creative', 'The Receptive', 'Difficulty at the Beginning',
  'Youthful Folly', 'Waiting', 'Conflict', 'The Army', 'Holding Together',
  'Small Taming', 'Treading', 'Peace', 'Standstill', 'Fellowship',
  'Great Possession', 'Modesty', 'Enthusiasm', 'Following',
  'Work on the Decayed', 'Approach', 'Contemplation', 'Biting Through',
  'Grace', 'Splitting Apart', 'Return', 'Innocence', 'Great Taming',
  'Nourishment', 'Great Exceeding', 'The Abysmal', 'The Clinging',
  'Influence', 'Duration', 'Retreat', 'Great Power', 'Progress',
  'Darkening of the Light', 'The Family', 'Opposition', 'Obstruction',
  'Deliverance', 'Decrease', 'Increase', 'Breakthrough', 'Coming to Meet',
  'Gathering', 'Pushing Upward', 'Oppression', 'The Well', 'Revolution',
  'The Cauldron', 'The Arousing', 'Keeping Still', 'Development',
  'The Marrying Maiden', 'Abundance', 'The Wanderer', 'The Gentle',
  'The Joyous', 'Dispersion', 'Limitation', 'Inner Truth',
  'Small Exceeding', 'After Completion', 'Before Completion',
]);

export const TOLTEC_NAMES = new Set([
  'Provoking Change', 'Sensing Creation', 'Consulting the Oracle',
  'Learning from Nature', 'Healing Old Wounds',
  'Waiting for Turbulence to Calm', 'Giving Birth to New Creation',
  'Balancing Dual Forces', 'Indomitable Intent', "Painting the Moon's Song",
  'Sustained Effort Rewarded', 'Window of Opportunity',
  'Mountain Concentration', 'Butterfly Metamorphosis', 'Family Rituals',
  'Dawn Ceremony', 'Shadow Journey', 'Creative Union',
  'Ritual of Rejoicing', 'Wisdom Journey', 'Undefeatable Will',
  'Ancestral Council', 'Liberating Passions', 'Hidden Treasures',
  'Community Purpose', 'Purposeful Action', 'Intuitive Guidance',
  'Dance of Renewal', 'Wind Flexibility', 'Death and Rebirth',
  'River of Life', 'Mastering Destructive Passions', 'Walking Alone',
  'Spiritual Protection', 'Confusion and Misdirection', 'Spirit Communion',
  'Disentanglement', 'Return to Wilderness', 'Ancestral Fire',
  "Spider's Web", 'Invisible Defiance', 'Artistic Vision',
  'Metamorphosis Journey', 'Spiritual Transformation', 'Freedom from Past',
  'Source of Contentment', 'True Self Emergence', 'Living Renewal',
  'Open Exploration', 'Focused Craft', 'Seed Spirit',
  'Blossoming Convictions', 'Reason Humbled', 'Healing the Past',
  'Living Purity', 'Reawakened Idealism', 'Unshakable Faith',
  'Spirit of Flower', 'Hidden Seeds', 'Graceful Transition',
  'Allied Warriors', 'Spirit Vision', 'Universal Metamorphosis',
  'Safeguarding Life',
]);

export const TOLTEC_TRAD_NAMES = new Set([
  'The Arousing (Thunder)', 'The Joyous (Lake)',
  'The Taming Power of the Small', 'The Cauldron', 'Increase', 'Duration',
  'Preponderance of the Great', 'Inner Truth', 'Difficulty at the Beginning',
  'Enthusiasm', 'Following', 'The Marrying Maiden',
  'Preponderance of the Small', 'Nourishment', 'Revolution', 'Treading',
  'Peace', 'Before Completion', 'Gathering Together', 'Limitation',
  'The Taming Power of the Great', 'The Wanderer', 'Opposition',
  'Break-through', 'The Power of the Great', 'Biting Through',
  'Possession in Great Measure', 'The Army', 'The Gentle', 'Keeping Still',
  'Darkening of the Light', 'Conflict', 'Abundance', 'Innocence', 'Decrease',
  'Influence', 'Obstruction', 'Splitting Apart', 'The Family',
  'Coming to Meet', 'Progress', 'Waiting', 'Return', 'Deliverance',
  'Approach', 'Oppression', 'Pushing Upward', 'Dispersion', 'Modesty',
  'Youthful Folly', 'The Creative', 'The Clinging', 'After Completion',
  'Standstill', 'Retreat', 'Grace', 'The Abysmal', 'The Receptive',
  'The Well', 'Contemplation', 'Holding Together', 'Fellowship',
  'Gradual Progress', 'Work on What Has Been Spoiled',
]);

export const TAROT_NAMES = new Set([
  'The Fool', 'The Magician', 'The High Priestess', 'The Empress',
  'The Emperor', 'The Hierophant', 'The Lovers', 'The Chariot',
  'Strength', 'The Hermit', 'Wheel of Fortune', 'Justice',
  'The Hanged Man', 'Death', 'Temperance', 'The Devil', 'The Tower',
  'The Star', 'The Moon', 'The Sun', 'Judgement', 'The World',
]);

export const TAROT_NUMERALS = new Set([
  '0', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
  'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX',
  'XX', 'XXI',
]);

// UPPERCASE_SNAKE concept names from the 3,033-entry dictionary.
// Starts with a capital letter, then 0-39 more A-Z or _ chars
// (matches the existing 40-char clamp ceiling).
export const CONCEPT_NAME_RE = /^[A-Z][A-Z_]{0,39}$/;
