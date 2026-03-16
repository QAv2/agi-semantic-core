-- AGI Semantic Core — SQLite Schema
-- 16D Dual Octonion Semantic Dictionary
--
-- Tables: concepts, aliases, relations, validation_log
-- Encoding: 8D essence [w,x,y,z,e,f,g,h] + 8D function [fw,fx,fy,fz,fe,ff,fg,fh]

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- =============================================================================
-- CONCEPTS — The dictionary itself
-- =============================================================================
CREATE TABLE IF NOT EXISTS concepts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE COLLATE NOCASE,

    -- Core 4D quaternion (essence)
    w           REAL    NOT NULL DEFAULT 1.0,
    x           REAL    NOT NULL DEFAULT 0.0,
    y           REAL    NOT NULL DEFAULT 0.0,
    z           REAL    NOT NULL DEFAULT 0.0,

    -- Domain 4D (essence)
    e           REAL    NOT NULL DEFAULT 0.0,   -- Spatial
    f           REAL    NOT NULL DEFAULT 0.0,   -- Temporal
    g           REAL    NOT NULL DEFAULT 0.0,   -- Relational
    h           REAL    NOT NULL DEFAULT 0.0,   -- Personal

    -- Function 8D (how it operates)
    fx          REAL    NOT NULL DEFAULT 0.0,
    fy          REAL    NOT NULL DEFAULT 0.0,
    fz          REAL    NOT NULL DEFAULT 0.0,
    fe          REAL    NOT NULL DEFAULT 0.0,
    ff          REAL    NOT NULL DEFAULT 0.0,
    fg          REAL    NOT NULL DEFAULT 0.0,
    fh          REAL    NOT NULL DEFAULT 0.0,

    -- Ontological metadata
    level       INTEGER NOT NULL DEFAULT 5,     -- ConceptLevel enum value (0-8)
    description TEXT    NOT NULL DEFAULT '',
    hexagram_ref INTEGER NOT NULL DEFAULT 0,    -- King Wen number (0 = unmapped)
    trigram     TEXT    NOT NULL DEFAULT '',     -- Computed: QIAN/KUN/ZHEN/etc.
    pos         TEXT    NOT NULL DEFAULT '',     -- Part of speech: noun/verb/adj/adv/prep/conj/det/pron/num/shape

    -- Provenance
    session     INTEGER NOT NULL DEFAULT 0,     -- Which session created/last modified this
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- =============================================================================
-- ALIASES — Alternate names pointing to canonical concepts
-- =============================================================================
CREATE TABLE IF NOT EXISTS aliases (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    alias       TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    concept_id  INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_aliases_concept ON aliases(concept_id);

-- =============================================================================
-- RELATIONS — Semantic links between concepts
-- =============================================================================
CREATE TABLE IF NOT EXISTS relations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    concept1_id INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    concept2_id INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    rel_type    TEXT    NOT NULL,                -- synonym/affinity/adjacent/complement/opposition
    angle_4d    REAL    NOT NULL DEFAULT 0.0,    -- Computed angle between 3D cores
    angle_8d    REAL    NOT NULL DEFAULT 0.0,    -- Computed angle in 7D space
    session     INTEGER NOT NULL DEFAULT 0,      -- Session that established this relation
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now')),

    UNIQUE(concept1_id, concept2_id)
);

CREATE INDEX IF NOT EXISTS idx_relations_c1 ON relations(concept1_id);
CREATE INDEX IF NOT EXISTS idx_relations_c2 ON relations(concept2_id);
CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(rel_type);

-- =============================================================================
-- VALIDATION_LOG — Audit trail for validation runs
-- =============================================================================
CREATE TABLE IF NOT EXISTS validation_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    total_concepts  INTEGER NOT NULL,
    total_relations INTEGER NOT NULL,
    complements_checked INTEGER NOT NULL DEFAULT 0,
    complements_valid   INTEGER NOT NULL DEFAULT 0,
    issues      TEXT    NOT NULL DEFAULT '[]',   -- JSON array of issue descriptions
    summary     TEXT    NOT NULL DEFAULT ''
);

-- =============================================================================
-- VIEWS — Common queries
-- =============================================================================

-- Full concept with trigram and level name
CREATE VIEW IF NOT EXISTS v_concepts AS
SELECT
    c.id, c.name,
    c.w, c.x, c.y, c.z,
    c.e, c.f, c.g, c.h,
    c.fx, c.fy, c.fz, c.fe, c.ff, c.fg, c.fh,
    c.level,
    CASE c.level
        WHEN 0 THEN 'UNITY'
        WHEN 1 THEN 'DYAD'
        WHEN 2 THEN 'TRIAD'
        WHEN 3 THEN 'TETRAD'
        WHEN 4 THEN 'QUALITY'
        WHEN 5 THEN 'DERIVED'
        WHEN 6 THEN 'VERB'
        WHEN 7 THEN 'ABSTRACT'
        WHEN 8 THEN 'INTERROGATIVE'
    END AS level_name,
    c.description,
    c.hexagram_ref,
    c.trigram,
    c.session,
    c.created_at,
    c.updated_at
FROM concepts c;

-- Relations with concept names
CREATE VIEW IF NOT EXISTS v_relations AS
SELECT
    r.id,
    c1.name AS concept1,
    c2.name AS concept2,
    r.rel_type,
    r.angle_4d,
    r.angle_8d,
    r.session
FROM relations r
JOIN concepts c1 ON r.concept1_id = c1.id
JOIN concepts c2 ON r.concept2_id = c2.id;

-- Concept + all aliases
CREATE VIEW IF NOT EXISTS v_aliases AS
SELECT
    c.id AS concept_id,
    c.name AS canonical,
    a.alias
FROM concepts c
LEFT JOIN aliases a ON a.concept_id = c.id;

-- Trigram distribution
CREATE VIEW IF NOT EXISTS v_trigram_dist AS
SELECT
    trigram,
    COUNT(*) AS count,
    GROUP_CONCAT(name, ', ') AS concepts
FROM concepts
WHERE trigram != ''
GROUP BY trigram
ORDER BY count DESC;
