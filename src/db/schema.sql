CREATE TABLE IF NOT EXISTS profiles (
    user_id TEXT PRIMARY KEY,
    stated_tone TEXT NOT NULL DEFAULT 'balanced',
    stated_length TEXT NOT NULL DEFAULT 'balanced',
    stated_sourcing TEXT NOT NULL DEFAULT 'balanced',
    stated_structure TEXT NOT NULL DEFAULT 'balanced',
    stated_revision TEXT NOT NULL DEFAULT 'balanced',
    observed_tone TEXT NOT NULL DEFAULT 'balanced',
    observed_length TEXT NOT NULL DEFAULT 'balanced',
    observed_sourcing TEXT NOT NULL DEFAULT 'balanced',
    observed_structure TEXT NOT NULL DEFAULT 'balanced',
    observed_revision TEXT NOT NULL DEFAULT 'balanced',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    user_message TEXT NOT NULL,
    agent_response TEXT NOT NULL DEFAULT '',
    model_used TEXT NOT NULL CHECK (model_used IN ('nano', 'super', 'ultra')),
    conflict_detected BOOLEAN NOT NULL DEFAULT 0,
    conflict_score FLOAT NOT NULL DEFAULT 0.0,
    routing_reason TEXT NOT NULL DEFAULT '',
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conflicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    stated_value TEXT NOT NULL,
    observed_value TEXT NOT NULL,
    conflict_type TEXT NOT NULL CHECK (conflict_type IN ('tone', 'length', 'sourcing', 'structure', 'revision')),
    confidence FLOAT NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved BOOLEAN NOT NULL DEFAULT 0
);

-- Added later; the three tables above are unchanged.
-- What the user did with an answer: behavior, as opposed to the words in their messages.
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    interaction_id INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('up', 'down', 'regenerate', 'copy')),
    reason TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Moments the note bot writes about: a preference set, a habit learned or faded, a
-- contradiction first seen, a suggestion accepted. `detail` may quote the user; it is
-- erased with the conversation it came from.
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    dimension TEXT NOT NULL DEFAULT '',
    value TEXT NOT NULL DEFAULT '',
    detail TEXT NOT NULL DEFAULT '',
    interaction_id INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Lessons: what Hope reviewed, why it failed, what fixed it. Codes and patterns only, never message text.
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL,
    kind TEXT NOT NULL,                    -- 'rule': a reader rule Hope added
    dimension TEXT NOT NULL DEFAULT '',
    value TEXT NOT NULL DEFAULT '',
    pattern TEXT NOT NULL DEFAULT '',
    persona TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Notes the user deleted ('hidden') or pinned ('saved'). The event log itself is never edited.
CREATE TABLE IF NOT EXISTS note_marks (
    user_id TEXT NOT NULL,
    note_key TEXT NOT NULL,
    state TEXT NOT NULL,
    PRIMARY KEY (user_id, note_key)
);

-- Archived past conversations: interactions first_id..last_id, compressed into `file`.
-- Deleted after expires_at unless saved (kept) = 1.
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    first_id INTEGER NOT NULL,
    last_id INTEGER NOT NULL,
    turns INTEGER NOT NULL,
    bytes INTEGER NOT NULL,
    file TEXT NOT NULL,
    archived_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    saved INTEGER NOT NULL DEFAULT 0
);

-- A simulated wallet per user: a $ limit and what answers have cost so far (list prices).
CREATE TABLE IF NOT EXISTS budgets (
    user_id TEXT PRIMARY KEY,
    limit_usd REAL NOT NULL DEFAULT 0,
    spent_usd REAL NOT NULL DEFAULT 0
);

-- What the Nano signal reader found in a message, so each message is read once.
CREATE TABLE IF NOT EXISTS signal_cache (
    text_hash TEXT PRIMARY KEY,
    signals TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions (user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback (user_id, interaction_id);
CREATE INDEX IF NOT EXISTS idx_conflicts_user ON conflicts (user_id, conflict_type, resolved);
