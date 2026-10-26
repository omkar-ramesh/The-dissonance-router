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

CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions (user_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_conflicts_user ON conflicts (user_id, conflict_type, resolved);
