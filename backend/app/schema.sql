PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runtime_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    csrf_token TEXT NOT NULL,
    expires_at INTEGER NOT NULL,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_user_idx ON sessions(user_id);
CREATE INDEX IF NOT EXISTS sessions_expiry_idx ON sessions(expires_at);

CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL REFERENCES users(id),
    name TEXT NOT NULL CHECK(length(name) BETWEEN 1 AND 80),
    model TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('provisioning', 'provisioning_error', 'ready')),
    paused INTEGER NOT NULL DEFAULT 0 CHECK(paused IN (0, 1)),
    chat_thread_id TEXT,
    chat_epoch INTEGER NOT NULL DEFAULT 0 CHECK(chat_epoch >= 0),
    blockers_json TEXT NOT NULL DEFAULT '[]',
    generation INTEGER NOT NULL DEFAULT 0 CHECK(generation >= 0),
    last_idle_at INTEGER,
    created_at INTEGER NOT NULL,
    UNIQUE(id, owner_id)
);
CREATE INDEX IF NOT EXISTS agents_owner_idx ON agents(owner_id, created_at, id);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(id),
    name TEXT NOT NULL CHECK(length(name) BETWEEN 1 AND 80),
    prompt TEXT NOT NULL,
    created_by TEXT NOT NULL CHECK(created_by IN ('user', 'agent')),
    run_at INTEGER NOT NULL,
    interval_seconds INTEGER CHECK(interval_seconds IS NULL OR interval_seconds BETWEEN 60 AND 31536000),
    next_run_at INTEGER,
    enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
    cancelled_at INTEGER,
    created_at INTEGER NOT NULL,
    UNIQUE(id, agent_id)
);
CREATE INDEX IF NOT EXISTS jobs_due_idx ON jobs(enabled, next_run_at);
CREATE INDEX IF NOT EXISTS jobs_agent_idx ON jobs(agent_id, created_at, id);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(id),
    source TEXT NOT NULL CHECK(source IN ('chat', 'job')),
    job_id TEXT,
    retry_of TEXT,
    prompt TEXT NOT NULL,
    scheduled_for INTEGER,
    coalesced_count INTEGER NOT NULL DEFAULT 0 CHECK(coalesced_count >= 0),
    status TEXT NOT NULL CHECK(status IN (
        'queued', 'starting', 'running', 'stopping', 'completed',
        'completed_with_errors', 'failed', 'interrupted', 'cancelled'
    )),
    model TEXT NOT NULL,
    chat_epoch INTEGER NOT NULL DEFAULT 0 CHECK(chat_epoch >= 0),
    thread_id TEXT,
    sdk_turn_id TEXT,
    run_token_hash TEXT,
    agent_generation INTEGER,
    termination_reason TEXT,
    intended_terminal_status TEXT CHECK(
        intended_terminal_status IS NULL OR intended_terminal_status IN (
            'completed', 'completed_with_errors', 'failed', 'interrupted', 'cancelled'
        )
    ),
    intended_termination_reason TEXT,
    cleanup_state TEXT NOT NULL DEFAULT 'not_required' CHECK(
        cleanup_state IN ('not_required', 'pending', 'confirmed', 'failed')
    ),
    created_at INTEGER NOT NULL,
    started_at INTEGER,
    finished_at INTEGER,
    error_code TEXT,
    error_message TEXT,
    had_tool_errors INTEGER NOT NULL DEFAULT 0 CHECK(had_tool_errors IN (0, 1)),
    FOREIGN KEY(job_id, agent_id) REFERENCES jobs(id, agent_id),
    FOREIGN KEY(retry_of, agent_id) REFERENCES runs(id, agent_id),
    CHECK((source = 'chat' AND job_id IS NULL) OR (source = 'job' AND job_id IS NOT NULL)),
    UNIQUE(id, agent_id)
);
CREATE INDEX IF NOT EXISTS runs_agent_created_idx ON runs(agent_id, created_at, id);
CREATE INDEX IF NOT EXISTS runs_queue_idx ON runs(status, created_at, id);
CREATE INDEX IF NOT EXISTS runs_job_idx ON runs(job_id, status);
CREATE UNIQUE INDEX IF NOT EXISTS runs_job_occurrence_idx
    ON runs(job_id, scheduled_for)
    WHERE job_id IS NOT NULL AND retry_of IS NULL AND scheduled_for IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS runs_one_nonterminal_job_idx
    ON runs(job_id)
    WHERE job_id IS NOT NULL AND status IN ('queued', 'starting', 'running', 'stopping');

CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL CHECK(sequence > 0),
    kind TEXT NOT NULL CHECK(kind IN (
        'message_delta', 'message_snapshot', 'activity_started',
        'activity_completed', 'error', 'state'
    )),
    sdk_item_id TEXT,
    payload_json TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY(run_id, sequence)
);

CREATE TABLE IF NOT EXISTS idempotency (
    actor_kind TEXT NOT NULL CHECK(actor_kind IN ('user', 'agent')),
    actor_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    body_hash TEXT NOT NULL,
    entity_id TEXT,
    operation_state TEXT NOT NULL,
    response_code INTEGER,
    response_json TEXT,
    created_at INTEGER NOT NULL,
    PRIMARY KEY(actor_kind, actor_id, operation, resource_id, request_id)
);
CREATE INDEX IF NOT EXISTS idempotency_created_idx ON idempotency(created_at);

CREATE TABLE IF NOT EXISTS model_run_starts (
    id INTEGER PRIMARY KEY,
    started_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS model_run_starts_time_idx ON model_run_starts(started_at);

CREATE TABLE IF NOT EXISTS quota_projects (
    agent_id TEXT PRIMARY KEY REFERENCES agents(id),
    project_id INTEGER NOT NULL UNIQUE CHECK(project_id > 0),
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS platform_storage_state (
    id INTEGER PRIMARY KEY CHECK(id = 1),
    protection_latched INTEGER NOT NULL DEFAULT 0 CHECK(protection_latched IN (0, 1)),
    updated_at INTEGER NOT NULL
);
INSERT OR IGNORE INTO platform_storage_state(id, protection_latched, updated_at)
VALUES (1, 0, 0);

INSERT OR IGNORE INTO runtime_metadata(key, value) VALUES ('schema_version', '1');
