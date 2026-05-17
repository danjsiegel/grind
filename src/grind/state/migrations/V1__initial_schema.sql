CREATE TABLE IF NOT EXISTS schema_version (
  version        INTEGER PRIMARY KEY,
  applied_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  description    TEXT
);

CREATE TABLE IF NOT EXISTS runs (
  run_id              TEXT PRIMARY KEY,
  repo_path           TEXT NOT NULL,
  policy_pack_path    TEXT NOT NULL,
  policy_schema_ver   TEXT NOT NULL,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  state               TEXT NOT NULL DEFAULT 'created',
  requested_objective TEXT NOT NULL,
  normalized_scope    JSON,
  operator_status     TEXT NOT NULL DEFAULT 'none',
  current_worker_id   TEXT,
  iteration_count     INTEGER NOT NULL DEFAULT 0,
  max_iterations      INTEGER NOT NULL DEFAULT 3,
  budget_limit_usd    DECIMAL(12,6),
  total_cost_usd      DECIMAL(12,6) NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id    TEXT PRIMARY KEY,
  run_id         TEXT NOT NULL REFERENCES runs(run_id),
  artifact_type  TEXT NOT NULL,
  path           TEXT NOT NULL,
  storage_kind   TEXT NOT NULL DEFAULT 'local',
  checksum       TEXT,
  size_bytes     INTEGER,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  metadata       JSON
);

CREATE TABLE IF NOT EXISTS retrieval_index_queue (
  queue_id       TEXT PRIMARY KEY,
  run_id         TEXT NOT NULL REFERENCES runs(run_id),
  artifact_id    TEXT NOT NULL REFERENCES artifacts(artifact_id),
  collection     TEXT NOT NULL,
  queue_status   TEXT NOT NULL DEFAULT 'pending',
  attempts       INTEGER NOT NULL DEFAULT 0,
  last_error     TEXT,
  queued_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at     TIMESTAMPTZ,
  completed_at   TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS workspace_checkpoints (
  checkpoint_id        TEXT PRIMARY KEY,
  run_id               TEXT NOT NULL REFERENCES runs(run_id),
  task_id              TEXT,
  stage_id             TEXT,
  iteration            INTEGER NOT NULL DEFAULT 0,
  checkpoint_kind      TEXT NOT NULL,
  capture_mode         TEXT NOT NULL,
  scope_paths          JSON NOT NULL,
  artifact_id          TEXT NOT NULL REFERENCES artifacts(artifact_id),
  status               TEXT NOT NULL DEFAULT 'available',
  created_by           TEXT NOT NULL,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
  restored_at          TIMESTAMPTZ
);

INSERT INTO schema_version (version, description)
SELECT 1, 'initial scaffold for verification, retrieval queue, and checkpoints'
WHERE NOT EXISTS (SELECT 1 FROM schema_version WHERE version = 1);
