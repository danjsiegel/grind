CREATE TABLE IF NOT EXISTS validations (
  validation_id        TEXT PRIMARY KEY,
  run_id               TEXT NOT NULL REFERENCES runs(run_id),
  task_id              TEXT NOT NULL REFERENCES tasks(task_id),
  stage_id             TEXT NOT NULL REFERENCES stages(stage_id),
  command              TEXT NOT NULL,
  status               TEXT NOT NULL,
  required             BOOLEAN NOT NULL DEFAULT TRUE,
  exit_code            INTEGER,
  stdout_artifact_id   TEXT,
  stderr_artifact_id   TEXT,
  summary              TEXT,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at         TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS operator_actions (
  action_id            TEXT PRIMARY KEY,
  run_id               TEXT NOT NULL REFERENCES runs(run_id),
  action_type          TEXT NOT NULL,
  note                 TEXT,
  checkpoint_id        TEXT,
  payload              JSON,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO schema_version (version, description)
SELECT 3, 'validation results and operator action ledger'
WHERE NOT EXISTS (SELECT 1 FROM schema_version WHERE version = 3);