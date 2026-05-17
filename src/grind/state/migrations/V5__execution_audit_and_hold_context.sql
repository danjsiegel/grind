ALTER TABLE runs ADD COLUMN IF NOT EXISTS current_hold_type TEXT;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS current_hold_reason TEXT;
ALTER TABLE runs ADD COLUMN IF NOT EXISTS current_hold_context JSON;

CREATE TABLE IF NOT EXISTS model_calls (
  model_call_id        TEXT PRIMARY KEY,
  run_id               TEXT NOT NULL REFERENCES runs(run_id),
  stage_id             TEXT NOT NULL REFERENCES stages(stage_id),
  model_role           TEXT NOT NULL,
  provider             TEXT NOT NULL,
  model_name           TEXT NOT NULL,
  runtime_agent        TEXT,
  runtime_variant      TEXT,
  command              JSON,
  status               TEXT NOT NULL,
  started_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at         TIMESTAMPTZ,
  latency_ms           INTEGER,
  input_tokens         INTEGER,
  output_tokens        INTEGER,
  estimated_cost_usd   DECIMAL(12,6),
  metadata             JSON,
  error_reason         TEXT
);

CREATE TABLE IF NOT EXISTS semantic_audits (
  semantic_audit_id           TEXT PRIMARY KEY,
  run_id                      TEXT NOT NULL REFERENCES runs(run_id),
  task_id                     TEXT NOT NULL REFERENCES tasks(task_id),
  stage_id                    TEXT NOT NULL REFERENCES stages(stage_id),
  iteration                   INTEGER NOT NULL,
  capability_level            TEXT NOT NULL,
  hard_fail                   BOOLEAN NOT NULL DEFAULT FALSE,
  blocking_findings           JSON,
  advisory_findings           JSON,
  unsupported_checks          JSON,
  report_artifact_id          TEXT NOT NULL REFERENCES artifacts(artifact_id),
  difference_surface_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
  summary                     TEXT,
  created_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS adjudication_panels (
  panel_id                    TEXT PRIMARY KEY,
  run_id                      TEXT NOT NULL REFERENCES runs(run_id),
  task_id                     TEXT NOT NULL REFERENCES tasks(task_id),
  stage_id                    TEXT NOT NULL REFERENCES stages(stage_id),
  iteration                   INTEGER NOT NULL,
  mode                        TEXT NOT NULL,
  primary_reason              TEXT,
  status                      TEXT NOT NULL,
  disagreement_artifact_id    TEXT,
  summary                     TEXT,
  created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at                TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS adjudication_votes (
  vote_id                     TEXT PRIMARY KEY,
  panel_id                    TEXT NOT NULL REFERENCES adjudication_panels(panel_id),
  run_id                      TEXT NOT NULL REFERENCES runs(run_id),
  stage_id                    TEXT NOT NULL REFERENCES stages(stage_id),
  member_label                TEXT NOT NULL,
  provider                    TEXT NOT NULL,
  model_name                  TEXT NOT NULL,
  runtime_agent               TEXT,
  runtime_variant             TEXT,
  response_artifact_id        TEXT,
  output_artifact_id          TEXT,
  payload                     JSON NOT NULL,
  summary                     TEXT,
  created_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO schema_version (version, description)
SELECT 5, 'execution audit ledger, semantic audit records, adjudication votes, and structured hold context'
WHERE NOT EXISTS (SELECT 1 FROM schema_version WHERE version = 5);