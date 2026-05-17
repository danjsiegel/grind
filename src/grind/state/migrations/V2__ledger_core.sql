CREATE TABLE IF NOT EXISTS tasks (
  task_id             TEXT PRIMARY KEY,
  run_id              TEXT NOT NULL REFERENCES runs(run_id),
  sequence            INTEGER NOT NULL,
  source_kind         TEXT NOT NULL,
  raw_input           TEXT NOT NULL,
  normalized_scope    JSON,
  phase_label         TEXT,
  acceptance_checks   JSON NOT NULL DEFAULT '[]',
  status              TEXT NOT NULL DEFAULT 'pending',
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS stages (
  stage_id                TEXT PRIMARY KEY,
  run_id                  TEXT NOT NULL REFERENCES runs(run_id),
  task_id                 TEXT NOT NULL REFERENCES tasks(task_id),
  stage_name              TEXT NOT NULL,
  started_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
  ended_at                TIMESTAMPTZ,
  status                  TEXT NOT NULL DEFAULT 'pending',
  model_role              TEXT,
  model_name              TEXT,
  provider                TEXT,
  runtime_agent           TEXT,
  runtime_variant         TEXT,
  prompt_artifact_id      TEXT,
  response_artifact_id    TEXT,
  output_artifact_id      TEXT,
  summary                 TEXT,
  iteration               INTEGER NOT NULL DEFAULT 1,
  latency_ms              INTEGER
);

CREATE TABLE IF NOT EXISTS transitions (
  transition_id       TEXT PRIMARY KEY,
  run_id              TEXT NOT NULL REFERENCES runs(run_id),
  from_state          TEXT NOT NULL,
  to_state            TEXT NOT NULL,
  reason              TEXT NOT NULL,
  actor               TEXT NOT NULL,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS findings (
  finding_id          TEXT PRIMARY KEY,
  run_id              TEXT NOT NULL REFERENCES runs(run_id),
  stage_id            TEXT NOT NULL REFERENCES stages(stage_id),
  stable_id           TEXT NOT NULL,
  title               TEXT NOT NULL,
  severity            TEXT NOT NULL,
  confidence          TEXT NOT NULL,
  category            TEXT NOT NULL,
  rationale           TEXT NOT NULL,
  exact_fix_action    TEXT NOT NULL,
  status              TEXT NOT NULL DEFAULT 'open',
  first_seen_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  adjudicated         BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS finding_evidence (
  evidence_id         TEXT PRIMARY KEY,
  finding_id          TEXT NOT NULL REFERENCES findings(finding_id),
  evidence_type       TEXT NOT NULL,
  artifact_id         TEXT REFERENCES artifacts(artifact_id),
  snippet             TEXT,
  source_ref          TEXT,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dispositions (
  disposition_id      TEXT PRIMARY KEY,
  finding_id          TEXT NOT NULL REFERENCES findings(finding_id),
  stage_id            TEXT NOT NULL REFERENCES stages(stage_id),
  iteration           INTEGER NOT NULL,
  decided_by          TEXT NOT NULL,
  decision            TEXT NOT NULL,
  justification       TEXT NOT NULL,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO schema_version (version, description)
SELECT 2, 'core ledger tables for tasks, stages, transitions, findings, and dispositions'
WHERE NOT EXISTS (SELECT 1 FROM schema_version WHERE version = 2);