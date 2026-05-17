CREATE TABLE IF NOT EXISTS workers (
  worker_id     TEXT PRIMARY KEY,
  hostname      TEXT NOT NULL,
  pid           INTEGER NOT NULL,
  registered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_seen_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS run_leases (
  lease_id     TEXT PRIMARY KEY,
  run_id       TEXT NOT NULL REFERENCES runs(run_id),
  worker_id    TEXT NOT NULL REFERENCES workers(worker_id),
  acquired_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  released_at  TIMESTAMPTZ,
  active_run_key TEXT,
  status       TEXT NOT NULL DEFAULT 'active'
    CHECK (status IN ('active', 'released', 'expired')),
  CHECK (
    (status = 'active' AND active_run_key = run_id)
    OR (status IN ('released', 'expired') AND active_run_key IS NULL)
  )
);

CREATE UNIQUE INDEX IF NOT EXISTS ix_run_leases_one_active
  ON run_leases (active_run_key);

INSERT INTO schema_version (version, description)
SELECT 6, 'worker registration and active run lease coordination'
WHERE NOT EXISTS (SELECT 1 FROM schema_version WHERE version = 6);