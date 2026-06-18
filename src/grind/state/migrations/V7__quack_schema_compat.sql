ALTER TABLE schema_version ALTER COLUMN applied_at DROP DEFAULT;

ALTER TABLE runs ALTER COLUMN created_at DROP DEFAULT;
ALTER TABLE runs ALTER COLUMN updated_at DROP DEFAULT;

ALTER TABLE artifacts ALTER COLUMN created_at DROP DEFAULT;

ALTER TABLE retrieval_index_queue ALTER COLUMN queued_at DROP DEFAULT;

ALTER TABLE workspace_checkpoints ALTER COLUMN created_at DROP DEFAULT;

ALTER TABLE tasks ALTER COLUMN created_at DROP DEFAULT;
ALTER TABLE tasks ALTER COLUMN updated_at DROP DEFAULT;

ALTER TABLE stages ALTER COLUMN started_at DROP DEFAULT;

ALTER TABLE transitions ALTER COLUMN created_at DROP DEFAULT;

ALTER TABLE findings ALTER COLUMN first_seen_at DROP DEFAULT;
ALTER TABLE findings ALTER COLUMN last_updated_at DROP DEFAULT;

ALTER TABLE finding_evidence ALTER COLUMN created_at DROP DEFAULT;

ALTER TABLE dispositions ALTER COLUMN created_at DROP DEFAULT;

ALTER TABLE validations ALTER COLUMN created_at DROP DEFAULT;

ALTER TABLE operator_actions ALTER COLUMN created_at DROP DEFAULT;

ALTER TABLE model_calls ALTER COLUMN started_at DROP DEFAULT;

ALTER TABLE semantic_audits ALTER COLUMN created_at DROP DEFAULT;

ALTER TABLE adjudication_panels ALTER COLUMN created_at DROP DEFAULT;

ALTER TABLE adjudication_votes ALTER COLUMN created_at DROP DEFAULT;

ALTER TABLE workers ALTER COLUMN registered_at DROP DEFAULT;
ALTER TABLE workers ALTER COLUMN last_seen_at DROP DEFAULT;

ALTER TABLE run_leases ALTER COLUMN acquired_at DROP DEFAULT;

INSERT INTO schema_version (version, applied_at, description)
SELECT 7, now(), 'drop timestamp defaults for Quack attach compatibility'
WHERE NOT EXISTS (SELECT 1 FROM schema_version WHERE version = 7);
