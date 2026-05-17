ALTER TABLE runs ADD COLUMN IF NOT EXISTS validation_commands_override JSON;
ALTER TABLE operator_actions ADD COLUMN IF NOT EXISTS payload JSON;

INSERT INTO schema_version (version, description)
SELECT 4, 'run policy overrides and structured operator action payloads'
WHERE NOT EXISTS (SELECT 1 FROM schema_version WHERE version = 4);