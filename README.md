# grind

`grind` is a plan/do/check/act engine for software change that treats evidence as
the product, not a nice-to-have. It separates planner, implementer, checker, and
adjudicator roles, keeps the run ledger in DuckDB, and makes validation, semantic
audit, findings, and holds part of the workflow instead of cleanup after the fact.

If a change cannot survive validation and independent review, Grind does not call
it done. The goal is fewer hallucinated fixes, fewer unsupported claims, and a
cleaner path from request to inspectable result.

## What this repo is today

Not the entire long-range spec. This repository now implements the v0.1 fleet-core contract of it:

- `grind init` scaffolds `.grind/engine.yaml`, `.grind/policy/project.yaml`, and the local storage layout
- DuckDB remains the canonical run ledger locally, and the state bootstrap layer is Quack-aware for shared deployments via `GRIND_DB_URI` or `state.db_uri`
- `grind run` creates a stored run, captures a baseline checkpoint, persists planner artifacts, and records the policy pack used for the run
- `grind resume` registers workers, acquires run leases, runs shell-free validation with timeout and risky-command holds, persists semantic-audit and evidence-verification artifacts, then drives checker, adjudicator, and acting loops
- operator commands such as `approve`, `reject`, `abort`, `hold-reason`, `patch-policy`, `inspect`, `report`, `retrieval-index`, and `retrieval-search` are implemented against the stored ledger

Still not full-spec or still intentionally narrowed:

- shared object-store artifact backends such as S3 or GCS
- OpenTelemetry spans and metrics
- the broader generic backend/platform surface beyond the shipped first-party runtime integrations
- some long-horizon spec areas that exist as design targets but are not v0.1 ship gates

So the honest answer is still simple: this is usable, real, and no longer just a local-only prototype, but it is not the full long-term spec.

---

## Prerequisites

`grind` delegates model execution to one or both of these backends. Install and
authenticate whichever you intend to use.

### GitHub Copilot CLI (`github_cli`)

```bash
brew install gh
gh auth login
gh auth status
```

`github_cli` uses `gh copilot`. Your GitHub account must have an active
Copilot entitlement.

### Kilo CLI (`kilo_cli`)

```bash
# install per vendor docs
kilo auth login
kilo auth list
kilo models
kilo agent list
```

For Kilo, use model IDs and agent names that actually exist in your local
environment. The examples below are realistic examples, not a guarantee that
those exact IDs exist in your install.

---

## Install

```bash
uv sync --extra dev
```

Optional, for remote DuckDB transport experiments over Quack (preview; current upstream docs use `core_nightly`):

```bash
uv run python -c "import duckdb; conn=duckdb.connect(); conn.execute('INSTALL quack FROM core_nightly;'); conn.execute('LOAD quack;')"
```

---

## Quick start

```bash
# 1. scaffold config + local storage directories + DuckDB schema
grind init

# 2. edit .grind/engine.yaml for your actual providers/models/agents

# 3. verify backend readiness
grind verify-backend --backend github_cli --role planner
grind verify-backend --backend kilo_cli --role checker

# 4. create a stored run from an inline objective
grind run "Build the persistence layer and stop at plan review"

# 5. or create a stored run from a handoff/spec file
grind run --objective-file /path/to/V1_1_HANDOFF.md

# 6. inspect the stored ledger
grind status
grind findings

# 7. continue from operator hold
grind resume <run_id>

# 8. restore the baseline workspace snapshot if needed
grind restore-checkpoint <run_id>
```

Running `grind init` creates:

- `.grind/engine.yaml`
- `.grind/policy/project.yaml`
- `.grind/state/grind.duckdb`
- `.grind/artifacts/`
- `.grind/archive/`

---

## Configuration

`grind init` writes `.grind/engine.yaml`. This file is also used by
`grind verify-backend --role <role>` when you omit `--model`.

If `.grind/engine.yaml` is missing, `grind run` and related commands fall back to built-in defaults. `grind init` is still the recommended path because it scaffolds an editable engine config and the default policy pack.

```yaml
# .grind/engine.yaml
state:
  kind: duckdb
  path: .grind/state/grind.duckdb

artifacts:
  root: .grind/artifacts

retention:
  mode: manual
  export_root: .grind/archive
  keep_artifacts_days:

validation:
  commands:
    - uv run pytest tests -q
  stop_on_failure: true

models:
  planner:
    provider: github_cli
    model: claude-sonnet-4.6

  implementer:
    provider: kilo_cli
    model: qwen-3.6-plus
    agent: code
    variant: thinking

  checker:
    provider: kilo_cli
    model: qwen-3.6-plus
    agent: ask
    variant: instant

  adjudicator:
    provider: github_cli
    model: claude-sonnet-4.6
```

Fields per role entry:

| Field | Required | Description |
|---|---|---|
| `provider` | yes | `github_cli` or `kilo_cli` |
| `model` | yes | provider-native model identifier accepted by that backend CLI |
| `agent` | no | Kilo agent name; if supplied, `kilo_cli.agent_presence` will verify it |
| `variant` | no | stored in resolved identity today; intended for runtime/profile selection |

Storage fields:

| Field | Required | Description |
|---|---|---|
| `state.kind` | yes | currently `duckdb` |
| `state.path` | yes | canonical ledger path; default `.grind/state/grind.duckdb` |
| `artifacts.root` | yes | local artifact root; default `.grind/artifacts` |
| `retention.mode` | yes | currently `manual`; Grind does not auto-prune ledger data |
| `retention.export_root` | yes | reserved for future export/prune flows |
| `retention.keep_artifacts_days` | no | placeholder for future retention policy |
| `validation.commands` | yes | shell commands run by `grind resume` during the validation slice |
| `validation.stop_on_failure` | yes | stop after the first failing validation command |

---

## Commands

### `grind init`

Initialize the local engine workspace.

```bash
grind init
grind init --cwd /path/to/repo
grind init --force
```

This creates the config file, storage directories, and applies the DuckDB schema
migrations.

### `grind verify-backend`

Checks that backend CLIs are installed, authenticated, and ready for model-bound
execution.

```bash
# verify all configured backends
grind verify-backend

# verify one backend explicitly
grind verify-backend --backend github_cli
grind verify-backend --backend kilo_cli

# verify a real GitHub CLI model
grind verify-backend --backend github_cli --model claude-sonnet-4.6

# verify a Kilo model/agent combination
grind verify-backend --backend kilo_cli --model qwen-3.6-plus --agent code --variant thinking

# resolve model settings from .grind/engine.yaml
grind verify-backend --backend github_cli --role planner
grind verify-backend --backend kilo_cli --role checker

# fail if skipped probes would otherwise be treated as not-applicable
grind verify-backend --strict

# machine-readable output
grind verify-backend --json
```

Exit codes: `0` all required probes passed, `1` a required probe failed, `2`
configuration error or inconclusive result.

### `grind run`

Create a real stored run in DuckDB.

```bash
# inline objective
grind run "Run the Kepler handoff benchmark"

# objective from file
grind run --objective-file /path/to/V1_1_HANDOFF.md

# machine-readable output
grind run --objective-file /path/to/V1_1_HANDOFF.md --json
```

Current behavior of `grind run`:

- creates a `runs` row
- creates a single `tasks` row
- captures a baseline `workspace_checkpoints` row and checkpoint artifact
- invokes the configured planner backend and persists planning prompt/response artifacts
- creates a completed planning stage and plan review artifact
- records `created -> planning -> plan_review -> awaiting_operator` transitions
- leaves the run in `awaiting_operator` with `operator_status = hold`

### `grind resume`

Resume a held run after operator review.

```bash
# continue the latest held run path
grind resume <run_id>

# restore the selected checkpoint before validation
grind resume <run_id> --restore-checkpoint

# restore a specific checkpoint before validation
grind resume <run_id> --restore-checkpoint --checkpoint-id <checkpoint_id>

# machine-readable output
grind resume <run_id> --json
```

Current behavior of `grind resume`:

- records an operator `resume` action
- optionally restores a stored checkpoint artifact into the workspace
- runs a real implementer-backed `doing` stage and persists prompt/response/output artifacts
- transitions the run through `plan_ready`, `doing`, `awaiting_validation`, and `validating`
- runs the configured `validation.commands`
- persists validation stdout/stderr artifacts and `validations` rows
- builds a `difference_surface` artifact and a `semantic_audit_report` artifact during `semantic_auditing`
- invokes real checker and adjudicator stages after validation passes
- captures a `pre_act` checkpoint and runs a real implementer-backed `acting` stage when adjudicated blockers remain
- returns to `awaiting_operator` on validation failure or unresolved post-act blockers
- completes the run when the adjudicated actionable set is empty

### `grind restore-checkpoint`

Restore the latest or specified checkpoint artifact into the workspace.

```bash
grind restore-checkpoint <run_id>
grind restore-checkpoint <run_id> --checkpoint-id <checkpoint_id>
grind restore-checkpoint <run_id> --json
```

### `grind status`

Summarize the latest run or a specific run directly from DuckDB.

```bash
grind status
grind status --run-id <run_id>
grind status --json
```

### `grind findings`

List stored findings for the latest run or a specific run.

```bash
grind findings
grind findings --run-id <run_id>
grind findings --json
```

---

## Persistence layout

The local canonical ledger is DuckDB.

- DB path: `.grind/state/grind.duckdb`
- Artifacts root: `.grind/artifacts`
- Export root: `.grind/archive`

The ledger currently stores:

- runs
- tasks
- stages
- transitions
- artifacts
- findings
- dispositions
- workspace checkpoints
- validations
- operator actions

Retention posture is currently manual. Grind does not auto-delete ledger rows or
artifact history.

---

## Project layout

```text
src/grind/
  cli.py               # grind init, grind run, grind verify-backend
  config.py            # engine config loading and default config rendering
  artifacts/
    store.py           # local artifact writer
  engine/
    orchestrator.py    # stored-run orchestration with planning/do/validate/audit/check/act slices
    state_machine.py   # RunState transition rules
  models/              # Pydantic domain models and ledger record models
  state/
    migrations/        # DuckDB schema migrations
    repositories.py    # typed DuckDB repositories
    store.py           # bootstrap/open helpers
  verification/        # backend probe infrastructure
tests/
```

## Tests

```bash
uv run pytest
```

