# grind

`grind` is a plan/do/check/act workflow for software change.

I first built this workflow for myself as a shell script. This version rebuilds
it around DuckDB as the canonical ledger, adds Quack-aware shared state for
multi-worker setups, and records the work as runs, artifacts, validation output,
findings, and holds instead of leaving everything in terminal scrollback.

The point is simple: do not trust a single pass. Plan the change, make the
change, validate what actually happened, and run an independent check before
calling it done. If a change cannot survive validation and review, Grind does
not call it done.

## What Grind does today

- scaffolds a `.grind/` workspace inside a target repository
- stores runs, tasks, stages, artifacts, validations, findings, checkpoints,
  and operator actions in DuckDB
- runs planner, implementer, checker, and adjudicator steps with resumable holds
- supports Quack-aware state bootstrapping for shared multi-worker setups
- keeps validation output and review artifacts instead of treating them as
  throwaway logs

## What it does not do yet

- shared object-store artifact backends such as S3 or GCS
- OpenTelemetry tracing and metrics
- a plugin marketplace or a broad integration surface
- automatic retention and pruning of ledger data and artifacts

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

If you are running `grind` from this source checkout against some other repository,
invoke it from here and point it at the target repo with `--cwd /path/to/target-repo`.
That target repo gets the `.grind/` directory, state DB, artifacts, and policy pack.

```bash
uv run --project /path/to/grind grind init --cwd /path/to/target-repo
uv run --project /path/to/grind grind verify-backend --cwd /path/to/target-repo --backend github_cli --role planner
uv run --project /path/to/grind grind run --cwd /path/to/target-repo "your objective here"
```

Optional, for remote DuckDB transport experiments over Quack (preview; current upstream docs use `core_nightly`):

```bash
uv run python -c "import duckdb; conn=duckdb.connect(); conn.execute('INSTALL quack FROM core_nightly;'); conn.execute('LOAD quack;')"
```

---

## Quick start

```bash
# 1. choose the repo you want Grind to operate on
TARGET_REPO=/path/to/target-repo

# 2. scaffold config + local storage directories + DuckDB schema in that repo
grind init --cwd "$TARGET_REPO"

# 3. edit $TARGET_REPO/.grind/engine.yaml for your actual providers/models/agents

# 4. verify backend readiness for that repo config
grind verify-backend --cwd "$TARGET_REPO" --backend github_cli --role planner
grind verify-backend --cwd "$TARGET_REPO" --backend kilo_cli --role checker

# 5. create a stored run from an inline objective
grind run --cwd "$TARGET_REPO" "Build the persistence layer and stop at plan review"

# 6. or create a stored run from a handoff/spec file
grind run --cwd "$TARGET_REPO" --objective-file /path/to/V1_1_HANDOFF.md

# 7. inspect the stored ledger in the target repo
grind status --cwd "$TARGET_REPO"
grind findings --cwd "$TARGET_REPO"

# 8. continue from operator hold
grind resume --cwd "$TARGET_REPO" <run_id>

# 9. restore the baseline workspace snapshot if needed
grind restore-checkpoint --cwd "$TARGET_REPO" <run_id>
```

Running `grind init --cwd "$TARGET_REPO"` creates these paths inside the target repo:

- `.grind/engine.yaml`
- `.grind/policy/project.yaml`
- `.grind/state/grind.duckdb`
- `.grind/artifacts/`
- `.grind/archive/`

---

## Configuration

`grind init --cwd /path/to/target-repo` writes `/path/to/target-repo/.grind/engine.yaml`.
That file is also used by `grind verify-backend --cwd /path/to/target-repo --role <role>`
when you omit `--model`.

If the target repo does not have `.grind/engine.yaml`, `grind run` and related commands
fall back to built-in defaults for that target repo. `grind init --cwd ...` is still the
recommended path because it scaffolds an editable engine config and the default policy pack
where the target repo will actually store its run state.

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
    model: kilo/anthropic/claude-sonnet-4.6
    agent: code
    variant: thinking

  checker:
    provider: kilo_cli
    model: kilo/anthropic/claude-sonnet-4.6
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
grind init --cwd /path/to/target-repo
grind init --force
```

Use `--cwd` when Grind is operating on a repo other than the one you launched the
command from. This creates the config file, storage directories, and applies the
DuckDB schema migrations in the target repo.

### `grind verify-backend`

Checks that backend CLIs are installed, authenticated, and ready for model-bound
execution.

```bash
# verify all configured backends
grind verify-backend

# verify one backend explicitly
grind verify-backend --cwd /path/to/target-repo --backend github_cli
grind verify-backend --cwd /path/to/target-repo --backend kilo_cli

# verify a real GitHub CLI model
grind verify-backend --cwd /path/to/target-repo --backend github_cli --model claude-sonnet-4.6

# verify a Kilo model/agent combination
grind verify-backend --cwd /path/to/target-repo --backend kilo_cli --model kilo/anthropic/claude-sonnet-4.6 --agent code --variant thinking

# resolve model settings from .grind/engine.yaml
grind verify-backend --cwd /path/to/target-repo --backend github_cli --role planner
grind verify-backend --cwd /path/to/target-repo --backend kilo_cli --role checker

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
grind run --cwd /path/to/target-repo "Run the Kepler handoff benchmark"

# objective from file
grind run --cwd /path/to/target-repo --objective-file /path/to/V1_1_HANDOFF.md

# machine-readable output
grind run --cwd /path/to/target-repo --objective-file /path/to/V1_1_HANDOFF.md --json
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
grind resume --cwd /path/to/target-repo <run_id>

# restore the selected checkpoint before validation
grind resume --cwd /path/to/target-repo <run_id> --restore-checkpoint

# restore a specific checkpoint before validation
grind resume --cwd /path/to/target-repo <run_id> --restore-checkpoint --checkpoint-id <checkpoint_id>

# machine-readable output
grind resume --cwd /path/to/target-repo <run_id> --json
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
grind restore-checkpoint --cwd /path/to/target-repo <run_id>
grind restore-checkpoint --cwd /path/to/target-repo <run_id> --checkpoint-id <checkpoint_id>
grind restore-checkpoint --cwd /path/to/target-repo <run_id> --json
```

### `grind status`

Summarize the latest run or a specific run directly from DuckDB.

```bash
grind status --cwd /path/to/target-repo
grind status --cwd /path/to/target-repo --run-id <run_id>
grind status --cwd /path/to/target-repo --json
```

### `grind findings`

List stored findings for the latest run or a specific run.

```bash
grind findings --cwd /path/to/target-repo
grind findings --cwd /path/to/target-repo --run-id <run_id>
grind findings --cwd /path/to/target-repo --json
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

