from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from pathlib import Path
import time

from grind.engine.leases import LeaseConflictError, acquire_lease, expire_stale_leases, heartbeat_worker, release_lease
from grind.engine.orchestrator import DoStageResponsePayload, MinimalOrchestrator
from grind.models import HoldType, OperatorStatus, Run, RunLease, RunState, TaskSourceKind, Worker
from grind.providers import ModelInvocationResult
from grind.validation import ValidationExecutionResult
from grind.state import bootstrap_state_store, open_state_store
from grind.config import init_engine_workspace


def _create_run(store: object, tmp_path: Path, *, run_id: str) -> Run:
    run = Run(
        run_id=run_id,
        repo_path=str(tmp_path),
        policy_pack_path=str(tmp_path / ".grind"),
        policy_schema_ver="0.1",
        requested_objective="lease test",
        state=RunState.CREATED,
        operator_status=OperatorStatus.NONE,
        current_hold_type=HoldType.PLAN_REVIEW,
        total_cost_usd=Decimal("0"),
    )
    store.runs.create(run)
    return run


def test_acquire_and_release_lease_updates_run_owner(tmp_path: Path) -> None:
    database_path = tmp_path / ".grind" / "state" / "grind.duckdb"
    bootstrap_state_store(database_path)

    with open_state_store(database_path) as store:
        run = _create_run(store, tmp_path, run_id="run_lease_1")
        worker = Worker(worker_id="worker_a", hostname="host-a", pid=111)
        store.workers.register(worker)

        lease = acquire_lease(store.connection, run.run_id, worker.worker_id)
        stored_run = store.runs.get(run.run_id)
        active_lease = store.run_leases.get_active_by_run(run.run_id)

        assert stored_run is not None
        assert stored_run.current_worker_id == worker.worker_id
        assert active_lease is not None
        assert active_lease.lease_id == lease.lease_id

        release_lease(store.connection, lease.lease_id)

        released_lease = store.run_leases.get(lease.lease_id)
        released_run = store.runs.get(run.run_id)

    assert released_lease is not None
    assert released_lease.status == "released"
    assert released_run is not None
    assert released_run.current_worker_id is None


def test_concurrent_acquisition_raises_lease_conflict(tmp_path: Path) -> None:
    database_path = tmp_path / ".grind" / "state" / "grind.duckdb"
    bootstrap_state_store(database_path)

    with open_state_store(database_path) as store:
        run = _create_run(store, tmp_path, run_id="run_lease_2")
        store.workers.register(Worker(worker_id="worker_a", hostname="host-a", pid=111))
        store.workers.register(Worker(worker_id="worker_b", hostname="host-b", pid=222))
        acquire_lease(store.connection, run.run_id, "worker_a")

    with open_state_store(database_path) as store:
        try:
            acquire_lease(store.connection, "run_lease_2", "worker_b")
        except LeaseConflictError as error:
            assert "worker_a" in str(error)
        else:  # pragma: no cover - explicit guard
            raise AssertionError("expected LeaseConflictError")


def test_expire_stale_leases_requires_missed_heartbeat(tmp_path: Path) -> None:
    database_path = tmp_path / ".grind" / "state" / "grind.duckdb"
    bootstrap_state_store(database_path)

    with open_state_store(database_path) as store:
        healthy_run = _create_run(store, tmp_path, run_id="run_lease_healthy")
        stale_run = _create_run(store, tmp_path, run_id="run_lease_stale")
        store.workers.register(Worker(worker_id="worker_healthy", hostname="host-a", pid=111))
        store.workers.register(Worker(worker_id="worker_stale", hostname="host-b", pid=222))
        healthy_lease = acquire_lease(store.connection, healthy_run.run_id, "worker_healthy")
        stale_lease = acquire_lease(store.connection, stale_run.run_id, "worker_stale")
        stale_worker = store.workers.get("worker_stale")
        assert stale_worker is not None
        store.workers.heartbeat(
            "worker_stale",
            last_seen_at=stale_worker.last_seen_at - timedelta(seconds=120),
        )

        heartbeat_worker(store.connection, "worker_healthy")
        expired = expire_stale_leases(store.connection, heartbeat_timeout_seconds=30)

        assert [lease.lease_id for lease in expired] == [stale_lease.lease_id]
        assert store.run_leases.get(healthy_lease.lease_id) == RunLease(
            lease_id=healthy_lease.lease_id,
            run_id=healthy_lease.run_id,
            worker_id=healthy_lease.worker_id,
            acquired_at=healthy_lease.acquired_at,
            released_at=None,
            status="active",
        )
        expired_lease = store.run_leases.get(stale_lease.lease_id)
        stale_run_after = store.runs.get(stale_run.run_id)

    assert expired_lease is not None
    assert expired_lease.status == "expired"
    assert stale_run_after is not None
    assert stale_run_after.current_worker_id is None


def test_release_allows_worker_handoff(tmp_path: Path) -> None:
    database_path = tmp_path / ".grind" / "state" / "grind.duckdb"
    bootstrap_state_store(database_path)

    with open_state_store(database_path) as store:
        run = _create_run(store, tmp_path, run_id="run_lease_handoff")
        store.workers.register(Worker(worker_id="worker_a", hostname="host-a", pid=111))
        store.workers.register(Worker(worker_id="worker_b", hostname="host-b", pid=222))
        first_lease = acquire_lease(store.connection, run.run_id, "worker_a")
        release_lease(store.connection, first_lease.lease_id)
        second_lease = acquire_lease(store.connection, run.run_id, "worker_b")
        active_lease = store.run_leases.get_active_by_run(run.run_id)
        stored_run = store.runs.get(run.run_id)

    assert active_lease is not None
    assert active_lease.lease_id == second_lease.lease_id
    assert stored_run is not None
    assert stored_run.current_worker_id == "worker_b"


def test_worker_heartbeat_during_long_stage(tmp_path: Path, monkeypatch) -> None:
    init_engine_workspace(tmp_path)
    orchestrator = MinimalOrchestrator(cwd=tmp_path)
    orchestrator.lease_heartbeat_interval_seconds = 0.02
    database_path = tmp_path / ".grind" / "state" / "grind.duckdb"
    observations: dict[str, object] = {}

    def fake_invoke_text_prompt(profile, *, prompt: str, cwd: Path) -> ModelInvocationResult:
        with open_state_store(database_path) as store:
            first_seen = store.workers.get(orchestrator.worker_id)
            assert first_seen is not None
            observations["first_seen_at"] = first_seen.last_seen_at
        time.sleep(0.08)
        with open_state_store(database_path) as store:
            second_seen = store.workers.get(orchestrator.worker_id)
            assert second_seen is not None
            observations["second_seen_at"] = second_seen.last_seen_at
        return ModelInvocationResult(command=["fake-planner"], stdout='{"plan":"ship it"}', stderr="", returncode=0)

    monkeypatch.setattr("grind.engine.orchestrator.invoke_text_prompt", fake_invoke_text_prompt)

    outcome = orchestrator.run(objective="lease heartbeat", source_kind=TaskSourceKind.INLINE)

    assert observations["second_seen_at"] > observations["first_seen_at"]
    with open_state_store(database_path) as store:
        assert store.run_leases.get_active_by_run(outcome.run_id) is None


def test_resume_after_hold_allows_worker_handoff(tmp_path: Path, monkeypatch) -> None:
    init_engine_workspace(tmp_path)
    database_path = tmp_path / ".grind" / "state" / "grind.duckdb"

    monkeypatch.setattr(
        "grind.engine.orchestrator.invoke_text_prompt",
        lambda profile, *, prompt, cwd: ModelInvocationResult(
            command=["fake-planner"],
            stdout='{"plan":"ship it"}',
            stderr="",
            returncode=0,
        ),
    )
    monkeypatch.setattr(
        MinimalOrchestrator,
        "_run_do_stage",
        lambda self, *, store, run, task, iteration: DoStageResponsePayload(
            touched_files=["README.md"],
            touched_symbols=[],
            validation_hints=[],
            claims_made=[],
            open_uncertainties=[],
            artifact_refs=[],
        ),
    )
    monkeypatch.setattr(
        MinimalOrchestrator,
        "_run_validation_review_cycle",
        lambda self, *, store, run, task, iteration, observed_delta: ("hold", "validation_stage_1", []),
    )
    monkeypatch.setattr(
        "grind.engine.orchestrator.run_validation_commands",
        lambda cwd, commands, *, stop_on_failure, timeout_seconds: [
            ValidationExecutionResult(command=commands[0], returncode=0, stdout="passed", stderr="")
        ],
    )

    worker_a = MinimalOrchestrator(cwd=tmp_path)
    run_outcome = worker_a.run(objective="handoff test", source_kind=TaskSourceKind.INLINE)

    worker_b = MinimalOrchestrator(cwd=tmp_path)
    resume_outcome = worker_b.resume(run_id=run_outcome.run_id)

    assert resume_outcome.final_state == RunState.AWAITING_OPERATOR
    with open_state_store(database_path) as store:
        leases = store.run_leases.list_by_run(run_outcome.run_id)
        latest_lease = leases[-1]

    assert len(leases) == 2
    assert latest_lease.worker_id == worker_b.worker_id
    assert latest_lease.status == "released"