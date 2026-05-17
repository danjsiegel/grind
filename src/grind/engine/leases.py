from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
import random
import secrets
import threading
import time

import duckdb

from grind.models.run_lease import RunLease
from grind.state.repositories import RunLeaseRepository, RunRepository, WorkerRepository


class LeaseConflictError(RuntimeError):
    pass


class BackgroundHeartbeat:
    def __init__(self, *, heartbeat: Callable[[], None], interval_seconds: float):
        self._heartbeat = heartbeat
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="grind-worker-heartbeat", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=max(self._interval_seconds * 4, 0.2))

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            self._heartbeat()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _retryable_writer_conflict(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        fragment in message
        for fragment in (
            "write-write conflict",
            "transaction conflict",
            "concurrent modification",
            "catalog write lock",
            "serialize",
            "conflict on update",
        )
    )


def acquire_lease(
    connection: duckdb.DuckDBPyConnection,
    run_id: str,
    worker_id: str,
    *,
    max_attempts: int = 4,
    min_retry_seconds: float = 0.01,
    max_retry_seconds: float = 0.05,
) -> RunLease:
    lease_repository = RunLeaseRepository(connection)
    run_repository = RunRepository(connection)
    worker_repository = WorkerRepository(connection)
    worker = worker_repository.get(worker_id)
    if worker is None:
        raise ValueError(f"worker not registered: {worker_id}")

    for attempt in range(max_attempts):
        try:
            existing = lease_repository.get_active_by_run(run_id)
            if existing is not None:
                if existing.worker_id == worker_id:
                    run_repository.set_current_worker(run_id, current_worker_id=worker_id)
                    return existing
                raise LeaseConflictError(
                    f"run {run_id} is already leased by worker {existing.worker_id}"
                )

            lease = RunLease(
                lease_id=f"lease_{secrets.token_hex(8)}",
                run_id=run_id,
                worker_id=worker_id,
            )
            lease_repository.create(lease)
            run_repository.set_current_worker(run_id, current_worker_id=worker_id)
            worker_repository.heartbeat(worker_id)
            return lease
        except LeaseConflictError:
            raise
        except Exception as error:
            active = lease_repository.get_active_by_run(run_id)
            if active is not None:
                if active.worker_id == worker_id:
                    run_repository.set_current_worker(run_id, current_worker_id=worker_id)
                    return active
                raise LeaseConflictError(
                    f"run {run_id} is already leased by worker {active.worker_id}"
                ) from error
            if attempt == max_attempts - 1 or not _retryable_writer_conflict(error):
                raise
            time.sleep(random.uniform(min_retry_seconds, max_retry_seconds))

    raise RuntimeError(f"unable to acquire lease for run {run_id}")


def release_lease(connection: duckdb.DuckDBPyConnection, lease_id: str) -> None:
    lease_repository = RunLeaseRepository(connection)
    lease = lease_repository.get(lease_id)
    if lease is None:
        return
    lease_repository.release(lease_id)
    RunRepository(connection).set_current_worker(lease.run_id, current_worker_id=None)


def heartbeat_worker(connection: duckdb.DuckDBPyConnection, worker_id: str) -> None:
    WorkerRepository(connection).heartbeat(worker_id)


def expire_stale_leases(
    connection: duckdb.DuckDBPyConnection,
    heartbeat_timeout_seconds: int,
) -> list[RunLease]:
    lease_repository = RunLeaseRepository(connection)
    worker_repository = WorkerRepository(connection)
    run_repository = RunRepository(connection)
    cutoff = _utc_now() - timedelta(seconds=heartbeat_timeout_seconds)
    expired: list[RunLease] = []
    for lease in lease_repository.list_active():
        worker = worker_repository.get(lease.worker_id)
        if worker is None or worker.last_seen_at < cutoff:
            lease_repository.expire(lease.lease_id)
            run_repository.set_current_worker(lease.run_id, current_worker_id=None)
            updated = lease_repository.get(lease.lease_id)
            if updated is not None:
                expired.append(updated)
    return expired