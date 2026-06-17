from __future__ import annotations

import os
from pathlib import Path

import duckdb

from grind.state.repositories import DuckDBStateStore
from grind.state.quack import QuackConnectionError, ensure_local_quack_server, is_local_quack_uri, quack_connect


MIGRATIONS_DIR = Path(__file__).with_name("migrations")


def _resolve_database_path(database_path: Path, *, db_uri: str | None) -> Path:
    if db_uri and not db_uri.startswith("quack:"):
        return Path(db_uri)
    return database_path


def _open_connection(
    database_path: Path,
    *,
    db_uri: str | None = None,
    quack_token: str | None = None,
    read_only: bool = False,
) -> duckdb.DuckDBPyConnection:
    effective_db_uri = db_uri or os.getenv("GRIND_DB_URI")
    if effective_db_uri and effective_db_uri.startswith("quack:"):
        effective_token = quack_token or os.getenv("GRIND_DB_TOKEN")
        if not effective_token and is_local_quack_uri(effective_db_uri):
            effective_token = ensure_local_quack_server(database_path, effective_db_uri)
        if not effective_token:
            raise QuackConnectionError(
                "GRIND_DB_TOKEN is required when GRIND_DB_URI points at Quack"
            )
        try:
            return quack_connect(effective_db_uri, effective_token)
        except QuackConnectionError:
            if not is_local_quack_uri(effective_db_uri):
                raise
            effective_token = ensure_local_quack_server(database_path, effective_db_uri, force_restart=True)
            return quack_connect(effective_db_uri, effective_token)

    resolved_path = _resolve_database_path(database_path, db_uri=effective_db_uri)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(resolved_path), read_only=read_only)


def bootstrap_state_store(
    database_path: Path,
    *,
    db_uri: str | None = None,
    quack_token: str | None = None,
) -> None:
    resolved_path = _resolve_database_path(database_path, db_uri=db_uri or os.getenv("GRIND_DB_URI"))
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    connection = _open_connection(database_path, db_uri=db_uri, quack_token=quack_token)
    try:
        for migration_path in sorted(MIGRATIONS_DIR.glob("V*.sql")):
            connection.execute(migration_path.read_text(encoding="utf-8"))
    finally:
        connection.close()


def current_schema_version(
    database_path: Path,
    *,
    db_uri: str | None = None,
    quack_token: str | None = None,
) -> int | None:
    connection = _open_connection(
        database_path,
        db_uri=db_uri,
        quack_token=quack_token,
        read_only=True,
    )
    try:
        row = connection.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        ).fetchone()
    finally:
        connection.close()

    if row is None:
        return None
    return int(row[0])


def open_state_store(
    database_path: Path,
    *,
    db_uri: str | None = None,
    quack_token: str | None = None,
) -> DuckDBStateStore:
    connection = _open_connection(database_path, db_uri=db_uri, quack_token=quack_token)
    return DuckDBStateStore(connection=connection, database_path=database_path)