from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import duckdb


class QuackConnectionError(RuntimeError):
    pass


class QuackCursor:
    def __init__(self, rows: list[tuple[Any, ...]]):
        self._rows = rows

    def fetchone(self) -> tuple[Any, ...] | None:
        if not self._rows:
            return None
        return self._rows[0]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._rows)


class QuackConnection:
    def __init__(self, *, connection: duckdb.DuckDBPyConnection, uri: str, token: str):
        self._connection = connection
        self._uri = uri
        self._token = token
        self._disable_ssl = is_local_quack_uri(uri)

    def execute(self, query: str, params: list[Any] | tuple[Any, ...] | None = None) -> QuackCursor:
        rendered = _render_sql(query, params)
        wrapper = (
            "SELECT * FROM quack_query("
            f"'{_sql_literal(self._uri)}', "
            f"'{_sql_literal(rendered)}', "
            f"disable_ssl => {'true' if self._disable_ssl else 'false'}, "
            f"token => '{_sql_literal(self._token)}'"
            ")"
        )
        rows = self._connection.execute(wrapper).fetchall()
        return QuackCursor(rows)

    def close(self) -> None:
        self._connection.close()


LOCAL_QUACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
MIGRATIONS_DIR = Path(__file__).with_name("migrations")


def is_local_quack_uri(uri: str) -> bool:
    if not uri.startswith("quack:"):
        return False
    target = uri[len("quack:") :]
    if not target:
        return False
    host = target
    if target.startswith("["):
        closing = target.find("]")
        host = target[1:closing] if closing != -1 else target
    elif ":" in target:
        host = target.rsplit(":", 1)[0]
    return host in LOCAL_QUACK_HOSTS


def local_quack_runtime_dir(database_path: Path) -> Path:
    return database_path.parent.parent / "quack"


def local_quack_database_path(database_path: Path) -> Path:
    return local_quack_runtime_dir(database_path) / "grind-quack.duckdb"


def ensure_local_quack_server(
    database_path: Path,
    uri: str,
    *,
    wait_timeout_seconds: float = 15.0,
    force_restart: bool = False,
) -> str:
    if not is_local_quack_uri(uri):
        raise QuackConnectionError(f"auto-start only supports local Quack URIs, got: {uri}")

    runtime_dir = local_quack_runtime_dir(database_path)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    info_path = runtime_dir / "server.json"
    existing = _read_server_info(info_path)
    if existing is not None:
        existing_pid = int(existing.get("pid", 0))
        existing_db_path = existing.get("db_path")
        if force_restart and _pid_is_running(existing_pid):
            _terminate_pid(existing_pid)
            existing = None
        elif (
            existing.get("uri") == uri
            and existing_db_path == str(database_path)
            and _pid_is_running(existing_pid)
        ):
            token = existing.get("auth_token")
            if isinstance(token, str) and token:
                return token

    log_path = runtime_dir / "server.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"starting local quack server for {database_path} at {uri}\n")

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "grind.state.quack",
            "serve",
            "--db-path",
            str(database_path),
            "--uri",
            uri,
            "--runtime-dir",
            str(runtime_dir),
        ],
        stdout=log_path.open("a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    deadline = time.monotonic() + wait_timeout_seconds
    while time.monotonic() < deadline:
        info = _read_server_info(info_path)
        if info is not None and info.get("uri") == uri:
            token = info.get("auth_token")
            if isinstance(token, str) and token:
                return token
        if process.poll() is not None:
            break
        time.sleep(0.1)

    tail = ""
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-20:])
    raise QuackConnectionError(
        f"local Quack server did not start for {database_path} at {uri}. Recent log output:\n{tail}"
    )


def _read_server_info(info_path: Path) -> dict[str, object] | None:
    if not info_path.exists():
        return None
    try:
        payload = json.loads(info_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _terminate_pid(pid: int) -> None:
    if pid <= 0:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not _pid_is_running(pid):
            return
        time.sleep(0.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _sql_literal(value: str) -> str:
    return value.replace("'", "''")


def _render_sql(query: str, params: list[Any] | tuple[Any, ...] | None) -> str:
    if not params:
        return query

    pieces = query.split("?")
    if len(pieces) - 1 != len(params):
        raise QuackConnectionError(
            f"parameter count mismatch while rendering SQL for Quack: expected {len(pieces) - 1}, got {len(params)}"
        )

    rendered = [pieces[0]]
    for value, suffix in zip(params, pieces[1:]):
        rendered.append(_value_literal(value))
        rendered.append(suffix)
    return "".join(rendered)


def _value_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float, Decimal)):
        return str(value)
    if isinstance(value, datetime):
        return f"'{_sql_literal(value.isoformat(sep=' '))}'"
    if isinstance(value, date):
        return f"'{_sql_literal(value.isoformat())}'"
    if isinstance(value, (dict, list, tuple)):
        return f"'{_sql_literal(json.dumps(value, default=str))}'"
    return f"'{_sql_literal(str(value))}'"


def quack_connect(uri: str, token: str) -> QuackConnection:
    connection = duckdb.connect()
    try:
        _load_or_install_quack(connection)
    except Exception as exc:
        connection.close()
        raise QuackConnectionError(f"unable to connect to Quack URI {uri}: {exc}") from exc
    return QuackConnection(connection=connection, uri=uri, token=token)


def _load_or_install_quack(connection: duckdb.DuckDBPyConnection) -> None:
    try:
        connection.execute("LOAD quack")
        return
    except Exception:
        connection.execute("INSTALL quack")
        connection.execute("LOAD quack")


def serve_local_quack(*, database_path: Path, uri: str, runtime_dir: Path) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    served_database_path = local_quack_database_path(database_path)
    served_database_path.parent.mkdir(parents=True, exist_ok=True)
    info_path = runtime_dir / "server.json"
    stop_event = threading.Event()

    def _handle_signal(signum, frame):  # pragma: no cover - signal wiring
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    connection = duckdb.connect(str(served_database_path))
    try:
        for migration_path in sorted(MIGRATIONS_DIR.glob("V*.sql")):
            connection.execute(_quack_compatible_sql(migration_path.read_text(encoding="utf-8")))
        _load_or_install_quack(connection)
        row = connection.execute(f"CALL quack_serve('{_sql_literal(uri)}')").fetchone()
        if row is None:
            raise QuackConnectionError(f"quack_serve returned no server details for {uri}")
        info = {
            "pid": os.getpid(),
            "source_db_path": str(database_path),
            "db_path": str(served_database_path),
            "uri": row[0] if len(row) > 0 else uri,
            "http_url": row[1] if len(row) > 1 else None,
            "auth_token": row[2] if len(row) > 2 else None,
        }
        if not info.get("auth_token"):
            raise QuackConnectionError(f"quack_serve did not return an auth token for {uri}")
        info_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        while not stop_event.wait(0.5):
            pass
    finally:
        try:
            if info_path.exists():
                info_path.unlink()
        finally:
            connection.close()


def _quack_compatible_sql(sql: str) -> str:
    sql = re.sub(
        r"\s+REFERENCES\s+[A-Za-z_][A-Za-z0-9_]*\s*\([A-Za-z_][A-Za-z0-9_]*\)",
        "",
        sql,
    )
    sql = re.sub(r"\s+DEFAULT\s+now\(\)", "", sql, flags=re.IGNORECASE)
    sql = re.sub(
        r"INSERT INTO schema_version \(version, description\)\s+SELECT\s+([^,]+),\s*([^\n]+)",
        r"INSERT INTO schema_version (version, applied_at, description)\nSELECT \1, now(), \2",
        sql,
        flags=re.IGNORECASE,
    )
    return sql


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m grind.state.quack")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve")
    serve.add_argument("--db-path", type=Path, required=True)
    serve.add_argument("--uri", required=True)
    serve.add_argument("--runtime-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        serve_local_quack(database_path=args.db_path, uri=args.uri, runtime_dir=args.runtime_dir)
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    raise SystemExit(main())