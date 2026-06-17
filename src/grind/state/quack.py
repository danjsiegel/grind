from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import duckdb


class QuackConnectionError(RuntimeError):
    pass


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


def quack_connect(uri: str, token: str) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    try:
        uri_sql = _sql_literal(uri)
        token_sql = _sql_literal(token)
        connection.execute("LOAD quack")
        connection.execute(
            f"CREATE SECRET (TYPE quack, TOKEN '{token_sql}', SCOPE '{uri_sql}')"
        )
        connection.execute(f"ATTACH '{uri_sql}' AS grind (TYPE quack)")
        connection.execute("USE grind")
        connection.execute("SET schema = 'main'")
    except Exception as exc:
        connection.close()
        raise QuackConnectionError(f"unable to connect to Quack URI {uri}: {exc}") from exc
    return connection


def serve_local_quack(*, database_path: Path, uri: str, runtime_dir: Path) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    info_path = runtime_dir / "server.json"
    stop_event = threading.Event()

    def _handle_signal(signum, frame):  # pragma: no cover - signal wiring
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    connection = duckdb.connect(str(database_path))
    try:
        for migration_path in sorted(MIGRATIONS_DIR.glob("V*.sql")):
            connection.execute(migration_path.read_text(encoding="utf-8"))
        connection.execute("LOAD quack")
        row = connection.execute(f"CALL quack_serve('{_sql_literal(uri)}')").fetchone()
        if row is None:
            raise QuackConnectionError(f"quack_serve returned no server details for {uri}")
        info = {
            "pid": os.getpid(),
            "db_path": str(database_path),
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