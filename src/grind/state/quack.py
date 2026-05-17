from __future__ import annotations

import duckdb


class QuackConnectionError(RuntimeError):
    pass


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