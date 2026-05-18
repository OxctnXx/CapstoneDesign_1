import sqlite3
from pathlib import Path

from database import BASE_DIR, MYSQL_CONFIG, _quote_identifier, get_db_connection, init_db


SQLITE_DB_PATH = BASE_DIR / "users.db"


TABLES = [
    "users",
    "omr_templates",
    "exams",
    "recognition_records",
]


def _sqlite_columns(conn, table_name):
    rows = conn.execute(f"PRAGMA table_info({_quote_identifier(table_name)})").fetchall()
    return [row["name"] for row in rows]


def _mysql_columns(conn, table_name):
    rows = conn.execute(f"SHOW COLUMNS FROM {_quote_identifier(table_name)}").fetchall()
    return [row["Field"] for row in rows]


def _select_rows(conn, table_name, columns):
    column_sql = ", ".join(_quote_identifier(column) for column in columns)
    return conn.execute(f"SELECT {column_sql} FROM {_quote_identifier(table_name)}").fetchall()


def _insert_or_update_sql(table_name, columns):
    column_sql = ", ".join(_quote_identifier(column) for column in columns)
    placeholders = ", ".join("?" for _ in columns)
    update_columns = [column for column in columns if column != "id"]
    if update_columns:
        update_sql = ", ".join(
            f"{_quote_identifier(column)} = VALUES({_quote_identifier(column)})"
            for column in update_columns
        )
    else:
        update_sql = "id = id"
    return (
        f"INSERT INTO {_quote_identifier(table_name)} ({column_sql}) "
        f"VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {update_sql}"
    )


def migrate(sqlite_path=SQLITE_DB_PATH):
    sqlite_path = Path(sqlite_path)
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {sqlite_path}")

    init_db()

    source = sqlite3.connect(sqlite_path)
    source.row_factory = sqlite3.Row
    target = get_db_connection()

    summary = {}
    try:
        target.execute("SET FOREIGN_KEY_CHECKS = 0")
        for table_name in TABLES:
            sqlite_columns = _sqlite_columns(source, table_name)
            mysql_columns = _mysql_columns(target, table_name)
            columns = [column for column in sqlite_columns if column in mysql_columns]
            if not columns:
                summary[table_name] = 0
                continue

            rows = _select_rows(source, table_name, columns)
            sql = _insert_or_update_sql(table_name, columns)
            for row in rows:
                target.execute(sql, tuple(row[column] for column in columns))
            target.commit()
            summary[table_name] = len(rows)
        target.execute("SET FOREIGN_KEY_CHECKS = 1")
        target.commit()
    except Exception:
        target.rollback()
        raise
    finally:
        source.close()
        target.close()

    return summary


if __name__ == "__main__":
    print(
        "Migrating SQLite data to MySQL "
        f"{MYSQL_CONFIG['user']}@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"
    )
    for table_name, row_count in migrate().items():
        print(f"{table_name}: {row_count} row(s)")
