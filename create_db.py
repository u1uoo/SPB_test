import argparse
import glob
import os
import sqlite3
from typing import Dict, List, Optional, Tuple

import pandas as pd


def find_time_column(columns: List[str]) -> Optional[str]:
    candidates = [
        "time",
        "open_time",
    ]
    lowered = {c.lower(): c for c in columns}
    for name in candidates:
        if name in lowered:
            return lowered[name]
    return None


def infer_sqlite_type(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series):
        return "REAL"
    if pd.api.types.is_bool_dtype(series):
        return "INTEGER"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "TEXT"
    return "TEXT"


def quote_ident(name: str) -> str:
    return f'"{name.replace("\"", "\"\"")}"'


def dataframe_to_table(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    table_name: str,
    primary_key: Optional[str] = None,) -> None:
    cur = conn.cursor()

    column_defs: List[str] = []
    for col in df.columns:
        sql_type = infer_sqlite_type(df[col])
        if primary_key is not None and col == primary_key:
            column_defs.append(f"{quote_ident(col)} {sql_type} PRIMARY KEY")
        else:
            column_defs.append(f"{quote_ident(col)} {sql_type}")

    ddl = f"CREATE TABLE {quote_ident(table_name)} (" + ", ".join(column_defs) + ")"

    cur.execute(f"DROP TABLE IF EXISTS {quote_ident(table_name)}")
    cur.execute(ddl)

    placeholders = ", ".join(["?" for _ in df.columns])
    insert_sql = (
        f"INSERT OR REPLACE INTO {quote_ident(table_name)} ("
        + ", ".join(quote_ident(c) for c in df.columns)
        + f") VALUES ({placeholders})"
    )

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    cur.executemany(insert_sql, df.itertuples(index=False, name=None))
    conn.commit()


def build_fundamentals_table(conn: sqlite3.Connection, fundamentals_csv: str) -> None:
    df = pd.read_csv(fundamentals_csv)

    if "symbol" not in df.columns:
        raise ValueError("fundamentals.csv must contain a 'symbol' column")
    df["symbol"] = df["symbol"].astype(str)

    dataframe_to_table(conn, df, table_name="fundamentals", primary_key="symbol")


def build_symbol_table(conn: sqlite3.Connection, csv_path: str, table_name: str) -> None:
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)

    if "symbol" not in df.columns:
        raise ValueError(f"{os.path.basename(csv_path)} must contain a 'symbol' column")

    df["symbol"] = df["symbol"].astype(str).str.strip()
    df = df.dropna(subset=["symbol"]).drop_duplicates(subset=["symbol"])

    dataframe_to_table(conn, df, table_name=table_name, primary_key="symbol")


def parse_symbol_from_filename(path: str) -> Optional[str]:
    base = os.path.basename(path)
    if base.startswith("ohlc_") and base.endswith(".csv"):
        symbol = base[len("ohlc_") : -len(".csv")]
        return symbol or None
    return None


def build_ohlc_tables(conn: sqlite3.Connection, data_dir: str) -> List[Tuple[str, str]]:
    created: List[Tuple[str, str]] = []
    csv_paths = sorted(glob.glob(os.path.join(data_dir, "ohlc_*.csv")))

    for path in csv_paths:
        symbol = parse_symbol_from_filename(path)
        if not symbol:
            continue

        df = pd.read_csv(path)

        if "symbol" in df.columns:
            df = df.drop(columns=["symbol"])

        time_col = find_time_column(df.columns.tolist())

        primary_key = None
        if time_col is not None:
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                try:
                    parsed = pd.to_datetime(df[time_col], errors="raise")
                    df[time_col] = parsed
                except Exception:
                    pass
            primary_key = time_col

        dataframe_to_table(conn, df, table_name=symbol, primary_key=primary_key)
        created.append((symbol, symbol))

    return created


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an SQLite database from CSVs: one OHLCV table per instrument "
            "(from ohlc_*.csv) and a fundamentals table keyed by symbol."
        )
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--data-dir",
        default=os.path.join(script_dir, "data"),
        help="Directory with CSV files (default: ./data next to this script)",
    )
    parser.add_argument(
        "--db-path",
        default=os.path.join(script_dir, "data", "market.db"),
        help="Output SQLite database path (default: ./data/market.db next to this script)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing database file if it exists.",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    db_path = os.path.abspath(args.db_path)

    if not os.path.isdir(data_dir):
        raise SystemExit(f"Data directory not found: {data_dir}")

    if os.path.exists(db_path) and args.overwrite:
        os.remove(db_path)
    ensure_parent_dir(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")

        build_symbol_table(conn, os.path.join(data_dir, "stocks.csv"), table_name="symbols_stocks")
        build_symbol_table(conn, os.path.join(data_dir, "crypto.csv"), table_name="symbols_crypto")

        fundamentals_csv = os.path.join(data_dir, "fundamentals.csv")
        if os.path.exists(fundamentals_csv):
            build_fundamentals_table(conn, fundamentals_csv)


        created = build_ohlc_tables(conn, data_dir)

    print(f"Database created at: {db_path}")


if __name__ == "__main__":
    main()
