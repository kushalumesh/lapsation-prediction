"""
Load the policyholder dataset into a local DuckDB database so the queries
in this folder can be run.

DuckDB is file-based, needs no server, and speaks standard SQL including
window functions and CTEs.

    pip install duckdb
    python sql/load_data.py
    duckdb lapsation.duckdb
"""

from pathlib import Path

import duckdb
import pandas as pd

DB = Path("lapsation.duckdb")
SOURCE = Path("data/policyholders.parquet")

if not SOURCE.exists():
    raise SystemExit("Run `python generate_data.py` first.")

con = duckdb.connect(str(DB))
con.execute("DROP TABLE IF EXISTS policyholders")
con.execute(f"CREATE TABLE policyholders AS SELECT * FROM read_parquet('{SOURCE}')")

rows = con.execute("SELECT count(*) FROM policyholders").fetchone()[0]
print(f"Loaded {rows:,} rows into {DB}")
print()
print(con.execute("DESCRIBE policyholders").df().to_string(index=False))
con.close()
