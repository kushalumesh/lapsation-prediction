"""
Load the policyholder dataset into a local DuckDB database.

    pip install duckdb
    python sql/load_data.py
"""

from pathlib import Path
import duckdb

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
