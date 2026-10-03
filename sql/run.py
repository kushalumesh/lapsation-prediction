"""Run a .sql file against lapsation.duckdb and print each result.

    python sql/run.py sql/01_lapse_rate_by_cohort.sql
    python sql/run.py            # runs every query in order
"""
import sys
from pathlib import Path

import duckdb

SQL_DIR = Path(__file__).parent
DB = SQL_DIR.parent / "lapsation.duckdb"

if not DB.exists():
    raise SystemExit("Database not found. Run: python sql/load_data.py")

files = ([Path(sys.argv[1])] if len(sys.argv) > 1
         else sorted(SQL_DIR.glob("[0-9][0-9]_*.sql")))

con = duckdb.connect(str(DB))
for f in files:
    print("=" * 78)
    print(f.name)
    print("=" * 78)
    body = "\n".join(l for l in f.read_text().split("\n")
                     if not l.strip().startswith("--"))
    for stmt in [s.strip() for s in body.split(";") if s.strip()]:
        print(con.execute(stmt).df().to_string(index=False))
        print()
con.close()
