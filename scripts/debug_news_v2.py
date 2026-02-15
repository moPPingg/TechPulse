import sqlite3
import pandas as pd
from pathlib import Path
import os

# Try relative path
db_path = Path("data/news/news.db")
if not db_path.exists():
    # Try absolute path from previous run
    db_path = Path("w:/TechPulse/data/news/news.db")

print(f"Checking DB at: {db_path.absolute()}")

if not db_path.exists():
    print(f"Database STILL not found at {db_path.absolute()}")
    # List files to help debug
    print("Files in data/news:")
    try:
        print(list(Path("data/news").glob("*")))
    except:
        pass
    exit(1)

try:
    conn = sqlite3.connect(str(db_path))
    query = "SELECT id, title, source, url FROM articles ORDER BY id DESC LIMIT 5"
    df = pd.read_sql_query(query, conn)
    print("\nRecent Articles:")
    print(df.to_string())
except Exception as e:
    print(f"Error querying database: {e}")
finally:
    if 'conn' in locals():
        conn.close()
