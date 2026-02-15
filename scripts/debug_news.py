import sqlite3
import pandas as pd
from pathlib import Path

# Adjust path as necessary
db_path = Path("w:/TechPulse/data/news/news.db")

if not db_path.exists():
    print(f"Database not found at {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
query = """
SELECT id, title, source, url, published_at 
FROM articles 
ORDER BY published_at DESC 
LIMIT 10
"""

try:
    df = pd.read_sql_query(query, conn)
    print("Recent Articles:")
    print(df.to_string())
except Exception as e:
    print(f"Error querying database: {e}")
finally:
    conn.close()
