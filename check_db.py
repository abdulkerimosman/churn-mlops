import sqlite3
import os

db_path = 'churn_crm.db'
if os.path.exists(db_path):
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = c.fetchall()
        print(f"Tables: {tables}")
        
        if ('customers',) in tables:
            c.execute("SELECT COUNT(*) FROM customers")
            count = c.fetchone()[0]
            print(f"Count: {count}")
        else:
            print("Table 'customers' does not exist.")
        conn.close()
    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"DB file {db_path} not found.")
