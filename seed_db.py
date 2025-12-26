import os
import sys
import pandas as pd
from dashboard import database

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(f"CSV Path: {CSV_PATH}")

if not os.path.exists(CSV_PATH):
    print("Error: CSV file not found!")
    sys.exit(1)

# Force init and seed
print("Initializing DB...")
database.init_db()

print("Seeding DB...")
try:
    database.seed_database(CSV_PATH)
    print("Seeding complete.")
except Exception as e:
    print(f"Seeding failed: {e}")
    import traceback
    traceback.print_exc()

# Verify
print("Verifying...")
df = database.load_customers()
print(f"Loaded {len(df)} customers from DB.")
