# view_db.py
import sqlite3
import pandas as pd

# Connect to your local database file
conn = sqlite3.connect('enterprise_itsm.db')

print("--- TABLES IN DATABASE ---")
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
print(tables)

print("\n--- RECORD QUEUE (TICKETS) ---")
try:
    # Fetch all rows from the tickets table
    df = pd.read_sql_query("SELECT * FROM tickets;", conn)
    
    if df.empty:
        print("The tickets table is currently empty.")
    else:
        # Display the full dataframe layout in the terminal
        print(df.to_string(index=False))
except Exception as e:
    print(f"Error reading table: {e}")

conn.close()