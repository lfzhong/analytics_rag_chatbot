import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect("feedback_log.db")

# Read the full feedback table into a DataFrame
df = pd.read_sql_query("SELECT * FROM feedback ORDER BY timestamp DESC", conn)

# Close the connection
conn.close()

# Display the data
print(df)  # or use df.to_csv(...) to export
df.to_csv('feedback_log.csv')
