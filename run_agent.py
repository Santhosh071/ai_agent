"""
run_agent.py - Script to run the AI agent from the terminal environment.

This script installs necessary dependencies, sets up the SQLite database,
loads raw CSV data from raw/, cleans and saves cleaned CSV files in clean/,
loads cleaned CSV for database insertion,
configures the Gemini LLM, and provides an interactive CLI for querying.
"""

# --- Phase 0: Initial Setup and Imports ---
import subprocess
import sys
import time
import os

def install_packages():
    print("Installing/upgrading required Python packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-U",
            "google-generativeai", "pandas", "matplotlib", "seaborn"
        ])
        print("Packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Package installation failed: {e}")
        sys.exit(1)

install_packages()

import pandas as pd
import sqlite3
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns

print("Libraries imported: pandas, sqlite3, google.generativeai, matplotlib, seaborn.")

# --- Phase 1: Database Setup and Connection ---
db_file_path = 'ecommerce_data.db'
print(f"Using temporary SQLite database at: {db_file_path}")

conn = sqlite3.connect(db_file_path)
cursor = conn.cursor()
print("Connected to SQLite database.")

# --- Phase 2: Create SQL Tables ---
create_eligibility_sql = """
CREATE TABLE IF NOT EXISTS product_level_eligibility_table (
    item_id INTEGER NOT NULL,
    eligibility_datetime_utc DATETIME NOT NULL,
    eligibility BOOLEAN,
    message TEXT,
    PRIMARY KEY (item_id, eligibility_datetime_utc)
);
"""
cursor.execute(create_eligibility_sql)
print("Table 'product_level_eligibility_table' ready.")

create_ad_sales_sql = """
CREATE TABLE IF NOT EXISTS product_level_ad_sales_metrics (
    date DATE NOT NULL,
    item_id INTEGER NOT NULL,
    ad_sales REAL,
    impressions INTEGER,
    ad_spend REAL,
    clicks INTEGER,
    units_sold INTEGER,
    PRIMARY KEY (date, item_id)
);
"""
cursor.execute(create_ad_sales_sql)
print("Table 'product_level_ad_sales_metrics' ready.")

create_total_sales_sql = """
CREATE TABLE IF NOT EXISTS product_level_total_sales_metrics (
    date DATE NOT NULL,
    item_id INTEGER NOT NULL,
    total_sales REAL,
    total_units_ordered INTEGER,
    PRIMARY KEY (date, item_id)
);
"""
cursor.execute(create_total_sales_sql)
print("Table 'product_level_total_sales_metrics' ready.")

conn.commit()
print("All table schemas committed.")

# --- Phase 3: Load raw CSV from raw/, clean, save to clean/, then load cleaned CSV ---
RAW_FOLDER = "raw"
CLEAN_FOLDER = "clean"

os.makedirs(CLEAN_FOLDER, exist_ok=True)

# Define paths for raw CSV files
raw_eligibility_path = os.path.join(RAW_FOLDER, "product_level_eligibility_table_improved.csv")
raw_ad_sales_path = os.path.join(RAW_FOLDER, "product_level_ad_sales_and_metrics_improved.csv")
raw_total_sales_path = os.path.join(RAW_FOLDER, "product_level_total_sales_and_metrics_improved.csv")

# Load raw CSV files
df_eligibility = pd.read_csv(raw_eligibility_path)
df_ad_sales = pd.read_csv(raw_ad_sales_path)
df_total_sales = pd.read_csv(raw_total_sales_path)

# Clean / preprocess
df_eligibility['eligibility_datetime_utc'] = pd.to_datetime(df_eligibility['eligibility_datetime_utc'])
df_eligibility['eligibility'] = df_eligibility['eligibility'].astype(int)

df_ad_sales['date'] = pd.to_datetime(df_ad_sales['date'])
df_total_sales['date'] = pd.to_datetime(df_total_sales['date'])

# Save cleaned CSV files
clean_eligibility_csv = os.path.join(CLEAN_FOLDER, "product_level_eligibility_table_improved.csv")
clean_ad_sales_csv = os.path.join(CLEAN_FOLDER, "product_level_ad_sales_and_metrics_improved.csv")
clean_total_sales_csv = os.path.join(CLEAN_FOLDER, "product_level_total_sales_and_metrics_improved.csv")

df_eligibility.to_csv(clean_eligibility_csv, index=False)
df_ad_sales.to_csv(clean_ad_sales_csv, index=False)
df_total_sales.to_csv(clean_total_sales_csv, index=False)

print("Raw CSV files loaded, cleaned CSV files saved.")

# Load cleaned CSV files for insertion
df_eligibility = pd.read_csv(clean_eligibility_csv)
df_eligibility['eligibility_datetime_utc'] = pd.to_datetime(df_eligibility['eligibility_datetime_utc'])
df_eligibility['eligibility'] = df_eligibility['eligibility'].astype(int)

df_ad_sales = pd.read_csv(clean_ad_sales_csv)
df_ad_sales['date'] = pd.to_datetime(df_ad_sales['date'])

df_total_sales = pd.read_csv(clean_total_sales_csv)
df_total_sales['date'] = pd.to_datetime(df_total_sales['date'])

print("Cleaned CSV files loaded and processed.")

# Insert data into SQLite tables
print("Inserting data into SQL tables...")
df_eligibility.to_sql('product_level_eligibility_table', conn, if_exists='replace', index=False)
print("Eligibility data inserted.")

df_ad_sales.to_sql('product_level_ad_sales_metrics', conn, if_exists='replace', index=False)
print("Ad sales data inserted.")

df_total_sales.to_sql('product_level_total_sales_metrics', conn, if_exists='replace', index=False)
print("Total sales data inserted.")

conn.commit()
print("All data committed successfully.")

# --- Phase 4: Configure Gemini Flash LLM ---
print("\nConfiguring Gemini Flash LLM...")

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print("ERROR: GOOGLE_API_KEY environment variable not found.")
    print("Set the GOOGLE_API_KEY environment variable before running this script.")
    sys.exit(1)

genai.configure(api_key=GOOGLE_API_KEY)
print("Gemini API key configured.")

gemini_model = genai.GenerativeModel('models/gemini-1.5-flash')
print("Gemini Flash model initialized.")

# --- Phase 5: Define Core Functions ---

def get_table_schema(cursor):
    """Retrieve schemas of all tables in the SQLite database."""
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    schema_info = {}
    for (table_name,) in tables:
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        column_descriptions = []
        for cid, name, ctype, notnull, dflt_value, pk in columns:
            desc = f"{name} {ctype}"
            if notnull:
                desc += " NOT NULL"
            if pk:
                desc += " PRIMARY KEY"
            column_descriptions.append(desc)
        schema_info[table_name] = ", ".join(column_descriptions)
    return schema_info

def ask_llm_for_sql(question: str, schema: dict, gemini_model) -> str:
    """Generate an SQL query from a natural language question using Gemini LLM."""
    schema_str = "\n".join([f"Table: {table}\nColumns: {cols}" for table, cols in schema.items()])

    prompt_parts = [
        "You are an expert SQL query generator. Your sole purpose is to convert user questions into precise and accurate SQL queries.",
        "Use the database schema below to identify correct tables and columns.",
        "Strictly adhere to the schema provided.",
        "",
        "Database schema:",
        schema_str,
        "",
        "**Important SQL Generation Rules:**",
        "- Select appropriate table(s) based on question context.",
        "- Use aggregate functions where applicable (SUM, AVG, MAX, MIN, COUNT).",
        "- Use JOINs only when needed based on common columns.",
        "- Filter with WHERE clauses for specific values.",
        "- Use GROUP BY, ORDER BY, LIMIT as relevant.",
        "- Match table and column names exactly as in schema.",
        "- Use float division for RoAS and CPC calculations.",
        "- Output only the SQL query, no explanations or formatting.",
        "",
        "Examples:",
        "User Question: What is my total sales?",
        "SQL Query: SELECT SUM(total_sales) FROM product_level_total_sales_metrics;",
        "",
        f"User Question: {question}",
        "SQL Query:"
    ]

    try:
        response = gemini_model.generate_content("\n".join(prompt_parts))
        generated_text = response.text.strip()

        # Clean possible markdown or code block wrappers
        for marker in ["```sql", "```"]:
            if generated_text.lower().startswith(marker):
                generated_text = generated_text[len(marker):].strip()
            if generated_text.endswith(marker):
                generated_text = generated_text[:-len(marker)].strip()

        # Extract SQL starting with valid SQL keywords
        sql_lines = generated_text.splitlines()
        sql_query_lines = []
        capture = False
        sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "WITH", "PRAGMA"]

        for line in sql_lines:
            line_upper = line.strip().upper()
            if not capture and any(line_upper.startswith(kw) for kw in sql_keywords):
                capture = True
            if capture:
                if line_upper == "" and not any(c.isalpha() for c in line_upper) and ';' not in line_upper:
                    break
                sql_query_lines.append(line)

        sql_query = "\n".join(sql_query_lines).strip()
        if sql_query and not sql_query.endswith(';'):
            if not any(sql_query.split()[-1].upper().endswith(k) for k in ["FROM", "WHERE", "BY", "LIMIT", "ON"]):
                sql_query += ';'

        return sql_query
    except Exception as e:
        print(f"Error generating or parsing SQL: {e}")
        return "ERROR: Failed to generate or parse SQL query."

def execute_sql_query(cursor, sql_query):
    """Execute an SQL query and return the result and column names."""
    try:
        cursor.execute(sql_query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        return results, columns
    except sqlite3.Error as e:
        print(f"SQL execution error: {e}")
        return None, None

def format_answer(question: str, results: list, columns: list) -> str:
    """Convert SQL results into a readable string format."""
    if results is None:
        return "An error occurred while querying the database."
    if not results:
        return "No relevant data found for that question."

    df = pd.DataFrame(results, columns=columns)
    if len(results) == 1 and len(results[0]) == 1:
        val = results[0][0]
        if isinstance(val, (int, float)):
            formatted_val = f"{val:,.2f}" if 'sales' in question.lower() or 'spend' in question.lower() else str(val)
            return f"The {question.lower()} is: {formatted_val}"
        else:
            return f"The answer to '{question}' is: {val}"
    else:
        return f"Data for '{question}':\n{df.to_string(index=False)}"

def stream_response(text: str, delay: float = 0.02):
    """Simulate typing effect when printing output."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def ask_ai_agent_cli(question: str, gemini_model, cursor) -> None:
    """Main AI agent function for CLI interaction."""
    stream_response("AI: Processing your request...", delay=0.01)

    schema = get_table_schema(cursor)
    stream_response("AI: Generating SQL query...", delay=0.01)
    sql_query = ask_llm_for_sql(question, schema, gemini_model)
    stream_response(f"AI: Generated SQL:\n{sql_query}", delay=0.005)

    if sql_query.startswith("ERROR:"):
        stream_response(sql_query, delay=0.03)
        return

    stream_response("AI: Executing SQL query...", delay=0.01)
    results, columns = execute_sql_query(cursor, sql_query)

    stream_response("AI: Formatting answer...", delay=0.01)
    answer = format_answer(question, results, columns)

    stream_response("AI: Here is the result:", delay=0.01)
    stream_response(answer, delay=0.005)

# --- Interactive CLI Loop ---
if __name__ == "__main__":
    schema = get_table_schema(cursor)
    stream_response("\n--- Database Schema ---", delay=0.01)
    for table, cols in schema.items():
        stream_response(f"{table}: {cols}", delay=0.005)
    stream_response("---------------------\n", delay=0.01)

    stream_response("--- AI Agent CLI ---", delay=0.01)
    stream_response("Enter your questions below (type 'exit' to quit).", delay=0.01)
    stream_response("Examples: 'What is my total sales?', 'Calculate the RoAS.'", delay=0.01)
    stream_response("---------------------\n", delay=0.01)

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            stream_response("AI: Goodbye!", delay=0.05)
            break
        ask_ai_agent_cli(user_input, gemini_model, cursor)

    conn.close()
    stream_response("AI: Database connection closed.", delay=0.01)
