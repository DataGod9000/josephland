# datatools-portfolio

Backend for an internal DataTools platform: DDL parse/apply, table compare, and validation. Uses **FastAPI**, **Supabase Postgres**, SQL-first design, safe DDL validation, and audit logging to `datatools.audit_log`.

## Tech

- **Backend:** Python FastAPI  
- **DB:** Supabase Postgres (connection via `DATABASE_URL` in `.env`)  
- **SQL:** SQL-first (no full-table load into Python); identifier validation; no raw arbitrary SQL execution  
- **Audit:** All write actions log to `datatools.audit_log`

## Prerequisites

- Python 3.10+
- Supabase project with `datatools` schema and metadata tables (e.g. `audit_log`, `table_registry`, `compare_runs`, `validation_runs`) and roles already set up (Steps 1–3 done).

## Setup (exact steps)

1. **Create a virtual environment**

   ```bash
   cd datatools-portfolio
   python3 -m venv .venv
   ```

2. **Activate the venv**

   - macOS/Linux: `source .venv/bin/activate`
   - Windows: `.venv\Scripts\activate`

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Create `.env` from example**

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set:

   - `DATABASE_URL` – your Supabase Postgres connection string (use only here, never hardcode in code).
   - `ALLOWED_SCHEMAS` – e.g. `dev,prod` (comma-separated; used for `env_schema` validation).

5. **Run the API**

   ```bash
   uvicorn main:app --reload --port 8000
   ```

6. **Open the app**

   - **Web UI:** **http://127.0.0.1:8000/** — simple frontend to try DDL parse/apply, compare, and validate.
   - **Swagger API docs:** **http://127.0.0.1:8000/docs**

## Sample workflow

### 1. DDL parse (no DB)

Parse a `CREATE TABLE` and get schema, table name, columns, and constraints:

```bash
curl -s -X POST http://127.0.0.1:8000/ddl/parse \
  -H "Content-Type: application/json" \
  -d '{"ddl": "CREATE TABLE dev.users (id INT PRIMARY KEY, name TEXT NOT NULL);"}'
```

### 2. DDL apply (create table in Postgres)

Apply the same DDL to a given schema (e.g. `dev`). Table is created and registered; audit log written.

```bash
curl -s -X POST http://127.0.0.1:8000/ddl/apply \
  -H "Content-Type: application/json" \
  -d '{"ddl": "CREATE TABLE dev.users (id INT PRIMARY KEY, name TEXT NOT NULL);", "env_schema": "dev"}'
```

### 3. Seed dummy data and run compare + validate

Create a second table and insert dummy rows (run in your DB client or Supabase SQL editor):

```sql
-- In dev schema (example)
CREATE TABLE dev.orders (id INT PRIMARY KEY, user_id INT, amount NUMERIC(10,2));
INSERT INTO dev.users (id, name) VALUES (1, 'alice'), (2, 'bob'), (3, 'carol');
INSERT INTO dev.orders (id, user_id, amount) VALUES (1, 1, 10.5), (2, 2, 20.0), (3, 99, 30.0);
```

Then:

**Suggest join keys (compare two tables):**

```bash
curl -s -X POST http://127.0.0.1:8000/compare/suggest-keys \
  -H "Content-Type: application/json" \
  -d '{"left_table": "users", "right_table": "orders", "env_schema": "dev", "max_candidates": 5}'
```

**Run compare (row counts, missing left/right, sample of differences):**

```bash
curl -s -X POST http://127.0.0.1:8000/compare/run \
  -H "Content-Type: application/json" \
  -d '{"left_table": "users", "right_table": "orders", "env_schema": "dev", "join_keys": ["id"], "compare_columns": null, "sample_limit": 50}'
```

**Run validation (row count, nulls, duplicate keys):**

```bash
curl -s -X POST http://127.0.0.1:8000/validate/run \
  -H "Content-Type: application/json" \
  -d '{"target_table": "users", "env_schema": "dev", "key_columns": ["id"]}'
```

## Example curl block (copy-paste)

```bash
# 1. DDL parse
curl -s -X POST http://127.0.0.1:8000/ddl/parse \
  -H "Content-Type: application/json" \
  -d '{"ddl": "CREATE TABLE dev.users (id INT PRIMARY KEY, name TEXT NOT NULL);"}'

# 2. DDL apply
curl -s -X POST http://127.0.0.1:8000/ddl/apply \
  -H "Content-Type: application/json" \
  -d '{"ddl": "CREATE TABLE dev.users (id INT PRIMARY KEY, name TEXT NOT NULL);", "env_schema": "dev"}'

# 3. Compare suggest-keys
curl -s -X POST http://127.0.0.1:8000/compare/suggest-keys \
  -H "Content-Type: application/json" \
  -d '{"left_table": "users", "right_table": "orders", "env_schema": "dev", "max_candidates": 5}'

# 4. Compare run
curl -s -X POST http://127.0.0.1:8000/compare/run \
  -H "Content-Type: application/json" \
  -d '{"left_table": "users", "right_table": "orders", "env_schema": "dev", "join_keys": ["id"], "compare_columns": null, "sample_limit": 50}'

# 5. Validate run
curl -s -X POST http://127.0.0.1:8000/validate/run \
  -H "Content-Type: application/json" \
  -d '{"target_table": "users", "env_schema": "dev", "key_columns": ["id"]}'
```

## API summary

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/ddl/parse` | Parse CREATE TABLE DDL (schema, table, columns, constraints). |
| POST | `/ddl/apply` | Apply CREATE TABLE to Postgres; register in `table_registry`; audit log. |
| POST | `/compare/suggest-keys` | Suggest join key candidates from common columns (name+type) and scores. |
| POST | `/compare/run` | Row counts, missing_in_left/right, sample; store in `compare_runs`; audit. |
| POST | `/validate/run` | Row count, null counts, duplicate-key check; store in `validation_runs`; audit. |
| GET | `/health` | Health check. |

## Security

- No endpoint executes arbitrary user SQL.
- Schema/table/column identifiers validated with `^[a-zA-Z_][a-zA-Z0-9_]*$`.
- Only `env_schema` values in `ALLOWED_SCHEMAS` (e.g. `dev`, `prod`) are allowed.
- DDL apply: single statement only; CREATE TABLE only; forbidden keywords (e.g. DROP, ALTER, TRUNCATE, COPY, GRANT, REVOKE) rejected.

## Project layout

```
datatools-portfolio/
  main.py           # FastAPI app, routes, DB helpers, audit, validation
  static/
    index.html      # Simple frontend UI (tabs: DDL parse/apply, compare, validate)
  requirements.txt
  .env.example
  .gitignore
  README.md
```

Keep `DATABASE_URL` and secrets only in `.env`; never commit `.env` or hardcode credentials in Python.
