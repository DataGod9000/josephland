"""
datatools-portfolio: FastAPI backend for internal DataTools platform.
SQL-first, Supabase Postgres, audit logging, safe DDL/compare/validate.
"""
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import sqlglot
from sqlglot.expressions import Create, ColumnDef
from dotenv import load_dotenv
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from psycopg import Connection, connect
from openai import OpenAI

# Load .env from the directory containing this file (so it works regardless of cwd)
load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI(
    title="DataTools Portfolio API",
    description="Internal DataTools platform: DDL parse/apply, compare, validate.",
    version="1.0.0",
)


@app.on_event("startup")
def ensure_deletion_schedule_table():
    """Create datatools schema and deletion_schedule table if they do not exist."""
    url = os.getenv("DATABASE_URL")
    if not url:
        return
    try:
        with connect(url) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("CREATE SCHEMA IF NOT EXISTS datatools")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS datatools.deletion_schedule (
                        id BIGSERIAL PRIMARY KEY,
                        env_schema TEXT NOT NULL,
                        original_table_name TEXT NOT NULL,
                        renamed_table_name TEXT NOT NULL,
                        delete_after TIMESTAMPTZ NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
    except Exception:
        pass


DATABASE_URL = os.getenv("DATABASE_URL")
ALLOWED_SCHEMAS_STR = os.getenv("ALLOWED_SCHEMAS", "dev,prod")
ALLOWED_SCHEMAS = [s.strip() for s in ALLOWED_SCHEMAS_STR.split(",") if s.strip()]

IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
DDL_FORBIDDEN = re.compile(
    r"\b(DROP|ALTER|TRUNCATE|COPY|GRANT|REVOKE)\b",
    re.IGNORECASE,
)
# Data governance: table name must be {layer}_josephco_{domain}_{tablename}_{granularity}
# layer=ods|dws|dim|ads|dwd, domain=trade|growth, granularity=di|df|hi|hf (underscores between each part)
TABLE_NAME_GOVERNANCE = re.compile(
    r"^(ods|dws|dim|ads|dwd)_josephco_(trade|growth)_[a-zA-Z0-9_]+_(di|df|hi|hf)$",
    re.IGNORECASE,
)


# ---------- Validation ----------


def validate_identifier(name: str) -> bool:
    """Allow only safe identifiers: letters, digits, underscore."""
    return bool(name and IDENTIFIER_RE.match(name))


def validate_env_schema(env_schema: str) -> None:
    if env_schema not in ALLOWED_SCHEMAS:
        raise HTTPException(
            status_code=400,
            detail=f"env_schema must be one of {ALLOWED_SCHEMAS}, got: {env_schema}",
        )


def validate_table_name(table_name: str) -> None:
    if not validate_identifier(table_name):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid table name: {table_name}. Use only [a-zA-Z_][a-zA-Z0-9_]*",
        )


def validate_table_name_governance(table_name: str) -> None:
    """Enforce naming: {ods|dws|dim|ads|dwd}_josephco_{trade|growth}_{tablename}_{di|df|hi|hf}."""
    if not table_name or not TABLE_NAME_GOVERNANCE.match(table_name):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Table name must follow data governance: "
                f"{{ods|dws|dim|ads|dwd}}_josephco_{{trade|growth}}_{{tablename}}_{{di|df|hi|hf}}. "
                f"Example: ods_josephco_growth_users_di. Got: {table_name!r}"
            ),
        )


# ---------- Database ----------


@contextmanager
def get_conn():
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL not set")
    with connect(DATABASE_URL) as conn:
        conn.autocommit = False
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def audit_log(conn: Connection, action: str, env_schema: Optional[str], details: dict[str, Any]) -> None:
    """Write one row to datatools.audit_log."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO datatools.audit_log (action, env_schema, details)
            VALUES (%s, %s, %s::jsonb)
            """,
            (action, env_schema, json.dumps(details)),
        )


# ---------- DDL parsing (sqlglot) ----------


def parse_create_table(ddl: str) -> dict[str, Any]:
    """
    Parse a single CREATE TABLE statement. Returns dict with schema_in_ddl, table, columns, constraints.
    Raises ValueError if not exactly one CREATE TABLE.
    """
    ddl_stripped = (ddl or "").strip()
    if not ddl_stripped:
        raise ValueError("DDL is empty")

    parsed = sqlglot.parse(ddl_stripped, dialect="postgres")
    if not parsed or len(parsed) == 0:
        raise ValueError("Could not parse DDL")

    if len(parsed) > 1:
        raise ValueError("Only a single statement is allowed")

    stmt = parsed[0]
    if not isinstance(stmt, Create):
        raise ValueError("Only CREATE TABLE is supported")

    table_ref = stmt.this
    if table_ref is None:
        raise ValueError("CREATE TABLE has no table reference")

    # sqlglot may wrap in Schema (e.g. "dev"."users"); then table_ref.name is empty and table is in table_ref.this
    if not (getattr(table_ref, "name", None) or "").strip() and getattr(table_ref, "this", None) is not None:
        table_ref = table_ref.this
    # Table name and schema: Table has .name (identifier) and .db (schema identifier)
    table_name = table_ref.name if hasattr(table_ref, "name") else str(getattr(table_ref, "this", ""))
    db = getattr(table_ref, "db", None)
    schema_in_ddl = db.name if db and hasattr(db, "name") else (db if isinstance(db, str) else "public")

    columns = []
    constraints = []

    # Column definitions: use find_all(ColumnDef) to get every column in the tree
    for col in stmt.find_all(ColumnDef):
        col_this = col.this
        col_name = col_this.name if hasattr(col_this, "name") else str(col_this)
        kind = getattr(col, "kind", None)
        dtype = kind.sql(dialect="postgres") if kind else "TEXT"
        nullable = True
        default_val = None
        for c in col.args.get("constraints") or []:
            cname = type(c).__name__
            if "NotNull" in cname or "PrimaryKey" in cname:
                nullable = False
            if "Default" in cname:
                try:
                    default_val = c.sql(dialect="postgres")
                except Exception:
                    default_val = None
        columns.append({
            "name": col_name,
            "type": dtype,
            "nullable": nullable,
            "default": default_val,
        })

    # Table-level constraints (e.g. PRIMARY KEY (id), UNIQUE (x)) - non-ColumnDef expressions in body
    expr = stmt.expression
    if expr and hasattr(expr, "expressions"):
        for child in expr.expressions:
            if not isinstance(child, ColumnDef):
                constraints.append({"raw": child.sql(dialect="postgres")})

    return {
        "schema_in_ddl": schema_in_ddl,
        "table": table_name,
        "columns": columns,
        "constraints": constraints,
    }


def build_create_table_sql(env_schema: str, table_name: str, parsed: dict[str, Any]) -> str:
    """Build CREATE TABLE schema_name.table_name (...) from parsed columns/constraints."""
    cols = parsed.get("columns") or []
    parts = [f'CREATE TABLE "{env_schema}"."{table_name}" (']
    col_defs = []
    for c in cols:
        name = c.get("name", "")
        typ = c.get("type", "text")
        nullable = c.get("nullable", True)
        default = c.get("default")
        seg = f'"{name}" {typ}'
        if not nullable:
            seg += " NOT NULL"
        if default:
            seg += " " + (default if str(default).upper().strip().startswith("DEFAULT") else f"DEFAULT {default}")
        col_defs.append(seg)
    for con in parsed.get("constraints") or []:
        col_defs.append(con.get("raw", ""))
    parts.append(", ".join(col_defs))
    parts.append(")")
    return "\n".join(parts)


# ---------- Request/Response models ----------


class DdlParseRequest(BaseModel):
    ddl: str = Field(..., description="CREATE TABLE statement")


class ColumnCommentInput(BaseModel):
    column_name: str
    comment_en: str = ""
    comment_zh: str = ""


class DdlApplyRequest(BaseModel):
    ddl: str = Field(..., description="CREATE TABLE statement")
    env_schema: str = Field(..., description="Target schema: dev or prod")
    column_comments: Optional[list[ColumnCommentInput]] = Field(
        default=None,
        description="Per-column comments (EN + ZH required by governance)",
    )


class SuggestColumnCommentsRequest(BaseModel):
    columns: list[dict[str, Any]] = Field(..., description="List of {name, type} for each column")
    table_name: Optional[str] = Field(default=None, description="Optional table name for context")


class CompareSuggestKeysRequest(BaseModel):
    left_table: str
    right_table: str
    env_schema: str
    max_candidates: int = Field(default=5, ge=1, le=20)


class CompareRunRequest(BaseModel):
    left_table: str
    right_table: str
    env_schema: str
    join_keys: list[str]
    compare_columns: Optional[list[str]] = None
    sample_limit: int = Field(default=50, ge=1, le=1000)


class ValidateRunRequest(BaseModel):
    target_table: str
    env_schema: str
    key_columns: Optional[list[str]] = None


class ScheduleDeleteRequest(BaseModel):
    env_schema: str
    table_name: str


class RestoreBackupRequest(BaseModel):
    env_schema: str
    table_name: str  # backup table name, e.g. back_up_users_20260224


class RunQueryRequest(BaseModel):
    sql: str = Field(..., min_length=1, description="Single SELECT statement only")


# ---------- Endpoints ----------


@app.post("/ddl/parse")
def ddl_parse(req: DdlParseRequest):
    """Parse CREATE TABLE DDL and return schema, table, columns, constraints."""
    try:
        result = parse_create_table(req.ddl)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.post("/ddl/suggest-column-comments")
def suggest_column_comments(req: SuggestColumnCommentsRequest):
    """Use AI to generate English and Chinese column comments. Requires OPENAI_API_KEY in .env."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not set. Add it to .env to use AI-generated comments.",
        )
    columns = req.columns or []
    if not columns:
        return {"suggestions": []}
    table_context = f" (table: {req.table_name})" if req.table_name else ""
    col_list = "\n".join(
        f"- {c.get('name', '')} ({c.get('type', '')})" for c in columns if c.get("name")
    )
    prompt = f"""Generate a short column comment in English and in Chinese for each column.
Table context:{table_context}
Columns:
{col_list}

Return a JSON array only, no other text. Each item: {{"column_name": "<name>", "comment_en": "<short English comment>", "comment_zh": "<简短中文注释>"}}
Example: [{{"column_name": "id", "comment_en": "Primary key identifier.", "comment_zh": "主键标识。"}}]"""
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Strip markdown code block if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rstrip().removesuffix("```").strip()
        suggestions = json.loads(text)
        if not isinstance(suggestions, list):
            suggestions = []
        # Ensure we have column_name, comment_en, comment_zh for each
        out = []
        for s in suggestions:
            if isinstance(s, dict) and s.get("column_name"):
                out.append({
                    "column_name": str(s["column_name"]),
                    "comment_en": str(s.get("comment_en", "")),
                    "comment_zh": str(s.get("comment_zh", "")),
                })
        return {"suggestions": out}
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"AI returned invalid JSON: {e!s}") from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI suggestion failed: {e!s}") from e


@app.post("/ddl/apply")
def ddl_apply(req: DdlApplyRequest):
    """Validate env_schema, parse DDL, force schema, execute CREATE TABLE, upsert table_registry, audit."""
    validate_env_schema(req.env_schema)
    if DDL_FORBIDDEN.search(req.ddl):
        raise HTTPException(
            status_code=400,
            detail="DDL must not contain DROP, ALTER, TRUNCATE, COPY, GRANT, REVOKE",
        )
    try:
        parsed = parse_create_table(req.ddl)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    table_name = parsed.get("table") or ""
    if not validate_identifier(table_name):
        raise HTTPException(status_code=400, detail=f"Invalid table name: {table_name}")
    validate_table_name_governance(table_name)

    # Data governance: every column must have both English and Chinese comment
    columns = parsed.get("columns") or []
    comments_by_col = {c.column_name.strip(): c for c in (req.column_comments or []) if c.column_name}
    missing = []
    for col in columns:
        cname = col.get("name", "")
        cc = comments_by_col.get(cname)
        if not cc or not (cc.comment_en and cc.comment_en.strip()) or not (cc.comment_zh and cc.comment_zh.strip()):
            missing.append(cname)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Data governance: each column must have both English and Chinese comments. Missing or empty for: {missing}. Use 'Generate comments with AI' or fill Comment (EN) and Comment (ZH) for every column.",
        )

    applied_sql = build_create_table_sql(req.env_schema, table_name, parsed)

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(applied_sql)
            # COMMENT ON COLUMN for each column (Postgres allows one comment per column; store "EN: ... | ZH: ...")
            for col in columns:
                cname = col.get("name", "")
                cc = comments_by_col.get(cname)
                if cc and cc.comment_en and cc.comment_zh:
                    combined = f"EN: {cc.comment_en.strip()} | ZH: {cc.comment_zh.strip()}"
                    # Escape single quotes for SQL: ' -> ''
                    combined_escaped = combined.replace("'", "''")
                    comment_sql = f'COMMENT ON COLUMN "{req.env_schema}"."{table_name}"."{cname}" IS \'{combined_escaped}\''
                    with conn.cursor() as c2:
                        c2.execute(comment_sql)
            # Upsert table_registry(env_schema, table_name, ddl, parsed_json)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO datatools.table_registry (env_schema, table_name, ddl, parsed_json)
                    VALUES (%s, %s, %s, %s::jsonb)
                    ON CONFLICT (env_schema, table_name)
                    DO UPDATE SET ddl = EXCLUDED.ddl, parsed_json = EXCLUDED.parsed_json
                    """,
                    (req.env_schema, table_name, req.ddl, json.dumps(parsed)),
                )
            audit_log(conn, "ddl_apply", req.env_schema, {
                "env_schema": req.env_schema,
                "table": table_name,
                "applied_sql": applied_sql,
            })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {e!s}. If you see 'relation datatools.audit_log does not exist', run the SQL in scripts/setup_datatools_schema.sql in your Supabase SQL editor.",
        ) from e

    return {"status": "ok", "applied_sql": applied_sql}


@app.post("/compare/suggest-keys")
def compare_suggest_keys(req: CompareSuggestKeysRequest):
    """Find common columns by name+type, score by uniqueness and null ratio, return top N."""
    validate_env_schema(req.env_schema)
    validate_table_name(req.left_table)
    validate_table_name(req.right_table)

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Common columns: same name and data_type in both tables
            cur.execute(
                """
                SELECT a.column_name, a.data_type
                FROM information_schema.columns a
                JOIN information_schema.columns b
                  ON a.column_name = b.column_name AND a.data_type = b.data_type
                WHERE a.table_schema = %s AND a.table_name = %s
                  AND b.table_schema = %s AND b.table_name = %s
                """,
                (req.env_schema, req.left_table, req.env_schema, req.right_table),
            )
            common = cur.fetchall()

        if not common:
            audit_log(conn, "compare_suggest_keys", req.env_schema, {
                "left_table": req.left_table,
                "right_table": req.right_table,
                "candidates": [],
            })
            return {"candidates": []}

        candidates = []
        for (col_name, data_type) in common:
            if not validate_identifier(col_name):
                continue
            with conn.cursor() as cur:
                # Row counts and distinct/null stats (SQL-only, safe from empty tables)
                cur.execute(
                    """
                    SELECT
                      (SELECT COUNT(*) FROM "%s"."%s") AS left_rows,
                      (SELECT COUNT(*) FROM "%s"."%s") AS right_rows,
                      (SELECT COUNT(DISTINCT "%s") FROM "%s"."%s") AS left_distinct,
                      (SELECT COUNT(*) - COUNT("%s") FROM "%s"."%s") AS left_nulls,
                      (SELECT COUNT(DISTINCT "%s") FROM "%s"."%s") AS right_distinct,
                      (SELECT COUNT(*) - COUNT("%s") FROM "%s"."%s") AS right_nulls
                    """ % (
                        req.env_schema, req.left_table,
                        req.env_schema, req.right_table,
                        col_name, req.env_schema, req.left_table,
                        col_name, req.env_schema, req.left_table,
                        col_name, req.env_schema, req.right_table,
                        col_name, req.env_schema, req.right_table,
                    ),
                )
                row = cur.fetchone()
            if not row or row[0] == 0 or row[1] == 0:
                uniq_score = 0.0
                null_penalty = 0.0
            else:
                left_rows, right_rows = int(row[0]), int(row[1])
                left_distinct, left_nulls = int(row[2]), int(row[3])
                right_distinct, right_nulls = int(row[4]), int(row[5])
                uniq_score = min(left_distinct / left_rows, right_distinct / right_rows)
                null_penalty = (left_nulls / left_rows) + (right_nulls / right_rows)
                uniq_score = min(uniq_score, 1.0)
            score = max(0.0, uniq_score - null_penalty)
            candidates.append({
                "column": col_name,
                "data_type": data_type,
                "score": round(score, 4),
            })

        candidates.sort(key=lambda x: -x["score"])
        top = candidates[: req.max_candidates]

        audit_log(conn, "compare_suggest_keys", req.env_schema, {
            "left_table": req.left_table,
            "right_table": req.right_table,
            "candidates": top,
        })

    return {"candidates": top}


@app.post("/compare/run")
def compare_run(req: CompareRunRequest):
    """Row counts, missing_in_left/right, sample of differences; store in compare_runs and audit."""
    validate_env_schema(req.env_schema)
    validate_table_name(req.left_table)
    validate_table_name(req.right_table)
    for k in req.join_keys:
        if not validate_identifier(k):
            raise HTTPException(status_code=400, detail=f"Invalid join key: {k}")

    # Build quoted key list for SQL
    keys_sql = ", ".join(f'"{k}"' for k in req.join_keys)
    join_on = " AND ".join(f'l."{k}" = r."{k}"' for k in req.join_keys)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f'SELECT COUNT(*) FROM "{req.env_schema}"."{req.left_table}"',
            )
            left_count = cur.fetchone()[0]
            cur.execute(
                f'SELECT COUNT(*) FROM "{req.env_schema}"."{req.right_table}"',
            )
            right_count = cur.fetchone()[0]

            # Missing in right: keys in left not in right
            k0 = req.join_keys[0]
            cur.execute(
                f"""
                SELECT COUNT(*) FROM "{req.env_schema}"."{req.left_table}" l
                LEFT JOIN "{req.env_schema}"."{req.right_table}" r ON {join_on}
                WHERE r."{k0}" IS NULL
                """,
            )
            missing_in_right = cur.fetchone()[0]

            cur.execute(
                f"""
                SELECT COUNT(*) FROM "{req.env_schema}"."{req.right_table}" r
                LEFT JOIN "{req.env_schema}"."{req.left_table}" l ON {join_on}
                WHERE l."{k0}" IS NULL
                """,
            )
            missing_in_left = cur.fetchone()[0]

            # Sample of differing keys (FULL OUTER JOIN, limit)
            sample_cols = ", ".join(f'l."{k}" AS left_{k}, r."{k}" AS right_{k}' for k in req.join_keys)
            cur.execute(
                f"""
                SELECT {sample_cols}
                FROM "{req.env_schema}"."{req.left_table}" l
                FULL OUTER JOIN "{req.env_schema}"."{req.right_table}" r ON {join_on}
                WHERE l."{k0}" IS NULL OR r."{k0}" IS NULL
                LIMIT %s
                """,
                (req.sample_limit,),
            )
            rows = cur.fetchall()
            col_names = [d[0] for d in cur.description]
            sample = [dict(zip(col_names, r)) for r in rows]

        result_json = {
            "left_count": left_count,
            "right_count": right_count,
            "missing_in_right": missing_in_right,
            "missing_in_left": missing_in_left,
            "sample": sample,
        }

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO datatools.compare_runs (left_table, right_table, join_keys, result_json)
                VALUES (%s, %s, %s::text[], %s::jsonb)
                """,
                (req.left_table, req.right_table, req.join_keys, json.dumps(result_json)),
            )
        audit_log(conn, "compare_run", req.env_schema, {
            "left_table": req.left_table,
            "right_table": req.right_table,
            "join_keys": req.join_keys,
            "left_count": left_count,
            "right_count": right_count,
        })

    return result_json


@app.post("/validate/run")
def validate_run(req: ValidateRunRequest):
    """Run validation checks: row count, null counts per column, duplicate key groups; store and audit."""
    validate_env_schema(req.env_schema)
    validate_table_name(req.target_table)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
                """,
                (req.env_schema, req.target_table),
            )
            columns = [{"name": r[0], "data_type": r[1]} for r in cur.fetchall()]

        if not columns:
            raise HTTPException(
                status_code=400,
                detail=f"Table {req.env_schema}.{req.target_table} not found or has no columns",
            )

        full_name = f'"{req.env_schema}"."{req.target_table}"'
        checks = []

        # Total row count
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {full_name}")
            total_rows = cur.fetchone()[0]
        checks.append({
            "check_item": "total_row_count",
            "result": "PASS" if total_rows >= 0 else "FAIL",
            "issue_count": 0 if total_rows >= 0 else 1,
        })

        # Null count per column
        for col in columns:
            cname = col["name"]
            if not validate_identifier(cname):
                continue
            with conn.cursor() as cur:
                cur.execute(
                    f'SELECT COUNT(*) - COUNT("{cname}") FROM {full_name}',
                )
                null_count = cur.fetchone()[0]
            checks.append({
                "check_item": f"null_count.{cname}",
                "result": "PASS" if null_count == 0 else "FAIL",
                "issue_count": null_count,
            })

        # Duplicate key groups if key_columns provided
        if req.key_columns:
            for k in req.key_columns:
                if not validate_identifier(k):
                    continue
            keys_sql = ", ".join(f'"{k}"' for k in req.key_columns)
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT COUNT(*) - COUNT(*) AS dup_groups
                    FROM (
                      SELECT {keys_sql}, COUNT(*) AS c
                      FROM {full_name}
                      GROUP BY {keys_sql}
                      HAVING COUNT(*) > 1
                    ) x
                    """,
                )
                # Actually we want count of duplicate groups
                cur.execute(
                    f"""
                    SELECT COUNT(*) FROM (
                      SELECT {keys_sql}
                      FROM {full_name}
                      GROUP BY {keys_sql}
                      HAVING COUNT(*) > 1
                    ) x
                    """,
                )
                dup_groups = cur.fetchone()[0]
            checks.append({
                "check_item": f"duplicate_keys.({','.join(req.key_columns)})",
                "result": "PASS" if dup_groups == 0 else "FAIL",
                "issue_count": dup_groups,
            })

        result_json = {"checks": checks, "total_rows": total_rows}

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO datatools.validation_runs (target_table, result_json)
                VALUES (%s, %s::jsonb)
                """,
                (req.target_table, json.dumps(result_json)),
            )
        audit_log(conn, "validate_run", req.env_schema, {
            "target_table": req.target_table,
            "checks_count": len(checks),
        })

    return result_json


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- Assets ----------


@app.get("/assets/tables")
def assets_list_tables(
    env_schema: Optional[str] = None,
    filter_type: Optional[str] = Query(None, alias="filter"),
    q: Optional[str] = None,
):
    """List tables. filter: tables (registry), backups (back_up_%), to_be_deleted (to_be_deleted_%). env_schema: dev|prod. q: search substring."""
    filter_type = (filter_type or "tables").lower()
    if filter_type not in ("tables", "backups", "to_be_deleted"):
        filter_type = "tables"
    schemas = [env_schema] if env_schema and env_schema in ALLOWED_SCHEMAS else list(ALLOWED_SCHEMAS)
    search = (q or "").strip().lower()
    out = []
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                if filter_type == "tables":
                    placeholders = ",".join("%s" for _ in schemas)
                    cur.execute(
                        f"""
                        SELECT env_schema, table_name FROM datatools.table_registry
                        WHERE env_schema IN ({placeholders})
                          AND table_name NOT LIKE 'back_up_%%'
                          AND table_name NOT LIKE 'to_be_deleted_%%'
                        ORDER BY env_schema, table_name
                        """,
                        schemas,
                    )
                    rows = cur.fetchall()
                    for r in rows:
                        out.append({"env_schema": r[0], "table_name": r[1], "type": "table"})
                elif filter_type == "backups":
                    for sch in schemas:
                        cur.execute(
                            """
                            SELECT table_schema, table_name FROM information_schema.tables
                            WHERE table_schema = %s AND table_name LIKE 'back_up_%%'
                            ORDER BY table_name
                            """,
                            (sch,),
                        )
                        for r in cur.fetchall():
                            out.append({"env_schema": r[0], "table_name": r[1], "type": "backup"})
                else:
                    placeholders = ",".join("%s" for _ in schemas)
                    cur.execute(
                        f"""
                        SELECT env_schema, renamed_table_name, delete_after
                        FROM datatools.deletion_schedule
                        WHERE env_schema IN ({placeholders})
                        ORDER BY env_schema, renamed_table_name
                        """,
                        schemas,
                    )
                    for r in cur.fetchall():
                        out.append({
                            "env_schema": r[0],
                            "table_name": r[1],
                            "type": "to_be_deleted",
                            "delete_after": r[2].isoformat() if hasattr(r[2], "isoformat") else str(r[2]),
                        })
        if search:
            out = [t for t in out if search in (t["table_name"] or "").lower() or search in (t["env_schema"] or "").lower()]
        return {"tables": out}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list tables: {e!s}") from e


@app.get("/assets/table-columns")
def assets_table_columns(
    env_schema: str = Query(..., description="Schema (dev/prod)"),
    table_name: str = Query(..., description="Table name"),
):
    """Return column names for a table (from information_schema) for generating SELECT."""
    validate_env_schema(env_schema)
    validate_table_name(table_name)
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                    """,
                    (env_schema, table_name),
                )
                rows = cur.fetchall()
        columns = [r[0] for r in rows]
        if not columns:
            raise HTTPException(status_code=404, detail=f"Table {env_schema}.{table_name} not found or has no columns")
        return {"columns": columns}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get columns: {e!s}") from e


def _format_size(n: int) -> str:
    """Format bytes as human-readable size."""
    if n is None or n < 0:
        return "—"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} PB"


@app.get("/assets/table-details")
def assets_table_details(
    env_schema: str = Query(..., description="Schema (dev/prod)"),
    table_name: str = Query(..., description="Table name"),
):
    """Return table stats: row count, size, owner, environment, sample rows."""
    validate_env_schema(env_schema)
    validate_table_name(table_name)
    qualified = f'"{env_schema}"."{table_name}"'
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {qualified}")
                row_count = cur.fetchone()[0]
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_total_relation_size(%s::regclass)",
                    (qualified,),
                )
                size_bytes = cur.fetchone()[0] or 0
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT tableowner FROM pg_tables WHERE schemaname = %s AND tablename = %s",
                    (env_schema, table_name),
                )
                row = cur.fetchone()
                owner = row[0] if row else None
            with conn.cursor() as cur:
                cur.execute(f"SELECT * FROM {qualified} LIMIT 10")
                columns = [d[0] for d in cur.description] if cur.description else []
                sample_rows = [list(r) for r in cur.fetchall()]
        return {
            "env_schema": env_schema,
            "table_name": table_name,
            "row_count": row_count,
            "size_bytes": size_bytes,
            "size_human": _format_size(size_bytes),
            "owner": owner,
            "sample_columns": columns,
            "sample_rows": sample_rows,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get table details: {e!s}") from e


@app.get("/assets/table-ddl")
def assets_table_ddl(
    env_schema: str = Query(..., description="Schema (dev/prod)"),
    table_name: str = Query(..., description="Table name"),
):
    """Return CREATE TABLE DDL for the table (from table_registry if available, else built from information_schema). Usable in Create table flow."""
    validate_env_schema(env_schema)
    validate_table_name(table_name)
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT ddl FROM datatools.table_registry WHERE env_schema = %s AND table_name = %s",
                    (env_schema, table_name),
                )
                row = cur.fetchone()
            if row and row[0]:
                return {"ddl": row[0]}
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                    """,
                    (env_schema, table_name),
                )
                cols = cur.fetchall()
        if not cols:
            raise HTTPException(status_code=404, detail=f"Table {env_schema}.{table_name} not found")
        parts = [f'CREATE TABLE "{env_schema}"."{table_name}" (']
        segs = []
        for (cname, dtype, nullable, default) in cols:
            s = f'  "{cname}" {dtype or "TEXT"}'
            if nullable == "NO":
                s += " NOT NULL"
            if default:
                s += " DEFAULT " + str(default)
            segs.append(s)
        parts.append(",\n".join(segs))
        parts.append(")")
        return {"ddl": "\n".join(parts)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get DDL: {e!s}") from e


@app.post("/assets/schedule-delete")
def assets_schedule_delete(req: ScheduleDeleteRequest):
    """Clone table to back_up_<name>_<YYYYMMDD>, then rename to to_be_deleted_<name>, schedule delete in 7 days, remove from table_registry."""
    validate_env_schema(req.env_schema)
    validate_table_name(req.table_name)
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    backup_name = f"back_up_{req.table_name}_{date_str}"
    renamed = f"to_be_deleted_{req.table_name}"
    delete_after = datetime.now(timezone.utc) + timedelta(days=7)
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'CREATE TABLE "{req.env_schema}"."{backup_name}" AS SELECT * FROM "{req.env_schema}"."{req.table_name}"',
                )
            with conn.cursor() as cur:
                cur.execute(
                    f'ALTER TABLE "{req.env_schema}"."{req.table_name}" RENAME TO "{renamed}"',
                )
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO datatools.deletion_schedule (env_schema, original_table_name, renamed_table_name, delete_after)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (req.env_schema, req.table_name, renamed, delete_after),
                )
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM datatools.table_registry WHERE env_schema = %s AND table_name = %s",
                    (req.env_schema, req.table_name),
                )
            audit_log(conn, "schedule_delete", req.env_schema, {
                "env_schema": req.env_schema,
                "table_name": req.table_name,
                "backup_name": backup_name,
                "renamed_to": renamed,
                "delete_after": delete_after.isoformat(),
            })
        return {
            "status": "ok",
            "backup_name": backup_name,
            "renamed_to": renamed,
            "delete_after": delete_after.isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to schedule delete: {e!s}") from e


@app.post("/assets/restore-backup")
def assets_restore_backup(req: RestoreBackupRequest):
    """Rename backup table back to original name; drop to_be_deleted_<name> if present; remove from deletion_schedule; add to table_registry."""
    validate_env_schema(req.env_schema)
    # backup name is back_up_{original}_{YYYYMMDD}
    m = re.match(r"^back_up_(.+)_\d{8}$", req.table_name)
    if not m:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid backup table name: {req.table_name}. Expected back_up_<name>_YYYYMMDD",
        )
    original_name = m.group(1)
    to_be_deleted_name = f"to_be_deleted_{original_name}"
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f'DROP TABLE IF EXISTS "{req.env_schema}"."{to_be_deleted_name}"',
                )
            with conn.cursor() as cur:
                cur.execute(
                    f'ALTER TABLE "{req.env_schema}"."{req.table_name}" RENAME TO "{original_name}"',
                )
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM datatools.deletion_schedule WHERE env_schema = %s AND renamed_table_name = %s",
                    (req.env_schema, to_be_deleted_name),
                )
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO datatools.table_registry (env_schema, table_name, ddl, parsed_json)
                    VALUES (%s, %s, %s, %s::jsonb)
                    ON CONFLICT (env_schema, table_name) DO UPDATE SET ddl = EXCLUDED.ddl, parsed_json = EXCLUDED.parsed_json
                    """,
                    (req.env_schema, original_name, "", "{}"),
                )
            audit_log(conn, "restore_backup", req.env_schema, {
                "env_schema": req.env_schema,
                "backup_table": req.table_name,
                "restored_as": original_name,
            })
        return {"status": "ok", "restored_as": original_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restore backup: {e!s}") from e


# ---------- Frontend (static) ----------

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.post("/query/run")
def query_run(req: RunQueryRequest):
    """Execute a single read-only SELECT query. Max 500 rows. Only SELECT is allowed."""
    sql = (req.sql or "").strip()
    if not sql:
        raise HTTPException(status_code=400, detail="SQL is empty")
    # Remove single-line and block comments for validation
    sql_no_comments = re.sub(r"--[^\n]*", "", sql)
    sql_no_comments = re.sub(r"/\*.*?\*/", "", sql_no_comments, flags=re.DOTALL)
    sql_no_comments = sql_no_comments.strip().upper()
    if not sql_no_comments.startswith("SELECT"):
        raise HTTPException(status_code=400, detail="Only SELECT queries are allowed")
    if ";" in sql_no_comments.rstrip(";"):
        idx = sql_no_comments.find(";")
        if idx >= 0 and sql_no_comments[idx + 1 :].strip():
            raise HTTPException(status_code=400, detail="Only a single statement is allowed")
    # Append LIMIT if not present to cap results
    if "LIMIT" not in sql_no_comments:
        sql = sql.rstrip().rstrip(";") + " LIMIT 500"
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                columns = [d[0] for d in cur.description] if cur.description else []
                rows = cur.fetchall()
        return {"columns": columns, "rows": [list(r) for r in rows]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Query failed: {e!s}") from e


@app.get("/")
def index():
    """Serve the DataTools Portfolio UI."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)
