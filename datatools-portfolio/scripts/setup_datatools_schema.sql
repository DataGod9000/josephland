-- Run this in Supabase SQL Editor (Dashboard → SQL Editor) once per project.
-- Creates the datatools schema and tables required by the DataTools Portfolio API.

CREATE SCHEMA IF NOT EXISTS datatools;

-- Audit log for all write actions (ddl_apply, compare_suggest_keys, compare_run, validate_run)
CREATE TABLE IF NOT EXISTS datatools.audit_log (
  id BIGSERIAL PRIMARY KEY,
  action TEXT NOT NULL,
  env_schema TEXT,
  details JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Registry of applied DDL (env_schema + table_name → ddl, parsed_json)
CREATE TABLE IF NOT EXISTS datatools.table_registry (
  env_schema TEXT NOT NULL,
  table_name TEXT NOT NULL,
  ddl TEXT NOT NULL,
  parsed_json JSONB,
  PRIMARY KEY (env_schema, table_name)
);

-- Compare runs (left_table, right_table, join_keys, result_json)
CREATE TABLE IF NOT EXISTS datatools.compare_runs (
  id BIGSERIAL PRIMARY KEY,
  left_table TEXT NOT NULL,
  right_table TEXT NOT NULL,
  join_keys TEXT[] NOT NULL,
  result_json JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Validation runs (target_table, result_json)
CREATE TABLE IF NOT EXISTS datatools.validation_runs (
  id BIGSERIAL PRIMARY KEY,
  target_table TEXT NOT NULL,
  result_json JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tables scheduled for deletion (renamed to to_be_deleted_*; actual DROP can be run later)
CREATE TABLE IF NOT EXISTS datatools.deletion_schedule (
  id BIGSERIAL PRIMARY KEY,
  env_schema TEXT NOT NULL,
  original_table_name TEXT NOT NULL,
  renamed_table_name TEXT NOT NULL,
  delete_after TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Optional: create dev and prod schemas so "Create table" can target them
CREATE SCHEMA IF NOT EXISTS dev;
CREATE SCHEMA IF NOT EXISTS prod;
