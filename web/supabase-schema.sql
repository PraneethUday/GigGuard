-- ============================================================
-- Worker Protection Insurance Platoform (WPIP) Auth — registered_workers table
-- Run in Supabase SQL Editor (Dashboard → SQL Editor)
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS registered_workers (
    id            UUID             PRIMARY KEY DEFAULT uuid_generate_v4(),
    name          TEXT             NOT NULL,
    age           INTEGER          NOT NULL,
    phone         TEXT             NOT NULL UNIQUE,
    email         TEXT             NOT NULL UNIQUE,
    password_hash TEXT             NOT NULL,
    city          TEXT             NOT NULL,
    area          TEXT             NOT NULL,
    delivery_id   TEXT             NOT NULL,
    platforms     TEXT[]           NOT NULL,
    verification_status TEXT NOT NULL DEFAULT 'pending'
                         CHECK (verification_status IN ('verified', 'pending', 'rejected')),
    pan           TEXT,
    aadhaar       TEXT             UNIQUE,
    upi           TEXT,
    bank          TEXT,
    consent       BOOLEAN          NOT NULL DEFAULT FALSE,
    gps_consent   BOOLEAN          NOT NULL DEFAULT FALSE,
    autopay       BOOLEAN          NOT NULL DEFAULT FALSE,
    tier          TEXT             NOT NULL DEFAULT 'standard'
                  CHECK (tier IN ('basic', 'standard', 'pro')),
    is_active     BOOLEAN          NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rw_email     ON registered_workers (email);
CREATE INDEX IF NOT EXISTS idx_rw_phone     ON registered_workers (phone);
CREATE INDEX IF NOT EXISTS idx_rw_platforms ON registered_workers USING GIN (platforms);

ALTER TABLE registered_workers ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Service role full access"
    ON registered_workers FOR ALL USING (true) WITH CHECK (true);
