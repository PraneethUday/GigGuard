-- ============================================================
-- Worker Protection Insurance Platoform (WPIP) Payments & Claims Schema
-- Run in Supabase SQL Editor after supabase-schema.sql
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Premium Payments ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS worker_payments (
    id                  UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    worker_id           UUID            NOT NULL,
    amount              NUMERIC(10,2)   NOT NULL,
    tier                TEXT            NOT NULL CHECK (tier IN ('basic', 'standard', 'pro')),
    payment_method      TEXT            NOT NULL DEFAULT 'upi'
                        CHECK (payment_method IN ('upi', 'card', 'netbanking')),
    razorpay_payment_id TEXT            NOT NULL,
    week_start          DATE            NOT NULL,
    week_end            DATE            NOT NULL,
    status              TEXT            NOT NULL DEFAULT 'success'
                        CHECK (status IN ('success', 'failed', 'pending')),
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_wp_worker_id  ON worker_payments (worker_id);
CREATE INDEX IF NOT EXISTS idx_wp_week_start ON worker_payments (week_start);

ALTER TABLE worker_payments ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Service role full access" ON worker_payments
    FOR ALL USING (true) WITH CHECK (true);

-- ── Insurance Claims ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS insurance_claims (
    id                  UUID            PRIMARY KEY DEFAULT uuid_generate_v4(),
    worker_id           UUID            NOT NULL,
    worker_name         TEXT            NOT NULL,
    claim_type          TEXT            NOT NULL
                        CHECK (claim_type IN ('accident', 'income_loss', 'weather_disruption', 'other')),
    description         TEXT            NOT NULL,
    incident_date       DATE            NOT NULL,
    claim_amount        NUMERIC(10,2)   NOT NULL,
    approved_amount     NUMERIC(10,2),
    status              TEXT            NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'under_review', 'approved', 'rejected', 'paid')),
    admin_notes         TEXT,
    razorpay_payout_id  TEXT,
    upi                 TEXT,
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ic_worker_id ON insurance_claims (worker_id);
CREATE INDEX IF NOT EXISTS idx_ic_status    ON insurance_claims (status);

ALTER TABLE insurance_claims ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Service role full access" ON insurance_claims
    FOR ALL USING (true) WITH CHECK (true);
