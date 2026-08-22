"""
Fraud scoring helpers for GigGuard claims.

This module exposes compatibility functions used by the trigger pipeline:
- build_fraud_features(...)
- compute_fraud_score(...)

Primary scoring path calls the external ML service.
Fallback scoring path uses deterministic rule-based logic.
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import requests

logger = logging.getLogger(__name__)

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8001")
MODEL_DIR = Path(__file__).parent / "saved_models"
FRAUD_MODEL_PATH = MODEL_DIR / "fraud_model.joblib"


def get_fraud_score(claim_data: dict) -> dict:
    """Call ML service for fraud scoring; fallback to rules if unavailable."""
    try:
        response = requests.post(
            f"{ML_SERVICE_URL}/predict/fraud-score",
            json=claim_data,
            timeout=5,
        )
        response.raise_for_status()
        result = response.json()

        return {
            "fraud_risk_score": float(result.get("fraud_risk_score", 0.0)),
            "disposition": result.get("claim_disposition", "Auto_Approved"),
            "signals": result.get("fraud_signals_found", []),
            "auto_action": result.get("auto_action", "Auto_Approved"),
            "explanation": result.get("explanation", "ML service response"),
        }
    except Exception as exc:
        logger.warning("ML service unavailable, using fallback fraud rules: %s", exc)
        return _rule_based_fallback(claim_data)


def _rule_based_fallback(claim_data: dict) -> dict:
    """Simple fallback if ML service is unreachable."""
    score = 0.0
    signals: list[str] = []

    if claim_data.get("delivery_activity_detected", False):
        score += 0.50
        signals.append("ACTIVE_DURING_CLAIM")
    if claim_data.get("duplicate_claim", False):
        score += 0.45
        signals.append("DUPLICATE_CLAIM")
    if claim_data.get("new_registration_fraud", False):
        score += 0.25
        signals.append("NEW_REGISTRATION_RISK")
    if claim_data.get("abnormal_claim_freq", False):
        score += 0.25
        signals.append("ABNORMAL_CLAIM_FREQUENCY")
    if claim_data.get("cross_platform_flag", 0) > 0:
        score += 0.35
        signals.append("CROSS_PLATFORM_ACTIVITY")
    if claim_data.get("curfew_zone_mismatch", 0) > 0:
        score += 0.50
        signals.append("CURFEW_ZONE_MISMATCH")

    payout_ratio = claim_data.get("payout_amount", 0) / max(
        claim_data.get("daily_wage", 1), 1
    )
    if payout_ratio > 1.2:
        score += 0.15
        signals.append("HIGH_PAYOUT_RATIO")

    claim_count = int(claim_data.get("claim_count_30d", 0) or 0)
    if claim_count > 5:
        score += 0.25
        signals.append("HIGH_CLAIM_FREQUENCY")
    elif claim_count > 3:
        score += 0.10
        signals.append("ELEVATED_CLAIM_FREQUENCY")

    if (claim_data.get("total_payout_30d", 0) or 0) > 3000:
        score += 0.10
        signals.append("HIGH_CUMULATIVE_PAYOUT")

    score = round(min(0.99, score), 4)

    if score >= 0.90:
        disposition = "Auto_Rejected"
    elif score >= 0.60:
        disposition = "Flagged_Manual_Review"
    else:
        disposition = "Auto_Approved"

    return {
        "fraud_risk_score": score,
        "disposition": disposition,
        "signals": signals,
        "auto_action": disposition,
        "explanation": "Rule-based fallback (ML service offline)",
    }


# ---------------------------------------------------------------------------
# Compatibility shims required by triggers.py
# ---------------------------------------------------------------------------

def build_fraud_features(
    worker_id: str,
    city: str,
    payout_amount: float,
    daily_wage: float,
    cross_platform_clear: bool,
    supabase_client=None,
    trigger_type: str = "",
    registration_date: str | None = None,
) -> dict:
    """Build feature dict consumed by compute_fraud_score()."""
    claim_count_30d = 0
    total_payout_30d = 0.0

    if supabase_client:
        try:
            # Only count claims since the worker's registration date.
            # This prevents a re-registered delivery_id from inheriting the
            # historical claim count of whoever previously held that ID.
            thirty_days_ago = (date.today() - timedelta(days=30)).isoformat()
            if registration_date:
                # Use whichever is more recent: 30 days ago or registration date
                reg_date = registration_date[:10]  # keep YYYY-MM-DD
                cutoff = reg_date if reg_date > thirty_days_ago else thirty_days_ago
            else:
                cutoff = thirty_days_ago

            resp = (
                supabase_client.table("claims")
                .select("payout_amount")
                .eq("worker_id", worker_id)
                .gte("created_at", cutoff)
                .execute()
            )
            claim_count_30d = len(resp.data)
            total_payout_30d = sum(r.get("payout_amount", 0) for r in resp.data)
        except Exception:
            pass

    if daily_wage >= 50.0:
        payout_ratio = payout_amount / daily_wage
    else:
        payout_ratio = 0.0

    cross_platform_flag = 0.0 if cross_platform_clear else 1.0

    curfew_zone_mismatch = 0.0
    if trigger_type == "curfew" and supabase_client:
        curfew_zone_mismatch = _check_curfew_zone_mismatch(
            worker_id, city, supabase_client
        )

    return {
        "claim_count_30d": int(claim_count_30d),
        "total_payout_30d": round(float(total_payout_30d), 2),
        "payout_amount": round(float(payout_amount), 2),
        "payout_ratio": round(float(payout_ratio), 4),
        "daily_wage": round(float(daily_wage), 2),
        "cross_platform_flag": float(cross_platform_flag),
        "curfew_zone_mismatch": float(curfew_zone_mismatch),
    }


FRAUD_FEATURE_COLS = [
    "claim_count_30d",
    "total_payout_30d",
    "payout_amount",
    "payout_ratio",
    "daily_wage",
    "cross_platform_flag",
    "curfew_zone_mismatch",
]


def _check_curfew_zone_mismatch(
    worker_id: str, city: str, supabase_client
) -> float:
    """Return curfew mismatch signal.

    GPS-based mismatch checks are currently disabled in this backend path,
    so this returns 0.0 (no mismatch penalty).
    """
    return 0.0
def compute_fraud_score(
    features: dict, trigger_type: str = ""
) -> tuple[float, list[str]]:
    """Score a claim and return `(fraud_score, fraud_flags)`.

    Primary path: call external ML service.
    Fallback path: local rule-based scoring.
    """
    claim_data = {
        "delivery_activity_detected": features.get("cross_platform_flag", 0) > 0,
        "duplicate_claim": (features.get("claim_count_30d", 0) or 0) > 5,
        "new_registration_fraud": False,
        "abnormal_claim_freq": (features.get("claim_count_30d", 0) or 0) > 3,
        "payout_amount": features.get("payout_amount", 0.0) or 0.0,
        "daily_wage": features.get("daily_wage", 0.0) or 0.0,
        "claim_count_30d": features.get("claim_count_30d", 0) or 0,
        "total_payout_30d": features.get("total_payout_30d", 0.0) or 0.0,
        "cross_platform_flag": features.get("cross_platform_flag", 0.0) or 0.0,
        "curfew_zone_mismatch": features.get("curfew_zone_mismatch", 0.0) or 0.0,
    }

    try:
        ml_result = get_fraud_score(claim_data)
        score = float(np.clip(float(ml_result.get("fraud_risk_score", 0.0)), 0.0, 1.0))
        raw_signals = ml_result.get("signals", [])
        if isinstance(raw_signals, list):
            flags = [str(s).lower() for s in raw_signals]
        else:
            flags = []
        return round(score, 4), flags
    except Exception as exc:
        logger.warning("Fraud scoring failed, using fallback rules: %s", exc)
        return _rule_based_score(features, trigger_type)


def _rule_based_score(
    features: dict, trigger_type: str = ""
) -> tuple[float, list[str]]:
    """Fallback rule-based fraud scoring when ML service/model is unavailable."""
    score = 0.0
    flags: list[str] = []

    if features.get("cross_platform_flag", 0) > 0:
        score += 0.35
        flags.append("cross_platform_activity")

    claim_count = int(features.get("claim_count_30d", 0) or 0)
    if claim_count > 5:
        score += 0.25
        flags.append("high_claim_frequency")
    elif claim_count > 3:
        score += 0.10
        flags.append("elevated_claim_frequency")

    if (features.get("payout_ratio", 0) or 0) > 0.45:
        score += 0.15
        flags.append("high_payout_ratio")

    if (features.get("total_payout_30d", 0) or 0) > 3000:
        score += 0.10
        flags.append("high_cumulative_payout")

    # Curfew mismatch is a strong fraud signal when available.
    if trigger_type == "curfew" and (features.get("curfew_zone_mismatch", 0) or 0) > 0:
        score += 0.50
        flags.append("curfew_zone_mismatch")

    return min(score, 1.0), flags


def train_fraud_model(supabase_client) -> dict:
    """Train a local Isolation Forest model on historical claims."""
    from sklearn.ensemble import IsolationForest

    try:
        resp = (
            supabase_client.table("claims")
            .select(
                "worker_id, payout_amount, daily_wage_est, cross_platform_clear, fraud_score, created_at"
            )
            .order("created_at", desc=True)
            .limit(1000)
            .execute()
        )
        claims = resp.data
    except Exception as exc:
        return {"error": f"Failed to fetch claims: {exc}"}

    if len(claims) < 20:
        return {"error": f"Need at least 20 claims to train, found {len(claims)}"}

    rows = []
    for c in claims:
        payout = c.get("payout_amount", 0) or 0
        wage = c.get("daily_wage_est", 0) or 1
        cross_clear = c.get("cross_platform_clear", True)
        rows.append(
            {
                "claim_count_30d": 1,
                "total_payout_30d": payout,
                "payout_amount": payout,
                "payout_ratio": payout / max(wage, 1),
                "daily_wage": wage,
                "cross_platform_flag": 0.0 if cross_clear else 1.0,
                "curfew_zone_mismatch": 0.0,
            }
        )

    X = np.array([[r[col] for col in FRAUD_FEATURE_COLS] for r in rows])

    model = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, FRAUD_MODEL_PATH)

    predictions = model.predict(X)
    n_anomalies = int((predictions == -1).sum())

    result = {
        "status": "ok",
        "n_samples": len(X),
        "n_anomalies": n_anomalies,
        "contamination": 0.1,
    }
    logger.info("[fraud] Trained Isolation Forest: %s", result)
    return result


def _load_fraud_model():
    """Load saved Isolation Forest model or return None."""
    if FRAUD_MODEL_PATH.exists():
        try:
            return joblib.load(FRAUD_MODEL_PATH)
        except Exception:
            return None
    return None
