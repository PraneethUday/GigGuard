"""
GigGuard — Chunk 7: FastAPI ML Service
Exposes all 4 trained models as REST endpoints

Endpoints:
  POST /predict/premium          → Model 1: weekly premium (Rs.)
  POST /predict/regional-multiplier → Model 2: city exposure factor
  POST /predict/fraud-score      → Model 3: fraud risk score + disposition
  POST /predict/risk-tier        → Model 4: zone/worker risk tier
  GET  /health                   → health check
  GET  /models/status            → all model load status

Run:
  cd ml_service
  uvicorn main:app --reload --port 8001
  
Test:
  curl http://localhost:8001/health
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

# ── App setup ─────────────────────────────────────────
app = FastAPI(
    title       = "GigGuard ML Service",
    description = "AI/ML endpoints for GigGuard parametric insurance platform",
    version     = "1.0.0",
)

MODELS_DIR = "models"

# ══════════════════════════════════════════════════════
# LOAD ALL MODELS AT STARTUP
# ══════════════════════════════════════════════════════
models = {}
load_status = {}

def try_load(key, path):
    try:
        models[key] = joblib.load(path)
        load_status[key] = "loaded"
        print(f"  ✓ {key}")
    except Exception as e:
        load_status[key] = f"failed: {str(e)}"
        print(f"  ✗ {key}: {e}")

print("\nLoading GigGuard ML models...")
try_load("model1_premium",         f"{MODELS_DIR}/model1_premium.pkl")
try_load("model2_xgb",             f"{MODELS_DIR}/model2_xgb_lossratio.pkl")
try_load("model3_fraud",           f"{MODELS_DIR}/model3_fraud_isolationforest.pkl")
try_load("model4_zones",           f"{MODELS_DIR}/model4_kmeans_zones.pkl")
try_load("model4_workers",         f"{MODELS_DIR}/model4_kmeans_workers.pkl")

# Load scalers (for zone/worker normalization)
try_load("scaler_zones",           f"{MODELS_DIR}/scaler_model4_zones.pkl")
try_load("scaler_workers",         f"{MODELS_DIR}/scaler_model4_workers.pkl")

# Load model metadata
try:
    with open(f"{MODELS_DIR}/model1_metadata.json") as f:
        models["meta1"] = json.load(f)
    with open(f"{MODELS_DIR}/model2_metadata.json") as f:
        models["meta2"] = json.load(f)
    with open(f"{MODELS_DIR}/model3_metadata.json") as f:
        models["meta3"] = json.load(f)
    with open(f"{MODELS_DIR}/model4_metadata.json") as f:
        models["meta4"] = json.load(f)
    print("  ✓ metadata files")
except Exception as e:
    print(f"  ✗ metadata: {e}")

print("Models ready.\n")

# ══════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════

# ── Model 1: Premium ──────────────────────────────────
class PremiumRequest(BaseModel):
    zone_risk_score:           float = Field(..., ge=0, le=1,   description="Zone disruption risk 0-1")
    rain_forecast_prob_7d:     float = Field(..., ge=0, le=1,   description="Rain probability next 7 days")
    aqi_forecast_7d:           float = Field(..., ge=0,         description="AQI forecast for zone")
    weekly_earnings_avg:       float = Field(..., gt=0,         description="Worker 4-week avg earnings (Rs.)")
    regional_exposure_factor:  float = Field(1.0, ge=0.5, le=3, description="City-level exposure multiplier")
    loyalty_weeks_at_purchase: int   = Field(0,   ge=0,         description="Consecutive weeks of payment")
    autopay_enabled:           bool  = Field(False,             description="AutoPay enabled?")
    multi_platform_score:      float = Field(0.2, ge=0, le=1,   description="Multi-platform reliability score")
    n_platforms:               int   = Field(1,   ge=1, le=5,   description="Number of registered platforms")
    public_holiday_flag:       bool  = Field(False,             description="Is this a public holiday week?")

class PremiumResponse(BaseModel):
    predicted_premium_rs:  float
    coverage_tier:         str
    base_premium:          float
    loyalty_discount_pct:  float
    autopay_discount_pct:  float
    max_weekly_payout:     int
    model_version:         str

# ── Model 2: Regional Multiplier ──────────────────────
class RegionalRequest(BaseModel):
    city:                str   = Field(..., description="City name")
    max_rainfall:        float = Field(0.0, description="Max rainfall mm/24h forecast")
    max_aqi:             float = Field(80.0, description="Max AQI forecast")
    max_temp:            float = Field(30.0, description="Max temperature °C")
    curfew_any:          bool  = Field(False)
    flood_any:           bool  = Field(False)
    avg_zone_risk:       float = Field(0.4, ge=0, le=1)
    avg_rain_forecast:   float = Field(0.3, ge=0, le=1)
    avg_aqi_forecast:    float = Field(100.0)
    n_active_policies:   int   = Field(1000)
    is_monsoon:          bool  = Field(False)
    is_winter:           bool  = Field(False)
    week_of_year:        int   = Field(1, ge=1, le=53)

class RegionalResponse(BaseModel):
    city:                    str
    predicted_loss_ratio:    float
    exposure_factor:         float
    premium_multiplier_pct:  float
    risk_level:              str
    interpretation:          str

# ── Model 3: Fraud Score ──────────────────────────────
class FraudRequest(BaseModel):
    gps_in_affected_zone:       bool  = Field(True)
    gps_spoof_detected:         bool  = Field(False)
    delivery_activity_detected: bool  = Field(False)
    duplicate_claim:            bool  = Field(False)
    new_registration_fraud:     bool  = Field(False)
    abnormal_claim_freq:        bool  = Field(False)
    payout_amount:              float = Field(0.0, ge=0)
    premium_paid:               float = Field(50.0, gt=0)
    weekly_earnings_avg:        float = Field(2000.0, gt=0)
    disrupted_hours:            float = Field(4.0, ge=0)
    rainfall_mm:                float = Field(0.0, ge=0)
    aqi_value:                  float = Field(80.0, ge=0)
    n_platforms_verified:       int   = Field(1, ge=1)
    eligibility_passed:         bool  = Field(True)

class FraudResponse(BaseModel):
    fraud_risk_score:    float
    claim_disposition:   str
    fraud_signals_found: List[str]
    rule_score:          float
    iso_score:           float
    auto_action:         str
    explanation:         str

# ── Model 4: Risk Tier ────────────────────────────────
class ZoneRiskRequest(BaseModel):
    zone_risk_score:              float = Field(..., ge=0, le=1)
    waterlogging_risk_score:      float = Field(..., ge=0, le=1)
    hist_claim_rate:              float = Field(..., ge=0)
    hist_disruption_freq_per_year:float = Field(..., ge=0)
    avg_aqi_baseline:             float = Field(..., ge=0)
    drainage_score:               float = Field(..., ge=0, le=1)
    curfew_history_count:         int   = Field(0, ge=0)
    platform_activity_density:    float = Field(50.0, ge=0)

class WorkerRiskRequest(BaseModel):
    zone_risk_score:         float = Field(..., ge=0, le=1)
    waterlogging_risk_score: float = Field(..., ge=0, le=1)
    weekly_earnings_avg:     float = Field(..., gt=0)
    hours_per_day:           float = Field(..., ge=0)
    active_days_per_week:    int   = Field(..., ge=1, le=7)
    n_platforms:             int   = Field(1, ge=1, le=5)
    loyalty_weeks:           int   = Field(0, ge=0)
    autopay_enabled:         bool  = Field(False)
    multi_platform_score:    float = Field(0.2, ge=0, le=1)
    hist_claim_count:        int   = Field(0, ge=0)
    weeks_active:            int   = Field(4, ge=0)

class RiskTierResponse(BaseModel):
    risk_tier:         str
    cluster_id:        int
    profile_type:      str
    confidence:        str
    recommended_action: str

# ══════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════

def compute_derived_premium_features(req: PremiumRequest):
    """Compute derived features matching training pipeline."""
    log_earnings             = float(np.log1p(req.weekly_earnings_avg))
    risk_weather_interaction = float(req.zone_risk_score * req.rain_forecast_prob_7d)
    aqi_normalized           = float(min(1.0, req.aqi_forecast_7d / 500))
    loyalty_weeks            = req.loyalty_weeks_at_purchase
    loyalty_tier             = 0 if loyalty_weeks == 0 else 1 if loyalty_weeks <= 4 else 2 if loyalty_weeks <= 12 else 3

    return [
        req.zone_risk_score,
        req.rain_forecast_prob_7d,
        req.aqi_forecast_7d,
        req.weekly_earnings_avg,
        req.regional_exposure_factor,
        req.loyalty_weeks_at_purchase,
        int(req.autopay_enabled),
        req.multi_platform_score,
        req.n_platforms,
        int(req.public_holiday_flag),
        log_earnings,
        risk_weather_interaction,
        aqi_normalized,
        loyalty_tier,
    ]

CITY_ENCODING = {
    "Ahmedabad": 0, "Bengaluru": 1, "Chennai": 2, "Delhi": 3,
    "Hyderabad": 4, "Kolkata": 5, "Mumbai": 6, "Pune": 7
}

TIER_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Extreme"}

RULE_WEIGHT = 0.60
ISO_WEIGHT  = 0.40

def apply_fraud_rules(req: FraudRequest):
    score   = 0.0
    signals = []
    if not req.gps_in_affected_zone:
        score += 0.35; signals.append("GPS_VIOLATION")
    if req.gps_spoof_detected:
        score += 0.40; signals.append("GPS_SPOOF")
    if req.delivery_activity_detected:
        score += 0.50; signals.append("ACTIVE_DURING_CLAIM")
    if req.duplicate_claim:
        score += 0.45; signals.append("DUPLICATE_CLAIM")
    if req.new_registration_fraud:
        score += 0.40; signals.append("NEW_REG_FRAUD")
    n_signals = len(signals)
    if n_signals >= 2:
        score += 0.30; signals.append("MULTI_SIGNAL")
    return min(1.0, score), signals

def classify_disposition(score: float) -> tuple:
    if score > 0.90:
        return "Auto_Rejected", "Claim automatically rejected — high fraud confidence"
    elif score > 0.75:
        return "Flagged_Manual_Review", "Claim flagged for manual review by operations team"
    else:
        return "Auto_Approved", "Claim approved — payout will be processed within 10 minutes"

# ══════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════

@app.get("/health")
def health_check():
    loaded = sum(1 for v in load_status.values() if v == "loaded")
    return {
        "status":        "healthy" if loaded >= 4 else "degraded",
        "models_loaded": loaded,
        "total_models":  len(load_status),
        "service":       "GigGuard ML Service v1.0.0",
    }

@app.get("/models/status")
def model_status():
    return {"load_status": load_status}

# ── POST /predict/premium ──────────────────────────────
@app.post("/predict/premium", response_model=PremiumResponse)
def predict_premium(req: PremiumRequest):
    if "model1_premium" not in models:
        raise HTTPException(status_code=503, detail="Premium model not loaded")

    features = compute_derived_premium_features(req)
    X        = np.array([features])

    predicted_premium = float(models["model1_premium"].predict(X)[0])
    predicted_premium = round(max(15.0, min(200.0, predicted_premium)), 2)

    # Determine coverage tier
    earnings = req.weekly_earnings_avg
    if earnings < 1500:
        tier = "Basic Shield";   max_payout = 500
    elif earnings < 3000:
        tier = "Standard Guard"; max_payout = 1200
    else:
        tier = "Pro Protect";    max_payout = 2500

    base_premium        = round(earnings * 0.035, 2)
    loyalty_discount    = round(min(0.15, req.loyalty_weeks_at_purchase * 0.01) * 100, 1)
    autopay_discount    = 5.0 if req.autopay_enabled else 0.0

    return PremiumResponse(
        predicted_premium_rs = predicted_premium,
        coverage_tier        = tier,
        base_premium         = base_premium,
        loyalty_discount_pct = loyalty_discount,
        autopay_discount_pct = autopay_discount,
        max_weekly_payout    = max_payout,
        model_version        = "1.0.0",
    )

# ── POST /predict/regional-multiplier ─────────────────
@app.post("/predict/regional-multiplier", response_model=RegionalResponse)
def predict_regional(req: RegionalRequest):
    if "model2_xgb" not in models:
        raise HTTPException(status_code=503, detail="Regional model not loaded")

    city_enc     = CITY_ENCODING.get(req.city, 0)
    exposure_score = min(1.0, (
        (req.max_rainfall / 200) * 0.4 +
        min(1.0, req.max_aqi / 500) * 0.3 +
        (0.15 if req.curfew_any else 0) +
        (0.15 if req.flood_any else 0)
    ))

    features = [
        req.max_rainfall, req.max_aqi, req.max_temp,
        int(req.curfew_any), int(req.flood_any),
        req.avg_zone_risk, req.avg_rain_forecast, req.avg_aqi_forecast,
        req.n_active_policies,
        int(req.is_monsoon), int(req.is_winter),
        exposure_score, req.week_of_year, city_enc
    ]
    X = np.array([features])

    pred_loss_ratio    = float(models["model2_xgb"].predict(X)[0].clip(0, 5))
    target_margin      = 0.25
    exposure_factor    = round(min(1.80, max(0.90, pred_loss_ratio + target_margin)), 4)
    multiplier_pct     = round((exposure_factor - 1.0) * 100, 1)

    if pred_loss_ratio > 1.0:
        risk_level = "Extreme"
        interp     = f"Expected payouts exceed premiums collected. Premium surcharge of {multiplier_pct:+.1f}% applied."
    elif pred_loss_ratio > 0.6:
        risk_level = "High"
        interp     = f"High claim volume expected. Premium adjustment of {multiplier_pct:+.1f}% applied."
    elif pred_loss_ratio > 0.3:
        risk_level = "Medium"
        interp     = f"Moderate risk week. Premium adjustment of {multiplier_pct:+.1f}% applied."
    else:
        risk_level = "Low"
        interp     = f"Low disruption risk. Premium discount of {multiplier_pct:.1f}% applied."

    return RegionalResponse(
        city                   = req.city,
        predicted_loss_ratio   = round(pred_loss_ratio, 4),
        exposure_factor        = exposure_factor,
        premium_multiplier_pct = multiplier_pct,
        risk_level             = risk_level,
        interpretation         = interp,
    )

# ── POST /predict/fraud-score ──────────────────────────
@app.post("/predict/fraud-score", response_model=FraudResponse)
def predict_fraud(req: FraudRequest):
    if "model3_fraud" not in models:
        raise HTTPException(status_code=503, detail="Fraud model not loaded")

    # Rule-based score
    rule_score, signals = apply_fraud_rules(req)

    # Build feature vector (matching training)
    gps_violation       = int(not req.gps_in_affected_zone)
    n_fraud_signals     = len([s for s in signals if s != "MULTI_SIGNAL"])
    payout_prem_ratio   = min(20.0, req.payout_amount / max(0.01, req.premium_paid))
    earnings_norm       = min(1.0, req.weekly_earnings_avg / 8000)
    disruption_severity = min(1.0, req.disrupted_hours / 24)
    rainfall_norm       = min(1.0, req.rainfall_mm / 200)
    aqi_norm            = min(1.0, req.aqi_value / 500)
    platforms_norm      = min(1.0, req.n_platforms_verified / 5)

    # Estimate fraud_risk_score (rule-based estimate for feature)
    frs_estimate = min(0.99, rule_score + 0.05 * n_fraud_signals)

    features = [
        gps_violation,
        int(req.gps_spoof_detected),
        int(req.delivery_activity_detected),
        int(req.duplicate_claim),
        int(req.new_registration_fraud),
        int(req.abnormal_claim_freq),
        float(n_fraud_signals),
        payout_prem_ratio,
        earnings_norm,
        disruption_severity,
        rainfall_norm,
        aqi_norm,
        platforms_norm,
        frs_estimate,
    ]
    X = np.array([features])

    # Isolation Forest score
    try:
        iso_raw   = models["model3_fraud"].decision_function(X)[0]
        # Normalize to 0-1 (higher = more anomalous = more fraudulent)
        iso_score = float(np.clip(1 - (iso_raw + 0.5), 0, 1))
    except:
        iso_score = 0.5

    # Combined score
    combined_score = float(RULE_WEIGHT * rule_score + ISO_WEIGHT * iso_score)
    combined_score = round(min(0.99, combined_score), 4)

    disposition, explanation = classify_disposition(combined_score)

    return FraudResponse(
        fraud_risk_score    = combined_score,
        claim_disposition   = disposition,
        fraud_signals_found = signals,
        rule_score          = round(rule_score, 4),
        iso_score           = round(iso_score, 4),
        auto_action         = disposition,
        explanation         = explanation,
    )

# ── POST /predict/risk-tier ────────────────────────────
@app.post("/predict/risk-tier/zone", response_model=RiskTierResponse)
def predict_zone_tier(req: ZoneRiskRequest):
    if "model4_zones" not in models:
        raise HTTPException(status_code=503, detail="Zone profiling model not loaded")

    features = [
        req.zone_risk_score, req.waterlogging_risk_score,
        req.hist_claim_rate, req.hist_disruption_freq_per_year,
        req.avg_aqi_baseline, req.drainage_score,
        req.curfew_history_count, req.platform_activity_density,
    ]
    X = np.array([features])

    # Scale if scaler available
    if "scaler_zones" in models:
        X = models["scaler_zones"].transform(X)

    cluster_id = int(models["model4_zones"].predict(X)[0])

    # Map cluster to tier using zone_risk_score heuristic
    if req.zone_risk_score >= 0.75:
        tier = "Extreme"
    elif req.zone_risk_score >= 0.50:
        tier = "High"
    elif req.zone_risk_score >= 0.25:
        tier = "Medium"
    else:
        tier = "Low"

    actions = {
        "Extreme": "Apply maximum premium multiplier (1.8×). Enable real-time monitoring.",
        "High":    "Apply elevated premium multiplier (1.3–1.5×). Weekly risk review.",
        "Medium":  "Standard premium pricing. Monthly risk review.",
        "Low":     "Apply loyalty discount eligible. Minimal monitoring needed.",
    }

    return RiskTierResponse(
        risk_tier          = tier,
        cluster_id         = cluster_id,
        profile_type       = "zone",
        confidence         = "high" if req.zone_risk_score > 0.6 or req.zone_risk_score < 0.3 else "medium",
        recommended_action = actions[tier],
    )

@app.post("/predict/risk-tier/worker", response_model=RiskTierResponse)
def predict_worker_segment(req: WorkerRiskRequest):
    if "model4_workers" not in models:
        raise HTTPException(status_code=503, detail="Worker profiling model not loaded")

    weekly_active_hours = req.hours_per_day * req.active_days_per_week
    earnings_per_hour   = req.weekly_earnings_avg / max(1, weekly_active_hours)
    log_earnings        = float(np.log1p(req.weekly_earnings_avg))
    claim_rate          = min(1.0, req.hist_claim_count / max(1, req.weeks_active))

    features = [
        req.zone_risk_score, req.waterlogging_risk_score,
        req.weekly_earnings_avg, req.hours_per_day,
        req.active_days_per_week, req.n_platforms, req.loyalty_weeks,
        int(req.autopay_enabled), req.multi_platform_score,
        req.hist_claim_count, req.weeks_active,
        log_earnings, claim_rate, weekly_active_hours, earnings_per_hour,
    ]
    X = np.array([features])

    if "scaler_workers" in models:
        X = models["scaler_workers"].transform(X)

    cluster_id = int(models["model4_workers"].predict(X)[0])

    # Segment by earnings
    if req.weekly_earnings_avg < 1500:
        segment = "Basic Shield"
        action  = "Assign Basic Shield tier. Weekly premium Rs.20-40."
    elif req.weekly_earnings_avg < 3000:
        segment = "Standard Guard"
        action  = "Assign Standard Guard tier. Weekly premium Rs.40-80."
    elif req.n_platforms >= 3:
        segment = "Super Active"
        action  = "Multi-platform Pro tier. Weekly premium Rs.80-130. Priority support."
    else:
        segment = "Pro Protect"
        action  = "Assign Pro Protect tier. Weekly premium Rs.80-130."

    return RiskTierResponse(
        risk_tier          = segment,
        cluster_id         = cluster_id,
        profile_type       = "worker",
        confidence         = "high",
        recommended_action = action,
    )

# ══════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)