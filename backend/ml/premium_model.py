"""
XGBoost-based weekly premium prediction model for GigGuard.

Features:
  - Worker activity: rolling 7d avg earnings/deliveries/hours, trends
  - Risk indicators: weather severity, AQI, past claims count
  - City + platform encoding
  - Tier multiplier

Target: weekly_premium (INR) = base_rate * risk_multiplier
  where base_rate = pct_of_avg_weekly_earnings
  and risk_multiplier accounts for weather/AQI/claims history
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "premium_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# Premium config per tier
# Rate is applied to weekly_earnings_est; min/max are the allowed output bounds
TIER_CONFIG = {
    "basic":    {"rate": 0.015, "min": 10,  "max": 60,  "max_payout": 500},
    "standard": {"rate": 0.025, "min": 20,  "max": 120, "max_payout": 1200},
    "pro":      {"rate": 0.035, "min": 40,  "max": 200, "max_payout": 2500},
}

# City risk weights (base — overridden by real weather data during prediction)
CITY_RISK_BASE = {
    "Chennai": 1.15, "Delhi": 1.20, "Mumbai": 1.10,
    "Bangalore": 1.00, "Bengaluru": 1.00,
    "Hyderabad": 0.95, "Pune": 0.90,
    "Kolkata": 1.05, "Ahmedabad": 0.95,
    "Jaipur": 0.90, "Surat": 0.90,
}


def build_features_from_history(history: list[dict], weather: dict | None = None) -> dict:
    """Build ML feature vector from a worker's recent activity history + weather.

    Args:
        history: list of daily activity rows (most recent first), each with
                 keys: earnings, deliveries, active_hours, rating, date, city, platform
        weather: dict from weather.py with temperature, aqi_index, rain_1h, etc.

    Returns:
        flat dict of numeric features ready for model input
    """
    if not history:
        return _empty_features()

    df = pd.DataFrame(history)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", ascending=True)

    # --- Rolling 7-day features ---
    earnings = df["earnings"].values
    deliveries = df["deliveries"].values
    hours = df["active_hours"].values
    ratings = df["rating"].values

    n = len(df)
    last7 = min(n, 7)
    last3 = min(n, 3)

    avg_earnings_7d = float(np.mean(earnings[-last7:]))
    avg_deliveries_7d = float(np.mean(deliveries[-last7:]))
    avg_hours_7d = float(np.mean(hours[-last7:]))
    avg_rating_7d = float(np.mean(ratings[-last7:]))

    # Weekly estimate
    weekly_earnings_est = avg_earnings_7d * 7

    # --- Trends (3d vs 7d) ---
    avg_earnings_3d = float(np.mean(earnings[-last3:])) if last3 > 0 else avg_earnings_7d
    earnings_trend = (avg_earnings_3d - avg_earnings_7d) / max(avg_earnings_7d, 1.0)

    avg_deliveries_3d = float(np.mean(deliveries[-last3:])) if last3 > 0 else avg_deliveries_7d
    delivery_trend = (avg_deliveries_3d - avg_deliveries_7d) / max(avg_deliveries_7d, 1.0)

    # --- Consistency (std dev of earnings) ---
    earnings_std = float(np.std(earnings[-last7:])) if last7 > 1 else 0.0
    earnings_cv = earnings_std / max(avg_earnings_7d, 1.0)

    # --- Total days active ---
    total_days = n

    # --- Earnings per delivery / per hour ---
    earn_per_delivery = avg_earnings_7d / max(avg_deliveries_7d, 1.0)
    earn_per_hour = avg_earnings_7d / max(avg_hours_7d, 0.1)

    # --- Weather features ---
    w = weather or {}
    temperature = w.get("temperature", 30.0)
    aqi_index = w.get("aqi_index", 100)
    rain_1h = w.get("rain_1h", 0.0)
    rain_3h = w.get("rain_3h", 0.0)
    humidity = w.get("humidity", 60)
    wind_speed = w.get("wind_speed", 5.0)
    is_heavy_rain = float(w.get("is_heavy_rain", False))
    is_extreme_heat = float(w.get("is_extreme_heat", False))
    is_severe_aqi = float(w.get("is_severe_aqi", False))

    # Composite risk score (0-1 scale)
    weather_risk = min(1.0, (
        (rain_1h / 50.0) * 0.3 +
        (aqi_index / 500.0) * 0.3 +
        (max(0, temperature - 40) / 15.0) * 0.2 +
        is_heavy_rain * 0.1 +
        is_severe_aqi * 0.1
    ))

    # City
    city = df["city"].iloc[-1] if "city" in df.columns else "Unknown"
    city_risk = CITY_RISK_BASE.get(city, 1.0)

    return {
        "avg_earnings_7d": round(avg_earnings_7d, 2),
        "avg_deliveries_7d": round(avg_deliveries_7d, 2),
        "avg_hours_7d": round(avg_hours_7d, 2),
        "avg_rating_7d": round(avg_rating_7d, 2),
        "weekly_earnings_est": round(weekly_earnings_est, 2),
        "earnings_trend": round(earnings_trend, 4),
        "delivery_trend": round(delivery_trend, 4),
        "earnings_cv": round(earnings_cv, 4),
        "earnings_std": round(earnings_std, 2),
        "total_days_active": total_days,
        "earn_per_delivery": round(earn_per_delivery, 2),
        "earn_per_hour": round(earn_per_hour, 2),
        "temperature": temperature,
        "aqi_index": aqi_index,
        "rain_1h": rain_1h,
        "rain_3h": rain_3h,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "is_heavy_rain": is_heavy_rain,
        "is_extreme_heat": is_extreme_heat,
        "is_severe_aqi": is_severe_aqi,
        "weather_risk": round(weather_risk, 4),
        "city_risk": city_risk,
    }


def _empty_features() -> dict:
    """Return zero-filled features for workers with no history."""
    return {
        "avg_earnings_7d": 0, "avg_deliveries_7d": 0, "avg_hours_7d": 0,
        "avg_rating_7d": 0, "weekly_earnings_est": 0,
        "earnings_trend": 0, "delivery_trend": 0, "earnings_cv": 0,
        "earnings_std": 0, "total_days_active": 0,
        "earn_per_delivery": 0, "earn_per_hour": 0,
        "temperature": 30, "aqi_index": 100, "rain_1h": 0, "rain_3h": 0,
        "humidity": 60, "wind_speed": 5,
        "is_heavy_rain": 0, "is_extreme_heat": 0, "is_severe_aqi": 0,
        "weather_risk": 0, "city_risk": 1.0,
    }


FEATURE_COLS = list(_empty_features().keys())


def compute_target_premium(features: dict, tier: str = "standard") -> float:
    """Compute the 'true' premium label for training data.

    Formula accounts for earnings, consistency, weather risk, city risk,
    and efficiency — producing varied targets across workers.
    The result is NOT clamped during training so the model learns real variance.
    Clamping is only applied at prediction/serving time.
    """
    cfg = TIER_CONFIG.get(tier, TIER_CONFIG["standard"])
    weekly = features["weekly_earnings_est"]

    base = weekly * cfg["rate"]

    # Weather risk: higher risk = higher premium (up to 1.5x)
    risk_mult = 1.0 + features["weather_risk"] * 0.5

    # City multiplier
    city_mult = features["city_risk"]

    # Consistency factor: inconsistent earners pay more (CV > 0.3 is high variance)
    cv = features.get("earnings_cv", 0)
    consistency_mult = 1.0 + min(cv, 0.5) * 0.4  # up to 1.2x for highly variable

    # Efficiency factor: less efficient workers are riskier
    earn_per_hour = features.get("earn_per_hour", 100)
    efficiency_factor = max(0.8, min(1.2, 1.0 - (earn_per_hour - 100) / 500))

    # Activity factor: fewer active days = higher premium (less data = more risk)
    days_active = features.get("total_days_active", 7)
    activity_mult = max(0.9, min(1.3, 1.0 + (7 - min(days_active, 7)) / 14))

    premium = base * risk_mult * city_mult * consistency_mult * efficiency_factor * activity_mult

    return round(premium, 2)


def clamp_premium(premium: float, tier: str = "standard") -> float:
    """Clamp a raw premium to tier bounds. Used only at serving time."""
    cfg = TIER_CONFIG.get(tier, TIER_CONFIG["standard"])
    return round(max(cfg["min"], min(cfg["max"], premium)), 2)


def build_training_data(
    all_worker_histories: list[tuple[str, list[dict], str]],
    city_weather: dict[str, dict] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build training dataset from worker histories.

    Args:
        all_worker_histories: list of (worker_id, history_rows, tier)
        city_weather: {city: weather_dict}

    Returns:
        X (DataFrame of features), y (Series of target premiums)
    """
    rows = []
    targets = []

    for worker_id, history, tier in all_worker_histories:
        if not history or len(history) < 3:
            continue

        city = history[0].get("city", "Unknown") if history else "Unknown"
        weather = (city_weather or {}).get(city)

        features = build_features_from_history(history, weather)
        target = compute_target_premium(features, tier)

        rows.append(features)
        targets.append(target)

    X = pd.DataFrame(rows, columns=FEATURE_COLS)
    y = pd.Series(targets, name="weekly_premium")
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple[XGBRegressor, float]:
    """Train XGBoost regressor and return (model, rmse).

    Uses 80/20 train/test split for evaluation.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    return model, rmse


def save_model(model: XGBRegressor, rmse: float, n_samples: int) -> None:
    """Save model + metadata to disk."""
    joblib.dump(model, MODEL_PATH)
    metadata = {
        "rmse": round(rmse, 4),
        "n_samples": n_samples,
        "trained_at": date.today().isoformat(),
        "features": FEATURE_COLS,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"[model] Saved model — RMSE: {rmse:.4f}, samples: {n_samples}")


def load_model() -> XGBRegressor | None:
    """Load saved model from disk, or None if not found."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def load_metadata() -> dict | None:
    """Load model metadata."""
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())
    return None


def predict_premium(
    model: XGBRegressor,
    features: dict,
    tier: str = "standard",
) -> dict:
    """Predict weekly premium for a single worker.

    Returns dict with predicted premium (clamped to tier bounds at serving time),
    plus breakdown info.
    """
    cfg = TIER_CONFIG.get(tier, TIER_CONFIG["standard"])

    X = pd.DataFrame([features], columns=FEATURE_COLS)
    raw_pred = float(model.predict(X)[0])

    # Clamp to tier bounds only at serving time
    premium = clamp_premium(raw_pred, tier)

    # Discount: autopay 5%
    premium_with_autopay = round(premium * 0.95, 2)

    return {
        "weekly_premium": premium,
        "weekly_premium_autopay": premium_with_autopay,
        "raw_prediction": round(raw_pred, 2),
        "tier": tier,
        "max_payout": cfg["max_payout"],
        "weather_risk": features.get("weather_risk", 0),
        "city_risk": features.get("city_risk", 1.0),
        "weekly_earnings_est": features.get("weekly_earnings_est", 0),
    }
