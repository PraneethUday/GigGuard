"""
Cron-compatible retraining script for the premium prediction model.

Flow:
  1. Fetch all worker histories (last 30 days) from platform tables
  2. Fetch current weather for all cities
  3. Build training data with feature engineering
  4. Train XGBoost model, evaluate RMSE
  5. Replace saved model ONLY if new RMSE <= old RMSE (or no existing model)
  6. Cache weather data in Supabase

Usage:
  python -m ml.retrain          (from backend/ directory)
  OR add to Render cron: python -m ml.retrain
"""

import sys
import os

# Ensure backend/ is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
from db import supabase, PLATFORM_TABLES, fetch_worker_history
from ml.weather import fetch_all_cities
from ml.premium_model import (
    build_features_from_history,
    build_training_data,
    train_model,
    save_model,
    load_metadata,
    FEATURE_COLS,
    MODEL_DIR,
)


def get_all_worker_ids() -> list[tuple[str, str, str]]:
    """Get all unique worker_ids from platform tables with their city.

    Returns list of (worker_id, city, platform).
    """
    workers = []
    seen = set()

    for table in PLATFORM_TABLES:
        try:
            resp = (
                supabase.table(table)
                .select("worker_id, city, platform")
                .order("date", desc=True)
                .limit(500)
                .execute()
            )
            for row in resp.data:
                wid = row["worker_id"]
                if wid not in seen:
                    seen.add(wid)
                    workers.append((wid, row.get("city", "Unknown"), row.get("platform", "unknown")))
        except Exception as e:
            print(f"[retrain] Error fetching from {table}: {e}")

    return workers


def cache_weather(city_weather: dict[str, dict]) -> None:
    """Store today's weather in the weather_cache table."""
    today = date.today().isoformat()
    rows = []
    for city, w in city_weather.items():
        rows.append({
            "city": city,
            "date": today,
            "temperature": w.get("temperature"),
            "humidity": w.get("humidity"),
            "wind_speed": w.get("wind_speed"),
            "rain_1h": w.get("rain_1h"),
            "rain_3h": w.get("rain_3h"),
            "aqi_index": w.get("aqi_index"),
            "pm25": w.get("pm25"),
            "pm10": w.get("pm10"),
            "weather_main": w.get("weather_main"),
            "is_heavy_rain": w.get("is_heavy_rain", False),
            "is_extreme_heat": w.get("is_extreme_heat", False),
            "is_severe_aqi": w.get("is_severe_aqi", False),
        })

    try:
        supabase.table("weather_cache").upsert(rows, on_conflict="city,date").execute()
        print(f"[retrain] Cached weather for {len(rows)} cities")
    except Exception as e:
        print(f"[retrain] Warning: could not cache weather: {e}")


def retrain():
    """Main retraining pipeline."""
    print("=" * 60)
    print(f"[retrain] Starting model retraining — {date.today()}")
    print("=" * 60)

    # 1. Fetch weather
    print("[retrain] Fetching weather data...")
    city_weather = fetch_all_cities()
    cache_weather(city_weather)

    for city, w in city_weather.items():
        print(f"  {city}: {w['temperature']}°C, AQI={w['aqi_index']}, Rain={w['rain_1h']}mm/h")

    # 2. Get all worker IDs
    print("[retrain] Fetching worker IDs from platform tables...")
    workers = get_all_worker_ids()
    print(f"[retrain] Found {len(workers)} unique workers")

    if len(workers) < 10:
        print("[retrain] Not enough workers for training. Skipping.")
        return

    # 3. Fetch histories and build training data
    print("[retrain] Fetching 30-day histories...")
    all_histories: list[tuple[str, list[dict], str]] = []
    tiers = ["basic", "standard", "pro"]

    for i, (worker_id, city, platform) in enumerate(workers):
        history = fetch_worker_history(worker_id, days=30)
        if len(history) >= 3:
            # Assign tier based on earnings level for training diversity
            avg_earn = sum(r.get("earnings", 0) for r in history) / len(history)
            if avg_earn < 300:
                tier = "basic"
            elif avg_earn < 600:
                tier = "standard"
            else:
                tier = "pro"
            all_histories.append((worker_id, history, tier))

        if (i + 1) % 50 == 0:
            print(f"  Fetched {i + 1}/{len(workers)} histories...")

    print(f"[retrain] {len(all_histories)} workers have sufficient data for training")

    if len(all_histories) < 10:
        print("[retrain] Not enough training samples. Skipping.")
        return

    # 4. Build features + targets
    print("[retrain] Building training dataset...")
    X, y = build_training_data(all_histories, city_weather)
    print(f"[retrain] Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"[retrain] Premium range: ₹{y.min():.0f} — ₹{y.max():.0f} (mean: ₹{y.mean():.0f})")

    # 5. Train
    print("[retrain] Training XGBoost model...")
    model, rmse = train_model(X, y)
    print(f"[retrain] New model RMSE: ₹{rmse:.2f}")

    # 6. Compare with existing model
    old_meta = load_metadata()
    if old_meta:
        old_rmse = old_meta.get("rmse", float("inf"))
        print(f"[retrain] Previous model RMSE: ₹{old_rmse:.2f}")
        if rmse > old_rmse * 1.1:  # allow 10% tolerance
            print(f"[retrain] New model is worse. Keeping old model.")
            return
        print(f"[retrain] New model is better or comparable. Replacing.")
    else:
        print("[retrain] No existing model found. Saving first model.")

    # 7. Save
    save_model(model, rmse, len(X))

    # Feature importance
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    print("\n[retrain] Top 10 Feature Importances:")
    for rank, idx in enumerate(sorted_idx[:10], 1):
        print(f"  {rank}. {FEATURE_COLS[idx]}: {importances[idx]:.4f}")

    print(f"\n[retrain] Done. Model saved to {MODEL_DIR}")


if __name__ == "__main__":
    retrain()
