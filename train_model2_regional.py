"""
GigGuard — Chunk 4: Train Model 2
Regional Profit Protection (Prophet + XGBoost)

What this does:
  - Trains Facebook Prophet per city to forecast weekly claim payouts
  - Trains XGBoost to predict loss_ratio (payout / premium collected)
  - Computes regional_exposure_factor per city for next week
  - Evaluates both models
  - Saves models/model2_prophet_{city}.pkl + models/model2_xgb.pkl

Run:
  python3 train_model2_regional.py
"""

import numpy as np
import pandas as pd
import joblib, json, os, warnings
warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    try:
        from fbprophet import Prophet
        PROPHET_AVAILABLE = True
    except ImportError:
        PROPHET_AVAILABLE = False
        print("⚠  Prophet not installed. Run: pip install prophet")
        print("   Will train XGBoost only.\n")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

FEATURES_DIR = "gigguard_features"
MODEL_OUT    = "models"
os.makedirs(MODEL_OUT, exist_ok=True)

print("=" * 58)
print("  GigGuard — Model 2: Regional Profit Protection")
print("=" * 58)

# ══════════════════════════════════════════════════════
# PART A — PROPHET: Per-city payout forecasting
# ══════════════════════════════════════════════════════
print("\n── PART A: Prophet Payout Forecaster ──")

prophet_df = pd.read_csv(f"{FEATURES_DIR}/features_model2_regional.csv")
prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

CITIES     = prophet_df["city"].unique()
REGRESSORS = ["max_rainfall","max_aqi","n_active_policies",
              "is_monsoon","is_winter","exposure_score","avg_zone_risk"]

prophet_results = {}

if PROPHET_AVAILABLE:
    print(f"  Training Prophet for {len(CITIES)} cities...")

    for city in sorted(CITIES):
        city_df = prophet_df[prophet_df["city"] == city].sort_values("ds").copy()
        city_df = city_df.fillna(0)

        # Train/test split — last 12 weeks as test (holdout)
        split_point = len(city_df) - 12
        train_c = city_df.iloc[:split_point].copy()
        test_c  = city_df.iloc[split_point:].copy()

        # Build Prophet model with regressors
        m = Prophet(
            yearly_seasonality  = True,
            weekly_seasonality  = False,
            daily_seasonality   = False,
            seasonality_mode    = "multiplicative",  # good for insurance (spikes in monsoon)
            changepoint_prior_scale = 0.05,          # conservative — insurance is stable
            seasonality_prior_scale = 10.0,
            interval_width      = 0.90,
        )

        for reg in REGRESSORS:
            m.add_regressor(reg)

        m.fit(train_c[["ds","y"] + REGRESSORS])

        # Predict on test
        forecast = m.predict(test_c[["ds"] + REGRESSORS])
        y_true   = test_c["y"].values
        y_pred   = forecast["yhat"].clip(lower=0).values

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)

        # Save city model
        city_model_path = f"{MODEL_OUT}/model2_prophet_{city.lower()}.pkl"
        joblib.dump(m, city_model_path)

        prophet_results[city] = {
            "rmse": round(rmse, 2),
            "mae":  round(mae, 2),
            "test_weeks": len(test_c),
            "avg_actual_payout": round(float(y_true.mean()), 2),
        }
        print(f"  {city:<12} RMSE=Rs.{rmse:>8.1f}  MAE=Rs.{mae:>8.1f}  "
              f"avg_payout=Rs.{y_true.mean():>8.0f}")

    print(f"\n  Prophet models saved for all {len(CITIES)} cities ✓")

    # Prophet forecast plot for Chennai (example)
    if MPL_AVAILABLE:
        city_df_c = prophet_df[prophet_df["city"]=="Chennai"].sort_values("ds").fillna(0)
        m_c = joblib.load(f"{MODEL_OUT}/model2_prophet_chennai.pkl")
        future   = m_c.make_future_dataframe(periods=12, freq="W")
        for reg in REGRESSORS:
            future[reg] = city_df_c[reg].median()
        forecast_c = m_c.predict(future)

        fig = m_c.plot(forecast_c, figsize=(12,5))
        plt.title("Chennai — Weekly Claim Payout Forecast (Prophet)", fontsize=13)
        plt.xlabel("Date"); plt.ylabel("Total Payout (Rs.)")
        plt.tight_layout()
        plt.savefig(f"{MODEL_OUT}/model2_prophet_chennai_forecast.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Forecast plot saved → models/model2_prophet_chennai_forecast.png")

else:
    print("  ⚠  Skipping Prophet (not installed)")
    prophet_results = {}

# ══════════════════════════════════════════════════════
# PART B — XGBOOST: Loss Ratio + Regional Multiplier
# ══════════════════════════════════════════════════════
print("\n── PART B: XGBoost Loss Ratio Predictor ──")

xgb_df = pd.read_csv(f"{FEATURES_DIR}/features_model2_xgb.csv").fillna(0)

FEATURE_COLS_M2 = [
    "max_rainfall","max_aqi","max_temp","curfew_any","flood_any",
    "avg_zone_risk","avg_rain_forecast","avg_aqi_forecast",
    "n_active_policies","is_monsoon","is_winter","exposure_score","week_of_year",
]
TARGET_M2 = "loss_ratio"

# Encode city as ordinal
city_map = {c: i for i, c in enumerate(sorted(xgb_df["city"].unique()))}
xgb_df["city_encoded"] = xgb_df["city"].map(city_map)
FEATURE_COLS_M2 = FEATURE_COLS_M2 + ["city_encoded"]

X2 = xgb_df[FEATURE_COLS_M2].values
y2 = xgb_df[TARGET_M2].values

# Stratified split by high/low loss ratio
high_loss = y2 > y2.mean()
X2_tr, X2_te, y2_tr, y2_te = train_test_split(
    X2, y2, test_size=0.20, random_state=42
)
X2_tr, X2_val, y2_tr, y2_val = train_test_split(
    X2_tr, y2_tr, test_size=0.15, random_state=42
)

print(f"  Train={len(X2_tr):,}  Val={len(X2_val):,}  Test={len(X2_te):,}")

# Baseline
baseline_loss = np.sqrt(mean_squared_error(y2_val, np.full_like(y2_val, y2_tr.mean())))
print(f"  Baseline RMSE: {baseline_loss:.4f}")

xgb2 = XGBRegressor(
    n_estimators          = 400,
    max_depth             = 5,
    learning_rate         = 0.04,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    reg_alpha             = 0.5,
    reg_lambda            = 1.5,
    min_child_weight      = 3,
    objective             = "reg:squarederror",
    early_stopping_rounds = 25,
    random_state          = 42,
    n_jobs                = -1,
    verbosity             = 0,
)

xgb2.fit(
    X2_tr, y2_tr,
    eval_set    = [(X2_val, y2_val)],
    verbose     = False,
)

# Evaluate
def eval_regression(model, X, y, name):
    p     = model.predict(X).clip(0, 5)
    rmse  = np.sqrt(mean_squared_error(y, p))
    mae   = mean_absolute_error(y, p)
    r2    = r2_score(y, p)
    print(f"  [{name}] RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    return {"rmse": round(rmse,4), "mae": round(mae,4), "r2": round(r2,4)}

val_m2  = eval_regression(xgb2, X2_val, y2_val, "VALIDATION")
test_m2 = eval_regression(xgb2, X2_te,  y2_te,  "TEST")

# Feature importance
imp2 = pd.DataFrame({
    "feature":    FEATURE_COLS_M2,
    "importance": xgb2.feature_importances_,
}).sort_values("importance", ascending=False)

print("\n  XGBoost Feature Importances (loss ratio):")
for _, row in imp2.iterrows():
    bar = "█" * int(row["importance"] * 150)
    print(f"    {row['feature']:<30} {row['importance']:.4f}  {bar}")

# Save XGBoost model
xgb2_path = f"{MODEL_OUT}/model2_xgb_lossratio.pkl"
joblib.dump(xgb2, xgb2_path)
imp2.to_csv(f"{MODEL_OUT}/model2_feature_importance.csv", index=False)
print(f"\n  XGBoost model saved → {xgb2_path}")

# ══════════════════════════════════════════════════════
# PART C — REGIONAL EXPOSURE FACTOR COMPUTATION
# ══════════════════════════════════════════════════════
print("\n── PART C: Regional Exposure Factor per City ──")
print("  (This is the multiplier applied to premiums next week)\n")

# Simulate 'next week' conditions per city
NEXT_WEEK_CONDITIONS = {
    "Chennai":    {"max_rainfall":70, "max_aqi":98,  "max_temp":33, "curfew_any":0, "flood_any":1,
                   "avg_zone_risk":0.42, "avg_rain_forecast":0.75, "avg_aqi_forecast":100,
                   "n_active_policies":8500, "is_monsoon":1, "is_winter":0,
                   "exposure_score":0.72, "week_of_year":32},
    "Delhi":      {"max_rainfall":5,  "max_aqi":310, "max_temp":28, "curfew_any":0, "flood_any":0,
                   "avg_zone_risk":0.55, "avg_rain_forecast":0.10, "avg_aqi_forecast":305,
                   "n_active_policies":12000,"is_monsoon":0,"is_winter":1,
                   "exposure_score":0.60, "week_of_year":48},
    "Mumbai":     {"max_rainfall":80, "max_aqi":130, "max_temp":30, "curfew_any":0, "flood_any":1,
                   "avg_zone_risk":0.50, "avg_rain_forecast":0.80, "avg_aqi_forecast":128,
                   "n_active_policies":11000,"is_monsoon":1,"is_winter":0,
                   "exposure_score":0.78, "week_of_year":32},
    "Bengaluru":  {"max_rainfall":40, "max_aqi":88,  "max_temp":26, "curfew_any":0, "flood_any":0,
                   "avg_zone_risk":0.38, "avg_rain_forecast":0.40, "avg_aqi_forecast":90,
                   "n_active_policies":9500, "is_monsoon":1,"is_winter":0,
                   "exposure_score":0.35, "week_of_year":32},
    "Hyderabad":  {"max_rainfall":30, "max_aqi":112, "max_temp":32, "curfew_any":0, "flood_any":0,
                   "avg_zone_risk":0.40, "avg_rain_forecast":0.30, "avg_aqi_forecast":110,
                   "n_active_policies":7000, "is_monsoon":1,"is_winter":0,
                   "exposure_score":0.28, "week_of_year":32},
    "Pune":       {"max_rainfall":20, "max_aqi":92,  "max_temp":27, "curfew_any":0, "flood_any":0,
                   "avg_zone_risk":0.32, "avg_rain_forecast":0.22, "avg_aqi_forecast":93,
                   "n_active_policies":5500, "is_monsoon":1,"is_winter":0,
                   "exposure_score":0.20, "week_of_year":32},
    "Kolkata":    {"max_rainfall":55, "max_aqi":178, "max_temp":31, "curfew_any":0, "flood_any":1,
                   "avg_zone_risk":0.48, "avg_rain_forecast":0.55, "avg_aqi_forecast":175,
                   "n_active_policies":7500, "is_monsoon":1,"is_winter":0,
                   "exposure_score":0.62, "week_of_year":32},
    "Ahmedabad":  {"max_rainfall":10, "max_aqi":145, "max_temp":35, "curfew_any":0, "flood_any":0,
                   "avg_zone_risk":0.33, "avg_rain_forecast":0.12, "avg_aqi_forecast":142,
                   "n_active_policies":4500, "is_monsoon":0,"is_winter":0,
                   "exposure_score":0.15, "week_of_year":32},
}

exposure_results = []
print(f"  {'City':<12} {'Loss Ratio':>12} {'Exposure Factor':>16} {'Premium Multiplier':>20}")
print(f"  {'-'*62}")

for city, conditions in NEXT_WEEK_CONDITIONS.items():
    c_enc = city_map.get(city, 0)
    row   = [conditions[f] for f in FEATURE_COLS_M2[:-1]] + [c_enc]
    X_row = np.array([row])

    pred_loss_ratio = float(xgb2.predict(X_row)[0].clip(0, 5))

    # Regional exposure factor formula:
    # Base = 1.0, scale up if predicted loss > 0.5 (company starts losing money)
    # Cap at 1.80 to avoid pricing workers out
    target_margin    = 0.25   # company wants 25% profit margin
    required_premium_coverage = pred_loss_ratio + target_margin
    exposure_factor  = round(min(1.80, max(0.90, required_premium_coverage / 1.0)), 4)

    # What this means for premium
    # A worker paying Rs.70 base premium gets multiplied by this factor
    example_base   = 70
    adjusted_prem  = round(example_base * exposure_factor, 1)

    exposure_results.append({
        "city":                city,
        "predicted_loss_ratio": round(pred_loss_ratio, 4),
        "exposure_factor":     exposure_factor,
        "example_premium_adj": adjusted_prem,
    })
    print(f"  {city:<12} {pred_loss_ratio:>12.4f} {exposure_factor:>16.4f} "
          f"  Rs.{example_base} → Rs.{adjusted_prem:>6.1f}")

exposure_df = pd.DataFrame(exposure_results)
exposure_df.to_csv(f"{MODEL_OUT}/model2_regional_exposure_factors.csv", index=False)
print(f"\n  Exposure factors saved → models/model2_regional_exposure_factors.csv")

# Exposure factor bar plot
if MPL_AVAILABLE:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#EF4444" if f > 1.3 else "#F59E0B" if f > 1.1 else "#10B981"
              for f in exposure_df["exposure_factor"]]
    axes[0].bar(exposure_df["city"], exposure_df["exposure_factor"], color=colors, edgecolor="white")
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Base (1.0)")
    axes[0].set_title("Regional Exposure Factor by City", fontsize=13)
    axes[0].set_ylabel("Exposure Factor")
    axes[0].set_ylim(0.7, 2.0)
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(exposure_df["city"], exposure_df["predicted_loss_ratio"], color="#6366F1", edgecolor="white")
    axes[1].axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="Break-even (0.5)")
    axes[1].set_title("Predicted Loss Ratio by City (Next Week)", fontsize=13)
    axes[1].set_ylabel("Loss Ratio (payout / premium)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("GigGuard — Model 2: Regional Profit Protection", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{MODEL_OUT}/model2_regional_exposure_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Exposure plot saved → models/model2_regional_exposure_plot.png")

# ── Save metadata ─────────────────────────────────────
meta2 = {
    "model_name":       "GigGuard Regional Profit Protection",
    "version":          "1.0.0",
    "xgb_features":     FEATURE_COLS_M2,
    "prophet_regressors": REGRESSORS,
    "target":           TARGET_M2,
    "city_encoding":    city_map,
    "xgb_val_metrics":  val_m2,
    "xgb_test_metrics": test_m2,
    "prophet_results":  prophet_results,
    "exposure_factors": exposure_results,
    "target_profit_margin": 0.25,
    "exposure_factor_cap":  1.80,
}
with open(f"{MODEL_OUT}/model2_metadata.json","w") as f:
    json.dump(meta2, f, indent=2)

# ── Final summary ─────────────────────────────────────
print("\n" + "=" * 58)
print("  MODEL 2 TRAINING COMPLETE")
print("=" * 58)
print(f"  XGBoost Loss Ratio  → RMSE={test_m2['rmse']:.4f}  R²={test_m2['r2']:.4f}")
if PROPHET_AVAILABLE and prophet_results:
    avg_prophet_rmse = np.mean([v["rmse"] for v in prophet_results.values()])
    print(f"  Prophet Avg RMSE    → Rs.{avg_prophet_rmse:.1f} per city per week")
print(f"\n  High risk cities this week:")
for r in sorted(exposure_results, key=lambda x: x["exposure_factor"], reverse=True)[:3]:
    print(f"    {r['city']:<12}: factor={r['exposure_factor']}  "
          f"Rs.70 → Rs.{r['example_premium_adj']}")
print(f"\n  Files saved:")
print(f"    models/model2_xgb_lossratio.pkl")
if PROPHET_AVAILABLE:
    print(f"    models/model2_prophet_{{city}}.pkl  (×{len(CITIES)})")
print(f"    models/model2_regional_exposure_factors.csv")
print(f"    models/model2_metadata.json")
print("=" * 58)
print("\n  ✅ Ready for Chunk 5 → Train Model 3 (Fraud Detection)")