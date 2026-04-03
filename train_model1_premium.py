"""
GigGuard — Chunk 3: Train Model 1
XGBoost Premium Calculator

What this does:
  - Loads features_model1_premium.csv
  - Trains XGBoost regressor to predict weekly premium (Rs.)
  - Tunes hyperparameters with cross-validation
  - Evaluates on validation + test set
  - Generates SHAP feature importance
  - Saves model to models/model1_premium.pkl

Run:
  python train_model1_premium.py
"""

import numpy as np
import pandas as pd
import joblib
import os
import json
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

# ── Try importing SHAP (optional but recommended) ─────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠  SHAP not installed. Run: pip install shap")
    print("   Continuing without SHAP plots...\n")

# ── Try importing matplotlib ───────────────────────────
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend (works on all systems)
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    print("⚠  matplotlib not installed. Skipping plots.")

# ══════════════════════════════════════════════════════
# PATHS — adjust if your folder layout is different
# ══════════════════════════════════════════════════════
FEATURES_PATH = "gigguard_features/features_model1_premium.csv"
MODEL_OUT_DIR = "models"
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════
print("=" * 55)
print("  GigGuard — Model 1: XGBoost Premium Calculator")
print("=" * 55)

print("\n[1/6] Loading feature data...")
df = pd.read_csv(FEATURES_PATH)
print(f"  Total rows : {len(df):,}")
print(f"  Columns    : {list(df.columns)}")

# Split back into train / val / test
train_df = df[df["split"] == "train"].copy()
val_df   = df[df["split"] == "val"].copy()
test_df  = df[df["split"] == "test"].copy()

print(f"  Train rows : {len(train_df):,}")
print(f"  Val rows   : {len(val_df):,}")
print(f"  Test rows  : {len(test_df):,}")

# ── Define feature columns (drop non-feature cols) ────
DROP_COLS = ["split", "final_weekly_premium"]
FEATURE_COLS = [c for c in df.columns if c not in DROP_COLS]

print(f"\n  Features used ({len(FEATURE_COLS)}):")
for f in FEATURE_COLS:
    print(f"    • {f}")

X_train = train_df[FEATURE_COLS].values
y_train = train_df["final_weekly_premium"].values

X_val   = val_df[FEATURE_COLS].values
y_val   = val_df["final_weekly_premium"].values

X_test  = test_df[FEATURE_COLS].values
y_test  = test_df["final_weekly_premium"].values

print(f"\n  Target range : Rs.{y_train.min():.0f} – Rs.{y_train.max():.0f}")
print(f"  Target mean  : Rs.{y_train.mean():.1f}")
print(f"  Target std   : Rs.{y_train.std():.1f}")

# ══════════════════════════════════════════════════════
# STEP 2 — BASELINE (mean predictor)
# ══════════════════════════════════════════════════════
print("\n[2/6] Computing baseline (mean predictor)...")
baseline_pred = np.full_like(y_val, y_train.mean())
baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_pred))
baseline_mae  = mean_absolute_error(y_val, baseline_pred)
print(f"  Baseline RMSE : Rs.{baseline_rmse:.2f}")
print(f"  Baseline MAE  : Rs.{baseline_mae:.2f}")
print(f"  (Our model must beat these numbers)")

# ══════════════════════════════════════════════════════
# STEP 3 — TRAIN XGBOOST
# ══════════════════════════════════════════════════════
print("\n[3/6] Training XGBoost Regressor...")

# These hyperparameters are tuned for tabular insurance data
# n_estimators   : 500 trees — enough depth without overfitting
# max_depth      : 6 — standard for tabular data
# learning_rate  : 0.05 — slow learner, more accurate
# subsample      : 0.8 — row sampling prevents overfitting
# colsample      : 0.8 — feature sampling
# reg_alpha/lambda: L1+L2 regularization for insurance generalizability
model = XGBRegressor(
    n_estimators      = 500,
    max_depth         = 6,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    min_child_weight  = 5,
    gamma             = 0.1,
    objective         = "reg:squarederror",
    eval_metric       = "rmse",
    early_stopping_rounds = 30,
    random_state      = 42,
    n_jobs            = -1,
    verbosity         = 0,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

best_iteration = model.best_iteration
print(f"  Best iteration : {best_iteration}")
print(f"  Trees used     : {best_iteration + 1} of {model.n_estimators}")

# ══════════════════════════════════════════════════════
# STEP 4 — EVALUATE
# ══════════════════════════════════════════════════════
print("\n[4/6] Evaluating model performance...")

def evaluate(model, X, y, split_name):
    preds = model.predict(X)
    rmse  = np.sqrt(mean_squared_error(y, preds))
    mae   = mean_absolute_error(y, preds)
    r2    = r2_score(y, preds)
    # Mean Absolute Percentage Error
    mape  = np.mean(np.abs((y - preds) / np.where(y==0, 1, y))) * 100
    print(f"\n  [{split_name}]")
    print(f"    RMSE  : Rs.{rmse:.2f}   (baseline: Rs.{baseline_rmse:.2f})")
    print(f"    MAE   : Rs.{mae:.2f}   (baseline: Rs.{baseline_mae:.2f})")
    print(f"    R²    : {r2:.4f}  (1.0 = perfect)")
    print(f"    MAPE  : {mape:.2f}%")
    improvement = (baseline_rmse - rmse) / baseline_rmse * 100
    print(f"    RMSE improvement over baseline: {improvement:.1f}%")
    return {"rmse": round(rmse,4), "mae": round(mae,4), "r2": round(r2,4), "mape": round(mape,4)}

val_metrics  = evaluate(model, X_val,  y_val,  "VALIDATION")
test_metrics = evaluate(model, X_test, y_test, "TEST (final)")

# Cross-validation on training data
print("\n  5-Fold Cross-Validation (on training set)...")
cv_scores = cross_val_score(
    XGBRegressor(
        n_estimators=best_iteration+1,
        max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0
    ),
    X_train, y_train,
    scoring="neg_root_mean_squared_error",
    cv=5, n_jobs=-1
)
cv_rmse = -cv_scores
print(f"  CV RMSE: {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")

# ── Sample predictions vs actual ─────────────────────
print("\n  Sample predictions (Test set — first 10):")
test_preds = model.predict(X_test)
print(f"  {'Actual (Rs.)':>14}  {'Predicted (Rs.)':>16}  {'Error':>8}")
print(f"  {'-'*42}")
for actual, pred in zip(y_test[:10], test_preds[:10]):
    err = pred - actual
    print(f"  {actual:>14.1f}  {pred:>16.1f}  {err:>+8.1f}")

# ── Tier-wise performance ─────────────────────────────
print("\n  Tier-wise MAE (Test set):")
test_df_eval = test_df.copy()
test_df_eval["predicted"] = test_preds
for tier, (lo, hi) in [
    ("Basic Shield",   (15,  60)),
    ("Standard Guard", (60, 120)),
    ("Pro Protect",    (120,201)),
]:
    mask = (test_df_eval["final_weekly_premium"] >= lo) & (test_df_eval["final_weekly_premium"] < hi)
    sub  = test_df_eval[mask]
    if len(sub) > 0:
        tier_mae = mean_absolute_error(sub["final_weekly_premium"], sub["predicted"])
        print(f"    {tier:<18}: MAE=Rs.{tier_mae:.2f}  n={len(sub):,}")

# ══════════════════════════════════════════════════════
# STEP 5 — FEATURE IMPORTANCE + SHAP
# ══════════════════════════════════════════════════════
print("\n[5/6] Feature importance...")

importance_df = pd.DataFrame({
    "feature":    FEATURE_COLS,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\n  XGBoost Feature Importances:")
for _, row in importance_df.iterrows():
    bar = "█" * int(row["importance"] * 200)
    print(f"    {row['feature']:<35} {row['importance']:.4f}  {bar}")

# Save importance
importance_df.to_csv(f"{MODEL_OUT_DIR}/model1_feature_importance.csv", index=False)

# SHAP values
if SHAP_AVAILABLE and MPL_AVAILABLE:
    print("\n  Computing SHAP values (sample of 500 rows)...")
    sample_idx = np.random.choice(len(X_test), min(500, len(X_test)), replace=False)
    X_shap = X_test[sample_idx]
    
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    
    # Summary plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_shap,
                      feature_names=FEATURE_COLS,
                      show=False, plot_type="bar")
    plt.title("GigGuard Model 1 — SHAP Feature Importance", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(f"{MODEL_OUT_DIR}/model1_shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP plot saved → models/model1_shap_importance.png")

# Predicted vs Actual plot
if MPL_AVAILABLE:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter: actual vs predicted
    axes[0].scatter(y_test[:1000], test_preds[:1000], alpha=0.3, s=10, color="#2563EB")
    axes[0].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()], "r--", linewidth=2, label="Perfect prediction")
    axes[0].set_xlabel("Actual Premium (Rs.)", fontsize=12)
    axes[0].set_ylabel("Predicted Premium (Rs.)", fontsize=12)
    axes[0].set_title("Actual vs Predicted Premium", fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residual distribution
    residuals = test_preds - y_test
    axes[1].hist(residuals, bins=60, color="#10B981", alpha=0.7, edgecolor="white")
    axes[1].axvline(0, color="red", linewidth=2, linestyle="--")
    axes[1].set_xlabel("Prediction Error (Rs.)", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title(f"Residual Distribution  (MAE=Rs.{test_metrics['mae']:.1f})", fontsize=13)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle("GigGuard — Model 1: XGBoost Premium Calculator", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{MODEL_OUT_DIR}/model1_evaluation_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Evaluation plot saved → models/model1_evaluation_plots.png")

# ══════════════════════════════════════════════════════
# STEP 6 — SAVE MODEL
# ══════════════════════════════════════════════════════
print("\n[6/6] Saving model...")

# Save model
MODEL_PATH = f"{MODEL_OUT_DIR}/model1_premium.pkl"
joblib.dump(model, MODEL_PATH)
model_size = os.path.getsize(MODEL_PATH) / 1024
print(f"  Model saved   → {MODEL_PATH}  ({model_size:.1f} KB)")

# Save feature column list (needed by FastAPI later)
METADATA_PATH = f"{MODEL_OUT_DIR}/model1_metadata.json"
model_metadata = {
    "model_name":       "GigGuard Premium Calculator",
    "model_type":       "XGBoostRegressor",
    "version":          "1.0.0",
    "feature_columns":  FEATURE_COLS,
    "target":           "final_weekly_premium",
    "best_iteration":   int(best_iteration),
    "val_metrics":      val_metrics,
    "test_metrics":     test_metrics,
    "cv_rmse_mean":     round(float(cv_rmse.mean()), 4),
    "cv_rmse_std":      round(float(cv_rmse.std()), 4),
    "baseline_rmse":    round(float(baseline_rmse), 4),
    "baseline_mae":     round(float(baseline_mae), 4),
    "input_example": {
        "zone_risk_score":           0.42,
        "rain_forecast_prob_7d":     0.35,
        "aqi_forecast_7d":           120.0,
        "weekly_earnings_avg":       2000.0,
        "regional_exposure_factor":  1.15,
        "loyalty_weeks_at_purchase": 6,
        "autopay_enabled":           1,
        "multi_platform_score":      0.4,
        "n_platforms":               2,
        "public_holiday_flag":       0,
        "log_earnings":              7.601,
        "risk_weather_interaction":  0.147,
        "aqi_normalized":            0.24,
        "loyalty_tier":              2,
    }
}
with open(METADATA_PATH, "w") as f:
    json.dump(model_metadata, f, indent=2)
print(f"  Metadata saved → {METADATA_PATH}")

# ── Quick inference test ──────────────────────────────
print("\n  Quick inference test (Kiran from the README):")
import numpy as np
kiran = np.array([[
    0.42,   # zone_risk_score
    0.08,   # rain_forecast_prob_7d (mild rain)
    120.0,  # aqi_forecast_7d
    2000.0, # weekly_earnings_avg
    1.05,   # regional_exposure_factor
    5,      # loyalty_weeks_at_purchase
    1,      # autopay_enabled
    0.4,    # multi_platform_score
    1,      # n_platforms
    0,      # public_holiday_flag
    7.601,  # log_earnings
    0.034,  # risk_weather_interaction
    0.24,   # aqi_normalized
    2,      # loyalty_tier
]])
kiran_premium = model.predict(kiran)[0]
print(f"  Kiran's predicted premium : Rs.{kiran_premium:.2f}/week")
print(f"  README expected           : ~Rs.72/week")

# ── Final summary ─────────────────────────────────────
print("\n" + "=" * 55)
print("  MODEL 1 TRAINING COMPLETE")
print("=" * 55)
print(f"  RMSE   : Rs.{test_metrics['rmse']:.2f}   (lower is better)")
print(f"  MAE    : Rs.{test_metrics['mae']:.2f}   (avg error per prediction)")
print(f"  R²     : {test_metrics['r2']:.4f}   (1.0 = perfect)")
print(f"  MAPE   : {test_metrics['mape']:.2f}%   (% error)")
beat = test_metrics['rmse'] < baseline_rmse
print(f"  Beats baseline? : {'✅ YES' if beat else '❌ NO'}")
print(f"\n  Files saved:")
print(f"    models/model1_premium.pkl")
print(f"    models/model1_metadata.json")
print(f"    models/model1_feature_importance.csv")
if MPL_AVAILABLE:
    print(f"    models/model1_evaluation_plots.png")
if SHAP_AVAILABLE:
    print(f"    models/model1_shap_importance.png")
print("=" * 55)
print("\n  ✅ Ready for Chunk 4 → Train Model 2 (Regional Profit Protection)")