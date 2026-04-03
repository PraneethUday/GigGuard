"""
GigGuard — Chunk 2: Feature Engineering
Produces 4 clean ML-ready feature matrices:
  features_model1_premium.csv       -> XGBoost Regressor
  features_model2_regional.csv      -> Prophet time-series
  features_model2_xgb.csv           -> XGBoost loss ratio
  features_model3_fraud.csv         -> Isolation Forest
  features_model4_zones.csv         -> K-Means (zones)
  features_model4_workers.csv       -> K-Means (workers)
  scaler_model3.pkl / scaler_model4_*.pkl
  feature_metadata.json
"""

import numpy as np
import pandas as pd
import json, pickle, os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

np.random.seed(42)

INPUT  = "/home/claude/gigguard_data"
OUTPUT = "/home/claude/gigguard_features"
os.makedirs(OUTPUT, exist_ok=True)
metadata = {}

# ── Load raw data ──────────────────────────────────────────────────
print("Loading raw datasets...")
zones    = pd.read_csv(f"{INPUT}/zone_risk.csv")
workers  = pd.read_csv(f"{INPUT}/workers.csv")
events   = pd.read_csv(f"{INPUT}/weather_events.csv")
policies = pd.read_csv(f"{INPUT}/policies.csv")
claims   = pd.read_csv(f"{INPUT}/claims.csv")
print(f"  zones={len(zones)}  workers={len(workers)}  events={len(events)}  policies={len(policies)}  claims={len(claims)}")

# ══════════════════════════════════════════════════════════════════
# MODEL 1 — PREMIUM CALCULATOR
# ══════════════════════════════════════════════════════════════════
print("\n── MODEL 1: Premium Calculator ──")

df1 = policies[policies["policy_status"] == "Active"].copy()
df1["autopay_enabled"]     = df1["autopay_enabled"].astype(int)
df1["public_holiday_flag"] = df1["public_holiday_flag"].astype(int)

# Derived features
df1["log_earnings"]              = np.log1p(df1["weekly_earnings_avg"])
df1["risk_weather_interaction"]  = df1["zone_risk_score"] * df1["rain_forecast_prob_7d"]
df1["aqi_normalized"]            = (df1["aqi_forecast_7d"] / 500).clip(0, 1)
df1["loyalty_tier"]              = pd.cut(
    df1["loyalty_weeks_at_purchase"], bins=[-1,0,4,12,999], labels=[0,1,2,3]
).astype(int)

FEATURES_M1 = [
    "zone_risk_score", "rain_forecast_prob_7d", "aqi_forecast_7d",
    "weekly_earnings_avg", "regional_exposure_factor",
    "loyalty_weeks_at_purchase", "autopay_enabled",
    "multi_platform_score", "n_platforms", "public_holiday_flag",
    "log_earnings", "risk_weather_interaction", "aqi_normalized", "loyalty_tier",
]
TARGET_M1 = "final_weekly_premium"

X1 = df1[FEATURES_M1].copy()
y1 = df1[TARGET_M1].copy()

X1_tr, X1_tmp, y1_tr, y1_tmp = train_test_split(X1, y1, test_size=0.30, random_state=42)
X1_val, X1_te, y1_val, y1_te = train_test_split(X1_tmp, y1_tmp, test_size=0.50, random_state=42)

df1_out = X1.copy()
df1_out[TARGET_M1] = y1.values
sp = pd.Series("train", index=X1.index)
sp.loc[X1_val.index] = "val"
sp.loc[X1_te.index]  = "test"
df1_out["split"] = sp
df1_out.to_csv(f"{OUTPUT}/features_model1_premium.csv", index=False)

print(f"  Rows={len(df1_out):,}  Features={len(FEATURES_M1)}")
print(f"  Train/Val/Test = {len(X1_tr):,} / {len(X1_val):,} / {len(X1_te):,}")
print(f"  Target: Rs.{y1.min():.0f}–Rs.{y1.max():.0f}  mean=Rs.{y1.mean():.1f}")
print(f"  Nulls={df1_out.isnull().sum().sum()}  Dupes={df1_out.duplicated().sum()}")

metadata["model1"] = {
    "features": FEATURES_M1, "target": TARGET_M1,
    "n_rows": len(df1_out), "train": len(X1_tr), "val": len(X1_val), "test": len(X1_te),
    "target_mean": round(float(y1.mean()),2), "target_std": round(float(y1.std()),2),
}

# ══════════════════════════════════════════════════════════════════
# MODEL 2 — REGIONAL PROFIT PROTECTION
# ══════════════════════════════════════════════════════════════════
print("\n── MODEL 2: Regional Profit Protection ──")

events["event_date"] = pd.to_datetime(events["event_date"])

# City-week aggregation from events
city_week = events.groupby(["city","week_number","event_date"]).agg(
    n_triggers          =("any_trigger_fired","sum"),
    total_disrupted_hrs =("disrupted_hours","sum"),
    max_rainfall        =("rainfall_mm_24h","max"),
    max_aqi             =("aqi_value","max"),
    max_temp            =("temperature_celsius","max"),
    curfew_any          =("curfew_imposed","max"),
    flood_any           =("flood_detected","max"),
).reset_index()

# City-week from policies
cp = policies.groupby(["city","policy_week"]).agg(
    n_active_policies       =("policy_id","count"),
    total_premium_collected =("final_weekly_premium","sum"),
    avg_zone_risk           =("zone_risk_score","mean"),
    avg_rain_forecast       =("rain_forecast_prob_7d","mean"),
    avg_aqi_forecast        =("aqi_forecast_7d","mean"),
    avg_regional_exposure   =("regional_exposure_factor","mean"),
).reset_index().rename(columns={"policy_week":"week_number"})

# City-week claims (actual payouts)
claims["week_number"] = claims["week_number"].astype(int)
cw_claims = claims.groupby(["city","week_number"]).agg(
    total_payout =("payout_amount","sum"),
    n_claims     =("claim_id","count"),
).reset_index()

df2 = city_week.merge(cp, on=["city","week_number"], how="left")
df2 = df2.merge(cw_claims, on=["city","week_number"], how="left")
df2["total_payout"]  = df2["total_payout"].fillna(0)
df2["n_claims"]      = df2["n_claims"].fillna(0).astype(int)
df2["curfew_any"]    = df2["curfew_any"].astype(int)
df2["flood_any"]     = df2["flood_any"].astype(int)

# Derived
df2["loss_ratio"] = (
    df2["total_payout"] / df2["total_premium_collected"].replace(0,np.nan)
).fillna(0).clip(0,5)
df2["claim_rate"] = (
    df2["n_claims"] / df2["n_active_policies"].replace(0,np.nan)
).fillna(0)
df2["exposure_score"] = (
    (df2["max_rainfall"]/200)*0.4 +
    (df2["max_aqi"]/500).clip(0,1)*0.3 +
    df2["curfew_any"]*0.15 +
    df2["flood_any"]*0.15
).clip(0,1)
df2["month"]        = df2["event_date"].dt.month
df2["is_monsoon"]   = df2["month"].isin([6,7,8,9,10]).astype(int)
df2["is_winter"]    = df2["month"].isin([11,12,1,2]).astype(int)
df2["week_of_year"] = df2["event_date"].dt.isocalendar().week.astype(int)

# Prophet format
prophet_rows = []
for city in df2["city"].unique():
    cdf = df2[df2["city"]==city].sort_values("event_date")
    for _, row in cdf.iterrows():
        prophet_rows.append({
            "ds": row["event_date"], "y": row["total_payout"], "city": city,
            "max_rainfall": row["max_rainfall"], "max_aqi": row["max_aqi"],
            "n_active_policies": row.get("n_active_policies",0) or 0,
            "is_monsoon": row["is_monsoon"], "is_winter": row["is_winter"],
            "exposure_score": row["exposure_score"],
            "avg_zone_risk": row.get("avg_zone_risk",0) or 0,
        })
df2_prophet = pd.DataFrame(prophet_rows)
df2_prophet.to_csv(f"{OUTPUT}/features_model2_regional.csv", index=False)

# XGBoost format
FEATURES_M2 = [
    "max_rainfall","max_aqi","max_temp","curfew_any","flood_any",
    "avg_zone_risk","avg_rain_forecast","avg_aqi_forecast",
    "n_active_policies","is_monsoon","is_winter","exposure_score","week_of_year",
]
df2_xgb = df2[FEATURES_M2 + ["loss_ratio","city","week_number"]].fillna(0).copy()
df2_xgb.to_csv(f"{OUTPUT}/features_model2_xgb.csv", index=False)

monsoon_lr  = df2[df2["is_monsoon"]==1]["loss_ratio"].mean()
offszn_lr   = df2[df2["is_monsoon"]==0]["loss_ratio"].mean()
print(f"  Prophet rows={len(df2_prophet):,}  XGB rows={len(df2_xgb):,}")
print(f"  Loss ratio: mean={df2['loss_ratio'].mean():.3f}  max={df2['loss_ratio'].max():.3f}")
print(f"  Monsoon loss={monsoon_lr:.3f}  Off-season={offszn_lr:.3f}")
print(f"  Nulls={df2_prophet.isnull().sum().sum()}")

metadata["model2"] = {
    "prophet_features": ["max_rainfall","max_aqi","n_active_policies","is_monsoon","is_winter","exposure_score","avg_zone_risk"],
    "xgb_features": FEATURES_M2, "target": "loss_ratio",
    "cities": list(df2["city"].unique()), "weeks_per_city": 156,
}

# ══════════════════════════════════════════════════════════════════
# MODEL 3 — FRAUD DETECTION
# ══════════════════════════════════════════════════════════════════
print("\n── MODEL 3: Fraud Detection ──")

df3 = claims.copy()
for col in ["gps_in_affected_zone","gps_spoof_detected","delivery_activity_detected",
            "duplicate_claim","eligibility_passed","new_registration_fraud",
            "abnormal_claim_freq","curfew_imposed"]:
    df3[col] = df3[col].astype(int)

# Derived
df3["gps_violation"]        = 1 - df3["gps_in_affected_zone"]
df3["n_fraud_signals"]      = (
    df3["gps_violation"] + df3["gps_spoof_detected"] +
    df3["delivery_activity_detected"] + df3["duplicate_claim"] +
    (1-df3["eligibility_passed"]) + df3["new_registration_fraud"] +
    df3["abnormal_claim_freq"]
)
df3["payout_premium_ratio"] = (df3["payout_amount"] / df3["premium_paid"].replace(0,np.nan)).fillna(0).clip(0,20)
df3["earnings_normalized"]  = (df3["weekly_earnings_avg"] / df3["weekly_earnings_avg"].max()).clip(0,1)
df3["disruption_severity"]  = (df3["disrupted_hours"] / 24).clip(0,1)
df3["rainfall_normalized"]  = (df3["rainfall_mm"] / 200).clip(0,1)
df3["aqi_normalized"]       = (df3["aqi_value"] / 500).clip(0,1)
df3["platforms_norm"]       = (df3["n_platforms_verified"] / 5).clip(0,1)

FEATURES_M3 = [
    "gps_violation","gps_spoof_detected","delivery_activity_detected",
    "duplicate_claim","new_registration_fraud","abnormal_claim_freq","n_fraud_signals",
    "payout_premium_ratio","earnings_normalized","disruption_severity",
    "rainfall_normalized","aqi_normalized","platforms_norm","fraud_risk_score",
]
TARGET_M3 = "is_fraud_ground_truth"

X3 = df3[FEATURES_M3].fillna(0)
y3 = df3[TARGET_M3].astype(int)

scaler3 = StandardScaler()
X3_sc   = pd.DataFrame(scaler3.fit_transform(X3), columns=FEATURES_M3)
with open(f"{OUTPUT}/scaler_model3.pkl","wb") as f: pickle.dump(scaler3, f)

X3_tr, X3_te, y3_tr, y3_te = train_test_split(X3_sc, y3, test_size=0.20, random_state=42, stratify=y3)

df3_out = X3_sc.copy()
df3_out[TARGET_M3]      = y3.values
df3_out["claim_status"] = df3["claim_status"].values
df3_out["claim_id"]     = df3["claim_id"].values
sp3 = pd.Series("train", index=X3_sc.index)
sp3.loc[X3_te.index] = "test"
df3_out["split"] = sp3
df3_out.to_csv(f"{OUTPUT}/features_model3_fraud.csv", index=False)

print(f"  Rows={len(df3_out):,}  Features={len(FEATURES_M3)}")
print(f"  Train/Test = {len(X3_tr):,} / {len(X3_te):,}")
print(f"  Fraud in train={y3_tr.sum()} ({y3_tr.mean()*100:.1f}%)  test={y3_te.sum()} ({y3_te.mean()*100:.1f}%)")
print(f"  Scaler saved.  Nulls={df3_out[FEATURES_M3].isnull().sum().sum()}")
print("  Signal separation (fraud mean vs legit mean):")
for feat in ["gps_violation","delivery_activity_detected","n_fraud_signals","payout_premium_ratio"]:
    fm = X3[feat][y3==1].mean(); lm = X3[feat][y3==0].mean()
    print(f"    {feat:<35}: fraud={fm:.3f}  legit={lm:.3f}  sep={abs(fm-lm):.3f}")

metadata["model3"] = {
    "features": FEATURES_M3, "target": TARGET_M3,
    "fraud_rate": round(float(y3.mean()),4),
    "contamination_param": 0.08,
    "scaler": "scaler_model3.pkl",
}

# ══════════════════════════════════════════════════════════════════
# MODEL 4 — RISK PROFILING (K-Means)
# ══════════════════════════════════════════════════════════════════
print("\n── MODEL 4: Risk Profiling ──")

# Zone features
df4z = zones.copy()
df4z["drainage_quality_enc"] = df4z["drainage_quality"].map({"Poor":0,"Moderate":1,"Good":2})
df4z["risk_tier_label"]      = df4z["risk_tier"].map({"Low":0,"Medium":1,"High":2,"Extreme":3})

ZONE_FEATS = [
    "zone_risk_score","waterlogging_risk_score","hist_claim_rate",
    "hist_disruption_freq_per_year","avg_aqi_baseline",
    "drainage_score","curfew_history_count","platform_activity_density",
]

X4z       = df4z[ZONE_FEATS].fillna(0)
scaler4z  = StandardScaler()
X4z_sc    = pd.DataFrame(scaler4z.fit_transform(X4z), columns=ZONE_FEATS)
with open(f"{OUTPUT}/scaler_model4_zones.pkl","wb") as f: pickle.dump(scaler4z, f)

df4z_out = X4z_sc.copy()
df4z_out["zone_id"]         = df4z["zone_id"].values
df4z_out["city"]            = df4z["city"].values
df4z_out["risk_tier"]       = df4z["risk_tier"].values
df4z_out["risk_tier_label"] = df4z["risk_tier_label"].values
df4z_out.to_csv(f"{OUTPUT}/features_model4_zones.csv", index=False)

# Worker features
df4w = workers.copy()
df4w["autopay_enabled"]   = df4w["autopay_enabled"].astype(int)
df4w["is_eligible"]       = df4w["is_eligible"].astype(int)
df4w = df4w.merge(
    df4z[["zone_id","zone_risk_score","waterlogging_risk_score","risk_tier_label"]],
    on="zone_id", how="left", suffixes=("","_zone")
)
df4w["weekly_active_hours"] = df4w["hours_per_day"] * df4w["active_days_per_week"]
df4w["earnings_per_hour"]   = (df4w["weekly_earnings_avg"] / df4w["weekly_active_hours"].replace(0,np.nan)).fillna(0)
df4w["log_earnings"]        = np.log1p(df4w["weekly_earnings_avg"])
df4w["claim_rate_personal"] = (df4w["hist_claim_count"] / df4w["weeks_active"].replace(0,np.nan)).fillna(0).clip(0,1)

WORKER_FEATS = [
    "zone_risk_score","waterlogging_risk_score","weekly_earnings_avg",
    "hours_per_day","active_days_per_week","n_platforms","loyalty_weeks",
    "autopay_enabled","multi_platform_score","hist_claim_count","weeks_active",
    "log_earnings","claim_rate_personal","weekly_active_hours","earnings_per_hour",
]

X4w       = df4w[WORKER_FEATS].fillna(0)
scaler4w  = StandardScaler()
X4w_sc    = pd.DataFrame(scaler4w.fit_transform(X4w), columns=WORKER_FEATS)
with open(f"{OUTPUT}/scaler_model4_workers.pkl","wb") as f: pickle.dump(scaler4w, f)

df4w_out = X4w_sc.copy()
df4w_out["worker_id"]   = df4w["worker_id"].values
df4w_out["city"]        = df4w["city"].values
df4w_out["worker_type"] = df4w["worker_type"].values
df4w_out["zone_id"]     = df4w["zone_id"].values
df4w_out.to_csv(f"{OUTPUT}/features_model4_workers.csv", index=False)

# Elbow method on zones
print("  Elbow method (k=2..8) on zone features:")
inertias = {}
for k in range(2,9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X4z_sc)
    inertias[k] = round(km.inertia_,2)
drops  = {k: inertias[k-1]-inertias[k] for k in range(3,9)}
best_k = max(drops, key=drops.get)
for k,v in inertias.items():
    marker = " ← recommended" if k==best_k else ""
    print(f"    k={k}: inertia={v:.1f}{marker}")

print(f"  Zone rows={len(df4z_out)}  Worker rows={len(df4w_out):,}")
print(f"  Zone features={len(ZONE_FEATS)}  Worker features={len(WORKER_FEATS)}")
print(f"  Tier distribution: {df4z['risk_tier'].value_counts().to_dict()}")
print(f"  Nulls zones={df4z_out[ZONE_FEATS].isnull().sum().sum()}  workers={df4w_out[WORKER_FEATS].isnull().sum().sum()}")

metadata["model4"] = {
    "zone_features": ZONE_FEATS, "worker_features": WORKER_FEATS,
    "recommended_k": best_k, "elbow_inertias": inertias,
    "zone_rows": len(df4z_out), "worker_rows": len(df4w_out),
}

# ── Save metadata ─────────────────────────────────────
with open(f"{OUTPUT}/feature_metadata.json","w") as f:
    json.dump(metadata, f, indent=2)

# ── Final summary ─────────────────────────────────────
print("\n" + "="*60)
print("  CHUNK 2 — FEATURE ENGINEERING COMPLETE")
print("="*60)
for fname in sorted(os.listdir(OUTPUT)):
    sz = os.path.getsize(f"{OUTPUT}/{fname}") / 1024
    print(f"  {fname:<46} {sz:>7.1f} KB")
print("="*60)
print("\n  Next → Chunk 3: Train Model 1 (XGBoost Premium Calculator)")
