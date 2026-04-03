"""
GigGuard — Chunk 6: Train Model 4
Worker & Zone Risk Profiling (K-Means Clustering)

What this does:
  - Loads features_model4_zones.csv + features_model4_workers.csv
  - Runs elbow method to confirm optimal K
  - Trains K-Means (k=4) on zones → Low/Medium/High/Extreme tiers
  - Trains K-Means (k=4) on workers → risk segments
  - Evaluates with silhouette score
  - Saves models/model4_kmeans_zones.pkl + model4_kmeans_workers.pkl

Run:
  python3 train_model4_profiling.py
"""

import numpy as np
import pandas as pd
import joblib, json, os, warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

FEATURES_DIR = "gigguard_features"
DATA_DIR     = "gigguard_data"
MODEL_OUT    = "models"
os.makedirs(MODEL_OUT, exist_ok=True)

print("=" * 58)
print("  GigGuard — Model 4: Risk Profiling (K-Means)")
print("=" * 58)

# ══════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════
print("\n[1/6] Loading profiling data...")

# Try loading pre-engineered features first
zones_feat_path   = f"{FEATURES_DIR}/features_model4_zones.csv"
workers_feat_path = f"{FEATURES_DIR}/features_model4_workers.csv"

ZONE_FEATURES = [
    "zone_risk_score", "waterlogging_risk_score", "hist_claim_rate",
    "hist_disruption_freq_per_year", "avg_aqi_baseline",
    "drainage_score", "curfew_history_count", "platform_activity_density",
]
WORKER_FEATURES = [
    "zone_risk_score", "waterlogging_risk_score", "weekly_earnings_avg",
    "hours_per_day", "active_days_per_week", "n_platforms", "loyalty_weeks",
    "autopay_enabled", "multi_platform_score", "hist_claim_count",
    "weeks_active", "log_earnings", "claim_rate_personal",
    "weekly_active_hours", "earnings_per_hour",
]

# Load zone features
if os.path.exists(zones_feat_path):
    df_zones_feat = pd.read_csv(zones_feat_path)
    # Use scaled feature columns if available
    zone_feat_cols = [c for c in ZONE_FEATURES if c in df_zones_feat.columns]
    X_zones = df_zones_feat[zone_feat_cols].fillna(0).values
    zone_meta = df_zones_feat[["zone_id","city","risk_tier"]].copy() if "zone_id" in df_zones_feat.columns else None
    print(f"  Zone feature file loaded: {len(df_zones_feat)} zones, {len(zone_feat_cols)} features")
else:
    # Build from raw data
    print("  Feature file not found — building from raw data...")
    df_zones_raw = pd.read_csv(f"{DATA_DIR}/zone_risk.csv")
    zone_feat_cols = [c for c in ZONE_FEATURES if c in df_zones_raw.columns]
    scaler_z = StandardScaler()
    X_zones  = scaler_z.fit_transform(df_zones_raw[zone_feat_cols].fillna(0))
    zone_meta = df_zones_raw[["zone_id","city","risk_tier"]].copy()
    joblib.dump(scaler_z, f"{MODEL_OUT}/scaler_model4_zones.pkl")
    print(f"  Built from raw: {len(df_zones_raw)} zones")

# Load worker features
if os.path.exists(workers_feat_path):
    df_workers_feat = pd.read_csv(workers_feat_path)
    worker_feat_cols = [c for c in WORKER_FEATURES if c in df_workers_feat.columns]
    X_workers = df_workers_feat[worker_feat_cols].fillna(0).values
    worker_meta = df_workers_feat[["worker_id","city","worker_type"]].copy() if "worker_id" in df_workers_feat.columns else None
    print(f"  Worker feature file loaded: {len(df_workers_feat):,} workers, {len(worker_feat_cols)} features")
else:
    print("  Worker feature file not found — building from raw data...")
    df_workers_raw = pd.read_csv(f"{DATA_DIR}/workers.csv")
    df_zones_raw   = pd.read_csv(f"{DATA_DIR}/zone_risk.csv")
    df_workers_raw = df_workers_raw.merge(
        df_zones_raw[["zone_id","zone_risk_score","waterlogging_risk_score"]],
        on="zone_id", how="left", suffixes=("","_zone")
    )
    df_workers_raw["autopay_enabled"]   = df_workers_raw["autopay_enabled"].astype(int)
    df_workers_raw["weekly_active_hours"]= df_workers_raw["hours_per_day"] * df_workers_raw["active_days_per_week"]
    df_workers_raw["earnings_per_hour"]  = (df_workers_raw["weekly_earnings_avg"] / df_workers_raw["weekly_active_hours"].replace(0,np.nan)).fillna(0)
    df_workers_raw["log_earnings"]       = np.log1p(df_workers_raw["weekly_earnings_avg"])
    df_workers_raw["claim_rate_personal"]= (df_workers_raw["hist_claim_count"] / df_workers_raw["weeks_active"].replace(0,np.nan)).fillna(0).clip(0,1)
    worker_feat_cols = [c for c in WORKER_FEATURES if c in df_workers_raw.columns]
    scaler_w  = StandardScaler()
    X_workers = scaler_w.fit_transform(df_workers_raw[worker_feat_cols].fillna(0))
    worker_meta = df_workers_raw[["worker_id","city","worker_type"]].copy()
    joblib.dump(scaler_w, f"{MODEL_OUT}/scaler_model4_workers.pkl")
    print(f"  Built from raw: {len(df_workers_raw):,} workers")

print(f"\n  Zone matrix   : {X_zones.shape}")
print(f"  Worker matrix : {X_workers.shape}")

# ══════════════════════════════════════════════════════
# STEP 2 — ELBOW METHOD (confirm best K)
# ══════════════════════════════════════════════════════
print("\n[2/6] Elbow Method — finding optimal K...")

inertias    = {}
sil_scores  = {}

for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_zones)
    inertias[k]   = round(km.inertia_, 2)
    if k >= 2:
        sil = silhouette_score(X_zones, labels)
        sil_scores[k] = round(sil, 4)

print(f"\n  {'K':>4}  {'Inertia':>12}  {'Silhouette':>12}  {'Drop':>10}")
print(f"  {'-'*42}")
for k in range(2, 9):
    drop = inertias[k-1] - inertias[k] if k > 2 else 0
    sil  = sil_scores.get(k, "-")
    print(f"  {k:>4}  {inertias[k]:>12.1f}  {sil:>12}  {drop:>10.1f}")

# Best K by silhouette
best_k_sil = max(sil_scores, key=sil_scores.get)
print(f"\n  Best K by silhouette : {best_k_sil}")
print(f"  We will use K=4 (aligned with Low/Medium/High/Extreme tiers)")
K = 4

# ══════════════════════════════════════════════════════
# STEP 3 — TRAIN K-MEANS ON ZONES
# ══════════════════════════════════════════════════════
print(f"\n[3/6] Training K-Means (k={K}) on Zones...")

km_zones = KMeans(
    n_clusters   = K,
    random_state = 42,
    n_init       = 20,    # more restarts = more stable solution
    max_iter     = 500,
    algorithm    = "lloyd",
)
zone_cluster_labels = km_zones.fit_predict(X_zones)
zone_inertia        = km_zones.inertia_
zone_sil            = silhouette_score(X_zones, zone_cluster_labels)

print(f"  Inertia     : {zone_inertia:.2f}")
print(f"  Silhouette  : {zone_sil:.4f}  (1.0=perfect, >0.5=good, >0.3=acceptable)")

# Map cluster IDs to risk tier labels
# Sort clusters by mean zone_risk_score (first feature)
# cluster with lowest mean = Low, highest = Extreme
if zone_meta is not None:
    zone_meta = zone_meta.copy()
    zone_meta["cluster"] = zone_cluster_labels

    # First feature is zone_risk_score — use it to order clusters
    cluster_means = {}
    for c in range(K):
        mask = zone_cluster_labels == c
        cluster_means[c] = X_zones[mask, 0].mean()

    sorted_clusters = sorted(cluster_means, key=cluster_means.get)
    tier_labels     = {sorted_clusters[0]: "Low", sorted_clusters[1]: "Medium",
                       sorted_clusters[2]: "High", sorted_clusters[3]: "Extreme"}

    zone_meta["predicted_tier"] = zone_meta["cluster"].map(tier_labels)

    print(f"\n  Cluster → Risk Tier Mapping:")
    for c in range(K):
        members = (zone_cluster_labels == c).sum()
        mean_risk = X_zones[zone_cluster_labels == c, 0].mean()
        print(f"    Cluster {c} → {tier_labels[c]:<8}  n={members:>4}  mean_risk={mean_risk:.3f}")

    # Accuracy vs ground truth tiers (how well do our clusters match?)
    if "risk_tier" in zone_meta.columns:
        correct = (zone_meta["predicted_tier"] == zone_meta["risk_tier"]).sum()
        total   = len(zone_meta)
        accuracy = correct / total
        print(f"\n  Cluster vs Ground Truth Accuracy: {correct}/{total} = {accuracy*100:.1f}%")

        print(f"\n  Cluster breakdown by city:")
        print(zone_meta.groupby(["city","predicted_tier"]).size().unstack(fill_value=0).to_string())

# ══════════════════════════════════════════════════════
# STEP 4 — TRAIN K-MEANS ON WORKERS
# ══════════════════════════════════════════════════════
print(f"\n[4/6] Training K-Means (k={K}) on Workers...")

km_workers = KMeans(
    n_clusters   = K,
    random_state = 42,
    n_init       = 20,
    max_iter     = 500,
    algorithm    = "lloyd",
)
worker_cluster_labels = km_workers.fit_predict(X_workers)
worker_inertia        = km_workers.inertia_
worker_sil            = silhouette_score(X_workers, worker_cluster_labels)

print(f"  Inertia     : {worker_inertia:.2f}")
print(f"  Silhouette  : {worker_sil:.4f}")

# Characterize each worker cluster
if worker_meta is not None:
    worker_meta = worker_meta.copy()
    worker_meta["cluster"] = worker_cluster_labels

    # Sort clusters by earnings (first is low earner, last is high earner + multi-platform)
    # Use earnings_per_hour (index 14 in worker features) if available, else first feature
    earnings_idx = worker_feat_cols.index("weekly_earnings_avg") if "weekly_earnings_avg" in worker_feat_cols else 2
    cluster_earnings = {}
    for c in range(K):
        mask = worker_cluster_labels == c
        cluster_earnings[c] = X_workers[mask, earnings_idx].mean()

    sorted_wc   = sorted(cluster_earnings, key=cluster_earnings.get)
    worker_labels = {
        sorted_wc[0]: "Basic Shield",
        sorted_wc[1]: "Standard Guard",
        sorted_wc[2]: "Pro Protect",
        sorted_wc[3]: "Super Active",
    }
    worker_meta["segment"] = worker_meta["cluster"].map(worker_labels)

    print(f"\n  Worker Segment Profiles:")
    print(f"  {'Segment':<18}  {'Count':>6}  {'Avg Earnings':>14}  {'Avg Platforms':>15}")
    print(f"  {'-'*58}")

    # Load raw worker data for readable stats
    try:
        raw_workers = pd.read_csv(f"{DATA_DIR}/workers.csv")
        raw_workers["cluster"]  = worker_cluster_labels
        raw_workers["segment"]  = worker_meta["segment"].values

        for seg in ["Basic Shield","Standard Guard","Pro Protect","Super Active"]:
            sub = raw_workers[raw_workers["segment"]==seg]
            if len(sub) > 0:
                print(f"  {seg:<18}  {len(sub):>6}  Rs.{sub['weekly_earnings_avg'].mean():>10.0f}  {sub['n_platforms'].mean():>15.2f}")
    except:
        for c in range(K):
            seg = worker_labels[c]
            n   = (worker_cluster_labels == c).sum()
            print(f"  {seg:<18}  {n:>6}")

# ══════════════════════════════════════════════════════
# STEP 5 — PLOTS
# ══════════════════════════════════════════════════════
print("\n[5/6] Generating plots...")

if MPL_AVAILABLE:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Elbow curve
    axes[0,0].plot(list(inertias.keys()), list(inertias.values()),
                   "bo-", linewidth=2, markersize=8)
    axes[0,0].axvline(K, color="red", linestyle="--", linewidth=2, label=f"Chosen K={K}")
    axes[0,0].set_xlabel("Number of Clusters (K)"); axes[0,0].set_ylabel("Inertia")
    axes[0,0].set_title("Elbow Method — Zone Clustering", fontsize=12)
    axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

    # Silhouette scores
    axes[0,1].bar(list(sil_scores.keys()), list(sil_scores.values()),
                  color=["#EF4444" if k==K else "#6366F1" for k in sil_scores.keys()])
    axes[0,1].set_xlabel("K"); axes[0,1].set_ylabel("Silhouette Score")
    axes[0,1].set_title("Silhouette Scores by K", fontsize=12)
    axes[0,1].grid(True, alpha=0.3, axis="y")
    for k, v in sil_scores.items():
        axes[0,1].text(k, v+0.005, f"{v:.3f}", ha="center", fontsize=9)

    # Zone cluster scatter (first 2 features)
    colors_z = ["#10B981","#F59E0B","#EF4444","#6366F1"]
    tier_order = ["Low","Medium","High","Extreme"]
    if zone_meta is not None and "predicted_tier" in zone_meta.columns:
        for i, tier in enumerate(tier_order):
            mask = zone_meta["predicted_tier"] == tier
            axes[1,0].scatter(X_zones[mask.values, 0], X_zones[mask.values, 1],
                            c=colors_z[i], label=tier, s=60, alpha=0.8)
        axes[1,0].set_xlabel("Zone Risk Score (scaled)")
        axes[1,0].set_ylabel("Waterlogging Risk (scaled)")
        axes[1,0].set_title("Zone Risk Clusters", fontsize=12)
        axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

    # Worker cluster scatter (earnings vs zone risk)
    colors_w = ["#10B981","#3B82F6","#F59E0B","#EF4444"]
    segments = ["Basic Shield","Standard Guard","Pro Protect","Super Active"]
    if worker_meta is not None and "segment" in worker_meta.columns:
        for i, seg in enumerate(segments):
            mask = worker_meta["segment"] == seg
            axes[1,1].scatter(X_workers[mask.values, earnings_idx],
                            X_workers[mask.values, 0],
                            c=colors_w[i], label=seg, s=10, alpha=0.5)
        axes[1,1].set_xlabel("Weekly Earnings (scaled)")
        axes[1,1].set_ylabel("Zone Risk Score (scaled)")
        axes[1,1].set_title("Worker Risk Segments", fontsize=12)
        axes[1,1].legend(fontsize=8); axes[1,1].grid(True, alpha=0.3)

    plt.suptitle("GigGuard — Model 4: Risk Profiling (K-Means)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{MODEL_OUT}/model4_profiling_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plots saved → models/model4_profiling_plots.png")

# ══════════════════════════════════════════════════════
# STEP 6 — SAVE MODELS + METADATA
# ══════════════════════════════════════════════════════
print("\n[6/6] Saving models...")

joblib.dump(km_zones,   f"{MODEL_OUT}/model4_kmeans_zones.pkl")
joblib.dump(km_workers, f"{MODEL_OUT}/model4_kmeans_workers.pkl")
print(f"  Zone model saved   → models/model4_kmeans_zones.pkl")
print(f"  Worker model saved → models/model4_kmeans_workers.pkl")

meta4 = {
    "model_name":          "GigGuard Risk Profiling",
    "version":             "1.0.0",
    "k":                   K,
    "zone_features":       zone_feat_cols,
    "worker_features":     worker_feat_cols,
    "zone_silhouette":     round(zone_sil, 4),
    "worker_silhouette":   round(worker_sil, 4),
    "zone_inertia":        round(zone_inertia, 2),
    "worker_inertia":      round(worker_inertia, 2),
    "elbow_inertias":      inertias,
    "silhouette_scores":   sil_scores,
    "zone_tier_mapping":   tier_labels if zone_meta is not None else {},
    "worker_segment_labels": worker_labels if worker_meta is not None else {},
}
with open(f"{MODEL_OUT}/model4_metadata.json","w") as f:
    json.dump(meta4, f, indent=2)
print(f"  Metadata saved     → models/model4_metadata.json")

# ── Quick inference demo ──────────────────────────────
print("\n  ── Quick Inference Demo ──────────────────────")
print("  Given a new zone — what risk tier is it?")

try:
    scaler_z2 = joblib.load(f"{MODEL_OUT}/scaler_model4_zones.pkl")
    new_zone = np.array([[0.72, 0.80, 15.0, 28.0, 190.0, 0.20, 3, 55.0]])
    new_zone_sc  = scaler_z2.transform(new_zone)
    cluster_pred = km_zones.predict(new_zone_sc)[0]
    tier_pred    = tier_labels.get(cluster_pred, f"Cluster {cluster_pred}")
    print(f"  New zone (high waterlogging, high AQI) → {tier_pred}")
except Exception as e:
    cluster_pred = km_zones.predict(new_zone_scaled := X_zones[:1])[0]
    print(f"  Sample zone → Cluster {cluster_pred} ({tier_labels.get(cluster_pred,'?')})")

print("\n  Given a new worker — what segment?")
try:
    scaler_w2 = joblib.load(f"{MODEL_OUT}/scaler_model4_workers.pkl")
    new_worker = np.array([[0.45, 0.50, 3200.0, 9.0, 6, 3, 12,
                            1, 0.65, 3, 85, 8.07, 0.035, 54.0, 59.26]])
    new_worker_sc   = scaler_w2.transform(new_worker[:, :len(worker_feat_cols)])
    seg_cluster     = km_workers.predict(new_worker_sc)[0]
    seg_label       = worker_labels.get(seg_cluster, f"Cluster {seg_cluster}")
    print(f"  Multi-platform rider, Rs.3200/week → {seg_label}")
except Exception as e:
    print(f"  (demo skipped: {e})")

# ── Final summary ─────────────────────────────────────
print("\n" + "=" * 58)
print("  MODEL 4 TRAINING COMPLETE")
print("=" * 58)
print(f"  Zone Silhouette  : {zone_sil:.4f}   (>0.3 = good clusters)")
print(f"  Worker Silhouette: {worker_sil:.4f}   (>0.3 = good clusters)")
print(f"  K used           : {K} (Low / Medium / High / Extreme)")
if zone_meta is not None and "risk_tier" in zone_meta.columns:
    correct = (zone_meta["predicted_tier"] == zone_meta["risk_tier"]).sum()
    print(f"  Tier accuracy    : {correct}/{len(zone_meta)} = {correct/len(zone_meta)*100:.1f}%")
print(f"\n  Files saved:")
print(f"    models/model4_kmeans_zones.pkl")
print(f"    models/model4_kmeans_workers.pkl")
print(f"    models/model4_metadata.json")
print(f"    models/model4_profiling_plots.png")
print("=" * 58)
print("\n  ✅ ALL 4 MODELS TRAINED!")
print("  Next → Chunk 7: Wrap all models into FastAPI endpoints")