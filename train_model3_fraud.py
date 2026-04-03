"""
GigGuard — Chunk 5: Train Model 3
Fraud Detection (Isolation Forest + Rule-Based Layer)

What this does:
  - Loads features_model3_fraud.csv
  - Trains Isolation Forest (unsupervised anomaly detection)
  - Evaluates using ground truth fraud labels
  - Builds rule-based hard-check layer on top
  - Shows confusion matrix + precision/recall/F1
  - Saves models/model3_fraud_isolationforest.pkl

Run:
  python3 train_model3_fraud.py
"""

import numpy as np
import pandas as pd
import joblib, json, os, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve
)

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
print("  GigGuard — Model 3: Fraud Detection Engine")
print("=" * 58)

# ══════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════
print("\n[1/6] Loading fraud feature data...")

df = pd.read_csv(f"{FEATURES_DIR}/features_model3_fraud.csv")

FEATURE_COLS = [
    "gps_violation", "gps_spoof_detected", "delivery_activity_detected",
    "duplicate_claim", "new_registration_fraud", "abnormal_claim_freq",
    "n_fraud_signals", "payout_premium_ratio", "earnings_normalized",
    "disruption_severity", "rainfall_normalized", "aqi_normalized",
    "platforms_norm", "fraud_risk_score",
]

# Keep only columns that exist
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

train_df = df[df["split"] == "train"].copy()
test_df  = df[df["split"] == "test"].copy()

X_train = train_df[FEATURE_COLS].values
y_train = train_df["is_fraud_ground_truth"].astype(int).values

X_test  = test_df[FEATURE_COLS].values
y_test  = test_df["is_fraud_ground_truth"].astype(int).values

print(f"  Train rows : {len(X_train):,}  (fraud={y_train.sum()}, legit={len(y_train)-y_train.sum()})")
print(f"  Test rows  : {len(X_test):,}   (fraud={y_test.sum()}, legit={len(y_test)-y_test.sum()})")
print(f"  Features   : {len(FEATURE_COLS)}")
print(f"  Fraud rate : {y_train.mean()*100:.1f}%")

# ══════════════════════════════════════════════════════
# STEP 2 — RULE-BASED LAYER (Phase 2 — hard checks)
# ══════════════════════════════════════════════════════
print("\n[2/6] Building Rule-Based Hard Check Layer...")

def apply_rule_based_checks(df_input):
    """
    Hard rules that auto-flag fraud regardless of ML score.
    Returns fraud_flag (0/1) and reason string.
    Returns series of rule-based fraud scores 0-1.
    """
    scores = pd.Series(0.0, index=df_input.index)
    reasons = pd.Series("", index=df_input.index)

    # Rule 1 — GPS not in affected zone
    if "gps_violation" in df_input.columns:
        mask = df_input["gps_violation"] >= 0.5
        scores[mask] += 0.35
        reasons[mask] += "GPS_VIOLATION|"

    # Rule 2 — GPS spoofing detected
    if "gps_spoof_detected" in df_input.columns:
        mask = df_input["gps_spoof_detected"] >= 0.5
        scores[mask] += 0.40
        reasons[mask] += "GPS_SPOOF|"

    # Rule 3 — Delivery activity during claimed disruption
    if "delivery_activity_detected" in df_input.columns:
        mask = df_input["delivery_activity_detected"] >= 0.5
        scores[mask] += 0.50
        reasons[mask] += "ACTIVE_DURING_CLAIM|"

    # Rule 4 — Duplicate claim
    if "duplicate_claim" in df_input.columns:
        mask = df_input["duplicate_claim"] >= 0.5
        scores[mask] += 0.45
        reasons[mask] += "DUPLICATE|"

    # Rule 5 — New registration fraud
    if "new_registration_fraud" in df_input.columns:
        mask = df_input["new_registration_fraud"] >= 0.5
        scores[mask] += 0.40
        reasons[mask] += "NEW_REG_FRAUD|"

    # Rule 6 — Multiple signals together
    if "n_fraud_signals" in df_input.columns:
        mask = df_input["n_fraud_signals"] >= 2.0
        scores[mask] += 0.30
        reasons[mask] += "MULTI_SIGNAL|"

    return scores.clip(0, 1), reasons

rule_scores_train, rule_reasons_train = apply_rule_based_checks(train_df[FEATURE_COLS])
rule_scores_test,  rule_reasons_test  = apply_rule_based_checks(test_df[FEATURE_COLS])

# Rule-based predictions (threshold 0.50)
rule_preds_test = (rule_scores_test >= 0.50).astype(int).values

rule_precision = precision_score(y_test, rule_preds_test, zero_division=0)
rule_recall    = recall_score(y_test, rule_preds_test, zero_division=0)
rule_f1        = f1_score(y_test, rule_preds_test, zero_division=0)

print(f"  Rule-Based Layer Performance (Test Set):")
print(f"    Precision : {rule_precision:.4f}  (of flagged claims, how many are real fraud)")
print(f"    Recall    : {rule_recall:.4f}  (of real fraud, how many did we catch)")
print(f"    F1 Score  : {rule_f1:.4f}")

# ══════════════════════════════════════════════════════
# STEP 3 — TRAIN ISOLATION FOREST
# ══════════════════════════════════════════════════════
print("\n[3/6] Training Isolation Forest...")
print("  (Trained on ALL data — unsupervised, no labels used)")

# Key params:
# contamination = 0.08 → tells model ~8% of data is anomalous (our fraud rate)
# n_estimators  = 200  → more trees = more stable anomaly scores
# max_samples   = 256  → standard for IsolationForest
# random_state  = 42   → reproducibility

iso_forest = IsolationForest(
    n_estimators  = 200,
    contamination = 0.08,
    max_samples   = 256,
    max_features  = 1.0,
    bootstrap     = False,
    random_state  = 42,
    n_jobs        = -1,
)

# Train on ALL data (unsupervised — no labels)
X_all = df[FEATURE_COLS].values
iso_forest.fit(X_all)
print(f"  Isolation Forest trained on {len(X_all):,} samples ✓")

# ══════════════════════════════════════════════════════
# STEP 4 — EVALUATE ISOLATION FOREST
# ══════════════════════════════════════════════════════
print("\n[4/6] Evaluating Isolation Forest on Test Set...")

# Isolation Forest returns:
#   predict()      → -1 (anomaly/fraud) or +1 (normal/legit)
#   score_samples() → raw anomaly score (more negative = more anomalous)
#   decision_function() → normalized score

iso_raw_preds   = iso_forest.predict(X_test)          # -1 or +1
iso_fraud_preds = (iso_raw_preds == -1).astype(int)   # convert to 0/1

iso_scores      = iso_forest.decision_function(X_test)  # anomaly scores
# Invert: higher score = more likely fraud (for intuitive thresholding)
iso_fraud_prob  = 1 - (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-10)

iso_precision = precision_score(y_test, iso_fraud_preds, zero_division=0)
iso_recall    = recall_score(y_test, iso_fraud_preds, zero_division=0)
iso_f1        = f1_score(y_test, iso_fraud_preds, zero_division=0)

try:
    iso_auc = roc_auc_score(y_test, iso_fraud_prob)
except:
    iso_auc = 0.0

print(f"  Isolation Forest Performance (Test Set):")
print(f"    Precision : {iso_precision:.4f}")
print(f"    Recall    : {iso_recall:.4f}")
print(f"    F1 Score  : {iso_f1:.4f}")
print(f"    ROC-AUC   : {iso_auc:.4f}")

# ══════════════════════════════════════════════════════
# STEP 5 — COMBINED MODEL (Rule + IsolationForest)
# ══════════════════════════════════════════════════════
print("\n[5/6] Combined Model (Rules + Isolation Forest)...")

# Final fraud score = weighted combination
# Rules carry more weight (they are deterministic and highly reliable)
# IsolationForest adds the anomaly detection layer
RULE_WEIGHT = 0.60
ISO_WEIGHT  = 0.40

combined_scores = (
    RULE_WEIGHT * rule_scores_test.values +
    ISO_WEIGHT  * iso_fraud_prob
)

# GigGuard thresholds (from README):
# > 0.90 → Auto Reject
# > 0.75 → Flag for Manual Review
# <= 0.75 → Auto Approve

def classify_claim(score):
    if score > 0.90:
        return "Auto_Rejected"
    elif score > 0.75:
        return "Flagged_Manual_Review"
    else:
        return "Auto_Approved"

combined_preds    = (combined_scores >= 0.50).astype(int)
combined_statuses = [classify_claim(s) for s in combined_scores]

comb_precision = precision_score(y_test, combined_preds, zero_division=0)
comb_recall    = recall_score(y_test, combined_preds, zero_division=0)
comb_f1        = f1_score(y_test, combined_preds, zero_division=0)
try:
    comb_auc = roc_auc_score(y_test, combined_scores)
except:
    comb_auc = 0.0

print(f"  Combined Model Performance (Test Set):")
print(f"    Precision : {comb_precision:.4f}")
print(f"    Recall    : {comb_recall:.4f}")
print(f"    F1 Score  : {comb_f1:.4f}")
print(f"    ROC-AUC   : {comb_auc:.4f}")

# Claim disposition breakdown
status_counts = pd.Series(combined_statuses).value_counts()
print(f"\n  Claim Disposition Breakdown (Test Set):")
for status, count in status_counts.items():
    pct = count / len(combined_statuses) * 100
    print(f"    {status:<25}: {count:>5} ({pct:.1f}%)")

# Confusion matrix
cm = confusion_matrix(y_test, combined_preds)
print(f"\n  Confusion Matrix:")
print(f"    {'':20}  Predicted Legit  Predicted Fraud")
print(f"    {'Actual Legit':<20}  {cm[0][0]:>15}  {cm[0][1]:>15}")
print(f"    {'Actual Fraud':<20}  {cm[1][0]:>15}  {cm[1][1]:>15}")

tn, fp, fn, tp = cm.ravel()
print(f"\n  True Positives  (fraud caught)     : {tp}")
print(f"  False Positives (legit flagged)    : {fp}")
print(f"  True Negatives  (legit approved)   : {tn}")
print(f"  False Negatives (fraud missed)     : {fn}")
print(f"\n  False Positive Rate : {fp/(fp+tn)*100:.1f}%  (legit claims wrongly flagged)")
print(f"  Fraud Catch Rate    : {tp/(tp+fn)*100:.1f}%  (fraud claims caught)")

# Comparison table
print(f"\n  ── Model Comparison ──────────────────────────")
print(f"  {'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print(f"  {'-'*55}")
print(f"  {'Rule-Based Only':<25} {rule_precision:>10.4f} {rule_recall:>10.4f} {rule_f1:>10.4f}")
print(f"  {'Isolation Forest Only':<25} {iso_precision:>10.4f} {iso_recall:>10.4f} {iso_f1:>10.4f}")
print(f"  {'Combined (Final)':<25} {comb_precision:>10.4f} {comb_recall:>10.4f} {comb_f1:>10.4f}")

# ── Plots ─────────────────────────────────────────────
if MPL_AVAILABLE:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Confusion matrix heatmap
    im = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks([0,1]); axes[0].set_yticks([0,1])
    axes[0].set_xticklabels(["Legit","Fraud"]); axes[0].set_yticklabels(["Legit","Fraud"])
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion Matrix\n(Combined Model)", fontsize=12)
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm[i,j]), ha="center", va="center",
                        fontsize=16, color="white" if cm[i,j] > cm.max()/2 else "black")

    # Fraud score distribution
    fraud_scores   = combined_scores[y_test == 1]
    legit_scores   = combined_scores[y_test == 0]
    axes[1].hist(legit_scores,  bins=40, alpha=0.6, color="#10B981", label="Legit")
    axes[1].hist(fraud_scores,  bins=40, alpha=0.6, color="#EF4444", label="Fraud")
    axes[1].axvline(0.75, color="orange", linestyle="--", linewidth=2, label="Review threshold (0.75)")
    axes[1].axvline(0.90, color="red",    linestyle="--", linewidth=2, label="Reject threshold (0.90)")
    axes[1].set_xlabel("Fraud Score"); axes[1].set_ylabel("Count")
    axes[1].set_title("Fraud Score Distribution", fontsize=12)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Precision-Recall curve
    try:
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, combined_scores)
        axes[2].plot(rec_curve, prec_curve, color="#6366F1", linewidth=2)
        axes[2].fill_between(rec_curve, prec_curve, alpha=0.1, color="#6366F1")
        axes[2].set_xlabel("Recall"); axes[2].set_ylabel("Precision")
        axes[2].set_title(f"Precision-Recall Curve\n(AUC={comb_auc:.3f})", fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim([0,1]); axes[2].set_ylim([0,1])
    except:
        axes[2].text(0.5, 0.5, "PR curve unavailable", ha="center")

    plt.suptitle("GigGuard — Model 3: Fraud Detection Engine", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{MODEL_OUT}/model3_fraud_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved → models/model3_fraud_evaluation.png")

# ── Save models ───────────────────────────────────────
iso_path = f"{MODEL_OUT}/model3_fraud_isolationforest.pkl"
joblib.dump(iso_forest, iso_path)
print(f"\n  Isolation Forest saved → {iso_path}")

# Save combined score pipeline config
meta3 = {
    "model_name":        "GigGuard Fraud Detection Engine",
    "version":           "1.0.0",
    "feature_columns":   FEATURE_COLS,
    "rule_weight":       RULE_WEIGHT,
    "iso_weight":        ISO_WEIGHT,
    "thresholds": {
        "auto_reject":         0.90,
        "manual_review":       0.75,
        "auto_approve":        0.75,
    },
    "contamination":     0.08,
    "performance": {
        "rule_based":     {"precision": round(rule_precision,4), "recall": round(rule_recall,4), "f1": round(rule_f1,4)},
        "isolation_forest":{"precision": round(iso_precision,4), "recall": round(iso_recall,4), "f1": round(iso_f1,4)},
        "combined":        {"precision": round(comb_precision,4),"recall": round(comb_recall,4), "f1": round(comb_f1,4), "roc_auc": round(comb_auc,4)},
    },
    "confusion_matrix": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
    "fraud_catch_rate":  round(float(tp/(tp+fn)),4),
    "false_positive_rate": round(float(fp/(fp+tn)),4),
}

with open(f"{MODEL_OUT}/model3_metadata.json","w") as f:
    json.dump(meta3, f, indent=2)
print(f"  Metadata saved → models/model3_metadata.json")

# ── Quick inference demo ──────────────────────────────
print("\n  ── Quick Inference Demo ──────────────────────")
print("  Scenario A: Ravi — legitimate claim (heavy rain, GPS ok)")
scenario_a = pd.DataFrame([{
    "gps_violation": 0, "gps_spoof_detected": 0, "delivery_activity_detected": 0,
    "duplicate_claim": 0, "new_registration_fraud": 0, "abnormal_claim_freq": 0,
    "n_fraud_signals": 0, "payout_premium_ratio": 2.5, "earnings_normalized": 0.6,
    "disruption_severity": 0.5, "rainfall_normalized": 0.4, "aqi_normalized": 0.2,
    "platforms_norm": 0.4, "fraud_risk_score": 0.05,
}])[FEATURE_COLS]

rule_a, reason_a = apply_rule_based_checks(scenario_a)
iso_a   = iso_forest.decision_function(scenario_a.values)
iso_pa  = float(1 - (iso_a - iso_forest.decision_function(X_all).min()) /
           (iso_forest.decision_function(X_all).max() - iso_forest.decision_function(X_all).min() + 1e-10))
final_a = float(RULE_WEIGHT * rule_a.values[0] + ISO_WEIGHT * iso_pa)
print(f"    Rule score: {rule_a.values[0]:.3f}  ISO score: {iso_pa:.3f}  FINAL: {final_a:.3f} → {classify_claim(final_a)}")

print("  Scenario B: Fraud — GPS spoofing + working during claim")
scenario_b = pd.DataFrame([{
    "gps_violation": 1, "gps_spoof_detected": 1, "delivery_activity_detected": 1,
    "duplicate_claim": 0, "new_registration_fraud": 0, "abnormal_claim_freq": 1,
    "n_fraud_signals": 3, "payout_premium_ratio": 8.0, "earnings_normalized": 0.3,
    "disruption_severity": 0.1, "rainfall_normalized": 0.1, "aqi_normalized": 0.1,
    "platforms_norm": 0.2, "fraud_risk_score": 0.88,
}])[FEATURE_COLS]

rule_b, reason_b = apply_rule_based_checks(scenario_b)
iso_b   = iso_forest.decision_function(scenario_b.values)
iso_pb  = float(1 - (iso_b - iso_forest.decision_function(X_all).min()) /
           (iso_forest.decision_function(X_all).max() - iso_forest.decision_function(X_all).min() + 1e-10))
final_b = float(RULE_WEIGHT * rule_b.values[0] + ISO_WEIGHT * iso_pb)
print(f"    Rule score: {rule_b.values[0]:.3f}  ISO score: {iso_pb:.3f}  FINAL: {final_b:.3f} → {classify_claim(final_b)}")
print(f"    Flags: {reason_b.values[0]}")

# ── Final summary ─────────────────────────────────────
print("\n" + "=" * 58)
print("  MODEL 3 TRAINING COMPLETE")
print("=" * 58)
print(f"  Combined F1       : {comb_f1:.4f}")
print(f"  Fraud Catch Rate  : {tp/(tp+fn)*100:.1f}%")
print(f"  False Positive Rate: {fp/(fp+tn)*100:.1f}%")
print(f"  ROC-AUC           : {comb_auc:.4f}")
print(f"\n  Files saved:")
print(f"    models/model3_fraud_isolationforest.pkl")
print(f"    models/model3_metadata.json")
print(f"    models/model3_fraud_evaluation.png")
print("=" * 58)
print("\n  ✅ Ready for Chunk 6 → Train Model 4 (Risk Profiling)")