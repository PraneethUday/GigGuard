import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { COLORS, FONTS, SIZES, SHADOWS } from "../constants/theme";
import { useAuth } from "../context/AuthContext";
import * as api from "../lib/api";

const TRIGGER_ICONS = {
  heavy_rain: "rainy",
  severe_aqi: "cloud",
  flood: "water",
  curfew: "ban",
  extreme_heat: "thermometer",
};

function money(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  return `₹${Math.round(value)}`;
}

function mapClaimStatus(payoutStatus, claimStatus) {
  const status = (payoutStatus || claimStatus || "").toLowerCase();
  if (
    status === "paid" ||
    status === "approved" ||
    status === "success" ||
    status === "completed"
  ) {
    return "paid";
  }
  if (status === "rejected") return "rejected";
  return "under-review";
}

function formatDate(value) {
  if (!value) return "Unknown date";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return "Unknown date";
  return d.toLocaleDateString("en-IN", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

function prettyTrigger(type) {
  if (!type) return "Disruption";
  return type
    .split("_")
    .map((chunk) => chunk.charAt(0).toUpperCase() + chunk.slice(1))
    .join(" ");
}

const STATUS = {
  paid: {
    bg: "#0A2E18",
    color: COLORS.success,
    label: "Settled",
    icon: "checkmark-circle",
  },
  "under-review": {
    bg: "#2A1A0A",
    color: COLORS.amber,
    label: "Processing",
    icon: "time",
  },
  rejected: {
    bg: "#2E0A0A",
    color: COLORS.error,
    label: "Rejected",
    icon: "close-circle",
  },
};

const FILTERS = ["all", "paid", "under-review", "rejected"];

export default function ClaimsScreen({ navigation }) {
  const { user, token } = useAuth();
  const [filter, setFilter] = useState("all");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");
  const [claims, setClaims] = useState([]);
  const [triggerStatus, setTriggerStatus] = useState({});

  const loadClaims = useCallback(
    async (isRefresh = false) => {
      if (!user?.delivery_id) {
        setLoading(false);
        setClaims([]);
        setError("Delivery ID missing for this account.");
        return;
      }

      if (isRefresh) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }

      setError("");

      try {
        const [claimRes, triggerRes] = await Promise.allSettled([
          api.getWorkerClaims(token, user.delivery_id),
          api.getTriggerStatus(),
        ]);

        if (claimRes.status === "fulfilled") {
          setClaims(claimRes.value?.data || []);
        } else {
          setClaims([]);
        }

        if (triggerRes.status === "fulfilled") {
          setTriggerStatus(triggerRes.value || {});
        }

        if (claimRes.status === "rejected") {
          setError("Unable to fetch claims right now.");
        }
      } catch {
        setError("Unable to fetch claims right now.");
      } finally {
        setLoading(false);
        setRefreshing(false);
      }
    },
    [token, user?.delivery_id],
  );

  useEffect(() => {
    loadClaims(false);
  }, [loadClaims]);

  const normalizedClaims = useMemo(() => {
    return claims.map((claim) => ({
      ...claim,
      _status: mapClaimStatus(claim.payout_status, claim.status),
      _icon: TRIGGER_ICONS[claim.trigger_type] || "document-text",
    }));
  }, [claims]);

  const filtered =
    filter === "all"
      ? normalizedClaims
      : normalizedClaims.filter((c) => c._status === filter);

  const totalPaid = normalizedClaims
    .filter((c) => c._status === "paid")
    .reduce((a, c) => a + (Number(c.payout_amount) || 0), 0);
  const paidCount = normalizedClaims.filter((c) => c._status === "paid").length;
  const rejectedCount = normalizedClaims.filter(
    (c) => c._status === "rejected",
  ).length;

  const cityLive = user?.city ? triggerStatus?.[user.city] : null;
  const liveTriggers = cityLive?.triggers_fired || [];

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loaderWrap}>
          <ActivityIndicator color={COLORS.primary} size="large" />
          <Text style={styles.loaderText}>Loading claims...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          onPress={() => navigation.goBack()}
          style={styles.backBtn}
        >
          <Ionicons name="arrow-back" size={20} color={COLORS.white} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Claims</Text>
        <TouchableOpacity
          onPress={() => loadClaims(true)}
          style={styles.refreshBtn}
          disabled={refreshing}
        >
          {refreshing ? (
            <ActivityIndicator color={COLORS.white} size="small" />
          ) : (
            <Ionicons name="refresh" size={18} color={COLORS.white} />
          )}
        </TouchableOpacity>
      </View>

      <ScrollView showsVerticalScrollIndicator={false}>
        {!!error && (
          <View style={styles.errorCard}>
            <Ionicons name="warning-outline" size={16} color={COLORS.error} />
            <Text style={styles.errorCardText}>{error}</Text>
          </View>
        )}

        {/* Live Disruption Alert */}
        <View style={styles.liveAlert}>
          <View style={styles.liveAlertTop}>
            <View style={styles.liveDot} />
            <Text style={styles.liveAlertTitle}>
              {liveTriggers.length > 0
                ? "DISRUPTION DETECTED"
                : "NO LIVE DISRUPTION"}
            </Text>
            <View style={styles.liveBadge}>
              <Text style={styles.liveBadgeText}>
                {liveTriggers.length > 0 ? "LIVE" : "CLEAR"}
              </Text>
            </View>
          </View>

          {liveTriggers.length > 0 ? (
            <>
              <Text style={styles.liveAlertSub}>
                {liveTriggers[0].description}
              </Text>
              <Text style={styles.liveAlertSub}>
                Auto-claims are being evaluated for your policy in{" "}
                {user?.city || "your city"}.
              </Text>
            </>
          ) : (
            <Text style={styles.liveAlertSub}>
              No active trigger events in {user?.city || "your city"} right now.
            </Text>
          )}

          {/* Progress steps */}
          <View style={styles.claimSteps}>
            {["Detected", "Validated", "Processing", "Payout"].map((s, i) => {
              const done = liveTriggers.length === 0 ? i < 4 : i < 2;
              const active = liveTriggers.length > 0 && i === 2;
              return (
                <React.Fragment key={s}>
                  <View style={styles.claimStep}>
                    <View
                      style={[
                        styles.claimStepDot,
                        done && styles.claimStepDone,
                        active && styles.claimStepActive,
                      ]}
                    >
                      {done && (
                        <Ionicons name="checkmark" size={9} color="#fff" />
                      )}
                    </View>
                    <Text
                      style={[
                        styles.claimStepText,
                        active && { color: COLORS.amber },
                      ]}
                    >
                      {s}
                    </Text>
                  </View>
                  {i < 3 && (
                    <View
                      style={[
                        styles.claimStepLine,
                        done && { backgroundColor: COLORS.success },
                      ]}
                    />
                  )}
                </React.Fragment>
              );
            })}
          </View>

          <View style={styles.progressBar}>
            <View
              style={[
                styles.progressFill,
                { width: liveTriggers.length > 0 ? "60%" : "100%" },
              ]}
            />
          </View>
          <Text style={styles.estPayout}>
            Est. payout:{" "}
            <Text style={{ color: COLORS.success, fontFamily: FONTS.bold }}>
              {money(totalPaid / Math.max(1, paidCount || 1))}
            </Text>
          </Text>
        </View>

        {/* Summary banner */}
        <View style={styles.summaryRow}>
          <View style={styles.summaryItem}>
            <Text
              style={styles.summaryValue}
              numberOfLines={1}
              adjustsFontSizeToFit
              minimumFontScale={0.75}
            >
              {normalizedClaims.length}
            </Text>
            <Text style={styles.summaryLabel}>Total</Text>
          </View>
          <View style={styles.summaryDivider} />
          <View style={styles.summaryItem}>
            <Text
              style={[styles.summaryValue, { color: COLORS.success }]}
              numberOfLines={1}
              adjustsFontSizeToFit
              minimumFontScale={0.75}
            >
              {money(totalPaid)}
            </Text>
            <Text style={styles.summaryLabel}>Received</Text>
          </View>
          <View style={styles.summaryDivider} />
          <View style={styles.summaryItem}>
            <Text
              style={styles.summaryValue}
              numberOfLines={1}
              adjustsFontSizeToFit
              minimumFontScale={0.75}
            >
              {paidCount}
            </Text>
            <Text style={styles.summaryLabel}>Settled</Text>
          </View>
          <View style={styles.summaryDivider} />
          <View style={styles.summaryItem}>
            <Text
              style={[styles.summaryValue, { color: COLORS.error }]}
              numberOfLines={1}
              adjustsFontSizeToFit
              minimumFontScale={0.75}
            >
              {rejectedCount}
            </Text>
            <Text style={styles.summaryLabel}>Rejected</Text>
          </View>
        </View>

        {/* Info pill */}
        <View style={styles.infoPill}>
          <Ionicons
            name="information-circle-outline"
            size={16}
            color={COLORS.info}
          />
          <Text style={styles.infoPillText}>
            All claims are triggered automatically — no filing required
          </Text>
        </View>

        {/* Filter chips */}
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          style={styles.filterRow}
          contentContainerStyle={{ paddingHorizontal: SIZES.padding, gap: 8 }}
        >
          {FILTERS.map((f) => (
            <TouchableOpacity
              key={f}
              style={[
                styles.filterChip,
                filter === f && styles.filterChipActive,
              ]}
              onPress={() => setFilter(f)}
            >
              <Text
                style={[
                  styles.filterChipText,
                  filter === f && { color: COLORS.primary },
                ]}
              >
                {f === "all"
                  ? "All"
                  : f === "paid"
                    ? "✓ Paid"
                    : f === "under-review"
                      ? "⏳ Processing"
                      : "✕ Rejected"}
              </Text>
            </TouchableOpacity>
          ))}
        </ScrollView>

        {/* Claims list */}
        <View style={styles.claimsList}>
          {filtered.length === 0 && (
            <View style={styles.emptyCard}>
              <Text style={styles.emptyText}>
                No claims found for this filter.
              </Text>
            </View>
          )}

          {filtered.map((claim) => {
            const sc = STATUS[claim._status] || STATUS["under-review"];
            return (
              <View
                key={claim.id || claim.claim_number}
                style={styles.claimCard}
              >
                <View style={styles.claimTop}>
                  <View
                    style={[styles.claimIconWrap, { backgroundColor: sc.bg }]}
                  >
                    <Ionicons name={claim._icon} size={20} color={sc.color} />
                  </View>
                  <View style={{ flex: 1 }}>
                    <Text style={styles.claimType}>
                      {prettyTrigger(claim.trigger_type)}
                    </Text>
                    <Text style={styles.claimDate}>
                      {formatDate(claim.created_at)} ·{" "}
                      {claim.city || user?.city || "Unknown"}
                    </Text>
                  </View>
                  <View
                    style={[styles.statusBadge, { backgroundColor: sc.bg }]}
                  >
                    <Text style={[styles.statusText, { color: sc.color }]}>
                      {sc.label}
                    </Text>
                  </View>
                </View>

                <View style={styles.claimBottom}>
                  <View style={styles.claimDetail}>
                    <Text style={styles.detailLabel}>Disrupted</Text>
                    <Text style={styles.detailValue}>
                      {claim.disrupted_hours || 0} hrs
                    </Text>
                  </View>
                  <View style={styles.claimDetail}>
                    <Text style={styles.detailLabel}>Claim ID</Text>
                    <Text style={styles.detailValue}>
                      {claim.claim_number || claim.id}
                    </Text>
                  </View>
                  <View style={styles.claimDetail}>
                    <Text style={styles.detailLabel}>Payout</Text>
                    <Text
                      style={[
                        styles.detailValue,
                        {
                          color:
                            claim._status === "rejected"
                              ? COLORS.error
                              : COLORS.success,
                        },
                      ]}
                    >
                      {claim._status === "rejected"
                        ? "—"
                        : money(Number(claim.payout_amount))}
                    </Text>
                  </View>
                </View>

                {claim._status === "paid" && (
                  <View style={styles.claimFooter}>
                    <Ionicons name="flash" size={12} color={COLORS.success} />
                    <Text style={styles.claimFooterText}>
                      {claim.transaction_id
                        ? `UPI transaction: ${claim.transaction_id}`
                        : "Approved for payout. Transaction pending."}
                    </Text>
                  </View>
                )}
                {claim._status === "rejected" && (
                  <View style={styles.claimFooterRejected}>
                    <Ionicons
                      name="alert-circle"
                      size={12}
                      color={COLORS.error}
                    />
                    <Text style={styles.claimFooterRejectedText}>
                      Reason:{" "}
                      {Array.isArray(claim.fraud_flags) &&
                      claim.fraud_flags.length > 0
                        ? claim.fraud_flags.join(", ")
                        : "Failed automated checks"}
                    </Text>
                  </View>
                )}
              </View>
            );
          })}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.surface },
  loaderWrap: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    gap: 12,
  },
  loaderText: {
    fontSize: SIZES.small,
    fontFamily: FONTS.medium,
    color: COLORS.textMuted,
  },

  header: {
    height: 56,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: SIZES.padding,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  backBtn: {
    width: 40,
    height: 40,
    borderRadius: 12,
    backgroundColor: COLORS.surfaceHigh,
    justifyContent: "center",
    alignItems: "center",
  },
  headerTitle: {
    fontSize: SIZES.h3,
    fontFamily: FONTS.bold,
    color: COLORS.white,
  },
  refreshBtn: {
    width: 40,
    height: 40,
    borderRadius: 12,
    backgroundColor: COLORS.surfaceHigh,
    justifyContent: "center",
    alignItems: "center",
  },
  errorCard: {
    margin: SIZES.padding,
    marginBottom: 0,
    backgroundColor: COLORS.errorContainer,
    borderWidth: 1,
    borderColor: COLORS.error + "40",
    borderRadius: SIZES.radius,
    padding: 10,
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  errorCardText: {
    fontSize: SIZES.small,
    fontFamily: FONTS.medium,
    color: COLORS.error,
    flex: 1,
  },

  // Live Alert
  liveAlert: {
    margin: SIZES.padding,
    borderRadius: SIZES.radius * 1.5,
    backgroundColor: COLORS.amberContainer,
    borderWidth: 1,
    borderColor: COLORS.amber + "50",
    padding: SIZES.padding,
  },
  liveAlertTop: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: SIZES.base,
  },
  liveDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: COLORS.amber,
  },
  liveAlertTitle: {
    flex: 1,
    fontSize: SIZES.small,
    fontFamily: FONTS.bold,
    color: COLORS.amber,
    letterSpacing: 0.5,
  },
  liveBadge: {
    backgroundColor: COLORS.amber,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: SIZES.radiusFull,
  },
  liveBadgeText: {
    fontSize: 9,
    fontFamily: FONTS.bold,
    color: "#fff",
    letterSpacing: 0.5,
  },
  liveAlertSub: {
    fontSize: SIZES.small,
    fontFamily: FONTS.medium,
    color: COLORS.amberDim,
    marginBottom: 4,
  },

  claimSteps: {
    flexDirection: "row",
    alignItems: "center",
    marginVertical: SIZES.padding * 0.75,
  },
  claimStep: { alignItems: "center", gap: 4 },
  claimStepDot: {
    width: 18,
    height: 18,
    borderRadius: 9,
    backgroundColor: COLORS.surfaceHighest,
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  claimStepDone: {
    backgroundColor: COLORS.success,
    borderColor: COLORS.success,
  },
  claimStepActive: { backgroundColor: COLORS.amber, borderColor: COLORS.amber },
  claimStepText: {
    fontSize: 9,
    fontFamily: FONTS.medium,
    color: COLORS.textFaint,
  },
  claimStepLine: {
    flex: 1,
    height: 2,
    backgroundColor: COLORS.border,
    marginBottom: 12,
  },
  progressBar: {
    height: 4,
    backgroundColor: COLORS.surfaceHighest,
    borderRadius: 2,
    overflow: "hidden",
    marginTop: SIZES.base,
  },
  progressFill: {
    height: "100%",
    backgroundColor: COLORS.amber,
    borderRadius: 2,
  },
  estPayout: {
    fontSize: SIZES.small,
    fontFamily: FONTS.medium,
    color: COLORS.textMuted,
    marginTop: SIZES.base,
  },

  // Summary
  summaryRow: {
    flexDirection: "row",
    backgroundColor: COLORS.surfaceContainer,
    marginHorizontal: SIZES.padding,
    borderRadius: SIZES.radius * 1.2,
    padding: SIZES.padding,
    marginBottom: SIZES.padding * 0.75,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  summaryItem: { flex: 1, alignItems: "center" },
  summaryValue: {
    fontSize: 20,
    fontFamily: FONTS.bold,
    color: COLORS.white,
    marginBottom: 2,
    textAlign: "center",
    width: "100%",
  },
  summaryLabel: {
    fontSize: SIZES.tiny,
    fontFamily: FONTS.medium,
    color: COLORS.textFaint,
  },
  summaryDivider: { width: 1, backgroundColor: COLORS.border },

  infoPill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingHorizontal: SIZES.padding,
    marginBottom: SIZES.padding * 0.75,
  },
  infoPillText: {
    flex: 1,
    fontSize: SIZES.small,
    fontFamily: FONTS.medium,
    color: COLORS.info,
    lineHeight: 18,
  },

  filterRow: { marginBottom: SIZES.padding },
  filterChip: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: SIZES.radiusFull,
    backgroundColor: COLORS.surfaceContainer,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  filterChipActive: {
    backgroundColor: COLORS.primaryContainer,
    borderColor: COLORS.primary,
  },
  filterChipText: {
    fontSize: SIZES.small,
    fontFamily: FONTS.semiBold,
    color: COLORS.textFaint,
  },

  // Claims list
  claimsList: {
    paddingHorizontal: SIZES.padding,
    paddingBottom: SIZES.padding * 2,
  },
  emptyCard: {
    backgroundColor: COLORS.surfaceContainer,
    borderRadius: SIZES.radius,
    borderWidth: 1,
    borderColor: COLORS.border,
    padding: SIZES.padding,
    marginBottom: SIZES.base,
  },
  emptyText: {
    color: COLORS.textMuted,
    fontSize: SIZES.small,
    fontFamily: FONTS.medium,
  },
  claimCard: {
    backgroundColor: COLORS.surfaceContainer,
    borderRadius: SIZES.radius * 1.2,
    borderWidth: 1,
    borderColor: COLORS.border,
    marginBottom: 12,
    overflow: "hidden",
  },
  claimTop: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    padding: SIZES.padding * 0.85,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  claimIconWrap: {
    width: 42,
    height: 42,
    borderRadius: 12,
    justifyContent: "center",
    alignItems: "center",
  },
  claimType: {
    fontSize: SIZES.body,
    fontFamily: FONTS.bold,
    color: COLORS.white,
    marginBottom: 2,
  },
  claimDate: {
    fontSize: SIZES.small,
    fontFamily: FONTS.medium,
    color: COLORS.textFaint,
  },
  statusBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: SIZES.radiusFull,
  },
  statusText: { fontSize: SIZES.tiny, fontFamily: FONTS.bold },
  claimBottom: {
    flexDirection: "row",
    justifyContent: "space-around",
    padding: SIZES.padding * 0.75,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  claimDetail: { alignItems: "center" },
  detailLabel: {
    fontSize: SIZES.tiny,
    fontFamily: FONTS.medium,
    color: COLORS.textFaint,
    marginBottom: 3,
    textTransform: "uppercase",
    letterSpacing: 0.3,
  },
  detailValue: {
    fontSize: SIZES.small,
    fontFamily: FONTS.bold,
    color: COLORS.white,
  },
  claimFooter: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    padding: SIZES.padding * 0.6,
    paddingHorizontal: SIZES.padding * 0.85,
    backgroundColor: COLORS.successContainer,
  },
  claimFooterText: {
    fontSize: SIZES.tiny,
    fontFamily: FONTS.semiBold,
    color: COLORS.success,
  },
  claimFooterRejected: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    padding: SIZES.padding * 0.6,
    paddingHorizontal: SIZES.padding * 0.85,
    backgroundColor: COLORS.errorContainer,
  },
  claimFooterRejectedText: {
    fontSize: SIZES.tiny,
    fontFamily: FONTS.semiBold,
    color: COLORS.error,
    flex: 1,
  },
});
