import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  ScrollView,
  TouchableOpacity,
  Platform,
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
  extreme_heat: "thermometer",
  curfew: "ban",
};

const PLATFORM_NAMES = {
  swiggy: "Swiggy",
  zomato: "Zomato",
  amazon: "Amazon Flex",
  blinkit: "Blinkit",
  zepto: "Zepto",
  meesho: "Meesho",
  porter: "Porter",
  dunzo: "Dunzo",
};

function money(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  return `₹${Math.round(value)}`;
}

function titleCaseTier(tier) {
  if (!tier) return "Standard";
  return tier.charAt(0).toUpperCase() + tier.slice(1);
}

function riskLabel(risk) {
  if (typeof risk !== "number") return "UNKNOWN";
  if (risk < 0.15) return "LOW RISK";
  if (risk < 0.4) return "MOD RISK";
  if (risk < 0.7) return "HIGH RISK";
  return "SEVERE";
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

function getStatusTone(status) {
  if (status === "paid") return { color: COLORS.success, label: "Paid" };
  if (status === "approved" || status === "pending")
    return { color: COLORS.amber, label: "Processing" };
  if (status === "rejected") return { color: COLORS.error, label: "Rejected" };
  return { color: COLORS.info, label: "Initiated" };
}

const QuickAction = ({ icon, label, onPress, color = COLORS.primary }) => (
  <TouchableOpacity
    style={styles.quickAction}
    onPress={onPress}
    activeOpacity={0.75}
  >
    <View style={[styles.quickActionIcon, { backgroundColor: color + "18" }]}>
      <Ionicons name={icon} size={22} color={color} />
    </View>
    <Text style={styles.quickActionLabel}>{label}</Text>
  </TouchableOpacity>
);

const HomeScreen = ({ navigation }) => {
  const { user, token, refreshUser } = useAuth();
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");
  const [premium, setPremium] = useState(null);
  const [claims, setClaims] = useState([]);
  const [triggerStatus, setTriggerStatus] = useState({});

  const deliveryId = user?.delivery_id;
  const userCity = user?.city;
  const userTier = user?.tier || "standard";

  const loadDashboard = useCallback(
    async (isRefresh = false) => {
      if (!deliveryId) {
        setLoading(false);
        setClaims([]);
        setPremium(null);
        setError("Missing delivery ID for this account.");
        return;
      }

      if (isRefresh) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }

      setError("");

      try {
        if (isRefresh) {
          await refreshUser();
        }

        const [premiumRes, claimsRes, triggersRes] = await Promise.allSettled([
          api.predictPremium(deliveryId, userCity, userTier),
          api.getWorkerClaims(token, deliveryId),
          api.getTriggerStatus(),
        ]);

        if (premiumRes.status === "fulfilled") {
          setPremium(premiumRes.value);
        }

        if (claimsRes.status === "fulfilled") {
          setClaims(claimsRes.value?.data || []);
        } else {
          setClaims([]);
        }

        if (triggersRes.status === "fulfilled") {
          setTriggerStatus(triggersRes.value || {});
        }

        if (
          premiumRes.status === "rejected" &&
          claimsRes.status === "rejected"
        ) {
          setError("Unable to load live dashboard data right now.");
        }
      } catch {
        setError("Unable to load live dashboard data right now.");
      } finally {
        setLoading(false);
        setRefreshing(false);
      }
    },
    [deliveryId, refreshUser, token, userCity, userTier],
  );

  useEffect(() => {
    loadDashboard(false);
  }, [loadDashboard]);

  const cityData = useMemo(() => {
    if (!user?.city) return null;
    return triggerStatus?.[user.city] || null;
  }, [triggerStatus, user?.city]);

  const weather = cityData?.weather || premium?.weather || null;
  const activeTriggers = cityData?.triggers_fired || [];

  const currentPremium = user?.autopay
    ? premium?.weekly_premium_autopay
    : premium?.weekly_premium;
  const coverageActive = user?.verification_status === "verified" && !!premium;
  const paidClaims = claims.filter(
    (c) => c.payout_status === "paid" || c.status === "paid",
  );
  const totalPaid = paidClaims.reduce(
    (sum, claim) => sum + (Number(claim.payout_amount) || 0),
    0,
  );

  const recentActivity = useMemo(() => {
    const claimItems = claims.slice(0, 3).map((claim) => {
      const tone = getStatusTone(claim.payout_status || claim.status);
      return {
        icon: TRIGGER_ICONS[claim.trigger_type] || "document-text",
        color: tone.color,
        title: `${tone.label} · ${claim.claim_number || "Claim"}`,
        sub: `${formatDate(claim.created_at)} · ${money(Number(claim.payout_amount))}`,
      };
    });

    if (claimItems.length > 0) return claimItems;

    return [
      {
        icon: "shield-checkmark",
        color: COLORS.success,
        title: "No claims yet",
        sub: "Your claims will appear automatically during disruptions.",
      },
    ];
  }, [claims]);

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loaderWrap}>
          <ActivityIndicator color={COLORS.primary} size="large" />
          <Text style={styles.loaderText}>
            Loading your GigGuard dashboard...
          </Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.scroll}
      >
        {/* Top bar */}
        <View style={styles.topBar}>
          <View>
            <Text style={styles.greeting}>Good morning,</Text>
            <Text style={styles.userName}>{user?.name || "GigGuard User"}</Text>
            <View style={styles.platformBadge}>
              <Text style={styles.platformBadgeText}>
                {(user?.platforms || []).length > 0
                  ? user.platforms
                      .map((p) => PLATFORM_NAMES[p] || p)
                      .join(" + ")
                  : "No platform linked"}
              </Text>
            </View>
          </View>
          <TouchableOpacity
            style={styles.notifBtn}
            onPress={() => loadDashboard(true)}
            disabled={refreshing}
          >
            {refreshing ? (
              <ActivityIndicator color={COLORS.white} size="small" />
            ) : (
              <Ionicons name="refresh" size={20} color={COLORS.white} />
            )}
          </TouchableOpacity>
        </View>

        {!!error && (
          <View style={styles.errorCard}>
            <Ionicons name="warning-outline" size={16} color={COLORS.error} />
            <Text style={styles.errorCardText}>{error}</Text>
          </View>
        )}

        {/* Active Shield Card */}
        <View style={styles.shieldCard}>
          <View style={styles.shieldGlow} />

          <View style={styles.shieldCardTop}>
            <Text style={styles.shieldCardLabel}>THIS WEEK'S COVERAGE</Text>
            <View style={styles.activeBadge}>
              <View style={styles.activePulse} />
              <Text style={styles.activeBadgeText}>
                {coverageActive ? "ACTIVE" : "INACTIVE"}
              </Text>
            </View>
          </View>

          <Text style={styles.payoutAmount}>{money(premium?.max_payout)}</Text>
          <Text style={styles.payoutLabel}>Maximum weekly payout</Text>

          <View style={styles.shieldCardMeta}>
            <Text style={styles.metaText}>
              {titleCaseTier(user?.tier)} Plan
            </Text>
            <View style={styles.metaDot} />
            <Text style={[styles.metaText, { color: COLORS.success }]}>
              Premium: {money(currentPremium)}
            </Text>
          </View>

          <View style={styles.progressTrack}>
            <View
              style={[
                styles.progressFill,
                { width: coverageActive ? "100%" : "35%" },
              ]}
            />
          </View>
        </View>

        {/* Weather Risk */}
        <View style={styles.weatherCard}>
          <View style={styles.weatherLeft}>
            <View style={styles.weatherIconWrap}>
              <Ionicons
                name={activeTriggers.length > 0 ? "warning" : "partly-sunny"}
                size={22}
                color={COLORS.amber}
              />
            </View>
            <View style={styles.weatherTextWrap}>
              <Text style={styles.weatherCity}>{user?.city || "Unknown"}</Text>
              <Text style={styles.weatherSub} numberOfLines={2}>
                Rain: {weather?.rain_1h ?? 0}mm/h · AQI:{" "}
                {weather?.aqi_index ?? "NA"} · Temp:{" "}
                {weather?.temperature ?? "NA"}°C
              </Text>
            </View>
          </View>
          <View style={styles.riskBadge}>
            <Text style={styles.riskBadgeText}>
              {riskLabel(premium?.weather_risk)}
            </Text>
          </View>
        </View>

        {activeTriggers.length > 0 && (
          <View style={styles.alertCard}>
            <View style={styles.alertCardTop}>
              <Ionicons name="alert-circle" size={16} color={COLORS.amber} />
              <Text style={styles.alertTitle}>Active Disruption Alerts</Text>
            </View>
            {activeTriggers.map((trigger) => (
              <Text key={trigger.trigger_id} style={styles.alertText}>
                • {trigger.description}
              </Text>
            ))}
          </View>
        )}

        {/* Premium Card */}
        <View style={styles.premiumCard}>
          <View style={styles.premiumCardTop}>
            <View>
              <Text style={styles.premiumAmount}>
                {money(currentPremium)}{" "}
                <Text style={styles.premiumPer}>/week</Text>
              </Text>
              <Text style={styles.premiumSub}>
                {user?.autopay ? "AutoPay active" : "AutoPay disabled"} ·
                history days: {premium?.history_days ?? 0}
              </Text>
            </View>
            <TouchableOpacity
              style={styles.payNowBtn}
              onPress={() => navigation.navigate("Policy")}
            >
              <Text style={styles.payNowText}>View Plan</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.premiumBreakdown}>
            {[
              ["Base", money(premium?.raw_prediction)],
              [
                "Weather Risk",
                typeof premium?.weather_risk === "number"
                  ? `${Math.round(premium.weather_risk * 100)}%`
                  : "—",
              ],
              [
                "City Risk",
                typeof premium?.city_risk === "number"
                  ? premium.city_risk.toFixed(2)
                  : "—",
              ],
              ["AutoPay", user?.autopay ? "−5%" : "Off"],
            ].map(([k, v]) => (
              <View key={k} style={styles.breakdownItem}>
                <Text style={styles.breakdownKey}>{k}</Text>
                <Text
                  style={[
                    styles.breakdownVal,
                    v.startsWith("−") && { color: COLORS.success },
                  ]}
                >
                  {v}
                </Text>
              </View>
            ))}
          </View>
        </View>

        {/* Quick Actions */}
        <Text style={styles.sectionTitle}>Quick Actions</Text>
        <View style={styles.quickActionsRow}>
          <QuickAction
            icon="shield-outline"
            label="My Policy"
            onPress={() => navigation.navigate("Policy")}
            color={COLORS.primary}
          />
          <QuickAction
            icon="document-text-outline"
            label="Claims"
            onPress={() => navigation.navigate("Claims")}
            color={COLORS.amber}
          />
          <QuickAction
            icon="card-outline"
            label="Payments"
            onPress={() => navigation.navigate("Payments")}
            color={COLORS.success}
          />
          <QuickAction
            icon="person-outline"
            label="Profile"
            onPress={() => navigation.navigate("Profile")}
            color={COLORS.info}
          />
        </View>

        {/* Recent Activity */}
        <Text style={styles.sectionTitle}>Recent Activity</Text>
        <View style={styles.activityCard}>
          {recentActivity.map((item, i) => (
            <View
              key={`${item.title}-${i}`}
              style={[
                styles.activityRow,
                i < recentActivity.length - 1 && styles.activityBorder,
              ]}
            >
              <View
                style={[
                  styles.activityIcon,
                  { backgroundColor: item.color + "18" },
                ]}
              >
                <Ionicons name={item.icon} size={18} color={item.color} />
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.activityTitle}>{item.title}</Text>
                <Text style={styles.activitySub}>{item.sub}</Text>
              </View>
            </View>
          ))}
        </View>
      </ScrollView>

      {/* Bottom Tab Bar */}
      <View style={styles.bottomNav}>
        <TouchableOpacity style={styles.navItem}>
          <Ionicons name="home" size={22} color={COLORS.primary} />
          <Text style={[styles.navText, { color: COLORS.primary }]}>Home</Text>
          <View style={styles.navActiveDot} />
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.navItem}
          onPress={() => navigation.navigate("Claims")}
        >
          <Ionicons
            name="document-outline"
            size={22}
            color={COLORS.textFaint}
          />
          <Text style={styles.navText}>Claims</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.navItem}
          onPress={() => navigation.navigate("Policy")}
        >
          <Ionicons name="shield-outline" size={22} color={COLORS.textFaint} />
          <Text style={styles.navText}>Policy</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.navItem}
          onPress={() => navigation.navigate("Profile")}
        >
          <Ionicons name="person-outline" size={22} color={COLORS.textFaint} />
          <Text style={styles.navText}>Profile</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.surface },
  scroll: { paddingBottom: 100 },
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

  topBar: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    paddingHorizontal: SIZES.padding,
    paddingTop: SIZES.padding,
    paddingBottom: SIZES.padding * 0.5,
  },
  greeting: {
    fontSize: SIZES.small,
    fontFamily: FONTS.medium,
    color: COLORS.textMuted,
  },
  userName: {
    fontSize: SIZES.h2,
    fontFamily: FONTS.bold,
    color: COLORS.white,
    marginTop: 2,
  },
  platformBadge: {
    backgroundColor: COLORS.primaryContainer,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: SIZES.radiusFull,
    alignSelf: "flex-start",
    marginTop: 6,
  },
  platformBadgeText: {
    fontSize: SIZES.tiny,
    fontFamily: FONTS.semiBold,
    color: COLORS.primaryDim,
  },
  notifBtn: {
    width: 42,
    height: 42,
    borderRadius: 14,
    backgroundColor: COLORS.surfaceHigh,
    justifyContent: "center",
    alignItems: "center",
    marginTop: 4,
  },
  errorCard: {
    marginHorizontal: SIZES.padding,
    marginBottom: SIZES.padding * 0.75,
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
    color: COLORS.error,
    fontSize: SIZES.small,
    fontFamily: FONTS.medium,
    flex: 1,
  },

  // Shield card
  shieldCard: {
    marginHorizontal: SIZES.padding,
    borderRadius: SIZES.radius * 1.5,
    backgroundColor: COLORS.primary,
    padding: SIZES.padding,
    marginBottom: SIZES.padding,
    overflow: "hidden",
    ...SHADOWS.button,
  },
  shieldGlow: {
    position: "absolute",
    width: 200,
    height: 200,
    borderRadius: 100,
    backgroundColor: "#fff",
    opacity: 0.05,
    top: -80,
    right: -60,
  },
  shieldCardTop: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: SIZES.padding * 0.75,
  },
  shieldCardLabel: {
    fontSize: SIZES.tiny,
    fontFamily: FONTS.bold,
    color: "rgba(255,255,255,0.6)",
    letterSpacing: 1,
  },
  activeBadge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
    backgroundColor: "rgba(255,255,255,0.15)",
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: SIZES.radiusFull,
  },
  activePulse: {
    width: 7,
    height: 7,
    borderRadius: 4,
    backgroundColor: COLORS.success,
  },
  activeBadgeText: {
    fontSize: SIZES.tiny,
    fontFamily: FONTS.bold,
    color: "#fff",
  },
  payoutAmount: {
    fontSize: 40,
    fontFamily: FONTS.bold,
    color: "#fff",
    marginBottom: 4,
  },
  payoutLabel: {
    fontSize: SIZES.small,
    fontFamily: FONTS.medium,
    color: "rgba(255,255,255,0.65)",
    marginBottom: SIZES.padding,
  },
  shieldCardMeta: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    marginBottom: SIZES.base,
  },
  metaText: {
    fontSize: SIZES.small,
    fontFamily: FONTS.medium,
    color: "rgba(255,255,255,0.7)",
  },
  metaDot: {
    width: 4,
    height: 4,
    borderRadius: 2,
    backgroundColor: "rgba(255,255,255,0.3)",
  },
  progressTrack: {
    height: 6,
    backgroundColor: "rgba(255,255,255,0.2)",
    borderRadius: 3,
    overflow: "hidden",
  },
  progressFill: { height: "100%", backgroundColor: "#fff", borderRadius: 3 },

  // Weather
  weatherCard: {
    marginHorizontal: SIZES.padding,
    borderRadius: SIZES.radius * 1.2,
    backgroundColor: COLORS.surfaceContainer,
    padding: SIZES.padding * 0.85,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: SIZES.padding,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  weatherLeft: {
    flexDirection: "row",
    alignItems: "center",
    gap: SIZES.padding * 0.6,
    flex: 1,
    marginRight: SIZES.base,
  },
  weatherTextWrap: { flex: 1, flexShrink: 1 },
  weatherIconWrap: {
    width: 42,
    height: 42,
    borderRadius: 12,
    backgroundColor: COLORS.amberContainer,
    justifyContent: "center",
    alignItems: "center",
  },
  weatherCity: { fontSize: 15, fontFamily: FONTS.bold, color: COLORS.white },
  weatherSub: {
    fontSize: SIZES.small,
    fontFamily: FONTS.medium,
    color: COLORS.textMuted,
    marginTop: 2,
  },
  riskBadge: {
    backgroundColor: COLORS.amberContainer,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: SIZES.radiusFull,
    borderWidth: 1,
    borderColor: COLORS.amber + "40",
  },
  riskBadgeText: {
    fontSize: SIZES.tiny,
    fontFamily: FONTS.bold,
    color: COLORS.amber,
    letterSpacing: 0.5,
  },

  alertCard: {
    marginHorizontal: SIZES.padding,
    borderRadius: SIZES.radius * 1.2,
    backgroundColor: COLORS.amberContainer,
    borderWidth: 1,
    borderColor: COLORS.amber + "40",
    padding: SIZES.padding * 0.8,
    marginBottom: SIZES.padding,
  },
  alertCardTop: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 6,
  },
  alertTitle: {
    fontSize: SIZES.small,
    fontFamily: FONTS.bold,
    color: COLORS.amber,
  },
  alertText: {
    fontSize: SIZES.small,
    fontFamily: FONTS.medium,
    color: COLORS.amberDim,
    marginTop: 2,
  },

  // Premium
  premiumCard: {
    marginHorizontal: SIZES.padding,
    borderRadius: SIZES.radius * 1.2,
    backgroundColor: COLORS.surfaceContainer,
    padding: SIZES.padding,
    marginBottom: SIZES.padding,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  premiumCardTop: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: SIZES.padding,
  },
  premiumAmount: { fontSize: 26, fontFamily: FONTS.bold, color: COLORS.white },
  premiumPer: {
    fontSize: SIZES.body,
    fontFamily: FONTS.medium,
    color: COLORS.textMuted,
  },
  premiumSub: {
    fontSize: SIZES.small,
    fontFamily: FONTS.medium,
    color: COLORS.textMuted,
    marginTop: 4,
  },
  payNowBtn: {
    backgroundColor: COLORS.primary,
    paddingHorizontal: 18,
    paddingVertical: 10,
    borderRadius: SIZES.radiusFull,
  },
  payNowText: { fontSize: SIZES.small, fontFamily: FONTS.bold, color: "#fff" },
  premiumBreakdown: {
    flexDirection: "row",
    justifyContent: "space-between",
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
    paddingTop: SIZES.padding * 0.75,
  },
  breakdownItem: { alignItems: "center" },
  breakdownKey: {
    fontSize: SIZES.tiny,
    fontFamily: FONTS.medium,
    color: COLORS.textFaint,
    marginBottom: 4,
  },
  breakdownVal: {
    fontSize: SIZES.small,
    fontFamily: FONTS.bold,
    color: COLORS.white,
  },

  // Quick Actions
  sectionTitle: {
    fontSize: SIZES.body,
    fontFamily: FONTS.bold,
    color: COLORS.white,
    paddingHorizontal: SIZES.padding,
    marginBottom: SIZES.padding,
    marginTop: SIZES.base,
  },
  quickActionsRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingHorizontal: SIZES.padding,
    marginBottom: SIZES.padding,
  },
  quickAction: { alignItems: "center" },
  quickActionIcon: {
    width: 54,
    height: 54,
    borderRadius: 16,
    justifyContent: "center",
    alignItems: "center",
    marginBottom: SIZES.base,
    borderWidth: 1,
    borderColor: COLORS.border,
  },
  quickActionLabel: {
    fontSize: SIZES.tiny,
    fontFamily: FONTS.semiBold,
    color: COLORS.textMuted,
    textAlign: "center",
  },

  // Activity
  activityCard: {
    marginHorizontal: SIZES.padding,
    borderRadius: SIZES.radius * 1.2,
    backgroundColor: COLORS.surfaceContainer,
    borderWidth: 1,
    borderColor: COLORS.border,
    marginBottom: SIZES.padding,
  },
  activityRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: SIZES.padding * 0.75,
    padding: SIZES.padding * 0.85,
  },
  activityBorder: { borderBottomWidth: 1, borderBottomColor: COLORS.border },
  activityIcon: {
    width: 38,
    height: 38,
    borderRadius: 12,
    justifyContent: "center",
    alignItems: "center",
  },
  activityTitle: {
    fontSize: SIZES.small,
    fontFamily: FONTS.semiBold,
    color: COLORS.white,
    marginBottom: 2,
  },
  activitySub: {
    fontSize: SIZES.tiny,
    fontFamily: FONTS.medium,
    color: COLORS.textFaint,
  },

  // Bottom Nav
  bottomNav: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    height: 72,
    backgroundColor: COLORS.surfaceContainer,
    flexDirection: "row",
    justifyContent: "space-around",
    alignItems: "center",
    paddingBottom: Platform.OS === "ios" ? 16 : 0,
    borderTopWidth: 1,
    borderTopColor: COLORS.border,
  },
  navItem: { alignItems: "center", justifyContent: "center" },
  navText: {
    fontSize: SIZES.tiny,
    fontFamily: FONTS.semiBold,
    color: COLORS.textFaint,
    marginTop: 3,
  },
  navActiveDot: {
    width: 4,
    height: 4,
    borderRadius: 2,
    backgroundColor: COLORS.primary,
    marginTop: 2,
  },
});

export default HomeScreen;
