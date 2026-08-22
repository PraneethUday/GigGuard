import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Modal,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
} from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { Ionicons } from "@expo/vector-icons";
import { SIZES, SHADOWS } from "../constants/theme";
import { useTheme } from "../context/ThemeContext";
import { useAuth } from "../context/AuthContext";
import { useLanguage } from "../context/LanguageContext";
import * as api from "../lib/api";

const PAY_STORAGE_KEY_PREFIX = "gg_payments_";

function formatCardNumber(raw) {
  return raw
    .replace(/\D/g, "")
    .slice(0, 16)
    .replace(/(.{4})/g, "$1 ")
    .trim();
}
function formatExpiry(raw) {
  const d = raw.replace(/\D/g, "").slice(0, 4);
  return d.length >= 3 ? `${d.slice(0, 2)}/${d.slice(2)}` : d;
}

const TRIGGER_ICONS = {
  heavy_rain: "rainy",
  severe_aqi: "cloud",
  flood: "water",
  extreme_heat: "thermometer",
  traffic_congestion: "car",
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
  if (typeof risk !== "number") return "—";
  if (risk < 0.15) return "Low Risk";
  if (risk < 0.4) return "Moderate";
  if (risk < 0.7) return "High Risk";
  return "Severe";
}
function ttiLabel(tti) {
  if (typeof tti !== "number") return "—";
  if (tti < 1.5) return "Free Flow";
  if (tti < 2.0) return "Light";
  if (tti < 2.5) return "Moderate";
  if (tti < 3.5) return "Severe";
  return "Gridlock";
}
function ttiColor(tti, COLORS) {
  if (typeof tti !== "number") return COLORS.textMuted;
  if (tti < 2.0) return COLORS.success;
  if (tti < 2.5) return COLORS.amber;
  return COLORS.error;
}
function formatDate(value) {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString("en-IN", { day: "numeric", month: "short" });
}
function formatMetricValue(value, suffix = "") {
  if (value === null || value === undefined || value === "") return "—";
  const numeric = Number(value);
  const display = Number.isFinite(numeric) ? numeric : value;
  return `${display}${suffix}`;
}
function humanizeLabel(value) {
  if (!value || typeof value !== "string") return "—";
  return value.replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();
}
function curfewLabel(confidence) {
  if (typeof confidence !== "number") return "—";
  if (confidence < 0.3) return "No signals";
  if (confidence < 0.6) return "Low activity";
  if (confidence < 0.8) return "Elevated risk";
  return "Active alert";
}
function curfewColor(confidence, COLORS) {
  if (typeof confidence !== "number") return COLORS.textMuted;
  if (confidence < 0.3) return COLORS.success;
  if (confidence < 0.6) return COLORS.amber;
  return COLORS.error;
}

function getStatusTone(status, COLORS) {
  if (status === "paid") return { color: COLORS.success, label: "Paid" };
  if (status === "approved" || status === "pending")
    return { color: COLORS.amber, label: "Processing" };
  if (status === "rejected") return { color: COLORS.error, label: "Rejected" };
  return { color: COLORS.info, label: "Initiated" };
}

const HomeScreen = ({ navigation }) => {
  const { user, token, refreshUser } = useAuth();
  const { COLORS, FONTS } = useTheme();
  const { t } = useLanguage();
  const styles = useMemo(() => createStyles(COLORS, FONTS), [COLORS, FONTS]);

  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");
  const [premium, setPremium] = useState(null);
  const [claims, setClaims] = useState([]);
  const [triggerStatus, setTriggerStatus] = useState({});
  const [trafficLive, setTrafficLive] = useState(null);
  const [curfewLive, setCurfewLive] = useState(null);

  // Chat state
  const [chatOpen, setChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const chatScrollRef = React.useRef(null);

  // Payment modal state
  const [showPayModal, setShowPayModal] = useState(false);
  const [payMethod, setPayMethod] = useState("upi");
  const [payForm, setPayForm] = useState({
    upiId: "",
    cardNumber: "",
    expiry: "",
    cvv: "",
    name: "",
  });
  const [paying, setPaying] = useState(false);
  const [payError, setPayError] = useState("");
  const [paySuccess, setPaySuccess] = useState(null);
  const [paidThisWeek, setPaidThisWeek] = useState(false);

  const storageKey = `${PAY_STORAGE_KEY_PREFIX}${user?.id || "guest"}`;

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
      if (isRefresh) setRefreshing(true);
      else setLoading(true);
      setError("");
      try {
        if (isRefresh) await refreshUser();
        const [premiumRes, claimsRes, triggersRes, trafficRes, curfewRes] =
          await Promise.allSettled([
            api.predictPremium(deliveryId, userCity, userTier),
            api.getWorkerClaims(token, deliveryId),
            api.getTriggerStatus(),
            userCity ? api.getCityTraffic(userCity) : Promise.resolve(null),
            userCity ? api.getCityCurfew(userCity) : Promise.resolve(null),
          ]);
        if (premiumRes.status === "fulfilled") setPremium(premiumRes.value);
        if (claimsRes.status === "fulfilled")
          setClaims(claimsRes.value?.data || []);
        else setClaims([]);
        if (triggersRes.status === "fulfilled")
          setTriggerStatus(triggersRes.value || {});
        if (trafficRes.status === "fulfilled")
          setTrafficLive(trafficRes.value?.traffic || null);
        if (curfewRes.status === "fulfilled")
          setCurfewLive(curfewRes.value?.curfew || null);
        if (premiumRes.status === "rejected" && claimsRes.status === "rejected")
          setError("Unable to load live data right now.");
      } catch {
        setError("Unable to load live data right now.");
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

  // Check paid-this-week from server first, fall back to local cache
  useEffect(() => {
    async function checkPaidThisWeek() {
      const isWithinWeek = (ts) =>
        ts && (new Date() - new Date(ts)) / (1000 * 60 * 60 * 24) < 7;

      // Try server first — this catches payments made on web/other devices
      if (token) {
        try {
          const res = await api.getPaymentHistory(token);
          const serverPayments = res?.payments ?? [];
          // Keep local cache in sync
          AsyncStorage.setItem(
            storageKey,
            JSON.stringify(serverPayments),
          ).catch(() => {});
          const paid = serverPayments.some(
            (p) => p.status === "success" && isWithinWeek(p.timestamp),
          );
          setPaidThisWeek(paid);
          return;
        } catch {
          // Fall through to local cache
        }
      }

      // Offline fallback
      try {
        const raw = await AsyncStorage.getItem(storageKey);
        if (!raw) return;
        const saved = JSON.parse(raw);
        setPaidThisWeek(
          saved.some(
            (p) => p.status === "success" && isWithinWeek(p.timestamp),
          ),
        );
      } catch {}
    }
    checkPaidThisWeek();
  }, [storageKey, token]);

  function openPayModal() {
    setPayForm({ upiId: "", cardNumber: "", expiry: "", cvv: "", name: "" });
    setPayError("");
    setPaySuccess(null);
    setPayMethod("upi");
    setShowPayModal(true);
  }
  function closePayModal() {
    setShowPayModal(false);
    setPaySuccess(null);
    setPayError("");
    setPaying(false);
  }

  function validatePayForm() {
    if (payMethod === "upi") {
      if (!payForm.upiId.trim()) return "Enter your UPI ID.";
      if (!/^[\w.\-+]+@[\w]+$/.test(payForm.upiId.trim()))
        return "Enter a valid UPI ID (e.g. name@upi).";
    } else {
      if (payForm.cardNumber.replace(/\s/g, "").length !== 16)
        return "Enter a valid 16-digit card number.";
      if (!payForm.expiry || payForm.expiry.length < 5)
        return "Enter a valid expiry (MM/YY).";
      if (!payForm.cvv || payForm.cvv.length < 3) return "Enter a valid CVV.";
      if (!payForm.name.trim()) return "Enter the cardholder name.";
    }
    return null;
  }

  async function handlePay() {
    const err = validatePayForm();
    if (err) {
      setPayError(err);
      return;
    }
    setPaying(true);
    setPayError("");
    try {
      const res = await api.payPremium(token, {
        method: payMethod,
        amount: weeklyPremiumAmt,
        tier: user?.tier || "standard",
        worker_id: user?.id,
      });
      const record = {
        transaction_id: res.transaction_id,
        method: payMethod,
        amount: res.amount ?? weeklyPremiumAmt,
        tier: res.tier || user?.tier || "standard",
        status: "success",
        timestamp: res.timestamp || new Date().toISOString(),
      };
      const saved = await AsyncStorage.getItem(storageKey)
        .then((r) => (r ? JSON.parse(r) : []))
        .catch(() => []);
      await AsyncStorage.setItem(
        storageKey,
        JSON.stringify([record, ...saved]),
      );
      setPaidThisWeek(true);
      setPaySuccess(record);
    } catch (e) {
      setPayError(e.message || "Payment failed. Please try again.");
    } finally {
      setPaying(false);
    }
  }

  async function sendChat() {
    const text = chatInput.trim();
    if (!text || chatLoading) return;
    const userMsg = { role: "user", content: text };
    const updated = [...chatMessages, userMsg];
    setChatMessages(updated);
    setChatInput("");
    setChatLoading(true);
    try {
      const data = await api.sendChatMessage(updated);
      const reply = data.reply || "Sorry, I couldn't get a response right now.";
      setChatMessages((prev) => [...prev, { role: "assistant", content: reply }]);
    } catch {
      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Something went wrong. Please try again." },
      ]);
    } finally {
      setChatLoading(false);
      setTimeout(() => chatScrollRef.current?.scrollToEnd({ animated: true }), 100);
    }
  }

  const cityData = useMemo(
    () => (user?.city ? triggerStatus?.[user.city] || null : null),
    [triggerStatus, user?.city],
  );
  const weather = cityData?.weather || premium?.weather || null;
  const trafficData = trafficLive || cityData?.traffic || null;
  const curfewData = curfewLive || null;
  const activeTriggers = cityData?.triggers_fired || [];

  const currentPremium = user?.autopay
    ? premium?.weekly_premium_autopay
    : premium?.weekly_premium;
  const weeklyPremiumAmt = currentPremium || premium?.weekly_premium || 0;
  const coverageActive = user?.verification_status === "verified" && !!premium;
  const paidClaims = claims.filter(
    (c) => c.payout_status === "paid" || c.status === "paid",
  );
  const totalPaid = paidClaims.reduce(
    (s, c) => s + (Number(c.payout_amount) || 0),
    0,
  );

  const recentActivity = useMemo(() => {
    const items = claims.slice(0, 3).map((claim) => {
      const tone = getStatusTone(claim.payout_status || claim.status, COLORS);
      return {
        icon: TRIGGER_ICONS[claim.trigger_type] || "document-text",
        color: tone.color,
        title: claim.trigger_type
          ? claim.trigger_type
              .split("_")
              .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
              .join(" ")
          : "Claim",
        status: tone.label,
        amount: money(Number(claim.payout_amount)),
        date: formatDate(claim.created_at),
      };
    });
    if (items.length > 0) return items;
    return [
      {
        icon: "shield-checkmark",
        color: COLORS.primary,
        title: t("no_claims"),
        status: t("protected_status"),
        amount: "—",
        date: "—",
      },
    ];
  }, [claims, COLORS, t]);

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loader}>
          <ActivityIndicator color={COLORS.primary} size="large" />
          <Text style={styles.loaderText}>{t("loading_dashboard")}</Text>
        </View>
      </SafeAreaView>
    );
  }

  const greeting = (() => {
    const h = new Date().getHours();
    if (h < 12) return t("greeting_morning");
    if (h < 17) return t("greeting_afternoon");
    return t("greeting_evening");
  })();

  const firstName = user?.name?.split(" ")[0] || "there";

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.scroll}
      >
        {/* ── Header ── */}
        <View style={styles.header}>
          <View style={styles.headerLeft}>
            <Text style={styles.greeting}>{greeting},</Text>
            <Text style={styles.userName}>{firstName}</Text>
          </View>
          <TouchableOpacity
            style={styles.refreshBtn}
            onPress={() => loadDashboard(true)}
            disabled={refreshing}
          >
            {refreshing ? (
              <ActivityIndicator color={COLORS.primary} size="small" />
            ) : (
              <Ionicons
                name="refresh-outline"
                size={20}
                color={COLORS.textMuted}
              />
            )}
          </TouchableOpacity>
        </View>

        {/* Platforms pill */}
        {(user?.platforms || []).length > 0 && (
          <View style={styles.pillRow}>
            <View style={styles.platformPill}>
              <Text style={styles.platformPillText}>
                {user.platforms.map((p) => PLATFORM_NAMES[p] || p).join(" · ")}
              </Text>
            </View>
            <View
              style={[
                styles.statusPill,
                {
                  backgroundColor: coverageActive
                    ? COLORS.successContainer
                    : COLORS.errorContainer,
                  borderColor: coverageActive
                    ? COLORS.success + "40"
                    : COLORS.error + "40",
                },
              ]}
            >
              <View
                style={[
                  styles.statusDot,
                  {
                    backgroundColor: coverageActive
                      ? COLORS.success
                      : COLORS.error,
                  },
                ]}
              />
              <Text
                style={[
                  styles.statusPillText,
                  { color: coverageActive ? COLORS.success : COLORS.error },
                ]}
              >
                {coverageActive ? t("protected_status") : t("inactive_status")}
              </Text>
            </View>
          </View>
        )}

        {!!error && (
          <View style={styles.errorCard}>
            <Ionicons name="warning-outline" size={15} color={COLORS.error} />
            <Text style={styles.errorCardText}>{error}</Text>
          </View>
        )}

        {/* ── Coverage Hero ── */}
        <View style={styles.heroCard}>
          <View style={styles.heroGlow} />
          <View style={styles.heroTop}>
            <View>
              <Text style={styles.heroLabel}>{t("this_weeks_coverage")}</Text>
              <Text style={styles.heroAmount}>
                {money(premium?.max_payout)}
              </Text>
              <Text style={styles.heroSub}>
                {t("max_payout_label")} · {titleCaseTier(user?.tier)}{" "}
                {t("plan_suffix")}
              </Text>
            </View>
            <View style={styles.heroShield}>
              <Ionicons
                name="shield-checkmark"
                size={32}
                color={coverageActive ? COLORS.primary : COLORS.textFaint}
              />
            </View>
          </View>
          <View style={styles.heroDivider} />
          <View style={styles.heroBottom}>
            <View style={styles.heroStat}>
              <Text style={styles.heroStatVal}>{money(currentPremium)}</Text>
              <Text style={styles.heroStatKey}>{t("weekly_premium")}</Text>
            </View>
            <View style={styles.heroStatDivider} />
            <View style={styles.heroStat}>
              <Text style={styles.heroStatVal}>{money(totalPaid)}</Text>
              <Text style={styles.heroStatKey}>{t("total_received")}</Text>
            </View>
            <View style={styles.heroStatDivider} />
            <View style={styles.heroStat}>
              <Text style={styles.heroStatVal}>{paidClaims.length}</Text>
              <Text style={styles.heroStatKey}>{t("claims_paid")}</Text>
            </View>
          </View>
        </View>

        {/* ── Pay Premium Card ── */}
        <View style={styles.payCard}>
          <View style={styles.payCardLeft}>
            <Text style={styles.payCardLabel}>{t("weekly_premium_label")}</Text>
            <Text style={styles.payCardAmount}>{money(weeklyPremiumAmt)}</Text>
            <Text style={styles.payCardSub}>
              {user?.autopay
                ? t("autopay_discount")
                : t("pay_to_keep_coverage")}
            </Text>
          </View>
          {paidThisWeek ? (
            <View style={styles.paidBadge}>
              <Ionicons
                name="checkmark-circle"
                size={15}
                color={COLORS.success}
              />
              <Text style={styles.paidBadgeText}>{t("paid")}</Text>
            </View>
          ) : (
            <TouchableOpacity
              style={styles.payNowBtn}
              onPress={openPayModal}
              activeOpacity={0.85}
            >
              <Text style={styles.payNowText}>{t("pay_now")}</Text>
            </TouchableOpacity>
          )}
        </View>

        {/* ── Active Disruption Alert ── */}
        {activeTriggers.length > 0 && (
          <View style={styles.alertCard}>
            <View style={styles.alertLeft}>
              <Ionicons name="flash" size={18} color={COLORS.amber} />
              <View style={{ flex: 1 }}>
                <Text style={styles.alertTitle}>
                  Disruption Active · {user?.city}
                </Text>
                <Text style={styles.alertSub}>
                  {activeTriggers[0].description}
                </Text>
              </View>
            </View>
            <View
              style={[styles.alertDot, { backgroundColor: COLORS.amber }]}
            />
          </View>
        )}

        {/* ── Live Conditions Card ── */}
        <View style={styles.envCard}>
          {/* Card header */}
          <View style={styles.envCardHeader}>
            <Ionicons
              name="analytics-outline"
              size={14}
              color={COLORS.primary}
            />
            <Text style={styles.envCardTitle}>Live Conditions</Text>
            <Text style={styles.envCardCity}>{user?.city || "—"}</Text>
          </View>

          {/* ── Weather row ── */}
          <View style={styles.envRow}>
            <View
              style={[
                styles.envIconWrap,
                { backgroundColor: COLORS.primaryContainer },
              ]}
            >
              <Ionicons
                name="partly-sunny-outline"
                size={18}
                color={COLORS.primary}
              />
            </View>
            <View style={styles.envRowBody}>
              <View style={styles.envRowTitleBar}>
                <Text style={styles.envTitle}>Weather</Text>
                <View style={styles.riskPill}>
                  <Text style={styles.riskPillText}>
                    {riskLabel(premium?.weather_risk)}
                  </Text>
                </View>
              </View>
              {weather ? (
                <>
                  <View style={styles.weatherGrid}>
                    <View style={styles.weatherMetric}>
                      <Text style={styles.weatherMetricLabel}>Temperature</Text>
                      <Text style={styles.weatherMetricValue}>
                        {formatMetricValue(weather.temperature, "°C")}
                      </Text>
                    </View>
                    <View style={styles.weatherMetric}>
                      <Text style={styles.weatherMetricLabel}>AQI</Text>
                      <Text style={styles.weatherMetricValue}>
                        {formatMetricValue(weather.aqi_index)}
                      </Text>
                    </View>
                    <View style={styles.weatherMetric}>
                      <Text style={styles.weatherMetricLabel}>Rainfall</Text>
                      <Text style={styles.weatherMetricValue}>
                        {formatMetricValue(weather.rain_1h ?? 0, " mm/h")}
                      </Text>
                    </View>
                    <View style={styles.weatherMetric}>
                      <Text style={styles.weatherMetricLabel}>Humidity</Text>
                      <Text style={styles.weatherMetricValue}>
                        {formatMetricValue(weather.humidity, "%")}
                      </Text>
                    </View>
                  </View>
                  <View style={styles.weatherConditionChip}>
                    <Ionicons
                      name="cloud-outline"
                      size={13}
                      color={COLORS.primary}
                    />
                    <Text style={styles.weatherConditionText}>
                      {weather.weather_main || "Condition unavailable"}
                    </Text>
                  </View>
                </>
              ) : (
                <Text style={styles.envSub}>{t("weather_unavailable")}</Text>
              )}
            </View>
          </View>

          {/* ── Traffic row ── */}
          {trafficData ? (
            <>
              <View style={styles.envSeparator} />
              <View style={styles.envRow}>
                <View
                  style={[
                    styles.envIconWrap,
                    {
                      backgroundColor: ttiColor(trafficData.tti, COLORS) + "18",
                    },
                  ]}
                >
                  <Ionicons
                    name="car-outline"
                    size={18}
                    color={ttiColor(trafficData.tti, COLORS)}
                  />
                </View>
                <View style={styles.envRowBody}>
                  <View style={styles.envRowTitleBar}>
                    <Text style={styles.envTitle}>Traffic</Text>
                    <View
                      style={[
                        styles.riskPill,
                        {
                          backgroundColor:
                            ttiColor(trafficData.tti, COLORS) + "18",
                          borderColor: ttiColor(trafficData.tti, COLORS) + "40",
                        },
                      ]}
                    >
                      <Text
                        style={[
                          styles.riskPillText,
                          { color: ttiColor(trafficData.tti, COLORS) },
                        ]}
                      >
                        {ttiLabel(trafficData.tti)}
                      </Text>
                    </View>
                  </View>
                  <View style={styles.chipWrap}>
                    <View style={styles.chip}>
                      <Text style={styles.chipText}>
                        TTI {(trafficData.tti || 1).toFixed(2)}
                      </Text>
                    </View>
                    <View style={styles.chip}>
                      <Text style={styles.chipText}>
                        Speed {Math.round(trafficData.current_speed_kmh || 0)}{" "}
                        km/h
                      </Text>
                    </View>
                    <View style={styles.chip}>
                      <Text style={styles.chipText}>
                        Free flow{" "}
                        {Math.round(trafficData.free_flow_speed_kmh || 0)} km/h
                      </Text>
                    </View>
                    <View style={styles.chip}>
                      <Text style={styles.chipText}>
                        Risk {Math.round((trafficData.traffic_risk || 0) * 100)}
                        %
                      </Text>
                    </View>
                    {trafficData.road_closure && (
                      <View
                        style={[
                          styles.chip,
                          {
                            backgroundColor: COLORS.errorContainer,
                            borderColor: COLORS.error + "40",
                          },
                        ]}
                      >
                        <Text
                          style={[styles.chipText, { color: COLORS.error }]}
                        >
                          Road closure
                        </Text>
                      </View>
                    )}
                    {trafficData.source ? (
                      <View style={styles.chip}>
                        <Text
                          style={[styles.chipText, { color: COLORS.textFaint }]}
                        >
                          {humanizeLabel(trafficData.source)}
                        </Text>
                      </View>
                    ) : null}
                  </View>
                </View>
              </View>
            </>
          ) : null}

          {/* ── Curfew / Unrest row ── */}
          {curfewData ? (
            <>
              <View style={styles.envSeparator} />
              <View style={styles.envRow}>
                <View
                  style={[
                    styles.envIconWrap,
                    {
                      backgroundColor:
                        curfewColor(curfewData.confidence, COLORS) + "18",
                    },
                  ]}
                >
                  <Ionicons
                    name="shield-outline"
                    size={18}
                    color={curfewColor(curfewData.confidence, COLORS)}
                  />
                </View>
                <View style={styles.envRowBody}>
                  <View style={styles.envRowTitleBar}>
                    <Text style={styles.envTitle}>Unrest / Curfew</Text>
                    <View
                      style={[
                        styles.riskPill,
                        {
                          backgroundColor:
                            curfewColor(curfewData.confidence, COLORS) + "18",
                          borderColor:
                            curfewColor(curfewData.confidence, COLORS) + "40",
                        },
                      ]}
                    >
                      <Text
                        style={[
                          styles.riskPillText,
                          { color: curfewColor(curfewData.confidence, COLORS) },
                        ]}
                      >
                        {curfewLabel(curfewData.confidence)}
                      </Text>
                    </View>
                  </View>
                  <View style={styles.chipWrap}>
                    <View style={styles.chip}>
                      <Text style={styles.chipText}>
                        Confidence{" "}
                        {Math.round((curfewData.confidence || 0) * 100)}%
                      </Text>
                    </View>
                    <View style={styles.chip}>
                      <Text style={styles.chipText}>
                        GDELT {curfewData.gdelt_events ?? 0} events
                      </Text>
                    </View>
                    {curfewData.nlp_score > 0 && (
                      <View style={styles.chip}>
                        <Text style={styles.chipText}>
                          NLP {humanizeLabel(curfewData.nlp_label)} (
                          {Math.round(curfewData.nlp_score * 100)}%)
                        </Text>
                      </View>
                    )}
                    {curfewData.fired && (
                      <View
                        style={[
                          styles.chip,
                          {
                            backgroundColor: COLORS.errorContainer,
                            borderColor: COLORS.error + "40",
                          },
                        ]}
                      >
                        <Text
                          style={[styles.chipText, { color: COLORS.error }]}
                        >
                          Trigger active
                        </Text>
                      </View>
                    )}
                    {curfewData.source ? (
                      <View style={styles.chip}>
                        <Text
                          style={[styles.chipText, { color: COLORS.textFaint }]}
                        >
                          {humanizeLabel(curfewData.source)}
                        </Text>
                      </View>
                    ) : null}
                  </View>
                </View>
              </View>
            </>
          ) : null}
        </View>

        {/* ── Quick Actions ── */}
        <Text style={styles.sectionLabel}>{t("quick_actions")}</Text>
        <View style={styles.actionsGrid}>
          {[
            {
              icon: "shield-outline",
              label: t("my_policy"),
              route: "Policy",
              color: COLORS.primary,
            },
            {
              icon: "document-text-outline",
              label: t("nav_claims"),
              route: "Claims",
              color: COLORS.amber,
            },
            {
              icon: "card-outline",
              label: t("nav_payments"),
              route: "Payments",
              color: COLORS.success,
            },
            {
              icon: "person-outline",
              label: t("nav_profile"),
              route: "Profile",
              color: COLORS.info,
            },
          ].map((action) => (
            <TouchableOpacity
              key={action.label}
              style={styles.actionItem}
              onPress={() => navigation.navigate(action.route)}
              activeOpacity={0.7}
            >
              <View
                style={[
                  styles.actionIcon,
                  {
                    backgroundColor: action.color + "15",
                    borderColor: action.color + "30",
                  },
                ]}
              >
                <Ionicons name={action.icon} size={22} color={action.color} />
              </View>
              <Text style={styles.actionLabel}>{action.label}</Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* ── Recent Activity ── */}
        <Text style={styles.sectionLabel}>{t("recent_activity")}</Text>
        <View style={styles.activityCard}>
          {recentActivity.map((item, i) => (
            <View
              key={`activity-${i}`}
              style={[
                styles.activityRow,
                i < recentActivity.length - 1 && {
                  borderBottomWidth: 1,
                  borderBottomColor: COLORS.border,
                },
              ]}
            >
              <View
                style={[
                  styles.activityIconWrap,
                  { backgroundColor: item.color + "15" },
                ]}
              >
                <Ionicons name={item.icon} size={17} color={item.color} />
              </View>
              <View style={styles.activityMeta}>
                <Text style={styles.activityTitle}>{item.title}</Text>
                <Text style={styles.activityDate}>{item.date}</Text>
              </View>
              <View style={styles.activityRight}>
                <Text style={[styles.activityAmount, { color: item.color }]}>
                  {item.amount}
                </Text>
                <Text style={[styles.activityStatus, { color: item.color }]}>
                  {item.status}
                </Text>
              </View>
            </View>
          ))}
        </View>
      </ScrollView>

      {/* ── Payment Modal ── */}
      <Modal
        visible={showPayModal}
        transparent
        animationType="slide"
        onRequestClose={closePayModal}
      >
        <KeyboardAvoidingView
          style={styles.modalOverlay}
          behavior={Platform.OS === "ios" ? "padding" : "height"}
        >
          <View style={styles.modalSheet}>
            {paySuccess ? (
              <View style={styles.successWrap}>
                <View style={styles.successCircle}>
                  <Ionicons name="checkmark" size={36} color={COLORS.success} />
                </View>
                <Text style={styles.successTitle}>
                  {t("payment_successful")}
                </Text>
                <Text style={styles.successTxId}>
                  {paySuccess.transaction_id}
                </Text>
                <Text style={styles.successAmt}>
                  {money(paySuccess.amount)} {t("paid")}
                </Text>
                <TouchableOpacity
                  style={styles.doneBtn}
                  onPress={closePayModal}
                >
                  <Text style={styles.doneBtnText}>{t("done")}</Text>
                </TouchableOpacity>
              </View>
            ) : (
              <>
                <View style={styles.modalHandle} />
                <View style={styles.modalTop}>
                  <View>
                    <Text style={styles.modalTitle}>{t("pay_premium")}</Text>
                    <Text style={styles.modalAmt}>
                      {money(weeklyPremiumAmt)}
                    </Text>
                  </View>
                  <TouchableOpacity
                    onPress={closePayModal}
                    style={styles.modalCloseBtn}
                  >
                    <Ionicons name="close" size={18} color={COLORS.textMuted} />
                  </TouchableOpacity>
                </View>

                {/* Method selector */}
                <View style={styles.methodRow}>
                  {[
                    { id: "upi", label: "UPI", icon: "phone-portrait-outline" },
                    { id: "debit", label: "Debit Card", icon: "card-outline" },
                    {
                      id: "credit",
                      label: "Credit Card",
                      icon: "wallet-outline",
                    },
                  ].map((m) => (
                    <TouchableOpacity
                      key={m.id}
                      style={[
                        styles.methodChip,
                        payMethod === m.id && styles.methodChipActive,
                      ]}
                      onPress={() => {
                        setPayMethod(m.id);
                        setPayError("");
                      }}
                    >
                      <Ionicons
                        name={m.icon}
                        size={14}
                        color={
                          payMethod === m.id ? COLORS.primary : COLORS.textFaint
                        }
                      />
                      <Text
                        style={[
                          styles.methodChipText,
                          payMethod === m.id && { color: COLORS.primary },
                        ]}
                      >
                        {m.label}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>

                {/* Form fields */}
                {payMethod === "upi" ? (
                  <View style={styles.field}>
                    <Text style={styles.fieldLabel}>UPI ID</Text>
                    <TextInput
                      style={styles.fieldInput}
                      placeholder="yourname@upi"
                      placeholderTextColor={COLORS.textFaint}
                      value={payForm.upiId}
                      onChangeText={(t) =>
                        setPayForm((f) => ({ ...f, upiId: t }))
                      }
                      autoCapitalize="none"
                      keyboardType="email-address"
                    />
                  </View>
                ) : (
                  <>
                    <View style={styles.field}>
                      <Text style={styles.fieldLabel}>Cardholder Name</Text>
                      <TextInput
                        style={styles.fieldInput}
                        placeholder="Name on card"
                        placeholderTextColor={COLORS.textFaint}
                        value={payForm.name}
                        onChangeText={(t) =>
                          setPayForm((f) => ({ ...f, name: t }))
                        }
                        autoCapitalize="words"
                      />
                    </View>
                    <View style={styles.field}>
                      <Text style={styles.fieldLabel}>Card Number</Text>
                      <TextInput
                        style={styles.fieldInput}
                        placeholder="0000 0000 0000 0000"
                        placeholderTextColor={COLORS.textFaint}
                        value={payForm.cardNumber}
                        onChangeText={(t) =>
                          setPayForm((f) => ({
                            ...f,
                            cardNumber: formatCardNumber(t),
                          }))
                        }
                        keyboardType="numeric"
                        maxLength={19}
                      />
                    </View>
                    <View style={{ flexDirection: "row", gap: 12 }}>
                      <View style={[styles.field, { flex: 1 }]}>
                        <Text style={styles.fieldLabel}>Expiry</Text>
                        <TextInput
                          style={styles.fieldInput}
                          placeholder="MM/YY"
                          placeholderTextColor={COLORS.textFaint}
                          value={payForm.expiry}
                          onChangeText={(t) =>
                            setPayForm((f) => ({
                              ...f,
                              expiry: formatExpiry(t),
                            }))
                          }
                          keyboardType="numeric"
                          maxLength={5}
                        />
                      </View>
                      <View style={[styles.field, { flex: 1 }]}>
                        <Text style={styles.fieldLabel}>CVV</Text>
                        <TextInput
                          style={styles.fieldInput}
                          placeholder="•••"
                          placeholderTextColor={COLORS.textFaint}
                          value={payForm.cvv}
                          onChangeText={(t) =>
                            setPayForm((f) => ({
                              ...f,
                              cvv: t.replace(/\D/g, "").slice(0, 4),
                            }))
                          }
                          keyboardType="numeric"
                          secureTextEntry
                          maxLength={4}
                        />
                      </View>
                    </View>
                  </>
                )}

                {!!payError && (
                  <View style={styles.payErrorBox}>
                    <Ionicons
                      name="warning-outline"
                      size={14}
                      color={COLORS.error}
                    />
                    <Text style={styles.payErrorText}>{payError}</Text>
                  </View>
                )}

                <TouchableOpacity
                  style={[styles.submitBtn, paying && { opacity: 0.65 }]}
                  onPress={handlePay}
                  disabled={paying}
                  activeOpacity={0.85}
                >
                  {paying ? (
                    <ActivityIndicator color="#fff" size="small" />
                  ) : (
                    <>
                      <Ionicons name="lock-closed" size={15} color="#fff" />
                      <Text style={styles.submitBtnText}>
                        {t("pay_now")} {money(weeklyPremiumAmt)}
                      </Text>
                    </>
                  )}
                </TouchableOpacity>
              </>
            )}
          </View>
        </KeyboardAvoidingView>
      </Modal>

      {/* ── Bottom Nav ── */}
      <View style={styles.bottomNav}>
        {/* Home */}
        <TouchableOpacity style={styles.navItem}>
          <View style={styles.navActiveWrap}>
            <Ionicons name="home" size={22} color={COLORS.primary} />
          </View>
          <Text style={[styles.navLabel, { color: COLORS.primary }]}>{t("nav_home")}</Text>
        </TouchableOpacity>

        {/* Claims */}
        <TouchableOpacity style={styles.navItem} onPress={() => navigation.navigate("Claims")}>
          <Ionicons name="document-outline" size={22} color={COLORS.textFaint} />
          <Text style={styles.navLabel}>{t("nav_claims")}</Text>
        </TouchableOpacity>

        {/* Center AI Chat button */}
        <View style={styles.navChatWrap}>
          <TouchableOpacity style={styles.navChatBtn} onPress={() => setChatOpen(true)} activeOpacity={0.85}>
            <Ionicons name={chatOpen ? "close" : "chatbubble-ellipses"} size={24} color="#fff" />
          </TouchableOpacity>
          <Text style={styles.navChatLabel}>AI Chat</Text>
        </View>

        {/* Policy */}
        <TouchableOpacity style={styles.navItem} onPress={() => navigation.navigate("Policy")}>
          <Ionicons name="shield-outline" size={22} color={COLORS.textFaint} />
          <Text style={styles.navLabel}>{t("nav_policy")}</Text>
        </TouchableOpacity>

        {/* Profile */}
        <TouchableOpacity style={styles.navItem} onPress={() => navigation.navigate("Profile")}>
          <Ionicons name="person-outline" size={22} color={COLORS.textFaint} />
          <Text style={styles.navLabel}>{t("nav_profile")}</Text>
        </TouchableOpacity>
      </View>

      {/* ── AI Chat Modal ── */}
      <Modal
        visible={chatOpen}
        animationType="slide"
        transparent
        onRequestClose={() => setChatOpen(false)}
      >
        <KeyboardAvoidingView
          style={styles.chatOverlay}
          behavior={Platform.OS === "ios" ? "padding" : "height"}
        >
          <TouchableOpacity style={styles.chatBackdrop} activeOpacity={1} onPress={() => setChatOpen(false)} />
          <View style={styles.chatSheet}>
            {/* Header */}
            <View style={styles.chatHeader}>
              <View style={styles.chatHeaderLeft}>
                <View style={styles.chatAvatar}>
                  <Text style={styles.chatAvatarText}>AI</Text>
                </View>
                <View>
                  <Text style={styles.chatTitle}>WPIP Assistant</Text>
                  <Text style={styles.chatSubtitle}>Ask me anything about your coverage</Text>
                </View>
              </View>
              <TouchableOpacity onPress={() => setChatOpen(false)} style={styles.chatCloseBtn}>
                <Ionicons name="close" size={20} color={COLORS.textFaint} />
              </TouchableOpacity>
            </View>

            {/* Messages */}
            <ScrollView
              ref={chatScrollRef}
              style={styles.chatBody}
              contentContainerStyle={styles.chatBodyContent}
              showsVerticalScrollIndicator={false}
              onContentSizeChange={() => chatScrollRef.current?.scrollToEnd({ animated: true })}
            >
              {chatMessages.length === 0 && (
                <View style={styles.chatEmpty}>
                  <View style={styles.chatEmptyIcon}>
                    <Ionicons name="chatbubble-ellipses-outline" size={32} color={COLORS.primary} />
                  </View>
                  <Text style={styles.chatEmptyTitle}>Hi! I&apos;m your WPIP assistant.</Text>
                  <Text style={styles.chatEmptyHint}>Ask me about claims, premiums, coverage, or how the platform works.</Text>
                  <View style={styles.chatSuggestions}>
                    {["How do claims work?", "What does Standard cover?", "When will I get paid?"].map((s) => (
                      <TouchableOpacity key={s} style={styles.chatSuggestion} onPress={() => setChatInput(s)}>
                        <Text style={styles.chatSuggestionText}>{s}</Text>
                      </TouchableOpacity>
                    ))}
                  </View>
                </View>
              )}
              {chatMessages.map((msg, i) => (
                <View key={i} style={[styles.chatBubbleRow, msg.role === "user" && styles.chatBubbleRowUser]}>
                  {msg.role === "assistant" && (
                    <View style={styles.chatBubbleAvatar}>
                      <Text style={styles.chatBubbleAvatarText}>AI</Text>
                    </View>
                  )}
                  <View style={[styles.chatBubble, msg.role === "user" ? styles.chatBubbleUser : styles.chatBubbleAssistant]}>
                    <Text style={[styles.chatBubbleText, msg.role === "user" && styles.chatBubbleTextUser]}>{msg.content}</Text>
                  </View>
                </View>
              ))}
              {chatLoading && (
                <View style={styles.chatBubbleRow}>
                  <View style={styles.chatBubbleAvatar}>
                    <Text style={styles.chatBubbleAvatarText}>AI</Text>
                  </View>
                  <View style={[styles.chatBubble, styles.chatBubbleAssistant]}>
                    <ActivityIndicator size="small" color={COLORS.textFaint} />
                  </View>
                </View>
              )}
            </ScrollView>

            {/* Input */}
            <View style={styles.chatInputRow}>
              <TextInput
                style={styles.chatInput}
                placeholder="Ask anything…"
                placeholderTextColor={COLORS.textFaint}
                value={chatInput}
                onChangeText={setChatInput}
                onSubmitEditing={sendChat}
                returnKeyType="send"
                editable={!chatLoading}
                multiline={false}
              />
              <TouchableOpacity
                style={[styles.chatSendBtn, (!chatInput.trim() || chatLoading) && styles.chatSendBtnDisabled]}
                onPress={sendChat}
                disabled={!chatInput.trim() || chatLoading}
                activeOpacity={0.8}
              >
                <Ionicons name="send" size={16} color="#fff" />
              </TouchableOpacity>
            </View>
          </View>
        </KeyboardAvoidingView>
      </Modal>
    </SafeAreaView>
  );
};

const createStyles = (COLORS, FONTS) =>
  StyleSheet.create({
    container: { flex: 1, backgroundColor: COLORS.surface },
    scroll: { paddingBottom: 88 },
    loader: {
      flex: 1,
      alignItems: "center",
      justifyContent: "center",
      gap: 14,
    },
    loaderText: {
      fontSize: SIZES.small,
      fontFamily: FONTS.medium,
      color: COLORS.textMuted,
    },

    // ── Header ───────────────────────────────────────────────
    header: {
      flexDirection: "row",
      justifyContent: "space-between",
      alignItems: "flex-start",
      paddingHorizontal: SIZES.padding,
      paddingTop: SIZES.padding,
      paddingBottom: SIZES.base,
    },
    headerLeft: { gap: 2 },
    greeting: {
      fontSize: SIZES.small,
      fontFamily: FONTS.regular,
      color: COLORS.textFaint,
    },
    userName: {
      fontSize: SIZES.h2,
      fontFamily: FONTS.display,
      color: COLORS.white,
    },
    refreshBtn: {
      width: 38,
      height: 38,
      borderRadius: 12,
      backgroundColor: COLORS.surfaceContainer,
      borderWidth: 1,
      borderColor: COLORS.border,
      justifyContent: "center",
      alignItems: "center",
      marginTop: 4,
    },

    pillRow: {
      flexDirection: "row",
      gap: 8,
      alignItems: "center",
      paddingHorizontal: SIZES.padding,
      marginBottom: SIZES.padding * 0.85,
    },
    platformPill: {
      flexDirection: "row",
      alignItems: "center",
      gap: 5,
      backgroundColor: COLORS.primaryContainer,
      borderWidth: 1,
      borderColor: COLORS.borderActive,
      paddingHorizontal: 10,
      paddingVertical: 5,
      borderRadius: SIZES.radiusFull,
    },
    platformPillText: {
      fontSize: SIZES.tiny,
      fontFamily: FONTS.semiBold,
      color: COLORS.primary,
    },
    statusPill: {
      flexDirection: "row",
      alignItems: "center",
      gap: 5,
      paddingHorizontal: 10,
      paddingVertical: 5,
      borderRadius: SIZES.radiusFull,
      borderWidth: 1,
    },
    statusDot: { width: 6, height: 6, borderRadius: 3 },
    statusPillText: { fontSize: SIZES.tiny, fontFamily: FONTS.semiBold },

    errorCard: {
      flexDirection: "row",
      alignItems: "center",
      gap: 8,
      marginHorizontal: SIZES.padding,
      marginBottom: SIZES.padding * 0.75,
      backgroundColor: COLORS.errorContainer,
      borderWidth: 1,
      borderColor: COLORS.error + "35",
      borderRadius: SIZES.radius,
      padding: 10,
    },
    errorCardText: {
      flex: 1,
      fontSize: SIZES.small,
      fontFamily: FONTS.medium,
      color: COLORS.error,
    },

    // ── Hero Card ────────────────────────────────────────────
    heroCard: {
      marginHorizontal: SIZES.padding,
      borderRadius: SIZES.radius * 1.5,
      backgroundColor: COLORS.surfaceContainer,
      borderWidth: 1,
      borderColor: COLORS.border,
      marginBottom: SIZES.padding,
      overflow: "hidden",
      ...SHADOWS.card,
    },
    heroGlow: {
      position: "absolute",
      width: 220,
      height: 220,
      borderRadius: 110,
      backgroundColor: COLORS.primary,
      opacity: 0.06,
      top: -60,
      right: -60,
    },
    heroTop: {
      flexDirection: "row",
      justifyContent: "space-between",
      alignItems: "flex-start",
      padding: SIZES.padding,
    },
    heroLabel: {
      fontSize: 10,
      fontFamily: FONTS.bold,
      color: COLORS.textFaint,
      letterSpacing: 1.2,
      textTransform: "uppercase",
      marginBottom: 8,
    },
    heroAmount: {
      fontSize: 36,
      fontFamily: FONTS.display,
      color: COLORS.white,
      lineHeight: 40,
    },
    heroSub: {
      fontSize: SIZES.small,
      fontFamily: FONTS.regular,
      color: COLORS.textFaint,
      marginTop: 4,
    },
    heroShield: {
      width: 56,
      height: 56,
      borderRadius: SIZES.radius,
      backgroundColor: COLORS.surfaceHigh,
      borderWidth: 1,
      borderColor: COLORS.border,
      justifyContent: "center",
      alignItems: "center",
    },
    heroDivider: {
      height: 1,
      backgroundColor: COLORS.border,
      marginHorizontal: SIZES.padding,
    },
    heroBottom: {
      flexDirection: "row",
      padding: SIZES.padding,
      paddingVertical: SIZES.padding * 0.85,
    },
    heroStat: { flex: 1, alignItems: "center" },
    heroStatVal: {
      fontSize: SIZES.body,
      fontFamily: FONTS.bold,
      color: COLORS.white,
      marginBottom: 3,
    },
    heroStatKey: {
      fontSize: SIZES.tiny,
      fontFamily: FONTS.regular,
      color: COLORS.textFaint,
    },
    heroStatDivider: { width: 1, backgroundColor: COLORS.border },

    // ── Alert Card ───────────────────────────────────────────
    alertCard: {
      marginHorizontal: SIZES.padding,
      borderRadius: SIZES.radius,
      backgroundColor: COLORS.amberContainer,
      borderWidth: 1,
      borderColor: COLORS.amber + "40",
      padding: SIZES.padding * 0.85,
      marginBottom: SIZES.padding,
      flexDirection: "row",
      alignItems: "center",
      justifyContent: "space-between",
    },
    alertLeft: { flexDirection: "row", alignItems: "center", gap: 10, flex: 1 },
    alertTitle: {
      fontSize: SIZES.small,
      fontFamily: FONTS.bold,
      color: COLORS.amber,
      marginBottom: 2,
    },
    alertSub: {
      fontSize: SIZES.tiny,
      fontFamily: FONTS.regular,
      color: COLORS.amberDim,
    },
    alertDot: { width: 8, height: 8, borderRadius: 4, opacity: 0.8 },

    // ── Environment / Live Conditions card ───────────────────
    envCard: {
      marginHorizontal: SIZES.padding,
      borderRadius: SIZES.radius * 1.2,
      backgroundColor: COLORS.surfaceContainer,
      borderWidth: 1,
      borderColor: COLORS.border,
      marginBottom: SIZES.padding,
      overflow: "hidden",
    },
    envCardHeader: {
      flexDirection: "row",
      alignItems: "center",
      gap: 6,
      paddingHorizontal: SIZES.padding,
      paddingTop: SIZES.padding * 0.75,
      paddingBottom: 0,
    },
    envCardTitle: {
      fontSize: 10,
      fontFamily: FONTS.bold,
      color: COLORS.primary,
      textTransform: "uppercase",
      letterSpacing: 1,
      flex: 1,
    },
    envCardCity: {
      fontSize: 10,
      fontFamily: FONTS.medium,
      color: COLORS.textFaint,
    },
    envRow: {
      flexDirection: "row",
      alignItems: "flex-start",
      gap: 12,
      padding: SIZES.padding * 0.85,
    },
    envRowBody: { flex: 1 },
    envRowTitleBar: {
      flexDirection: "row",
      alignItems: "center",
      justifyContent: "space-between",
      marginBottom: 8,
    },
    envSeparator: { height: 1, backgroundColor: COLORS.border },
    envIconWrap: {
      width: 36,
      height: 36,
      borderRadius: 10,
      backgroundColor: COLORS.surfaceHigh,
      justifyContent: "center",
      alignItems: "center",
      flexShrink: 0,
      marginTop: 2,
    },
    // keep legacy aliases so nothing else breaks
    envTextWrap: { flex: 1 },
    envTitle: {
      fontSize: SIZES.small,
      fontFamily: FONTS.semiBold,
      color: COLORS.white,
    },
    envSub: {
      fontSize: SIZES.tiny,
      fontFamily: FONTS.regular,
      color: COLORS.textFaint,
    },
    riskPill: {
      paddingHorizontal: 9,
      paddingVertical: 3,
      borderRadius: SIZES.radiusFull,
      backgroundColor: COLORS.primaryContainer,
      borderWidth: 1,
      borderColor: COLORS.borderActive,
    },
    riskPillText: {
      fontSize: 10,
      fontFamily: FONTS.bold,
      color: COLORS.primary,
      letterSpacing: 0.3,
    },

    // data chips
    chipWrap: { flexDirection: "row", flexWrap: "wrap", gap: 5 },
    chip: {
      paddingHorizontal: 8,
      paddingVertical: 4,
      borderRadius: SIZES.radiusFull,
      backgroundColor: COLORS.surfaceHigh,
      borderWidth: 1,
      borderColor: COLORS.border,
    },
    chipText: {
      fontSize: 11,
      fontFamily: FONTS.medium,
      color: COLORS.textMuted,
    },
    weatherGrid: { flexDirection: "row", flexWrap: "wrap", gap: 8 },
    weatherMetric: {
      flexBasis: "48%",
      flexGrow: 1,
      backgroundColor: COLORS.surfaceHigh,
      borderWidth: 1,
      borderColor: COLORS.border,
      borderRadius: SIZES.radius,
      paddingHorizontal: 10,
      paddingVertical: 8,
    },
    weatherMetricLabel: {
      fontSize: 10,
      fontFamily: FONTS.semiBold,
      color: COLORS.textFaint,
      marginBottom: 4,
      textTransform: "uppercase",
      letterSpacing: 0.4,
    },
    weatherMetricValue: {
      fontSize: SIZES.small,
      fontFamily: FONTS.bold,
      color: COLORS.white,
    },
    weatherConditionChip: {
      flexDirection: "row",
      alignItems: "center",
      gap: 6,
      alignSelf: "flex-start",
      marginTop: 8,
      paddingHorizontal: 10,
      paddingVertical: 5,
      borderRadius: SIZES.radiusFull,
      backgroundColor: COLORS.primaryContainer,
      borderWidth: 1,
      borderColor: COLORS.borderActive,
    },
    weatherConditionText: {
      fontSize: 11,
      fontFamily: FONTS.semiBold,
      color: COLORS.primary,
    },

    // ── Quick Actions ────────────────────────────────────────
    sectionLabel: {
      fontSize: SIZES.small,
      fontFamily: FONTS.bold,
      color: COLORS.textMuted,
      textTransform: "uppercase",
      letterSpacing: 0.8,
      paddingHorizontal: SIZES.padding,
      marginBottom: SIZES.padding * 0.75,
    },
    actionsGrid: {
      flexDirection: "row",
      justifyContent: "space-between",
      paddingHorizontal: SIZES.padding,
      marginBottom: SIZES.padding * 1.25,
    },
    actionItem: { alignItems: "center", width: "22%" },
    actionIcon: {
      width: 52,
      height: 52,
      borderRadius: SIZES.radius,
      justifyContent: "center",
      alignItems: "center",
      marginBottom: 7,
      borderWidth: 1,
    },
    actionLabel: {
      fontSize: 11,
      fontFamily: FONTS.medium,
      color: COLORS.textMuted,
      textAlign: "center",
    },

    // ── Premium Card ─────────────────────────────────────────
    premiumCard: {
      marginHorizontal: SIZES.padding,
      borderRadius: SIZES.radius,
      backgroundColor: COLORS.surfaceContainer,
      borderWidth: 1,
      borderColor: COLORS.border,
      padding: SIZES.padding,
      marginBottom: SIZES.padding * 1.25,
    },
    premiumHeader: {
      flexDirection: "row",
      justifyContent: "space-between",
      alignItems: "flex-end",
      marginBottom: SIZES.padding,
    },
    premiumTitle: {
      fontSize: SIZES.small,
      fontFamily: FONTS.semiBold,
      color: COLORS.textFaint,
      marginBottom: 4,
    },
    premiumAmount: {
      fontSize: 26,
      fontFamily: FONTS.display,
      color: COLORS.white,
    },
    viewPlanBtn: { flexDirection: "row", alignItems: "center", gap: 3 },
    viewPlanText: {
      fontSize: SIZES.small,
      fontFamily: FONTS.semiBold,
      color: COLORS.primary,
    },
    breakdownRow: {
      flexDirection: "row",
      justifyContent: "space-between",
      borderTopWidth: 1,
      borderTopColor: COLORS.border,
      paddingTop: SIZES.padding * 0.75,
    },
    breakdownItem: { alignItems: "center" },
    breakdownKey: {
      fontSize: SIZES.tiny,
      fontFamily: FONTS.regular,
      color: COLORS.textFaint,
      marginBottom: 4,
    },
    breakdownVal: {
      fontSize: SIZES.small,
      fontFamily: FONTS.bold,
      color: COLORS.white,
    },

    // ── Activity Card ────────────────────────────────────────
    activityCard: {
      marginHorizontal: SIZES.padding,
      borderRadius: SIZES.radius,
      backgroundColor: COLORS.surfaceContainer,
      borderWidth: 1,
      borderColor: COLORS.border,
      marginBottom: SIZES.padding,
    },
    activityRow: {
      flexDirection: "row",
      alignItems: "center",
      gap: 12,
      padding: SIZES.padding * 0.85,
    },
    activityIconWrap: {
      width: 36,
      height: 36,
      borderRadius: 10,
      justifyContent: "center",
      alignItems: "center",
    },
    activityMeta: { flex: 1 },
    activityTitle: {
      fontSize: SIZES.small,
      fontFamily: FONTS.semiBold,
      color: COLORS.white,
      marginBottom: 2,
    },
    activityDate: {
      fontSize: SIZES.tiny,
      fontFamily: FONTS.regular,
      color: COLORS.textFaint,
    },
    activityRight: { alignItems: "flex-end" },
    activityAmount: {
      fontSize: SIZES.small,
      fontFamily: FONTS.bold,
      marginBottom: 2,
    },
    activityStatus: { fontSize: SIZES.tiny, fontFamily: FONTS.medium },

    // ── Pay Premium Card ─────────────────────────────────────
    payCard: {
      marginHorizontal: SIZES.padding,
      borderRadius: SIZES.radius,
      backgroundColor: COLORS.primaryContainer,
      borderWidth: 1,
      borderColor: COLORS.borderActive,
      padding: SIZES.padding,
      marginBottom: SIZES.padding,
      flexDirection: "row",
      alignItems: "center",
      justifyContent: "space-between",
    },
    payCardLeft: { flex: 1 },
    payCardLabel: {
      fontSize: 10,
      fontFamily: FONTS.bold,
      color: COLORS.primary,
      letterSpacing: 1,
      textTransform: "uppercase",
      marginBottom: 5,
    },
    payCardAmount: {
      fontSize: 24,
      fontFamily: FONTS.display,
      color: COLORS.white,
      lineHeight: 28,
    },
    payCardSub: {
      fontSize: SIZES.tiny,
      fontFamily: FONTS.regular,
      color: COLORS.textFaint,
      marginTop: 3,
    },
    paidBadge: {
      flexDirection: "row",
      alignItems: "center",
      gap: 5,
      backgroundColor: COLORS.successContainer,
      borderWidth: 1,
      borderColor: COLORS.success + "40",
      paddingHorizontal: 12,
      paddingVertical: 7,
      borderRadius: SIZES.radiusFull,
    },
    paidBadgeText: {
      fontSize: SIZES.small,
      fontFamily: FONTS.bold,
      color: COLORS.success,
    },
    payNowBtn: {
      backgroundColor: COLORS.primary,
      paddingHorizontal: 18,
      paddingVertical: 11,
      borderRadius: SIZES.radiusFull,
      ...SHADOWS.button,
    },
    payNowText: {
      fontSize: SIZES.small,
      fontFamily: FONTS.bold,
      color: "#fff",
    },

    // ── Payment Modal ─────────────────────────────────────────
    modalOverlay: {
      flex: 1,
      justifyContent: "flex-end",
      backgroundColor: "rgba(0,0,0,0.6)",
    },
    modalSheet: {
      backgroundColor: COLORS.surfaceContainer,
      borderTopLeftRadius: 24,
      borderTopRightRadius: 24,
      padding: SIZES.padding,
      paddingBottom: SIZES.padding * 2.5,
      borderTopWidth: 1,
      borderColor: COLORS.border,
    },
    modalHandle: {
      width: 36,
      height: 4,
      borderRadius: 2,
      backgroundColor: COLORS.border,
      alignSelf: "center",
      marginBottom: SIZES.padding,
    },
    modalTop: {
      flexDirection: "row",
      justifyContent: "space-between",
      alignItems: "flex-start",
      marginBottom: SIZES.padding,
    },
    modalTitle: {
      fontSize: SIZES.small,
      fontFamily: FONTS.semiBold,
      color: COLORS.textFaint,
      marginBottom: 4,
    },
    modalAmt: { fontSize: 32, fontFamily: FONTS.display, color: COLORS.white },
    modalCloseBtn: {
      width: 34,
      height: 34,
      borderRadius: 10,
      backgroundColor: COLORS.surfaceHigh,
      justifyContent: "center",
      alignItems: "center",
    },
    methodRow: { flexDirection: "row", gap: 8, marginBottom: SIZES.padding },
    methodChip: {
      flex: 1,
      flexDirection: "row",
      alignItems: "center",
      justifyContent: "center",
      gap: 5,
      paddingVertical: 9,
      borderRadius: SIZES.radiusFull,
      backgroundColor: COLORS.surfaceHigh,
      borderWidth: 1,
      borderColor: COLORS.border,
    },
    methodChipActive: {
      backgroundColor: COLORS.primaryContainer,
      borderColor: COLORS.borderActive,
    },
    methodChipText: {
      fontSize: 11,
      fontFamily: FONTS.semiBold,
      color: COLORS.textFaint,
    },
    field: { marginBottom: SIZES.padding * 0.75 },
    fieldLabel: {
      fontSize: SIZES.small,
      fontFamily: FONTS.semiBold,
      color: COLORS.textMuted,
      marginBottom: 7,
    },
    fieldInput: {
      height: 48,
      backgroundColor: COLORS.surfaceHigh,
      borderRadius: SIZES.radius,
      paddingHorizontal: 14,
      fontSize: SIZES.body,
      fontFamily: FONTS.regular,
      color: COLORS.white,
      borderWidth: 1,
      borderColor: COLORS.border,
    },
    payErrorBox: {
      flexDirection: "row",
      alignItems: "center",
      gap: 7,
      backgroundColor: COLORS.errorContainer,
      borderRadius: SIZES.radius,
      padding: 10,
      marginBottom: SIZES.base,
      borderWidth: 1,
      borderColor: COLORS.error + "35",
    },
    payErrorText: {
      flex: 1,
      fontSize: SIZES.small,
      fontFamily: FONTS.medium,
      color: COLORS.error,
    },
    submitBtn: {
      height: 52,
      borderRadius: SIZES.radiusFull,
      backgroundColor: COLORS.primary,
      flexDirection: "row",
      justifyContent: "center",
      alignItems: "center",
      gap: 10,
      marginTop: SIZES.base,
      ...SHADOWS.button,
    },
    submitBtnText: {
      fontSize: SIZES.body,
      fontFamily: FONTS.bold,
      color: "#fff",
    },
    successWrap: { alignItems: "center", paddingVertical: SIZES.padding * 1.5 },
    successCircle: {
      width: 80,
      height: 80,
      borderRadius: 40,
      backgroundColor: COLORS.successContainer,
      borderWidth: 1,
      borderColor: COLORS.success + "40",
      justifyContent: "center",
      alignItems: "center",
      marginBottom: SIZES.padding,
    },
    successTitle: {
      fontSize: SIZES.h3,
      fontFamily: FONTS.display,
      color: COLORS.white,
      marginBottom: 6,
    },
    successTxId: {
      fontSize: SIZES.small,
      fontFamily: FONTS.medium,
      color: COLORS.textFaint,
      marginBottom: 4,
    },
    successAmt: {
      fontSize: SIZES.h2,
      fontFamily: FONTS.bold,
      color: COLORS.success,
      marginBottom: SIZES.padding * 1.5,
    },
    doneBtn: {
      height: 50,
      width: "100%",
      borderRadius: SIZES.radiusFull,
      backgroundColor: COLORS.primary,
      justifyContent: "center",
      alignItems: "center",
      ...SHADOWS.button,
    },
    doneBtnText: {
      fontSize: SIZES.body,
      fontFamily: FONTS.bold,
      color: "#fff",
    },

    // ── Bottom Nav ───────────────────────────────────────────
    bottomNav: {
      position: "absolute",
      bottom: 0,
      left: 0,
      right: 0,
      height: 68,
      backgroundColor: COLORS.surfaceContainer,
      borderTopWidth: 1,
      borderTopColor: COLORS.border,
      flexDirection: "row",
      justifyContent: "space-around",
      alignItems: "center",
      paddingBottom: Platform.OS === "ios" ? 14 : 0,
    },
    navItem: { alignItems: "center", justifyContent: "center", flex: 1, paddingVertical: 4 },
    navActiveWrap: {
      backgroundColor: COLORS.primaryContainer,
      width: 40,
      height: 28,
      borderRadius: 8,
      justifyContent: "center",
      alignItems: "center",
    },
    navLabel: {
      fontSize: 10,
      fontFamily: FONTS.medium,
      color: COLORS.textFaint,
      marginTop: 3,
    },

    // Center chat button in nav
    navChatWrap: {
      flex: 1,
      alignItems: "center",
      justifyContent: "center",
      marginTop: -22,
    },
    navChatBtn: {
      width: 52,
      height: 52,
      borderRadius: 26,
      backgroundColor: COLORS.primary,
      alignItems: "center",
      justifyContent: "center",
      borderWidth: 3,
      borderColor: COLORS.surfaceContainer,
      ...SHADOWS.button,
    },
    navChatLabel: {
      fontSize: 10,
      fontFamily: FONTS.medium,
      color: COLORS.primary,
      marginTop: 4,
    },

    // ── Chat Modal ──
    chatOverlay: {
      flex: 1,
      justifyContent: "flex-end",
    },
    chatBackdrop: {
      flex: 1,
      backgroundColor: "rgba(0,0,0,0.45)",
    },
    chatSheet: {
      backgroundColor: COLORS.surfaceContainer,
      borderTopLeftRadius: 24,
      borderTopRightRadius: 24,
      maxHeight: "82%",
      borderTopWidth: 1,
      borderColor: COLORS.border,
    },
    chatHeader: {
      flexDirection: "row",
      alignItems: "center",
      justifyContent: "space-between",
      padding: 16,
      borderBottomWidth: 1,
      borderBottomColor: COLORS.border,
    },
    chatHeaderLeft: {
      flexDirection: "row",
      alignItems: "center",
      gap: 10,
    },
    chatAvatar: {
      width: 36,
      height: 36,
      borderRadius: 18,
      backgroundColor: COLORS.primary,
      alignItems: "center",
      justifyContent: "center",
    },
    chatAvatarText: {
      fontSize: 11,
      fontFamily: FONTS.bold,
      color: "#fff",
    },
    chatTitle: {
      fontSize: 14,
      fontFamily: FONTS.semiBold,
      color: COLORS.text,
    },
    chatSubtitle: {
      fontSize: 11,
      fontFamily: FONTS.regular,
      color: COLORS.textFaint,
      marginTop: 1,
    },
    chatCloseBtn: {
      padding: 6,
      borderRadius: 8,
    },
    chatBody: {
      flexGrow: 0,
      maxHeight: 380,
    },
    chatBodyContent: {
      padding: 14,
      gap: 10,
    },
    chatEmpty: {
      alignItems: "center",
      paddingVertical: 20,
      gap: 8,
    },
    chatEmptyIcon: {
      width: 60,
      height: 60,
      borderRadius: 30,
      backgroundColor: COLORS.primaryContainer,
      alignItems: "center",
      justifyContent: "center",
      marginBottom: 4,
    },
    chatEmptyTitle: {
      fontSize: 14,
      fontFamily: FONTS.semiBold,
      color: COLORS.text,
      textAlign: "center",
    },
    chatEmptyHint: {
      fontSize: 12,
      fontFamily: FONTS.regular,
      color: COLORS.textFaint,
      textAlign: "center",
      lineHeight: 18,
      paddingHorizontal: 10,
    },
    chatSuggestions: {
      flexDirection: "row",
      flexWrap: "wrap",
      justifyContent: "center",
      gap: 8,
      marginTop: 8,
    },
    chatSuggestion: {
      backgroundColor: COLORS.surfaceHigh,
      borderRadius: 20,
      paddingHorizontal: 12,
      paddingVertical: 6,
      borderWidth: 1,
      borderColor: COLORS.border,
    },
    chatSuggestionText: {
      fontSize: 11,
      fontFamily: FONTS.medium,
      color: COLORS.textMuted,
    },
    chatBubbleRow: {
      flexDirection: "row",
      alignItems: "flex-end",
      gap: 8,
      marginBottom: 2,
    },
    chatBubbleRowUser: {
      flexDirection: "row-reverse",
    },
    chatBubbleAvatar: {
      width: 26,
      height: 26,
      borderRadius: 13,
      backgroundColor: COLORS.primary,
      alignItems: "center",
      justifyContent: "center",
      flexShrink: 0,
    },
    chatBubbleAvatarText: {
      fontSize: 9,
      fontFamily: FONTS.bold,
      color: "#fff",
    },
    chatBubble: {
      maxWidth: "78%",
      borderRadius: 16,
      paddingHorizontal: 12,
      paddingVertical: 9,
    },
    chatBubbleAssistant: {
      backgroundColor: COLORS.surfaceHigh,
      borderBottomLeftRadius: 4,
    },
    chatBubbleUser: {
      backgroundColor: COLORS.primary,
      borderBottomRightRadius: 4,
    },
    chatBubbleText: {
      fontSize: 13,
      fontFamily: FONTS.regular,
      color: COLORS.text,
      lineHeight: 19,
    },
    chatBubbleTextUser: {
      color: "#fff",
    },
    chatInputRow: {
      flexDirection: "row",
      alignItems: "center",
      gap: 8,
      padding: 12,
      borderTopWidth: 1,
      borderTopColor: COLORS.border,
      paddingBottom: Platform.OS === "ios" ? 28 : 12,
    },
    chatInput: {
      flex: 1,
      height: 40,
      backgroundColor: COLORS.surfaceHigh,
      borderRadius: 20,
      paddingHorizontal: 14,
      fontSize: 13,
      fontFamily: FONTS.regular,
      color: COLORS.text,
      borderWidth: 1,
      borderColor: COLORS.border,
    },
    chatSendBtn: {
      width: 40,
      height: 40,
      borderRadius: 20,
      backgroundColor: COLORS.primary,
      alignItems: "center",
      justifyContent: "center",
      ...SHADOWS.button,
    },
    chatSendBtnDisabled: {
      opacity: 0.4,
    },
  });

export default HomeScreen;
