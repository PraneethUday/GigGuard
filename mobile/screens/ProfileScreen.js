import React, { useMemo, useState } from "react";
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

function labelTier(tier) {
  if (tier === "basic") return "Basic Shield";
  if (tier === "pro") return "Pro Protect";
  return "Standard Guard";
}

const Row = ({ label, value, icon, onPress }) => (
  <TouchableOpacity
    onPress={onPress}
    style={styles.row}
    activeOpacity={onPress ? 0.7 : 1}
  >
    <View style={styles.rowLeft}>
      <View style={styles.rowIcon}>
        <Ionicons name={icon} size={16} color={COLORS.primary} />
      </View>
      <Text style={styles.rowLabel}>{label}</Text>
    </View>
    <View style={styles.rowRight}>
      <Text style={styles.rowValue}>{value}</Text>
      {onPress && (
        <Ionicons name="chevron-forward" size={14} color={COLORS.textFaint} />
      )}
    </View>
  </TouchableOpacity>
);

export default function ProfileScreen({ navigation }) {
  const { user, logout, refreshUser } = useAuth();
  const [refreshing, setRefreshing] = useState(false);

  const initials = useMemo(() => {
    if (!user?.name) return "GG";
    const parts = user.name.trim().split(" ").filter(Boolean);
    if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
    return `${parts[0][0] || ""}${parts[1][0] || ""}`.toUpperCase();
  }, [user?.name]);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await refreshUser();
    } finally {
      setRefreshing(false);
    }
  };

  const handleLogout = async () => {
    await logout();
    navigation.reset({ index: 0, routes: [{ name: "Landing" }] });
  };

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
        <Text style={styles.headerTitle}>Profile</Text>
        <View style={{ width: 40 }} />
      </View>

      <ScrollView
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.scroll}
      >
        {/* Avatar section */}
        <View style={styles.avatarSection}>
          <View style={styles.avatarGlow} />
          <View style={styles.avatar}>
            <Text style={styles.avatarText}>{initials}</Text>
          </View>
          <Text style={styles.name}>{user?.name || "WPIP User"}</Text>
          <Text style={styles.zone}>
            {user?.city || "Unknown City"} · {labelTier(user?.tier)}
          </Text>
          <View style={styles.protectedBadge}>
            <Ionicons
              name="shield-checkmark"
              size={14}
              color={COLORS.success}
            />
            <Text style={styles.protectedText}>
              {user?.verification_status === "verified"
                ? "Income Protected"
                : "Verification Pending"}
            </Text>
          </View>
        </View>

        {/* Personal Info */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Personal Information</Text>
          <Row
            icon="person-outline"
            label="Full Name"
            value={user?.name || "—"}
          />
          <Row icon="call-outline" label="Phone" value={user?.phone || "—"} />
          <Row icon="mail-outline" label="Email" value={user?.email || "—"} />
          <Row icon="location-outline" label="City" value={user?.city || "—"} />
          <Row
            icon="map-outline"
            label="Area / Zone"
            value={user?.area || "—"}
          />
          <Row
            icon="card-outline"
            label="Delivery Partner ID"
            value={user?.delivery_id || "—"}
          />
        </View>

        {/* Connected Platforms */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Connected Platforms</Text>
          {(user?.platforms || []).length === 0 && (
            <View style={styles.emptyCard}>
              <Text style={styles.emptyText}>No platforms linked yet.</Text>
            </View>
          )}

          {(user?.platforms || []).map((platform) => (
            <View key={platform} style={styles.platformCard}>
              <View style={styles.platformLeft}>
                <View style={styles.platformIconWrap}>
                  <Text style={styles.platformInitial}>
                    {platform[0]?.toUpperCase() || "P"}
                  </Text>
                </View>
                <View>
                  <Text style={styles.platformName}>{platform}</Text>
                  <Text style={styles.platformId}>
                    Delivery ID: {user?.delivery_id || "—"}
                  </Text>
                </View>
              </View>
              <View style={styles.verifiedBadge}>
                <Text style={styles.verifiedText}>
                  {user?.verification_status === "verified"
                    ? "Verified ✓"
                    : "Pending"}
                </Text>
              </View>
            </View>
          ))}
        </View>

        {/* Coverage Settings */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Coverage Settings</Text>
          <Row
            icon="shield-outline"
            label="Current Plan"
            value={labelTier(user?.tier)}
          />
          <Row
            icon="repeat-outline"
            label="AutoPay"
            value={user?.autopay ? "Enabled (5% discount)" : "Disabled"}
          />
          <Row
            icon="checkmark-done-outline"
            label="Verification"
            value={user?.verification_status || "pending"}
          />
          <Row icon="wallet-outline" label="UPI ID" value={user?.upi || "—"} />
        </View>

        <TouchableOpacity
          style={styles.refreshProfileBtn}
          onPress={handleRefresh}
          disabled={refreshing}
        >
          {refreshing ? (
            <ActivityIndicator color={COLORS.white} size="small" />
          ) : (
            <Ionicons name="refresh" size={18} color={COLORS.white} />
          )}
          <Text style={styles.refreshProfileText}>Refresh Profile</Text>
        </TouchableOpacity>

        {/* Logout */}
        <TouchableOpacity style={styles.logoutBtn} onPress={handleLogout}>
          <Ionicons name="log-out-outline" size={18} color={COLORS.error} />
          <Text style={styles.logoutText}>Sign Out</Text>
        </TouchableOpacity>

        <Text style={styles.footer}>WPIP v1.0.0 · IRDAI Registered</Text>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.surface },

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
  scroll: { paddingBottom: SIZES.padding * 3 },

  // Avatar
  avatarSection: {
    alignItems: "center",
    paddingVertical: SIZES.padding * 1.5,
    backgroundColor: COLORS.surfaceContainer,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
    position: "relative",
    overflow: "hidden",
  },
  avatarGlow: {
    position: "absolute",
    width: 160,
    height: 160,
    borderRadius: 80,
    backgroundColor: COLORS.primary,
    opacity: 0.06,
    top: -40,
  },
  avatar: {
    width: 72,
    height: 72,
    borderRadius: 36,
    backgroundColor: COLORS.primaryContainer,
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 12,
    borderWidth: 2,
    borderColor: COLORS.primary + "50",
  },
  avatarText: { fontSize: 24, fontFamily: FONTS.bold, color: COLORS.primary },
  name: {
    fontSize: SIZES.h2,
    fontFamily: FONTS.bold,
    color: COLORS.white,
    marginBottom: 4,
  },
  zone: {
    fontSize: SIZES.small,
    fontFamily: FONTS.medium,
    color: COLORS.textMuted,
    marginBottom: 12,
  },
  protectedBadge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: COLORS.successContainer,
    paddingHorizontal: 14,
    paddingVertical: 6,
    borderRadius: SIZES.radiusFull,
    borderWidth: 1,
    borderColor: COLORS.success + "30",
  },
  protectedText: {
    fontSize: SIZES.small,
    fontFamily: FONTS.bold,
    color: COLORS.success,
  },

  // Sections
  section: {
    marginHorizontal: SIZES.padding,
    marginTop: SIZES.padding,
    borderRadius: SIZES.radius * 1.2,
    backgroundColor: COLORS.surfaceContainer,
    borderWidth: 1,
    borderColor: COLORS.border,
    overflow: "hidden",
  },
  sectionTitle: {
    fontSize: SIZES.tiny,
    fontFamily: FONTS.bold,
    color: COLORS.textFaint,
    textTransform: "uppercase",
    letterSpacing: 1,
    padding: SIZES.padding * 0.75,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },

  // Rows
  row: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    padding: SIZES.padding * 0.7,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  rowLeft: { flexDirection: "row", alignItems: "center", gap: 12, flex: 1 },
  rowIcon: {
    width: 32,
    height: 32,
    borderRadius: 10,
    backgroundColor: COLORS.primaryContainer,
    justifyContent: "center",
    alignItems: "center",
  },
  rowLabel: {
    fontSize: SIZES.small,
    fontFamily: FONTS.medium,
    color: COLORS.white,
  },
  rowRight: { flexDirection: "row", alignItems: "center", gap: 4 },
  rowValue: {
    fontSize: SIZES.small,
    fontFamily: FONTS.medium,
    color: COLORS.textFaint,
    maxWidth: 140,
    textAlign: "right",
  },

  // Platforms
  platformCard: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    padding: SIZES.padding * 0.75,
    borderBottomWidth: 1,
    borderBottomColor: COLORS.border,
  },
  platformLeft: { flexDirection: "row", alignItems: "center", gap: 12 },
  platformIconWrap: {
    width: 40,
    height: 40,
    borderRadius: 12,
    backgroundColor: COLORS.primaryContainer,
    justifyContent: "center",
    alignItems: "center",
  },
  platformInitial: {
    fontSize: 18,
    fontFamily: FONTS.bold,
    color: COLORS.primary,
  },
  platformName: {
    fontSize: SIZES.small,
    fontFamily: FONTS.bold,
    color: COLORS.white,
  },
  platformId: {
    fontSize: SIZES.tiny,
    fontFamily: FONTS.medium,
    color: COLORS.textFaint,
    marginTop: 1,
  },
  verifiedBadge: {
    backgroundColor: COLORS.successContainer,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: SIZES.radiusFull,
    borderWidth: 1,
    borderColor: COLORS.success + "30",
  },
  verifiedText: {
    fontSize: SIZES.tiny,
    fontFamily: FONTS.bold,
    color: COLORS.success,
  },
  emptyCard: { padding: SIZES.padding * 0.8 },
  emptyText: {
    fontSize: SIZES.small,
    fontFamily: FONTS.medium,
    color: COLORS.textMuted,
  },

  // Toggles
  refreshProfileBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    marginHorizontal: SIZES.padding,
    marginTop: SIZES.padding * 1.2,
    height: 48,
    borderRadius: SIZES.radius,
    backgroundColor: COLORS.primary,
    ...SHADOWS.button,
  },
  refreshProfileText: {
    fontSize: SIZES.body,
    fontFamily: FONTS.bold,
    color: COLORS.white,
  },

  // Logout
  logoutBtn: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 10,
    margin: SIZES.padding,
    marginTop: SIZES.padding * 1.5,
    height: 52,
    borderRadius: SIZES.radius,
    borderWidth: 1,
    borderColor: COLORS.error + "40",
    backgroundColor: COLORS.errorContainer,
  },
  logoutText: {
    fontSize: SIZES.body,
    fontFamily: FONTS.bold,
    color: COLORS.error,
  },
  footer: {
    textAlign: "center",
    fontSize: 11,
    fontFamily: FONTS.regular,
    color: COLORS.textFaint,
    paddingBottom: SIZES.padding,
  },
});
