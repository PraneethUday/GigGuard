import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, SafeAreaView, ScrollView, TouchableOpacity, ActivityIndicator } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { COLORS, FONTS, SIZES } from '../constants/theme';
import { useAuth } from '../context/AuthContext';
import * as api from '../lib/api';

const TIERS = ['basic', 'standard', 'pro'];
const PLAN_DETAILS = {
  basic:    { label: 'Basic Shield',   tag: 'Low-cost entry cover' },
  standard: { label: 'Standard Guard', tag: 'Balanced weekly protection' },
  pro:      { label: 'Pro Protect',    tag: 'Highest payout priority' },
};

const TRIGGERS = [
  { icon: 'rainy-outline',       label: 'Heavy Rainfall',      threshold: '>64.5mm/24h' },
  { icon: 'cloud-outline',       label: 'Air Pollution (AQI)',  threshold: 'AQI > 400' },
  { icon: 'ban-outline',         label: 'Curfew / Sec. 144',   threshold: 'Official Order' },
  { icon: 'water-outline',       label: 'Flood / Waterlogging', threshold: 'Zone Tagged' },
  { icon: 'thermometer-outline', label: 'Extreme Heat',         threshold: '>45°C' },
];

export default function PolicyScreen({ navigation }) {
  const { user } = useAuth();
  const [currentPremium, setCurrentPremium] = useState(null);
  const [quotes, setQuotes] = useState({});
  const [nextWeekTier, setNextWeekTier] = useState(user?.tier || 'standard');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!user?.delivery_id) { setLoading(false); return; }

    // Load saved next-week plan
    AsyncStorage.getItem(`gg_next_week_plan_${user.id}`)
      .then(saved => { if (saved && TIERS.includes(saved)) setNextWeekTier(saved); })
      .catch(() => {});

    // Fetch current tier premium
    api.predictPremium(user.delivery_id, user.city, user.tier || 'standard')
      .then(setCurrentPremium)
      .catch(() => {});

    // Fetch all tier quotes
    Promise.allSettled(
      TIERS.map(t => api.predictPremium(user.delivery_id, user.city, t).then(q => [t, q]))
    ).then(results => {
      const q = {};
      results.forEach(r => { if (r.status === 'fulfilled') { const [t, data] = r.value; q[t] = data; } });
      setQuotes(q);
    }).finally(() => setLoading(false));
  }, [user]);

  const selectNextWeek = (tier) => {
    setNextWeekTier(tier);
    if (user?.id) AsyncStorage.setItem(`gg_next_week_plan_${user.id}`, tier).catch(() => {});
  };

  const currentTier = user?.tier || 'standard';
  const tierLabel = PLAN_DETAILS[currentTier]?.label || 'Standard Guard';
  const verified = user?.verification_status === 'verified';
  const wp = currentPremium ? (user?.autopay ? currentPremium.weekly_premium_autopay : currentPremium.weekly_premium) : null;
  const platformCount = user?.platforms?.length || 0;

  const premiumRows = currentPremium ? [
    { label: `Base premium`, value: `₹${currentPremium.raw_prediction || '-'}` },
    { label: 'Weather risk factor', value: currentPremium.weather_risk != null ? `×${(1 + currentPremium.weather_risk).toFixed(2)}` : '-' },
    { label: 'City risk factor', value: currentPremium.city_risk != null ? `×${currentPremium.city_risk.toFixed(2)}` : '-' },
    ...(user?.autopay ? [{ label: 'AutoPay discount', value: '−5%' }] : []),
    { label: 'Final weekly premium', value: wp != null ? `₹${Math.round(wp)}` : '-', bold: true },
  ] : [];

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backBtn}>
          <Ionicons name="arrow-back" size={22} color={COLORS.white} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>My Policy</Text>
        <View style={{ width: 40 }} />
      </View>

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scroll}>
        {loading && <ActivityIndicator size="small" color={COLORS.primary} style={{ marginBottom: 16 }} />}

        {/* Active Policy Card */}
        <View style={styles.policyCard}>
          <View style={styles.policyCardTop}>
            <View>
              <Text style={styles.policyCardLabel}>ACTIVE POLICY</Text>
              <Text style={styles.policyCardTier}>{tierLabel}</Text>
              <Text style={styles.policyCardId}>{user?.city || ''} · {user?.delivery_id || ''}</Text>
            </View>
            <View style={[styles.activeBadge, !verified && { backgroundColor: '#d97706' }]}>
              <Text style={styles.activeBadgeText}>{verified ? 'Protected ✓' : 'Pending'}</Text>
            </View>
          </View>
          <View style={styles.policyStats}>
            <View style={styles.policyStat}>
              <Text style={styles.policyStatLabel}>Weekly Premium</Text>
              <Text style={styles.policyStatValue}>{wp != null ? `₹${Math.round(wp)}` : '...'}</Text>
            </View>
            <View style={styles.policyStat}>
              <Text style={styles.policyStatLabel}>Max Payout</Text>
              <Text style={styles.policyStatValue}>{currentPremium?.max_payout ? `₹${currentPremium.max_payout.toLocaleString()}` : '...'}</Text>
            </View>
            <View style={styles.policyStat}>
              <Text style={styles.policyStatLabel}>Platforms</Text>
              <Text style={styles.policyStatValue}>{platformCount} Active</Text>
            </View>
          </View>
        </View>

        {/* Premium Breakdown */}
        {premiumRows.length > 0 && (
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Ionicons name="calculator-outline" size={20} color={COLORS.primary} />
              <Text style={styles.sectionTitle}>Premium Breakdown</Text>
            </View>
            {premiumRows.map((row, i) => (
              <View key={i} style={[styles.premiumRow, i === premiumRows.length - 1 && styles.premiumRowLast]}>
                <Text style={styles.premiumLabel}>{row.label}</Text>
                <Text style={[styles.premiumValue, row.bold && styles.premiumValueBold]}>{row.value}</Text>
              </View>
            ))}
          </View>
        )}

        {/* Next Week Plan Selection */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Ionicons name="calendar-outline" size={20} color={COLORS.primary} />
            <Text style={styles.sectionTitle}>Next Week Plan</Text>
          </View>
          {TIERS.map((t) => {
            const q = quotes[t];
            const selected = t === nextWeekTier;
            const price = q ? (user?.autopay ? q.weekly_premium_autopay : q.weekly_premium) : null;
            return (
              <TouchableOpacity
                key={t}
                style={[styles.tierCard, selected && styles.tierCardSelected]}
                onPress={() => selectNextWeek(t)}
              >
                <View style={{ flex: 1 }}>
                  <Text style={[styles.tierLabel, selected && { color: COLORS.white }]}>
                    {PLAN_DETAILS[t].label}
                  </Text>
                  <Text style={[styles.tierSub, selected && { color: 'rgba(255,255,255,0.7)' }]}>
                    {price != null ? `₹${Math.round(price)}/wk` : 'Loading...'}
                    {q?.max_payout ? ` · Max ₹${q.max_payout.toLocaleString()}` : ''}
                  </Text>
                </View>
                <View style={[styles.radioOuter, selected && { borderColor: COLORS.white }]}>
                  {selected && <View style={styles.radioInner} />}
                </View>
              </TouchableOpacity>
            );
          })}
        </View>

        {/* Coverage Triggers */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Ionicons name="shield-checkmark-outline" size={20} color={COLORS.primary} />
            <Text style={styles.sectionTitle}>Coverage Triggers</Text>
          </View>
          {TRIGGERS.map((t, i) => (
            <View key={i} style={styles.triggerRow}>
              <View style={styles.triggerIcon}>
                <Ionicons name={t.icon} size={18} color={COLORS.primary} />
              </View>
              <View style={styles.triggerInfo}>
                <Text style={styles.triggerLabel}>{t.label}</Text>
                <Text style={styles.triggerThreshold}>{t.threshold}</Text>
              </View>
              <View style={styles.triggerActiveBadge}>
                <Text style={styles.triggerActiveBadgeText}>Active</Text>
              </View>
            </View>
          ))}
        </View>

        {/* Policy Details */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Ionicons name="document-text-outline" size={20} color={COLORS.primary} />
            <Text style={styles.sectionTitle}>Policy Details</Text>
          </View>
          {[
            ['Delivery ID', user?.delivery_id || '-'],
            ['City / Zone', `${user?.city || '-'}${user?.area ? ` · ${user.area}` : ''}`],
            ['Platforms', (user?.platforms || []).join(', ') || '-'],
            ['Tier', tierLabel],
            ['Verification', verified ? 'Verified' : 'Pending'],
          ].map(([k, v]) => (
            <View key={k} style={styles.detailRow}>
              <Text style={styles.detailKey}>{k}</Text>
              <Text style={styles.detailValue}>{v}</Text>
            </View>
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.background },
  header: { height: 60, backgroundColor: COLORS.secondary, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: SIZES.padding },
  backBtn: { width: 40, height: 40, justifyContent: 'center' },
  headerTitle: { fontSize: 18, fontFamily: FONTS.bold, color: COLORS.white },
  scroll: { padding: SIZES.padding, paddingBottom: SIZES.padding * 3 },

  policyCard: { backgroundColor: COLORS.secondary, borderRadius: SIZES.radius * 1.5, padding: SIZES.padding, marginBottom: SIZES.padding, overflow: 'hidden' },
  policyCardTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: SIZES.padding },
  policyCardLabel: { fontSize: 10, fontFamily: FONTS.bold, color: 'rgba(255,255,255,0.5)', letterSpacing: 1, textTransform: 'uppercase', marginBottom: 4 },
  policyCardTier: { fontSize: 20, fontFamily: FONTS.bold, color: COLORS.white, marginBottom: 2 },
  policyCardId: { fontSize: 13, fontFamily: FONTS.medium, color: 'rgba(255,255,255,0.5)' },
  activeBadge: { backgroundColor: COLORS.success, paddingHorizontal: 12, paddingVertical: 6, borderRadius: 100 },
  activeBadgeText: { fontSize: 12, fontFamily: FONTS.bold, color: COLORS.white },
  policyStats: { flexDirection: 'row', justifyContent: 'space-between', paddingTop: SIZES.base, borderTopWidth: 1, borderTopColor: 'rgba(255,255,255,0.1)' },
  policyStat: {},
  policyStatLabel: { fontSize: 10, fontFamily: FONTS.bold, color: 'rgba(255,255,255,0.45)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 4 },
  policyStatValue: { fontSize: 17, fontFamily: FONTS.bold, color: COLORS.white },

  section: { backgroundColor: COLORS.surface, borderRadius: SIZES.radius, padding: SIZES.padding, marginBottom: SIZES.base * 2, borderWidth: 1, borderColor: 'rgba(27,27,58,0.06)' },
  sectionHeader: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: SIZES.padding * 0.75 },
  sectionTitle: { fontSize: 16, fontFamily: FONTS.bold, color: COLORS.accent, marginLeft: 8 },

  premiumRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 9, borderBottomWidth: 1, borderBottomColor: 'rgba(27,27,58,0.06)' },
  premiumRowLast: { borderBottomWidth: 0 },
  premiumLabel: { fontSize: 14, fontFamily: FONTS.medium, color: 'rgba(27,27,58,0.6)', flex: 1 },
  premiumValue: { fontSize: 14, fontFamily: FONTS.bold, color: COLORS.accent },
  premiumValueBold: { fontSize: 16, color: COLORS.primary },

  tierCard: {
    flexDirection: 'row', alignItems: 'center', padding: SIZES.padding * 0.75,
    borderRadius: SIZES.radius, borderWidth: 1.5, borderColor: 'rgba(27,27,58,0.1)',
    marginBottom: SIZES.base, backgroundColor: COLORS.surface,
  },
  tierCardSelected: { backgroundColor: COLORS.secondary, borderColor: COLORS.secondary },
  tierLabel: { fontSize: 15, fontFamily: FONTS.bold, color: COLORS.accent },
  tierSub: { fontSize: 12, fontFamily: FONTS.medium, color: 'rgba(27,27,58,0.5)', marginTop: 2 },
  radioOuter: { width: 22, height: 22, borderRadius: 11, borderWidth: 2, borderColor: 'rgba(27,27,58,0.2)', justifyContent: 'center', alignItems: 'center' },
  radioInner: { width: 10, height: 10, borderRadius: 5, backgroundColor: COLORS.white },

  triggerRow: { flexDirection: 'row', alignItems: 'center', paddingVertical: 10, borderBottomWidth: 1, borderBottomColor: 'rgba(27,27,58,0.05)' },
  triggerIcon: { width: 36, height: 36, borderRadius: 10, backgroundColor: 'rgba(165,28,48,0.08)', justifyContent: 'center', alignItems: 'center', marginRight: 12 },
  triggerInfo: { flex: 1 },
  triggerLabel: { fontSize: 14, fontFamily: FONTS.bold, color: COLORS.accent },
  triggerThreshold: { fontSize: 12, fontFamily: FONTS.medium, color: 'rgba(27,27,58,0.45)', marginTop: 2 },
  triggerActiveBadge: { backgroundColor: 'rgba(34,197,94,0.1)', paddingHorizontal: 10, paddingVertical: 4, borderRadius: 100 },
  triggerActiveBadgeText: { fontSize: 11, fontFamily: FONTS.bold, color: '#16a34a' },

  detailRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 9, borderBottomWidth: 1, borderBottomColor: 'rgba(27,27,58,0.05)' },
  detailKey: { fontSize: 14, fontFamily: FONTS.medium, color: 'rgba(27,27,58,0.5)' },
  detailValue: { fontSize: 14, fontFamily: FONTS.bold, color: COLORS.accent, maxWidth: '55%', textAlign: 'right' },
});
