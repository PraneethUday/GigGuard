import React from 'react';
import {
  View, Text, StyleSheet, SafeAreaView,
  ScrollView, TouchableOpacity,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, FONTS, SIZES, SHADOWS } from '../constants/theme';

const triggers = [
  { icon: 'rainy-outline', label: 'Heavy Rainfall', threshold: '>64.5mm/24h', active: true },
  { icon: 'cloud-outline', label: 'Air Pollution (AQI)', threshold: 'AQI > 400', active: true },
  { icon: 'ban-outline', label: 'Curfew / Sec. 144', threshold: 'Official Order', active: true },
  { icon: 'water-outline', label: 'Flood / Waterlogging', threshold: 'Zone Tagged', active: true },
  { icon: 'thermometer-outline', label: 'Extreme Heat', threshold: '>45°C', active: true },
];

const premiumRows = [
  { label: 'Base Premium (3.5% × ₹2,800)', value: '₹98', muted: true },
  { label: 'Zone Risk (Moderate)', value: '×1.05', muted: true },
  { label: 'Weather Forecast', value: '×1.10', muted: true },
  { label: 'Loyalty Discount (8 wks)', value: '−8%', positive: true },
  { label: 'AutoPay Discount', value: '−5%', positive: true },
  { label: 'Final Premium', value: '₹99', bold: true },
];

export default function PolicyScreen({ navigation }) {
  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backBtn}>
          <Ionicons name="arrow-back" size={20} color={COLORS.white} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>My Policy</Text>
        <View style={{ width: 40 }} />
      </View>

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scroll}>

        {/* Active Policy Hero Card */}
        <View style={styles.policyHero}>
          <View style={styles.heroGlow} />
          <View style={styles.heroTop}>
            <View>
              <Text style={styles.heroTierLabel}>ACTIVE POLICY</Text>
              <Text style={styles.heroTier}>Standard Guard</Text>
              <Text style={styles.heroId}>GG-2026-44821</Text>
            </View>
            <View style={styles.activeBadge}>
              <View style={styles.activeDot} />
              <Text style={styles.activeBadgeText}>Protected ✓</Text>
            </View>
          </View>

          <View style={styles.heroStats}>
            {[
              { label: 'Weekly Premium', value: '₹99' },
              { label: 'Max Payout', value: '₹1,200' },
              { label: 'Platforms', value: '2 Active' },
            ].map((s, i) => (
              <View key={i} style={styles.heroStat}>
                <Text style={styles.heroStatLabel}>{s.label}</Text>
                <Text style={styles.heroStatValue}>{s.value}</Text>
              </View>
            ))}
          </View>
        </View>

        {/* Premium Breakdown */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Ionicons name="calculator-outline" size={18} color={COLORS.primary} />
            <Text style={styles.sectionTitle}>Premium Breakdown</Text>
          </View>
          {premiumRows.map((row, i) => (
            <View key={i} style={[styles.premiumRow, i === premiumRows.length - 1 && styles.premiumRowLast]}>
              <Text style={styles.premiumLabel}>{row.label}</Text>
              <Text style={[
                styles.premiumValue,
                row.positive && { color: COLORS.success },
                row.bold && { color: COLORS.primary, fontSize: 16 },
              ]}>{row.value}</Text>
            </View>
          ))}
        </View>

        {/* Loyalty Progress */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Ionicons name="star-outline" size={18} color={COLORS.amber} />
            <Text style={styles.sectionTitle}>Loyalty Progress</Text>
          </View>
          <View style={styles.loyaltyInfo}>
            <Text style={styles.loyaltyWeeks}>8 weeks</Text>
            <Text style={styles.loyaltyDiscount}>8% discount</Text>
          </View>
          <View style={styles.progressTrack}>
            <View style={[styles.progressFill, { width: '53%' }]} />
          </View>
          <View style={styles.progressLabels}>
            <Text style={styles.progressLabel}>0 wks</Text>
            <Text style={styles.progressLabel}>15 wks → 15% max</Text>
          </View>
        </View>

        {/* Coverage Triggers */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Ionicons name="shield-checkmark-outline" size={18} color={COLORS.primary} />
            <Text style={styles.sectionTitle}>Coverage Triggers</Text>
          </View>
          {triggers.map((t, i) => (
            <View key={i} style={[styles.triggerRow, i < triggers.length - 1 && styles.triggerBorder]}>
              <View style={styles.triggerIcon}>
                <Ionicons name={t.icon} size={16} color={COLORS.primary} />
              </View>
              <View style={{ flex: 1 }}>
                <Text style={styles.triggerLabel}>{t.label}</Text>
                <Text style={styles.triggerThreshold}>{t.threshold}</Text>
              </View>
              <View style={styles.triggerBadge}>
                <Text style={styles.triggerBadgeText}>Active</Text>
              </View>
            </View>
          ))}
        </View>

        {/* Policy Details */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Ionicons name="document-text-outline" size={18} color={COLORS.primary} />
            <Text style={styles.sectionTitle}>Policy Details</Text>
          </View>
          {[
            ['Policy Number', 'GG-2026-44821'],
            ['Coverage Start', '20 Jan 2026'],
            ['Valid Until', '31 Dec 2026'],
            ['Zone', 'Chennai South'],
            ['Platforms', 'Zomato, Amazon Flex'],
          ].map(([k, v]) => (
            <View key={k} style={[styles.detailRow]}>
              <Text style={styles.detailKey}>{k}</Text>
              <Text style={styles.detailValue}>{v}</Text>
            </View>
          ))}
        </View>

        {/* Upgrade card */}
        <View style={styles.upgradeCard}>
          <View style={styles.upgradeLeft}>
            <Ionicons name="rocket-outline" size={22} color={COLORS.amber} />
            <View>
              <Text style={styles.upgradeTitle}>Upgrade to Pro Protect</Text>
              <Text style={styles.upgradeSub}>Max payout ₹2,500 · All 5 triggers</Text>
            </View>
          </View>
          <TouchableOpacity style={styles.upgradeBtn}>
            <Text style={styles.upgradeBtnText}>Upgrade</Text>
          </TouchableOpacity>
        </View>

      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.surface },

  header: { height: 56, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: SIZES.padding, borderBottomWidth: 1, borderBottomColor: COLORS.border },
  backBtn: { width: 40, height: 40, borderRadius: 12, backgroundColor: COLORS.surfaceHigh, justifyContent: 'center', alignItems: 'center' },
  headerTitle: { fontSize: SIZES.h3, fontFamily: FONTS.bold, color: COLORS.white },
  scroll: { padding: SIZES.padding, paddingBottom: SIZES.padding * 3 },

  // Hero
  policyHero: { borderRadius: SIZES.radius * 1.5, backgroundColor: COLORS.primary, padding: SIZES.padding, marginBottom: SIZES.padding, overflow: 'hidden', ...SHADOWS.button },
  heroGlow: { position: 'absolute', width: 180, height: 180, borderRadius: 90, backgroundColor: '#fff', opacity: 0.06, top: -60, right: -40 },
  heroTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: SIZES.padding },
  heroTierLabel: { fontSize: SIZES.tiny, fontFamily: FONTS.bold, color: 'rgba(255,255,255,0.55)', letterSpacing: 1, marginBottom: 4 },
  heroTier: { fontSize: SIZES.h2, fontFamily: FONTS.bold, color: '#fff', marginBottom: 2 },
  heroId: { fontSize: SIZES.small, fontFamily: FONTS.medium, color: 'rgba(255,255,255,0.5)' },
  activeBadge: { flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: COLORS.successContainer, paddingHorizontal: 12, paddingVertical: 6, borderRadius: SIZES.radiusFull },
  activeDot: { width: 7, height: 7, borderRadius: 4, backgroundColor: COLORS.success },
  activeBadgeText: { fontSize: SIZES.tiny, fontFamily: FONTS.bold, color: COLORS.success },
  heroStats: { flexDirection: 'row', justifyContent: 'space-between', borderTopWidth: 1, borderTopColor: 'rgba(255,255,255,0.15)', paddingTop: SIZES.padding },
  heroStat: {},
  heroStatLabel: { fontSize: SIZES.tiny, fontFamily: FONTS.medium, color: 'rgba(255,255,255,0.5)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: 0.3 },
  heroStatValue: { fontSize: 17, fontFamily: FONTS.bold, color: '#fff' },

  // Sections
  section: { backgroundColor: COLORS.surfaceContainer, borderRadius: SIZES.radius * 1.2, padding: SIZES.padding, marginBottom: SIZES.padding, borderWidth: 1, borderColor: COLORS.border },
  sectionHeader: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: SIZES.padding * 0.75 },
  sectionTitle: { fontSize: SIZES.body, fontFamily: FONTS.bold, color: COLORS.white },

  // Premium
  premiumRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 10, borderBottomWidth: 1, borderBottomColor: COLORS.border },
  premiumRowLast: { borderBottomWidth: 0 },
  premiumLabel: { fontSize: SIZES.small, fontFamily: FONTS.medium, color: COLORS.textMuted, flex: 1 },
  premiumValue: { fontSize: SIZES.small, fontFamily: FONTS.bold, color: COLORS.white },

  // Loyalty
  loyaltyInfo: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: SIZES.base },
  loyaltyWeeks: { fontSize: SIZES.body, fontFamily: FONTS.bold, color: COLORS.white },
  loyaltyDiscount: { fontSize: SIZES.body, fontFamily: FONTS.bold, color: COLORS.success },
  progressTrack: { height: 8, backgroundColor: COLORS.surfaceHighest, borderRadius: 4, overflow: 'hidden', marginBottom: SIZES.base },
  progressFill: { height: '100%', backgroundColor: COLORS.primary, borderRadius: 4 },
  progressLabels: { flexDirection: 'row', justifyContent: 'space-between' },
  progressLabel: { fontSize: SIZES.tiny, fontFamily: FONTS.medium, color: COLORS.textFaint },

  // Triggers
  triggerRow: { flexDirection: 'row', alignItems: 'center', paddingVertical: 11 },
  triggerBorder: { borderBottomWidth: 1, borderBottomColor: COLORS.border },
  triggerIcon: { width: 34, height: 34, borderRadius: 10, backgroundColor: COLORS.primaryContainer, justifyContent: 'center', alignItems: 'center', marginRight: 12 },
  triggerLabel: { fontSize: SIZES.small, fontFamily: FONTS.bold, color: COLORS.white, marginBottom: 2 },
  triggerThreshold: { fontSize: SIZES.tiny, fontFamily: FONTS.medium, color: COLORS.textFaint },
  triggerBadge: { backgroundColor: COLORS.successContainer, paddingHorizontal: 10, paddingVertical: 4, borderRadius: SIZES.radiusFull, borderWidth: 1, borderColor: COLORS.success + '30' },
  triggerBadgeText: { fontSize: SIZES.tiny, fontFamily: FONTS.bold, color: COLORS.success },

  // Details
  detailRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 10, borderBottomWidth: 1, borderBottomColor: COLORS.border },
  detailKey: { fontSize: SIZES.small, fontFamily: FONTS.medium, color: COLORS.textMuted },
  detailValue: { fontSize: SIZES.small, fontFamily: FONTS.bold, color: COLORS.white, maxWidth: '55%', textAlign: 'right' },

  // Upgrade
  upgradeCard: { borderRadius: SIZES.radius * 1.2, backgroundColor: COLORS.amberContainer, borderWidth: 1, borderColor: COLORS.amber + '40', padding: SIZES.padding, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  upgradeLeft: { flexDirection: 'row', alignItems: 'center', gap: 12, flex: 1 },
  upgradeTitle: { fontSize: SIZES.small, fontFamily: FONTS.bold, color: COLORS.amber, marginBottom: 2 },
  upgradeSub: { fontSize: SIZES.tiny, fontFamily: FONTS.medium, color: COLORS.amberDim },
  upgradeBtn: { backgroundColor: COLORS.amber, paddingHorizontal: 16, paddingVertical: 8, borderRadius: SIZES.radiusFull },
  upgradeBtnText: { fontSize: SIZES.small, fontFamily: FONTS.bold, color: '#fff' },
});
