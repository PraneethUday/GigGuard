import React, { useState } from 'react';
import {
  View, Text, StyleSheet, SafeAreaView,
  ScrollView, TouchableOpacity,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, FONTS, SIZES, SHADOWS } from '../constants/theme';

const CLAIMS = [
  { id: 'CLM-4821', type: 'Heavy Rain', icon: 'rainy', date: 'Mar 28, 2026', zone: 'Chennai South', hours: 4, amount: 175, status: 'paid', time: '9 min 32 sec' },
  { id: 'CLM-4755', type: 'AQI Alert', icon: 'cloud', date: 'Mar 15, 2026', zone: 'Chennai Central', hours: 6, amount: 263, status: 'paid', time: '11 min' },
  { id: 'CLM-4690', type: 'Curfew', icon: 'ban', date: 'Mar 2, 2026', zone: 'Chennai North', hours: 3, amount: 131, status: 'paid', time: '7 min 44 sec' },
  { id: 'CLM-4610', type: 'Heavy Rain', icon: 'rainy', date: 'Feb 21, 2026', zone: 'Chennai South', hours: 5, amount: 219, status: 'rejected', reason: 'Active deliveries detected during window' },
];

const STATUS = {
  paid: { bg: '#0A2E18', color: COLORS.success, label: 'Paid', icon: 'checkmark-circle' },
  'under-review': { bg: '#2A1A0A', color: COLORS.amber, label: 'Processing', icon: 'time' },
  rejected: { bg: '#2E0A0A', color: COLORS.error, label: 'Rejected', icon: 'close-circle' },
};

const FILTERS = ['all', 'paid', 'under-review', 'rejected'];

export default function ClaimsScreen({ navigation }) {
  const [filter, setFilter] = useState('all');
  const filtered = filter === 'all' ? CLAIMS : CLAIMS.filter(c => c.status === filter);
  const totalPaid = CLAIMS.filter(c => c.status === 'paid').reduce((a, c) => a + c.amount, 0);
  const paidCount = CLAIMS.filter(c => c.status === 'paid').length;

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backBtn}>
          <Ionicons name="arrow-back" size={20} color={COLORS.white} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Claims</Text>
        <View style={{ width: 40 }} />
      </View>

      <ScrollView showsVerticalScrollIndicator={false}>

        {/* Live Disruption Alert */}
        <View style={styles.liveAlert}>
          <View style={styles.liveAlertTop}>
            <View style={styles.liveDot} />
            <Text style={styles.liveAlertTitle}>DISRUPTION DETECTED</Text>
            <View style={styles.liveBadge}><Text style={styles.liveBadgeText}>LIVE</Text></View>
          </View>
          <Text style={styles.liveAlertSub}>Heavy rainfall in Chennai — 68mm/24h (Red Alert IMD)</Text>
          <Text style={styles.liveAlertSub}>Auto-claim initiated for your policy</Text>

          {/* Progress steps */}
          <View style={styles.claimSteps}>
            {['Detected', 'Validated', 'Processing...', 'Payout'].map((s, i) => {
              const done = i < 2;
              const active = i === 2;
              return (
                <React.Fragment key={s}>
                  <View style={styles.claimStep}>
                    <View style={[styles.claimStepDot, done && styles.claimStepDone, active && styles.claimStepActive]}>
                      {done && <Ionicons name="checkmark" size={9} color="#fff" />}
                    </View>
                    <Text style={[styles.claimStepText, active && { color: COLORS.amber }]}>{s}</Text>
                  </View>
                  {i < 3 && <View style={[styles.claimStepLine, done && { backgroundColor: COLORS.success }]} />}
                </React.Fragment>
              );
            })}
          </View>

          <View style={styles.progressBar}><View style={[styles.progressFill, { width: '60%' }]} /></View>
          <Text style={styles.estPayout}>Est. payout: <Text style={{ color: COLORS.success, fontFamily: FONTS.bold }}>₹175</Text> in ~8 minutes</Text>
        </View>

        {/* Summary banner */}
        <View style={styles.summaryRow}>
          <View style={styles.summaryItem}>
            <Text style={styles.summaryValue}>{CLAIMS.length}</Text>
            <Text style={styles.summaryLabel}>Total</Text>
          </View>
          <View style={styles.summaryDivider} />
          <View style={styles.summaryItem}>
            <Text style={[styles.summaryValue, { color: COLORS.success }]}>₹{totalPaid}</Text>
            <Text style={styles.summaryLabel}>Received</Text>
          </View>
          <View style={styles.summaryDivider} />
          <View style={styles.summaryItem}>
            <Text style={styles.summaryValue}>{paidCount}</Text>
            <Text style={styles.summaryLabel}>Paid</Text>
          </View>
          <View style={styles.summaryDivider} />
          <View style={styles.summaryItem}>
            <Text style={[styles.summaryValue, { color: COLORS.error }]}>{CLAIMS.length - paidCount}</Text>
            <Text style={styles.summaryLabel}>Rejected</Text>
          </View>
        </View>

        {/* Info pill */}
        <View style={styles.infoPill}>
          <Ionicons name="information-circle-outline" size={16} color={COLORS.info} />
          <Text style={styles.infoPillText}>All claims are triggered automatically — no filing required</Text>
        </View>

        {/* Filter chips */}
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.filterRow} contentContainerStyle={{ paddingHorizontal: SIZES.padding, gap: 8 }}>
          {FILTERS.map(f => (
            <TouchableOpacity
              key={f}
              style={[styles.filterChip, filter === f && styles.filterChipActive]}
              onPress={() => setFilter(f)}
            >
              <Text style={[styles.filterChipText, filter === f && { color: COLORS.primary }]}>
                {f === 'all' ? 'All' : f === 'paid' ? '✓ Paid' : f === 'under-review' ? '⏳ Processing' : '✕ Rejected'}
              </Text>
            </TouchableOpacity>
          ))}
        </ScrollView>

        {/* Claims list */}
        <View style={styles.claimsList}>
          {filtered.map(claim => {
            const sc = STATUS[claim.status] || STATUS.paid;
            return (
              <View key={claim.id} style={styles.claimCard}>
                <View style={styles.claimTop}>
                  <View style={[styles.claimIconWrap, { backgroundColor: sc.bg }]}>
                    <Ionicons name={claim.icon} size={20} color={sc.color} />
                  </View>
                  <View style={{ flex: 1 }}>
                    <Text style={styles.claimType}>{claim.type}</Text>
                    <Text style={styles.claimDate}>{claim.date} · {claim.zone}</Text>
                  </View>
                  <View style={[styles.statusBadge, { backgroundColor: sc.bg }]}>
                    <Text style={[styles.statusText, { color: sc.color }]}>{sc.label}</Text>
                  </View>
                </View>

                <View style={styles.claimBottom}>
                  <View style={styles.claimDetail}>
                    <Text style={styles.detailLabel}>Disrupted</Text>
                    <Text style={styles.detailValue}>{claim.hours} hrs</Text>
                  </View>
                  <View style={styles.claimDetail}>
                    <Text style={styles.detailLabel}>Claim ID</Text>
                    <Text style={styles.detailValue}>{claim.id}</Text>
                  </View>
                  <View style={styles.claimDetail}>
                    <Text style={styles.detailLabel}>Payout</Text>
                    <Text style={[styles.detailValue, { color: claim.status === 'paid' ? COLORS.success : COLORS.error }]}>
                      {claim.status === 'rejected' ? '—' : `₹${claim.amount}`}
                    </Text>
                  </View>
                </View>

                {claim.status === 'paid' && (
                  <View style={styles.claimFooter}>
                    <Ionicons name="flash" size={12} color={COLORS.success} />
                    <Text style={styles.claimFooterText}>UPI credited in {claim.time}</Text>
                  </View>
                )}
                {claim.status === 'rejected' && (
                  <View style={styles.claimFooterRejected}>
                    <Ionicons name="alert-circle" size={12} color={COLORS.error} />
                    <Text style={styles.claimFooterRejectedText}>Reason: {claim.reason}</Text>
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

  header: { height: 56, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: SIZES.padding, borderBottomWidth: 1, borderBottomColor: COLORS.border },
  backBtn: { width: 40, height: 40, borderRadius: 12, backgroundColor: COLORS.surfaceHigh, justifyContent: 'center', alignItems: 'center' },
  headerTitle: { fontSize: SIZES.h3, fontFamily: FONTS.bold, color: COLORS.white },

  // Live Alert
  liveAlert: { margin: SIZES.padding, borderRadius: SIZES.radius * 1.5, backgroundColor: COLORS.amberContainer, borderWidth: 1, borderColor: COLORS.amber + '50', padding: SIZES.padding },
  liveAlertTop: { flexDirection: 'row', alignItems: 'center', gap: 8, marginBottom: SIZES.base },
  liveDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: COLORS.amber },
  liveAlertTitle: { flex: 1, fontSize: SIZES.small, fontFamily: FONTS.bold, color: COLORS.amber, letterSpacing: 0.5 },
  liveBadge: { backgroundColor: COLORS.amber, paddingHorizontal: 8, paddingVertical: 3, borderRadius: SIZES.radiusFull },
  liveBadgeText: { fontSize: 9, fontFamily: FONTS.bold, color: '#fff', letterSpacing: 0.5 },
  liveAlertSub: { fontSize: SIZES.small, fontFamily: FONTS.medium, color: COLORS.amberDim, marginBottom: 4 },

  claimSteps: { flexDirection: 'row', alignItems: 'center', marginVertical: SIZES.padding * 0.75 },
  claimStep: { alignItems: 'center', gap: 4 },
  claimStepDot: { width: 18, height: 18, borderRadius: 9, backgroundColor: COLORS.surfaceHighest, justifyContent: 'center', alignItems: 'center', borderWidth: 1, borderColor: COLORS.border },
  claimStepDone: { backgroundColor: COLORS.success, borderColor: COLORS.success },
  claimStepActive: { backgroundColor: COLORS.amber, borderColor: COLORS.amber },
  claimStepText: { fontSize: 9, fontFamily: FONTS.medium, color: COLORS.textFaint },
  claimStepLine: { flex: 1, height: 2, backgroundColor: COLORS.border, marginBottom: 12 },
  progressBar: { height: 4, backgroundColor: COLORS.surfaceHighest, borderRadius: 2, overflow: 'hidden', marginTop: SIZES.base },
  progressFill: { height: '100%', backgroundColor: COLORS.amber, borderRadius: 2 },
  estPayout: { fontSize: SIZES.small, fontFamily: FONTS.medium, color: COLORS.textMuted, marginTop: SIZES.base },

  // Summary
  summaryRow: { flexDirection: 'row', backgroundColor: COLORS.surfaceContainer, marginHorizontal: SIZES.padding, borderRadius: SIZES.radius * 1.2, padding: SIZES.padding, marginBottom: SIZES.padding * 0.75, borderWidth: 1, borderColor: COLORS.border },
  summaryItem: { flex: 1, alignItems: 'center' },
  summaryValue: { fontSize: 20, fontFamily: FONTS.bold, color: COLORS.white, marginBottom: 2 },
  summaryLabel: { fontSize: SIZES.tiny, fontFamily: FONTS.medium, color: COLORS.textFaint },
  summaryDivider: { width: 1, backgroundColor: COLORS.border },

  infoPill: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingHorizontal: SIZES.padding, marginBottom: SIZES.padding * 0.75 },
  infoPillText: { flex: 1, fontSize: SIZES.small, fontFamily: FONTS.medium, color: COLORS.info, lineHeight: 18 },

  filterRow: { marginBottom: SIZES.padding },
  filterChip: { paddingHorizontal: 16, paddingVertical: 8, borderRadius: SIZES.radiusFull, backgroundColor: COLORS.surfaceContainer, borderWidth: 1, borderColor: COLORS.border },
  filterChipActive: { backgroundColor: COLORS.primaryContainer, borderColor: COLORS.primary },
  filterChipText: { fontSize: SIZES.small, fontFamily: FONTS.semiBold, color: COLORS.textFaint },

  // Claims list
  claimsList: { paddingHorizontal: SIZES.padding, paddingBottom: SIZES.padding * 2 },
  claimCard: { backgroundColor: COLORS.surfaceContainer, borderRadius: SIZES.radius * 1.2, borderWidth: 1, borderColor: COLORS.border, marginBottom: 12, overflow: 'hidden' },
  claimTop: { flexDirection: 'row', alignItems: 'center', gap: 12, padding: SIZES.padding * 0.85, borderBottomWidth: 1, borderBottomColor: COLORS.border },
  claimIconWrap: { width: 42, height: 42, borderRadius: 12, justifyContent: 'center', alignItems: 'center' },
  claimType: { fontSize: SIZES.body, fontFamily: FONTS.bold, color: COLORS.white, marginBottom: 2 },
  claimDate: { fontSize: SIZES.small, fontFamily: FONTS.medium, color: COLORS.textFaint },
  statusBadge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: SIZES.radiusFull },
  statusText: { fontSize: SIZES.tiny, fontFamily: FONTS.bold },
  claimBottom: { flexDirection: 'row', justifyContent: 'space-around', padding: SIZES.padding * 0.75, borderBottomWidth: 1, borderBottomColor: COLORS.border },
  claimDetail: { alignItems: 'center' },
  detailLabel: { fontSize: SIZES.tiny, fontFamily: FONTS.medium, color: COLORS.textFaint, marginBottom: 3, textTransform: 'uppercase', letterSpacing: 0.3 },
  detailValue: { fontSize: SIZES.small, fontFamily: FONTS.bold, color: COLORS.white },
  claimFooter: { flexDirection: 'row', alignItems: 'center', gap: 6, padding: SIZES.padding * 0.6, paddingHorizontal: SIZES.padding * 0.85, backgroundColor: COLORS.successContainer },
  claimFooterText: { fontSize: SIZES.tiny, fontFamily: FONTS.semiBold, color: COLORS.success },
  claimFooterRejected: { flexDirection: 'row', alignItems: 'center', gap: 6, padding: SIZES.padding * 0.6, paddingHorizontal: SIZES.padding * 0.85, backgroundColor: COLORS.errorContainer },
  claimFooterRejectedText: { fontSize: SIZES.tiny, fontFamily: FONTS.semiBold, color: COLORS.error, flex: 1 },
});
