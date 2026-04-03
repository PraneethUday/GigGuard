import React from 'react';
import {
  View, Text, StyleSheet, SafeAreaView,
  ScrollView, TouchableOpacity, Platform,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, FONTS, SIZES, SHADOWS } from '../constants/theme';

const RECENT_ACTIVITY = [
  { icon: 'refresh-circle', color: COLORS.primary, title: 'Policy renewed', sub: '3 days ago · ₹72 deducted', },
  { icon: 'shield-checkmark', color: COLORS.success, title: 'No disruptions this week', sub: 'All clear in Chennai South', },
  { icon: 'star', color: COLORS.amber, title: 'Loyalty milestone: 8 weeks!', sub: 'Discount now 8% — keep it up', },
];

const QuickAction = ({ icon, label, onPress, color = COLORS.primary }) => (
  <TouchableOpacity style={styles.quickAction} onPress={onPress} activeOpacity={0.75}>
    <View style={[styles.quickActionIcon, { backgroundColor: color + '18' }]}>
      <Ionicons name={icon} size={22} color={color} />
    </View>
    <Text style={styles.quickActionLabel}>{label}</Text>
  </TouchableOpacity>
);

const HomeScreen = ({ navigation }) => {
  return (
    <SafeAreaView style={styles.container}>
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scroll}>

        {/* Top bar */}
        <View style={styles.topBar}>
          <View>
            <Text style={styles.greeting}>Good morning,</Text>
            <Text style={styles.userName}>Arjun Kumar 👋</Text>
            <View style={styles.platformBadge}>
              <Text style={styles.platformBadgeText}>Swiggy + Amazon Flex</Text>
            </View>
          </View>
          <TouchableOpacity style={styles.notifBtn}>
            <Ionicons name="notifications-outline" size={22} color={COLORS.white} />
          </TouchableOpacity>
        </View>

        {/* Active Shield Card */}
        <View style={styles.shieldCard}>
          <View style={styles.shieldGlow} />

          <View style={styles.shieldCardTop}>
            <Text style={styles.shieldCardLabel}>THIS WEEK'S COVERAGE</Text>
            <View style={styles.activeBadge}>
              <View style={styles.activePulse} />
              <Text style={styles.activeBadgeText}>ACTIVE</Text>
            </View>
          </View>

          <Text style={styles.payoutAmount}>₹1,200</Text>
          <Text style={styles.payoutLabel}>Maximum weekly payout</Text>

          <View style={styles.shieldCardMeta}>
            <Text style={styles.metaText}>Week 8 of 12</Text>
            <View style={styles.metaDot} />
            <Text style={[styles.metaText, { color: COLORS.success }]}>Loyalty: 8% off</Text>
          </View>

          <View style={styles.progressTrack}>
            <View style={[styles.progressFill, { width: '67%' }]} />
          </View>
        </View>

        {/* Weather Risk */}
        <View style={styles.weatherCard}>
          <View style={styles.weatherLeft}>
            <View style={styles.weatherIconWrap}>
              <Ionicons name="rainy" size={22} color={COLORS.amber} />
            </View>
            <View>
              <Text style={styles.weatherCity}>Chennai</Text>
              <Text style={styles.weatherSub}>Rainfall: 12mm · AQI: 142 · 32°C</Text>
            </View>
          </View>
          <View style={styles.riskBadge}>
            <Text style={styles.riskBadgeText}>MOD RISK</Text>
          </View>
        </View>

        {/* Premium Card */}
        <View style={styles.premiumCard}>
          <View style={styles.premiumCardTop}>
            <View>
              <Text style={styles.premiumAmount}>₹72 <Text style={styles.premiumPer}>/week</Text></Text>
              <Text style={styles.premiumSub}>AutoPay active · next deduction Monday</Text>
            </View>
            <TouchableOpacity style={styles.payNowBtn}>
              <Text style={styles.payNowText}>Pay Now</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.premiumBreakdown}>
            {[
              ['Base', '₹70'],
              ['Weather', '+4%'],
              ['Loyalty', '−8%'],
              ['AutoPay', '−5%'],
            ].map(([k, v]) => (
              <View key={k} style={styles.breakdownItem}>
                <Text style={styles.breakdownKey}>{k}</Text>
                <Text style={[styles.breakdownVal, v.startsWith('−') && { color: COLORS.success }]}>{v}</Text>
              </View>
            ))}
          </View>
        </View>

        {/* Quick Actions */}
        <Text style={styles.sectionTitle}>Quick Actions</Text>
        <View style={styles.quickActionsRow}>
          <QuickAction icon="shield-outline" label="My Policy" onPress={() => navigation.navigate('Policy')} color={COLORS.primary} />
          <QuickAction icon="document-text-outline" label="Claims" onPress={() => navigation.navigate('Claims')} color={COLORS.amber} />
          <QuickAction icon="bar-chart-outline" label="Platform Data" color={COLORS.success} />
          <QuickAction icon="time-outline" label="History" color={COLORS.info} />
        </View>

        {/* Recent Activity */}
        <Text style={styles.sectionTitle}>Recent Activity</Text>
        <View style={styles.activityCard}>
          {RECENT_ACTIVITY.map((item, i) => (
            <View key={i} style={[styles.activityRow, i < RECENT_ACTIVITY.length - 1 && styles.activityBorder]}>
              <View style={[styles.activityIcon, { backgroundColor: item.color + '18' }]}>
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
        <TouchableOpacity style={styles.navItem} onPress={() => navigation.navigate('Claims')}>
          <Ionicons name="document-outline" size={22} color={COLORS.textFaint} />
          <Text style={styles.navText}>Claims</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem} onPress={() => navigation.navigate('Policy')}>
          <Ionicons name="shield-outline" size={22} color={COLORS.textFaint} />
          <Text style={styles.navText}>Policy</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem} onPress={() => navigation.navigate('Profile')}>
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

  topBar: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', paddingHorizontal: SIZES.padding, paddingTop: SIZES.padding, paddingBottom: SIZES.padding * 0.5 },
  greeting: { fontSize: SIZES.small, fontFamily: FONTS.medium, color: COLORS.textMuted },
  userName: { fontSize: SIZES.h2, fontFamily: FONTS.bold, color: COLORS.white, marginTop: 2 },
  platformBadge: { backgroundColor: COLORS.primaryContainer, paddingHorizontal: 10, paddingVertical: 4, borderRadius: SIZES.radiusFull, alignSelf: 'flex-start', marginTop: 6 },
  platformBadgeText: { fontSize: SIZES.tiny, fontFamily: FONTS.semiBold, color: COLORS.primaryDim },
  notifBtn: { width: 42, height: 42, borderRadius: 14, backgroundColor: COLORS.surfaceHigh, justifyContent: 'center', alignItems: 'center', marginTop: 4 },

  // Shield card
  shieldCard: { marginHorizontal: SIZES.padding, borderRadius: SIZES.radius * 1.5, backgroundColor: COLORS.primary, padding: SIZES.padding, marginBottom: SIZES.padding, overflow: 'hidden', ...SHADOWS.button },
  shieldGlow: { position: 'absolute', width: 200, height: 200, borderRadius: 100, backgroundColor: '#fff', opacity: 0.05, top: -80, right: -60 },
  shieldCardTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: SIZES.padding * 0.75 },
  shieldCardLabel: { fontSize: SIZES.tiny, fontFamily: FONTS.bold, color: 'rgba(255,255,255,0.6)', letterSpacing: 1 },
  activeBadge: { flexDirection: 'row', alignItems: 'center', gap: 5, backgroundColor: 'rgba(255,255,255,0.15)', paddingHorizontal: 10, paddingVertical: 4, borderRadius: SIZES.radiusFull },
  activePulse: { width: 7, height: 7, borderRadius: 4, backgroundColor: COLORS.success },
  activeBadgeText: { fontSize: SIZES.tiny, fontFamily: FONTS.bold, color: '#fff' },
  payoutAmount: { fontSize: 40, fontFamily: FONTS.bold, color: '#fff', marginBottom: 4 },
  payoutLabel: { fontSize: SIZES.small, fontFamily: FONTS.medium, color: 'rgba(255,255,255,0.65)', marginBottom: SIZES.padding },
  shieldCardMeta: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: SIZES.base },
  metaText: { fontSize: SIZES.small, fontFamily: FONTS.medium, color: 'rgba(255,255,255,0.7)' },
  metaDot: { width: 4, height: 4, borderRadius: 2, backgroundColor: 'rgba(255,255,255,0.3)' },
  progressTrack: { height: 6, backgroundColor: 'rgba(255,255,255,0.2)', borderRadius: 3, overflow: 'hidden' },
  progressFill: { height: '100%', backgroundColor: '#fff', borderRadius: 3 },

  // Weather
  weatherCard: { marginHorizontal: SIZES.padding, borderRadius: SIZES.radius * 1.2, backgroundColor: COLORS.surfaceContainer, padding: SIZES.padding * 0.85, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: SIZES.padding, borderWidth: 1, borderColor: COLORS.border },
  weatherLeft: { flexDirection: 'row', alignItems: 'center', gap: SIZES.padding * 0.6 },
  weatherIconWrap: { width: 42, height: 42, borderRadius: 12, backgroundColor: COLORS.amberContainer, justifyContent: 'center', alignItems: 'center' },
  weatherCity: { fontSize: 15, fontFamily: FONTS.bold, color: COLORS.white },
  weatherSub: { fontSize: SIZES.small, fontFamily: FONTS.medium, color: COLORS.textMuted, marginTop: 2 },
  riskBadge: { backgroundColor: COLORS.amberContainer, paddingHorizontal: 10, paddingVertical: 5, borderRadius: SIZES.radiusFull, borderWidth: 1, borderColor: COLORS.amber + '40' },
  riskBadgeText: { fontSize: SIZES.tiny, fontFamily: FONTS.bold, color: COLORS.amber, letterSpacing: 0.5 },

  // Premium
  premiumCard: { marginHorizontal: SIZES.padding, borderRadius: SIZES.radius * 1.2, backgroundColor: COLORS.surfaceContainer, padding: SIZES.padding, marginBottom: SIZES.padding, borderWidth: 1, borderColor: COLORS.border },
  premiumCardTop: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: SIZES.padding },
  premiumAmount: { fontSize: 26, fontFamily: FONTS.bold, color: COLORS.white },
  premiumPer: { fontSize: SIZES.body, fontFamily: FONTS.medium, color: COLORS.textMuted },
  premiumSub: { fontSize: SIZES.small, fontFamily: FONTS.medium, color: COLORS.textMuted, marginTop: 4 },
  payNowBtn: { backgroundColor: COLORS.primary, paddingHorizontal: 18, paddingVertical: 10, borderRadius: SIZES.radiusFull },
  payNowText: { fontSize: SIZES.small, fontFamily: FONTS.bold, color: '#fff' },
  premiumBreakdown: { flexDirection: 'row', justifyContent: 'space-between', borderTopWidth: 1, borderTopColor: COLORS.border, paddingTop: SIZES.padding * 0.75 },
  breakdownItem: { alignItems: 'center' },
  breakdownKey: { fontSize: SIZES.tiny, fontFamily: FONTS.medium, color: COLORS.textFaint, marginBottom: 4 },
  breakdownVal: { fontSize: SIZES.small, fontFamily: FONTS.bold, color: COLORS.white },

  // Quick Actions
  sectionTitle: { fontSize: SIZES.body, fontFamily: FONTS.bold, color: COLORS.white, paddingHorizontal: SIZES.padding, marginBottom: SIZES.padding, marginTop: SIZES.base },
  quickActionsRow: { flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: SIZES.padding, marginBottom: SIZES.padding },
  quickAction: { alignItems: 'center' },
  quickActionIcon: { width: 54, height: 54, borderRadius: 16, justifyContent: 'center', alignItems: 'center', marginBottom: SIZES.base, borderWidth: 1, borderColor: COLORS.border },
  quickActionLabel: { fontSize: SIZES.tiny, fontFamily: FONTS.semiBold, color: COLORS.textMuted, textAlign: 'center' },

  // Activity
  activityCard: { marginHorizontal: SIZES.padding, borderRadius: SIZES.radius * 1.2, backgroundColor: COLORS.surfaceContainer, borderWidth: 1, borderColor: COLORS.border, marginBottom: SIZES.padding },
  activityRow: { flexDirection: 'row', alignItems: 'center', gap: SIZES.padding * 0.75, padding: SIZES.padding * 0.85 },
  activityBorder: { borderBottomWidth: 1, borderBottomColor: COLORS.border },
  activityIcon: { width: 38, height: 38, borderRadius: 12, justifyContent: 'center', alignItems: 'center' },
  activityTitle: { fontSize: SIZES.small, fontFamily: FONTS.semiBold, color: COLORS.white, marginBottom: 2 },
  activitySub: { fontSize: SIZES.tiny, fontFamily: FONTS.medium, color: COLORS.textFaint },

  // Bottom Nav
  bottomNav: { position: 'absolute', bottom: 0, left: 0, right: 0, height: 72, backgroundColor: COLORS.surfaceContainer, flexDirection: 'row', justifyContent: 'space-around', alignItems: 'center', paddingBottom: Platform.OS === 'ios' ? 16 : 0, borderTopWidth: 1, borderTopColor: COLORS.border },
  navItem: { alignItems: 'center', justifyContent: 'center' },
  navText: { fontSize: SIZES.tiny, fontFamily: FONTS.semiBold, color: COLORS.textFaint, marginTop: 3 },
  navActiveDot: { width: 4, height: 4, borderRadius: 2, backgroundColor: COLORS.primary, marginTop: 2 },
});

export default HomeScreen;
