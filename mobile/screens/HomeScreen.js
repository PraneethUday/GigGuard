import React, { useEffect, useState, useCallback } from 'react';
import { View, Text, StyleSheet, SafeAreaView, ScrollView, TouchableOpacity, Platform, ActivityIndicator } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, FONTS, SIZES } from '../constants/theme';
import { useAuth } from '../context/AuthContext';
import * as api from '../lib/api';

const PLAN_NAMES = { basic: 'Basic Shield', standard: 'Standard Guard', pro: 'Pro Protect' };

function getGreeting() {
  const h = new Date().getHours();
  if (h < 12) return 'Good morning';
  if (h < 17) return 'Good afternoon';
  return 'Good evening';
}

export default function HomeScreen({ navigation }) {
  const { user } = useAuth();
  const [premium, setPremium] = useState(null);
  const [claims, setClaims] = useState([]);
  const [weather, setWeather] = useState(null);
  const [triggers, setTriggers] = useState([]);
  const [loadingData, setLoadingData] = useState(true);

  const fetchDashboard = useCallback(async () => {
    if (!user?.delivery_id) { setLoadingData(false); return; }
    setLoadingData(true);
    try {
      const [premRes, claimsRes] = await Promise.allSettled([
        api.predictPremium(user.delivery_id, user.city, user.tier || 'standard'),
        api.getWorkerClaims(user.delivery_id),
      ]);
      if (premRes.status === 'fulfilled') setPremium(premRes.value);
      if (claimsRes.status === 'fulfilled' && claimsRes.value?.data) setClaims(claimsRes.value.data);

      // Weather + triggers (non-critical — fire and forget)
      if (user.city) {
        api.getCityWeather(user.city)
          .then(res => res?.weather && setWeather(res.weather))
          .catch(() => {});
        api.getTriggerStatus()
          .then(res => {
            const cityData = res?.[user.city];
            if (cityData?.triggers_fired?.length) setTriggers(cityData.triggers_fired);
          })
          .catch(() => {});
      }
    } catch { /* handled per-call */ }
    finally { setLoadingData(false); }
  }, [user]);

  useEffect(() => { fetchDashboard(); }, [fetchDashboard]);

  const totalPaid = claims.filter(c => c.payout_status === 'paid').reduce((s, c) => s + (c.payout_amount || 0), 0);
  const approvedCount = claims.filter(c => c.payout_status === 'paid' || c.payout_status === 'approved').length;
  const tierLabel = PLAN_NAMES[user?.tier] || 'Standard Guard';
  const verified = user?.verification_status === 'verified';
  const weeklyPremium = premium
    ? (user?.autopay ? premium.weekly_premium_autopay : premium.weekly_premium)
    : null;

  return (
    <SafeAreaView style={styles.container}>
      {/* Top Header Section */}
      <View style={styles.topHeader}>
        <View style={styles.headerRow}>
          <View style={styles.logoAndTitle}>
            <View style={styles.logoCircle}>
              <View style={styles.logoInner}>
                <Ionicons name="location" size={16} color={COLORS.secondary} />
              </View>
            </View>
            <Text style={styles.brandName}>GigGuard</Text>
          </View>
          <TouchableOpacity style={styles.notificationBtn} onPress={fetchDashboard}>
            <Ionicons name="refresh-outline" size={22} color={COLORS.white} />
          </TouchableOpacity>
        </View>

        <View style={styles.greetingSection}>
          <Text style={styles.greetingText}>{getGreeting()},</Text>
          <Text style={styles.userName}>{user?.name || 'Partner'}</Text>
        </View>

        <View style={styles.activePlanCard}>
          <View style={styles.planInfo}>
            <Text style={styles.planLabel}>ACTIVE PLAN</Text>
            <Text style={styles.planName}>{tierLabel}</Text>
            <Text style={styles.planValidity}>
              {weeklyPremium != null ? `₹${Math.round(weeklyPremium)}/week` : 'Calculating...'}
              {premium?.max_payout ? `  ·  Max ₹${premium.max_payout}` : ''}
            </Text>
          </View>
          <View style={[styles.protectedBadge, !verified && { backgroundColor: '#d97706' }]}>
            <Text style={styles.protectedText}>{verified ? 'Protected' : 'Pending'}</Text>
          </View>
        </View>
      </View>

      {/* Main Content */}
      <View style={styles.contentContainer}>
        <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollPadding}>
          {loadingData && (
            <ActivityIndicator size="small" color={COLORS.primary} style={{ marginBottom: 16 }} />
          )}

          {/* Stats Summary */}
          <View style={styles.statsRow}>
            <View style={styles.statCard}>
              <View style={[styles.statIconContainer, { backgroundColor: 'rgba(245,158,11,0.15)' }]}>
                <Ionicons name="star" size={20} color="#F59E0B" />
              </View>
              <Text style={styles.statValue}>₹{Math.round(totalPaid).toLocaleString()}</Text>
              <Text style={styles.statLabel}>Total claimed</Text>
            </View>
            <View style={styles.statCard}>
              <View style={[styles.statIconContainer, { backgroundColor: COLORS.primary + '15' }]}>
                <Ionicons name="checkmark-circle" size={20} color={COLORS.primary} />
              </View>
              <Text style={styles.statValue}>{approvedCount}</Text>
              <Text style={styles.statLabel}>Claims approved</Text>
            </View>
          </View>

          {/* Weather Alert Banner (only if active triggers) */}
          {triggers.length > 0 && (
            <View style={[styles.alertBanner, { backgroundColor: COLORS.primary + '15', borderColor: COLORS.primary + '30' }]}>
              <View style={[styles.alertIconContainer, { backgroundColor: COLORS.primary }]}>
                <Ionicons name="warning-outline" size={24} color={COLORS.white} />
              </View>
              <View style={styles.alertTextContent}>
                <Text style={styles.alertTitle}>Active disruption in {user?.city}</Text>
                <Text style={[styles.alertSubtitle, { color: COLORS.primary }]}>
                  {triggers.map(t => t.description || t.trigger_type).join(', ')}
                </Text>
              </View>
            </View>
          )}

          {/* Weather info (if no triggers but weather data exists) */}
          {triggers.length === 0 && weather && (
            <View style={[styles.alertBanner, { backgroundColor: 'rgba(59,130,246,0.08)', borderColor: 'rgba(59,130,246,0.15)' }]}>
              <View style={[styles.alertIconContainer, { backgroundColor: '#3B82F6' }]}>
                <Ionicons name="partly-sunny-outline" size={24} color={COLORS.white} />
              </View>
              <View style={styles.alertTextContent}>
                <Text style={styles.alertTitle}>{weather.weather_main || 'Clear'} in {user?.city}</Text>
                <Text style={[styles.alertSubtitle, { color: '#3B82F6' }]}>
                  {weather.temperature ? `${weather.temperature}°C` : ''}
                  {weather.humidity ? `  ·  ${weather.humidity}% humidity` : ''}
                  {weather.aqi_index ? `  ·  AQI ${weather.aqi_index}` : ''}
                </Text>
              </View>
            </View>
          )}

          {/* Quick Actions */}
          <Text style={styles.sectionTitle}>QUICK ACTIONS</Text>
          <View style={styles.quickActionsRow}>
            <TouchableOpacity style={styles.quickActionItem} onPress={() => navigation.navigate('Claims')}>
              <View style={[styles.actionIconBg, { backgroundColor: COLORS.accent + '15' }]}>
                <Ionicons name="document-text-outline" size={24} color={COLORS.accent} />
              </View>
              <Text style={styles.actionText}>My Claims</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.quickActionItem} onPress={() => navigation.navigate('Policy')}>
              <View style={[styles.actionIconBg, { backgroundColor: COLORS.primary + '15' }]}>
                <Ionicons name="shield-checkmark-outline" size={24} color={COLORS.primary} />
              </View>
              <Text style={styles.actionText}>Coverage</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.quickActionItem} onPress={() => navigation.navigate('Profile')}>
              <View style={[styles.actionIconBg, { backgroundColor: 'rgba(245,158,11,0.15)' }]}>
                <Ionicons name="person-outline" size={24} color="#F59E0B" />
              </View>
              <Text style={styles.actionText}>Profile</Text>
            </TouchableOpacity>
          </View>
        </ScrollView>
      </View>

      {/* Bottom Navigation */}
      <View style={styles.bottomNav}>
        <TouchableOpacity style={styles.navItem}>
          <Ionicons name="home" size={24} color={COLORS.primary} />
          <Text style={[styles.navText, { color: COLORS.primary }]}>Home</Text>
          <View style={styles.activeDot} />
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem} onPress={() => navigation.navigate('Claims')}>
          <Ionicons name="document-outline" size={24} color="#A1A1AA" />
          <Text style={styles.navText}>Claims</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem} onPress={() => navigation.navigate('Policy')}>
          <Ionicons name="shield-outline" size={24} color="#A1A1AA" />
          <Text style={styles.navText}>Coverage</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem} onPress={() => navigation.navigate('Profile')}>
          <Ionicons name="person-outline" size={24} color="#A1A1AA" />
          <Text style={styles.navText}>Profile</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#A01E22' },
  topHeader: { paddingHorizontal: SIZES.padding, paddingTop: SIZES.base * 1.5, paddingBottom: SIZES.padding },
  headerRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: SIZES.base * 1.5 },
  logoAndTitle: { flexDirection: 'row', alignItems: 'center' },
  logoCircle: { width: 36, height: 36, borderRadius: 12, backgroundColor: COLORS.accent, justifyContent: 'center', alignItems: 'center', marginRight: SIZES.base },
  logoInner: { width: 20, height: 20, borderRadius: 10, backgroundColor: COLORS.white, justifyContent: 'center', alignItems: 'center' },
  brandName: { fontSize: 20, fontFamily: FONTS.bold, color: COLORS.white },
  notificationBtn: { width: 40, height: 40, borderRadius: 20, backgroundColor: 'rgba(255,255,255,0.12)', justifyContent: 'center', alignItems: 'center' },
  greetingSection: { marginBottom: SIZES.padding * 0.5 },
  greetingText: { fontSize: 14, fontFamily: FONTS.medium, color: COLORS.white, opacity: 0.7 },
  userName: { fontSize: 22, fontFamily: FONTS.bold, color: COLORS.white, marginTop: SIZES.base * 0.2 },
  activePlanCard: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.08)', borderRadius: SIZES.radius,
    padding: SIZES.padding * 0.5, borderWidth: 1, borderColor: 'rgba(255,255,255,0.1)',
  },
  planInfo: {},
  planLabel: { fontSize: 12, fontFamily: FONTS.bold, color: COLORS.primary, letterSpacing: 1, marginBottom: 4 },
  planName: { fontSize: 16, fontFamily: FONTS.bold, color: COLORS.white },
  planValidity: { fontSize: 13, fontFamily: FONTS.medium, color: COLORS.white, opacity: 0.5, marginTop: 2 },
  protectedBadge: { backgroundColor: COLORS.primary, paddingHorizontal: 12, paddingVertical: 6, borderRadius: 20 },
  protectedText: { fontSize: 12, fontFamily: FONTS.bold, color: COLORS.white },
  contentContainer: { flex: 1, backgroundColor: COLORS.background, borderTopLeftRadius: 30, borderTopRightRadius: 30, marginTop: -20 },
  scrollPadding: { paddingHorizontal: SIZES.padding, paddingTop: SIZES.padding * 1.5, paddingBottom: 100 },
  statsRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: SIZES.padding },
  statCard: {
    flex: 0.48, backgroundColor: COLORS.white, borderRadius: SIZES.radius, padding: SIZES.padding * 0.75,
    shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.04, shadowRadius: 8, elevation: 1,
  },
  statIconContainer: { width: 40, height: 40, borderRadius: 12, justifyContent: 'center', alignItems: 'center', marginBottom: SIZES.base },
  statValue: { fontSize: 24, fontFamily: FONTS.bold, color: COLORS.secondary },
  statLabel: { fontSize: 13, fontFamily: FONTS.medium, color: COLORS.text, opacity: 0.6, marginTop: 2 },
  alertBanner: {
    flexDirection: 'row', alignItems: 'center', borderRadius: SIZES.radius * 1.5,
    padding: SIZES.padding * 0.75, marginBottom: SIZES.padding, borderWidth: 1,
  },
  alertIconContainer: { width: 48, height: 48, borderRadius: 16, justifyContent: 'center', alignItems: 'center', marginRight: SIZES.padding * 0.5 },
  alertTextContent: { flex: 1 },
  alertTitle: { fontSize: 16, fontFamily: FONTS.bold, color: COLORS.secondary },
  alertSubtitle: { fontSize: 13, fontFamily: FONTS.medium, marginTop: 2 },
  sectionTitle: { fontSize: 14, fontFamily: FONTS.bold, color: COLORS.secondary, letterSpacing: 1, marginBottom: SIZES.padding },
  quickActionsRow: { flexDirection: 'row', justifyContent: 'space-between' },
  quickActionItem: { flex: 1, alignItems: 'center' },
  actionIconBg: {
    width: 60, height: 60, borderRadius: 20, justifyContent: 'center', alignItems: 'center',
    backgroundColor: COLORS.white, shadowColor: '#000', shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.05, shadowRadius: 8, elevation: 2, marginBottom: SIZES.base,
  },
  actionText: { fontSize: 13, fontFamily: FONTS.bold, color: COLORS.secondary },
  bottomNav: {
    position: 'absolute', bottom: 0, left: 0, right: 0, height: 80,
    backgroundColor: COLORS.white, flexDirection: 'row', justifyContent: 'space-around', alignItems: 'center',
    paddingBottom: Platform.OS === 'ios' ? 20 : 0, borderTopLeftRadius: 24, borderTopRightRadius: 24,
    shadowColor: '#000', shadowOffset: { width: 0, height: -4 }, shadowOpacity: 0.05, shadowRadius: 10, elevation: 20,
  },
  navItem: { alignItems: 'center', justifyContent: 'center' },
  navText: { fontSize: 11, fontFamily: FONTS.bold, color: '#A1A1AA', marginTop: 4 },
  activeDot: { width: 4, height: 4, borderRadius: 2, backgroundColor: COLORS.primary, marginTop: 2 },
});
