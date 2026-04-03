import React from 'react';
import {
  View, Text, StyleSheet, SafeAreaView,
  ScrollView, TouchableOpacity,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, FONTS, SIZES, SHADOWS } from '../constants/theme';

const FEATURES = [
  { icon: 'flash', label: 'Auto-Claims' },
  { icon: 'rainy', label: 'Weather Alerts' },
  { icon: 'cash', label: 'Instant Payouts' },
];

const PLATFORMS = ['Zomato', 'Swiggy', 'Amazon', 'Blinkit', 'Zepto'];

const StatCard = ({ icon, text }) => (
  <View style={styles.featurePill}>
    <Ionicons name={icon} size={14} color={COLORS.amber} />
    <Text style={styles.featurePillText}>{text}</Text>
  </View>
);

const LandingScreen = ({ navigation }) => {
  return (
    <SafeAreaView style={styles.container}>
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scroll}>

        {/* Header */}
        <View style={styles.header}>
          <View style={styles.logoRow}>
            <View style={styles.logoIcon}>
              <Ionicons name="shield-checkmark" size={18} color={COLORS.white} />
            </View>
            <Text style={styles.logoText}>GigGuard</Text>
          </View>
          <TouchableOpacity onPress={() => navigation.navigate('Login')}>
            <Text style={styles.loginLink}>Login</Text>
          </TouchableOpacity>
        </View>

        {/* Hero */}
        <View style={styles.heroSection}>
          <View style={styles.heroBadge}>
            <View style={styles.activeDot} />
            <Text style={styles.heroBadgeText}>AI-Powered Protection</Text>
          </View>

          <Text style={styles.heroTitle}>Your Income.{'\n'}Protected.</Text>
          <Text style={styles.heroSubtitle}>
            Automatic payouts when rain, AQI, or curfews stop your deliveries. No paperwork. No waiting.
          </Text>

          {/* Feature pills */}
          <View style={styles.featureRow}>
            {FEATURES.map(f => (
              <StatCard key={f.label} icon={f.icon} text={f.label} />
            ))}
          </View>
        </View>

        {/* Illustration / Shield card */}
        <View style={styles.shieldCard}>
          <View style={styles.shieldGlow} />
          <Ionicons name="shield-checkmark" size={64} color={COLORS.primary} style={{ zIndex: 1 }} />
          <Text style={styles.shieldCardTitle}>Income Shield Active</Text>
          <View style={styles.shieldMetrics}>
            <View style={styles.shieldMetric}>
              <Text style={styles.shieldMetricValue}>₹1,200</Text>
              <Text style={styles.shieldMetricLabel}>Max Weekly Cover</Text>
            </View>
            <View style={styles.shieldMetricDivider} />
            <View style={styles.shieldMetric}>
              <Text style={styles.shieldMetricValue}>15 min</Text>
              <Text style={styles.shieldMetricLabel}>Trigger Polling</Text>
            </View>
            <View style={styles.shieldMetricDivider} />
            <View style={styles.shieldMetric}>
              <Text style={styles.shieldMetricValue}>5 triggers</Text>
              <Text style={styles.shieldMetricLabel}>Events Covered</Text>
            </View>
          </View>
        </View>

        {/* How it works */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>How it works</Text>
          {[
            { n: '1', text: 'Register & verify your delivery platforms' },
            { n: '2', text: 'Pay ₹20–130/week based on your earnings & risk' },
            { n: '3', text: 'Auto-claims fire when disruptions hit your zone' },
          ].map(step => (
            <View key={step.n} style={styles.stepRow}>
              <View style={styles.stepNum}>
                <Text style={styles.stepNumText}>{step.n}</Text>
              </View>
              <Text style={styles.stepText}>{step.text}</Text>
            </View>
          ))}
        </View>

        {/* CTA */}
        <TouchableOpacity style={styles.ctaButton} onPress={() => navigation.navigate('SignUp')} activeOpacity={0.85}>
          <Text style={styles.ctaText}>Get Protected — Free Setup</Text>
          <Ionicons name="arrow-forward" size={18} color={COLORS.white} />
        </TouchableOpacity>

        <TouchableOpacity onPress={() => navigation.navigate('Login')} style={{ alignItems: 'center', marginTop: SIZES.padding }}>
          <Text style={styles.loginHint}>Already insured? <Text style={{ color: COLORS.primary }}>Log in</Text></Text>
        </TouchableOpacity>

        {/* Platform row */}
        <View style={styles.platformRow}>
          {PLATFORMS.map(p => (
            <View key={p} style={styles.platformPill}>
              <Text style={styles.platformText}>{p}</Text>
            </View>
          ))}
        </View>

        <Text style={styles.footer}>© 2026 GigGuard Technologies · IRDAI Registered</Text>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.surface },
  scroll: { paddingBottom: SIZES.padding * 3 },

  // Header
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: SIZES.padding, paddingTop: SIZES.padding, paddingBottom: SIZES.base },
  logoRow: { flexDirection: 'row', alignItems: 'center', gap: SIZES.base },
  logoIcon: { width: 34, height: 34, borderRadius: 10, backgroundColor: COLORS.primary, justifyContent: 'center', alignItems: 'center' },
  logoText: { fontSize: 18, fontFamily: FONTS.bold, color: COLORS.white },
  loginLink: { fontSize: 14, fontFamily: FONTS.semiBold, color: COLORS.primaryDim },

  // Hero
  heroSection: { paddingHorizontal: SIZES.padding, paddingTop: SIZES.padding * 1.5, paddingBottom: SIZES.padding },
  heroBadge: { flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: COLORS.primaryContainer, paddingHorizontal: 12, paddingVertical: 6, borderRadius: SIZES.radiusFull, alignSelf: 'flex-start', marginBottom: SIZES.padding },
  activeDot: { width: 7, height: 7, borderRadius: 4, backgroundColor: COLORS.success },
  heroBadgeText: { fontSize: SIZES.tiny, fontFamily: FONTS.semiBold, color: COLORS.primaryDim, letterSpacing: 0.5 },
  heroTitle: { fontSize: 38, fontFamily: FONTS.bold, color: COLORS.white, lineHeight: 46, marginBottom: SIZES.padding * 0.75, letterSpacing: -0.5 },
  heroSubtitle: { fontSize: SIZES.body, fontFamily: FONTS.regular, color: COLORS.textMuted, lineHeight: 24, marginBottom: SIZES.padding },
  featureRow: { flexDirection: 'row', gap: SIZES.base, flexWrap: 'wrap' },
  featurePill: { flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: COLORS.surfaceHigh, paddingHorizontal: 12, paddingVertical: 7, borderRadius: SIZES.radiusFull, borderWidth: 1, borderColor: COLORS.border },
  featurePillText: { fontSize: SIZES.small, fontFamily: FONTS.semiBold, color: COLORS.white },

  // Shield Card
  shieldCard: { marginHorizontal: SIZES.padding, borderRadius: SIZES.radius * 1.5, backgroundColor: COLORS.surfaceContainer, padding: SIZES.padding * 1.25, alignItems: 'center', marginBottom: SIZES.padding, overflow: 'hidden', ...SHADOWS.card },
  shieldGlow: { position: 'absolute', width: 180, height: 180, borderRadius: 90, backgroundColor: COLORS.primary, opacity: 0.08, top: -40, alignSelf: 'center' },
  shieldCardTitle: { fontSize: SIZES.h3, fontFamily: FONTS.bold, color: COLORS.white, marginTop: SIZES.base, marginBottom: SIZES.padding },
  shieldMetrics: { flexDirection: 'row', justifyContent: 'space-between', width: '100%', borderTopWidth: 1, borderTopColor: COLORS.border, paddingTop: SIZES.padding },
  shieldMetric: { flex: 1, alignItems: 'center' },
  shieldMetricValue: { fontSize: 16, fontFamily: FONTS.bold, color: COLORS.primary, marginBottom: 4 },
  shieldMetricLabel: { fontSize: SIZES.tiny, fontFamily: FONTS.medium, color: COLORS.textFaint, textAlign: 'center' },
  shieldMetricDivider: { width: 1, backgroundColor: COLORS.border },

  // Steps
  section: { paddingHorizontal: SIZES.padding, marginBottom: SIZES.padding },
  sectionTitle: { fontSize: SIZES.h3, fontFamily: FONTS.bold, color: COLORS.white, marginBottom: SIZES.padding },
  stepRow: { flexDirection: 'row', alignItems: 'flex-start', gap: SIZES.padding * 0.75, marginBottom: SIZES.padding * 0.75 },
  stepNum: { width: 28, height: 28, borderRadius: 14, backgroundColor: COLORS.primaryContainer, justifyContent: 'center', alignItems: 'center', borderWidth: 1, borderColor: COLORS.primary },
  stepNumText: { fontSize: SIZES.small, fontFamily: FONTS.bold, color: COLORS.primary },
  stepText: { flex: 1, fontSize: SIZES.body, fontFamily: FONTS.regular, color: COLORS.textMuted, lineHeight: 22, paddingTop: 3 },

  // CTA
  ctaButton: { marginHorizontal: SIZES.padding, height: 56, borderRadius: SIZES.radiusFull, backgroundColor: COLORS.primary, flexDirection: 'row', justifyContent: 'center', alignItems: 'center', gap: 10, ...SHADOWS.button },
  ctaText: { fontSize: 16, fontFamily: FONTS.bold, color: COLORS.white },
  loginHint: { fontSize: SIZES.small, fontFamily: FONTS.medium, color: COLORS.textFaint },

  // Platforms
  platformRow: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', gap: SIZES.base, paddingHorizontal: SIZES.padding, marginTop: SIZES.padding * 1.5 },
  platformPill: { backgroundColor: COLORS.surfaceHigh, paddingHorizontal: 12, paddingVertical: 5, borderRadius: SIZES.radiusFull, borderWidth: 1, borderColor: COLORS.border },
  platformText: { fontSize: SIZES.tiny, fontFamily: FONTS.semiBold, color: COLORS.textFaint },

  footer: { textAlign: 'center', fontSize: 11, fontFamily: FONTS.regular, color: COLORS.textFaint, marginTop: SIZES.padding, paddingHorizontal: SIZES.padding },
});

export default LandingScreen;
