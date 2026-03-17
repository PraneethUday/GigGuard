import React from 'react';
import { View, Text, StyleSheet, SafeAreaView, ScrollView, TouchableOpacity, Image, Platform } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, FONTS, SIZES } from '../constants/theme';

const HomeScreen = ({ navigation }) => {
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
          <TouchableOpacity style={styles.notificationBtn}>
            <Ionicons name="notifications-outline" size={24} color={COLORS.white} />
          </TouchableOpacity>
        </View>

        <View style={styles.greetingSection}>
          <Text style={styles.greetingText}>Good morning,</Text>
          <Text style={styles.userName}>Arjun Kumar</Text>
        </View>

        <View style={styles.activePlanCard}>
          <View style={styles.planInfo}>
            <Text style={styles.planLabel}>ACTIVE PLAN</Text>
            <Text style={styles.planName}>GigGuard Premium</Text>
            <Text style={styles.planValidity}>Valid till 31 Dec 2026</Text>
          </View>
          <View style={styles.protectedBadge}>
            <Text style={styles.protectedText}>Protected</Text>
          </View>
        </View>
      </View>

      {/* Main Content (Scrollable) */}
      <View style={styles.contentContainer}>
        <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollPadding}>
          {/* Stats Summary Card Group */}
          <View style={styles.statsRow}>
            <View style={styles.statCard}>
              <View style={[styles.statIconContainer, { backgroundColor: COLORS.milkTea + '15' }]}>
                <Ionicons name="star" size={20} color={COLORS.milkTea} />
              </View>
              <Text style={styles.statValue}>₹4,200</Text>
              <Text style={styles.statLabel}>Total claimed</Text>
            </View>
            <View style={styles.statCard}>
              <View style={[styles.statIconContainer, { backgroundColor: COLORS.primary + '15' }]}>
                <Ionicons name="checkmark-circle" size={20} color={COLORS.primary} />
              </View>
              <Text style={styles.statValue}>3</Text>
              <Text style={styles.statLabel}>Claims approved</Text>
            </View>
          </View>

          {/* Rain Detection Alert - Saffron/Orange theme */}
          <View style={[styles.alertBanner, { backgroundColor: COLORS.primary + '15', borderColor: COLORS.primary + '30' }]}>
            <View style={[styles.alertIconContainer, { backgroundColor: COLORS.primary }]}>
              <Ionicons name="umbrella-outline" size={24} color={COLORS.white} />
            </View>
            <View style={styles.alertTextContent}>
              <Text style={styles.alertTitle}>Rain detected in your area</Text>
              <Text style={[styles.alertSubtitle, { color: COLORS.primary }]}>Heavy rain forecast for next 4 hours</Text>
            </View>
          </View>

          {/* File Claim Button */}
          <TouchableOpacity style={styles.fileClaimBtn}>
            <Text style={styles.fileClaimBtnText}>File a claim now</Text>
          </TouchableOpacity>

          {/* Quick Actions Header */}
          <Text style={styles.sectionTitle}>QUICK ACTIONS</Text>

          {/* Quick Actions Grid Mockup */}
          <View style={styles.quickActionsRow}>
            <TouchableOpacity style={styles.quickActionItem}>
              <View style={[styles.actionIconBg, { backgroundColor: COLORS.accent + '15' }]}>
                <Ionicons name="document-text-outline" size={24} color={COLORS.accent} />
              </View>
              <Text style={styles.actionText}>My claims</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.quickActionItem}>
              <View style={[styles.actionIconBg, { backgroundColor: COLORS.milkTea + '15' }]}>
                <Ionicons name="time-outline" size={24} color={COLORS.milkTea} />
              </View>
              <Text style={styles.actionText}>History</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.quickActionItem}>
              <View style={[styles.actionIconBg, { backgroundColor: COLORS.primary + '15' }]}>
                <Ionicons name="shield-checkmark-outline" size={24} color={COLORS.primary} />
              </View>
              <Text style={styles.actionText}>Coverage</Text>
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
        <TouchableOpacity style={styles.navItem}>
          <Ionicons name="document-outline" size={24} color="#A1A1AA" />
          <Text style={styles.navText}>Claims</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem}>
          <Ionicons name="shield-outline" size={24} color="#A1A1AA" />
          <Text style={styles.navText}>Coverage</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem}>
          <Ionicons name="person-outline" size={24} color="#A1A1AA" />
          <Text style={styles.navText}>Profile</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#A01E22', // Back to Deep Red
  },
  topHeader: {
    paddingHorizontal: SIZES.padding,
    paddingTop: SIZES.base * 1.5,
    paddingBottom: SIZES.padding,
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: SIZES.base * 1.5,
  },
  logoAndTitle: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  logoCircle: {
    width: 36,
    height: 36,
    borderRadius: 12,
    backgroundColor: COLORS.accent,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: SIZES.base,
  },
  logoInner: {
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: COLORS.white,
    justifyContent: 'center',
    alignItems: 'center',
  },
  brandName: {
    fontSize: 20,
    fontFamily: FONTS.bold,
    color: COLORS.white,
  },
  notificationBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.12)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  greetingSection: {
    marginBottom: SIZES.padding * 0.5,
  },
  greetingText: {
    fontSize: 14,
    fontFamily: FONTS.medium,
    color: COLORS.white,
    opacity: 0.7,
  },
  userName: {
    fontSize: 22,
    fontFamily: FONTS.bold,
    color: COLORS.white,
    marginTop: SIZES.base * 0.2,
  },
  activePlanCard: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    borderRadius: SIZES.radius,
    padding: SIZES.padding * 0.5,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  planLabel: {
    fontSize: 12,
    fontFamily: FONTS.bold,
    color: COLORS.primary, // Sakura for "ACTIVE PLAN"
    letterSpacing: 1,
    marginBottom: 4,
  },
  planName: {
    fontSize: 16,
    fontFamily: FONTS.bold,
    color: COLORS.white,
  },
  planValidity: {
    fontSize: 13,
    fontFamily: FONTS.medium,
    color: COLORS.white,
    opacity: 0.5,
    marginTop: 2,
  },
  protectedBadge: {
    backgroundColor: COLORS.primary, // Amaranth status
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  protectedText: {
    fontSize: 12,
    fontFamily: FONTS.bold,
    color: COLORS.white,
  },
  contentContainer: {
    flex: 1,
    backgroundColor: COLORS.background, // Off-white for the body
    borderTopLeftRadius: 30,
    borderTopRightRadius: 30,
    marginTop: -20,
  },
  scrollPadding: {
    paddingHorizontal: SIZES.padding,
    paddingTop: SIZES.padding * 1.5,
    paddingBottom: 100, // Extra space for footer
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: SIZES.padding,
  },
  statCard: {
    flex: 0.48,
    backgroundColor: COLORS.white,
    borderRadius: SIZES.radius,
    padding: SIZES.padding * 0.75,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.04,
    shadowRadius: 8,
    elevation: 1,
  },
  statIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: SIZES.base,
  },
  statValue: {
    fontSize: 24,
    fontFamily: FONTS.bold,
    color: COLORS.secondary,
  },
  statLabel: {
    fontSize: 13,
    fontFamily: FONTS.medium,
    color: COLORS.text,
    opacity: 0.6,
    marginTop: 2,
  },
  alertBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#3B82F6' + '15', 
    borderRadius: SIZES.radius * 1.5,
    padding: SIZES.padding * 0.75,
    marginBottom: SIZES.padding,
    borderWidth: 1,
    borderColor: '#3B82F6' + '30',
  },
  alertIconContainer: {
    width: 48,
    height: 48,
    borderRadius: 16,
    backgroundColor: COLORS.primary,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: SIZES.padding * 0.5,
  },
  alertTextContent: {
    flex: 1,
  },
  alertTitle: {
    fontSize: 16,
    fontFamily: FONTS.bold,
    color: COLORS.secondary,
  },
  alertSubtitle: {
    fontSize: 13,
    fontFamily: FONTS.medium,
    color: COLORS.primary,
    marginTop: 2,
  },
  fileClaimBtn: {
    backgroundColor: COLORS.accent,
    borderRadius: SIZES.radius * 1.5,
    height: 56,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: SIZES.padding * 1.5,
    shadowColor: COLORS.accent,
    shadowOffset: { width: 0, height: 8 },
    elevation: 3,
  },
  fileClaimBtnText: {
    fontSize: 16,
    fontFamily: FONTS.bold,
    color: COLORS.secondary,
  },
  sectionTitle: {
    fontSize: 14,
    fontFamily: FONTS.bold,
    color: COLORS.secondary,
    letterSpacing: 1,
    marginBottom: SIZES.padding,
  },
  quickActionsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  quickActionItem: {
    flex: 1,
    alignItems: 'center',
  },
  actionIconBg: {
    width: 60,
    height: 60,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: COLORS.white,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
    marginBottom: SIZES.base,
  },
  actionText: {
    fontSize: 13,
    fontFamily: FONTS.bold,
    color: COLORS.secondary,
  },
  bottomNav: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    height: 80,
    backgroundColor: COLORS.white,
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingBottom: Platform.OS === 'ios' ? 20 : 0,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -4 },
    shadowOpacity: 0.05,
    shadowRadius: 10,
    elevation: 20,
  },
  navItem: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  navText: {
    fontSize: 11,
    fontFamily: FONTS.bold,
    color: '#A1A1AA',
    marginTop: 4,
  },
  activeDot: {
    width: 4,
    height: 4,
    borderRadius: 2,
    backgroundColor: COLORS.primary,
    marginTop: 2,
  },
});

export default HomeScreen;
