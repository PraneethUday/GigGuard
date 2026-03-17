import React from 'react';
import { View, Text, StyleSheet, SafeAreaView, Image, ScrollView, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, FONTS, SIZES } from '../constants/theme';
import Button from '../components/Button';

const LandingScreen = ({ navigation }) => {
  return (
    <SafeAreaView style={styles.container}>
      {/* Header Bar */}
      <View style={styles.header}>
        <View style={styles.headerLogoContainer}>
          <View style={styles.logoBackground}>
            <Image
              source={require('../assets/logo.png')}
              style={styles.headerLogo}
              resizeMode="contain"
            />
          </View>
          <Text style={styles.headerTitle}>GigGuard</Text>
        </View>
        <View style={styles.headerActions}>
          <TouchableOpacity onPress={() => navigation.navigate('Login')}>
            <Text style={styles.loginText}>Login</Text>
          </TouchableOpacity>
          <TouchableOpacity 
            style={styles.getStartedSmall}
            onPress={() => navigation.navigate('SignUp')}
          >
            <Text style={styles.getStartedSmallText}>Join</Text>
          </TouchableOpacity>
        </View>
      </View>

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
        {/* Hero Section */}
        <View style={styles.heroSection}>
          {/* Delivery Illustration */}
          <View style={styles.illustrationContainer}>
            <Image
              source={require('../assets/delivery.png')}
              style={styles.illustration}
              resizeMode="contain"
            />
          </View>
          
          <View style={styles.textContainer}>
            <View style={[styles.divider, { backgroundColor: COLORS.primary }]} />
            <Text style={styles.motto}>
              Protecting your hustle,{"\n"}rain or shine.
            </Text>
          </View>
        </View>

        {/* About Section Card */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Ionicons name="information-circle-outline" size={24} color={COLORS.primary} style={{ marginRight: 10 }} />
            <Text style={styles.sectionTitle}>About GigGuard</Text>
          </View>
          <Text style={styles.sectionText}>
            GigGuard is a premium insurance platform designed specifically for the unique needs of freelancers. We provide instant, parametric protection that traditional insurance misses.
          </Text>
        </View>

        {/* How It Works Card */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Ionicons name="git-branch-outline" size={24} color={COLORS.primary} style={{ marginRight: 10 }} />
            <Text style={styles.sectionTitle}>How It Works</Text>
          </View>
          <View style={styles.stepContainer}>
            <View style={styles.stepItem}>
              <View style={styles.stepNumber}><Text style={styles.stepNumberText}>1</Text></View>
              <Text style={styles.stepText}>Connect your gig platform and location services.</Text>
            </View>
            <View style={styles.stepItem}>
              <View style={styles.stepNumber}><Text style={styles.stepNumberText}>2</Text></View>
              <Text style={styles.stepText}>Our AI monitors weather triggers in real-time.</Text>
            </View>
            <View style={styles.stepItem}>
              <View style={styles.stepNumber}><Text style={styles.stepNumberText}>3</Text></View>
              <Text style={styles.stepText}>Receive instant payouts directly to your wallet.</Text>
            </View>
          </View>
        </View>

        {/* Benefits Grid Card */}
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Ionicons name="diamond-outline" size={24} color={COLORS.primary} style={{ marginRight: 10 }} />
            <Text style={styles.sectionTitle}>Why Choose Us</Text>
          </View>
          <View style={styles.benefitsGrid}>
            <View style={styles.benefitItem}>
              <View style={styles.benefitIconContainer}>
                <Ionicons name="flash-outline" size={22} color={COLORS.primary} />
              </View>
              <Text style={styles.benefitTitle}>Instant Pay</Text>
            </View>
            <View style={styles.benefitItem}>
              <View style={styles.benefitIconContainer}>
                <Ionicons name="analytics-outline" size={22} color={COLORS.primary} />
              </View>
              <Text style={styles.benefitTitle}>AI-Driven</Text>
            </View>
            <View style={styles.benefitItem}>
              <View style={styles.benefitIconContainer}>
                <Ionicons name="shield-outline" size={22} color={COLORS.primary} />
              </View>
              <Text style={styles.benefitTitle}>Secure</Text>
            </View>
            <View style={styles.benefitItem}>
              <View style={styles.benefitIconContainer}>
                <Ionicons name="people-outline" size={22} color={COLORS.primary} />
              </View>
              <Text style={styles.benefitTitle}>Community</Text>
            </View>
          </View>
        </View>

        {/* Footer */}
        <View style={styles.footer}>
          <Text style={styles.footerText}>© 2026 GigGuard Technologies. All rights reserved.</Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  scrollContent: {
    paddingHorizontal: SIZES.padding,
    paddingBottom: SIZES.padding * 2,
  },
  header: {
    height: 65,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: SIZES.padding,
    backgroundColor: '#A01E22', // Back to Deep Red Header
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
  },
  headerLogoContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  logoBackground: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: COLORS.white,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: SIZES.base,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
  },
  headerLogo: {
    width: 24,
    height: 24,
  },
  headerTitle: {
    fontSize: 18,
    fontFamily: FONTS.bold,
    color: COLORS.white,
    letterSpacing: 1,
  },
  headerActions: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  loginText: {
    fontFamily: FONTS.bold,
    color: COLORS.white,
    marginRight: SIZES.padding,
    fontSize: 14,
  },
  getStartedSmall: {
    backgroundColor: 'rgba(255, 255, 255, 0.25)', // Restore semi-transparent white
    paddingHorizontal: SIZES.padding,
    paddingVertical: SIZES.base * 0.8,
    borderRadius: SIZES.radius,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.4)',
  },
  getStartedSmallText: {
    color: COLORS.white,
    fontFamily: FONTS.bold,
    fontSize: 14,
  },
  heroSection: {
    alignItems: 'center',
    paddingVertical: SIZES.padding * 3,
    paddingHorizontal: SIZES.padding,
  },
  illustrationContainer: {
    width: '100%',
    height: 240, // Balanced size for a neater look
    justifyContent: 'center',
    alignItems: 'center',
    marginVertical: SIZES.base,
  },
  illustration: {
    width: '100%',
    height: '100%',
  },
  textContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
  },
  divider: {
    width: 60,
    height: 4,
    backgroundColor: '#A51C30', // Ruby Red Divider
    borderRadius: 2,
    marginBottom: SIZES.padding,
  },
  motto: {
    fontSize: SIZES.h2,
    fontFamily: FONTS.medium,
    color: '#1B1B3A', // Restoring dark motto text
    textAlign: 'center',
    lineHeight: 38,
    maxWidth: '90%',
  },
  buttonContainer: {
    width: '100%',
    marginBottom: SIZES.padding * 1.5,
  },
  button: {
    marginVertical: SIZES.base,
  },
  section: {
    backgroundColor: COLORS.surface,
    padding: SIZES.padding,
    borderRadius: SIZES.radius,
    marginBottom: SIZES.padding,
    borderWidth: 1,
    borderColor: 'rgba(27, 27, 58, 0.05)',
  },
  sectionTitle: {
    fontSize: SIZES.h2,
    fontFamily: FONTS.bold,
    color: COLORS.secondary,
    marginBottom: SIZES.padding * 0.5,
  },
  sectionText: {
    fontSize: SIZES.body,
    fontFamily: FONTS.medium,
    color: COLORS.text,
    lineHeight: 24,
    opacity: 0.8,
  },
  stepItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: SIZES.base * 1.5,
  },
  stepNumber: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: COLORS.primary,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: SIZES.base,
  },
  stepNumberText: {
    color: COLORS.white,
    fontFamily: FONTS.bold,
    fontSize: 14,
  },
  stepText: {
    flex: 1,
    fontSize: SIZES.body,
    fontFamily: FONTS.medium,
    color: COLORS.text,
    opacity: 0.8,
  },
  benefitRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: SIZES.padding * 0.5,
  },
  benefitItem: {
    flex: 0.48,
    alignItems: 'center',
    padding: SIZES.base,
    backgroundColor: COLORS.brookGreen + '15',
    borderRadius: SIZES.radius * 0.5,
  },
  benefitIcon: {
    marginBottom: SIZES.base * 0.5,
  },
  benefitTitle: {
    fontSize: SIZES.body * 0.9,
    fontFamily: FONTS.bold,
    color: COLORS.secondary,
    textAlign: 'center',
  },
  footer: {
    paddingVertical: SIZES.padding,
    alignItems: 'center',
  },
  footerText: {
    fontSize: 12,
    fontFamily: FONTS.medium,
    color: COLORS.text,
    opacity: 0.5,
  },
});

export default LandingScreen;
