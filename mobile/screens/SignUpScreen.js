import React, { useState } from 'react';
import {
  View, Text, StyleSheet, SafeAreaView,
  ScrollView, TouchableOpacity, TextInput,
  KeyboardAvoidingView, Platform, Switch,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, FONTS, SIZES, SHADOWS } from '../constants/theme';

const ALL_PLATFORMS = [
  { id: 'swiggy', name: 'Swiggy' },
  { id: 'zomato', name: 'Zomato' },
  { id: 'amazon', name: 'Amazon Flex' },
  { id: 'blinkit', name: 'Blinkit' },
  { id: 'zepto', name: 'Zepto' },
  { id: 'meesho', name: 'Meesho' },
  { id: 'porter', name: 'Porter' },
  { id: 'dunzo', name: 'Dunzo' },
];

const CITIES = ['Chennai', 'Delhi', 'Mumbai', 'Bangalore', 'Hyderabad', 'Pune', 'Kolkata', 'Ahmedabad'];

const TIERS = [
  { id: 'basic', name: 'Basic Shield', price: '₹20–40/wk', payout: '₹500' },
  { id: 'standard', name: 'Standard Guard', price: '₹40–80/wk', payout: '₹1,200', recommended: true },
  { id: 'pro', name: 'Pro Protect', price: '₹80–130/wk', payout: '₹2,500' },
];

const STEPS = ['Platform', 'Personal', 'Documents', 'Coverage'];

const StepIndicator = ({ current }) => (
  <View style={styles.stepper}>
    {STEPS.map((s, i) => {
      const done = i + 1 < current;
      const active = i + 1 === current;
      return (
        <React.Fragment key={s}>
          <View style={styles.stepItem}>
            <View style={[styles.stepCircle, done && styles.stepDone, active && styles.stepActive]}>
              {done
                ? <Ionicons name="checkmark" size={12} color="#fff" />
                : <Text style={[styles.stepNum, active && { color: '#fff' }]}>{i + 1}</Text>
              }
            </View>
            <Text style={[styles.stepLabel, active && { color: COLORS.white }]}>{s}</Text>
          </View>
          {i < STEPS.length - 1 && (
            <View style={[styles.stepLine, done && { backgroundColor: COLORS.primary }]} />
          )}
        </React.Fragment>
      );
    })}
  </View>
);

const Field = ({ label, placeholder, value, onChangeText, keyboardType, secureTextEntry, hint }) => (
  <View style={styles.fieldWrap}>
    <Text style={styles.fieldLabel}>{label}</Text>
    <TextInput
      style={styles.fieldInput}
      placeholder={placeholder}
      placeholderTextColor={COLORS.textFaint}
      value={value}
      onChangeText={onChangeText}
      keyboardType={keyboardType || 'default'}
      secureTextEntry={secureTextEntry}
      autoCapitalize="none"
    />
    {hint ? <Text style={styles.fieldHint}>{hint}</Text> : null}
  </View>
);

const SignUpScreen = ({ navigation }) => {
  const [step, setStep] = useState(1);
  const [form, setForm] = useState({
    platforms: [],
    name: '', age: '', phone: '', email: '', password: '', confirmPassword: '',
    city: '', area: '', deliveryId: '',
    pan: '', aadhaar: '', upi: '', bank: '',
    consent: false, gpsConsent: false, autopay: false,
    tier: 'standard',
  });

  const update = (key, val) => setForm(p => ({ ...p, [key]: val }));
  const togglePlatform = id => setForm(p => ({
    ...p,
    platforms: p.platforms.includes(id)
      ? p.platforms.filter(x => x !== id)
      : [...p.platforms, id],
  }));

  const nextStep = () => setStep(s => Math.min(s + 1, 4));
  const prevStep = () => setStep(s => Math.max(s - 1, 1));

  const handleSubmit = () => {
    console.log('Registering', form);
    navigation.navigate('Home');
  };

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : 'height'} style={{ flex: 1 }}>
        <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scroll}>

          {/* Header */}
          <View style={styles.header}>
            <TouchableOpacity onPress={step === 1 ? () => navigation.goBack() : prevStep} style={styles.backBtn}>
              <Ionicons name="arrow-back" size={20} color={COLORS.white} />
            </TouchableOpacity>
            <Text style={styles.headerTitle}>Create Account</Text>
            <View style={{ width: 40 }} />
          </View>

          <StepIndicator current={step} />

          {/* ── STEP 1: Platform ── */}
          {step === 1 && (
            <View style={styles.stepContent}>
              <Text style={styles.stepHeading}>Choose your platforms</Text>
              <Text style={styles.stepSub}>Select all delivery platforms you work with.</Text>

              <View style={styles.platformGrid}>
                {ALL_PLATFORMS.map(p => {
                  const sel = form.platforms.includes(p.id);
                  return (
                    <TouchableOpacity
                      key={p.id}
                      style={[styles.platformCard, sel && styles.platformCardSelected]}
                      onPress={() => togglePlatform(p.id)}
                      activeOpacity={0.75}
                    >
                      <Text style={[styles.platformCardName, sel && { color: COLORS.primary }]}>{p.name}</Text>
                      {sel && <Text style={styles.platformSelected}>Selected ✓</Text>}
                    </TouchableOpacity>
                  );
                })}
              </View>

              {form.platforms.length > 0 && (
                <View style={styles.selectionBadge}>
                  <Ionicons name="checkmark-circle" size={16} color={COLORS.success} />
                  <Text style={styles.selectionBadgeText}>
                    {form.platforms.length} platform{form.platforms.length > 1 ? 's' : ''} selected
                  </Text>
                </View>
              )}

              <TouchableOpacity
                style={[styles.ctaBtn, form.platforms.length === 0 && { opacity: 0.4 }]}
                onPress={nextStep}
                disabled={form.platforms.length === 0}
              >
                <Text style={styles.ctaBtnText}>Continue →</Text>
              </TouchableOpacity>
            </View>
          )}

          {/* ── STEP 2: Personal Info ── */}
          {step === 2 && (
            <View style={styles.stepContent}>
              <Text style={styles.stepHeading}>Personal information</Text>

              <Field label="Full Name" placeholder="Your full name" value={form.name} onChangeText={v => update('name', v)} />
              <View style={styles.row}>
                <View style={{ flex: 1, marginRight: 10 }}>
                  <Field label="Age" placeholder="25" value={form.age} onChangeText={v => update('age', v)} keyboardType="numeric" />
                </View>
                <View style={{ flex: 2 }}>
                  <Field label="Phone" placeholder="+91 98765 43210" value={form.phone} onChangeText={v => update('phone', v)} keyboardType="phone-pad" />
                </View>
              </View>
              <Field label="Email Address" placeholder="you@example.com" value={form.email} onChangeText={v => update('email', v)} />
              <View style={styles.row}>
                <View style={{ flex: 1, marginRight: 10 }}>
                  <Field label="Password" placeholder="Min. 6 chars" value={form.password} onChangeText={v => update('password', v)} secureTextEntry />
                </View>
                <View style={{ flex: 1 }}>
                  <Field label="Confirm Password" placeholder="Re-enter" value={form.confirmPassword} onChangeText={v => update('confirmPassword', v)} secureTextEntry />
                </View>
              </View>

              <Text style={styles.fieldLabel}>City</Text>
              <ScrollView horizontal showsHorizontalScrollIndicator={false} style={{ marginBottom: SIZES.padding }}>
                {CITIES.map(c => (
                  <TouchableOpacity
                    key={c}
                    style={[styles.cityPill, form.city === c && styles.cityPillSelected]}
                    onPress={() => update('city', c)}
                  >
                    <Text style={[styles.cityPillText, form.city === c && { color: COLORS.primary }]}>{c}</Text>
                  </TouchableOpacity>
                ))}
              </ScrollView>

              <Field label="Delivery Area / Zone" placeholder="e.g. Koramangala" value={form.area} onChangeText={v => update('area', v)} />
              <Field label="Primary Delivery Partner ID" placeholder="Your ID from the platform app" value={form.deliveryId} onChangeText={v => update('deliveryId', v)} hint="Used to verify your worker status on the selected platforms." />

              <View style={styles.navRow}>
                <TouchableOpacity style={styles.ghostBtn} onPress={prevStep}><Text style={styles.ghostBtnText}>← Back</Text></TouchableOpacity>
                <TouchableOpacity style={styles.ctaBtnSmall} onPress={nextStep}><Text style={styles.ctaBtnText}>Continue →</Text></TouchableOpacity>
              </View>
            </View>
          )}

          {/* ── STEP 3: Documents ── */}
          {step === 3 && (
            <View style={styles.stepContent}>
              <Text style={styles.stepHeading}>Identity & Payment</Text>

              <View style={styles.row}>
                <View style={{ flex: 1, marginRight: 10 }}>
                  <Field label="PAN Card" placeholder="ABCDE1234F" value={form.pan} onChangeText={v => update('pan', v.toUpperCase())} />
                </View>
                <View style={{ flex: 1 }}>
                  <Field label="Aadhaar Number" placeholder="1234 5678 9012" value={form.aadhaar} onChangeText={v => update('aadhaar', v)} keyboardType="numeric" />
                </View>
              </View>

              <View style={styles.secureNote}>
                <Ionicons name="lock-closed" size={14} color={COLORS.primary} />
                <Text style={styles.secureNoteText}>Your documents are encrypted and secure</Text>
              </View>

              <Field label="UPI ID (for payouts)" placeholder="yourname@upi" value={form.upi} onChangeText={v => update('upi', v)} />
              <Field label="Bank Account (optional)" placeholder="IFSC + Account Number" value={form.bank} onChangeText={v => update('bank', v)} />

              {/* Consent section */}
              <View style={styles.consentCard}>
                <Text style={styles.consentTitle}>Consent & Authorisations</Text>
                {[
                  { key: 'consent', label: 'I authorise GigGuard to monitor weather and disruption data in my delivery zone for insurance claims.' },
                  { key: 'gpsConsent', label: 'I authorise GPS location validation during disruption events for fraud prevention.' },
                  { key: 'autopay', label: 'Enable AutoPay — auto-deduct weekly premium from platform payout.', tag: '5% discount' },
                ].map(item => (
                  <View key={item.key} style={styles.consentRow}>
                    <Switch
                      value={!!form[item.key]}
                      onValueChange={v => update(item.key, v)}
                      trackColor={{ true: COLORS.primary, false: COLORS.surfaceHighest }}
                      thumbColor={COLORS.white}
                    />
                    <Text style={styles.consentLabel}>{item.label}</Text>
                    {item.tag && (
                      <View style={styles.discountTag}>
                        <Text style={styles.discountTagText}>{item.tag}</Text>
                      </View>
                    )}
                  </View>
                ))}
              </View>

              <View style={styles.navRow}>
                <TouchableOpacity style={styles.ghostBtn} onPress={prevStep}><Text style={styles.ghostBtnText}>← Back</Text></TouchableOpacity>
                <TouchableOpacity style={styles.ctaBtnSmall} onPress={nextStep}><Text style={styles.ctaBtnText}>Continue →</Text></TouchableOpacity>
              </View>
            </View>
          )}

          {/* ── STEP 4: Coverage ── */}
          {step === 4 && (
            <View style={styles.stepContent}>
              <Text style={styles.stepHeading}>Choose your coverage</Text>

              {TIERS.map(t => {
                const sel = form.tier === t.id;
                return (
                  <TouchableOpacity
                    key={t.id}
                    style={[styles.tierCard, sel && styles.tierCardSelected]}
                    onPress={() => update('tier', t.id)}
                    activeOpacity={0.8}
                  >
                    {t.recommended && (
                      <View style={styles.recommendedBadge}>
                        <Text style={styles.recommendedText}>Recommended</Text>
                      </View>
                    )}
                    <View style={styles.tierCardInner}>
                      <View>
                        <Text style={[styles.tierName, sel && { color: COLORS.primary }]}>{t.name}</Text>
                        <Text style={[styles.tierPrice, sel && { color: COLORS.primary }]}>{t.price}</Text>
                        <Text style={styles.tierPayout}>Max payout: {t.payout}</Text>
                      </View>
                      <View style={[styles.tierRadio, sel && styles.tierRadioSelected]}>
                        {sel && <View style={styles.tierRadioDot} />}
                      </View>
                    </View>
                  </TouchableOpacity>
                );
              })}

              {/* Summary */}
              <View style={styles.summaryCard}>
                <Text style={styles.summaryTitle}>Summary</Text>
                {[
                  ['Platforms', form.platforms.map(id => ALL_PLATFORMS.find(p => p.id === id)?.name).join(', ') || '–'],
                  ['Coverage', TIERS.find(t => t.id === form.tier)?.name || '–'],
                  ['AutoPay', form.autopay ? 'Enabled (5% discount)' : 'Disabled'],
                ].map(([k, v]) => (
                  <View key={k} style={styles.summaryRow}>
                    <Text style={styles.summaryKey}>{k}</Text>
                    <Text style={styles.summaryVal}>{v}</Text>
                  </View>
                ))}
              </View>

              <View style={styles.navRow}>
                <TouchableOpacity style={styles.ghostBtn} onPress={prevStep}><Text style={styles.ghostBtnText}>← Back</Text></TouchableOpacity>
                <TouchableOpacity style={styles.ctaBtnSmall} onPress={handleSubmit}><Text style={styles.ctaBtnText}>Create Account</Text></TouchableOpacity>
              </View>
            </View>
          )}

          <TouchableOpacity onPress={() => navigation.navigate('Login')} style={{ alignItems: 'center', paddingVertical: SIZES.padding }}>
            <Text style={styles.loginHint}>Already have an account? <Text style={{ color: COLORS.primary }}>Sign in</Text></Text>
          </TouchableOpacity>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.surface },
  scroll: { paddingBottom: SIZES.padding * 3 },

  header: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: SIZES.padding, paddingTop: SIZES.padding, paddingBottom: SIZES.base },
  backBtn: { width: 40, height: 40, borderRadius: 12, backgroundColor: COLORS.surfaceHigh, justifyContent: 'center', alignItems: 'center' },
  headerTitle: { fontSize: SIZES.h3, fontFamily: FONTS.bold, color: COLORS.white },

  // Stepper
  stepper: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingHorizontal: SIZES.padding, paddingVertical: SIZES.padding, gap: 4 },
  stepItem: { alignItems: 'center', gap: 4 },
  stepCircle: { width: 28, height: 28, borderRadius: 14, backgroundColor: COLORS.surfaceHighest, justifyContent: 'center', alignItems: 'center', borderWidth: 1, borderColor: COLORS.border },
  stepActive: { backgroundColor: COLORS.primary, borderColor: COLORS.primary },
  stepDone: { backgroundColor: COLORS.primary, borderColor: COLORS.primary },
  stepNum: { fontSize: SIZES.tiny, fontFamily: FONTS.bold, color: COLORS.textFaint },
  stepLabel: { fontSize: 9, fontFamily: FONTS.medium, color: COLORS.textFaint },
  stepLine: { width: 24, height: 2, backgroundColor: COLORS.border, marginBottom: 12 },

  stepContent: { paddingHorizontal: SIZES.padding },
  stepHeading: { fontSize: SIZES.h2, fontFamily: FONTS.bold, color: COLORS.white, marginBottom: 6, marginTop: 4 },
  stepSub: { fontSize: SIZES.small, fontFamily: FONTS.medium, color: COLORS.textMuted, marginBottom: SIZES.padding },

  // Platform grid
  platformGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10, marginBottom: SIZES.padding },
  platformCard: { width: '47%', padding: 14, borderRadius: SIZES.radius, backgroundColor: COLORS.surfaceContainer, borderWidth: 1, borderColor: COLORS.border, alignItems: 'center' },
  platformCardSelected: { backgroundColor: COLORS.primaryContainer, borderColor: COLORS.primary },
  platformCardName: { fontSize: SIZES.small, fontFamily: FONTS.bold, color: COLORS.white },
  platformSelected: { fontSize: SIZES.tiny, fontFamily: FONTS.medium, color: COLORS.primary, marginTop: 4 },

  selectionBadge: { flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: COLORS.successContainer, paddingHorizontal: 14, paddingVertical: 8, borderRadius: SIZES.radiusFull, alignSelf: 'flex-start', marginBottom: SIZES.padding },
  selectionBadgeText: { fontSize: SIZES.small, fontFamily: FONTS.semiBold, color: COLORS.success },

  // Form fields
  row: { flexDirection: 'row' },
  fieldWrap: { marginBottom: SIZES.padding * 0.85 },
  fieldLabel: { fontSize: SIZES.small, fontFamily: FONTS.semiBold, color: COLORS.textMuted, marginBottom: 6 },
  fieldInput: { height: 46, backgroundColor: COLORS.surfaceHighest, borderRadius: SIZES.radius, paddingHorizontal: 14, fontSize: SIZES.body, fontFamily: FONTS.regular, color: COLORS.white, borderWidth: 1, borderColor: COLORS.border },
  fieldHint: { fontSize: 11, fontFamily: FONTS.regular, color: COLORS.textFaint, marginTop: 4 },

  // City pills
  cityPill: { paddingHorizontal: 14, paddingVertical: 8, borderRadius: SIZES.radiusFull, backgroundColor: COLORS.surfaceHighest, marginRight: 8, borderWidth: 1, borderColor: COLORS.border },
  cityPillSelected: { backgroundColor: COLORS.primaryContainer, borderColor: COLORS.primary },
  cityPillText: { fontSize: SIZES.small, fontFamily: FONTS.semiBold, color: COLORS.textMuted },

  // Secure note
  secureNote: { flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: COLORS.primaryContainer, paddingHorizontal: 12, paddingVertical: 8, borderRadius: SIZES.radius, marginBottom: SIZES.padding },
  secureNoteText: { fontSize: SIZES.small, fontFamily: FONTS.medium, color: COLORS.primaryDim },

  // Consent
  consentCard: { backgroundColor: COLORS.surfaceContainer, borderRadius: SIZES.radius * 1.2, padding: SIZES.padding, marginTop: SIZES.base, marginBottom: SIZES.padding, borderWidth: 1, borderColor: COLORS.border },
  consentTitle: { fontSize: SIZES.small, fontFamily: FONTS.bold, color: COLORS.amber, letterSpacing: 0.5, marginBottom: SIZES.padding, textTransform: 'uppercase' },
  consentRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 12, marginBottom: SIZES.padding * 0.75 },
  consentLabel: { flex: 1, fontSize: SIZES.small, fontFamily: FONTS.regular, color: COLORS.textMuted, lineHeight: 19 },
  discountTag: { backgroundColor: COLORS.successContainer, paddingHorizontal: 8, paddingVertical: 3, borderRadius: SIZES.radiusFull },
  discountTagText: { fontSize: SIZES.tiny, fontFamily: FONTS.bold, color: COLORS.success },

  // Tiers
  tierCard: { borderRadius: SIZES.radius * 1.2, backgroundColor: COLORS.surfaceContainer, borderWidth: 1, borderColor: COLORS.border, padding: SIZES.padding, marginBottom: 12, position: 'relative', overflow: 'hidden' },
  tierCardSelected: { borderColor: COLORS.primary, backgroundColor: COLORS.primaryContainer },
  tierCardInner: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  tierName: { fontSize: SIZES.body, fontFamily: FONTS.bold, color: COLORS.white, marginBottom: 4 },
  tierPrice: { fontSize: SIZES.h2, fontFamily: FONTS.bold, color: COLORS.white },
  tierPayout: { fontSize: SIZES.small, fontFamily: FONTS.medium, color: COLORS.textFaint, marginTop: 4 },
  tierRadio: { width: 22, height: 22, borderRadius: 11, borderWidth: 2, borderColor: COLORS.border, justifyContent: 'center', alignItems: 'center' },
  tierRadioSelected: { borderColor: COLORS.primary },
  tierRadioDot: { width: 10, height: 10, borderRadius: 5, backgroundColor: COLORS.primary },
  recommendedBadge: { position: 'absolute', top: -1, right: 16, backgroundColor: COLORS.amber, paddingHorizontal: 10, paddingVertical: 3, borderBottomLeftRadius: 8, borderBottomRightRadius: 8 },
  recommendedText: { fontSize: SIZES.tiny, fontFamily: FONTS.bold, color: '#fff' },

  // Summary
  summaryCard: { backgroundColor: COLORS.surfaceContainer, borderRadius: SIZES.radius * 1.2, padding: SIZES.padding, marginBottom: SIZES.padding, borderWidth: 1, borderColor: COLORS.border },
  summaryTitle: { fontSize: SIZES.small, fontFamily: FONTS.bold, color: COLORS.amber, textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: SIZES.padding * 0.75 },
  summaryRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 8, borderBottomWidth: 1, borderBottomColor: COLORS.border },
  summaryKey: { fontSize: SIZES.small, fontFamily: FONTS.medium, color: COLORS.textMuted },
  summaryVal: { fontSize: SIZES.small, fontFamily: FONTS.bold, color: COLORS.white, flex: 1, textAlign: 'right' },

  // Buttons
  ctaBtn: { height: 52, borderRadius: SIZES.radiusFull, backgroundColor: COLORS.primary, justifyContent: 'center', alignItems: 'center', marginTop: SIZES.padding },
  ctaBtnSmall: { flex: 2, height: 48, borderRadius: SIZES.radiusFull, backgroundColor: COLORS.primary, justifyContent: 'center', alignItems: 'center' },
  ctaBtnText: { fontSize: SIZES.body, fontFamily: FONTS.bold, color: '#fff' },
  ghostBtn: { flex: 1, height: 48, borderRadius: SIZES.radiusFull, backgroundColor: 'transparent', justifyContent: 'center', alignItems: 'center', borderWidth: 1, borderColor: COLORS.border, marginRight: 12 },
  ghostBtnText: { fontSize: SIZES.body, fontFamily: FONTS.semiBold, color: COLORS.textMuted },
  navRow: { flexDirection: 'row', marginTop: SIZES.padding, marginBottom: SIZES.base },

  loginHint: { fontSize: SIZES.small, fontFamily: FONTS.medium, color: COLORS.textFaint },
});

export default SignUpScreen;
