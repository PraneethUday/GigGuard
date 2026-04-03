import React, { useState } from 'react';
import { View, Text, StyleSheet, SafeAreaView, KeyboardAvoidingView, Platform, TouchableOpacity, ScrollView } from 'react-native';
import { COLORS, FONTS, SIZES } from '../constants/theme';
import InputField from '../components/InputField';
import Button from '../components/Button';
import Checkbox from '../components/Checkbox';
import { useAuth } from '../context/AuthContext';

const PLATFORMS = [
  { id: 'swiggy', label: 'Swiggy' },
  { id: 'zomato', label: 'Zomato' },
  { id: 'amazon_flex', label: 'Amazon Flex' },
  { id: 'blinkit', label: 'Blinkit' },
  { id: 'zepto', label: 'Zepto' },
  { id: 'meesho', label: 'Meesho' },
  { id: 'porter', label: 'Porter' },
  { id: 'dunzo', label: 'Dunzo' },
];

const CITIES = ['Chennai', 'Bangalore', 'Hyderabad', 'Mumbai', 'Delhi', 'Pune', 'Kolkata', 'Jaipur', 'Ahmedabad', 'Lucknow'];

const TIERS = [
  { id: 'basic', label: 'Basic Shield', range: '₹20–40/wk', payout: '₹500 max' },
  { id: 'standard', label: 'Standard Guard', range: '₹40–80/wk', payout: '₹1,200 max' },
  { id: 'pro', label: 'Pro Protect', range: '₹80–130/wk', payout: '₹2,500 max' },
];

const SignUpScreen = ({ navigation }) => {
  const { register } = useAuth();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [platforms, setPlatforms] = useState([]);
  const [name, setName] = useState('');
  const [age, setAge] = useState('');
  const [phone, setPhone] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [city, setCity] = useState('');
  const [area, setArea] = useState('');
  const [deliveryId, setDeliveryId] = useState('');
  const [pan, setPan] = useState('');
  const [aadhaar, setAadhaar] = useState('');
  const [upi, setUpi] = useState('');
  const [tier, setTier] = useState('standard');
  const [consent, setConsent] = useState(false);
  const [gpsConsent, setGpsConsent] = useState(false);
  const [autopay, setAutopay] = useState(false);

  const togglePlatform = (id) => {
    setPlatforms(prev => prev.includes(id) ? prev.filter(p => p !== id) : [...prev, id]);
  };

  const handleSignUp = async () => {
    if (platforms.length === 0) { setError('Select at least one platform'); return; }
    if (!name || !email || !password || !phone || !deliveryId) {
      setError('Please fill in all required fields');
      return;
    }
    if (password.length < 6) { setError('Password must be at least 6 characters'); return; }
    if (!city) { setError('Please select a city'); return; }
    if (!consent) { setError('Please provide consent to proceed'); return; }

    setLoading(true);
    setError('');
    try {
      await register({
        name,
        age: age ? parseInt(age, 10) : null,
        phone,
        email: email.trim().toLowerCase(),
        password,
        confirmPassword: password,
        city,
        area,
        deliveryId,
        platforms,
        pan: pan.toUpperCase(),
        aadhaar,
        upi,
        bank: '',
        consent,
        gpsConsent,
        autopay,
        tier,
      });
      navigation.reset({ index: 0, routes: [{ name: 'Home' }] });
    } catch (e) {
      setError(e.message || 'Registration failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : 'height'} style={styles.content}>
        <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
          <View style={styles.card}>
            <View style={styles.header}>
              <Text style={styles.title}>Partner Registration</Text>
              <Text style={styles.subtitle}>Join GigGuard's protection network</Text>
            </View>

            <View style={styles.form}>
              {error ? <Text style={styles.errorText}>{error}</Text> : null}

              {/* Platform Selection */}
              <Text style={styles.label}>Select your platforms *</Text>
              <View style={styles.platformContainer}>
                {PLATFORMS.map((p) => (
                  <TouchableOpacity
                    key={p.id}
                    style={[styles.platformItem, platforms.includes(p.id) && styles.selectedPlatform]}
                    onPress={() => togglePlatform(p.id)}
                  >
                    <Text style={[styles.platformText, platforms.includes(p.id) && styles.selectedPlatformText]}>
                      {p.label}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>

              <InputField label="Full Name *" placeholder="Enter your name" value={name} onChangeText={setName} />

              <View style={styles.row}>
                <View style={{ flex: 1, marginRight: SIZES.base }}>
                  <InputField label="Age" placeholder="25" value={age} onChangeText={setAge} keyboardType="numeric" />
                </View>
                <View style={{ flex: 2 }}>
                  <InputField label="Phone *" placeholder="+91 XXXXX XXXXX" value={phone} onChangeText={setPhone} keyboardType="phone-pad" />
                </View>
              </View>

              <InputField label="Email *" placeholder="example@partner.com" value={email} onChangeText={setEmail} keyboardType="email-address" autoCapitalize="none" />
              <InputField label="Password *" placeholder="Min 6 characters" value={password} onChangeText={setPassword} secureTextEntry />

              {/* City Selection */}
              <Text style={styles.label}>City *</Text>
              <View style={styles.platformContainer}>
                {CITIES.map((c) => (
                  <TouchableOpacity
                    key={c}
                    style={[styles.platformItem, city === c && styles.selectedPlatform]}
                    onPress={() => setCity(c)}
                  >
                    <Text style={[styles.platformText, city === c && styles.selectedPlatformText]}>{c}</Text>
                  </TouchableOpacity>
                ))}
              </View>

              <InputField label="Service Area / Zone" placeholder="e.g. Anna Nagar, South Delhi" value={area} onChangeText={setArea} />
              <InputField label="Delivery Partner ID *" placeholder="e.g. w_sw_001" value={deliveryId} onChangeText={setDeliveryId} />

              <View style={styles.row}>
                <View style={{ flex: 1, marginRight: SIZES.base }}>
                  <InputField label="PAN Card" placeholder="ABCDE1234F" value={pan} onChangeText={setPan} autoCapitalize="characters" />
                </View>
                <View style={{ flex: 1 }}>
                  <InputField label="Aadhaar" placeholder="XXXX XXXX XXXX" value={aadhaar} onChangeText={setAadhaar} keyboardType="numeric" />
                </View>
              </View>

              <InputField label="UPI ID (for payouts)" placeholder="name@upi" value={upi} onChangeText={setUpi} />

              {/* Tier Selection */}
              <Text style={styles.label}>Coverage Tier</Text>
              {TIERS.map((t) => (
                <TouchableOpacity
                  key={t.id}
                  style={[styles.tierCard, tier === t.id && styles.tierCardSelected]}
                  onPress={() => setTier(t.id)}
                >
                  <View style={{ flex: 1 }}>
                    <Text style={[styles.tierLabel, tier === t.id && styles.tierLabelSelected]}>{t.label}</Text>
                    <Text style={[styles.tierRange, tier === t.id && { color: 'rgba(255,255,255,0.7)' }]}>{t.range} · {t.payout}</Text>
                  </View>
                  <View style={[styles.tierRadio, tier === t.id && styles.tierRadioSelected]}>
                    {tier === t.id && <View style={styles.tierRadioDot} />}
                  </View>
                </TouchableOpacity>
              ))}

              {/* Consents */}
              <Checkbox label="I consent to weather monitoring for insurance payouts *" checked={consent} onChange={setConsent} style={styles.checkbox} />
              <Checkbox label="I consent to GPS location validation during disruptions" checked={gpsConsent} onChange={setGpsConsent} style={{ marginBottom: SIZES.base }} />
              <Checkbox label="Enable AutoPay (5% discount on premium)" checked={autopay} onChange={setAutopay} style={{ marginBottom: SIZES.padding }} />

              <Button title="Create Account" onPress={handleSignUp} loading={loading} style={styles.signUpButton} />

              <View style={styles.footer}>
                <Text style={styles.footerText}>Existing partner? </Text>
                <TouchableOpacity onPress={() => navigation.navigate('Login')}>
                  <Text style={styles.footerLink}>Login here</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.background },
  content: { flex: 1 },
  scrollContent: { paddingHorizontal: SIZES.padding, paddingVertical: SIZES.padding * 1.5 },
  card: {
    backgroundColor: COLORS.white,
    padding: SIZES.padding,
    borderRadius: SIZES.radius * 1.5,
    shadowColor: COLORS.secondary,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.04,
    shadowRadius: 20,
    elevation: 2,
    borderWidth: 1,
    borderColor: 'rgba(4, 5, 46, 0.03)',
  },
  header: { marginBottom: SIZES.padding, alignItems: 'center' },
  title: { fontSize: 24, fontFamily: FONTS.bold, color: COLORS.secondary, marginBottom: SIZES.base * 0.5 },
  subtitle: { fontSize: 14, fontFamily: FONTS.medium, color: COLORS.secondary, opacity: 0.6 },
  form: { width: '100%' },
  errorText: {
    fontSize: 13,
    fontFamily: FONTS.medium,
    color: COLORS.error,
    textAlign: 'center',
    marginBottom: SIZES.base,
    backgroundColor: 'rgba(211,47,47,0.08)',
    padding: 10,
    borderRadius: SIZES.radius,
  },
  label: { fontSize: 14, fontFamily: FONTS.bold, color: COLORS.secondary, marginBottom: SIZES.base, marginTop: SIZES.base },
  platformContainer: { flexDirection: 'row', flexWrap: 'wrap', marginBottom: SIZES.padding },
  platformItem: {
    paddingHorizontal: SIZES.base * 1.5,
    paddingVertical: SIZES.base,
    borderRadius: SIZES.radius * 0.5,
    backgroundColor: COLORS.background,
    marginRight: SIZES.base,
    marginBottom: SIZES.base,
    borderWidth: 1,
    borderColor: 'rgba(4, 5, 46, 0.1)',
  },
  selectedPlatform: { backgroundColor: COLORS.secondary, borderColor: COLORS.secondary },
  platformText: { fontFamily: FONTS.bold, fontSize: 13, color: COLORS.secondary },
  selectedPlatformText: { color: COLORS.white },
  row: { flexDirection: 'row' },
  tierCard: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: SIZES.padding * 0.75,
    borderRadius: SIZES.radius,
    borderWidth: 1.5,
    borderColor: 'rgba(4,5,46,0.1)',
    marginBottom: SIZES.base,
    backgroundColor: COLORS.surface,
  },
  tierCardSelected: { backgroundColor: COLORS.secondary, borderColor: COLORS.secondary },
  tierLabel: { fontSize: 15, fontFamily: FONTS.bold, color: COLORS.accent },
  tierLabelSelected: { color: COLORS.white },
  tierRange: { fontSize: 12, fontFamily: FONTS.medium, color: 'rgba(27,27,58,0.5)', marginTop: 2 },
  tierRadio: {
    width: 22, height: 22, borderRadius: 11,
    borderWidth: 2, borderColor: 'rgba(27,27,58,0.2)',
    justifyContent: 'center', alignItems: 'center',
  },
  tierRadioSelected: { borderColor: COLORS.white },
  tierRadioDot: { width: 10, height: 10, borderRadius: 5, backgroundColor: COLORS.white },
  checkbox: { marginTop: SIZES.padding, marginBottom: SIZES.base },
  signUpButton: { marginTop: SIZES.base },
  footer: { flexDirection: 'row', justifyContent: 'center', marginTop: SIZES.padding, marginBottom: SIZES.padding },
  footerText: { fontSize: 14, fontFamily: FONTS.medium, color: COLORS.secondary, opacity: 0.6 },
  footerLink: { fontSize: 14, fontFamily: FONTS.bold, color: COLORS.primary },
});

export default SignUpScreen;
