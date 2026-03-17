import React, { useState } from 'react';
import { View, Text, StyleSheet, SafeAreaView, KeyboardAvoidingView, Platform, TouchableOpacity, ScrollView } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, FONTS, SIZES } from '../constants/theme';
import InputField from '../components/InputField';
import Button from '../components/Button';
import Checkbox from '../components/Checkbox';

const PLATFORMS = ['Swiggy', 'Zomato', 'Amazon', 'Meesho', 'Savana'];

const SignUpScreen = ({ navigation }) => {
  const [formData, setFormData] = useState({
    platform: '',
    name: '',
    age: '',
    email: '',
    phone: '',
    pancard: '',
    aadhaar: '',
    location: '',
    address: '',
    deliveryId: '',
    consent: false,
  });

  const updateField = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSignUp = () => {
    if (!formData.consent) {
      alert('Please provide consent to proceed.');
      return;
    }
    console.log('Registering Delivery Partner:', formData);
    // Navigation or registration logic
  };

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.content}
      >
        <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
          <View style={styles.card}>
            <View style={styles.header}>
              <Text style={styles.title}>Partner Registration</Text>
              <Text style={styles.subtitle}>Join GigGuard's professional network.</Text>
            </View>

            <View style={styles.form}>
              {/* Platform Selection */}
              <Text style={styles.label}>Choose your platform</Text>
              <View style={styles.platformContainer}>
                {PLATFORMS.map((p) => (
                  <TouchableOpacity
                    key={p}
                    style={[
                      styles.platformItem,
                      formData.platform === p && styles.selectedPlatform
                    ]}
                    onPress={() => updateField('platform', p)}
                  >
                    <Text style={[
                      styles.platformText,
                      formData.platform === p && styles.selectedPlatformText
                    ]}>{p}</Text>
                  </TouchableOpacity>
                ))}
              </View>

              <InputField
                label="Full Name"
                placeholder="Enter your name"
                value={formData.name}
                onChangeText={(v) => updateField('name', v)}
              />
              
              <View style={styles.row}>
                <View style={{ flex: 1, marginRight: SIZES.base }}>
                  <InputField
                    label="Age"
                    placeholder="25"
                    value={formData.age}
                    onChangeText={(v) => updateField('age', v)}
                    keyboardType="numeric"
                  />
                </View>
                <View style={{ flex: 2 }}>
                  <InputField
                    label="Phone Number"
                    placeholder="+91 XXXXX XXXXX"
                    value={formData.phone}
                    onChangeText={(v) => updateField('phone', v)}
                    keyboardType="phone-pad"
                  />
                </View>
              </View>

              <InputField
                label="Email Address"
                placeholder="example@partner.com"
                value={formData.email}
                onChangeText={(v) => updateField('email', v)}
                keyboardType="email-address"
                autoCapitalize="none"
              />

              <View style={styles.row}>
                <View style={{ flex: 1, marginRight: SIZES.base }}>
                  <InputField
                    label="PAN Card"
                    placeholder="ABCDE1234F"
                    value={formData.pancard}
                    onChangeText={(v) => updateField('pancard', v)}
                    autoCapitalize="characters"
                  />
                </View>
                <View style={{ flex: 1 }}>
                  <InputField
                    label="Aadhaar"
                    placeholder="XXXX XXXX XXXX"
                    value={formData.aadhaar}
                    onChangeText={(v) => updateField('aadhaar', v)}
                    keyboardType="numeric"
                  />
                </View>
              </View>

              <InputField
                label="Location of Service"
                placeholder="City, Area"
                value={formData.location}
                onChangeText={(v) => updateField('location', v)}
              />

              <InputField
                label="Delivery Partner ID"
                placeholder="e.g. SW-12345"
                value={formData.deliveryId}
                onChangeText={(v) => updateField('deliveryId', v)}
              />

              <Checkbox
                label="I consent to weather monitoring for insurance payouts."
                checked={formData.consent}
                onChange={(v) => updateField('consent', v)}
                style={styles.checkbox}
              />

              <Button
                title="Create Account"
                onPress={handleSignUp}
                style={styles.signUpButton}
              />

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
  container: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  content: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: SIZES.padding,
    paddingVertical: SIZES.padding * 1.5,
  },
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
  header: {
    marginBottom: SIZES.padding,
    alignItems: 'center',
  },
  title: {
    fontSize: 24,
    fontFamily: FONTS.bold,
    color: COLORS.secondary,
    marginBottom: SIZES.base * 0.5,
  },
  subtitle: {
    fontSize: 14,
    fontFamily: FONTS.medium,
    color: COLORS.secondary,
    opacity: 0.6,
  },
  form: {
    width: '100%',
  },
  label: {
    fontSize: 14,
    fontFamily: FONTS.bold,
    color: COLORS.secondary,
    marginBottom: SIZES.base,
    marginTop: SIZES.base,
  },
  platformContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: SIZES.padding,
  },
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
  selectedPlatform: {
    backgroundColor: COLORS.secondary,
    borderColor: COLORS.secondary,
  },
  platformText: {
    fontFamily: FONTS.bold,
    fontSize: 13,
    color: COLORS.secondary,
  },
  selectedPlatformText: {
    color: COLORS.white,
  },
  row: {
    flexDirection: 'row',
  },
  checkbox: {
    marginTop: SIZES.padding,
    marginBottom: SIZES.padding,
  },
  signUpButton: {
    marginTop: SIZES.base,
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: SIZES.padding,
    marginBottom: SIZES.padding,
  },
  footerText: {
    fontSize: 14,
    fontFamily: FONTS.medium,
    color: COLORS.secondary,
    opacity: 0.6,
  },
  footerLink: {
    fontSize: 14,
    fontFamily: FONTS.bold,
    color: COLORS.primary,
  },
});

export default SignUpScreen;
