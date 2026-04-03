import React, { useState } from 'react';
import { View, Text, StyleSheet, SafeAreaView, KeyboardAvoidingView, Platform, TouchableOpacity } from 'react-native';
import { COLORS, FONTS, SIZES } from '../constants/theme';
import InputField from '../components/InputField';
import Button from '../components/Button';
import { useAuth } from '../context/AuthContext';

const LoginScreen = ({ navigation }) => {
  const { login } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleLogin = async () => {
    if (!email || !password) {
      setError('Please enter both email and password');
      return;
    }
    setLoading(true);
    setError('');
    try {
      await login(email.trim().toLowerCase(), password);
      navigation.reset({ index: 0, routes: [{ name: 'Home' }] });
    } catch (e) {
      setError(e.message || 'Login failed. Check your credentials.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.content}
      >
        <View style={styles.card}>
          <View style={styles.header}>
            <Text style={styles.title}>Welcome Back</Text>
            <Text style={styles.subtitle}>Sign in to your GigGuard account</Text>
          </View>

          <View style={styles.form}>
            {error ? <Text style={styles.errorText}>{error}</Text> : null}

            <InputField
              label="Email Address"
              placeholder="example@gig.com"
              value={email}
              onChangeText={setEmail}
              keyboardType="email-address"
              autoCapitalize="none"
            />
            <InputField
              label="Password"
              placeholder="••••••••"
              value={password}
              onChangeText={setPassword}
              secureTextEntry
            />

            <TouchableOpacity style={styles.forgotPassword}>
              <Text style={styles.forgotPasswordText}>Forgot Password?</Text>
            </TouchableOpacity>

            <Button
              title="Login"
              onPress={handleLogin}
              loading={loading}
              style={styles.loginButton}
            />

            <View style={styles.footer}>
              <Text style={styles.footerText}>Don't have an account? </Text>
              <TouchableOpacity onPress={() => navigation.navigate('SignUp')}>
                <Text style={styles.footerLink}>Sign Up</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.background },
  content: { flex: 1, paddingHorizontal: SIZES.padding, justifyContent: 'center', backgroundColor: COLORS.background },
  card: {
    backgroundColor: COLORS.white,
    padding: SIZES.padding,
    borderRadius: SIZES.radius * 2,
    shadowColor: COLORS.secondary,
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.05,
    shadowRadius: 20,
    elevation: 3,
    borderWidth: 1,
    borderColor: 'rgba(4, 5, 46, 0.03)',
  },
  header: { marginBottom: SIZES.padding, alignItems: 'center' },
  title: { fontSize: 28, fontFamily: FONTS.bold, color: COLORS.secondary, marginBottom: SIZES.base * 0.5 },
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
  forgotPassword: { alignSelf: 'flex-end', marginBottom: SIZES.padding },
  forgotPasswordText: { fontSize: 13, fontFamily: FONTS.bold, color: COLORS.primary },
  loginButton: { marginTop: SIZES.base },
  footer: { flexDirection: 'row', justifyContent: 'center', marginTop: SIZES.padding * 2 },
  footerText: { fontSize: 14, fontFamily: FONTS.medium, color: COLORS.secondary, opacity: 0.7 },
  footerLink: { fontSize: 14, fontFamily: FONTS.bold, color: COLORS.primary },
});

export default LoginScreen;
