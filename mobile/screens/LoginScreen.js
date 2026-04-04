import React, { useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  KeyboardAvoidingView,
  TouchableOpacity,
  TextInput,
  Platform,
  ActivityIndicator,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { COLORS, FONTS, SIZES, SHADOWS } from "../constants/theme";
import { useAuth } from "../context/AuthContext";

const LoginScreen = ({ navigation }) => {
  const { login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPwd, setShowPwd] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleLogin = async () => {
    if (!email.trim() || !password) {
      setError("Please enter your email and password.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      await login(email.trim(), password);
      navigation.reset({ index: 0, routes: [{ name: "Home" }] });
    } catch (err) {
      setError(err.message || "Login failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        style={{ flex: 1 }}
      >
        {/* Top glow */}
        <View style={styles.bgGlow} />

        <View style={styles.content}>
          {/* Logo */}
          <View style={styles.logoWrap}>
            <View style={styles.logoIcon}>
              <Ionicons
                name="shield-checkmark"
                size={24}
                color={COLORS.white}
              />
            </View>
            <Text style={styles.logoText}>WPIP</Text>
          </View>

          <Text style={styles.title}>Welcome back</Text>
          <Text style={styles.subtitle}>
            Sign in to your income protection account
          </Text>

          {/* Card */}
          <View style={styles.card}>
            {!!error && (
              <View style={styles.errorBox}>
                <Text style={styles.errorText}>{error}</Text>
              </View>
            )}

            <Text style={styles.fieldLabel}>Email Address</Text>
            <TextInput
              style={styles.fieldInput}
              placeholder="you@example.com"
              placeholderTextColor={COLORS.textFaint}
              value={email}
              onChangeText={setEmail}
              keyboardType="email-address"
              autoCapitalize="none"
            />

            <Text style={styles.fieldLabel}>Password</Text>
            <View style={styles.passwordWrap}>
              <TextInput
                style={[styles.fieldInput, { flex: 1, borderWidth: 0 }]}
                placeholder="••••••••"
                placeholderTextColor={COLORS.textFaint}
                value={password}
                onChangeText={setPassword}
                secureTextEntry={!showPwd}
                autoCapitalize="none"
              />
              <TouchableOpacity
                onPress={() => setShowPwd((v) => !v)}
                style={styles.eyeBtn}
              >
                <Ionicons
                  name={showPwd ? "eye-off-outline" : "eye-outline"}
                  size={18}
                  color={COLORS.textFaint}
                />
              </TouchableOpacity>
            </View>

            <TouchableOpacity
              style={[styles.loginBtn, loading && { opacity: 0.7 }]}
              onPress={handleLogin}
              activeOpacity={0.85}
              disabled={loading}
            >
              {loading ? (
                <ActivityIndicator color={COLORS.white} />
              ) : (
                <>
                  <Text style={styles.loginBtnText}>Sign In</Text>
                  <Ionicons name="arrow-forward" size={18} color="#fff" />
                </>
              )}
            </TouchableOpacity>
          </View>

          <TouchableOpacity
            onPress={() => navigation.navigate("SignUp")}
            style={styles.signupLink}
          >
            <Text style={styles.signupLinkText}>
              Don't have an account?{" "}
              <Text style={{ color: COLORS.primary }}>Register free</Text>
            </Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: COLORS.surface },
  bgGlow: {
    position: "absolute",
    width: 300,
    height: 300,
    borderRadius: 150,
    backgroundColor: COLORS.primary,
    opacity: 0.06,
    top: -100,
    alignSelf: "center",
  },
  content: {
    flex: 1,
    paddingHorizontal: SIZES.padding,
    justifyContent: "center",
    paddingBottom: SIZES.padding * 2,
  },

  logoWrap: {
    flexDirection: "row",
    alignItems: "center",
    gap: SIZES.base,
    justifyContent: "center",
    marginBottom: SIZES.padding * 2,
  },
  logoIcon: {
    width: 42,
    height: 42,
    borderRadius: 13,
    backgroundColor: COLORS.primary,
    justifyContent: "center",
    alignItems: "center",
  },
  logoText: { fontSize: SIZES.h2, fontFamily: FONTS.bold, color: COLORS.white },

  title: {
    fontSize: SIZES.h1,
    fontFamily: FONTS.bold,
    color: COLORS.white,
    marginBottom: 6,
  },
  subtitle: {
    fontSize: SIZES.small,
    fontFamily: FONTS.regular,
    color: COLORS.textMuted,
    marginBottom: SIZES.padding * 1.5,
    lineHeight: 20,
  },

  card: {
    backgroundColor: COLORS.surfaceContainer,
    borderRadius: SIZES.radius * 1.5,
    padding: SIZES.padding,
    borderWidth: 1,
    borderColor: COLORS.border,
    ...SHADOWS.card,
  },

  errorBox: {
    backgroundColor: COLORS.errorContainer,
    borderRadius: SIZES.radius,
    paddingHorizontal: 12,
    paddingVertical: 10,
    marginBottom: SIZES.base,
    borderWidth: 1,
    borderColor: COLORS.error + "40",
  },
  errorText: {
    color: COLORS.error,
    fontSize: SIZES.small,
    fontFamily: FONTS.medium,
  },

  fieldLabel: {
    fontSize: SIZES.small,
    fontFamily: FONTS.semiBold,
    color: COLORS.textMuted,
    marginBottom: 6,
    marginTop: SIZES.padding * 0.5,
  },
  fieldInput: {
    height: 48,
    backgroundColor: COLORS.surfaceHighest,
    borderRadius: SIZES.radius,
    paddingHorizontal: 14,
    fontSize: SIZES.body,
    fontFamily: FONTS.regular,
    color: COLORS.white,
    borderWidth: 1,
    borderColor: COLORS.border,
  },

  passwordWrap: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: COLORS.surfaceHighest,
    borderRadius: SIZES.radius,
    borderWidth: 1,
    borderColor: COLORS.border,
    height: 48,
    paddingLeft: 14,
  },
  eyeBtn: { paddingHorizontal: 14 },

  loginBtn: {
    height: 52,
    borderRadius: SIZES.radiusFull,
    backgroundColor: COLORS.primary,
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    gap: 10,
    ...SHADOWS.button,
  },
  loginBtnText: { fontSize: SIZES.body, fontFamily: FONTS.bold, color: "#fff" },

  signupLink: { alignItems: "center", marginTop: SIZES.padding * 1.5 },
  signupLinkText: {
    fontSize: SIZES.small,
    fontFamily: FONTS.medium,
    color: COLORS.textFaint,
  },
});

export default LoginScreen;
