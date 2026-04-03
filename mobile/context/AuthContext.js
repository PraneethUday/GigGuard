import React, { createContext, useContext, useState, useEffect } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as api from '../lib/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [loading, setLoading] = useState(true);

  // Restore session on mount
  useEffect(() => {
    (async () => {
      try {
        const storedToken = await AsyncStorage.getItem('gg_token');
        const storedUser = await AsyncStorage.getItem('gg_user');
        if (storedToken && storedUser) {
          setToken(storedToken);
          setUser(JSON.parse(storedUser));
          // Verify token is still valid
          try {
            const data = await api.getMe(storedToken);
            if (data.user) {
              setUser(data.user);
              await AsyncStorage.setItem('gg_user', JSON.stringify(data.user));
            }
          } catch {
            // Token expired — clear session
            await clearSession();
          }
        }
      } catch {
        // ignore storage errors
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const saveSession = async (newToken, newUser) => {
    setToken(newToken);
    setUser(newUser);
    await AsyncStorage.setItem('gg_token', newToken);
    await AsyncStorage.setItem('gg_user', JSON.stringify(newUser));
  };

  const clearSession = async () => {
    setToken(null);
    setUser(null);
    await AsyncStorage.removeItem('gg_token');
    await AsyncStorage.removeItem('gg_user');
  };

  const login = async (email, password) => {
    const data = await api.login(email, password);
    await saveSession(data.token, data.user);
    return data;
  };

  const register = async (formData) => {
    const data = await api.register(formData);
    await saveSession(data.token, data.user);
    return data;
  };

  const logout = async () => {
    await clearSession();
  };

  const refreshUser = async () => {
    if (!token) return;
    try {
      const data = await api.getMe(token);
      if (data.user) {
        setUser(data.user);
        await AsyncStorage.setItem('gg_user', JSON.stringify(data.user));
      }
    } catch {
      // silently fail
    }
  };

  return (
    <AuthContext.Provider value={{ user, token, loading, login, register, logout, refreshUser }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
