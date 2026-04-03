/**
 * Central API configuration for GigGuard mobile app.
 *
 * AUTH endpoints go to the Next.js web server (which owns JWT logic).
 * DATA endpoints go directly to the FastAPI backend.
 *
 * For physical devices, replace "localhost" with your computer's LAN IP.
 */
import { Platform } from 'react-native';

const devHost = Platform.OS === 'android' ? '10.0.2.2' : 'localhost';

// Next.js web server — handles auth (login / register / me)
const WEB_URL = `http://${devHost}:3000`;

// FastAPI backend — handles data (claims, premium, weather, triggers)
const BACKEND_URL = `http://${devHost}:8000`;

async function request(url, options = {}) {
  const res = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || data.message || 'Request failed');
  }
  return data;
}

// ── Auth (via Next.js) ─────────────────────────────────────────────
export function login(email, password) {
  return request(`${WEB_URL}/api/auth/login`, {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  });
}

export function register(formData) {
  return request(`${WEB_URL}/api/auth/register`, {
    method: 'POST',
    body: JSON.stringify(formData),
  });
}

export function getMe(token) {
  return request(`${WEB_URL}/api/auth/me`, {
    headers: { Authorization: `Bearer ${token}` },
  });
}

// ── Premium (via FastAPI) ──────────────────────────────────────────
export function predictPremium(delivery_id, city, tier) {
  return request(`${BACKEND_URL}/api/premium/predict`, {
    method: 'POST',
    body: JSON.stringify({ delivery_id, city: city || 'Unknown', tier: tier || 'standard' }),
  });
}

// ── Claims (via FastAPI) ───────────────────────────────────────────
export function getWorkerClaims(deliveryId) {
  return request(`${BACKEND_URL}/api/claims/worker/${deliveryId}`);
}

// ── Weather & Triggers (via FastAPI) ───────────────────────────────
export function getCityWeather(city) {
  return request(`${BACKEND_URL}/api/weather/${city}`);
}

export function getTriggerStatus() {
  return request(`${BACKEND_URL}/api/triggers/status`);
}

// ── GPS (via FastAPI) ──────────────────────────────────────────────
export function gpsCheckin(worker_id, latitude, longitude) {
  return request(`${BACKEND_URL}/api/gps/checkin`, {
    method: 'POST',
    body: JSON.stringify({ worker_id, latitude, longitude }),
  });
}
