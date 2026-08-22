import { Platform } from "react-native";
import { WEB_API_URL, BACKEND_API_URL } from "@env";

const fallbackHost = Platform.OS === "android" ? "10.0.2.2" : "localhost";

const WEB_URL = (WEB_API_URL || `http://${fallbackHost}:3000`).replace(
  /\/$/,
  "",
);
const BACKEND_URL = (BACKEND_API_URL || `http://${fallbackHost}:8000`).replace(
  /\/$/,
  "",
);

async function request(url, options = {}, timeoutMs = 8000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let response;
  try {
    response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
    });
  } catch (err) {
    const msg = err?.name === "AbortError"
      ? `Request timed out. Could not reach ${url}`
      : `Network request failed. Could not reach ${url}`;
    throw new Error(msg);
  } finally {
    clearTimeout(timer);
  }

  const raw = await response.text();
  let data = {};

  if (raw) {
    try {
      data = JSON.parse(raw);
    } catch {
      data = { message: raw };
    }
  }

  if (!response.ok) {
    throw new Error(
      data.error ||
        data.detail ||
        data.message ||
        `Request failed (${response.status})`,
    );
  }

  return data;
}

function shouldFallback(error) {
  const msg = (error?.message || "").toLowerCase();
  return (
    msg.includes("network request failed") ||
    msg.includes("failed to fetch") ||
    msg.includes("(404)") ||
    msg.includes("(502)") ||
    msg.includes("(503)")
  );
}

async function requestWithFallback(primaryUrl, fallbackUrl, options = {}) {
  try {
    return await request(primaryUrl, options);
  } catch (error) {
    if (!fallbackUrl || !shouldFallback(error)) {
      throw error;
    }
    return request(fallbackUrl, options);
  }
}

function authHeaders(token) {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

// Auth routes — always prefer the web (Next.js) URL so tokens are issued
// and verified by the same secret used by the payment and history APIs.
// Backend is kept as fallback only.
export function login(email, password) {
  return requestWithFallback(
    `${WEB_URL}/api/auth/login`,
    `${BACKEND_URL}/api/auth/login`,
    {
      method: "POST",
      body: JSON.stringify({ email, password }),
    },
  );
}

export function register(formData) {
  return requestWithFallback(
    `${WEB_URL}/api/auth/register`,
    `${BACKEND_URL}/api/auth/register`,
    {
      method: "POST",
      body: JSON.stringify(formData),
    },
  );
}

export function getMe(token) {
  return requestWithFallback(
    `${WEB_URL}/api/auth/me`,
    `${BACKEND_URL}/api/auth/me`,
    {
      headers: authHeaders(token),
    },
  );
}

export function verifyDeliveryId(deliveryId, platforms) {
  return requestWithFallback(
    `${BACKEND_URL}/api/verify-id`,
    `${WEB_URL}/api/verify-id`,
    {
      method: "POST",
      body: JSON.stringify({ deliveryId, platforms }),
    },
  );
}

// Premium routes
export async function predictPremium(delivery_id, city, tier = "standard") {
  const payload = { delivery_id, city: city || "Unknown", tier };

  try {
    return await request(`${WEB_URL}/api/premium/predict`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
  } catch {
    return request(`${BACKEND_URL}/api/premium/predict`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }
}

export async function getPremiumQuotes(deliveryId, city) {
  const tiers = ["basic", "standard", "pro"];
  const results = await Promise.allSettled(
    tiers.map((tier) => predictPremium(deliveryId, city, tier)),
  );

  return tiers.reduce((acc, tier, index) => {
    const result = results[index];
    if (result.status === "fulfilled") {
      acc[tier] = result.value;
    }
    return acc;
  }, {});
}

// Claims routes
export async function getWorkerClaims(token, deliveryId) {
  if (deliveryId) {
    try {
      return await request(
        `${WEB_URL}/api/backend/claims/worker/${encodeURIComponent(deliveryId)}`,
      );
    } catch {
      return request(
        `${BACKEND_URL}/api/claims/worker/${encodeURIComponent(deliveryId)}`,
      );
    }
  }

  if (token) {
    return request(`${WEB_URL}/api/claims/worker`, {
      headers: authHeaders(token),
    });
  }

  throw new Error("deliveryId or token is required to fetch claims");
}

// Weather / triggers routes
export async function getCityWeather(city) {
  try {
    return await request(
      `${WEB_URL}/api/backend/weather/${encodeURIComponent(city)}`,
    );
  } catch {
    return request(`${BACKEND_URL}/api/weather/${encodeURIComponent(city)}`);
  }
}

export async function getTriggerStatus() {
  try {
    return await request(`${WEB_URL}/api/backend/triggers/status`);
  } catch {
    return request(`${BACKEND_URL}/api/triggers/status`);
  }
}

// Payment routes
export function payPremium(token, payload = {}) {
  if (!token) {
    throw new Error("Authentication required. Please log in again.");
  }

  return request(`${WEB_URL}/api/payment/pay`, {
    method: "POST",
    headers: authHeaders(token),
    body: JSON.stringify(payload),
  });
}

export function getPaymentHistory(token) {
  if (!token) {
    throw new Error("Authentication required. Please log in again.");
  }

  return request(`${WEB_URL}/api/payment/history`, {
    headers: authHeaders(token),
  });
}

// Traffic & Curfew routes
export async function getCityTraffic(city) {
  try {
    return await request(
      `${WEB_URL}/api/backend/traffic/${encodeURIComponent(city)}`,
    );
  } catch {
    return request(
      `${BACKEND_URL}/api/traffic/${encodeURIComponent(city)}`,
    );
  }
}

export async function getCityCurfew(city) {
  try {
    return await request(
      `${WEB_URL}/api/backend/curfew/${encodeURIComponent(city)}`,
    );
  } catch {
    return request(
      `${BACKEND_URL}/api/curfew/${encodeURIComponent(city)}`,
    );
  }
}


// AI Chat
export function sendChatMessage(messages) {
  return request(`${WEB_URL}/api/chat`, {
    method: "POST",
    body: JSON.stringify({ messages }),
  });
}

// Support / Escalation
export function submitSupportTicket(token, payload) {
  return request(`${WEB_URL}/api/support/tickets`, {
    method: "POST",
    headers: authHeaders(token),
    body: JSON.stringify(payload),
  });
}
