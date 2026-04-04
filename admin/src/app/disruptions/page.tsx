"use client";
import { useEffect, useState } from "react";

/* ─── Types ─── */

type WeatherInfo = {
  temperature: number;
  humidity: number;
  wind_speed: number;
  rain_1h: number;
  rain_3h: number;
  aqi_index: number;
  weather_main: string;
  is_heavy_rain: boolean;
  is_extreme_heat: boolean;
  is_severe_aqi: boolean;
  is_flood_risk: boolean;
};

type TriggerFired = {
  trigger_id: string;
  trigger_type: string;
  description: string;
  severity: string;
};

type CityStatus = {
  weather: WeatherInfo;
  triggers_fired: TriggerFired[];
  has_active_disruption: boolean;
};

type DisruptionEvent = {
  id: string;
  trigger_id: string;
  trigger_type: string;
  city: string;
  severity: string;
  description: string;
  temperature: number;
  rain_1h: number;
  rain_3h: number;
  aqi_index: number;
  status: string;
  event_date: string;
  created_at: string;
};

type Claim = {
  id: string;
  claim_number: string;
  worker_id: string;
  platform: string;
  city: string;
  trigger_id: string;
  trigger_type: string;
  payout_amount: number;
  daily_wage_est: number;
  disrupted_hours: number;
  payout_status: string;
  fraud_score: number;
  fraud_flags: string[];
  cross_platform_clear: boolean;
  status: string;
  created_at: string;
};

/* ─── Constants ─── */

const SEVERITY_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  extreme: { bg: "#fef2f2", text: "#dc2626", border: "#fecaca" },
  severe: { bg: "#fff7ed", text: "#ea580c", border: "#fed7aa" },
  moderate: { bg: "#fffbeb", text: "#d97706", border: "#fde68a" },
};

const STATUS_COLORS: Record<string, { bg: string; text: string }> = {
  active: { bg: "#dcfce7", text: "#166534" },
  resolved: { bg: "#f1f5f9", text: "#475569" },
  expired: { bg: "#f1f5f9", text: "#94a3b8" },
  auto_initiated: { bg: "#dbeafe", text: "#1d4ed8" },
  under_review: { bg: "#fef3c7", text: "#92400e" },
  approved: { bg: "#dcfce7", text: "#166534" },
  paid: { bg: "#d1fae5", text: "#065f46" },
  rejected: { bg: "#fef2f2", text: "#dc2626" },
  pending: { bg: "#fef3c7", text: "#92400e" },
};

const WEATHER_EMOJI: Record<string, string> = {
  Clear: "☀️",
  Clouds: "☁️",
  Rain: "🌧️",
  Drizzle: "🌦️",
  Thunderstorm: "⛈️",
  Snow: "❄️",
  Mist: "🌫️",
  Haze: "🌫️",
  Fog: "🌫️",
};

/* ─── Page ─── */

export default function DisruptionsPage() {
  const [triggerStatus, setTriggerStatus] = useState<Record<string, CityStatus>>({});
  const [disruptions, setDisruptions] = useState<DisruptionEvent[]>([]);
  const [claims, setClaims] = useState<Claim[]>([]);
  const [loading, setLoading] = useState(true);
  const [actionMsg, setActionMsg] = useState("");
  const [actionErr, setActionErr] = useState("");
  const [testCity, setTestCity] = useState("Chennai");
  const [testTrigger, setTestTrigger] = useState("T-01");
  const [firing, setFiring] = useState(false);
  const [activeTab, setActiveTab] = useState<"overview" | "claims">("overview");

  useEffect(() => {
    refresh();
  }, []);

  const refresh = async () => {
    setLoading(true);
    try {
      const [tsRes, dRes, cRes] = await Promise.all([
        fetch("/api/backend/triggers/status"),
        fetch("/api/backend/disruptions?limit=50"),
        fetch("/api/backend/claims?limit=100"),
      ]);
      if (tsRes.ok) setTriggerStatus(await tsRes.json());
      if (dRes.ok) {
        const d = await dRes.json();
        setDisruptions(d.data || []);
      }
      if (cRes.ok) {
        const c = await cRes.json();
        setClaims(c.data || []);
      }
    } catch {
      setActionErr("Failed to connect to backend.");
    } finally {
      setLoading(false);
    }
  };

  const handleTestFire = async () => {
    setFiring(true);
    setActionMsg("");
    setActionErr("");
    try {
      const res = await fetch("/api/backend/triggers/test-fire", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ city: testCity, trigger_id: testTrigger }),
      });
      const data = await res.json();
      if (data.error) {
        setActionErr(data.error);
      } else {
        setActionMsg(`🚨 Trigger ${data.trigger_id} fired in ${data.city} — ${data.claims_created} claims auto-created`);
        refresh();
      }
    } catch {
      setActionErr("Backend unavailable.");
    } finally {
      setFiring(false);
    }
  };

  const cities = Object.keys(triggerStatus);
  const activeCities = cities.filter((c) => triggerStatus[c].has_active_disruption);
  const totalClaims = claims.length;
  const pendingClaims = claims.filter((c) => c.payout_status === "pending" || c.status === "under_review").length;
  const totalPayout = claims.reduce((sum, c) => sum + (c.payout_amount || 0), 0);

  return (
    <div style={{ minHeight: "100vh", background: "#f8fafc", display: "flex", flexDirection: "column" }}>
      {/* Header */}
      <header style={{ background: "#0f172a", padding: "0 24px", height: 60, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 36, height: 36, background: "#4f46e5", borderRadius: 10, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, fontWeight: 800, color: "#fff" }}>WP</div>
          <span style={{ fontSize: 17, fontWeight: 700, color: "#fff" }}>WPIP</span>
          <span style={{ background: "#dc2626", color: "#fff", fontSize: 11, fontWeight: 700, padding: "3px 10px", borderRadius: 6, marginLeft: 4 }}>Disruptions</span>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <a href="/admin" style={{ color: "#94a3b8", fontSize: 13, fontWeight: 600, textDecoration: "none", padding: "6px 14px", border: "1px solid #334155", borderRadius: 6 }}>Admin</a>
          <a href="/control-center" style={{ color: "#94a3b8", fontSize: 13, fontWeight: 600, textDecoration: "none", padding: "6px 14px", border: "1px solid #334155", borderRadius: 6 }}>Control Center</a>
          <button type="button" onClick={refresh} style={{ background: "#1e293b", color: "#94a3b8", border: "1px solid #334155", borderRadius: 6, padding: "6px 14px", fontSize: 12, fontWeight: 600, cursor: "pointer" }}>↻ Refresh</button>
        </div>
      </header>

      <main style={{ flex: 1, padding: 32 }}>
        <div style={{ maxWidth: 1300, margin: "0 auto" }}>
          <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4 }}>Parametric Trigger Monitor</h1>
          <p style={{ color: "#64748b", fontSize: 14, marginBottom: 24 }}>
            Real-time weather monitoring, disruption events, and automated claims dashboard.
          </p>

          {actionMsg && <div style={{ marginBottom: 12, background: "#f0fdf4", border: "1px solid #86efac", color: "#166534", fontSize: 13, fontWeight: 600, borderRadius: 8, padding: "10px 14px" }}>{actionMsg}</div>}
          {actionErr && <div style={{ marginBottom: 12, background: "#fef2f2", border: "1px solid #fecaca", color: "#dc2626", fontSize: 13, fontWeight: 600, borderRadius: 8, padding: "10px 14px" }}>{actionErr}</div>}

          {loading ? (
            <div style={{ padding: 60, textAlign: "center", color: "#64748b" }}>Loading trigger data...</div>
          ) : (
            <>
              {/* Summary Cards */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, marginBottom: 28 }}>
                <SummaryCard label="Monitored Cities" value={cities.length.toString()} icon="🏙️" color="#4f46e5" />
                <SummaryCard label="Active Disruptions" value={activeCities.length.toString()} icon="🚨" color={activeCities.length > 0 ? "#dc2626" : "#059669"} />
                <SummaryCard label="Total Claims" value={totalClaims.toString()} icon="📋" color="#d97706" />
                <SummaryCard label="Total Payout" value={`₹${totalPayout.toLocaleString("en-IN")}`} icon="💰" color="#059669" />
              </div>

              {/* Tabs */}
              <div style={{ display: "flex", gap: 0, marginBottom: 24 }}>
                {(["overview", "claims"] as const).map((tab) => (
                  <button key={tab} type="button" onClick={() => setActiveTab(tab)} style={{
                    padding: "10px 24px", fontSize: 14, fontWeight: 600, cursor: "pointer", border: "1px solid #e2e8f0", borderBottom: activeTab === tab ? "2px solid #4f46e5" : "1px solid #e2e8f0",
                    background: activeTab === tab ? "#fff" : "#f8fafc", color: activeTab === tab ? "#4f46e5" : "#64748b",
                    borderRadius: tab === "overview" ? "8px 0 0 0" : "0 8px 0 0",
                  }}>
                    {tab === "overview" ? "🌦️ Weather & Triggers" : `📋 Claims (${totalClaims})`}
                  </button>
                ))}
              </div>

              {activeTab === "overview" && (
                <>
                  {/* Live Weather Grid */}
                  <div style={{ background: "#fff", borderRadius: 12, border: "1px solid #e2e8f0", padding: 24, marginBottom: 24 }}>
                    <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16, color: "#0f172a" }}>Live Weather & Trigger Status</h3>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
                      {cities.map((city) => {
                        const cs = triggerStatus[city];
                        const w = cs.weather;
                        const emoji = WEATHER_EMOJI[w.weather_main] || "🌤️";
                        return (
                          <div key={city} style={{
                            borderRadius: 12, border: cs.has_active_disruption ? "2px solid #dc2626" : "1px solid #e2e8f0",
                            padding: 16, background: cs.has_active_disruption ? "#fef2f2" : "#f8fafc",
                            position: "relative", overflow: "hidden",
                          }}>
                            {cs.has_active_disruption && (
                              <div style={{ position: "absolute", top: 0, right: 0, background: "#dc2626", color: "#fff", fontSize: 10, fontWeight: 700, padding: "3px 10px", borderRadius: "0 10px 0 10px" }}>
                                ⚠ ALERT
                              </div>
                            )}
                            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                              <span style={{ fontSize: 28 }}>{emoji}</span>
                              <div>
                                <div style={{ fontSize: 15, fontWeight: 700, color: "#0f172a" }}>{city}</div>
                                <div style={{ fontSize: 12, color: "#64748b" }}>{w.weather_main} • {w.temperature}°C</div>
                              </div>
                            </div>
                            <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 6, fontSize: 12 }}>
                              <MiniStat label="Rain 1h" value={`${w.rain_1h}mm`} warn={w.rain_1h > 20} />
                              <MiniStat label="Rain 3h" value={`${w.rain_3h}mm`} warn={w.rain_3h > 64.5} />
                              <MiniStat label="AQI" value={w.aqi_index.toString()} warn={w.aqi_index >= 300} />
                              <MiniStat label="Wind" value={`${w.wind_speed} m/s`} warn={w.wind_speed > 20} />
                            </div>
                            {cs.triggers_fired.length > 0 && (
                              <div style={{ marginTop: 10 }}>
                                {cs.triggers_fired.map((t) => (
                                  <div key={t.trigger_id} style={{
                                    background: SEVERITY_COLORS[t.severity]?.bg || "#fef3c7",
                                    color: SEVERITY_COLORS[t.severity]?.text || "#92400e",
                                    border: `1px solid ${SEVERITY_COLORS[t.severity]?.border || "#fde68a"}`,
                                    fontSize: 11, fontWeight: 600, padding: "4px 8px", borderRadius: 6, marginTop: 4,
                                  }}>
                                    {t.trigger_id}: {t.description} [{t.severity.toUpperCase()}]
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Test Fire */}
                  <div style={{ background: "#fff", borderRadius: 12, border: "1px solid #e2e8f0", padding: 24, marginBottom: 24 }}>
                    <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 12, color: "#0f172a" }}>🧪 Test Fire Trigger</h3>
                    <p style={{ fontSize: 13, color: "#64748b", marginBottom: 16 }}>Simulate a parametric disruption for testing. This creates a mock disruption event and auto-generates claims for all workers in that city.</p>
                    <div style={{ display: "flex", gap: 12, alignItems: "flex-end" }}>
                      <div>
                        <label style={labelStyle}>City</label>
                        <select value={testCity} onChange={(e) => setTestCity(e.target.value)} style={selectStyle}>
                          {["Chennai", "Bangalore", "Hyderabad", "Mumbai", "Delhi", "Pune"].map((c) => <option key={c} value={c}>{c}</option>)}
                        </select>
                      </div>
                      <div>
                        <label style={labelStyle}>Trigger</label>
                        <select value={testTrigger} onChange={(e) => setTestTrigger(e.target.value)} style={selectStyle}>
                          <option value="T-01">T-01: Heavy Rain</option>
                          <option value="T-02">T-02: Extreme Heat</option>
                          <option value="T-03">T-03: Severe AQI</option>
                          <option value="T-04">T-04: Flood</option>
                        </select>
                      </div>
                      <button type="button" onClick={handleTestFire} disabled={firing} style={{ height: 38, background: "#dc2626", color: "#fff", border: "none", borderRadius: 8, padding: "0 24px", fontSize: 13, fontWeight: 700, cursor: firing ? "not-allowed" : "pointer", opacity: firing ? 0.6 : 1 }}>
                        {firing ? "Firing..." : "🔥 Fire Trigger"}
                      </button>
                    </div>
                  </div>

                  {/* Disruption Events Table */}
                  <div style={{ background: "#fff", borderRadius: 12, border: "1px solid #e2e8f0", padding: 24 }}>
                    <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16, color: "#0f172a" }}>Disruption Events History</h3>
                    {disruptions.length === 0 ? (
                      <p style={{ color: "#94a3b8", fontSize: 13 }}>No disruption events yet. Use the test fire above or wait for real weather triggers.</p>
                    ) : (
                      <div style={{ overflowX: "auto" }}>
                        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                          <thead>
                            <tr style={{ background: "#f8fafc" }}>
                              {["Trigger", "Type", "City", "Severity", "Status", "Temp", "Rain 1h", "Rain 3h", "AQI", "Date"].map((h) => (
                                <th key={h} style={thStyle}>{h}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {disruptions.map((d) => (
                              <tr key={d.id} style={{ borderBottom: "1px solid #f1f5f9" }}>
                                <td style={tdStyle}><span style={{ fontWeight: 700 }}>{d.trigger_id}</span></td>
                                <td style={tdStyle}>{d.trigger_type}</td>
                                <td style={tdStyle}><span style={{ fontWeight: 600 }}>{d.city}</span></td>
                                <td style={tdStyle}><Badge color={SEVERITY_COLORS[d.severity]} text={d.severity.toUpperCase()} /></td>
                                <td style={tdStyle}><Badge color={STATUS_COLORS[d.status]} text={d.status} /></td>
                                <td style={tdStyle}>{d.temperature ?? "-"}°C</td>
                                <td style={tdStyle}>{d.rain_1h ?? "-"}mm</td>
                                <td style={tdStyle}>{d.rain_3h ?? "-"}mm</td>
                                <td style={tdStyle}>{d.aqi_index ?? "-"}</td>
                                <td style={tdStyle}>{d.event_date}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                </>
              )}

              {activeTab === "claims" && (
                <div style={{ background: "#fff", borderRadius: 12, border: "1px solid #e2e8f0", padding: 24 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                    <h3 style={{ fontSize: 16, fontWeight: 700, color: "#0f172a" }}>Auto-Generated Claims</h3>
                    <div style={{ display: "flex", gap: 12 }}>
                      <StatPill label="Pending" value={pendingClaims.toString()} color="#d97706" />
                      <StatPill label="Total Payout" value={`₹${totalPayout.toLocaleString("en-IN")}`} color="#059669" />
                    </div>
                  </div>
                  {claims.length === 0 ? (
                    <p style={{ color: "#94a3b8", fontSize: 13 }}>No claims yet. Fire a test trigger to see auto-generated claims.</p>
                  ) : (
                    <div style={{ overflowX: "auto" }}>
                      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                        <thead>
                          <tr style={{ background: "#f8fafc" }}>
                            {["Claim #", "Worker ID", "Platform", "City", "Trigger", "Payout", "Daily Wage", "Fraud Score", "Status", "Payout Status"].map((h) => (
                              <th key={h} style={thStyle}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {claims.map((c) => (
                            <tr key={c.id} style={{ borderBottom: "1px solid #f1f5f9" }}>
                              <td style={tdStyle}><span style={{ fontFamily: "monospace", fontSize: 11, fontWeight: 600 }}>{c.claim_number}</span></td>
                              <td style={tdStyle}><span style={{ fontFamily: "monospace", fontSize: 11 }}>{c.worker_id.slice(0, 8)}...</span></td>
                              <td style={tdStyle}>{c.platform}</td>
                              <td style={tdStyle}>{c.city}</td>
                              <td style={tdStyle}><span style={{ fontWeight: 600 }}>{c.trigger_id}</span> <span style={{ color: "#94a3b8" }}>{c.trigger_type}</span></td>
                              <td style={tdStyle}><span style={{ fontWeight: 700, color: "#059669" }}>₹{c.payout_amount?.toFixed(2)}</span></td>
                              <td style={tdStyle}>₹{c.daily_wage_est?.toFixed(2) ?? "-"}</td>
                              <td style={tdStyle}>
                                <span style={{
                                  fontWeight: 700,
                                  color: c.fraud_score >= 0.75 ? "#dc2626" : c.fraud_score >= 0.5 ? "#d97706" : "#059669",
                                }}>
                                  {(c.fraud_score * 100).toFixed(0)}%
                                </span>
                                {c.fraud_flags?.length > 0 && (
                                  <span style={{ fontSize: 10, color: "#dc2626", marginLeft: 4 }}>⚠</span>
                                )}
                              </td>
                              <td style={tdStyle}><Badge color={STATUS_COLORS[c.status]} text={c.status} /></td>
                              <td style={tdStyle}><Badge color={STATUS_COLORS[c.payout_status]} text={c.payout_status} /></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </main>
    </div>
  );
}

/* ─── Reusable Components ─── */

const labelStyle: React.CSSProperties = { display: "block", fontSize: 12, fontWeight: 600, color: "#374151", marginBottom: 4 };
const selectStyle: React.CSSProperties = { width: 180, height: 38, padding: "0 10px", fontSize: 13, border: "1px solid #d1d5db", borderRadius: 8, outline: "none", cursor: "pointer" };
const thStyle: React.CSSProperties = { padding: "8px 12px", textAlign: "left", fontWeight: 600, color: "#64748b", borderBottom: "1px solid #e2e8f0", fontSize: 11, textTransform: "uppercase" };
const tdStyle: React.CSSProperties = { padding: "8px 12px" };

function SummaryCard({ label, value, icon, color }: { label: string; value: string; icon: string; color: string }) {
  return (
    <div style={{ background: "#fff", borderRadius: 12, border: "1px solid #e2e8f0", padding: "16px 20px", display: "flex", alignItems: "center", gap: 14 }}>
      <div style={{ width: 44, height: 44, borderRadius: 12, background: `${color}15`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 22 }}>{icon}</div>
      <div>
        <div style={{ fontSize: 11, color: "#64748b", fontWeight: 500, textTransform: "uppercase" }}>{label}</div>
        <div style={{ fontSize: 20, fontWeight: 800, color }}>{value}</div>
      </div>
    </div>
  );
}

function MiniStat({ label, value, warn }: { label: string; value: string; warn: boolean }) {
  return (
    <div style={{ background: warn ? "#fef2f2" : "#f1f5f9", padding: "4px 8px", borderRadius: 6, border: warn ? "1px solid #fecaca" : "none" }}>
      <span style={{ color: "#94a3b8", fontSize: 10 }}>{label} </span>
      <span style={{ fontWeight: 600, color: warn ? "#dc2626" : "#374151" }}>{value}</span>
    </div>
  );
}

function Badge({ color, text }: { color?: { bg: string; text: string }; text: string }) {
  const c = color || { bg: "#f1f5f9", text: "#475569" };
  return (
    <span style={{ background: c.bg, color: c.text, fontSize: 11, fontWeight: 600, padding: "3px 8px", borderRadius: 6, whiteSpace: "nowrap" }}>
      {text}
    </span>
  );
}

function StatPill({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 13 }}>
      <span style={{ color: "#64748b", fontWeight: 500 }}>{label}:</span>
      <span style={{ fontWeight: 700, color }}>{value}</span>
    </div>
  );
}
