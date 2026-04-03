"use client";
import { useEffect, useRef, useState } from "react";

type PlatformStat = {
  platform: string;
  worker_count: number;
  avg_earnings: number;
  min_earnings: number;
  max_earnings: number;
  avg_deliveries: number;
  total_earnings: number;
};

type DataSummary = {
  date: string;
  platforms: PlatformStat[];
  total_workers: number;
  overall_avg_earnings: number;
};

type ModelStatus = {
  model_loaded: boolean;
  metadata: {
    rmse: number;
    n_samples: number;
    trained_at: string;
    features: string[];
  } | null;
  tiers: Record<string, { rate: number; min: number; max: number; max_payout: number }>;
};

type PremiumResult = {
  weekly_premium: number;
  weekly_premium_autopay: number;
  raw_prediction: number;
  tier: string;
  max_payout: number;
  weather_risk: number;
  city_risk: number;
  weekly_earnings_est: number;
  history_days: number;
};

const PLATFORMS = [
  "swiggy", "zomato", "amazon_flex", "blinkit", "zepto", "meesho", "porter", "dunzo",
];
const CITIES = ["Chennai", "Bangalore", "Hyderabad", "Mumbai", "Delhi", "Pune"];

export default function ControlCenterPage() {
  const [dataSummary, setDataSummary] = useState<DataSummary | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionMsg, setActionMsg] = useState("");
  const [actionErr, setActionErr] = useState("");
  const [actionLoading, setActionLoading] = useState("");
  const [retrainLogs, setRetrainLogs] = useState<string[]>([]);
  const [retrainStatus, setRetrainStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Income adjustment form
  const [adjPlatform, setAdjPlatform] = useState("");
  const [adjEarnings, setAdjEarnings] = useState("1.0");
  const [adjDeliveries, setAdjDeliveries] = useState("1.0");

  // Premium test form
  const [testWorkerId, setTestWorkerId] = useState("");
  const [testCity, setTestCity] = useState("Chennai");
  const [testTier, setTestTier] = useState("standard");
  const [premiumResult, setPremiumResult] = useState<PremiumResult | null>(null);

  useEffect(() => {
    refresh();
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, []);

  const handleRetrain = async () => {
    setActionLoading("Retrain Model");
    setActionMsg("");
    setActionErr("");
    setRetrainLogs(["Starting retrain..."]);
    setRetrainStatus("running");
    try {
      const res = await fetch("/api/backend/model/retrain", { method: "POST" });
      const data = await res.json();
      if (!res.ok || data.status === "error") {
        setActionErr(data.message || "Failed to start retrain.");
        setRetrainStatus("error");
        setActionLoading("");
        return;
      }
      // Poll for logs every 1.5s
      pollRef.current = setInterval(async () => {
        try {
          const r = await fetch("/api/backend/model/retrain/logs");
          const d = await r.json();
          setRetrainLogs(d.logs?.length ? d.logs : ["Waiting for logs..."]);
          if (!d.running) {
            clearInterval(pollRef.current!);
            pollRef.current = null;
            setActionLoading("");
            if (d.error) {
              setRetrainStatus("error");
              setActionErr(`Retrain failed: ${d.error}`);
            } else {
              setRetrainStatus("done");
              setActionMsg("Retrain completed successfully.");
              refresh();
            }
          }
        } catch {
          // backend blip — keep polling
        }
      }, 1500);
    } catch {
      setActionErr("Backend unavailable.");
      setRetrainStatus("error");
      setActionLoading("");
    }
  };

  const refresh = async () => {
    setLoading(true);
    try {
      const [summaryRes, modelRes] = await Promise.all([
        fetch("/api/backend/admin/data-summary"),
        fetch("/api/backend/model/status"),
      ]);
      if (summaryRes.ok) setDataSummary(await summaryRes.json());
      if (modelRes.ok) setModelStatus(await modelRes.json());
    } catch {
      setActionErr("Failed to connect to backend.");
    } finally {
      setLoading(false);
    }
  };

  const doAction = async (
    label: string,
    url: string,
    method: string = "POST",
    body?: object,
  ) => {
    setActionLoading(label);
    setActionMsg("");
    setActionErr("");
    try {
      const res = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: body ? JSON.stringify(body) : undefined,
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        setActionErr(data.error || `${label} failed.`);
      } else {
        setActionMsg(data.message || `${label} completed.`);
        refresh();
      }
    } catch {
      setActionErr(`${label} failed — backend may be offline.`);
    } finally {
      setActionLoading("");
    }
  };

  const handleAdjustIncome = () => {
    doAction("Adjust Income", "/api/backend/admin/adjust-income", "POST", {
      platform: adjPlatform || null,
      earnings_multiplier: parseFloat(adjEarnings) || 1.0,
      deliveries_multiplier: parseFloat(adjDeliveries) || 1.0,
    });
  };

  const handleTestPremium = async () => {
    if (!testWorkerId) {
      setActionErr("Enter a worker ID to test.");
      return;
    }
    setActionLoading("Test Premium");
    setActionMsg("");
    setActionErr("");
    setPremiumResult(null);
    try {
      const res = await fetch("/api/backend/premium/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          delivery_id: testWorkerId,
          city: testCity,
          tier: testTier,
        }),
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        setActionErr(data.error || "Premium prediction failed.");
      } else {
        setPremiumResult(data);
        setActionMsg("Premium computed successfully.");
      }
    } catch {
      setActionErr("Backend unavailable.");
    } finally {
      setActionLoading("");
    }
  };

  return (
    <div style={{ minHeight: "100vh", background: "#11131E", display: "flex", flexDirection: "column", fontFamily: '"Inter", -apple-system, sans-serif', color: "#E1E1F2" }}>
      <header style={{ background: "#0C0E18", padding: "0 24px", height: 60, display: "flex", alignItems: "center", justifyContent: "space-between", borderBottom: "1px solid rgba(70,69,85,0.6)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 36, height: 36, background: "#4f46e5", borderRadius: 10, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, fontWeight: 800, color: "#fff" }}>GG</div>
          <span style={{ fontSize: 17, fontWeight: 700, color: "#fff" }}>GigGuard</span>
          <span style={{ background: "#059669", color: "#fff", fontSize: 11, fontWeight: 700, padding: "3px 10px", borderRadius: 6, marginLeft: 4 }}>Control Center</span>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <a href="/disruptions" style={{ color: "#fff", fontSize: 13, fontWeight: 600, textDecoration: "none", padding: "6px 14px", background: "#dc2626", borderRadius: 6 }}>🚨 Disruptions</a>
          <a href="/admin" style={{ color: "#94a3b8", fontSize: 13, fontWeight: 600, textDecoration: "none", padding: "6px 14px", border: "1px solid #334155", borderRadius: 6 }}>Admin Dashboard</a>
          <button type="button" onClick={refresh} style={{ background: "#1e293b", color: "#94a3b8", border: "1px solid #334155", borderRadius: 6, padding: "6px 14px", fontSize: 12, fontWeight: 600, cursor: "pointer" }}>Refresh</button>
        </div>
      </header>

      <main style={{ flex: 1, padding: 32 }}>
        <div style={{ maxWidth: 1200, margin: "0 auto" }}>
          <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4, color: "#E1E1F2" }}>Backend Control Center</h1>
          <p style={{ color: "#918FA1", fontSize: 14, marginBottom: 24 }}>
            Monitor data, adjust income parameters, retrain the ML model, and test premium predictions.
          </p>

          {actionMsg && <div style={{ marginBottom: 12, background: "#0A2E18", border: "1px solid rgba(34,197,94,0.3)", color: "#22C55E", fontSize: 13, fontWeight: 600, borderRadius: 8, padding: "8px 12px" }}>{actionMsg}</div>}
          {actionErr && <div style={{ marginBottom: 12, background: "#2E0A0A", border: "1px solid rgba(239,68,68,0.3)", color: "#EF4444", fontSize: 13, fontWeight: 600, borderRadius: 8, padding: "8px 12px" }}>{actionErr}</div>}

          {loading ? (
            <div style={{ padding: 40, textAlign: "center", color: "#918FA1" }}>Loading backend data...</div>
          ) : (
            <>
              {/* Quick Actions */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16, marginBottom: 28 }}>
                <ActionCard
                  title="Generate Data"
                  desc="Trigger one cycle of synthetic data generation for all platforms."
                  btnLabel={actionLoading === "Generate Data" ? "Generating..." : "Generate Now"}
                  disabled={!!actionLoading}
                  onClick={() => doAction("Generate Data", "/api/backend/admin/generate-data")}
                  color="#4f46e5"
                />
                <ActionCard
                  title="Retrain Model"
                  desc="Retrain the XGBoost model on latest data. Takes 10-30 seconds."
                  btnLabel={actionLoading === "Retrain Model" ? "Training..." : "Retrain Now"}
                  disabled={!!actionLoading}
                  onClick={handleRetrain}
                  color="#059669"
                />
                <ActionCard
                  title="Refresh All"
                  desc="Reload data summary and model status from the backend."
                  btnLabel="Refresh"
                  disabled={!!actionLoading}
                  onClick={refresh}
                  color="#0f172a"
                />
              </div>

              {/* Retrain Log Panel */}
              {retrainStatus !== "idle" && (
                <div style={{ background: "#0f172a", borderRadius: 12, border: "1px solid #1e293b", padding: 20, marginBottom: 28 }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <span style={{ fontSize: 14, fontWeight: 700, color: "#f8fafc" }}>Retrain Logs</span>
                      {retrainStatus === "running" && (
                        <span style={{ display: "inline-flex", gap: 4, alignItems: "center" }}>
                          <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#4ade80", animation: "pulse 1.5s infinite" }} />
                          <span style={{ fontSize: 11, color: "#4ade80", fontWeight: 600 }}>RUNNING</span>
                        </span>
                      )}
                      {retrainStatus === "done" && <span style={{ fontSize: 11, color: "#4ade80", fontWeight: 600, background: "#14532d", padding: "2px 8px", borderRadius: 4 }}>DONE</span>}
                      {retrainStatus === "error" && <span style={{ fontSize: 11, color: "#f87171", fontWeight: 600, background: "#450a0a", padding: "2px 8px", borderRadius: 4 }}>ERROR</span>}
                    </div>
                    <button type="button" onClick={() => setRetrainStatus("idle")} style={{ background: "transparent", border: "none", color: "#64748b", cursor: "pointer", fontSize: 18, lineHeight: 1 }}>×</button>
                  </div>
                  {retrainStatus === "running" && (
                    <div style={{ height: 4, background: "#1e293b", borderRadius: 4, marginBottom: 12, overflow: "hidden" }}>
                      <div style={{ height: "100%", background: "linear-gradient(90deg, #4f46e5, #059669)", borderRadius: 4, animation: "progress-slide 2s ease-in-out infinite", width: "60%" }} />
                    </div>
                  )}
                  <div style={{ fontFamily: "monospace", fontSize: 12, color: "#94a3b8", maxHeight: 260, overflowY: "auto", display: "flex", flexDirection: "column", gap: 2 }}>
                    {retrainLogs.map((line, i) => (
                      <div key={i} style={{ color: line.includes("ERROR") ? "#f87171" : line.includes("Done") || line.includes("better") ? "#4ade80" : "#94a3b8" }}>
                        {line}
                      </div>
                    ))}
                    {retrainStatus === "running" && (
                      <div style={{ color: "#4f46e5" }}>_</div>
                    )}
                  </div>
                </div>
              )}
              <style>{`
                @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
                @keyframes progress-slide { 0% { transform: translateX(-100%); } 100% { transform: translateX(200%); } }
              `}</style>

              {/* Model Status */}
              <Section title="ML Model Status">
                {modelStatus ? (
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
                    <InfoCard label="Model Loaded" value={modelStatus.model_loaded ? "Yes" : "No (using formula fallback)"} />
                    <InfoCard label="RMSE" value={modelStatus.metadata ? `${modelStatus.metadata.rmse.toFixed(2)}` : "N/A"} />
                    <InfoCard label="Training Samples" value={modelStatus.metadata?.n_samples?.toString() || "N/A"} />
                    <InfoCard label="Last Trained" value={modelStatus.metadata?.trained_at || "Never"} />
                  </div>
                ) : (
                  <p style={{ color: "#64748b", fontSize: 13 }}>Could not load model status.</p>
                )}
                {modelStatus?.tiers && (
                  <div style={{ marginTop: 16 }}>
                    <h4 style={{ fontSize: 13, fontWeight: 600, color: "#C7C4D8", marginBottom: 8 }}>Tier Configuration</h4>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                      <thead>
                        <tr style={{ background: "#272935" }}>
                          {["Tier", "Rate", "Min Premium", "Max Premium", "Max Payout"].map(h => (
                            <th key={h} style={{ padding: "8px 12px", textAlign: "left", fontWeight: 700, color: "#918FA1", borderBottom: "1px solid rgba(70,69,85,0.6)", fontSize: 11, textTransform: "uppercase" }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(modelStatus.tiers).map(([tier, cfg]) => (
                          <tr key={tier} style={{ borderBottom: "1px solid rgba(70,69,85,0.4)" }}>
                            <td style={{ padding: "8px 12px", fontWeight: 600, color: "#E1E1F2" }}>{tier.charAt(0).toUpperCase() + tier.slice(1)}</td>
                            <td style={{ padding: "8px 12px", color: "#C7C4D8" }}>{(cfg.rate * 100).toFixed(1)}%</td>
                            <td style={{ padding: "8px 12px", color: "#C7C4D8" }}>{cfg.min}</td>
                            <td style={{ padding: "8px 12px", color: "#C7C4D8" }}>{cfg.max}</td>
                            <td style={{ padding: "8px 12px", color: "#C7C4D8" }}>{cfg.max_payout}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </Section>

              {/* Data Summary */}
              <Section title="Platform Data Summary (Today)">
                {dataSummary ? (
                  <>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16, marginBottom: 16 }}>
                      <InfoCard label="Date" value={dataSummary.date} />
                      <InfoCard label="Total Workers (Today)" value={dataSummary.total_workers.toString()} />
                      <InfoCard label="Overall Avg Earnings" value={`${dataSummary.overall_avg_earnings}`} />
                    </div>
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                      <thead>
                        <tr style={{ background: "#272935" }}>
                          {["Platform", "Workers", "Avg Earnings", "Min", "Max", "Avg Deliveries", "Total Earnings"].map(h => (
                            <th key={h} style={{ padding: "8px 12px", textAlign: "left", fontWeight: 700, color: "#918FA1", borderBottom: "1px solid rgba(70,69,85,0.6)", fontSize: 11, textTransform: "uppercase" }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {dataSummary.platforms.map(p => (
                          <tr key={p.platform} style={{ borderBottom: "1px solid rgba(70,69,85,0.4)" }}>
                            <td style={{ padding: "8px 12px", fontWeight: 600, color: "#E1E1F2" }}>{p.platform}</td>
                            <td style={{ padding: "8px 12px", color: "#C7C4D8" }}>{p.worker_count}</td>
                            <td style={{ padding: "8px 12px", color: "#C7C4D8" }}>{p.avg_earnings}</td>
                            <td style={{ padding: "8px 12px", color: "#C7C4D8" }}>{p.min_earnings}</td>
                            <td style={{ padding: "8px 12px", color: "#C7C4D8" }}>{p.max_earnings}</td>
                            <td style={{ padding: "8px 12px", color: "#C7C4D8" }}>{p.avg_deliveries}</td>
                            <td style={{ padding: "8px 12px", color: "#C7C4D8" }}>{p.total_earnings}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </>
                ) : (
                  <p style={{ color: "#64748b", fontSize: 13 }}>No data available. Generate data first.</p>
                )}
              </Section>

              {/* Adjust Income */}
              <Section title="Adjust Income Data">
                <p style={{ fontSize: 13, color: "#64748b", marginBottom: 16 }}>
                  Multiply today&apos;s earnings/deliveries for a platform to test how the ML model responds. Use this to verify the model produces different premiums for different income levels.
                </p>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 16 }}>
                  <div>
                    <label style={labelStyle}>Platform (empty = all)</label>
                    <select value={adjPlatform} onChange={e => setAdjPlatform(e.target.value)} style={selectStyle}>
                      <option value="">All platforms</option>
                      {PLATFORMS.map(p => <option key={p} value={p}>{p}</option>)}
                    </select>
                  </div>
                  <div>
                    <label style={labelStyle}>Earnings Multiplier</label>
                    <input type="number" step="0.1" min="0.1" max="10" value={adjEarnings} onChange={e => setAdjEarnings(e.target.value)} style={inputStyle} />
                  </div>
                  <div>
                    <label style={labelStyle}>Deliveries Multiplier</label>
                    <input type="number" step="0.1" min="0.1" max="10" value={adjDeliveries} onChange={e => setAdjDeliveries(e.target.value)} style={inputStyle} />
                  </div>
                  <div style={{ display: "flex", alignItems: "flex-end" }}>
                    <button type="button" onClick={handleAdjustIncome} disabled={!!actionLoading} style={{ ...btnStyle, background: "#d97706", width: "100%" }}>
                      {actionLoading === "Adjust Income" ? "Applying..." : "Apply Adjustment"}
                    </button>
                  </div>
                </div>
                <p style={{ fontSize: 12, color: "#94a3b8" }}>
                  Example: Set earnings to 0.3x to simulate low-income scenario, or 3.0x for high-income. Then retrain and check premium predictions.
                </p>
              </Section>

              {/* Test Premium */}
              <Section title="Test Premium Prediction">
                <p style={{ fontSize: 13, color: "#64748b", marginBottom: 16 }}>
                  Enter a worker ID from the data above and compute their predicted premium to verify the model is working.
                </p>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 16 }}>
                  <div>
                    <label style={labelStyle}>Worker ID</label>
                    <input type="text" placeholder="UUID from platform data" value={testWorkerId} onChange={e => setTestWorkerId(e.target.value)} style={inputStyle} />
                  </div>
                  <div>
                    <label style={labelStyle}>City</label>
                    <select value={testCity} onChange={e => setTestCity(e.target.value)} style={selectStyle}>
                      {CITIES.map(c => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                  <div>
                    <label style={labelStyle}>Tier</label>
                    <select value={testTier} onChange={e => setTestTier(e.target.value)} style={selectStyle}>
                      <option value="basic">Basic</option>
                      <option value="standard">Standard</option>
                      <option value="pro">Pro</option>
                    </select>
                  </div>
                  <div style={{ display: "flex", alignItems: "flex-end" }}>
                    <button type="button" onClick={handleTestPremium} disabled={!!actionLoading} style={{ ...btnStyle, background: "#4f46e5", width: "100%" }}>
                      {actionLoading === "Test Premium" ? "Computing..." : "Compute Premium"}
                    </button>
                  </div>
                </div>
                {premiumResult && (
                  <div style={{ background: "#1D1B45", border: "1px solid rgba(108,99,255,0.3)", borderRadius: 10, padding: 20, marginTop: 8 }}>
                    <h4 style={{ fontSize: 14, fontWeight: 700, marginBottom: 12, color: "#8B84FF" }}>Premium Result</h4>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, fontSize: 13 }}>
                      <ResultItem label="Weekly Premium" value={`${premiumResult.weekly_premium}`} />
                      <ResultItem label="With AutoPay" value={`${premiumResult.weekly_premium_autopay}`} />
                      <ResultItem label="Raw Prediction" value={`${premiumResult.raw_prediction}`} />
                      <ResultItem label="Tier" value={premiumResult.tier} />
                      <ResultItem label="Max Payout" value={`${premiumResult.max_payout}`} />
                      <ResultItem label="Weather Risk" value={`${(premiumResult.weather_risk * 100).toFixed(1)}%`} />
                      <ResultItem label="City Risk" value={`${premiumResult.city_risk}`} />
                      <ResultItem label="Weekly Earnings Est" value={`${premiumResult.weekly_earnings_est}`} />
                      <ResultItem label="History Days" value={`${premiumResult.history_days}`} />
                    </div>
                  </div>
                )}
              </Section>
            </>
          )}
        </div>
      </main>
    </div>
  );
}

const labelStyle: React.CSSProperties = { display: "block", fontSize: 12, fontWeight: 600, color: "#918FA1", marginBottom: 4 };
const inputStyle: React.CSSProperties = { width: "100%", height: 38, padding: "0 10px", fontSize: 13, border: "1px solid rgba(70,69,85,0.6)", borderRadius: 8, outline: "none", background: "#323440", color: "#E1E1F2" };
const selectStyle: React.CSSProperties = { ...inputStyle, cursor: "pointer" };
const btnStyle: React.CSSProperties = { height: 38, color: "#fff", border: "none", borderRadius: 8, fontSize: 13, fontWeight: 600, cursor: "pointer", padding: "0 16px" };

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ background: "#1D1F2B", borderRadius: 12, border: "1px solid rgba(70,69,85,0.6)", padding: 24, marginBottom: 24 }}>
      <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16, color: "#E1E1F2" }}>{title}</h3>
      {children}
    </div>
  );
}

function ActionCard({ title, desc, btnLabel, disabled, onClick, color }: {
  title: string; desc: string; btnLabel: string; disabled: boolean; onClick: () => void; color: string;
}) {
  return (
    <div style={{ background: "#1D1F2B", borderRadius: 12, border: "1px solid rgba(70,69,85,0.6)", padding: 20 }}>
      <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 4, color: "#E1E1F2" }}>{title}</h3>
      <p style={{ fontSize: 12, color: "#918FA1", marginBottom: 14, lineHeight: 1.5 }}>{desc}</p>
      <button type="button" disabled={disabled} onClick={onClick} style={{ background: color, color: "#fff", border: "none", borderRadius: 8, padding: "8px 18px", fontSize: 13, fontWeight: 600, cursor: disabled ? "not-allowed" : "pointer", opacity: disabled ? 0.6 : 1, width: "100%" }}>
        {btnLabel}
      </button>
    </div>
  );
}

function InfoCard({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ background: "#272935", borderRadius: 8, padding: "12px 16px", border: "1px solid rgba(70,69,85,0.6)" }}>
      <div style={{ fontSize: 11, color: "#918FA1", fontWeight: 600, textTransform: "uppercase", marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 15, fontWeight: 700, color: "#E1E1F2" }}>{value}</div>
    </div>
  );
}

function ResultItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div style={{ fontSize: 11, color: "#918FA1", fontWeight: 500 }}>{label}</div>
      <div style={{ fontSize: 14, fontWeight: 700, color: "#E1E1F2" }}>{value}</div>
    </div>
  );
}
