"use client";
import React, { useEffect, useState } from "react";

type WorkerRef = {
  id: string;
  name: string;
  email: string;
  city: string;
  delivery_id?: string;
} | null;

type Payment = {
  transaction_id: string;
  worker_id: string;
  amount: number;
  method: string;
  tier: string;
  status: string;
  created_at: string;
  worker: WorkerRef;
};

type WeekEntry = {
  week: string;
  total: number;
  count: number;
};

type Analytics = {
  totalReceived: number;
  thisWeekTotal: number;
  transactionCount: number;
  weeklyBreakdown: WeekEntry[];
};

const METHOD_LABELS: Record<string, string> = {
  upi: "UPI",
  debit: "Debit Card",
  credit: "Credit Card",
};

const TIER_COLORS: Record<string, { bg: string; color: string }> = {
  basic: { bg: "#EFF6FF", color: "#1D4ED8" },
  standard: { bg: "#F0FDF4", color: "#15803D" },
  pro: { bg: "#FDF4FF", color: "#7E22CE" },
};

const IconRefresh = () => (
  <svg width="14" height="14" fill="none" viewBox="0 0 14 14">
    <path d="M12.25 7A5.25 5.25 0 1 1 7 1.75" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    <path d="M7 1.75L9.5 4.25 7 6.75" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const IconDownload = () => (
  <svg width="14" height="14" fill="none" viewBox="0 0 14 14">
    <path d="M7 2v7M4 7l3 3 3-3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M2 11h10" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
  </svg>
);

const IconSearch = () => (
  <svg width="14" height="14" fill="none" viewBox="0 0 14 14">
    <circle cx="6" cy="6" r="4.5" stroke="#94A3B8" strokeWidth="1.3" />
    <path d="M10 10l2.5 2.5" stroke="#94A3B8" strokeWidth="1.3" strokeLinecap="round" />
  </svg>
);

function money(n: number) {
  return "₹" + n.toLocaleString("en-IN", { minimumFractionDigits: 0, maximumFractionDigits: 0 });
}

function fmtDate(iso: string) {
  return new Date(iso).toLocaleDateString("en-IN", {
    day: "numeric", month: "short", year: "numeric",
  });
}

function fmtWeek(iso: string) {
  const start = new Date(iso);
  const end = new Date(iso);
  end.setDate(end.getDate() + 6);
  return `${start.toLocaleDateString("en-IN", { day: "numeric", month: "short" })} – ${end.toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" })}`;
}

export default function PaymentsPage() {
  const [payments, setPayments] = useState<Payment[]>([]);
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [methodFilter, setMethodFilter] = useState("all");
  const [tierFilter, setTierFilter] = useState("all");
  const [tab, setTab] = useState<"transactions" | "weekly">("transactions");

  const load = () => {
    setLoading(true);
    setError("");
    fetch("/api/admin/payments")
      .then((r) => r.json())
      .then((data) => {
        if (data.payments) setPayments(data.payments);
        if (data.analytics) setAnalytics(data.analytics);
        else if (data.error) setError(data.error);
      })
      .catch(() => setError("Failed to fetch payments."))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const filtered = payments.filter((p) => {
    const q = search.toLowerCase();
    const matchSearch =
      (p.worker?.name ?? "").toLowerCase().includes(q) ||
      (p.worker?.email ?? "").toLowerCase().includes(q) ||
      p.transaction_id.toLowerCase().includes(q) ||
      (p.worker?.city ?? "").toLowerCase().includes(q);
    const matchMethod = methodFilter === "all" || p.method === methodFilter;
    const matchTier = tierFilter === "all" || p.tier === tierFilter;
    return matchSearch && matchMethod && matchTier;
  });

  const maxWeek = analytics?.weeklyBreakdown?.[0]?.total ?? 1;

  // Profit estimate: total received - (assume platform keeps 20% as admin fee, rest for claims)
  const claimsReserve = analytics ? Math.round(analytics.totalReceived * 0.80) : 0;
  const adminFee = analytics ? Math.round(analytics.totalReceived * 0.20) : 0;

  return (
    <div>
      {/* Header */}
      <div className="pageHead">
        <div>
          <h1 className="pageTitle">Payments</h1>
          <p className="pageSubtitle">
            Weekly premium collection, transaction history, and revenue analytics.
          </p>
        </div>
        <div className="pageActions">
          <button type="button" className="btn btnSecondary" onClick={load}>
            <IconRefresh /> Refresh
          </button>
          <button type="button" className="btn btnPrimary">
            <IconDownload /> Export
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="gridStats" style={{ gridTemplateColumns: "repeat(4, 1fr)", marginBottom: 24 }}>
        <div className="statCard">
          <div className="statLabel">Total Collected</div>
          <div className="statValue">{loading ? "—" : money(analytics?.totalReceived ?? 0)}</div>
          <div className="statSub">{loading ? "" : `${analytics?.transactionCount ?? 0} successful payments`}</div>
        </div>
        <div className="statCard">
          <div className="statLabel">This Week</div>
          <div className="statValue" style={{ color: "#16A34A" }}>
            {loading ? "—" : money(analytics?.thisWeekTotal ?? 0)}
          </div>
          <div className="statSub">Current week collection</div>
        </div>
        <div className="statCard">
          <div className="statLabel">Claims Reserve (80%)</div>
          <div className="statValue" style={{ color: "#2563EB" }}>
            {loading ? "—" : money(claimsReserve)}
          </div>
          <div className="statSub">Held for claim payouts</div>
        </div>
        <div className="statCard">
          <div className="statLabel">Platform Revenue (20%)</div>
          <div className="statValue" style={{ color: "#D97706" }}>
            {loading ? "—" : money(adminFee)}
          </div>
          <div className="statSub">Administration & operations</div>
        </div>
      </div>

      {error && <div className="alertError">{error}</div>}

      {/* Tabs */}
      <div className="card">
        <div className="sectionHead" style={{ borderBottom: "1px solid var(--border)", marginBottom: 0, paddingBottom: 0 }}>
          <div className="tabs" style={{ borderBottom: "none" }}>
            {(["transactions", "weekly"] as const).map((t) => (
              <button
                key={t}
                type="button"
                className={`tabBtn ${tab === t ? "tabBtnActive" : ""}`}
                onClick={() => setTab(t)}
              >
                {t === "transactions" ? "All Transactions" : "Weekly Breakdown"}
              </button>
            ))}
          </div>
        </div>

        {tab === "transactions" && (
          <>
            {/* Filters */}
            <div
              style={{
                display: "flex", gap: 8, alignItems: "center",
                padding: "14px 20px", borderBottom: "1px solid var(--border)",
              }}
            >
              <div
                style={{
                  display: "flex", alignItems: "center", gap: 8,
                  background: "var(--elevated)", border: "1px solid var(--border)",
                  borderRadius: 8, padding: "0 12px", height: 36, flex: 1, maxWidth: 320,
                }}
              >
                <IconSearch />
                <input
                  placeholder="Search worker, city, or transaction ID…"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  style={{
                    border: "none", background: "transparent", outline: "none",
                    fontSize: 13, color: "var(--white)", width: "100%", fontFamily: "inherit",
                  }}
                />
              </div>
              <select
                className="select"
                value={methodFilter}
                onChange={(e) => setMethodFilter(e.target.value)}
              >
                <option value="all">All methods</option>
                <option value="upi">UPI</option>
                <option value="debit">Debit Card</option>
                <option value="credit">Credit Card</option>
              </select>
              <select
                className="select"
                value={tierFilter}
                onChange={(e) => setTierFilter(e.target.value)}
              >
                <option value="all">All tiers</option>
                <option value="basic">Basic</option>
                <option value="standard">Standard</option>
                <option value="pro">Pro</option>
              </select>
              <span
                style={{
                  fontSize: 12, fontWeight: 500, color: "var(--faint)",
                  background: "var(--elevated)", border: "1px solid var(--border)",
                  borderRadius: 20, padding: "3px 10px",
                }}
              >
                {filtered.length}
              </span>
            </div>

            {loading ? (
              <div className="loading">Loading transactions…</div>
            ) : filtered.length === 0 ? (
              <div className="empty">No transactions found.</div>
            ) : (
              <>
                <div className="tableWrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Worker</th>
                        <th>Transaction ID</th>
                        <th>Amount</th>
                        <th>Tier</th>
                        <th>Method</th>
                        <th>Status</th>
                        <th>Date</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filtered.map((p) => {
                        const tierStyle = TIER_COLORS[p.tier] ?? { bg: "#F1F5F9", color: "#64748B" };
                        return (
                          <tr key={p.transaction_id}>
                            <td>
                              <div style={{ fontWeight: 600, fontSize: 13 }}>
                                {p.worker?.name ?? <span style={{ color: "var(--faint)" }}>Unknown</span>}
                              </div>
                              <div style={{ fontSize: 12, color: "var(--faint)" }}>
                                {p.worker?.email ?? p.worker_id.slice(0, 16) + "…"}
                              </div>
                              {p.worker?.city && (
                                <div style={{ fontSize: 11, color: "var(--faint)", marginTop: 1 }}>
                                  {p.worker.city}
                                </div>
                              )}
                            </td>
                            <td>
                              <span className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>
                                {p.transaction_id}
                              </span>
                            </td>
                            <td>
                              <span style={{ fontWeight: 700, fontSize: 14, color: "#16A34A" }}>
                                {money(p.amount)}
                              </span>
                            </td>
                            <td>
                              <span
                                style={{
                                  background: tierStyle.bg, color: tierStyle.color,
                                  padding: "2px 8px", borderRadius: 4,
                                  fontSize: 11, fontWeight: 600,
                                }}
                              >
                                {p.tier.charAt(0).toUpperCase() + p.tier.slice(1)}
                              </span>
                            </td>
                            <td style={{ color: "var(--muted)", fontSize: 13 }}>
                              {METHOD_LABELS[p.method] ?? p.method}
                            </td>
                            <td>
                              <span className={`badge ${p.status === "success" ? "badgeGreen" : "badgeRed"}`}>
                                {p.status === "success" ? "Success" : "Failed"}
                              </span>
                            </td>
                            <td style={{ fontSize: 12, color: "var(--faint)", whiteSpace: "nowrap" }}>
                              {fmtDate(p.created_at)}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <div className="paginationRow">
                  <span>Showing {filtered.length} of {payments.length} transactions</span>
                </div>
              </>
            )}
          </>
        )}

        {tab === "weekly" && (
          <div style={{ padding: "20px" }}>
            {loading ? (
              <div className="loading">Loading weekly data…</div>
            ) : !analytics?.weeklyBreakdown?.length ? (
              <div className="empty">No weekly data available.</div>
            ) : (
              <>
                {/* Weekly bar chart */}
                <div style={{ marginBottom: 28 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: "var(--faint)", letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 16 }}>
                    Revenue per Week
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                    {analytics.weeklyBreakdown.map((w) => {
                      const pct = maxWeek > 0 ? Math.max(4, Math.round((w.total / maxWeek) * 100)) : 4;
                      return (
                        <div key={w.week} style={{ display: "flex", alignItems: "center", gap: 16 }}>
                          <div style={{ width: 160, fontSize: 12, color: "var(--muted)", flexShrink: 0 }}>
                            {fmtWeek(w.week)}
                          </div>
                          <div style={{ flex: 1, height: 10, background: "var(--elevated)", borderRadius: 6, overflow: "hidden" }}>
                            <div
                              style={{
                                height: "100%", width: `${pct}%`,
                                background: "linear-gradient(90deg, #1E293B, #3B82F6)",
                                borderRadius: 6,
                                transition: "width 0.4s ease",
                              }}
                            />
                          </div>
                          <div style={{ width: 80, textAlign: "right", fontWeight: 700, fontSize: 13, color: "var(--white)", flexShrink: 0 }}>
                            {money(w.total)}
                          </div>
                          <div style={{ width: 60, textAlign: "right", fontSize: 12, color: "var(--faint)", flexShrink: 0 }}>
                            {w.count} txn{w.count !== 1 ? "s" : ""}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Weekly table */}
                <div className="tableWrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Week</th>
                        <th>Transactions</th>
                        <th>Total Collected</th>
                        <th>Claims Reserve (80%)</th>
                        <th>Platform Revenue (20%)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analytics.weeklyBreakdown.map((w) => (
                        <tr key={w.week}>
                          <td style={{ fontWeight: 500 }}>{fmtWeek(w.week)}</td>
                          <td style={{ color: "var(--muted)" }}>{w.count}</td>
                          <td>
                            <span style={{ fontWeight: 700, color: "#16A34A" }}>{money(w.total)}</span>
                          </td>
                          <td style={{ color: "#2563EB", fontWeight: 600 }}>
                            {money(Math.round(w.total * 0.8))}
                          </td>
                          <td style={{ color: "#D97706", fontWeight: 600 }}>
                            {money(Math.round(w.total * 0.2))}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
