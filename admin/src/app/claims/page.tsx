"use client";
import React, { useEffect, useState } from "react";

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
  reviewed_at: string | null;
  paid_at: string | null;
  transaction_id: string | null;
  fuzzy_payout_multiplier: number;
};

const TRIGGER_TYPE_LABELS: Record<string, string> = {
  heavy_rain: "Heavy Rain",
  extreme_heat: "Extreme Heat",
  severe_aqi: "Severe AQI",
  flood: "Flood",
  traffic_congestion: "Traffic Congestion",
  curfew: "Curfew / Unrest",
};

const PLATFORM_NAMES: Record<string, string> = {
  swiggy: "Swiggy",
  zomato: "Zomato",
  amazon: "Amazon Flex",
  blinkit: "Blinkit",
  zepto: "Zepto",
  meesho: "Meesho",
  porter: "Porter",
  dunzo: "Dunzo",
};

function ClaimStatusBadge({ status }: { status: string }) {
  const s = status?.toLowerCase();
  if (s === "approved")
    return <span className="badge badgeGreen">Approved</span>;
  if (s === "paid") return <span className="badge badgeGreen">Paid</span>;
  if (s === "rejected") return <span className="badge badgeRed">Rejected</span>;
  if (s === "under_review")
    return <span className="badge badgeBlue">Under Review</span>;
  if (s === "pending") return <span className="badge badgeAmber">Pending</span>;
  return <span className="badge badgeGray">{status}</span>;
}

function FraudBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color = pct >= 75 ? "#DC2626" : pct >= 40 ? "#D97706" : "#16A34A";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div
        style={{
          width: 48,
          height: 5,
          borderRadius: 3,
          background: "#E2E8F0",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${pct}%`,
            background: color,
            borderRadius: 3,
          }}
        />
      </div>
      <span style={{ fontSize: 12, fontWeight: 600, color }}>{pct}%</span>
    </div>
  );
}

const IconRefresh = () => (
  <svg width="14" height="14" fill="none" viewBox="0 0 14 14">
    <path
      d="M12.25 7A5.25 5.25 0 1 1 7 1.75"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
    />
    <path
      d="M7 1.75L9.5 4.25 7 6.75"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const IconChevron = ({ open }: { open: boolean }) => (
  <svg
    width="14"
    height="14"
    fill="none"
    viewBox="0 0 14 14"
    style={{
      transform: open ? "rotate(180deg)" : "rotate(0deg)",
      transition: "transform 0.2s",
    }}
  >
    <path
      d="M3 5l4 4 4-4"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

// delivery_id → sorted payment dates (ms) for coverage check
type PaymentCoverage = Map<string, number[]>;

function hasPremiumCoverage(coverage: PaymentCoverage, deliveryId: string, claimCreatedAt: string): boolean {
  const dates = coverage.get(deliveryId.toLowerCase());
  if (!dates || dates.length === 0) return false;
  const claimMs = new Date(claimCreatedAt).getTime();
  const weekMs = 7 * 24 * 60 * 60 * 1000;
  return dates.some((d) => d <= claimMs && d >= claimMs - weekMs);
}

export default function ClaimsPage() {
  const [claims, setClaims] = useState<Claim[]>([]);
  const [loading, setLoading] = useState(true);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [expandedClaim, setExpandedClaim] = useState<string | null>(null);
  const [approveAmounts, setApproveAmounts] = useState<Record<string, string>>(
    {},
  );
  const [updatingId, setUpdatingId] = useState<string | null>(null);
  const [coverage, setCoverage] = useState<PaymentCoverage>(new Map());

  useEffect(() => {
    fetchClaims();
    fetchCoverage();
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    const q = (params.get("q") || "").trim().toLowerCase();
    setSearchQuery(q);
  }, []);

  const fetchCoverage = () => {
    fetch("/api/admin/payments")
      .then((r) => r.json())
      .then((data) => {
        const map: PaymentCoverage = new Map();
        for (const p of data.payments || []) {
          if (p.status !== "success" || !p.worker?.delivery_id) continue;
          const key = String(p.worker.delivery_id).toLowerCase();
          if (!map.has(key)) map.set(key, []);
          map.get(key)!.push(new Date(p.created_at).getTime());
        }
        setCoverage(map);
      })
      .catch(() => { /* silently fail — coverage badges become "unknown" */ });
  };

  const fetchClaims = () => {
    setLoading(true);
    fetch("/api/claims/all")
      .then((r) => r.json())
      .then((data) => {
        setClaims(data.claims || data.data || []);
      })
      .catch(() => setError("Failed to load claims"))
      .finally(() => setLoading(false));
  };

  const rescoreClaim = async (claimId: string) => {
    setUpdatingId(claimId);
    setNotice("");
    setError("");
    try {
      const res = await fetch("/api/claims/rescore", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ claimId }),
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        setError(data.error || "Rescore failed");
        return;
      }
      setClaims((prev) =>
        prev.map((c) =>
          c.id !== claimId ? c : {
            ...c,
            fraud_score: data.fraud_score ?? c.fraud_score,
            fraud_flags: data.fraud_flags ?? c.fraud_flags,
            ...(data.claim ?? {}),
          }
        )
      );
      setNotice(`Fraud score recalculated: ${Math.round((data.fraud_score ?? 0) * 100)}%`);
    } catch {
      setError("Rescore request failed");
    } finally {
      setUpdatingId(null);
    }
  };

  const updateClaim = async (
    claimId: string,
    status: string,
    amount?: number,
  ) => {
    setUpdatingId(claimId);
    setNotice("");
    setError("");
    try {
      const res = await fetch("/api/claims/update", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          claimId,
          status,
          approved_amount: amount ?? null,
        }),
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        setError(data.error || "Failed to update claim");
        return;
      }
      setClaims((prev) =>
        prev.map((c) => (c.id === claimId ? { ...c, ...data.claim } : c)),
      );
      setNotice(`Claim ${status === "paid" ? "marked as paid" : status}.`);
      setExpandedClaim(null);
      fetchClaims();
    } catch {
      setError("Failed to update claim");
    } finally {
      setUpdatingId(null);
    }
  };

  const filtered = claims.filter((c) => {
    const statusMatch =
      statusFilter === "all" ||
      c.status === statusFilter ||
      c.payout_status === statusFilter;

    if (!statusMatch) return false;
    if (!searchQuery) return true;

    const triggerLabel = TRIGGER_TYPE_LABELS[c.trigger_type] || c.trigger_type;
    const platformLabel = PLATFORM_NAMES[c.platform] || c.platform;
    const searchable = [
      c.claim_number,
      c.worker_id,
      c.city,
      c.platform,
      platformLabel,
      c.trigger_id,
      c.trigger_type,
      triggerLabel,
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();

    return searchable.includes(searchQuery);
  });

  const stats = {
    total: claims.length,
    pending: claims.filter(
      (c) => c.payout_status === "pending" || c.status === "under_review",
    ).length,
    approved: claims.filter(
      (c) => c.payout_status === "approved" || c.status === "approved",
    ).length,
    paid: claims.filter((c) => c.payout_status === "paid").length,
    totalPayout: claims.reduce((s, c) => s + (c.payout_amount || 0), 0),
  };

  return (
    <div>
      {/* Header */}
      <div className="pageHead">
        <div>
          <h1 className="pageTitle">Claims</h1>
          <p className="pageSubtitle">
            Insurance claim lifecycle — review, approve, reject, and track
            payouts.
          </p>
        </div>
        <div className="pageActions">
          <button
            type="button"
            className="btn btnSecondary"
            onClick={fetchClaims}
          >
            <IconRefresh /> Refresh
          </button>
        </div>
      </div>

      {/* Stats */}
      <div
        className="gridStats"
        style={{ gridTemplateColumns: "repeat(5, 1fr)", marginBottom: 24 }}
      >
        {[
          {
            label: "Total Claims",
            value: loading ? "—" : stats.total.toLocaleString(),
          },
          {
            label: "Awaiting Action",
            value: loading ? "—" : stats.pending.toLocaleString(),
          },
          {
            label: "Approved",
            value: loading ? "—" : stats.approved.toLocaleString(),
          },
          {
            label: "Paid Out",
            value: loading ? "—" : stats.paid.toLocaleString(),
          },
          {
            label: "Total Payout",
            value: loading ? "—" : `₹${stats.totalPayout.toLocaleString()}`,
          },
        ].map(({ label, value }) => (
          <div key={label} className="statCard">
            <div className="statLabel">{label}</div>
            <div className="statValue" style={{ fontSize: 22 }}>
              {value}
            </div>
          </div>
        ))}
      </div>

      {notice && <div className="alertSuccess">{notice}</div>}
      {error && <div className="alertError">{error}</div>}

      {/* Table */}
      <div className="card">
        <div className="sectionHead">
          <div className="sectionTitle">Insurance Claims</div>
          <select
            className="select"
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
          >
            <option value="all">All statuses</option>
            <option value="pending">Pending</option>
            <option value="under_review">Under Review</option>
            <option value="approved">Approved</option>
            <option value="paid">Paid</option>
            <option value="rejected">Rejected</option>
          </select>
        </div>

        {loading ? (
          <div className="loading">Loading claims…</div>
        ) : filtered.length === 0 ? (
          <div className="empty">No claims found.</div>
        ) : (
          <div className="tableWrap">
            <table>
              <thead>
                <tr>
                  <th>Claim #</th>
                  <th>Worker / City</th>
                  <th>Platform</th>
                  <th>Trigger</th>
                  <th>Payout</th>
                  <th>Fraud Score</th>
                  <th>Coverage</th>
                  <th>Status</th>
                  <th>Created</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((c) => (
                  <React.Fragment key={c.id}>
                    <tr
                      style={{
                        background:
                          expandedClaim === c.id ? "#FAFBFD" : undefined,
                        borderBottom:
                          expandedClaim === c.id ? "none" : undefined,
                      }}
                    >
                      <td>
                        <span
                          className="mono"
                          style={{ fontWeight: 700, fontSize: 12 }}
                        >
                          {c.claim_number}
                        </span>
                      </td>
                      <td>
                        <div
                          className="mono"
                          style={{
                            fontSize: 11,
                            color: "var(--muted)",
                            marginBottom: 2,
                          }}
                        >
                          {c.worker_id?.slice(0, 12)}…
                        </div>
                        <div style={{ fontSize: 11, color: "var(--faint)" }}>
                          {c.city}
                        </div>
                      </td>
                      <td style={{ fontSize: 12, color: "var(--muted)" }}>
                        {PLATFORM_NAMES[c.platform] || c.platform}
                      </td>
                      <td style={{ fontSize: 13, color: "var(--muted)" }}>
                        {TRIGGER_TYPE_LABELS[c.trigger_type] || c.trigger_type}
                      </td>
                      <td style={{ fontWeight: 700, fontSize: 14 }}>
                        ₹{c.payout_amount?.toLocaleString()}
                      </td>
                      <td>
                        <FraudBar score={c.fraud_score} />
                      </td>
                      <td>
                        {coverage.size === 0 ? (
                          <span className="badge badgeGray">—</span>
                        ) : hasPremiumCoverage(coverage, c.worker_id, c.created_at) ? (
                          <span className="badge badgeGreen">Paid ✓</span>
                        ) : (
                          <span className="badge badgeRed">Unpaid ✗</span>
                        )}
                      </td>
                      <td>
                        <ClaimStatusBadge
                          status={c.payout_status || c.status}
                        />
                      </td>
                      <td
                        style={{
                          fontSize: 12,
                          color: "var(--faint)",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {new Date(c.created_at).toLocaleDateString("en-IN", {
                          day: "numeric",
                          month: "short",
                        })}
                      </td>
                      <td>
                        <div
                          style={{ display: "flex", gap: 4, flexWrap: "wrap" }}
                        >
                          <button
                            type="button"
                            className="btn btnSm btnOutline"
                            onClick={() =>
                              setExpandedClaim(
                                expandedClaim === c.id ? null : c.id,
                              )
                            }
                          >
                            {expandedClaim === c.id ? "Close" : "Review"}
                            <IconChevron open={expandedClaim === c.id} />
                          </button>
                          {c.payout_status === "approved" && (
                            <button
                              type="button"
                              className="btn btnSm"
                              disabled={updatingId === c.id}
                              onClick={() => updateClaim(c.id, "paid")}
                              style={{
                                background: "#F5F3FF",
                                color: "#7C3AED",
                                border: "1px solid #DDD6FE",
                              }}
                            >
                              Mark Paid
                            </button>
                          )}
                          {(c.payout_status === "pending" ||
                            c.status === "under_review") && (
                            <>
                              <button
                                type="button"
                                className="btn btnSm"
                                disabled={updatingId === c.id}
                                onClick={() => updateClaim(c.id, "approved")}
                                style={{
                                  background: "#F0FDF4",
                                  color: "#166534",
                                  border: "1px solid #86EFAC",
                                }}
                              >
                                Approve
                              </button>
                              <button
                                type="button"
                                className="btn btnSm"
                                disabled={updatingId === c.id}
                                onClick={() => updateClaim(c.id, "rejected")}
                                style={{
                                  background: "#FEF2F2",
                                  color: "#B91C1C",
                                  border: "1px solid #FECACA",
                                }}
                              >
                                Reject
                              </button>
                            </>
                          )}
                          {c.fraud_score >= 0.75 && (
                            <button
                              type="button"
                              className="btn btnSm"
                              disabled={updatingId === c.id}
                              onClick={() => rescoreClaim(c.id)}
                              style={{
                                background: "#F0F9FF",
                                color: "#0369A1",
                                border: "1px solid #BAE6FD",
                              }}
                            >
                              {updatingId === c.id ? "…" : "Rescore"}
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>

                    {expandedClaim === c.id && (
                      <tr>
                        <td
                          colSpan={9}
                          style={{ padding: 0, background: "#FAFBFD" }}
                        >
                          <div
                            style={{
                              display: "grid",
                              gridTemplateColumns: "1fr 1fr",
                              gap: 24,
                              padding: "16px 20px 20px",
                              borderBottom: "1px solid var(--border)",
                            }}
                          >
                            {/* Left: claim details */}
                            <div>
                              <div
                                style={{
                                  fontSize: 11,
                                  fontWeight: 700,
                                  textTransform: "uppercase",
                                  letterSpacing: "0.05em",
                                  color: "var(--faint)",
                                  marginBottom: 10,
                                }}
                              >
                                Claim Details
                              </div>
                              <div
                                style={{
                                  display: "grid",
                                  gridTemplateColumns: "1fr 1fr",
                                  gap: "8px 16px",
                                }}
                              >
                                {[
                                  ["Daily Wage Est.", `₹${c.daily_wage_est}`],
                                  ["Disrupted Hours", `${c.disrupted_hours}h`],
                                  [
                                    "Cross-platform",
                                    c.cross_platform_clear
                                      ? "Clear ✓"
                                      : "Failed ✗",
                                  ],
                                  [
                                    "Multiplier",
                                    `${c.fuzzy_payout_multiplier?.toFixed(2)}x`,
                                  ],
                                  ["Trigger ID", c.trigger_id || "—"],
                                ].map(([k, v]) => (
                                  <div key={k}>
                                    <div
                                      style={{
                                        fontSize: 11,
                                        color: "var(--faint)",
                                        marginBottom: 2,
                                      }}
                                    >
                                      {k}
                                    </div>
                                    <div
                                      style={{
                                        fontSize: 13,
                                        fontWeight: 600,
                                        color:
                                          v === "Yes ✓" || v === "Clear ✓"
                                            ? "#16A34A"
                                            : v === "No ✗" || v === "Failed ✗"
                                              ? "#DC2626"
                                              : "var(--white)",
                                      }}
                                    >
                                      {v}
                                    </div>
                                  </div>
                                ))}
                              </div>

                              {c.fraud_flags?.length > 0 && (
                                <div style={{ marginTop: 14 }}>
                                  <div
                                    style={{
                                      fontSize: 11,
                                      fontWeight: 700,
                                      color: "var(--faint)",
                                      textTransform: "uppercase",
                                      letterSpacing: "0.05em",
                                      marginBottom: 6,
                                    }}
                                  >
                                    Fraud Flags
                                  </div>
                                  <div
                                    style={{
                                      display: "flex",
                                      gap: 6,
                                      flexWrap: "wrap",
                                    }}
                                  >
                                    {c.fraud_flags.map((flag) => (
                                      <span
                                        key={flag}
                                        className="badge badgeAmber"
                                        style={{ fontSize: 10 }}
                                      >
                                        {flag.replace(/_/g, " ")}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              )}

                              {c.transaction_id && (
                                <div
                                  style={{
                                    marginTop: 12,
                                    fontSize: 11,
                                    color: "var(--faint)",
                                  }}
                                >
                                  TXN:{" "}
                                  <span className="mono">
                                    {c.transaction_id}
                                  </span>
                                </div>
                              )}
                            </div>

                            {/* Right: payout adjustment */}
                            <div>
                              <div
                                style={{
                                  fontSize: 11,
                                  fontWeight: 700,
                                  textTransform: "uppercase",
                                  letterSpacing: "0.05em",
                                  color: "var(--faint)",
                                  marginBottom: 10,
                                }}
                              >
                                Payout Adjustment
                              </div>
                              <label
                                style={{
                                  display: "block",
                                  fontSize: 12,
                                  color: "var(--muted)",
                                  marginBottom: 6,
                                }}
                              >
                                Override payout amount (₹)
                              </label>
                              <input
                                type="number"
                                className="input"
                                placeholder={`Current ₹${c.payout_amount}`}
                                value={approveAmounts[c.id] ?? ""}
                                onChange={(e) =>
                                  setApproveAmounts((prev) => ({
                                    ...prev,
                                    [c.id]: e.target.value,
                                  }))
                                }
                                style={{ width: "100%", marginBottom: 10 }}
                              />
                              <div style={{ display: "flex", gap: 8 }}>
                                <button
                                  type="button"
                                  className="btn btnSm"
                                  disabled={updatingId === c.id}
                                  onClick={() =>
                                    updateClaim(
                                      c.id,
                                      "approved",
                                      approveAmounts[c.id]
                                        ? parseFloat(approveAmounts[c.id])
                                        : c.payout_amount,
                                    )
                                  }
                                  style={{
                                    background: "#F0FDF4",
                                    color: "#166534",
                                    border: "1px solid #86EFAC",
                                  }}
                                >
                                  Approve with Amount
                                </button>
                                <button
                                  type="button"
                                  className="btn btnSm"
                                  disabled={updatingId === c.id}
                                  onClick={() => updateClaim(c.id, "rejected")}
                                  style={{
                                    background: "#FEF2F2",
                                    color: "#B91C1C",
                                    border: "1px solid #FECACA",
                                  }}
                                >
                                  Reject
                                </button>
                              </div>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
