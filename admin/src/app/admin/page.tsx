"use client";
import { useEffect, useState } from "react";

type Claim = {
  id: string;
  worker_id: string;
  worker_name: string;
  claim_type: string;
  description: string;
  incident_date: string;
  claim_amount: number;
  approved_amount: number | null;
  status: string;
  admin_notes: string | null;
  razorpay_payout_id: string | null;
  upi: string | null;
  created_at: string;
};

type Worker = {
  id: string;
  name: string;
  email: string;
  phone: string;
  city: string;
  area: string;
  platforms: string[];
  tier: string;
  verification_status: string;
  delivery_id: string;
  autopay: boolean;
  is_active: boolean;
  created_at: string;
  auto_verified: boolean;
  auto_verified_platforms: string[];
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

const TIER_OPTIONS = ["basic", "standard", "pro"] as const;

const CLAIM_TYPE_LABELS: Record<string, string> = {
  accident: "Accident",
  income_loss: "Income Loss",
  weather_disruption: "Weather Disruption",
  other: "Other",
};

export default function AdminPage() {
  const [tab, setTab] = useState<"workers" | "claims">("workers");

  // Workers state
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [search, setSearch] = useState("");
  const [updatingId, setUpdatingId] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>("all");

  // Claims state
  const [claims, setClaims] = useState<Claim[]>([]);
  const [claimsLoading, setClaimsLoading] = useState(false);
  const [claimStatusFilter, setClaimStatusFilter] = useState("all");
  const [updatingClaimId, setUpdatingClaimId] = useState<string | null>(null);
  const [claimNotice, setClaimNotice] = useState("");
  const [expandedClaim, setExpandedClaim] = useState<string | null>(null);
  const [approveAmounts, setApproveAmounts] = useState<Record<string, string>>({});
  const [adminNotes, setAdminNotes] = useState<Record<string, string>>({});

  useEffect(() => {
    fetchWorkers();
    fetchClaims();
  }, []);

  const fetchClaims = () => {
    setClaimsLoading(true);
    fetch("/api/claims/all")
      .then(r => r.json())
      .then(data => { if (data.claims) setClaims(data.claims); })
      .catch(() => {})
      .finally(() => setClaimsLoading(false));
  };

  const updateClaim = async (
    claimId: string,
    status: string,
    approvedAmount?: number,
    notes?: string,
  ) => {
    setUpdatingClaimId(claimId);
    setClaimNotice("");
    try {
      const res = await fetch("/api/claims/update", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ claimId, status, approved_amount: approvedAmount ?? null, admin_notes: notes ?? null }),
      });
      const data = await res.json();
      if (!res.ok || data.error) { setError(data.error || "Failed to update claim."); return; }
      setClaims(prev => prev.map(c => c.id === claimId ? { ...c, ...data.claim } : c));
      setClaimNotice(`Claim ${status === "paid" ? "marked as paid" : status}.`);
      setExpandedClaim(null);
    } catch { setError("Failed to update claim."); }
    finally { setUpdatingClaimId(null); }
  };

  const fetchWorkers = () => {
    setLoading(true);
    fetch("/api/admin/workers")
      .then((r) => r.json())
      .then((data) => {
        if (data.workers) setWorkers(data.workers);
        else setError(data.error || "Failed to load");
      })
      .catch(() => setError("Failed to fetch workers"))
      .finally(() => setLoading(false));
  };

  const filtered = workers.filter((w) => {
    const matchesSearch =
      w.name.toLowerCase().includes(search.toLowerCase()) ||
      w.email.toLowerCase().includes(search.toLowerCase()) ||
      w.city.toLowerCase().includes(search.toLowerCase()) ||
      w.delivery_id.toLowerCase().includes(search.toLowerCase());
    const matchesStatus =
      statusFilter === "all" || w.verification_status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const stats = {
    total: workers.length,
    verified: workers.filter((w) => w.verification_status === "verified").length,
    pending: workers.filter((w) => w.verification_status === "pending").length,
    rejected: workers.filter((w) => w.verification_status === "rejected").length,
    active: workers.filter((w) => w.is_active).length,
  };

  const updateWorker = async (
    workerId: string,
    action?: "approve" | "reject" | "pending",
    tier?: string,
  ) => {
    setUpdatingId(workerId);
    try {
      const body: Record<string, string> = {};
      if (action) body.action = action;
      if (tier) body.tier = tier;

      const res = await fetch(`/api/admin/workers/${workerId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || "Failed to update worker");
        setNotice("");
        return;
      }

      setWorkers((prev) =>
        prev.map((worker) => {
          if (worker.id !== workerId) return worker;
          return {
            ...worker,
            verification_status:
              data.worker.verification_status ?? worker.verification_status,
            tier: data.worker.tier ?? worker.tier,
            auto_verified:
              data.worker.auto_verified ?? worker.auto_verified,
            auto_verified_platforms:
              data.worker.auto_verified_platforms ?? worker.auto_verified_platforms,
          };
        }),
      );
      setError("");
      if (action === "approve") setNotice("Worker approved successfully.");
      else if (action === "reject") setNotice("Worker rejected successfully.");
      else if (action === "pending") setNotice("Worker set back to pending.");
      else if (tier) setNotice(`Worker tier changed to ${tier}.`);
    } catch {
      setError("Failed to update worker");
      setNotice("");
    } finally {
      setUpdatingId(null);
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#11131E",
        display: "flex",
        flexDirection: "column",
        fontFamily: '"Inter", -apple-system, sans-serif',
        color: "#E1E1F2",
      }}
    >
      <header
        style={{
          background: "#0C0E18",
          borderBottom: "1px solid rgba(70,69,85,0.6)",
          padding: "0 24px",
          height: 60,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div
            style={{
              width: 36,
              height: 36,
              background: "#4f46e5",
              borderRadius: 10,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 14,
              fontWeight: 800,
              color: "#fff",
            }}
          >
            GG
          </div>
          <span style={{ fontSize: 17, fontWeight: 700, color: "#fff" }}>
            GigGuard
          </span>
          <span
            style={{
              background: "#4f46e5",
              color: "#fff",
              fontSize: 11,
              fontWeight: 700,
              padding: "3px 10px",
              borderRadius: 6,
              marginLeft: 4,
            }}
          >
            Admin
          </span>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <a
            href="/disruptions"
            style={{
              display: "inline-flex",
              alignItems: "center",
              background: "#dc2626",
              color: "#fff",
              border: "none",
              borderRadius: 6,
              padding: "6px 14px",
              fontSize: 12,
              fontWeight: 600,
              textDecoration: "none",
            }}
          >
            🚨 Disruptions
          </a>
          <a
            href="/control-center"
            style={{
              display: "inline-flex",
              alignItems: "center",
              background: "#059669",
              color: "#fff",
              border: "none",
              borderRadius: 6,
              padding: "6px 14px",
              fontSize: 12,
              fontWeight: 600,
              textDecoration: "none",
            }}
          >
            Control Center
          </a>
          <button
            type="button"
            onClick={fetchWorkers}
            style={{
              background: "#1e293b",
              color: "#94a3b8",
              border: "1px solid #334155",
              borderRadius: 6,
              padding: "6px 14px",
              fontSize: 12,
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            Refresh
          </button>
        </div>
      </header>

      <main style={{ flex: 1, padding: 32 }}>
        <div style={{ maxWidth: 1200, margin: "0 auto" }}>
          <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4, color: "#E1E1F2" }}>
            Admin Dashboard
          </h1>
          <p style={{ color: "#918FA1", fontSize: 14, marginBottom: 20 }}>
            Manage registered delivery partners — approve, reject, change status and tier at any time
          </p>

          {/* Tab nav */}
          <div style={{ display: "flex", gap: 4, marginBottom: 24, background: "#1D1F2B", borderRadius: 10, padding: 4, width: "fit-content", border: "1px solid rgba(70,69,85,0.6)" }}>
            {(["workers", "claims"] as const).map(t => (
              <button
                key={t}
                type="button"
                onClick={() => setTab(t)}
                style={{
                  padding: "7px 20px", borderRadius: 7, border: "none", cursor: "pointer",
                  fontSize: 13, fontWeight: 600,
                  background: tab === t ? "#6C63FF" : "transparent",
                  color: tab === t ? "#fff" : "#918FA1",
                  boxShadow: tab === t ? "0 2px 8px rgba(108,99,255,0.4)" : "none",
                  transition: "all 0.15s",
                }}
              >
                {t === "workers" ? "Workers" : `Claims${claims.length ? ` (${claims.filter(c => c.status === "pending" || c.status === "under_review").length})` : ""}`}
              </button>
            ))}
          </div>

          {(notice || claimNotice) && (
            <div style={{ marginBottom: 12, background: "#0A2E18", border: "1px solid rgba(34,197,94,0.3)", color: "#22C55E", fontSize: 13, fontWeight: 600, borderRadius: 8, padding: "8px 12px" }}>
              {notice || claimNotice}
            </div>
          )}

          {error && (
            <div
              style={{
                marginBottom: 12,
                background: "#2E0A0A",
                border: "1px solid rgba(239,68,68,0.3)",
                color: "#EF4444",
                fontSize: 13,
                fontWeight: 600,
                borderRadius: 8,
                padding: "8px 12px",
              }}
            >
              {error}
            </div>
          )}

          {tab === "workers" && <>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(5, 1fr)",
              gap: 16,
              marginBottom: 28,
            }}
          >
            <StatCard label="Total Workers" value={stats.total} color="#0f172a" />
            <StatCard label="Verified" value={stats.verified} color="#16a34a" />
            <StatCard label="Pending" value={stats.pending} color="#d97706" />
            <StatCard label="Rejected" value={stats.rejected} color="#dc2626" />
            <StatCard label="Active" value={stats.active} color="#4f46e5" />
          </div>

          <div
            style={{
              background: "#1D1F2B",
              borderRadius: 12,
              border: "1px solid rgba(70,69,85,0.6)",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                padding: "16px 20px",
                borderBottom: "1px solid rgba(70,69,85,0.6)",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                gap: 12,
              }}
            >
              <h3 style={{ fontSize: 15, fontWeight: 600, margin: 0, color: "#E1E1F2" }}>
                Registered Workers
              </h3>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  style={{
                    height: 36, padding: "0 12px", fontSize: 13,
                    border: "1px solid rgba(70,69,85,0.6)", borderRadius: 8,
                    outline: "none", cursor: "pointer",
                    background: "#323440", color: "#E1E1F2",
                  }}
                >
                  <option value="all">All statuses</option>
                  <option value="pending">Pending</option>
                  <option value="verified">Verified</option>
                  <option value="rejected">Rejected</option>
                </select>
                <input
                  placeholder="Search name, email, city, or ID..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  style={{
                    width: 280, height: 36, padding: "0 12px", fontSize: 13,
                    border: "1px solid rgba(70,69,85,0.6)", borderRadius: 8,
                    outline: "none", background: "#323440", color: "#E1E1F2",
                  }}
                />
              </div>
            </div>

            {loading ? (
              <div style={{ padding: 40, textAlign: "center", color: "#918FA1" }}>
                Loading workers...
              </div>
            ) : filtered.length === 0 ? (
              <div style={{ padding: 40, textAlign: "center", color: "#918FA1" }}>
                {workers.length === 0
                  ? "No workers registered yet."
                  : "No workers match your filters."}
              </div>
            ) : (
              <div style={{ overflowX: "auto" }}>
                <table
                  style={{
                    width: "100%",
                    borderCollapse: "collapse",
                    fontSize: 13,
                  }}
                >
                  <thead>
                    <tr style={{ background: "#272935" }}>
                      {[
                        "Name",
                        "Email",
                        "City",
                        "Delivery ID",
                        "Platforms",
                        "Tier",
                        "Auto Check",
                        "Status",
                        "Actions",
                        "Registered",
                      ].map((h) => (
                        <th
                          key={h}
                          style={{
                            padding: "10px 14px",
                            textAlign: "left",
                            fontWeight: 700,
                            color: "#918FA1",
                            borderBottom: "1px solid rgba(70,69,85,0.6)",
                            fontSize: 11,
                            textTransform: "uppercase",
                            letterSpacing: "0.05em",
                            whiteSpace: "nowrap",
                          }}
                        >
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map((w) => (
                      <tr
                        key={w.id}
                        style={{ borderBottom: "1px solid rgba(70,69,85,0.4)" }}
                      >
                        <td
                          style={{
                            padding: "12px 14px",
                            fontWeight: 600,
                            color: "#E1E1F2",
                          }}
                        >
                          {w.name}
                        </td>
                        <td style={{ padding: "12px 14px", color: "#918FA1", fontSize: 12 }}>
                          {w.email}
                        </td>
                        <td style={{ padding: "12px 14px", color: "#C7C4D8" }}>
                          {w.city}
                        </td>
                        <td style={{ padding: "12px 14px", color: "#918FA1", fontSize: 11, fontFamily: "monospace" }}>
                          {w.delivery_id?.slice(0, 12)}...
                        </td>
                        <td style={{ padding: "12px 14px" }}>
                          <div
                            style={{
                              display: "flex",
                              gap: 4,
                              flexWrap: "wrap",
                            }}
                          >
                            {w.platforms.map((p) => (
                              <span
                                key={p}
                                style={{
                                  background: "#1D1B45",
                                  color: "#8B84FF",
                                  padding: "2px 8px",
                                  borderRadius: 4,
                                  fontSize: 10,
                                  fontWeight: 700,
                                }}
                              >
                                {PLATFORM_NAMES[p] || p}
                              </span>
                            ))}
                          </div>
                        </td>
                        <td style={{ padding: "12px 14px" }}>
                          <select
                            value={w.tier}
                            onChange={(e) =>
                              updateWorker(w.id, undefined, e.target.value)
                            }
                            disabled={updatingId === w.id}
                            style={{
                              padding: "4px 8px",
                              borderRadius: 6,
                              fontSize: 12,
                              fontWeight: 600,
                              border: "1px solid rgba(70,69,85,0.6)",
                              cursor: "pointer",
                              background: "#323440",
                              color:
                                w.tier === "pro"
                                  ? "#c084fc"
                                  : w.tier === "standard"
                                    ? "#6C63FF"
                                    : "#22C55E",
                            }}
                          >
                            {TIER_OPTIONS.map((t) => (
                              <option key={t} value={t}>
                                {t.charAt(0).toUpperCase() + t.slice(1)}
                              </option>
                            ))}
                          </select>
                        </td>
                        <td style={{ padding: "12px 14px" }}>
                          <span
                            style={{
                              padding: "3px 8px",
                              borderRadius: 6,
                              fontSize: 11,
                              fontWeight: 600,
                              background: w.auto_verified
                                ? "#f0fdf4"
                                : "#fef2f2",
                              color: w.auto_verified ? "#16a34a" : "#dc2626",
                            }}
                            title={
                              w.auto_verified_platforms?.length
                                ? `Matched: ${w.auto_verified_platforms.join(", ")}`
                                : "No platform match"
                            }
                          >
                            {w.auto_verified ? "Matched" : "Not matched"}
                          </span>
                        </td>
                        <td style={{ padding: "12px 14px" }}>
                          <span
                            style={{
                              padding: "3px 8px",
                              borderRadius: 6,
                              fontSize: 11,
                              fontWeight: 600,
                              background:
                                w.verification_status === "verified"
                                  ? "#f0fdf4"
                                  : w.verification_status === "rejected"
                                    ? "#fef2f2"
                                    : "#fffbeb",
                              color:
                                w.verification_status === "verified"
                                  ? "#16a34a"
                                  : w.verification_status === "rejected"
                                    ? "#dc2626"
                                    : "#d97706",
                            }}
                          >
                            {w.verification_status.charAt(0).toUpperCase() +
                              w.verification_status.slice(1)}
                          </span>
                        </td>
                        <td style={{ padding: "12px 14px" }}>
                          <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                            {w.verification_status !== "verified" && (
                              <ActionBtn
                                label="Approve"
                                bg="#f0fdf4"
                                border="#86efac"
                                color="#166534"
                                disabled={updatingId === w.id}
                                onClick={() => updateWorker(w.id, "approve")}
                              />
                            )}
                            {w.verification_status !== "rejected" && (
                              <ActionBtn
                                label="Reject"
                                bg="#fef2f2"
                                border="#fecaca"
                                color="#b91c1c"
                                disabled={updatingId === w.id}
                                onClick={() => updateWorker(w.id, "reject")}
                              />
                            )}
                            {w.verification_status !== "pending" && (
                              <ActionBtn
                                label="Pending"
                                bg="#fffbeb"
                                border="#fde68a"
                                color="#92400e"
                                disabled={updatingId === w.id}
                                onClick={() => updateWorker(w.id, "pending")}
                              />
                            )}
                          </div>
                        </td>
                        <td
                          style={{
                            padding: "12px 14px",
                            color: "#918FA1",
                            fontSize: 12,
                            whiteSpace: "nowrap",
                          }}
                        >
                          {new Date(w.created_at).toLocaleDateString("en-IN", {
                            day: "numeric",
                            month: "short",
                            year: "numeric",
                          })}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
          </>}

          {tab === "claims" && (
            <div>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
                <h3 style={{ fontSize: 15, fontWeight: 600, margin: 0, color: "#E1E1F2" }}>Insurance Claims</h3>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <select
                    value={claimStatusFilter}
                    onChange={e => setClaimStatusFilter(e.target.value)}
                    style={{ height: 36, padding: "0 12px", fontSize: 13, border: "1px solid rgba(70,69,85,0.6)", borderRadius: 8, outline: "none", cursor: "pointer", background: "#323440", color: "#E1E1F2" }}
                  >
                    <option value="all">All statuses</option>
                    <option value="pending">Pending</option>
                    <option value="under_review">Under Review</option>
                    <option value="approved">Approved</option>
                    <option value="paid">Paid</option>
                    <option value="rejected">Rejected</option>
                  </select>
                  <button type="button" onClick={fetchClaims} style={{ height: 36, padding: "0 14px", fontSize: 12, fontWeight: 600, background: "#1D1F2B", color: "#918FA1", border: "1px solid rgba(70,69,85,0.6)", borderRadius: 6, cursor: "pointer" }}>Refresh</button>
                </div>
              </div>

              {claimsLoading ? (
                <div style={{ padding: 40, textAlign: "center", color: "#918FA1" }}>Loading claims...</div>
              ) : (
                <div style={{ background: "#1D1F2B", borderRadius: 12, border: "1px solid rgba(70,69,85,0.6)", overflow: "hidden" }}>
                  {claims.filter(c => claimStatusFilter === "all" || c.status === claimStatusFilter).length === 0 ? (
                    <div style={{ padding: 40, textAlign: "center", color: "#918FA1" }}>No claims found.</div>
                  ) : (
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                      <thead>
                        <tr style={{ background: "#272935" }}>
                          {["Worker", "Type", "Date", "Claimed", "Approved", "Status", "Actions"].map(h => (
                            <th key={h} style={{ padding: "10px 14px", textAlign: "left", fontWeight: 700, color: "#918FA1", borderBottom: "1px solid rgba(70,69,85,0.6)", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em", whiteSpace: "nowrap" }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {claims
                          .filter(c => claimStatusFilter === "all" || c.status === claimStatusFilter)
                          .map(c => (
                          <>
                            <tr key={c.id} style={{ borderBottom: expandedClaim === c.id ? "none" : "1px solid rgba(70,69,85,0.4)" }}>
                              <td style={{ padding: "12px 14px", fontWeight: 600, color: "#E1E1F2" }}>{c.worker_name}</td>
                              <td style={{ padding: "12px 14px", color: "#C7C4D8" }}>{CLAIM_TYPE_LABELS[c.claim_type] || c.claim_type}</td>
                              <td style={{ padding: "12px 14px", color: "#918FA1" }}>{c.incident_date}</td>
                              <td style={{ padding: "12px 14px", fontWeight: 600, color: "#E1E1F2" }}>₹{c.claim_amount}</td>
                              <td style={{ padding: "12px 14px", color: c.approved_amount ? "#22C55E" : "#918FA1", fontWeight: c.approved_amount ? 700 : 400 }}>
                                {c.approved_amount ? `₹${c.approved_amount}` : "—"}
                              </td>
                              <td style={{ padding: "12px 14px" }}>
                                <ClaimBadge status={c.status} />
                              </td>
                              <td style={{ padding: "12px 14px" }}>
                                <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                                  <button type="button" onClick={() => setExpandedClaim(expandedClaim === c.id ? null : c.id)} style={{ border: "1px solid #d1d5db", background: "#f8fafc", color: "#374151", borderRadius: 6, padding: "4px 8px", fontSize: 11, fontWeight: 600, cursor: "pointer" }}>
                                    {expandedClaim === c.id ? "Close" : "Review"}
                                  </button>
                                  {c.status === "approved" && (
                                    <ActionBtn label="Mark Paid" bg="#f5f3ff" border="#ddd6fe" color="#7c3aed" disabled={updatingClaimId === c.id} onClick={() => updateClaim(c.id, "paid")} />
                                  )}
                                  {c.status === "pending" && (
                                    <ActionBtn label="Review" bg="#eff6ff" border="#bfdbfe" color="#2563eb" disabled={updatingClaimId === c.id} onClick={() => updateClaim(c.id, "under_review")} />
                                  )}
                                </div>
                              </td>
                            </tr>
                            {expandedClaim === c.id && (
                              <tr key={`${c.id}-expand`} style={{ borderBottom: "1px solid rgba(70,69,85,0.4)" }}>
                                <td colSpan={7} style={{ padding: "0 14px 16px", background: "#272935" }}>
                                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, paddingTop: 12 }}>
                                    <div>
                                      <div style={{ fontSize: 11, fontWeight: 700, color: "#918FA1", textTransform: "uppercase", marginBottom: 6, letterSpacing: "0.05em" }}>Description</div>
                                      <p style={{ fontSize: 13, color: "#C7C4D8", margin: 0, lineHeight: 1.5 }}>{c.description}</p>
                                      {c.upi && <p style={{ fontSize: 12, color: "#918FA1", marginTop: 8 }}>UPI: <strong style={{ color: "#E1E1F2" }}>{c.upi}</strong></p>}
                                      {c.razorpay_payout_id && <p style={{ fontSize: 11, fontFamily: "monospace", color: "#918FA1", marginTop: 4 }}>Payout ID: {c.razorpay_payout_id}</p>}
                                    </div>
                                    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                                      <div>
                                        <label style={{ display: "block", fontSize: 11, fontWeight: 600, color: "#918FA1", marginBottom: 4 }}>Approved amount (₹)</label>
                                        <input
                                          type="number"
                                          placeholder={`Max ₹${c.claim_amount}`}
                                          value={approveAmounts[c.id] ?? ""}
                                          onChange={e => setApproveAmounts(prev => ({ ...prev, [c.id]: e.target.value }))}
                                          style={{ width: "100%", height: 36, padding: "0 10px", fontSize: 13, border: "1px solid rgba(70,69,85,0.6)", borderRadius: 8, outline: "none", boxSizing: "border-box", background: "#323440", color: "#E1E1F2" }}
                                        />
                                      </div>
                                      <div>
                                        <label style={{ display: "block", fontSize: 11, fontWeight: 600, color: "#918FA1", marginBottom: 4 }}>Admin notes</label>
                                        <input
                                          type="text"
                                          placeholder="Optional note for worker"
                                          value={adminNotes[c.id] ?? ""}
                                          onChange={e => setAdminNotes(prev => ({ ...prev, [c.id]: e.target.value }))}
                                          style={{ width: "100%", height: 36, padding: "0 10px", fontSize: 13, border: "1px solid rgba(70,69,85,0.6)", borderRadius: 8, outline: "none", boxSizing: "border-box", background: "#323440", color: "#E1E1F2" }}
                                        />
                                      </div>
                                      <div style={{ display: "flex", gap: 6 }}>
                                        <ActionBtn label="Approve" bg="#f0fdf4" border="#86efac" color="#166534" disabled={updatingClaimId === c.id} onClick={() => updateClaim(c.id, "approved", approveAmounts[c.id] ? parseFloat(approveAmounts[c.id]) : c.claim_amount, adminNotes[c.id])} />
                                        <ActionBtn label="Reject" bg="#fef2f2" border="#fecaca" color="#b91c1c" disabled={updatingClaimId === c.id} onClick={() => updateClaim(c.id, "rejected", undefined, adminNotes[c.id])} />
                                      </div>
                                    </div>
                                  </div>
                                </td>
                              </tr>
                            )}
                          </>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

function ClaimBadge({ status }: { status: string }) {
  const styles: Record<string, React.CSSProperties> = {
    pending:      { background: "#2A1A0A", color: "#FF8C42", border: "1px solid rgba(255,140,66,0.3)" },
    under_review: { background: "#0a1a3a", color: "#3B82F6", border: "1px solid rgba(59,130,246,0.3)" },
    approved:     { background: "#0A2E18", color: "#22C55E", border: "1px solid rgba(34,197,94,0.3)" },
    rejected:     { background: "#2E0A0A", color: "#EF4444", border: "1px solid rgba(239,68,68,0.3)" },
    paid:         { background: "#1D1B45", color: "#6C63FF", border: "1px solid rgba(108,99,255,0.3)" },
  };
  const labels: Record<string, string> = {
    pending: "Pending", under_review: "Under Review", approved: "Approved", rejected: "Rejected", paid: "Paid",
  };
  const s = styles[status] || styles.pending;
  return (
    <span style={{ ...s, padding: "3px 8px", borderRadius: 6, fontSize: 11, fontWeight: 600 }}>
      {labels[status] || status}
    </span>
  );
}

function ActionBtn({
  label,
  bg,
  border,
  color,
  disabled,
  onClick,
}: {
  label: string;
  bg: string;
  border: string;
  color: string;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      style={{
        border: `1px solid ${border}`,
        background: bg,
        color,
        borderRadius: 6,
        padding: "4px 8px",
        fontSize: 11,
        fontWeight: 600,
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.6 : 1,
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </button>
  );
}

function StatCard({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div
      style={{
        background: "#fff",
        borderRadius: 12,
        border: "1px solid #e2e8f0",
        padding: 20,
      }}
    >
      <div
        style={{
          fontSize: 12,
          color: "#64748b",
          fontWeight: 500,
          marginBottom: 6,
          textTransform: "uppercase",
          letterSpacing: "0.05em",
        }}
      >
        {label}
      </div>
      <div style={{ fontSize: 28, fontWeight: 700, color }}>{value}</div>
    </div>
  );
}
