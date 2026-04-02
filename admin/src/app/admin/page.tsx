"use client";
import { useEffect, useState } from "react";

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

export default function AdminPage() {
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [search, setSearch] = useState("");
  const [updatingId, setUpdatingId] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>("all");

  useEffect(() => {
    fetchWorkers();
  }, []);

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
        background: "#f8fafc",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <header
        style={{
          background: "#0f172a",
          borderBottom: "1px solid #1e293b",
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
          <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4 }}>
            Admin Dashboard
          </h1>
          <p style={{ color: "#64748b", fontSize: 14, marginBottom: 28 }}>
            Manage registered delivery partners — approve, reject, change status and tier at any time
          </p>

          {notice && (
            <div
              style={{
                marginBottom: 12,
                background: "#f0fdf4",
                border: "1px solid #86efac",
                color: "#166534",
                fontSize: 13,
                fontWeight: 600,
                borderRadius: 8,
                padding: "8px 12px",
              }}
            >
              {notice}
            </div>
          )}
          {error && (
            <div
              style={{
                marginBottom: 12,
                background: "#fef2f2",
                border: "1px solid #fecaca",
                color: "#dc2626",
                fontSize: 13,
                fontWeight: 600,
                borderRadius: 8,
                padding: "8px 12px",
              }}
            >
              {error}
            </div>
          )}

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
              background: "#fff",
              borderRadius: 12,
              border: "1px solid #e2e8f0",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                padding: "16px 20px",
                borderBottom: "1px solid #e2e8f0",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                gap: 12,
              }}
            >
              <h3 style={{ fontSize: 15, fontWeight: 600, margin: 0 }}>
                Registered Workers
              </h3>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  style={{
                    height: 36,
                    padding: "0 12px",
                    fontSize: 13,
                    border: "1px solid #d1d5db",
                    borderRadius: 8,
                    outline: "none",
                    cursor: "pointer",
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
                    width: 280,
                    height: 36,
                    padding: "0 12px",
                    fontSize: 13,
                    border: "1px solid #d1d5db",
                    borderRadius: 8,
                    outline: "none",
                  }}
                />
              </div>
            </div>

            {loading ? (
              <div style={{ padding: 40, textAlign: "center", color: "#64748b" }}>
                Loading workers...
              </div>
            ) : filtered.length === 0 ? (
              <div style={{ padding: 40, textAlign: "center", color: "#64748b" }}>
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
                    <tr style={{ background: "#f8fafc" }}>
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
                            fontWeight: 600,
                            color: "#64748b",
                            borderBottom: "1px solid #e2e8f0",
                            fontSize: 11,
                            textTransform: "uppercase",
                            letterSpacing: "0.04em",
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
                        style={{ borderBottom: "1px solid #f1f5f9" }}
                      >
                        <td
                          style={{
                            padding: "12px 14px",
                            fontWeight: 600,
                            color: "#0f172a",
                          }}
                        >
                          {w.name}
                        </td>
                        <td style={{ padding: "12px 14px", color: "#64748b", fontSize: 12 }}>
                          {w.email}
                        </td>
                        <td style={{ padding: "12px 14px", color: "#64748b" }}>
                          {w.city}
                        </td>
                        <td style={{ padding: "12px 14px", color: "#64748b", fontSize: 11, fontFamily: "monospace" }}>
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
                                  background: "#eef2ff",
                                  color: "#4f46e5",
                                  padding: "2px 6px",
                                  borderRadius: 4,
                                  fontSize: 10,
                                  fontWeight: 600,
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
                              border: "1px solid #d1d5db",
                              cursor: "pointer",
                              background:
                                w.tier === "pro"
                                  ? "#faf5ff"
                                  : w.tier === "standard"
                                    ? "#eef2ff"
                                    : "#f0fdf4",
                              color:
                                w.tier === "pro"
                                  ? "#7c3aed"
                                  : w.tier === "standard"
                                    ? "#4f46e5"
                                    : "#16a34a",
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
                            color: "#64748b",
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
        </div>
      </main>
    </div>
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
