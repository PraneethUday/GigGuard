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

export default function AdminPage() {
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [search, setSearch] = useState("");
  const [updatingId, setUpdatingId] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/admin/workers")
      .then((r) => r.json())
      .then((data) => {
        if (data.workers) setWorkers(data.workers);
        else setError(data.error || "Failed to load");
      })
      .catch(() => setError("Failed to fetch workers"))
      .finally(() => setLoading(false));
  }, []);

  const filtered = workers.filter(
    (w) =>
      w.name.toLowerCase().includes(search.toLowerCase()) ||
      w.email.toLowerCase().includes(search.toLowerCase()) ||
      w.city.toLowerCase().includes(search.toLowerCase()),
  );

  const stats = {
    total: workers.length,
    verified: workers.filter((w) => w.verification_status === "verified")
      .length,
    pending: workers.filter((w) => w.verification_status === "pending").length,
    active: workers.filter((w) => w.is_active).length,
  };

  const updateApproval = async (
    workerId: string,
    action: "approve" | "reject",
  ) => {
    setUpdatingId(workerId);
    try {
      const res = await fetch(`/api/admin/workers/${workerId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || "Failed to update worker status");
        setNotice("");
        return;
      }

      setWorkers((prev) =>
        prev.map((worker) => {
          if (worker.id !== workerId) return worker;
          return {
            ...worker,
            verification_status: data.worker.verification_status,
            auto_verified: data.worker.auto_verified,
            auto_verified_platforms: data.worker.auto_verified_platforms,
          };
        }),
      );
      setError("");
      setNotice(
        action === "approve"
          ? "Worker approved successfully."
          : "Worker rejected successfully.",
      );
    } catch {
      setError("Failed to update worker status");
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
      </header>

      <main style={{ flex: 1, padding: 32 }}>
        <div style={{ maxWidth: 1100, margin: "0 auto" }}>
          <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4 }}>
            Admin Dashboard
          </h1>
          <p style={{ color: "#64748b", fontSize: 14, marginBottom: 28 }}>
            Manage registered delivery partners
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

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(4, 1fr)",
              gap: 16,
              marginBottom: 28,
            }}
          >
            <StatCard
              label="Total Workers"
              value={stats.total}
              color="#0f172a"
            />
            <StatCard label="Verified" value={stats.verified} color="#16a34a" />
            <StatCard label="Pending" value={stats.pending} color="#d97706" />
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
              }}
            >
              <h3 style={{ fontSize: 15, fontWeight: 600, margin: 0 }}>
                Registered Workers
              </h3>
              <input
                placeholder="Search by name, email, or city..."
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

            {loading ? (
              <div
                style={{ padding: 40, textAlign: "center", color: "#64748b" }}
              >
                Loading workers...
              </div>
            ) : error ? (
              <div
                style={{ padding: 40, textAlign: "center", color: "#dc2626" }}
              >
                {error}
              </div>
            ) : filtered.length === 0 ? (
              <div
                style={{ padding: 40, textAlign: "center", color: "#64748b" }}
              >
                {workers.length === 0
                  ? "No workers registered yet."
                  : "No workers match your search."}
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
                        "Phone",
                        "City",
                        "Platforms",
                        "Tier",
                        "Auto Check",
                        "Status",
                        "Action",
                        "Registered",
                      ].map((h) => (
                        <th
                          key={h}
                          style={{
                            padding: "10px 16px",
                            textAlign: "left",
                            fontWeight: 600,
                            color: "#64748b",
                            borderBottom: "1px solid #e2e8f0",
                            fontSize: 12,
                            textTransform: "uppercase",
                            letterSpacing: "0.04em",
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
                            padding: "12px 16px",
                            fontWeight: 600,
                            color: "#0f172a",
                          }}
                        >
                          {w.name}
                        </td>
                        <td style={{ padding: "12px 16px", color: "#64748b" }}>
                          {w.email}
                        </td>
                        <td style={{ padding: "12px 16px", color: "#64748b" }}>
                          {w.phone}
                        </td>
                        <td style={{ padding: "12px 16px", color: "#64748b" }}>
                          {w.city}
                        </td>
                        <td style={{ padding: "12px 16px" }}>
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
                                  padding: "2px 8px",
                                  borderRadius: 4,
                                  fontSize: 11,
                                  fontWeight: 600,
                                }}
                              >
                                {PLATFORM_NAMES[p] || p}
                              </span>
                            ))}
                          </div>
                        </td>
                        <td style={{ padding: "12px 16px" }}>
                          <span
                            style={{
                              padding: "3px 10px",
                              borderRadius: 6,
                              fontSize: 12,
                              fontWeight: 600,
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
                            {w.tier === "basic"
                              ? "Basic"
                              : w.tier === "pro"
                                ? "Pro"
                                : "Standard"}
                          </span>
                        </td>
                        <td style={{ padding: "12px 16px" }}>
                          <span
                            style={{
                              padding: "3px 10px",
                              borderRadius: 6,
                              fontSize: 12,
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
                        <td style={{ padding: "12px 16px" }}>
                          <span
                            style={{
                              padding: "3px 10px",
                              borderRadius: 6,
                              fontSize: 12,
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
                            {w.verification_status}
                          </span>
                        </td>
                        <td style={{ padding: "12px 16px" }}>
                          <div style={{ display: "flex", gap: 6 }}>
                            <button
                              disabled={
                                updatingId === w.id ||
                                !w.auto_verified ||
                                w.verification_status === "verified"
                              }
                              onClick={() => updateApproval(w.id, "approve")}
                              style={{
                                border: "1px solid #86efac",
                                background: "#f0fdf4",
                                color: "#166534",
                                borderRadius: 6,
                                padding: "4px 8px",
                                fontSize: 11,
                                fontWeight: 600,
                                cursor: "pointer",
                                opacity:
                                  updatingId === w.id ||
                                  !w.auto_verified ||
                                  w.verification_status === "verified"
                                    ? 0.6
                                    : 1,
                              }}
                            >
                              Approve
                            </button>
                            <button
                              disabled={
                                updatingId === w.id ||
                                w.verification_status === "rejected"
                              }
                              onClick={() => updateApproval(w.id, "reject")}
                              style={{
                                border: "1px solid #fecaca",
                                background: "#fef2f2",
                                color: "#b91c1c",
                                borderRadius: 6,
                                padding: "4px 8px",
                                fontSize: 11,
                                fontWeight: 600,
                                cursor: "pointer",
                                opacity:
                                  updatingId === w.id ||
                                  w.verification_status === "rejected"
                                    ? 0.6
                                    : 1,
                              }}
                            >
                              Reject
                            </button>
                          </div>
                        </td>
                        <td
                          style={{
                            padding: "12px 16px",
                            color: "#64748b",
                            fontSize: 12,
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
