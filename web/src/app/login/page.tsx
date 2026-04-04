"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";

// ── Design Tokens ──────────────────────────────────────────────────────────
const C = {
  bg: "#0C0E18",
  surface: "#11131E",
  card: "#1D1F2B",
  elevated: "#272935",
  input: "#323440",
  primary: "#6C63FF",
  primaryDim: "#8B84FF",
  amber: "#FF8C42",
  success: "#22C55E",
  error: "#EF4444",
  white: "#E1E1F2",
  muted: "#C7C4D8",
  faint: "#918FA1",
  border: "rgba(70,69,85,0.6)",
};
const inpStyle: React.CSSProperties = {
  width: "100%",
  height: 46,
  padding: "0 14px",
  fontSize: 14,
  background: C.input,
  border: `1px solid ${C.border}`,
  borderRadius: 10,
  color: C.white,
  outline: "none",
  boxSizing: "border-box",
};
const labelSt: React.CSSProperties = {
  display: "block",
  fontSize: 13,
  fontWeight: 600,
  color: C.muted,
  marginBottom: 6,
};

const featureHighlights = [
  {
    title: "Rapid Claim Filing",
    desc: "Submit incidents with guided steps and evidence upload in under 5 minutes.",
    stat: "5 min",
  },
  {
    title: "Risk-Aware Coverage",
    desc: "Premiums adapt to weather, route disruption, and historical risk signals.",
    stat: "Live",
  },
  {
    title: "Always-On Support",
    desc: "Round-the-clock help for policy updates, renewals, and emergency guidance.",
    stat: "24/7",
  },
];

export default function LoginPage() {
  const router = useRouter();
  const [form, setForm] = useState({ email: "", password: "" });
  const [showPwd, setShowPwd] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleLogin = async () => {
    if (!form.email || !form.password) {
      setError("Please enter your email and password.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: form.email, password: form.password }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || "Login failed.");
        return;
      }
      localStorage.setItem("gg_token", data.token);
      localStorage.setItem("gg_user", JSON.stringify(data.user));
      router.push("/dashboard");
    } catch {
      setError("Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: C.bg,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
        fontFamily: "Inter, sans-serif",
      }}
    >
      {/* bg glow */}
      <div
        style={{
          position: "fixed",
          top: -100,
          left: "50%",
          transform: "translateX(-50%)",
          width: 400,
          height: 400,
          borderRadius: "50%",
          background: C.primary,
          opacity: 0.05,
          pointerEvents: "none",
        }}
      />
      <div
        style={{
          position: "fixed",
          bottom: -220,
          left: -120,
          width: 420,
          height: 420,
          borderRadius: "50%",
          background: C.primaryDim,
          opacity: 0.08,
          filter: "blur(8px)",
          pointerEvents: "none",
        }}
      />

      <div
        className="loginShell"
        style={{ width: "100%", maxWidth: 1080, position: "relative" }}
      >
        <aside
          className="promoPanel"
          style={{
            background:
              "linear-gradient(145deg, #1A1D2C 0%, #141726 42%, #0F1220 100%)",
            border: `1px solid ${C.border}`,
            borderRadius: 26,
            padding: "34px 30px",
            boxShadow: "0 12px 36px rgba(0,0,0,0.35)",
            position: "relative",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 8,
              border: `1px solid ${C.primaryDim}80`,
              borderRadius: 999,
              padding: "6px 12px",
              fontSize: 12,
              fontWeight: 700,
              color: C.primaryDim,
              marginBottom: 18,
              background: "rgba(108,99,255,0.08)",
            }}
          >
            FEATURED PLAN
          </div>
          <h2
            style={{
              fontSize: 34,
              lineHeight: 1.18,
              color: C.white,
              margin: "0 0 12px",
            }}
          >
            Protection built for every shift and route.
          </h2>
          <p
            style={{
              color: C.muted,
              fontSize: 16,
              lineHeight: 1.6,
              margin: "0 0 24px",
            }}
          >
            Join thousands of independent workers using GigGuard to stay
            insured, reduce claim stress, and keep earnings protected through
            every trip.
          </p>

          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 14,
              marginBottom: 20,
            }}
          >
            {featureHighlights.map((item) => (
              <div
                key={item.title}
                style={{
                  display: "grid",
                  gridTemplateColumns: "80px minmax(0,1fr)",
                  gap: 12,
                  alignItems: "center",
                  border: `1px solid ${C.border}`,
                  borderRadius: 14,
                  padding: "12px 14px",
                  background: "rgba(255,255,255,0.02)",
                }}
              >
                <div
                  style={{
                    height: 38,
                    borderRadius: 10,
                    background:
                      "linear-gradient(90deg, #6C63FF 0%, #8B84FF 100%)",
                    color: "#fff",
                    fontWeight: 800,
                    fontSize: 13,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    letterSpacing: 0.4,
                  }}
                >
                  {item.stat}
                </div>
                <div>
                  <p
                    style={{
                      margin: 0,
                      color: C.white,
                      fontSize: 14,
                      fontWeight: 700,
                    }}
                  >
                    {item.title}
                  </p>
                  <p
                    style={{
                      margin: "3px 0 0",
                      color: C.faint,
                      fontSize: 12,
                      lineHeight: 1.45,
                    }}
                  >
                    {item.desc}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </aside>

        <div style={{ width: "100%", maxWidth: 420, position: "relative" }}>
          {/* Logo */}
          <div style={{ textAlign: "center", marginBottom: 36 }}>
            <div
              style={{
                minHeight: 52,
                maxWidth: 320,
                padding: "10px 16px",
                background: C.primary,
                borderRadius: 16,
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                textAlign: "center",
                lineHeight: 1.2,
                fontSize: 13,
                fontWeight: 800,
                letterSpacing: 0.2,
                color: "#fff",
                marginBottom: 16,
                boxShadow: `0 8px 24px ${C.primary}40`,
              }}
            >
              Worker Protection Insurance System
            </div>
            <h1
              style={{
                fontSize: 26,
                fontWeight: 800,
                color: C.white,
                margin: 0,
              }}
            >
              Welcome back
            </h1>
            <p style={{ color: C.faint, fontSize: 14, marginTop: 6 }}>
              Sign in to your Worker Protection Insurance Platform (WPIP)
              account
            </p>
          </div>

          {/* Card */}
          <div
            style={{
              background: C.card,
              borderRadius: 20,
              padding: 28,
              border: `1px solid ${C.border}`,
              boxShadow: `0 8px 32px rgba(0,0,0,0.3)`,
            }}
          >
            {error && (
              <div
                style={{
                  background: "#2E0A0A",
                  border: `1px solid ${C.error}40`,
                  borderRadius: 10,
                  padding: "10px 14px",
                  marginBottom: 16,
                  fontSize: 13,
                  color: C.error,
                  fontWeight: 500,
                }}
              >
                {error}
              </div>
            )}

            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <div>
                <label style={labelSt}>Email Address</label>
                <input
                  type="email"
                  placeholder="you@example.com"
                  value={form.email}
                  onChange={(e) =>
                    setForm((p) => ({ ...p, email: e.target.value }))
                  }
                  onKeyDown={(e) => e.key === "Enter" && handleLogin()}
                  style={inpStyle}
                />
              </div>
              <div>
                <label style={labelSt}>Password</label>
                <div style={{ position: "relative" }}>
                  <input
                    type={showPwd ? "text" : "password"}
                    placeholder="••••••••"
                    value={form.password}
                    onChange={(e) =>
                      setForm((p) => ({ ...p, password: e.target.value }))
                    }
                    onKeyDown={(e) => e.key === "Enter" && handleLogin()}
                    style={{ ...inpStyle, paddingRight: 68 }}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPwd((v) => !v)}
                    style={{
                      position: "absolute",
                      right: 10,
                      top: "50%",
                      transform: "translateY(-50%)",
                      background: "none",
                      border: "none",
                      color: C.primaryDim,
                      cursor: "pointer",
                      fontSize: 12,
                      fontWeight: 700,
                      letterSpacing: 0.2,
                    }}
                  >
                    {showPwd ? "Hide" : "Show"}
                  </button>
                </div>
              </div>

              <div style={{ textAlign: "right" }}>
                <span
                  style={{
                    fontSize: 13,
                    fontWeight: 600,
                    color: C.primaryDim,
                    cursor: "pointer",
                  }}
                >
                  Forgot password?
                </span>
              </div>

              <button
                onClick={handleLogin}
                disabled={loading}
                style={{
                  height: 50,
                  background: loading ? C.elevated : C.primary,
                  color: "#fff",
                  border: "none",
                  borderRadius: 50,
                  fontSize: 15,
                  fontWeight: 700,
                  cursor: loading ? "not-allowed" : "pointer",
                  boxShadow: `0 6px 20px ${C.primary}45`,
                  transition: "all 0.2s",
                }}
              >
                {loading ? "Signing in..." : "Sign In →"}
              </button>
            </div>
          </div>

          <p
            style={{
              textAlign: "center",
              fontSize: 14,
              color: C.faint,
              marginTop: 24,
            }}
          >
            Don&apos;t have an account?{" "}
            <span
              style={{ color: C.primary, fontWeight: 700, cursor: "pointer" }}
              onClick={() => router.push("/register")}
            >
              Register free
            </span>
          </p>
        </div>

        <style jsx>{`
          .loginShell {
            display: grid;
            grid-template-columns: minmax(540px, 1fr) minmax(360px, 420px);
            gap: 34px;
            align-items: center;
            margin-left: -18px;
          }

          .promoPanel {
            width: 100%;
            max-width: 620px;
            justify-self: start;
          }

          .promoPanel::after {
            content: "";
            position: absolute;
            width: 220px;
            height: 220px;
            border-radius: 999px;
            top: -100px;
            right: -80px;
            background: rgba(108, 99, 255, 0.24);
            filter: blur(4px);
          }

          @media (max-width: 960px) {
            .loginShell {
              grid-template-columns: 1fr;
              max-width: 420px;
              margin: 0 auto;
              margin-left: 0;
            }

            .promoPanel {
              display: none;
            }
          }
        `}</style>
      </div>
    </div>
  );
}
