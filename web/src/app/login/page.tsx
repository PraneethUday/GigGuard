"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";

// ── Design Tokens ──────────────────────────────────────────────────────────
const C = {
  bg:        "#0C0E18",
  surface:   "#11131E",
  card:      "#1D1F2B",
  elevated:  "#272935",
  input:     "#323440",
  primary:   "#6C63FF",
  primaryDim:"#8B84FF",
  amber:     "#FF8C42",
  success:   "#22C55E",
  error:     "#EF4444",
  white:     "#E1E1F2",
  muted:     "#C7C4D8",
  faint:     "#918FA1",
  border:    "rgba(70,69,85,0.6)",
};
const inpStyle: React.CSSProperties = {
  width:"100%", height:46, padding:"0 14px", fontSize:14,
  background:C.input, border:`1px solid ${C.border}`, borderRadius:10,
  color:C.white, outline:"none", boxSizing:"border-box",
};
const labelSt: React.CSSProperties = {
  display:"block", fontSize:13, fontWeight:600, color:C.muted, marginBottom:6,
};

export default function LoginPage() {
  const router = useRouter();
  const [form, setForm] = useState({ email: "", password: "" });
  const [showPwd, setShowPwd] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleLogin = async () => {
    if (!form.email || !form.password) { setError("Please enter your email and password."); return; }
    setLoading(true); setError("");
    try {
      const res = await fetch("/api/auth/login", {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({ email:form.email, password:form.password }),
      });
      const data = await res.json();
      if (!res.ok) { setError(data.error || "Login failed."); return; }
      localStorage.setItem("gg_token", data.token);
      localStorage.setItem("gg_user", JSON.stringify(data.user));
      router.push("/dashboard");
    } catch { setError("Something went wrong. Please try again."); }
    finally { setLoading(false); }
  };

  return (
    <div style={{ minHeight:"100vh", background:C.bg, display:"flex", alignItems:"center", justifyContent:"center", padding:24, fontFamily:"Inter, sans-serif" }}>
      {/* bg glow */}
      <div style={{ position:"fixed", top:-100, left:"50%", transform:"translateX(-50%)", width:400, height:400, borderRadius:"50%", background:C.primary, opacity:0.05, pointerEvents:"none" }} />

      <div style={{ width:"100%", maxWidth:420, position:"relative" }}>
        {/* Logo */}
        <div style={{ textAlign:"center", marginBottom:36 }}>
          <div style={{ width:52, height:52, background:C.primary, borderRadius:16, display:"inline-flex", alignItems:"center", justifyContent:"center", fontSize:20, marginBottom:16, boxShadow:`0 8px 24px ${C.primary}40` }}>
            🛡️
          </div>
          <h1 style={{ fontSize:26, fontWeight:800, color:C.white, margin:0 }}>Welcome back</h1>
          <p style={{ color:C.faint, fontSize:14, marginTop:6 }}>Sign in to your GigGuard account</p>
        </div>

        {/* Card */}
        <div style={{ background:C.card, borderRadius:20, padding:28, border:`1px solid ${C.border}`, boxShadow:`0 8px 32px rgba(0,0,0,0.3)` }}>
          {error && (
            <div style={{ background:"#2E0A0A", border:`1px solid ${C.error}40`, borderRadius:10, padding:"10px 14px", marginBottom:16, fontSize:13, color:C.error, fontWeight:500 }}>{error}</div>
          )}

          <div style={{ display:"flex", flexDirection:"column", gap:16 }}>
            <div>
              <label style={labelSt}>Email Address</label>
              <input type="email" placeholder="you@example.com" value={form.email}
                onChange={e => setForm(p => ({ ...p, email: e.target.value }))}
                onKeyDown={e => e.key === "Enter" && handleLogin()}
                style={inpStyle} />
            </div>
            <div>
              <label style={labelSt}>Password</label>
              <div style={{ position:"relative" }}>
                <input type={showPwd ? "text" : "password"} placeholder="••••••••" value={form.password}
                  onChange={e => setForm(p => ({ ...p, password: e.target.value }))}
                  onKeyDown={e => e.key === "Enter" && handleLogin()}
                  style={{ ...inpStyle, paddingRight:44 }} />
                <button type="button" onClick={() => setShowPwd(v => !v)}
                  style={{ position:"absolute", right:12, top:"50%", transform:"translateY(-50%)", background:"none", border:"none", color:C.faint, cursor:"pointer", fontSize:16 }}>
                  {showPwd ? "🙈" : "👁"}
                </button>
              </div>
            </div>

            <div style={{ textAlign:"right" }}>
              <span style={{ fontSize:13, fontWeight:600, color:C.primaryDim, cursor:"pointer" }}>Forgot password?</span>
            </div>

            <button onClick={handleLogin} disabled={loading}
              style={{ height:50, background:loading ? C.elevated : C.primary, color:"#fff", border:"none", borderRadius:50, fontSize:15, fontWeight:700, cursor:loading ? "not-allowed" : "pointer", boxShadow:`0 6px 20px ${C.primary}45`, transition:"all 0.2s" }}>
              {loading ? "Signing in..." : "Sign In →"}
            </button>
          </div>
        </div>

        <p style={{ textAlign:"center", fontSize:14, color:C.faint, marginTop:24 }}>
          Don&apos;t have an account?{" "}
          <span style={{ color:C.primary, fontWeight:700, cursor:"pointer" }} onClick={() => router.push("/register")}>Register free</span>
        </p>
      </div>
    </div>
  );
}
