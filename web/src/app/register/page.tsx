"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import styles from "./page.module.css";
import {
  SwiggyIcon,
  ZomatoIcon,
  BlinkitIcon,
  ZeptoIcon,
  MeeshoIcon,
  DunzoIcon,
  PorterIcon,
} from "../dashboard/PlatformIcons";

const PLATFORMS = [
  { id: "zomato",  name: "Zomato",       logo: <ZomatoIcon height={22} />,  symClass: "symZomato"  },
  { id: "swiggy",  name: "Swiggy",       logo: <SwiggyIcon height={22} />,  symClass: "symSwiggy"  },
  { id: "amazon",  name: "Amazon Flex",  logo: null,                         symClass: "symAmazon"  },
  { id: "blinkit", name: "Blinkit",      logo: <BlinkitIcon height={22} />, symClass: "symBlinkit" },
  { id: "zepto",   name: "Zepto",        logo: <ZeptoIcon height={18} />,   symClass: "symZepto"   },
  { id: "meesho",  name: "Meesho",       logo: <MeeshoIcon height={16} />,  symClass: "symMeesho"  },
  { id: "porter",  name: "Porter",       logo: <PorterIcon size={22} />,    symClass: "symPorter"  },
  { id: "dunzo",   name: "Dunzo",        logo: <DunzoIcon height={16} />,   symClass: "symDunzo"   },
];

const CITIES = [
  "Bengaluru", "Chennai", "Delhi", "Hyderabad", "Mumbai", "Pune",
];

const STEPS = ["Platform", "Personal Info", "Documents", "Coverage"];

const platformLabel = (id: string) => {
  const p = PLATFORMS.find((x) => x.id === id);
  return p ? p.name : id;
};

type Form = {
  platforms: string[];
  name: string;
  age: string;
  phone: string;
  email: string;
  password: string;
  confirmPassword: string;
  pan: string;
  aadhaar: string;
  city: string;
  area: string;
  deliveryId: string;
  upi: string;
  bank: string;
  consent: boolean;
  autopay: boolean;
  tier: string;
};

export default function RegisterPage() {
  const router = useRouter();
  const [step, setStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [verifying, setVerifying] = useState(false);
  const [verifyResult, setVerifyResult] = useState<{
    verified: boolean;
    matched_platforms: string[];
    detected_city?: string | null;
  } | null>(null);

  const [form, setForm] = useState<Form>({
    platforms: [],
    name: "", age: "", phone: "", email: "",
    password: "", confirmPassword: "",
    pan: "", aadhaar: "", city: "", area: "",
    deliveryId: "", upi: "", bank: "",
    consent: false, autopay: false,
    tier: "standard",
  });

  const update = (key: keyof Form, val: unknown) =>
    setForm((p) => ({ ...p, [key]: val }));
  const togglePlatform = (id: string) =>
    setForm((p) => ({
      ...p,
      platforms: p.platforms.includes(id)
        ? p.platforms.filter((x) => x !== id)
        : [...p.platforms, id],
    }));
  const nextStep = () => { setError(""); setStep((s) => Math.min(s + 1, 4)); };
  const prevStep = () => { setError(""); setStep((s) => Math.max(s - 1, 1)); };

  const weeklyPremium = form.tier === "basic" ? 30 : form.tier === "standard" ? 60 : 105;
  const maxPayout = form.tier === "basic" ? 500 : form.tier === "standard" ? 1200 : 2500;

  const handleSubmit = async () => {
    if (form.password !== form.confirmPassword) { setError("Passwords do not match."); return; }
    if (form.password.length < 6) { setError("Password must be at least 6 characters."); return; }
    setLoading(true);
    setError("");
    try {
      const res = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: form.name, age: form.age, phone: form.phone,
          email: form.email, password: form.password,
          city: form.city, area: form.area, deliveryId: form.deliveryId,
          platforms: form.platforms, pan: form.pan, aadhaar: form.aadhaar,
          upi: form.upi, bank: form.bank,
          consent: form.consent,
          autopay: form.autopay, tier: form.tier,
        }),
      });
      const data = await res.json();
      if (!res.ok) { setError(data.error || "Registration failed."); return; }
      localStorage.setItem("gg_token", data.token);
      localStorage.setItem("gg_user", JSON.stringify(data.user));
      router.push("/dashboard");
    } catch {
      setError("Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleVerifyId = async () => {
    if (!form.deliveryId || form.platforms.length === 0) {
      setError("Enter your Delivery Partner ID and select at least one platform first.");
      return;
    }
    setVerifying(true);
    setError("");
    setVerifyResult(null);
    try {
      const res = await fetch("/api/verify-id", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ deliveryId: form.deliveryId, platforms: form.platforms }),
      });
      const data = await res.json();
      if (!res.ok) { setError(data.error || "Verification failed."); return; }
      setVerifyResult(data);
      if (data.verified && data.detected_city) {
        update("city", data.detected_city);
      }
      if (!data.verified) {
        setError("Your Delivery Partner ID was not found. You can still register, but manual admin verification will be required.");
      }
    } catch {
      setError("Verification service unavailable. You can still proceed.");
    } finally {
      setVerifying(false);
    }
  };

  return (
    <div className={styles.page}>
      <div className={styles.container}>
        {/* Header */}
        <div className={styles.header}>
          <div className={styles.logoMark}>GG</div>
          <h1 className={styles.title}>Create your account</h1>
          <p className={styles.subtitle}>
            Register as a delivery partner on WPIP (WPIP)
          </p>
        </div>

        {/* Stepper */}
        <div className={styles.stepper}>
          {STEPS.map((s, i) => (
            <div key={s} className={styles.stepItem}>
              <div
                className={`${styles.stepCircle} ${
                  step > i + 1
                    ? styles.stepCircleDone
                    : step === i + 1
                    ? styles.stepCircleActive
                    : ""
                }`}
              >
                {step > i + 1 ? "✓" : i + 1}
              </div>
              <span
                className={`${styles.stepLabel} ${step === i + 1 ? styles.stepLabelActive : ""}`}
              >
                {s}
              </span>
              {i < STEPS.length - 1 && (
                <div
                  className={`${styles.stepConnector} ${step > i + 1 ? styles.stepConnectorDone : ""}`}
                />
              )}
            </div>
          ))}
        </div>

        {/* Error */}
        {error && <div className={styles.errorBox}>{error}</div>}

        {/* Card */}
        <div className={styles.card}>
          {/* ── Step 1: Platforms ── */}
          {step === 1 && (
            <div>
              <h2 className={styles.stepTitle}>Choose your platforms</h2>
              <p className={styles.stepDesc}>
                Select all delivery platforms you work with.
              </p>
              <div className={styles.platformGrid}>
                {PLATFORMS.map((p) => {
                  const sel = form.platforms.includes(p.id);
                  return (
                    <div
                      key={p.id}
                      onClick={() => togglePlatform(p.id)}
                      className={`${styles.platformCard} ${sel ? styles.platformCardSelected : ""}`}
                    >
                      <div
                        className={`${styles.platformSymbol} ${!p.logo ? (styles[p.symClass] ?? styles.symDefault) : styles.platformLogoWrap}`}
                      >
                        {p.logo ?? p.id.slice(0, 2).toUpperCase()}
                      </div>
                      <div
                        className={`${styles.platformName} ${sel ? styles.platformNameSelected : ""}`}
                      >
                        {p.name}
                      </div>
                      {sel && (
                        <div className={styles.platformSelectedBadge}>
                          Selected
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
              {form.platforms.length > 0 && (
                <div className={styles.selectionSuccess}>
                  {form.platforms.length} platform
                  {form.platforms.length > 1 ? "s" : ""} selected
                </div>
              )}
              <button
                type="button"
                disabled={form.platforms.length === 0}
                onClick={nextStep}
                className={`${styles.btnPrimary} ${styles.btnFull}`}
              >
                Continue
              </button>
            </div>
          )}

          {/* ── Step 2: Personal Info ── */}
          {step === 2 && (
            <div>
              <h2 className={styles.stepTitle}>Personal information</h2>
              <div className={styles.formGrid}>
                <div className={styles.fullSpan}>
                  <label className={styles.fieldLabel}>Full Name</label>
                  <input
                    className={styles.input}
                    placeholder="Your full name"
                    value={form.name}
                    onChange={(e) => update("name", e.target.value)}
                  />
                </div>
                <div>
                  <label className={styles.fieldLabel}>Age</label>
                  <input
                    className={styles.input}
                    placeholder="25"
                    type="number"
                    value={form.age}
                    onChange={(e) => update("age", e.target.value)}
                  />
                </div>
                <div>
                  <label className={styles.fieldLabel}>Phone</label>
                  <input
                    className={styles.input}
                    placeholder="+91 98765 43210"
                    value={form.phone}
                    onChange={(e) => update("phone", e.target.value)}
                  />
                </div>
                <div className={styles.fullSpan}>
                  <label className={styles.fieldLabel}>Email</label>
                  <input
                    className={styles.input}
                    type="email"
                    placeholder="you@example.com"
                    value={form.email}
                    onChange={(e) => update("email", e.target.value)}
                  />
                </div>
                <div>
                  <label className={styles.fieldLabel}>Password</label>
                  <input
                    className={styles.input}
                    type="password"
                    placeholder="Min. 6 characters"
                    value={form.password}
                    onChange={(e) => update("password", e.target.value)}
                  />
                </div>
                <div>
                  <label className={styles.fieldLabel}>Confirm Password</label>
                  <input
                    className={styles.input}
                    type="password"
                    placeholder="Re-enter password"
                    value={form.confirmPassword}
                    onChange={(e) => update("confirmPassword", e.target.value)}
                  />
                </div>
                <div>
                  <label className={styles.fieldLabel}>City</label>
                  <select
                    className={styles.select}
                    value={form.city}
                    onChange={(e) => update("city", e.target.value)}
                    title="Select your city"
                    disabled={!!(verifyResult?.verified && verifyResult?.detected_city)}
                  >
                    <option value="">Select city</option>
                    {CITIES.map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                  {verifyResult?.verified && verifyResult?.detected_city && (
                    <p className={styles.fieldHintAccent}>
                      City auto-detected from your platform: {verifyResult.detected_city}
                    </p>
                  )}
                </div>
                <div>
                  <label className={styles.fieldLabel}>Delivery Area / Zone</label>
                  <input
                    className={styles.input}
                    placeholder="e.g. Koramangala"
                    value={form.area}
                    onChange={(e) => update("area", e.target.value)}
                  />
                </div>
                <div className={styles.fullSpan}>
                  <label className={styles.fieldLabel}>
                    Primary Delivery Partner ID
                  </label>
                  <div className={styles.verifyRow}>
                    <input
                      className={`${styles.input} ${styles.verifyInput}`}
                      placeholder="Your ID from the platform app"
                      value={form.deliveryId}
                      onChange={(e) => {
                        update("deliveryId", e.target.value);
                        setVerifyResult(null);
                      }}
                    />
                    <button
                      type="button"
                      onClick={handleVerifyId}
                      disabled={verifying || !form.deliveryId}
                      className={styles.verifyBtn}
                    >
                      {verifying ? "Verifying..." : "Verify ID"}
                    </button>
                  </div>
                  <p className={styles.fieldHint}>
                    Click &quot;Verify ID&quot; to check against selected platform databases.
                  </p>
                  {verifyResult?.verified && (
                    <div className={styles.verifyOk}>
                      ID verified on:{" "}
                      {verifyResult.matched_platforms
                        .map((id) => platformLabel(id))
                        .join(", ")}
                    </div>
                  )}
                  {verifyResult && !verifyResult.verified && (
                    <div className={styles.verifyFail}>
                      ID not found. You can still register, but admin manual
                      verification will be required.
                    </div>
                  )}
                </div>
              </div>
              <div className={styles.actions}>
                <button type="button" onClick={prevStep} className={styles.btnOutline}>Back</button>
                <button type="button" onClick={nextStep} className={styles.btnPrimary}>Continue</button>
              </div>
            </div>
          )}

          {/* ── Step 3: Documents ── */}
          {step === 3 && (
            <div>
              <h2 className={styles.stepTitle}>Identity & payment</h2>
              <div className={styles.formGrid}>
                <div>
                  <label className={styles.fieldLabel}>PAN Card Number</label>
                  <input
                    className={styles.input}
                    placeholder="ABCDE1234F"
                    value={form.pan}
                    onChange={(e) => update("pan", e.target.value.toUpperCase())}
                    maxLength={10}
                  />
                </div>
                <div>
                  <label className={styles.fieldLabel}>Aadhaar Number</label>
                  <input
                    className={styles.input}
                    placeholder="1234 5678 9012"
                    value={form.aadhaar}
                    onChange={(e) => update("aadhaar", e.target.value)}
                    maxLength={12}
                  />
                </div>
                <div className={styles.fullSpan}>
                  <label className={styles.fieldLabel}>UPI ID (for payouts)</label>
                  <input
                    className={styles.input}
                    placeholder="yourname@upi"
                    value={form.upi}
                    onChange={(e) => update("upi", e.target.value)}
                  />
                </div>
                <div className={styles.fullSpan}>
                  <label className={styles.fieldLabel}>Bank Account (optional)</label>
                  <input
                    className={styles.input}
                    placeholder="IFSC + Account Number"
                    value={form.bank}
                    onChange={(e) => update("bank", e.target.value)}
                  />
                </div>
              </div>

              <div className={styles.consentSection}>
                <p className={styles.consentHeading}>Consent & authorisations</p>
                {[
                  {
                    key: "consent" as keyof Form,
                    label: "I authorise WPIP to monitor weather and disruption data in my delivery zone for insurance claims.",
                  },
                  {
                    key: "autopay" as keyof Form,
                    label: "Enable AutoPay — auto-deduct weekly premium from platform payout (5% discount).",
                  },
                ].map((item) => (
                  <label key={item.key} className={styles.consentItem}>
                    <input
                      type="checkbox"
                      className={styles.consentCheckbox}
                      checked={!!form[item.key]}
                      onChange={(e) => update(item.key, e.target.checked)}
                    />
                    <span className={styles.consentText}>{item.label}</span>
                  </label>
                ))}
              </div>

              <div className={styles.actions}>
                <button type="button" onClick={prevStep} className={styles.btnOutline}>Back</button>
                <button type="button" onClick={nextStep} className={styles.btnPrimary}>Continue</button>
              </div>
            </div>
          )}

          {/* ── Step 4: Coverage ── */}
          {step === 4 && (
            <div>
              <h2 className={styles.stepTitle}>Choose your coverage</h2>
              <div className={styles.dynamicNotice}>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
                Premiums are dynamically adjusted every week based on your city&apos;s weather, AQI, and disruption risk.
              </div>
              <div className={styles.tierGrid}>
                {[
                  { id: "basic",    tier: "Basic",    base: 30,  payout: "500" },
                  { id: "standard", tier: "Standard", base: 60,  payout: "1,200", recommended: true },
                  { id: "pro",      tier: "Pro",      base: 105, payout: "2,500" },
                ].map((t) => {
                  const sel = form.tier === t.id;
                  return (
                    <div
                      key={t.id}
                      onClick={() => update("tier", t.id)}
                      className={`${styles.tierCard} ${sel ? styles.tierCardSelected : ""}`}
                    >
                      {t.recommended && (
                        <div className={styles.recommendedBadge}>Recommended</div>
                      )}
                      <div className={`${styles.tierName} ${sel ? styles.tierNameSelected : ""}`}>
                        {t.tier}
                      </div>
                      <div className={`${styles.tierPrice} ${sel ? styles.tierPriceSelected : ""}`}>
                        ₹{t.base}
                      </div>
                      <div className={styles.tierUnit}>base / week</div>
                      <div className={styles.tierAdjusted}>adjusted weekly</div>
                      <div className={styles.tierPayout}>
                        Max payout: ₹{t.payout}
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className={styles.summaryBox}>
                <p className={styles.summaryHeading}>Summary</p>
                {[
                  ["Platforms", form.platforms.map((id) => platformLabel(id)).join(", ") || "–"],
                  ["Base Weekly Premium", `₹${weeklyPremium} (adjusted dynamically)`],
                  ["Max Weekly Payout", `₹${maxPayout}`],
                  ["AutoPay", form.autopay ? "Enabled (5% discount)" : "Disabled"],
                ].map(([k, v]) => (
                  <div key={k} className={styles.summaryRow}>
                    <span className={styles.summaryKey}>{k}</span>
                    <span className={styles.summaryVal}>{v}</span>
                  </div>
                ))}
              </div>

              <div className={styles.actions}>
                <button type="button" onClick={prevStep} className={styles.btnOutline}>Back</button>
                <button
                  type="button"
                  onClick={handleSubmit}
                  disabled={loading}
                  className={styles.btnPrimary}
                >
                  {loading ? "Creating account..." : "Create account"}
                </button>
              </div>
            </div>
          )}
        </div>

        <p className={styles.footerText}>
          Already have an account?{" "}
          <span className={styles.footerLink} onClick={() => router.push("/login")}>
            Sign in
          </span>
        </p>
      </div>
    </div>
  );
}
