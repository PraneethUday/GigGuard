"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import styles from "./page.module.css";

// ─── Types ────────────────────────────────────────────────────────────────────

type User = {
  id: string;
  name: string;
  email: string;
  platforms: string[];
  tier: string;
  verification_status: string;
  city: string;
  area?: string;
  delivery_id?: string;
  autopay?: boolean;
  phone?: string;
};

type Premium = {
  weekly_premium: number;
  weekly_premium_autopay: number;
  raw_prediction: number;
  tier: string;
  max_payout: number;
  weather_risk: number;
  city_risk: number;
  weekly_earnings_est: number;
  history_days?: number;
  weather?: {
    temperature: number;
    aqi_index: number;
    rain_1h: number;
    weather_main: string;
    humidity: number;
  };
};

type Tier = "basic" | "standard" | "pro";
type QuoteMap = Partial<Record<Tier, Premium>>;

type PaymentRecord = {
  id: string;
  amount: number;
  method: "upi" | "debit" | "credit";
  status: "success";
  timestamp: string;
  tier: string;
};

// ─── Constants ────────────────────────────────────────────────────────────────

const PLATFORM_META: Record<
  string,
  { name: string; symbol: string; iconClass: string }
> = {
  swiggy: { name: "Swiggy", symbol: "SW", iconClass: "platformIconSwiggy" },
  zomato: { name: "Zomato", symbol: "ZO", iconClass: "platformIconZomato" },
  amazon: {
    name: "Amazon Flex",
    symbol: "AF",
    iconClass: "platformIconAmazon",
  },
  blinkit: { name: "Blinkit", symbol: "BL", iconClass: "platformIconBlinkit" },
  zepto: { name: "Zepto", symbol: "ZE", iconClass: "platformIconZepto" },
  meesho: { name: "Meesho", symbol: "ME", iconClass: "platformIconMeesho" },
  porter: { name: "Porter", symbol: "PO", iconClass: "platformIconPorter" },
  dunzo: { name: "Dunzo", symbol: "DZ", iconClass: "platformIconDunzo" },
};

const TIERS: Tier[] = ["basic", "standard", "pro"];

const PLAN_DETAILS: Record<
  Tier,
  { label: string; tag: string; includes: string[] }
> = {
  basic: {
    label: "Basic Shield",
    tag: "Low-cost entry cover",
    includes: ["Daily income interruption support"],
  },
  standard: {
    label: "Standard Guard",
    tag: "Balanced weekly protection",
    includes: ["Income loss support", "Weather disruption support"],
  },
  pro: {
    label: "Pro Protect",
    tag: "Highest payout priority",
    includes: [
      "Income loss support",
      "Weather disruption support",
      "Priority claims processing",
    ],
  },
};

const NAV = [
  { id: "home", label: "Dashboard" },
  { id: "claims", label: "Claims History" },
  { id: "payments", label: "Payments" },
  { id: "profile", label: "Profile" },
];

// ─── Helpers ──────────────────────────────────────────────────────────────────

function normalizeTier(tier: string | undefined): Tier {
  if (tier === "basic" || tier === "pro") return tier;
  return "standard";
}

function planStorageKey(userId: string) {
  return `gg_next_week_plan_${userId}`;
}

function paymentsStorageKey(userId: string) {
  return `gg_payments_${userId}`;
}

function getNextWeekWindow(): { label: string } {
  const now = new Date();
  const dayFromMonday = (now.getDay() + 6) % 7;
  const from = new Date(now);
  from.setDate(now.getDate() + (7 - dayFromMonday));
  const to = new Date(from);
  to.setDate(from.getDate() + 6);
  const fmt = new Intl.DateTimeFormat("en-IN", {
    day: "numeric",
    month: "short",
  });
  return { label: `${fmt.format(from)} – ${fmt.format(to)}` };
}

function money(value: number | undefined | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "–";
  return `₹${Math.round(value)}`;
}

function riskLabel(risk: number): string {
  if (risk < 0.15) return "Low";
  if (risk < 0.4) return "Moderate";
  if (risk < 0.7) return "High";
  return "Severe";
}

function formatCardNumber(val: string): string {
  return val
    .replace(/\D/g, "")
    .slice(0, 16)
    .replace(/(.{4})/g, "$1 ")
    .trim();
}

function formatExpiry(val: string): string {
  const d = val.replace(/\D/g, "").slice(0, 4);
  return d.length >= 3 ? `${d.slice(0, 2)}/${d.slice(2)}` : d;
}

function methodLabel(m: string): string {
  if (m === "upi") return "UPI";
  if (m === "debit") return "Debit Card";
  return "Credit Card";
}

async function fetchPremiumForTier(user: User, tier: Tier): Promise<Premium> {
  const res = await fetch("/api/premium/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      delivery_id: user.delivery_id,
      city: user.city || "Unknown",
      tier,
    }),
  });
  const data = await res.json();
  if (!res.ok || data.error)
    throw new Error(data.error || "Premium request failed");
  return data as Premium;
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function StatusPill({
  label,
  ok,
  okText,
  badText,
}: {
  label: string;
  ok: boolean;
  okText: string;
  badText: string;
}) {
  return (
    <div
      className={`${styles.statusPill} ${ok ? styles.statusOk : styles.statusBad}`}
    >
      <span>{label}</span>
      <strong>{ok ? okText : badText}</strong>
    </div>
  );
}

function MetricCard({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: "blue" | "green" | "amber" | "ink";
}) {
  return (
    <div className={`${styles.metricCard} ${styles[`tone_${tone}`]}`}>
      <div className={styles.metricLabel}>{label}</div>
      <div className={styles.metricValue}>{value}</div>
    </div>
  );
}

function ProfileRow({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className={styles.profileRow}>
      <span>{label}</span>
      <strong className={mono ? styles.profileRowMono : undefined}>
        {value}
      </strong>
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const router = useRouter();

  // user / data
  const [user, setUser] = useState<User | null>(null);
  const [tab, setTab] = useState("home");
  const [currentPremium, setCurrentPremium] = useState<Premium | null>(null);
  const [quotes, setQuotes] = useState<QuoteMap>({});
  const [nextWeekTier, setNextWeekTier] = useState<Tier>("standard");
  const [loadingCurrent, setLoadingCurrent] = useState(false);
  const [loadingQuotes, setLoadingQuotes] = useState(false);
  const [premiumError, setPremiumError] = useState("");
  const [planMessage, setPlanMessage] = useState("");
  const [claims, setClaims] = useState<Record<string, unknown>[]>([]);
  const [loadingClaims, setLoadingClaims] = useState(false);

  // payment
  const [payments, setPayments] = useState<PaymentRecord[]>([]);
  const [showPayModal, setShowPayModal] = useState(false);
  const [payMethod, setPayMethod] = useState<"upi" | "debit" | "credit">("upi");
  const [payForm, setPayForm] = useState({
    upi: "",
    cardNumber: "",
    expiry: "",
    cvv: "",
    name: "",
  });
  const [payLoading, setPayLoading] = useState(false);
  const [payError, setPayError] = useState("");
  const [paySuccess, setPaySuccess] = useState<{
    amount: number;
    txId: string;
  } | null>(null);

  // ── auth
  const logout = () => {
    localStorage.removeItem("gg_token");
    localStorage.removeItem("gg_user");
    router.push("/login");
  };

  const refreshUserFromServer = async (token: string, fallback: User) => {
    try {
      const res = await fetch("/api/auth/me", {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) {
        if (res.status === 401) logout();
        return;
      }
      const data = await res.json();
      if (!data.user) return;
      setUser(data.user);
      localStorage.setItem("gg_user", JSON.stringify(data.user));
      const savedPlan = localStorage.getItem(planStorageKey(data.user.id));
      setNextWeekTier(normalizeTier(savedPlan || data.user.tier));
    } catch {
      setUser(fallback);
    }
  };

  useEffect(() => {
    const token = localStorage.getItem("gg_token");
    const raw = localStorage.getItem("gg_user");
    if (!token || !raw) {
      router.replace("/login");
      return;
    }
    try {
      const parsed = JSON.parse(raw) as User;
      setUser(parsed);
      refreshUserFromServer(token, parsed);
      const interval = setInterval(
        () => refreshUserFromServer(token, parsed),
        15000,
      );
      return () => clearInterval(interval);
    } catch {
      router.replace("/login");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [router]);

  // fetch data when user id / tier changes
  useEffect(() => {
    if (!user?.delivery_id) return;
    const savedPlan = localStorage.getItem(planStorageKey(user.id));
    setNextWeekTier(normalizeTier(savedPlan || user.tier));
    fetchCurrentPremium(user);
    fetchTierQuotes(user);
    fetchClaims(user);
    // load stored payments
    try {
      const stored = localStorage.getItem(paymentsStorageKey(user.id));
      if (stored) setPayments(JSON.parse(stored));
    } catch {
      /* ignore */
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.id, user?.tier]);

  const fetchClaims = async (u: User) => {
    if (!u.delivery_id) return;
    setLoadingClaims(true);
    try {
      const res = await fetch(`/api/backend/claims/worker/${u.delivery_id}`);
      const data = await res.json();
      if (data?.data) setClaims(data.data);
    } catch {
      /* silently fail */
    } finally {
      setLoadingClaims(false);
    }
  };

  const fetchCurrentPremium = async (u: User) => {
    if (!u.delivery_id) return;
    setLoadingCurrent(true);
    setPremiumError("");
    try {
      const data = await fetchPremiumForTier(u, normalizeTier(u.tier));
      setCurrentPremium(data);
    } catch {
      setPremiumError("Could not fetch premium. Backend may be offline.");
    } finally {
      setLoadingCurrent(false);
    }
  };

  const fetchTierQuotes = async (u: User) => {
    if (!u.delivery_id) return;
    setLoadingQuotes(true);
    try {
      const results = await Promise.allSettled(
        TIERS.map(
          async (tier) => [tier, await fetchPremiumForTier(u, tier)] as const,
        ),
      );
      const next: QuoteMap = {};
      results.forEach((r) => {
        if (r.status === "fulfilled") next[r.value[0]] = r.value[1];
      });
      setQuotes(next);
    } finally {
      setLoadingQuotes(false);
    }
  };

  const applyNextWeekPlan = (tier: Tier) => {
    if (!user) return;
    setNextWeekTier(tier);
    localStorage.setItem(planStorageKey(user.id), tier);
    setPlanMessage(`${PLAN_DETAILS[tier].label} scheduled for next week.`);
  };

  // ── payment
  const openPayModal = () => {
    setPayError("");
    setPaySuccess(null);
    setPayForm({ upi: "", cardNumber: "", expiry: "", cvv: "", name: "" });
    setPayMethod("upi");
    setShowPayModal(true);
  };

  const closePayModal = () => {
    if (payLoading) return;
    setShowPayModal(false);
    setPaySuccess(null);
    setPayError("");
  };

  const handlePay = async () => {
    if (!user) return;
    setPayError("");

    if (payMethod === "upi") {
      if (!payForm.upi.includes("@")) {
        setPayError("Enter a valid UPI ID (e.g. yourname@paytm)");
        return;
      }
    } else {
      const digits = payForm.cardNumber.replace(/\s/g, "");
      if (digits.length < 16) {
        setPayError("Enter a valid 16-digit card number");
        return;
      }
      if (!/^\d{2}\/\d{2}$/.test(payForm.expiry)) {
        setPayError("Enter expiry as MM/YY (e.g. 09/27)");
        return;
      }
      if (payForm.cvv.length < 3) {
        setPayError("Enter a valid 3-digit CVV");
        return;
      }
      if (!payForm.name.trim()) {
        setPayError("Enter the cardholder name");
        return;
      }
    }

    setPayLoading(true);
    // simulate processing delay
    await new Promise((r) => setTimeout(r, 2000));

    try {
      const token = localStorage.getItem("gg_token");
      const res = await fetch("/api/payment/pay", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          amount: currentWeeklyPremium,
          method: payMethod,
          tier: user.tier,
          upi: payMethod === "upi" ? payForm.upi : undefined,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        setPayError(data.error || "Payment failed. Please try again.");
        return;
      }

      setPaySuccess({ amount: data.amount, txId: data.transaction_id });

      const record: PaymentRecord = {
        id: data.transaction_id,
        amount: data.amount,
        method: payMethod,
        status: "success",
        timestamp: new Date().toISOString(),
        tier: user.tier,
      };
      const updated = [record, ...payments];
      setPayments(updated);
      localStorage.setItem(
        paymentsStorageKey(user.id),
        JSON.stringify(updated),
      );
    } catch {
      setPayError("Something went wrong. Please try again.");
    } finally {
      setPayLoading(false);
    }
  };

  // ── derived
  if (!user) return null;

  const currentTier = normalizeTier(user.tier);
  const tierLabel = PLAN_DETAILS[currentTier].label;
  const verified = user.verification_status === "verified";
  const nextWindow = getNextWeekWindow();
  const selectedNextQuote = quotes[nextWeekTier];
  const coveredNow = verified && !!currentPremium;
  const coveredNextWeek = verified && !!selectedNextQuote;

  const currentWeeklyPremium = currentPremium
    ? user.autopay
      ? currentPremium.weekly_premium_autopay
      : currentPremium.weekly_premium
    : null;
  const nextWeeklyPremium = selectedNextQuote
    ? user.autopay
      ? selectedNextQuote.weekly_premium_autopay
      : selectedNextQuote.weekly_premium
    : null;

  const today = new Date().toDateString();
  const paidToday =
    payments.length > 0 &&
    new Date(payments[0].timestamp).toDateString() === today;

  return (
    <div className={styles.pageRoot}>
      {/* ── Header ── */}
      <header className={styles.header}>
        <div className={styles.brandBlock}>
          <div className={styles.brandLogo}>WP</div>
          <span className={styles.brandText}>WPIP</span>
        </div>
        <div className={styles.headerActions}>
          <div className={styles.userBadge}>
            <div className={styles.userInitial}>
              {user.name.charAt(0).toUpperCase()}
            </div>
            <span className={styles.userName}>{user.name}</span>
          </div>
          <button type="button" onClick={logout} className={styles.logoutBtn}>
            Logout
          </button>
        </div>
      </header>

      <div className={styles.layout}>
        {/* ── Sidebar ── */}
        <aside className={styles.sidebar}>
          {NAV.map((item) => (
            <button
              type="button"
              key={item.id}
              onClick={() => setTab(item.id)}
              className={`${styles.navBtn} ${tab === item.id ? styles.navBtnActive : ""}`}
            >
              {item.label}
            </button>
          ))}
        </aside>

        <main className={styles.main}>
          <div className={styles.content}>
            {/* ══ HOME TAB ══ */}
            {tab === "home" && (
              <div>
                <div className={styles.hero}>
                  <h1 className={styles.heroTitle}>
                    Coverage dashboard for {user.name}
                  </h1>
                  <p className={styles.heroSub}>
                    Live insurance status, next-week plan, and weather-adjusted
                    premium insights.
                  </p>
                  <div className={styles.heroStatusRow}>
                    <StatusPill
                      label="Now"
                      ok={coveredNow}
                      okText="Covered"
                      badText="Not covered"
                    />
                    <StatusPill
                      label="Next week"
                      ok={coveredNextWeek}
                      okText="Planned"
                      badText="Not scheduled"
                    />
                    <StatusPill
                      label="Verification"
                      ok={verified}
                      okText="Verified"
                      badText="Pending"
                    />
                  </div>
                </div>

                <div className={styles.grid4}>
                  <MetricCard
                    label="Current plan"
                    value={tierLabel}
                    tone="blue"
                  />
                  <MetricCard
                    label="Weekly premium"
                    value={money(currentWeeklyPremium)}
                    tone="green"
                  />
                  <MetricCard
                    label="Max payout"
                    value={money(currentPremium?.max_payout)}
                    tone="amber"
                  />
                  <MetricCard
                    label="City"
                    value={user.city || "–"}
                    tone="ink"
                  />
                </div>

                {/* Pay Banner */}
                <div className={styles.payBanner}>
                  <div>
                    <div className={styles.payBannerTitle}>
                      This week&apos;s premium due
                    </div>
                    <div className={styles.payBannerAmount}>
                      {money(currentWeeklyPremium)}
                    </div>
                    <div className={styles.bannerMeta}>
                      {tierLabel} · {user.city}
                    </div>
                  </div>
                  {paidToday ? (
                    <div className={styles.payBannerPaid}>✓ Paid today</div>
                  ) : (
                    <button
                      type="button"
                      className={styles.payBannerBtn}
                      onClick={openPayModal}
                      disabled={!currentPremium}
                    >
                      Pay Now →
                    </button>
                  )}
                </div>

                {/* Coverage this week */}
                <section className={styles.panel}>
                  <div className={styles.panelHead}>
                    <h3 className={styles.panelTitle}>Coverage this week</h3>
                    <p className={styles.panelSub}>
                      Active plan: {PLAN_DETAILS[currentTier].label}. Coverage
                      requires verification and a computed premium.
                    </p>
                  </div>

                  {loadingCurrent ? (
                    <div className={styles.muted}>
                      Calculating current-week premium…
                    </div>
                  ) : premiumError ? (
                    <div className={styles.errorText}>{premiumError}</div>
                  ) : currentPremium ? (
                    <div className={styles.coverageGrid}>
                      <div className={styles.coverageCard}>
                        <div className={styles.coverageLabel}>Status</div>
                        <div className={styles.coverageValue}>
                          {coveredNow
                            ? "Active coverage"
                            : "Coverage unavailable"}
                        </div>
                      </div>
                      <div className={styles.coverageCard}>
                        <div className={styles.coverageLabel}>
                          Weekly premium
                        </div>
                        <div className={styles.coverageValue}>
                          {money(currentWeeklyPremium)}
                        </div>
                      </div>
                      <div className={styles.coverageCard}>
                        <div className={styles.coverageLabel}>
                          Max payout per claim
                        </div>
                        <div className={styles.coverageValue}>
                          {money(currentPremium.max_payout)}
                        </div>
                      </div>
                      <div className={styles.coverageCard}>
                        <div className={styles.coverageLabel}>Risk level</div>
                        <div className={styles.coverageValue}>
                          {riskLabel(currentPremium.weather_risk)}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className={styles.muted}>
                      Current plan data is not available.
                    </div>
                  )}

                  {currentPremium?.weather && (
                    <div className={styles.weatherStrip}>
                      <span>
                        Temp:{" "}
                        <strong>{currentPremium.weather.temperature}°C</strong>
                      </span>
                      <span>
                        AQI: <strong>{currentPremium.weather.aqi_index}</strong>
                      </span>
                      <span>
                        Rain:{" "}
                        <strong>{currentPremium.weather.rain_1h} mm/h</strong>
                      </span>
                      <span>
                        Humidity:{" "}
                        <strong>{currentPremium.weather.humidity}%</strong>
                      </span>
                      <span>
                        Condition:{" "}
                        <strong>{currentPremium.weather.weather_main}</strong>
                      </span>
                    </div>
                  )}

                  <ul className={styles.coverList}>
                    {PLAN_DETAILS[currentTier].includes.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </section>

                {/* Next week plan */}
                <section className={styles.panel}>
                  <div className={styles.panelHead}>
                    <h3 className={styles.panelTitle}>
                      Insurance for next week
                    </h3>
                    <p className={styles.panelSub}>
                      {nextWindow.label} · select a plan to schedule it for next
                      week.
                    </p>
                  </div>

                  <div className={styles.grid3}>
                    {TIERS.map((tier) => {
                      const quote = quotes[tier];
                      const selected = tier === nextWeekTier;
                      return (
                        <div
                          key={tier}
                          className={`${styles.planCard} ${selected ? styles.planCardSelected : ""}`}
                        >
                          <div className={styles.planTopRow}>
                            <h4>{PLAN_DETAILS[tier].label}</h4>
                            <span className={styles.planTag}>
                              {PLAN_DETAILS[tier].tag}
                            </span>
                          </div>
                          <div className={styles.planPremium}>
                            {loadingQuotes && !quote
                              ? "Calculating…"
                              : money(
                                  user.autopay
                                    ? quote?.weekly_premium_autopay
                                    : quote?.weekly_premium,
                                )}
                          </div>
                          <div className={styles.planPayout}>
                            Max payout: {money(quote?.max_payout)}
                          </div>
                          <button
                            type="button"
                            className={
                              selected ? styles.planBtnSelected : styles.planBtn
                            }
                            onClick={() => applyNextWeekPlan(tier)}
                          >
                            {selected
                              ? "Scheduled for next week"
                              : "Choose for next week"}
                          </button>
                        </div>
                      );
                    })}
                  </div>

                  <div className={styles.nextWeekSummary}>
                    <div>
                      <div className={styles.coverageLabel}>Scheduled plan</div>
                      <div className={styles.coverageValue}>
                        {PLAN_DETAILS[nextWeekTier].label}
                      </div>
                    </div>
                    <div>
                      <div className={styles.coverageLabel}>
                        Projected premium
                      </div>
                      <div className={styles.coverageValue}>
                        {money(nextWeeklyPremium)}
                      </div>
                    </div>
                    <div>
                      <div className={styles.coverageLabel}>Coverage state</div>
                      <div className={styles.coverageValue}>
                        {coveredNextWeek
                          ? "Will be covered"
                          : "Not covered yet"}
                      </div>
                    </div>
                  </div>

                  <ul className={styles.coverList}>
                    {PLAN_DETAILS[nextWeekTier].includes.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>

                  {planMessage && (
                    <div className={styles.successText}>{planMessage}</div>
                  )}
                </section>

                {/* Platforms */}
                <section className={styles.panel}>
                  <h3 className={styles.panelTitle}>Connected platforms</h3>
                  <div className={styles.platformWrap}>
                    {user.platforms.map((p) => {
                      const meta = PLATFORM_META[p] ?? {
                        name: p,
                        symbol: p.slice(0, 2).toUpperCase(),
                        iconClass: "platformIconDefault",
                      };
                      return (
                        <span key={p} className={styles.platformChip}>
                          <span
                            className={`${styles.platformChipIcon} ${styles[meta.iconClass]}`}
                          >
                            {meta.symbol}
                          </span>
                          {meta.name}
                        </span>
                      );
                    })}
                  </div>
                </section>

                {!verified && (
                  <div className={styles.warningBox}>
                    Your account is pending verification. Insurance remains
                    inactive until verification completes.
                  </div>
                )}
              </div>
            )}

            {/* ══ CLAIMS TAB ══ */}
            {tab === "claims" && (
              <div>
                <h1 className={styles.profileTitle}>Claims History</h1>
                <p className={`${styles.muted} ${styles.claimsIntro}`}>
                  Claims are automatically generated during severe weather
                  disruptions.
                </p>
                {loadingClaims ? (
                  <div className={styles.muted}>Loading claims…</div>
                ) : claims.length === 0 ? (
                  <div className={styles.emptyState}>
                    No claims yet. Claims appear automatically when a disruption
                    event is detected in your city.
                  </div>
                ) : (
                  <div className={styles.claimsList}>
                    {claims.map((c) => (
                      <div
                        key={String(c.id)}
                        className={`${styles.panel} ${styles.claimPanel}`}
                      >
                        <div className={styles.planTopRow}>
                          <h4 className={styles.claimHeading}>
                            {String(c.claim_number ?? "–")}
                          </h4>
                          <span
                            className={
                              c.payout_status === "paid"
                                ? styles.badgeGreen
                                : c.payout_status === "rejected"
                                  ? styles.badgeRed
                                  : styles.badgeAmber
                            }
                          >
                            {String(c.payout_status ?? "").toUpperCase()}
                          </span>
                        </div>
                        <div className={styles.claimBody}>
                          <ProfileRow
                            label="Disruption Event"
                            value={String(c.trigger_type ?? "").replace(
                              /_/g,
                              " ",
                            )}
                          />
                          <ProfileRow
                            label="Payout Amount"
                            value={`₹${c.payout_amount ?? 0}`}
                          />
                          <ProfileRow
                            label="City"
                            value={String(c.city ?? "–")}
                          />
                          <ProfileRow
                            label="Date Initiated"
                            value={
                              c.created_at
                                ? new Date(
                                    String(c.created_at),
                                  ).toLocaleDateString("en-IN")
                                : "–"
                            }
                          />
                          {c.payout_status === "paid" && (
                            <ProfileRow
                              label="Transaction ID"
                              value={String(c.transaction_id ?? "–")}
                              mono
                            />
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* ══ PAYMENTS TAB ══ */}
            {tab === "payments" && (
              <div>
                <div className={styles.tabHeader}>
                  <div>
                    <h1 className={styles.profileTitle}>Payments</h1>
                    <p className={styles.muted}>
                      Pay your weekly premium securely and view your payment
                      history.
                    </p>
                  </div>
                  {paidToday ? (
                    <div className={styles.paidBadge}>✓ Paid today</div>
                  ) : (
                    <button
                      type="button"
                      className={styles.primaryBtn}
                      onClick={openPayModal}
                      disabled={!currentPremium}
                    >
                      Pay {money(currentWeeklyPremium)} →
                    </button>
                  )}
                </div>

                {/* Pay banner */}
                <div className={styles.payBanner}>
                  <div>
                    <div className={styles.payBannerTitle}>
                      This week&apos;s premium due
                    </div>
                    <div className={styles.payBannerAmount}>
                      {money(currentWeeklyPremium)}
                    </div>
                    <div className={styles.bannerMeta}>
                      {tierLabel} · AutoPay{" "}
                      {user.autopay ? "enabled (−5%)" : "disabled"}
                    </div>
                  </div>
                  {paidToday ? (
                    <div className={styles.payBannerPaid}>✓ Paid today</div>
                  ) : (
                    <button
                      type="button"
                      className={styles.payBannerBtn}
                      onClick={openPayModal}
                      disabled={!currentPremium}
                    >
                      Pay Now →
                    </button>
                  )}
                </div>

                {/* History */}
                <section className={styles.panel}>
                  <h3
                    className={`${styles.panelTitle} ${styles.panelTitleSpaced}`}
                  >
                    Payment History
                  </h3>
                  {payments.length === 0 ? (
                    <div className={styles.emptyState}>
                      No payments yet. Use the &ldquo;Pay Now&rdquo; button
                      above to pay your weekly premium.
                    </div>
                  ) : (
                    <table className={styles.dataTable}>
                      <thead>
                        <tr>
                          <th>Transaction ID</th>
                          <th>Amount</th>
                          <th>Method</th>
                          <th>Plan</th>
                          <th>Date</th>
                          <th>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {payments.map((p) => (
                          <tr key={p.id}>
                            <td>
                              <span className={styles.monoSmall}>{p.id}</span>
                            </td>
                            <td className={styles.greenText}>
                              <strong>{money(p.amount)}</strong>
                            </td>
                            <td>{methodLabel(p.method)}</td>
                            <td>
                              <span
                                className={
                                  p.tier === "pro"
                                    ? styles.tier_pro
                                    : p.tier === "basic"
                                      ? styles.tier_basic
                                      : styles.tier_standard
                                }
                              >
                                {p.tier}
                              </span>
                            </td>
                            <td>
                              {new Date(p.timestamp).toLocaleDateString(
                                "en-IN",
                                {
                                  day: "numeric",
                                  month: "short",
                                  year: "numeric",
                                },
                              )}
                            </td>
                            <td>
                              <span className={styles.badgeGreen}>Success</span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </section>
              </div>
            )}

            {/* ══ PROFILE TAB ══ */}
            {tab === "profile" && (
              <div>
                <h1 className={styles.profileTitle}>Profile</h1>
                <div className={styles.profileGrid}>
                  <div className={styles.panel}>
                    <h3 className={styles.panelTitle}>Personal information</h3>
                    <ProfileRow label="Name" value={user.name} />
                    <ProfileRow label="Email" value={user.email} />
                    <ProfileRow label="Phone" value={user.phone || "–"} />
                    <ProfileRow label="City" value={user.city || "–"} />
                    <ProfileRow label="Area" value={user.area || "–"} />
                    <ProfileRow label="Account UUID" value={user.id} mono />
                    <ProfileRow
                      label="Delivery Partner ID"
                      value={user.delivery_id || "–"}
                      mono
                    />
                  </div>
                  <div className={styles.panel}>
                    <h3 className={styles.panelTitle}>Insurance details</h3>
                    <ProfileRow label="Current plan" value={tierLabel} />
                    <ProfileRow
                      label="Next week plan"
                      value={PLAN_DETAILS[nextWeekTier].label}
                    />
                    <ProfileRow
                      label="Current premium"
                      value={money(currentWeeklyPremium)}
                    />
                    <ProfileRow
                      label="Next week premium"
                      value={money(nextWeeklyPremium)}
                    />
                    <ProfileRow
                      label="Verification"
                      value={verified ? "Verified" : "Pending"}
                    />
                    <ProfileRow
                      label="AutoPay"
                      value={user.autopay ? "Enabled" : "Disabled"}
                    />
                    <ProfileRow
                      label="Platforms"
                      value={user.platforms
                        .map((p) => PLATFORM_META[p]?.name || p)
                        .join(", ")}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        </main>
      </div>

      {/* ══ PAYMENT MODAL ══ */}
      {showPayModal && (
        <div className={styles.modalOverlay} onClick={closePayModal}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            {paySuccess ? (
              /* ── Success screen ── */
              <div className={styles.paySuccessBox}>
                <div className={styles.paySuccessIcon}>✅</div>
                <div className={styles.paySuccessTitle}>
                  Payment Successful!
                </div>
                <div className={styles.paySuccessAmount}>
                  {money(paySuccess.amount)}
                </div>
                <div className={styles.paySuccessId}>{paySuccess.txId}</div>
                <button
                  type="button"
                  className={`${styles.modalPayBtn} ${styles.modalDoneBtn}`}
                  onClick={closePayModal}
                >
                  Done
                </button>
              </div>
            ) : (
              /* ── Payment form ── */
              <>
                <h2 className={styles.modalTitle}>Pay Weekly Premium</h2>
                <p className={styles.modalSub}>
                  Mock payment — no real money is charged.
                </p>

                {/* Amount box */}
                <div className={styles.modalAmountBox}>
                  <div className={styles.modalAmountLabel}>Amount Due</div>
                  <div className={styles.modalAmount}>
                    {money(currentWeeklyPremium)}
                  </div>
                  <div className={styles.modalTier}>
                    {tierLabel} · {user.city}
                  </div>
                </div>

                {/* Method selector */}
                <div className={styles.modalField}>
                  <label className={styles.modalLabel}>Payment Method</label>
                  <div className={styles.methodGrid}>
                    {(["upi", "debit", "credit"] as const).map((m) => (
                      <button
                        type="button"
                        key={m}
                        className={
                          payMethod === m
                            ? styles.methodBtnSelected
                            : styles.methodBtn
                        }
                        onClick={() => {
                          setPayMethod(m);
                          setPayError("");
                        }}
                      >
                        {methodLabel(m)}
                      </button>
                    ))}
                  </div>
                </div>

                {/* UPI form */}
                {payMethod === "upi" && (
                  <div className={styles.modalField}>
                    <label className={styles.modalLabel}>UPI ID</label>
                    <input
                      className={styles.modalInput}
                      placeholder="yourname@paytm / @gpay / @upi"
                      value={payForm.upi}
                      onChange={(e) =>
                        setPayForm((p) => ({ ...p, upi: e.target.value }))
                      }
                      autoComplete="off"
                    />
                    <div className={styles.upiHint}>
                      Supports Paytm, Google Pay, PhonePe, BHIM, and all UPI
                      apps.
                    </div>
                  </div>
                )}

                {/* Card form */}
                {(payMethod === "debit" || payMethod === "credit") && (
                  <>
                    <div className={styles.modalField}>
                      <label className={styles.modalLabel}>Card Number</label>
                      <input
                        className={styles.modalInput}
                        placeholder="1234 5678 9012 3456"
                        value={payForm.cardNumber}
                        maxLength={19}
                        autoComplete="cc-number"
                        onChange={(e) =>
                          setPayForm((p) => ({
                            ...p,
                            cardNumber: formatCardNumber(e.target.value),
                          }))
                        }
                      />
                    </div>
                    <div className={styles.cardGrid}>
                      <div className={styles.modalField}>
                        <label className={styles.modalLabel}>
                          Expiry (MM/YY)
                        </label>
                        <input
                          className={styles.modalInput}
                          placeholder="09/27"
                          value={payForm.expiry}
                          maxLength={5}
                          autoComplete="cc-exp"
                          onChange={(e) =>
                            setPayForm((p) => ({
                              ...p,
                              expiry: formatExpiry(e.target.value),
                            }))
                          }
                        />
                      </div>
                      <div className={styles.modalField}>
                        <label className={styles.modalLabel}>CVV</label>
                        <input
                          className={styles.modalInput}
                          placeholder="•••"
                          type="password"
                          value={payForm.cvv}
                          maxLength={4}
                          autoComplete="cc-csc"
                          onChange={(e) =>
                            setPayForm((p) => ({
                              ...p,
                              cvv: e.target.value.replace(/\D/g, ""),
                            }))
                          }
                        />
                      </div>
                    </div>
                    <div className={styles.modalField}>
                      <label className={styles.modalLabel}>
                        Cardholder Name
                      </label>
                      <input
                        className={styles.modalInput}
                        placeholder="Name as on card"
                        value={payForm.name}
                        autoComplete="cc-name"
                        onChange={(e) =>
                          setPayForm((p) => ({ ...p, name: e.target.value }))
                        }
                      />
                    </div>
                  </>
                )}

                {payError && (
                  <div className={styles.modalError}>{payError}</div>
                )}

                <div className={styles.modalActions}>
                  <button
                    type="button"
                    className={styles.modalCancelBtn}
                    onClick={closePayModal}
                    disabled={payLoading}
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    className={styles.modalPayBtn}
                    onClick={handlePay}
                    disabled={payLoading}
                  >
                    {payLoading
                      ? "Processing…"
                      : `Pay ${money(currentWeeklyPremium)}`}
                  </button>
                </div>

                <div className={styles.modalDisclaimer}>
                  🔒 This is a mock payment. No real transaction will occur.
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
