"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import styles from "./page.module.css";

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

const TIERS: Tier[] = ["basic", "standard", "pro"];

const PLAN_DETAILS: Record<
  Tier,
  { label: string; tag: string; includes: string[] }
> = {
  basic: {
    label: "Basic Shield",
    tag: "Low-cost entry cover",
    includes: ["Accident support", "Daily income interruption support"],
  },
  standard: {
    label: "Standard Guard",
    tag: "Balanced weekly protection",
    includes: [
      "Accident support",
      "Income loss support",
      "Weather disruption support",
    ],
  },
  pro: {
    label: "Pro Protect",
    tag: "Highest payout priority",
    includes: [
      "Accident support",
      "Income loss support",
      "Weather disruption support",
      "Priority claims processing",
    ],
  },
};

const NAV = [
  { id: "home", label: "Dashboard" },
  { id: "profile", label: "Profile" },
];

export default function DashboardPage() {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [tab, setTab] = useState("home");
  const [currentPremium, setCurrentPremium] = useState<Premium | null>(null);
  const [quotes, setQuotes] = useState<QuoteMap>({});
  const [nextWeekTier, setNextWeekTier] = useState<Tier>("standard");
  const [loadingCurrent, setLoadingCurrent] = useState(false);
  const [loadingQuotes, setLoadingQuotes] = useState(false);
  const [premiumError, setPremiumError] = useState("");
  const [planMessage, setPlanMessage] = useState("");

  const refreshUserFromServer = async (token: string, fallback: User) => {
    try {
      const res = await fetch("/api/auth/me", {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) {
        if (res.status === 401) {
          logout();
        }
        return;
      }

      const data = await res.json();
      if (!data.user) return;

      setUser(data.user);
      localStorage.setItem("gg_user", JSON.stringify(data.user));

      // If admin changed user's tier externally, keep next week selection coherent.
      const savedPlan = localStorage.getItem(planStorageKey(data.user.id));
      const parsedTier = normalizeTier(savedPlan || data.user.tier);
      setNextWeekTier(parsedTier);
    } catch {
      // Keep using fallback local user when offline.
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

      const interval = setInterval(() => {
        refreshUserFromServer(token, parsed);
      }, 15000);

      return () => clearInterval(interval);
    } catch {
      router.replace("/login");
    }
  }, [router]);

  useEffect(() => {
    if (!user?.delivery_id) return;

    const savedPlan = localStorage.getItem(planStorageKey(user.id));
    const parsedTier = normalizeTier(savedPlan || user.tier);
    setNextWeekTier(parsedTier);

    fetchCurrentPremium(user);
    fetchTierQuotes(user);
  }, [user]);

  const logout = () => {
    localStorage.removeItem("gg_token");
    localStorage.removeItem("gg_user");
    router.push("/login");
  };

  const fetchCurrentPremium = async (targetUser: User) => {
    if (!targetUser.delivery_id) return;

    setLoadingCurrent(true);
    setPremiumError("");
    try {
      const data = await fetchPremiumForTier(
        targetUser,
        normalizeTier(targetUser.tier),
      );
      setCurrentPremium(data);
    } catch {
      setPremiumError("Could not fetch premium. Backend may be offline.");
    } finally {
      setLoadingCurrent(false);
    }
  };

  const fetchTierQuotes = async (targetUser: User) => {
    if (!targetUser.delivery_id) return;

    setLoadingQuotes(true);
    try {
      const results = await Promise.allSettled(
        TIERS.map(async (tier) => {
          const quote = await fetchPremiumForTier(targetUser, tier);
          return [tier, quote] as const;
        }),
      );

      const nextQuotes: QuoteMap = {};
      results.forEach((entry) => {
        if (entry.status === "fulfilled") {
          const [tier, quote] = entry.value;
          nextQuotes[tier] = quote;
        }
      });
      setQuotes(nextQuotes);
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

  return (
    <div className={styles.pageRoot}>
      <header className={styles.header}>
        <div className={styles.brandBlock}>
          <div className={styles.brandLogo}>GG</div>
          <span className={styles.brandText}>GigGuard</span>
        </div>
        <div className={styles.headerActions}>
          <div className={styles.userBadge}>
            <div className={styles.userInitial}>
              {user.name.charAt(0).toUpperCase()}
            </div>
            <span className={styles.userName}>{user.name}</span>
          </div>
          <button onClick={logout} className={styles.logoutBtn}>
            Logout
          </button>
        </div>
      </header>

      <div className={styles.layout}>
        <aside className={styles.sidebar}>
          {NAV.map((item) => (
            <button
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
                      okText="Planned and covered"
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
                    label="Current premium"
                    value={money(currentWeeklyPremium)}
                    tone="green"
                  />
                  <MetricCard
                    label="Current max payout"
                    value={money(currentPremium?.max_payout)}
                    tone="amber"
                  />
                  <MetricCard
                    label="City"
                    value={user.city || "-"}
                    tone="ink"
                  />
                </div>

                <section className={styles.panel}>
                  <div className={styles.panelHead}>
                    <div>
                      <h3 className={styles.panelTitle}>Coverage this week</h3>
                      <p className={styles.panelSub}>
                        Your active plan is {PLAN_DETAILS[currentTier].label}.
                        Coverage depends on verification and premium
                        computation.
                      </p>
                    </div>
                  </div>

                  {loadingCurrent ? (
                    <div className={styles.muted}>
                      Calculating current-week premium...
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
                        <strong>{currentPremium.weather.temperature}C</strong>
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

                <section className={styles.panel}>
                  <div className={styles.panelHead}>
                    <div>
                      <h3 className={styles.panelTitle}>
                        Insurance for next week
                      </h3>
                      <p className={styles.panelSub}>
                        {nextWindow.label} • select a plan now to schedule it
                        for next week.
                      </p>
                    </div>
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
                              ? "Calculating..."
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
                      <div className={styles.coverageLabel}>
                        Coverage state next week
                      </div>
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

                <section className={styles.panel}>
                  <h3 className={styles.panelTitle}>Connected platforms</h3>
                  <div className={styles.platformWrap}>
                    {user.platforms.map((p) => (
                      <span key={p} className={styles.platformChip}>
                        {PLATFORM_NAMES[p] || p}
                      </span>
                    ))}
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

            {tab === "profile" && (
              <div>
                <h1 className={styles.profileTitle}>Profile</h1>
                <div className={styles.profileGrid}>
                  <div className={styles.panel}>
                    <h3 className={styles.panelTitle}>Personal information</h3>
                    <ProfileRow label="Name" value={user.name} />
                    <ProfileRow label="Email" value={user.email} />
                    <ProfileRow label="Phone" value={user.phone || "-"} />
                    <ProfileRow label="City" value={user.city || "-"} />
                    <ProfileRow label="Area" value={user.area || "-"} />
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
                        .map((p) => PLATFORM_NAMES[p] || p)
                        .join(", ")}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

async function fetchPremiumForTier(user: User, tier: Tier): Promise<Premium> {
  const response = await fetch("/api/premium/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      delivery_id: user.delivery_id,
      city: user.city || "Unknown",
      tier,
    }),
  });
  const data = await response.json();
  if (!response.ok || data.error) {
    throw new Error(data.error || "Premium request failed");
  }
  return data as Premium;
}

function normalizeTier(tier: string | undefined): Tier {
  if (tier === "basic" || tier === "pro") return tier;
  return "standard";
}

function planStorageKey(userId: string): string {
  return `gg_next_week_plan_${userId}`;
}

function getNextWeekWindow(): { label: string } {
  const now = new Date();
  const dayFromMonday = (now.getDay() + 6) % 7;
  const daysUntilNextMonday = 7 - dayFromMonday;

  const from = new Date(now);
  from.setDate(now.getDate() + daysUntilNextMonday);

  const to = new Date(from);
  to.setDate(from.getDate() + 6);

  const format = new Intl.DateTimeFormat("en-IN", {
    day: "numeric",
    month: "short",
  });
  return { label: `${format.format(from)} - ${format.format(to)}` };
}

function money(value: number | undefined | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return `\u20B9${Math.round(value)}`;
}

function riskLabel(risk: number): string {
  if (risk < 0.15) return "Low";
  if (risk < 0.4) return "Moderate";
  if (risk < 0.7) return "High";
  return "Severe";
}

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

function ProfileRow({ label, value }: { label: string; value: string }) {
  return (
    <div className={styles.profileRow}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
