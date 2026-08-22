"use client";
import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import styles from "./page.module.css";
import { LANGUAGES, Language, makeT } from "@/lib/translations";

// ─── Platform logos ───────────────────────────────────────────────────────────
import {
  SwiggyIcon,
  ZomatoIcon,
  BlinkitIcon,
  ZeptoIcon,
  MeeshoIcon,
  DunzoIcon,
  PorterIcon,
} from "./PlatformIcons";

const PLATFORM_LOGOS: Partial<Record<string, React.ReactNode>> = {
  swiggy: <SwiggyIcon height={14} />,
  zomato: <ZomatoIcon height={14} />,
  blinkit: <BlinkitIcon height={14} />,
  zepto: <ZeptoIcon height={14} />,
  meesho: <MeeshoIcon height={12} />,
  dunzo: <DunzoIcon height={12} />,
  porter: <PorterIcon size={14} />,
};

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
  created_at?: string;
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

type TrafficData = {
  tti: number;
  current_speed_kmh: number;
  free_flow_speed_kmh: number;
  confidence: number;
  road_closure: boolean;
  traffic_risk: number;
  source: string;
};

type CurfewData = {
  confidence: number;
  gdelt_events: number;
  gdelt_score: number;
  nlp_score: number;
  nlp_label: string;
  fired: boolean;
  source: string;
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

type NotifType = "settled" | "rejected" | "review" | "premium" | "alert";

type Notif = {
  id: string;
  type: NotifType;
  title: string;
  message: string;
  time: string;
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
    tag: "Balanced Coverage",
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
  { id: "home", tKey: "nav_dashboard" },
  { id: "claims", tKey: "nav_claims" },
  { id: "payments", tKey: "nav_payments" },
  { id: "profile", tKey: "nav_profile" },
  { id: "simulator", tKey: "nav_simulator" },
];

const NAV_ICONS: Record<string, React.ReactNode> = {
  home: (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect x="3" y="3" width="7" height="7" rx="1.5" />
      <rect x="14" y="3" width="7" height="7" rx="1.5" />
      <rect x="3" y="14" width="7" height="7" rx="1.5" />
      <rect x="14" y="14" width="7" height="7" rx="1.5" />
    </svg>
  ),
  claims: (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M14 2H7a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7z" />
      <path d="M14 2v5h5" />
      <path d="m9 14 2 2 4-4" />
    </svg>
  ),
  payments: (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect x="2.5" y="5.5" width="19" height="13" rx="2.5" />
      <path d="M2.5 10h19" />
      <path d="M7 14h4" />
    </svg>
  ),
  profile: (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="12" cy="8" r="4" />
      <path d="M4 20a8 8 0 0 1 16 0" />
    </svg>
  ),
  simulator: (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M4 19h16" />
      <path d="m6 15 4-4 3 3 5-6" />
      <circle cx="6" cy="15" r="1" fill="currentColor" stroke="none" />
      <circle cx="10" cy="11" r="1" fill="currentColor" stroke="none" />
      <circle cx="13" cy="14" r="1" fill="currentColor" stroke="none" />
      <circle cx="18" cy="8" r="1" fill="currentColor" stroke="none" />
    </svg>
  ),
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

function normalizeTier(tier: string | undefined): Tier {
  if (tier === "basic" || tier === "pro") return tier;
  return "standard";
}

function planStorageKey(userId: string) {
  return `gg_next_week_plan_${userId}`;
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

function ttiLabel(tti: number): {
  text: string;
  level: "normal" | "moderate" | "severe";
} {
  if (tti < 1.5) return { text: "Free Flow", level: "normal" };
  if (tti < 2.0) return { text: "Light Congestion", level: "normal" };
  if (tti < 2.5) return { text: "Moderate Congestion", level: "moderate" };
  if (tti < 3.5) return { text: "Severe Congestion", level: "severe" };
  return { text: "Gridlock", level: "severe" };
}

function curfewLabel(confidence: number): string {
  if (confidence < 0.3) return "No signals";
  if (confidence < 0.6) return "Low activity";
  if (confidence < 0.8) return "Elevated risk";
  return "Active alert";
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

function payoutStatusValue(value: unknown): string {
  return String(value ?? "").toLowerCase();
}

function isSettledPayoutStatus(value: unknown): boolean {
  const status = payoutStatusValue(value);
  return status === "paid" || status === "approved";
}

function formatNotifTime(ts: string): string {
  const d = new Date(ts);
  if (isNaN(d.getTime())) return "";
  const diff = Date.now() - d.getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 7) return `${days}d ago`;
  return d.toLocaleDateString("en-IN", { day: "numeric", month: "short" });
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

// ─── Helpers ──────────────────────────────────────────────────────────────────

function greeting(): string {
  const h = new Date().getHours();
  if (h < 12) return "morning";
  if (h < 17) return "afternoon";
  return "evening";
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
  const [claimsFilter, setClaimsFilter] = useState<
    "all" | "paid" | "review" | "rejected"
  >("all");

  // traffic & curfew
  const [trafficData, setTrafficData] = useState<TrafficData | null>(null);
  const [curfewData, setCurfewData] = useState<CurfewData | null>(null);
  const [loadingTraffic, setLoadingTraffic] = useState(false);
  const [loadingCurfew, setLoadingCurfew] = useState(false);

  // chatbot
  type ChatMsg = { role: "user" | "assistant"; content: string };
  const [chatOpen, setChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMsg[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // language
  const [language, setLanguageState] = useState<Language>("en");

  useEffect(() => {
    const stored = localStorage.getItem("gg_language") as Language | null;
    if (stored && ["en", "hi", "te", "ta", "ml"].includes(stored)) {
      setLanguageState(stored);
    }
  }, []);

  const setLanguage = (lang: Language) => {
    setLanguageState(lang);
    localStorage.setItem("gg_language", lang);
  };

  const t = makeT(language);

  // theme
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  useEffect(() => {
    // Read from the DOM attribute (already set by the anti-flash script before
    // React hydrates) so state matches CSS on the very first render.
    const attr = document.documentElement.getAttribute("data-theme");
    if (attr === "light") setTheme("light");
  }, []);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    localStorage.setItem("gg_theme", next);
    if (next === "light") {
      document.documentElement.setAttribute("data-theme", "light");
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
  };

  // notifications
  const [dismissedIds, setDismissedIds] = useState<Set<string>>(new Set());
  const [showNotifPanel, setShowNotifPanel] = useState(false);
  const notifRef = useRef<HTMLDivElement>(null);
  const nextWeekSectionRef = useRef<HTMLElement>(null);
  const [pendingUpgradeScroll, setPendingUpgradeScroll] = useState(false);

  const scrollToNextWeekPlan = () => {
    if (tab !== "home") {
      setPendingUpgradeScroll(true);
      setTab("home");
      return;
    }
    nextWeekSectionRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  };

  useEffect(() => {
    try {
      const raw = localStorage.getItem("gg_dismissed_notifs");
      if (raw) setDismissedIds(new Set(JSON.parse(raw) as string[]));
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    if (!showNotifPanel) return;
    const handler = (e: MouseEvent) => {
      if (notifRef.current && !notifRef.current.contains(e.target as Node)) {
        setShowNotifPanel(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [showNotifPanel]);

  useEffect(() => {
    if (!pendingUpgradeScroll || tab !== "home") return;
    const rafId = window.requestAnimationFrame(() => {
      nextWeekSectionRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
      setPendingUpgradeScroll(false);
    });
    return () => window.cancelAnimationFrame(rafId);
  }, [pendingUpgradeScroll, tab]);

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
  const [showSupportModal, setShowSupportModal] = useState(false);
  const [supportMode, setSupportMode] = useState<
    "support" | "claim_escalation"
  >("support");
  const [supportSubject, setSupportSubject] = useState("");
  const [supportMessage, setSupportMessage] = useState("");
  const [supportClaimContext, setSupportClaimContext] = useState<{
    id: string;
    claimNumber: string;
    triggerType: string;
  } | null>(null);
  const [supportLoading, setSupportLoading] = useState(false);
  const [supportError, setSupportError] = useState("");
  const [supportSuccess, setSupportSuccess] = useState("");

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
    // Phase 3: fetch traffic and curfew data for user's city
    if (user.city) {
      fetchTraffic(user.city);
      fetchCurfew(user.city);
    }
    fetchPayments();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.id, user?.tier]);

  // Re-fetch claims once created_at becomes available (first server refresh populates it).
  // This ensures the backend date filter is applied even when localStorage lacked created_at.
  useEffect(() => {
    if (user?.delivery_id && user?.created_at) {
      fetchClaims(user);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.created_at]);

  const fetchClaims = async (u: User) => {
    if (!u.delivery_id) return;
    setLoadingClaims(true);
    try {
      const token = localStorage.getItem("gg_token");
      // Use the dedicated route that fetches the registration date from Supabase
      // server-side — no dependency on client-side user.created_at being populated.
      const res = await fetch(`/api/claims/worker/${u.delivery_id}`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      const data = await res.json();
      if (data?.data) setClaims(data.data);
    } catch {
      /* silently fail */
    } finally {
      setLoadingClaims(false);
    }
  };

  const fetchTraffic = async (city: string) => {
    setLoadingTraffic(true);
    try {
      const res = await fetch(
        `/api/backend/traffic/${encodeURIComponent(city)}`,
      );
      const data = await res.json();
      if (data?.traffic) setTrafficData(data.traffic);
    } catch {
      /* silently fail */
    } finally {
      setLoadingTraffic(false);
    }
  };

  const fetchCurfew = async (city: string) => {
    setLoadingCurfew(true);
    try {
      const res = await fetch(
        `/api/backend/curfew/${encodeURIComponent(city)}`,
      );
      const data = await res.json();
      if (data?.curfew) setCurfewData(data.curfew);
    } catch {
      /* silently fail */
    } finally {
      setLoadingCurfew(false);
    }
  };

  const fetchPayments = async () => {
    const token = localStorage.getItem("gg_token");
    if (!token) return;
    try {
      const res = await fetch("/api/payment/history", {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) return;
      const data = await res.json();
      if (data?.payments) {
        setPayments(
          // server returns transaction_id; map to id for table key/display
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          data.payments.map((p: any) => ({ ...p, id: p.transaction_id })),
        );
      }
    } catch {
      /* silently fail */
    }
  };

  const sendChat = async () => {
    const text = chatInput.trim();
    if (!text || chatLoading) return;
    const userMsg: ChatMsg = { role: "user", content: text };
    const updated = [...chatMessages, userMsg];
    setChatMessages(updated);
    setChatInput("");
    setChatLoading(true);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: updated }),
      });
      const data = await res.json();
      const reply = data.reply || "Sorry, I couldn't get a response right now.";
      setChatMessages((prev) => [...prev, { role: "assistant", content: reply }]);
    } catch {
      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Something went wrong. Please try again." },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  // Auto-scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);


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

  const openSupportModal = (
    mode: "support" | "claim_escalation",
    claim?: Record<string, unknown>,
  ) => {
    const claimId = claim ? String(claim.id ?? "") : "";
    const claimNumber = claim
      ? String(claim.claim_number ?? claim.id ?? "")
      : "";
    const triggerType = claim ? String(claim.trigger_type ?? "") : "";

    setSupportMode(mode);
    setSupportClaimContext(
      mode === "claim_escalation"
        ? {
            id: claimId,
            claimNumber,
            triggerType,
          }
        : null,
    );
    setSupportSubject(
      mode === "claim_escalation"
        ? claimNumber
          ? `Escalation for claim ${claimNumber}`
          : "Claim escalation"
        : "Support request",
    );
    setSupportMessage("");
    setSupportError("");
    setSupportSuccess("");
    setShowSupportModal(true);
  };

  const closeSupportModal = () => {
    if (supportLoading) return;
    setShowSupportModal(false);
    setSupportError("");
    setSupportSuccess("");
  };

  const handleSubmitSupportTicket = async () => {
    if (!user) return;
    setSupportError("");
    setSupportSuccess("");

    if (!supportSubject.trim()) {
      setSupportError("Please add a subject for your issue.");
      return;
    }
    if (!supportMessage.trim()) {
      setSupportError("Please describe the issue before submitting.");
      return;
    }

    setSupportLoading(true);
    try {
      const token = localStorage.getItem("gg_token");
      const res = await fetch("/api/support/tickets", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          ticket_type: supportMode,
          subject: supportSubject.trim(),
          message: supportMessage.trim(),
          source_tab: tab,
          worker_id: user.id,
          delivery_id: user.delivery_id,
          claim_id: supportClaimContext?.id || null,
          claim_number: supportClaimContext?.claimNumber || null,
          claim_trigger_type: supportClaimContext?.triggerType || null,
        }),
      });

      const data = await res.json();
      if (!res.ok) {
        setSupportError(data.error || "Failed to submit ticket.");
        return;
      }

      setSupportSuccess(
        "Issue submitted successfully. Our team will review it.",
      );
      setSupportMessage("");
    } catch {
      setSupportError("Unable to submit right now. Please try again.");
    } finally {
      setSupportLoading(false);
    }
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
      await fetchPayments();
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

  // Compare date strings only (YYYY-MM-DD) to avoid same-day time edge cases.
  const registrationDateStr = user?.created_at ? user.created_at.slice(0, 10) : null;
  const claimsAfterRegistration = claims.filter((c) => {
    if (!registrationDateStr || !c.created_at) return true;
    return String(c.created_at).slice(0, 10) >= registrationDateStr;
  });
  const claimsSettled = claimsAfterRegistration.filter((c) =>
    isSettledPayoutStatus(c.payout_status),
  );
  const claimsUnderReview = claimsAfterRegistration.filter(
    (c) =>
      !isSettledPayoutStatus(c.payout_status) &&
      payoutStatusValue(c.payout_status) !== "rejected",
  );
  const claimsRejected = claimsAfterRegistration.filter(
    (c) => payoutStatusValue(c.payout_status) === "rejected",
  );
  const claimsTotalReceived = claimsSettled.reduce(
    (sum, c) => sum + (Number(c.payout_amount) || 0),
    0,
  );
  const claimsFiltered =
    claimsFilter === "all"
      ? claimsAfterRegistration
      : claimsFilter === "paid"
        ? claimsSettled
        : claimsFilter === "review"
          ? claimsUnderReview
          : claimsRejected;

  // ── Notifications ──────────────────────────────────────────────────────────
  const allNotifs: Notif[] = [];

  // Welcome notification — only for accounts registered in the last 7 days
  if (user.created_at) {
    const daysSinceReg = (Date.now() - new Date(user.created_at).getTime()) / (1000 * 60 * 60 * 24);
    if (daysSinceReg <= 7) {
      allNotifs.push({
        id: `welcome_${user.id}`,
        type: "alert",
        title: "Welcome to WPIP!",
        message: `Your coverage is now active. Claims are filed automatically whenever a disruption is detected in ${user.city}.`,
        time: user.created_at,
      });
    }
  }

  // Premium due
  if (!paidToday && currentWeeklyPremium) {
    allNotifs.push({
      id: `premium_due_${user.id}`,
      type: "premium",
      title: "Premium Due",
      message: `Your weekly premium of ${money(currentWeeklyPremium)} is due. Pay now to keep coverage active.`,
      time: new Date().toISOString(),
    });
  }

  // Curfew / unrest alert
  if (curfewData?.fired) {
    allNotifs.push({
      id: `curfew_${user.city}_${new Date().toDateString()}`,
      type: "alert",
      title: "Curfew Trigger Active",
      message: `T-06 unrest trigger is active in ${user.city}. Your coverage is protecting you.`,
      time: new Date().toISOString(),
    });
  }

  // From claims (only post-registration, most recent first, max 8)
  claimsAfterRegistration.slice(0, 8).forEach((c) => {
    const id = `claim_${String(c.id)}`;
    const triggerName = String(c.trigger_type ?? "disruption").replace(
      /_/g,
      " ",
    );
    if (isSettledPayoutStatus(c.payout_status)) {
      const settledStatus = payoutStatusValue(c.payout_status);
      allNotifs.push({
        id,
        type: "settled",
        title: "Claim Settled",
        message:
          settledStatus === "approved"
            ? `₹${c.payout_amount ?? 0} approved for ${triggerName}.`
            : `₹${c.payout_amount ?? 0} payout processed for ${triggerName}.`,
        time: String(c.created_at ?? ""),
      });
    } else if (payoutStatusValue(c.payout_status) === "rejected") {
      allNotifs.push({
        id,
        type: "rejected",
        title: "Claim Rejected",
        message: `Your ${triggerName} claim did not pass automated checks.`,
        time: String(c.created_at ?? ""),
      });
    } else {
      allNotifs.push({
        id,
        type: "review",
        title: "Claim Filed",
        message: `${triggerName} claim auto-filed and is under review.`,
        time: String(c.created_at ?? ""),
      });
    }
  });

  const activeNotifs = allNotifs.filter((n) => !dismissedIds.has(n.id));
  const unreadCount = activeNotifs.length;

  const dismissNotif = (id: string) => {
    const next = new Set([...dismissedIds, id]);
    setDismissedIds(next);
    localStorage.setItem("gg_dismissed_notifs", JSON.stringify([...next]));
  };

  const dismissAll = () => {
    const next = new Set(allNotifs.map((n) => n.id));
    setDismissedIds(next);
    localStorage.setItem("gg_dismissed_notifs", JSON.stringify([...next]));
  };

  const NOTIF_ICONS: Record<NotifType, string> = {
    settled: "✓",
    rejected: "✕",
    review: "⏳",
    premium: "₹",
    alert: "⚡",
  };

  return (
    <div className={styles.pageRoot}>
      {/* ── Header ── */}
      <header className={styles.header}>
        <div className={styles.headerMain}>
          <div className={styles.brandBlock}>
            <div className={styles.brandLogo}>GG</div>
            <span className={styles.brandText}>WPIP</span>
          </div>
          <label className={styles.headerSearch} style={{visibility: 'hidden'}}>
            <svg
              className={styles.headerSearchIcon}
              width="15"
              height="15"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="11" cy="11" r="7" />
              <line x1="21" y1="21" x2="16.65" y2="16.65" />
            </svg>
            <input
              className={styles.headerSearchInput}
              type="text"
              placeholder="Search claims, payouts, and risk alerts..."
              aria-label="Search dashboard"
            />
          </label>
        </div>

        <div className={styles.headerActions}>
          {/* ── Notification bell ── */}
          <div className={styles.notifWrapper} ref={notifRef}>
            <button
              type="button"
              className={styles.notifBtn}
              onClick={() => setShowNotifPanel((v) => !v)}
              aria-label="Notifications"
            >
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
                <path d="M13.73 21a2 2 0 0 1-3.46 0" />
              </svg>
              {unreadCount > 0 && (
                <span className={styles.notifBadge}>
                  {unreadCount > 9 ? "9+" : unreadCount}
                </span>
              )}
            </button>

            {showNotifPanel && (
              <div className={styles.notifPanel}>
                <div className={styles.notifPanelHead}>
                  <span className={styles.notifPanelTitle}>
                    Notifications
                    {unreadCount > 0 && (
                      <span className={styles.notifPanelCount}>
                        {unreadCount}
                      </span>
                    )}
                  </span>
                  {activeNotifs.length > 0 && (
                    <button
                      type="button"
                      className={styles.notifMarkAll}
                      onClick={dismissAll}
                    >
                      Mark all read
                    </button>
                  )}
                </div>

                {activeNotifs.length === 0 ? (
                  <div className={styles.notifEmpty}>
                    <svg
                      width="32"
                      height="32"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
                      <path d="M13.73 21a2 2 0 0 1-3.46 0" />
                    </svg>
                    <p>All caught up</p>
                    <span>No new notifications</span>
                  </div>
                ) : (
                  <div className={styles.notifList}>
                    {activeNotifs.map((n) => (
                      <div key={n.id} className={styles.notifItem}>
                        <div
                          className={`${styles.notifIcon} ${styles[`notifIcon_${n.type}`]}`}
                        >
                          {NOTIF_ICONS[n.type]}
                        </div>
                        <div className={styles.notifBody}>
                          <div className={styles.notifItemTitle}>{n.title}</div>
                          <div className={styles.notifItemMsg}>{n.message}</div>
                          <div className={styles.notifItemTime}>
                            {formatNotifTime(n.time)}
                          </div>
                        </div>
                        <button
                          type="button"
                          className={styles.notifDismiss}
                          onClick={() => dismissNotif(n.id)}
                          aria-label="Dismiss"
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          <button
            type="button"
            onClick={toggleTheme}
            className={`${styles.themeBtn} ${theme === "light" ? styles.themeBtnLight : ""}`}
            aria-label={
              theme === "dark" ? "Switch to light mode" : "Switch to dark mode"
            }
          >
            {theme === "dark" ? (
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="12" cy="12" r="4" />
                <line x1="12" y1="2" x2="12" y2="5" />
                <line x1="12" y1="19" x2="12" y2="22" />
                <line x1="4.93" y1="4.93" x2="7.05" y2="7.05" />
                <line x1="16.95" y1="16.95" x2="19.07" y2="19.07" />
                <line x1="2" y1="12" x2="5" y2="12" />
                <line x1="19" y1="12" x2="22" y2="12" />
                <line x1="4.93" y1="19.07" x2="7.05" y2="16.95" />
                <line x1="16.95" y1="7.05" x2="19.07" y2="4.93" />
              </svg>
            ) : (
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
              </svg>
            )}
            {theme === "dark" ? "Light" : "Dark"}
          </button>
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
        {/* ── Left Sidebar ── */}
        <aside className={styles.sidebar}>
          <div className={styles.sidebarBrand}>
            <h2 className={styles.sidebarBrandTitle}>WPIP</h2>
            <p className={styles.sidebarBrandTag}>Worker Safety Command</p>
          </div>

          <nav className={styles.sidebarNav}>
            {NAV.map((item) => {
              const isActive = tab === item.id;
              return (
                <button
                  type="button"
                  key={item.id}
                  onClick={() => {
                    if (item.id === "simulator") {
                      router.push("/simulator");
                      return;
                    }
                    setTab(item.id);
                  }}
                  className={`${styles.navBtn} ${isActive ? styles.navBtnActive : ""}`}
                >
                  <span className={styles.navBtnIcon}>
                    {NAV_ICONS[item.id] ?? (
                      <svg
                        width="14"
                        height="14"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        aria-hidden="true"
                      >
                        <circle cx="12" cy="12" r="4" />
                      </svg>
                    )}
                  </span>
                  <span>{t(item.tKey)}</span>
                </button>
              );
            })}
          </nav>

          <div className={styles.sidebarBottom}>
            <button
              type="button"
              className={styles.upgradeBtn}
              onClick={scrollToNextWeekPlan}
            >
              Upgrade Protection
            </button>

            <div className={styles.sidebarMeta}>
              <button
                type="button"
                className={styles.sidebarMetaLink}
                onClick={() => setTab("profile")}
              >
                Settings
              </button>
              <button
                type="button"
                className={styles.sidebarMetaLink}
                onClick={() => openSupportModal("support")}
              >
                Support
              </button>
            </div>
          </div>
        </aside>

        <main className={styles.main}>
          <div>
            {/* ══ HOME TAB ══ */}
            {tab === "home" && (
              <div className={styles.homeGrid}>
                <div className={styles.homeMain}>
                  {/* Greeting */}
                  <div className={styles.pageGreeting}>
                    <h1 className={styles.greetTitle}>
                      {t(`greeting_${greeting()}`)}, {user.name.split(" ")[0]}.
                    </h1>
                    <p className={styles.greetSub}>
                      {coveredNow ? t("coverage_on") : t("coverage_off")} ·{" "}
                      {user.city}
                    </p>
                  </div>

                  <div className={styles.grid4}>
                    <MetricCard
                      label={t("current_plan")}
                      value={tierLabel}
                      tone="blue"
                    />
                    <MetricCard
                      label={t("weekly_premium")}
                      value={money(currentWeeklyPremium)}
                      tone="green"
                    />
                    <MetricCard
                      label={t("max_payout")}
                      value={money(currentPremium?.max_payout)}
                      tone="amber"
                    />
                    <MetricCard
                      label={t("city")}
                      value={user.city || "–"}
                      tone="ink"
                    />
                  </div>

                  {/* Coverage this week */}
                  <section className={styles.panel}>
                    <div className={styles.panelHead}>
                      <h3 className={styles.panelTitle}>
                        {t("cover_this_week")}
                      </h3>
                      <p className={styles.panelSub}>
                        {t("active_plan_prefix")}:{" "}
                        {PLAN_DETAILS[currentTier].label}.{" "}
                        {t("coverage_needs_note")}
                      </p>
                    </div>

                    {loadingCurrent ? (
                      <div className={styles.muted}>{t("loading_premium")}</div>
                    ) : premiumError ? (
                      <div className={styles.errorText}>{premiumError}</div>
                    ) : currentPremium ? (
                      <div className={styles.coverageGrid}>
                        <div className={styles.coverageCard}>
                          <div className={styles.coverageLabel}>
                            {t("status")}
                          </div>
                          <div className={styles.coverageValue}>
                            {coveredNow
                              ? t("active_coverage")
                              : t("coverage_unavailable")}
                          </div>
                        </div>
                        <div className={styles.coverageCard}>
                          <div className={styles.coverageLabel}>
                            {t("weekly_premium_label")}
                          </div>
                          <div className={styles.coverageValue}>
                            {money(currentWeeklyPremium)}
                          </div>
                        </div>
                        <div className={styles.coverageCard}>
                          <div className={styles.coverageLabel}>
                            {t("max_payout_per_claim")}
                          </div>
                          <div className={styles.coverageValue}>
                            {money(currentPremium.max_payout)}
                          </div>
                        </div>
                        <div className={styles.coverageCard}>
                          <div className={styles.coverageLabel}>
                            {t("risk_level")}
                          </div>
                          <div className={styles.coverageValue}>
                            {riskLabel(currentPremium.weather_risk)}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className={styles.muted}>{t("no_plan_data")}</div>
                    )}

                    {currentPremium?.weather && (
                      <div className={styles.weatherStrip}>
                        <span>
                          Temp:{" "}
                          <strong>
                            {currentPremium.weather.temperature}°C
                          </strong>
                        </span>
                        <span>
                          AQI:{" "}
                          <strong>{currentPremium.weather.aqi_index}</strong>
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

                    {/* Traffic strip */}
                    {loadingTraffic ? (
                      <div className={styles.stripMuted}>
                        Loading traffic data...
                      </div>
                    ) : trafficData ? (
                      <div className={styles.trafficStrip}>
                        <span>
                          TTI: <strong>{trafficData.tti.toFixed(2)}</strong>{" "}
                          <span
                            className={`${styles.ttiBadge} ${styles[`ttiBadge${ttiLabel(trafficData.tti).level.charAt(0).toUpperCase() + ttiLabel(trafficData.tti).level.slice(1)}`]}`}
                          >
                            {ttiLabel(trafficData.tti).text}
                          </span>
                        </span>
                        <span>
                          Speed:{" "}
                          <strong>{trafficData.current_speed_kmh} km/h</strong>
                        </span>
                        <span>
                          Free flow:{" "}
                          <strong>
                            {trafficData.free_flow_speed_kmh} km/h
                          </strong>
                        </span>
                        <span>
                          Traffic risk:{" "}
                          <strong>
                            {(trafficData.traffic_risk * 100).toFixed(0)}%
                          </strong>
                        </span>
                        {trafficData.road_closure && (
                          <span>
                            <strong className={styles.roadClosureAlert}>
                              Road Closure Detected
                            </strong>
                          </span>
                        )}
                        <span className={styles.stripSource}>
                          Source: {trafficData.source}
                        </span>
                      </div>
                    ) : null}

                    {/* Curfew / unrest strip */}
                    {loadingCurfew ? (
                      <div className={styles.stripMuted}>
                        Loading curfew data...
                      </div>
                    ) : curfewData ? (
                      <div className={styles.curfewStrip}>
                        <span>
                          Unrest status:{" "}
                          <strong>{curfewLabel(curfewData.confidence)}</strong>
                        </span>
                        <span>
                          Confidence:{" "}
                          <strong>
                            {(curfewData.confidence * 100).toFixed(0)}%
                          </strong>
                        </span>
                        <span>
                          GDELT events:{" "}
                          <strong>{curfewData.gdelt_events}</strong>
                        </span>
                        {curfewData.nlp_score > 0 && (
                          <span>
                            NLP match:{" "}
                            <strong>
                              {curfewData.nlp_label} (
                              {(curfewData.nlp_score * 100).toFixed(0)}%)
                            </strong>
                          </span>
                        )}
                        {curfewData.fired && (
                          <span>
                            <strong className={styles.roadClosureAlert}>
                              T-06 Trigger Active
                            </strong>
                          </span>
                        )}
                        <span className={styles.stripSource}>
                          Source: {curfewData.source}
                        </span>
                      </div>
                    ) : null}

                    <ul className={styles.coverList}>
                      {PLAN_DETAILS[currentTier].includes.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </section>

                  {/* Next week plan */}
                  <section
                    ref={nextWeekSectionRef}
                    className={`${styles.panel} ${styles.nextWeekSection}`}
                  >
                    <div className={styles.panelHead}>
                      <h3 className={styles.panelTitle}>
                        {t("insurance_next_week")}
                      </h3>
                      <p className={styles.panelSub}>
                        {nextWindow.label} · {t("select_plan_note")}
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
                              {t("max_payout")}: {money(quote?.max_payout)}
                            </div>
                            <button
                              type="button"
                              className={
                                selected
                                  ? styles.planBtnSelected
                                  : styles.planBtn
                              }
                              onClick={() => applyNextWeekPlan(tier)}
                            >
                              {selected
                                ? t("scheduled_next_week")
                                : t("choose_next_week")}
                            </button>
                          </div>
                        );
                      })}
                    </div>

                    <div className={styles.nextWeekSummary}>
                      <div>
                        <div className={styles.coverageLabel}>
                          {t("scheduled_plan")}
                        </div>
                        <div className={styles.coverageValue}>
                          {PLAN_DETAILS[nextWeekTier].label}
                        </div>
                      </div>
                      <div>
                        <div className={styles.coverageLabel}>
                          {t("projected_premium")}
                        </div>
                        <div className={styles.coverageValue}>
                          {money(nextWeeklyPremium)}
                        </div>
                      </div>
                      <div>
                        <div className={styles.coverageLabel}>
                          {t("coverage_state")}
                        </div>
                        <div className={styles.coverageValue}>
                          {coveredNextWeek
                            ? t("will_be_covered")
                            : t("not_covered_yet")}
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
                    <h3 className={styles.panelTitle}>
                      {t("connected_platforms")}
                    </h3>
                    <div className={styles.platformWrap}>
                      {user.platforms.map((p) => {
                        const meta = PLATFORM_META[p] ?? {
                          name: p,
                          symbol: p.slice(0, 2).toUpperCase(),
                          iconClass: "platformIconDefault",
                        };
                        const logo = PLATFORM_LOGOS[p];
                        return (
                          <span key={p} className={styles.platformChip}>
                            {logo ? (
                              <span className={styles.platformChipLogo}>
                                {logo}
                              </span>
                            ) : (
                              <span
                                className={`${styles.platformChipIcon} ${styles[meta.iconClass]}`}
                              >
                                {meta.symbol}
                              </span>
                            )}
                            {meta.name}
                          </span>
                        );
                      })}
                    </div>
                  </section>

                  {!verified && (
                    <div className={styles.warningBox}>
                      {t("verification_warning")}
                    </div>
                  )}
                </div>

                {/* ── Right aside ── */}
                <aside className={styles.homeAside}>
                  <div className={styles.asideCard}>
                    <div className={styles.asideCardLabel}>
                      {t("upcoming_premium")}
                    </div>
                    <div className={styles.asideCardAmount}>
                      {money(currentWeeklyPremium)}
                    </div>
                    <div className={styles.asideCardMeta}>
                      {tierLabel} · {t("per_week")}
                    </div>
                    <div className={styles.asideCardMeta}>{user.city}</div>
                    {paidToday ? (
                      <div className={styles.asidePaidBadge}>
                        {t("paid_today")}
                      </div>
                    ) : (
                      <button
                        type="button"
                        className={styles.asidePayBtn}
                        onClick={openPayModal}
                        disabled={!currentPremium}
                      >
                        {t("pay_premium")}
                      </button>
                    )}
                  </div>

                  <div className={styles.asideCard}>
                    <div className={styles.asideCardLabel}>
                      {t("quick_actions")}
                    </div>
                    <ul className={styles.quickActions}>
                      <li>
                        <button
                          type="button"
                          className={styles.quickActionItem}
                          onClick={() => setTab("payments")}
                        >
                          {t("pay_this_weeks_premium")}
                        </button>
                      </li>
                      <li>
                        <button
                          type="button"
                          className={styles.quickActionItem}
                          onClick={() => setTab("claims")}
                        >
                          {t("view_claims_history")}
                        </button>
                      </li>
                      <li>
                        <button
                          type="button"
                          className={styles.quickActionItem}
                          onClick={() => setTab("profile")}
                        >
                          {t("update_profile")}
                        </button>
                      </li>
                    </ul>
                  </div>

                  {currentPremium?.weather && (
                    <div
                      className={`${styles.asideCard} ${styles.asideRiskCard}`}
                    >
                      <div className={styles.asideCardLabel}>
                        {t("weather_risk")}
                      </div>
                      <div className={styles.riskBadge}>
                        {riskLabel(currentPremium.weather_risk)} Risk
                      </div>
                      <div className={styles.riskDetails}>
                        <span>{currentPremium.weather.weather_main}</span>
                        <span>{currentPremium.weather.temperature}°C</span>
                        <span>AQI {currentPremium.weather.aqi_index}</span>
                        <span>Humidity {currentPremium.weather.humidity}%</span>
                      </div>
                    </div>
                  )}
                </aside>
              </div>
            )}

            {/* ══ CLAIMS TAB ══ */}
            {tab === "claims" && (
              <div>
                <h1 className={styles.profileTitle}>{t("claims_history")}</h1>

                {!loadingClaims && (
                  <>
                    {/* ── Hero dashboard card ── */}
                    <div className={styles.claimsHeroCard}>
                      <div className={styles.claimsHeroGlow} />
                      <div className={styles.claimsHeroTop}>
                        <div>
                          <div className={styles.claimsHeroEyebrow}>
                            {t("total_received")}
                          </div>
                          <div className={styles.claimsHeroAmount}>
                            ₹
                            {Math.round(claimsTotalReceived).toLocaleString(
                              "en-IN",
                            )}
                          </div>
                        </div>
                        <div className={styles.claimsHeroBadge}>
                          <div className={styles.claimsHeroBadgeNum}>
                            {claimsAfterRegistration.length}
                          </div>
                          <div className={styles.claimsHeroBadgeLabel}>
                            {t("claims_badge")}
                          </div>
                        </div>
                      </div>
                      <div className={styles.claimsHeroStats}>
                        <div className={styles.claimsHeroStat}>
                          <div
                            className={`${styles.claimsHeroStatDot} ${styles.dotGreen}`}
                          />
                          <span className={styles.claimsHeroStatNum}>
                            {claimsSettled.length}
                          </span>
                          <span className={styles.claimsHeroStatLabel}>
                            {t("settled")}
                          </span>
                        </div>
                        <div className={styles.claimsHeroStatSep} />
                        <div className={styles.claimsHeroStat}>
                          <div
                            className={`${styles.claimsHeroStatDot} ${styles.dotAmber}`}
                          />
                          <span className={styles.claimsHeroStatNum}>
                            {claimsUnderReview.length}
                          </span>
                          <span className={styles.claimsHeroStatLabel}>
                            {t("review")}
                          </span>
                        </div>
                        <div className={styles.claimsHeroStatSep} />
                        <div className={styles.claimsHeroStat}>
                          <div
                            className={`${styles.claimsHeroStatDot} ${styles.dotRed}`}
                          />
                          <span className={styles.claimsHeroStatNum}>
                            {claimsRejected.length}
                          </span>
                          <span className={styles.claimsHeroStatLabel}>
                            {t("rejected")}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* ── Auto-claim note ── */}
                    <div className={styles.claimsNoteRow}>
                      <svg
                        width="14"
                        height="14"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        className={styles.claimsNoteIcon}
                      >
                        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                        <polyline points="9 12 11 14 15 10" />
                      </svg>
                      <span className={styles.claimsNoteText}>
                        {t("auto_claim_note")}
                      </span>
                    </div>

                    {/* ── Filter tabs ── */}
                    <div className={styles.claimsFilterTabs}>
                      {(
                        [
                          { key: "all", tKey: "all", count: claimsAfterRegistration.length },
                          {
                            key: "paid",
                            tKey: "settled",
                            count: claimsSettled.length,
                          },
                          {
                            key: "review",
                            tKey: "review",
                            count: claimsUnderReview.length,
                          },
                          {
                            key: "rejected",
                            tKey: "rejected",
                            count: claimsRejected.length,
                          },
                        ] as const
                      ).map((opt) => (
                        <button
                          key={opt.key}
                          type="button"
                          className={`${styles.claimsFilterTab} ${claimsFilter === opt.key ? styles.claimsFilterTabActive : ""}`}
                          onClick={() => setClaimsFilter(opt.key)}
                        >
                          {t(opt.tKey)}
                          <span
                            className={`${styles.claimsFilterBadge} ${claimsFilter === opt.key ? styles.claimsFilterBadgeActive : ""}`}
                          >
                            {opt.count}
                          </span>
                        </button>
                      ))}
                    </div>
                  </>
                )}

                {loadingClaims ? (
                  <div className={styles.muted}>{t("loading_claims")}</div>
                ) : claimsFiltered.length === 0 ? (
                  <div className={styles.emptyState}>
                    {claimsFilter === "all"
                      ? "No claims yet. Claims appear automatically when a disruption event is detected in your city."
                      : `No ${claimsFilter === "paid" ? "settled" : claimsFilter === "review" ? "under review" : "rejected"} claims found.`}
                  </div>
                ) : (
                  <div className={styles.claimsList}>
                    {claimsFiltered.map((c) => {
                      const triggerType = String(c.trigger_type ?? "");
                      const triggerLabel = triggerType
                        .replace(/_/g, " ")
                        .replace(/\b\w/g, (ch) => ch.toUpperCase());
                      const settled = isSettledPayoutStatus(c.payout_status);
                      const rejected =
                        payoutStatusValue(c.payout_status) === "rejected";
                      const statusKey = settled
                        ? "settled"
                        : rejected
                          ? "rejected"
                          : "review";
                      const disrupted = c.disrupted_hours
                        ? `${c.disrupted_hours} hrs`
                        : "—";
                      const planLabel = user?.tier
                        ? String(user.tier).charAt(0).toUpperCase() +
                          String(user.tier).slice(1)
                        : "—";
                      const dateStr = c.created_at
                        ? new Date(String(c.created_at)).toLocaleDateString(
                            "en-IN",
                            { day: "numeric", month: "short", year: "2-digit" },
                          )
                        : "—";

                      // Trigger icon + colour
                      const triggerTheme: Record<
                        string,
                        { icon: string; bg: string; color: string }
                      > = {
                        heavy_rain: {
                          icon: "🌧️",
                          bg: "rgba(59,130,246,0.10)",
                          color: "#2563eb",
                        },
                        flood: {
                          icon: "🌊",
                          bg: "rgba(6,182,212,0.10)",
                          color: "#0891b2",
                        },
                        extreme_heat: {
                          icon: "🌡️",
                          bg: "rgba(239,68,68,0.10)",
                          color: "#dc2626",
                        },
                        severe_aqi: {
                          icon: "💨",
                          bg: "rgba(100,116,139,0.10)",
                          color: "#475569",
                        },
                        traffic_congestion: {
                          icon: "🚦",
                          bg: "rgba(245,158,11,0.10)",
                          color: "#d97706",
                        },
                        curfew: {
                          icon: "🔒",
                          bg: "rgba(139,92,246,0.10)",
                          color: "#7c3aed",
                        },
                      };
                      const theme = triggerTheme[triggerType] ?? {
                        icon: "⚡",
                        bg: "rgba(100,116,139,0.08)",
                        color: "#64748b",
                      };

                      return (
                        <div
                          key={String(c.id)}
                          className={styles.claimCard}
                          data-status={statusKey}
                        >
                          {/* ── Header ── */}
                          <div className={styles.claimCardHeader}>
                            <div
                              className={styles.claimTriggerIcon}
                              style={{
                                background: theme.bg,
                              }}
                            >
                              {theme.icon}
                            </div>

                            <div className={styles.claimCardMeta}>
                              <p className={styles.claimCardTitle}>
                                {triggerLabel}
                              </p>
                              <div className={styles.claimCardSub}>
                                <span>
                                  {String(c.city ?? "—")} · {dateStr}
                                </span>
                                <span
                                  className={`${styles.claimStatusDot} ${
                                    settled
                                      ? styles.claimStatusDotGreen
                                      : rejected
                                        ? styles.claimStatusDotRed
                                        : styles.claimStatusDotAmber
                                  }`}
                                >
                                  {settled
                                    ? "Settled"
                                    : rejected
                                      ? "Rejected"
                                      : "Processing"}
                                </span>
                              </div>
                            </div>

                            <div className={styles.claimCardAmount}>
                              ₹
                              {Number(c.payout_amount ?? 0).toLocaleString(
                                "en-IN",
                              )}
                            </div>
                          </div>

                          {/* ── Stats grid (disrupted + plan) ── */}
                          <div className={styles.claimStats}>
                            <div className={styles.claimStatCell}>
                              <span className={styles.claimStatLabel}>
                                Disrupted
                              </span>
                              <span className={styles.claimStatValue}>
                                {disrupted}
                              </span>
                            </div>
                            <div className={styles.claimStatDivider} />
                            <div className={styles.claimStatCell}>
                              <span className={styles.claimStatLabel}>
                                Plan
                              </span>
                              <span className={styles.claimStatValue}>
                                {planLabel}
                              </span>
                            </div>
                          </div>

                          {/* ── Claim ID — full width, never truncates ── */}
                          <div className={styles.claimIdRow}>
                            <span className={styles.claimIdLabel}>
                              Claim ID
                            </span>
                            <span className={styles.claimIdValue}>
                              {String(c.claim_number ?? c.id ?? "—")}
                            </span>
                          </div>

                          {/* ── Transaction ID (only when paid) ── */}
                          {settled &&
                            payoutStatusValue(c.payout_status) === "paid" &&
                            !!c.transaction_id && (
                              <div className={styles.claimTxnRow}>
                                <span className={styles.claimTxnLabel}>
                                  Transaction ID
                                </span>
                                <span className={styles.claimTxnValue}>
                                  {String(c.transaction_id)}
                                </span>
                              </div>
                            )}

                          {/* ── Status message ── */}
                          {!settled && !rejected && (
                            <div
                              className={`${styles.claimStatusMsg} ${styles.claimStatusMsgAmber}`}
                            >
                              <svg
                                width="13"
                                height="13"
                                viewBox="0 0 14 14"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="1.6"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                style={{ flexShrink: 0, marginTop: 1 }}
                              >
                                <circle cx="7" cy="7" r="5.5" />
                                <path d="M7 4.5V7l1.5 1.5" />
                              </svg>
                              Under review — typically resolves within 24 hrs
                            </div>
                          )}
                          {rejected && (
                            <div
                              className={`${styles.claimStatusMsg} ${styles.claimStatusMsgRed}`}
                            >
                              <svg
                                width="13"
                                height="13"
                                viewBox="0 0 14 14"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="1.6"
                                strokeLinecap="round"
                                style={{ flexShrink: 0, marginTop: 1 }}
                              >
                                <circle cx="7" cy="7" r="5.5" />
                                <path d="M5 5l4 4M9 5l-4 4" />
                              </svg>
                              This claim was not eligible for payout. Escalate
                              below if you believe this is incorrect.
                            </div>
                          )}
                          {settled && (
                            <div
                              className={`${styles.claimStatusMsg} ${styles.claimStatusMsgGreen}`}
                            >
                              <svg
                                width="13"
                                height="13"
                                viewBox="0 0 14 14"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="1.6"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                style={{ flexShrink: 0, marginTop: 1 }}
                              >
                                <circle cx="7" cy="7" r="5.5" />
                                <path d="M4.5 7l2 2 3-3" />
                              </svg>
                              Payout of ₹
                              {Number(c.payout_amount ?? 0).toLocaleString(
                                "en-IN",
                              )}{" "}
                              successfully processed.
                            </div>
                          )}

                          {/* ── Footer ── */}
                          <div className={styles.claimCardFooter}>
                            <button
                              type="button"
                              className={styles.claimEscalateBtn}
                              onClick={() =>
                                openSupportModal("claim_escalation", c)
                              }
                            >
                              Escalate Claim
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            {/* ══ PAYMENTS TAB ══ */}
            {tab === "payments" && (
              <div>
                <div className={styles.tabHeader}>
                  <div>
                    <h1 className={styles.profileTitle}>{t("payments")}</h1>
                    <p className={styles.muted}>{t("payments_sub")}</p>
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
                      {t("this_weeks_due")}
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
                    {t("payment_history")}
                  </h3>
                  {payments.length === 0 ? (
                    <div className={styles.emptyState}>{t("no_payments")}</div>
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
                <h1 className={styles.profileTitle}>{t("profile")}</h1>

                {/* ── Account status pills ── */}
                <div className={styles.profileStatusRow}>
                  <StatusPill
                    label="Coverage"
                    ok={coveredNow}
                    okText="Active"
                    badText="Inactive"
                  />
                  <StatusPill
                    label="Verification"
                    ok={verified}
                    okText="Verified"
                    badText="Pending"
                  />
                </div>

                <div className={styles.profileGrid}>
                  <div className={styles.panel}>
                    <h3 className={styles.panelTitle}>
                      {t("section_personal")}
                    </h3>
                    <ProfileRow label={t("row_name")} value={user.name} />
                    <ProfileRow label={t("row_email")} value={user.email} />
                    <ProfileRow
                      label={t("row_phone")}
                      value={user.phone || "–"}
                    />
                    <ProfileRow
                      label={t("row_city")}
                      value={user.city || "–"}
                    />
                    <ProfileRow
                      label={t("row_area")}
                      value={user.area || "–"}
                    />
                    <ProfileRow
                      label={t("row_account_id")}
                      value={user.id}
                      mono
                    />
                    <ProfileRow
                      label={t("row_delivery_id")}
                      value={user.delivery_id || "–"}
                      mono
                    />
                  </div>
                  <div className={styles.panel}>
                    <h3 className={styles.panelTitle}>
                      {t("section_insurance")}
                    </h3>
                    <ProfileRow label={t("row_plan")} value={tierLabel} />
                    <ProfileRow
                      label={t("row_next_plan")}
                      value={PLAN_DETAILS[nextWeekTier].label}
                    />
                    <ProfileRow
                      label={t("row_premium")}
                      value={money(currentWeeklyPremium)}
                    />
                    <ProfileRow
                      label={t("row_next_premium")}
                      value={money(nextWeeklyPremium)}
                    />
                    <ProfileRow
                      label={t("row_verification")}
                      value={verified ? t("verified") : t("pending")}
                    />
                    <ProfileRow
                      label={t("row_autopay")}
                      value={user.autopay ? t("autopay_on") : t("autopay_off")}
                    />
                    <ProfileRow
                      label={t("row_platforms")}
                      value={user.platforms
                        .map((p) => PLATFORM_META[p]?.name || p)
                        .join(", ")}
                    />
                  </div>
                </div>

                {/* ── Language selector ── */}
                <div className={`${styles.panel} ${styles.langPanel}`}>
                  <h3 className={styles.panelTitle}>{t("section_language")}</h3>
                  <p className={styles.panelSub}>{t("select_language")}</p>
                  <div className={styles.langGrid}>
                    {LANGUAGES.map((lang) => {
                      const active = language === lang.code;
                      return (
                        <button
                          key={lang.code}
                          type="button"
                          className={`${styles.langBtn} ${active ? styles.langBtnActive : ""}`}
                          onClick={() => setLanguage(lang.code)}
                        >
                          <span className={styles.langNative}>
                            {lang.native}
                          </span>
                          <span className={styles.langLabel}>{lang.label}</span>
                          {active && (
                            <span className={styles.langCheck}>✓</span>
                          )}
                        </button>
                      );
                    })}
                  </div>
                </div>

              </div>
            )}
          </div>
        </main>
      </div>

      {/* ══ SUPPORT MODAL ══ */}
      {showSupportModal && (
        <div className={styles.modalOverlay} onClick={closeSupportModal}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <h2 className={styles.modalTitle}>
              {supportMode === "claim_escalation"
                ? "Escalate Claim"
                : "Support Request"}
            </h2>
            <p className={styles.modalSub}>
              {supportMode === "claim_escalation"
                ? "Escalation will be sent to the owner dashboard for review."
                : "Describe your issue and our team will get back to you."}
            </p>

            {supportClaimContext && (
              <div className={styles.supportClaimMeta}>
                <div className={styles.supportClaimMetaLine}>
                  <strong>Claim:</strong>{" "}
                  {supportClaimContext.claimNumber || "–"}
                </div>
                <div className={styles.supportClaimMetaLine}>
                  <strong>Type:</strong>{" "}
                  {supportClaimContext.triggerType
                    ? supportClaimContext.triggerType.replace(/_/g, " ")
                    : "–"}
                </div>
              </div>
            )}

            <div className={styles.modalField}>
              <label className={styles.modalLabel}>Subject</label>
              <input
                className={styles.modalInput}
                placeholder="Brief subject"
                value={supportSubject}
                maxLength={120}
                onChange={(e) => setSupportSubject(e.target.value)}
              />
            </div>

            <div className={styles.modalField}>
              <label className={styles.modalLabel}>Issue Details</label>
              <textarea
                className={styles.modalTextarea}
                placeholder="Explain what happened and what help you need"
                value={supportMessage}
                rows={5}
                maxLength={1000}
                onChange={(e) => setSupportMessage(e.target.value)}
              />
            </div>

            {supportError && (
              <div className={styles.modalError}>{supportError}</div>
            )}
            {supportSuccess && (
              <div className={styles.modalSuccess}>{supportSuccess}</div>
            )}

            <div className={styles.modalActions}>
              <button
                type="button"
                className={styles.modalCancelBtn}
                onClick={closeSupportModal}
                disabled={supportLoading}
              >
                Cancel
              </button>
              <button
                type="button"
                className={styles.modalPayBtn}
                onClick={handleSubmitSupportTicket}
                disabled={supportLoading}
              >
                {supportLoading ? "Submitting..." : "Submit Issue"}
              </button>
            </div>
          </div>
        </div>
      )}

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
                  This is a mock payment. No real transaction will occur.
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* ── Mobile Bottom Tab Bar ── */}
      <nav className={styles.mobileNav}>
        {/* Home */}
        <button type="button" className={`${styles.mobileNavBtn} ${tab === "home" ? styles.mobileNavBtnActive : ""}`} onClick={() => setTab("home")}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>
          </svg>
          <span>Home</span>
        </button>
        {/* Claims */}
        <button type="button" className={`${styles.mobileNavBtn} ${tab === "claims" ? styles.mobileNavBtnActive : ""}`} onClick={() => setTab("claims")}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H7a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7z"/><path d="M14 2v5h5"/><path d="m9 14 2 2 4-4"/>
          </svg>
          <span>Claims</span>
        </button>
        {/* Center Chat Button */}
        <div className={styles.mobileNavChatWrap}>
          <button type="button" className={`${styles.mobileNavChat} ${chatOpen ? styles.mobileNavChatActive : ""}`} onClick={() => setChatOpen((o) => !o)} aria-label="Open AI assistant">
            {chatOpen ? (
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
              </svg>
            ) : (
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
              </svg>
            )}
          </button>
          <span className={styles.mobileNavChatLabel}>AI Chat</span>
        </div>
        {/* Payments */}
        <button type="button" className={`${styles.mobileNavBtn} ${tab === "payments" ? styles.mobileNavBtnActive : ""}`} onClick={() => setTab("payments")}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="2.5" y="5.5" width="19" height="13" rx="2.5"/><path d="M2.5 10h19"/><path d="M7 14h4"/>
          </svg>
          <span>Pay</span>
        </button>
        {/* Profile */}
        <button type="button" className={`${styles.mobileNavBtn} ${tab === "profile" ? styles.mobileNavBtnActive : ""}`} onClick={() => setTab("profile")}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="8" r="4"/><path d="M4 20a8 8 0 0 1 16 0"/>
          </svg>
          <span>Profile</span>
        </button>
      </nav>

      {/* ── Floating Chat Button (desktop only) ── */}
      <button
        type="button"
        className={styles.chatFab}
        onClick={() => setChatOpen((o) => !o)}
        aria-label="Open AI assistant"
      >
        {chatOpen ? (
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        ) : (
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
          </svg>
        )}
      </button>

      {/* ── Chat Modal ── */}
      {chatOpen && (
        <div className={styles.chatModal}>
          <div className={styles.chatHeader}>
            <div className={styles.chatHeaderLeft}>
              <div className={styles.chatAvatar}>AI</div>
              <div>
                <div className={styles.chatTitle}>WPIP Assistant</div>
                <div className={styles.chatSubtitle}>Ask me anything about your coverage</div>
              </div>
            </div>
            <button type="button" className={styles.chatCloseBtn} title="Close chat" onClick={() => setChatOpen(false)}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
              </svg>
            </button>
          </div>

          <div className={styles.chatBody}>
            {chatMessages.length === 0 && (
              <div className={styles.chatEmpty}>
                <div className={styles.chatEmptyIcon}>
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                  </svg>
                </div>
                <p className={styles.chatEmptyText}>Hi! I&apos;m your WPIP assistant.</p>
                <p className={styles.chatEmptyHint}>Ask me about claims, premiums, coverage tiers, or how the platform works.</p>
                <div className={styles.chatSuggestions}>
                  {["How do claims work?", "What does Standard cover?", "When will I get paid?"].map((s) => (
                    <button key={s} type="button" className={styles.chatSuggestion} onClick={() => { setChatInput(s); }}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {chatMessages.map((msg, i) => (
              <div key={i} className={`${styles.chatBubbleRow} ${msg.role === "user" ? styles.chatBubbleRowUser : ""}`}>
                {msg.role === "assistant" && <div className={styles.chatBubbleAvatar}>AI</div>}
                <div className={`${styles.chatBubble} ${msg.role === "user" ? styles.chatBubbleUser : styles.chatBubbleAssistant}`}>
                  {msg.content}
                </div>
              </div>
            ))}
            {chatLoading && (
              <div className={styles.chatBubbleRow}>
                <div className={styles.chatBubbleAvatar}>AI</div>
                <div className={`${styles.chatBubble} ${styles.chatBubbleAssistant} ${styles.chatTyping}`}>
                  <span/><span/><span/>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          <div className={styles.chatInputRow}>
            <input
              className={styles.chatInput}
              placeholder="Ask anything…"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); } }}
              disabled={chatLoading}
            />
            <button
              type="button"
              className={styles.chatSendBtn}
              title="Send message"
              onClick={sendChat}
              disabled={!chatInput.trim() || chatLoading}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
              </svg>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
