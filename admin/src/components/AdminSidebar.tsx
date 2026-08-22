"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import styles from "./AdminSidebar.module.css";

type NavItem = {
  href: string;
  label: string;
  icon: React.ReactNode;
  exact?: boolean;
};

const IconOverview = () => (
  <svg width="16" height="16" fill="none" viewBox="0 0 16 16">
    <rect
      x="1"
      y="1"
      width="6"
      height="6"
      rx="1.5"
      stroke="currentColor"
      strokeWidth="1.5"
    />
    <rect
      x="9"
      y="1"
      width="6"
      height="6"
      rx="1.5"
      stroke="currentColor"
      strokeWidth="1.5"
    />
    <rect
      x="1"
      y="9"
      width="6"
      height="6"
      rx="1.5"
      stroke="currentColor"
      strokeWidth="1.5"
    />
    <rect
      x="9"
      y="9"
      width="6"
      height="6"
      rx="1.5"
      stroke="currentColor"
      strokeWidth="1.5"
    />
  </svg>
);

const IconWorkers = () => (
  <svg width="16" height="16" fill="none" viewBox="0 0 16 16">
    <circle cx="6" cy="5" r="2.5" stroke="currentColor" strokeWidth="1.5" />
    <path
      d="M1.5 13.5c0-2.485 2.015-4.5 4.5-4.5s4.5 2.015 4.5 4.5"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
    />
    <circle cx="12" cy="5.5" r="2" stroke="currentColor" strokeWidth="1.5" />
    <path
      d="M14.5 13.5c0-2.071-1.343-3.845-3.228-4.363"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
    />
  </svg>
);

const IconClaims = () => (
  <svg width="16" height="16" fill="none" viewBox="0 0 16 16">
    <rect
      x="2"
      y="1"
      width="10"
      height="14"
      rx="1.5"
      stroke="currentColor"
      strokeWidth="1.5"
    />
    <path
      d="M5 5h4M5 8h4M5 11h2"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
    />
    <path
      d="M12 5h1.5a.5.5 0 0 1 .5.5v9a.5.5 0 0 1-.5.5H5"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
    />
  </svg>
);

const IconSupport = () => (
  <svg width="16" height="16" fill="none" viewBox="0 0 16 16">
    <circle cx="8" cy="8" r="6.5" stroke="currentColor" strokeWidth="1.5" />
    <path
      d="M8 12v-1"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
    />
    <path
      d="M8 10c0-2 3-2.5 3-4.5a3 3 0 0 0-6 0"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
    />
  </svg>
);

const IconDisruptions = () => (
  <svg width="16" height="16" fill="none" viewBox="0 0 16 16">
    <path
      d="M8 2L2 13h12L8 2z"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path
      d="M8 6v3.5"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
    />
    <circle cx="8" cy="11" r=".75" fill="currentColor" />
  </svg>
);

const IconControlCenter = () => (
  <svg width="16" height="16" fill="none" viewBox="0 0 16 16">
    <circle cx="8" cy="8" r="2" stroke="currentColor" strokeWidth="1.5" />
    <path
      d="M8 1v2M8 13v2M1 8h2M13 8h2M3.22 3.22l1.41 1.41M11.37 11.37l1.41 1.41M3.22 12.78l1.41-1.41M11.37 4.63l1.41-1.41"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
    />
  </svg>
);

const IconAnalytics = () => (
  <svg width="16" height="16" fill="none" viewBox="0 0 16 16">
    <path
      d="M2 13V8M6 13V5M10 13V3M14 13V7"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const IconReports = () => (
  <svg width="16" height="16" fill="none" viewBox="0 0 16 16">
    <rect
      x="2"
      y="1"
      width="12"
      height="14"
      rx="1.5"
      stroke="currentColor"
      strokeWidth="1.5"
    />
    <path
      d="M5 5h6M5 8h6M5 11h4"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
    />
  </svg>
);

const IconPayments = () => (
  <svg width="16" height="16" fill="none" viewBox="0 0 16 16">
    <rect
      x="1"
      y="3.5"
      width="14"
      height="10"
      rx="1.5"
      stroke="currentColor"
      strokeWidth="1.5"
    />
    <path
      d="M1 7h14"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
    />
    <path
      d="M4 11h3M11 11h1"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
    />
  </svg>
);

const IconShield = () => (
  <svg width="18" height="18" fill="none" viewBox="0 0 24 24">
    <path
      d="M12 2L3 7v5c0 5.25 3.75 10.15 9 11.35C17.25 22.15 21 17.25 21 12V7L12 2z"
      fill="currentColor"
      opacity="0.9"
    />
    <path
      d="M9 12l2 2 4-4"
      stroke="#fff"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const NAV_MAIN: NavItem[] = [
  { href: "/admin", label: "Overview", icon: <IconOverview />, exact: true },
  { href: "/workers", label: "Workers", icon: <IconWorkers /> },
  { href: "/claims", label: "Claims", icon: <IconClaims /> },
  { href: "/payments", label: "Payments", icon: <IconPayments /> },
  { href: "/support", label: "Support", icon: <IconSupport /> },
  { href: "/disruptions", label: "Disruptions", icon: <IconDisruptions /> },
  {
    href: "/control-center",
    label: "Control Center",
    icon: <IconControlCenter />,
  },
];

const NAV_SECONDARY: NavItem[] = [
  { href: "/analytics", label: "Analytics", icon: <IconAnalytics /> },
  { href: "/reports", label: "Reports", icon: <IconReports /> },
];

function isActive(pathname: string, item: NavItem): boolean {
  if (item.exact) return pathname === item.href || pathname === "/";
  return pathname.startsWith(item.href);
}

export default function AdminSidebar() {
  const pathname = usePathname();

  return (
    <aside className={styles.sidebar}>
      {/* Brand */}
      <div className={styles.brand}>
        <div className={styles.brandIcon}>
          <IconShield />
        </div>
        <div>
          <div className={styles.brandName}>WPIP</div>
          <div className={styles.brandSub}>ADMIN PORTAL</div>
        </div>
      </div>

      {/* Main nav */}
      <nav className={styles.nav}>
        <div className={styles.navGroup}>
          <div className={styles.navGroupLabel}>MAIN MENU</div>
          {NAV_MAIN.map((item) => {
            const active = isActive(pathname, item);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`${styles.navItem} ${active ? styles.navItemActive : ""}`}
              >
                <span className={styles.navIcon}>{item.icon}</span>
                <span>{item.label}</span>
              </Link>
            );
          })}
        </div>

        <div className={styles.navGroup}>
          <div className={styles.navGroupLabel}>INSIGHTS</div>
          {NAV_SECONDARY.map((item) => {
            const active = isActive(pathname, item);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`${styles.navItem} ${active ? styles.navItemActive : ""}`}
              >
                <span className={styles.navIcon}>{item.icon}</span>
                <span>{item.label}</span>
              </Link>
            );
          })}
        </div>
      </nav>
    </aside>
  );
}
