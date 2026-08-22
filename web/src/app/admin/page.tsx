"use client";
import { useEffect, useState, useCallback } from "react";
import styles from "./page.module.css";

type CooldownCity = {
  last_claim_at: string;
  remaining_hours: number;
  remaining_minutes: number;
};

type CooldownStatus = {
  enabled: boolean;
  cooldown_hours: number;
  cities_in_cooldown: Record<string, CooldownCity>;
};

export default function AdminPage() {
  const [cooldown, setCooldown] = useState<CooldownStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchCooldown = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch("/api/admin/cooldown");
      if (!res.ok) throw new Error("Failed to fetch");
      setCooldown(await res.json());
    } catch {
      setError("Could not reach backend.");
    } finally {
      setLoading(false);
    }
  }, []);

  const toggleCooldown = async (val: boolean) => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch("/api/admin/cooldown", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: val }),
      });
      if (!res.ok) throw new Error("Failed to toggle");
      setCooldown(await res.json());
    } catch {
      setError("Could not update setting.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCooldown();
    const interval = setInterval(fetchCooldown, 30000);
    return () => clearInterval(interval);
  }, [fetchCooldown]);

  const cooldownEnabled = cooldown?.enabled ?? false;
  const citiesInCooldown = cooldown ? Object.entries(cooldown.cities_in_cooldown) : [];

  return (
    <div className={styles.root}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.brand}>
          <div className={styles.brandLogo}>WP</div>
          <span className={styles.brandText}>WPIP Admin</span>
        </div>
        <a href="/dashboard" className={styles.backLink}>
          ← Back to Dashboard
        </a>
      </header>

      <main className={styles.main}>
        <div className={styles.pageHead}>
          <h1 className={styles.pageTitle}>Admin Controls</h1>
          <p className={styles.pageSub}>
            System-level settings for claim processing rules.
          </p>
        </div>

        {error && <div className={styles.errorBanner}>{error}</div>}

        {/* City Claim Cooldown card */}
        <div className={styles.card}>
          <div className={styles.cardHead}>
            <div>
              <h2 className={styles.cardTitle}>City Claim Cooldown</h2>
              <p className={styles.cardDesc}>
                After a claim batch fires in a city, block any further claims
                for that city for{" "}
                <strong>{cooldown?.cooldown_hours ?? 6} hours</strong>.
                Designed to prevent claim flooding after a single disruption
                event. <strong>Off by default.</strong>
              </p>
            </div>

            {/* Toggle */}
            <button
              type="button"
              role="switch"
              aria-checked={cooldownEnabled ? "true" : "false"}
              disabled={loading}
              className={`${styles.toggle} ${cooldownEnabled ? styles.toggleOn : ""}`}
              onClick={() => toggleCooldown(!cooldownEnabled)}
              title="Toggle city claim cooldown"
            >
              <span className={styles.toggleThumb} />
            </button>
          </div>

          {/* Status badge */}
          <div className={styles.statusRow}>
            <span className={`${styles.statusDot} ${cooldownEnabled ? styles.dotOn : styles.dotOff}`} />
            <span className={styles.statusText}>
              {loading
                ? "Updating…"
                : cooldownEnabled
                ? "Enabled — new claims blocked during cooldown window"
                : "Disabled — claims fire freely in all cities"}
            </span>
          </div>

          {/* Per-city cooldown list — only shown when feature is ON */}
          {cooldownEnabled && (
            <div className={styles.cityList}>
              <h3 className={styles.cityListTitle}>Cities currently in cooldown</h3>
              {citiesInCooldown.length === 0 ? (
                <p className={styles.cityListEmpty}>
                  No cities are currently in a cooldown window.
                </p>
              ) : (
                citiesInCooldown.map(([city, info]) => (
                  <div key={city} className={styles.cityRow}>
                    <div className={styles.cityInfo}>
                      <span className={styles.cityName}>
                        {city.charAt(0).toUpperCase() + city.slice(1)}
                      </span>
                      <span className={styles.cityMeta}>
                        Last claim:{" "}
                        {new Date(info.last_claim_at).toLocaleTimeString()}
                      </span>
                    </div>
                    <span className={styles.cityBadge}>
                      {info.remaining_minutes}m remaining
                    </span>
                  </div>
                ))
              )}
            </div>
          )}
        </div>

        {/* Refresh button */}
        <button
          type="button"
          className={styles.refreshBtn}
          onClick={fetchCooldown}
          disabled={loading}
        >
          {loading ? "Refreshing…" : "Refresh Status"}
        </button>
      </main>
    </div>
  );
}
