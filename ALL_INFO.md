# WPIP — Worker Protection Insurance Platform
## Complete Technical & Business Reference

---

## Table of Contents

1. [The Big Idea](#1-the-big-idea)
2. [System Architecture — Top to Bottom](#2-system-architecture--top-to-bottom)
3. [Database Schema](#3-database-schema)
4. [Trigger System — How Claims Auto-Fire](#4-trigger-system--how-claims-auto-fire)
5. [Fuzzy Severity Engine — Proportional Payouts](#5-fuzzy-severity-engine--proportional-payouts)
6. [Premium Model — How Cost Varies City to City](#6-premium-model--how-cost-varies-city-to-city)
7. [Fraud Detection — How the Score is Calculated](#7-fraud-detection--how-the-score-is-calculated)
8. [Demo Guide — How to Show High Fraud Score](#8-demo-guide--how-to-show-high-fraud-score)
9. [Web App Architecture](#9-web-app-architecture)
10. [Admin Portal Architecture](#10-admin-portal-architecture)
11. [Mobile App — Full Architecture Deep Dive](#11-mobile-app--full-architecture-deep-dive)
12. [AI Chatbot Integration](#12-ai-chatbot-integration)
13. [Data Flow — End to End](#13-data-flow--end-to-end)
14. [City-by-City Risk Breakdown](#14-city-by-city-risk-breakdown)

---

## 1. The Big Idea

WPIP (Worker Protection Insurance Platform) is a **parametric, weekly micro-insurance platform** for gig delivery workers in India — Zomato, Swiggy, Blinkit, Zepto, Amazon Flex, Meesho, Porter, Dunzo.

### Why Parametric?

Traditional insurance requires workers to file a claim, submit documents, wait weeks, and hope it gets approved. That does not work for someone who earns ₹700/day and cannot afford to be idle.

**Parametric insurance** removes all of that friction:

- There is a **predefined trigger event** (flood, extreme heat, severe AQI, curfew, etc.)
- The system **detects the event automatically** using real-time data (OpenWeatherMap, TomTom Traffic, GDELT news)
- The claim is **auto-created instantly** — the worker does not file anything
- The payout is calculated proportionally using a **fuzzy severity score** (not a binary on/off)
- Money is credited to the worker's UPI ID within 24–48 hours

### What a Worker Pays

Workers pay a **weekly premium** (auto-deducted from platform payouts with a 5% discount, or paid manually). Three tiers:

| Tier | Base Premium / Week | Max Weekly Payout |
|------|--------------------|--------------------|
| Basic | ₹30–₹80 (adjusted) | ₹500 |
| Standard | ₹60–₹100 (adjusted) | ₹1,200 |
| Pro | ₹80–₹120 (adjusted) | ₹2,500 |

Premiums are dynamically re-priced every week based on city weather risk, AQI, and the worker's own delivery history. Workers in Delhi (high pollution/heat) pay more than workers in Pune (lower risk).

---

## 2. System Architecture — Top to Bottom

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENTS                              │
│  Mobile App (React Native/Expo)  │  Web App (Next.js 15)   │
│  Admin Portal (Next.js 15)                                  │
└──────────────────┬──────────────────┬───────────────────────┘
                   │                  │
         REST / JSON over HTTPS       │
                   │                  │
┌──────────────────▼──────────────────▼───────────────────────┐
│              Next.js API Routes (web/src/app/api)            │
│  /api/auth/login   /api/auth/register   /api/auth/me        │
│  /api/payment/pay  /api/payment/history                     │
│  /api/claims/worker/[deliveryId]                            │
│  /api/backend/[...path]  (catch-all proxy to FastAPI)       │
│  /api/chat         (OpenAI GPT-4o-mini proxy)               │
│  /api/verify-id                                             │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP (localhost:8000)
┌──────────────────────────────▼──────────────────────────────┐
│                  FastAPI Backend (Python)                    │
│  backend/main.py — 1200+ lines                              │
│                                                             │
│  Auth endpoints:   /api/auth/login  /api/auth/register      │
│  Premium:         /api/premium/predict                      │
│  Claims:          /api/claims/worker/{id}  /api/claims/all  │
│  Triggers:        /api/triggers/status  /api/triggers/fire  │
│  Payments:        /api/payments  /api/workers              │
│  Rescore:         /api/claims/{id}/rescore                  │
│                                                             │
│  Scheduler — runs poll_triggers() every 15 minutes          │
└───────┬──────────────────────┬───────────────────────────────┘
        │                      │
┌───────▼───────┐   ┌──────────▼──────────────────────────────┐
│  ML Services  │   │         Supabase (PostgreSQL)           │
│               │   │                                         │
│ ml/weather.py │   │  registered_workers  (auth + profile)   │
│ ml/traffic.py │   │  claims              (all claim records) │
│ ml/curfew.py  │   │  disruption_events   (trigger events)   │
│ ml/fraud_     │   │  worker_payments     (premium payments)  │
│   model.py    │   │  zomato_workers      (platform data)     │
│ ml/premium_   │   │  swiggy_workers      (platform data)     │
│   model.py    │   │  blinkit_workers     (platform data)     │
│               │   │  ... (8 platform tables total)          │
│ fuzzy.py      │   │  traffic_tti_history (congestion audit)  │
│ claims.py     │   │  premium_predictions (weekly pricing)    │
└───────────────┘   └─────────────────────────────────────────┘
        │
┌───────▼───────────────────────────────────────────────────┐
│               External Data APIs                           │
│  OpenWeatherMap — temp, rain, AQI (5-min cache)           │
│  TomTom Traffic — TTI (Travel Time Index) per city        │
│  GDELT Project  — news events for curfew detection        │
│  Hugging Face   — NLP model for sentiment/curfew labels   │
└───────────────────────────────────────────────────────────┘
```

### Service Ports

| Service | Port | Technology |
|---------|------|-----------|
| Web App | 3000 | Next.js 15 (App Router) |
| Admin Portal | 3001 | Next.js 15 (App Router) |
| FastAPI Backend | 8000 | Python, Uvicorn |
| ML Service | 8001 | FastAPI (separate process) |
| Mobile App | Expo | React Native 0.83 |

---

## 3. Database Schema

All data lives in **Supabase** (PostgreSQL with row-level security).

### `registered_workers`
The single source of truth for all WPIP members.

| Column | Type | Purpose |
|--------|------|---------|
| `id` | UUID (PK) | Internal user ID — used in JWT `sub` claim |
| `delivery_id` | UUID | The platform's own worker ID (from Zomato/Swiggy etc.) |
| `email`, `phone` | text | Login credentials |
| `password_hash` | text | bcrypt hash (12 rounds) |
| `city` | text | Registered city — must match platform DB city |
| `platforms` | text[] | e.g. `["zomato", "swiggy"]` |
| `tier` | text | `basic` / `standard` / `pro` |
| `verification_status` | text | `pending` / `verified` / `rejected` |
| `is_active` | bool | Soft-delete flag |
| `created_at` | timestamptz | Registration timestamp — used as fraud cutoff |
| `autopay`, `upi` | bool, text | Payment preferences |

### `claims`
Every auto-generated claim lives here.

| Column | Type | Purpose |
|--------|------|---------|
| `id` | UUID (PK) | Claim UUID |
| `claim_number` | text | Human readable: `CLM-20260425-a1b2c3` |
| `worker_id` | UUID | = delivery_id (links to platform tables) |
| `city` | text | Must match registered city |
| `trigger_type` | text | `flood`, `extreme_heat`, `heavy_rain`, etc. |
| `payout_amount` | float | Calculated payout in ₹ |
| `fuzzy_payout_multiplier` | float | Severity score (0–1) |
| `fraud_score` | float | 0.0 = clean, 1.0 = certain fraud |
| `fraud_flags` | text[] | e.g. `["high_claim_frequency"]` |
| `status` | text | `auto_initiated` / `under_review` / `paid` |
| `payout_status` | text | `approved` / `pending` / `rejected` |
| `cross_platform_clear` | bool | Was worker idle during event? |
| `created_at` | timestamptz | Claim creation time |

### Platform Tables (8 total)
`zomato_workers`, `swiggy_workers`, `blinkit_workers`, `zepto_workers`, `amazon_flex_workers`, `meesho_workers`, `porter_workers`, `dunzo_workers`

Each has the same shape:

| Column | Purpose |
|--------|---------|
| `worker_id` | Platform's UUID for this worker |
| `platform` | e.g. `"zomato"` |
| `date` | Activity date |
| `deliveries` | Delivery count that day |
| `earnings` | ₹ earned that day |
| `active_hours` | Hours online |
| `city` | City of operation |
| `verified` | Platform-verified flag |

---

## 4. Trigger System — How Claims Auto-Fire

The trigger system (`backend/triggers.py`) is the heart of WPIP. It runs on a **15-minute polling loop** via the backend scheduler.

### What Happens Every 15 Minutes

```
1. For each of 6 cities (Chennai, Bengaluru, Hyderabad, Mumbai, Delhi, Pune):
   a. Fetch live weather from OpenWeatherMap (temp, rain_1h, rain_3h, AQI)
   b. Fetch live traffic TTI from TomTom
   c. Evaluate curfew risk from GDELT + NLP (async, 60s timeout)

2. Run all 6 trigger rules against the live data

3. For each fired trigger:
   a. Upsert a disruption_event row (idempotent — same trigger+city+date = no duplicate)
   b. Find all registered workers in that city
   c. Check each worker was NOT delivering during the event (cross-platform check)
   d. Calculate payout = (hourly_rate × coverage% × disrupted_hours × severity)
   e. Run fraud scoring on the claim
   f. Insert/upsert claim row
```

### The 6 Triggers

| ID | Name | Entry | Full Severity | Data Source |
|----|------|-------|--------------|-------------|
| T-01 | Heavy Rainfall | 20mm | 100mm | OpenWeatherMap rain |
| T-02 | Extreme Heat | 39.5°C | 45°C | OpenWeatherMap temp |
| T-03 | Severe AQI | 300 index | 450 index | OpenWeatherMap air_pollution |
| T-04 | Flood Risk | 64.5mm (3h) | 150mm (3h) | OpenWeatherMap rain_3h |
| T-05 | Traffic Congestion | TTI > 2.5 (2 consecutive cycles) | TTI 3.5 | TomTom Traffic |
| T-06 | Curfew / Unrest | 0.80 confidence | 0.95 confidence | GDELT + NLP |

### T-05 Special Rule — Consecutive Cycles

Traffic congestion only fires if TTI > 2.5 for **two consecutive 15-minute polling cycles**. A brief traffic spike does not trigger a payout. This prevents false positives from accidents that clear within minutes.

```python
# Stored between poll cycles per city
_previous_tti: dict[str, float] = {}

# Only fires if BOTH current AND previous TTI > 2.5
if not (current_tti > 2.5 and prev_tti > 2.5):
    continue  # skip
```

### T-06 Curfew — GDELT + NLP Pipeline

1. GDELT (Global Database of Events, Language, and Tone) is queried for the city's recent news events
2. RSS feeds and news articles are parsed
3. A HuggingFace NLP model classifies the text: `"section 144"`, `"curfew"`, `"protest"`, etc.
4. A combined confidence score is computed: `0.4 × gdelt_score + 0.6 × nlp_score`
5. If confidence ≥ 0.80, T-06 fires

### Cross-Platform Activity Check

Before creating a claim, the system checks all 8 platform tables to see if the worker had any delivery on the event date. If they were actively working during a "disruption," they did not lose income, so no claim is created.

```python
def check_cross_platform_activity(worker_id, event_date):
    # Returns True if NO activity found (worker is clear)
    # Returns False if activity detected (skip claim)
    for table in PLATFORM_TABLES:
        resp = supabase.table(table)
            .select("worker_id")
            .eq("worker_id", worker_id)
            .eq("date", event_date)
            .execute()
        if resp.data:
            return False  # was working, skip claim
    return True
```

---

## 5. Fuzzy Severity Engine — Proportional Payouts

Traditional parametric insurance is binary: either the event happened (full payout) or it did not (₹0). This creates **basis risk** — a worker in a flood with 70mm of rain gets the same payout as one with 150mm.

WPIP uses a **fuzzy logic severity engine** (`backend/fuzzy.py`) that produces a continuous score between 0.0 and 1.0.

### How It Works

```python
def fuzzy_score(value, low, high):
    """Linear membership function."""
    if value <= low:  return 0.0   # below entry — no disruption
    if value >= high: return 1.0   # at ceiling — maximum disruption
    return (value - low) / (high - low)  # proportional between entry and ceiling
```

### Calibration Examples

**T-02 Extreme Heat** (entry = 39.5°C, ceiling = 45°C):
- 39.5°C → score = 0.00 → 0% payout
- 40.0°C → score = 0.09 → 9% of max payout
- 42.0°C → score = 0.45 → 45% of max payout
- 45.0°C → score = 1.00 → 100% of max payout

**T-04 Flood** (entry = 64.5mm, ceiling = 150mm):
- 64.5mm → score = 0.00 → no payout
- 100mm  → score = 0.42 → 42% of max payout
- 150mm  → score = 1.00 → full payout

### Payout Formula

```
hourly_rate     = daily_wage / 8
coverage_pct    = 50% (basic) | 80% (standard) | 100% (pro)
raw_payout      = hourly_rate × coverage_pct × 6 hours × severity_score
final_payout    = min(raw_payout, absolute_tier_cap × severity_score)
```

**Example:** Standard worker, daily wage ₹700, flood severity 0.60:
```
hourly_rate  = ₹700 / 8 = ₹87.50
raw_payout   = ₹87.50 × 0.80 × 6 × 0.60 = ₹252
tier_cap     = ₹1,200 × 0.60 = ₹720
final_payout = min(₹252, ₹720) = ₹252
```

---

## 6. Premium Model — How Cost Varies City to City

The weekly premium is calculated by `backend/ml/premium_model.py` using a **trained ML model** (or a formula fallback) on 25 features.

### Input Features (25 total)

**Worker Activity (from platform tables, last 7 days):**
- `avg_earnings_7d` — average daily earnings
- `avg_deliveries_7d` — average deliveries per day
- `avg_hours_7d` — average active hours
- `avg_rating_7d` — customer rating
- `earnings_trend` — 3-day vs 7-day trend
- `earnings_cv` — earnings coefficient of variation (volatility)
- `earn_per_delivery`, `earn_per_hour`
- `total_days` — days of data available

**Live Environmental Risk:**
- `temperature`, `aqi_index`, `rain_1h`, `humidity`, `weather_risk`
- `tti` — Traffic Time Index
- `curfew_confidence` — unrest risk

**Tier:**
- `tier_basic`, `tier_standard`, `tier_pro` (one-hot encoded)

### City Base Risk Multipliers

These are the **baseline multipliers** before real weather data is applied:

| City | Base Risk | Reason |
|------|-----------|--------|
| Delhi | 1.20× | Extreme summers (48°C+), severe AQI (500+) in winter |
| Chennai | 1.15× | Cyclone-prone, northeast monsoon flooding |
| Mumbai | 1.10× | Heavy monsoon (300mm+ in 24h), coastal flooding |
| Kolkata | 1.05× | Monsoon flooding, Nor'westers |
| Bangalore | 1.00× | Baseline — moderate climate year-round |
| Hyderabad | 0.95× | Mild summers, moderate rainfall |
| Pune | 0.90× | Lowest risk — Western Ghats rain shadow |

### Tier Pricing Config

```python
TIER_CONFIG = {
    "basic":    {"rate": 0.008, "min": 40, "max": 80,  "max_payout": 500},
    "standard": {"rate": 0.012, "min": 60, "max": 100, "max_payout": 1200},
    "pro":      {"rate": 0.016, "min": 80, "max": 120, "max_payout": 2500},
}
```

- `rate` = base multiplier on weekly earnings estimate
- `min` / `max` = hard floor and ceiling for the premium (₹/week)
- Premium = `max(min, min(max, weekly_earnings_est × rate × city_risk × weather_risk))`

### Real Pricing Example

Worker in Delhi (risk 1.20), weekly earnings estimate ₹4,000, Standard tier:
```
base        = ₹4,000 × 0.012 = ₹48
city_adj    = ₹48 × 1.20 = ₹57.6
weather_adj = ₹57.6 × 1.15 (active AQI season) = ₹66.2
final       = clamp(₹66.2, min=60, max=100) = ₹66.2/week
```

Same worker in Pune (risk 0.90):
```
base        = ₹4,000 × 0.012 = ₹48
city_adj    = ₹48 × 0.90 = ₹43.2
weather_adj = ₹43.2 × 1.00 (mild weather) = ₹43.2
final       = clamp(₹43.2, min=60, max=100) = ₹60/week (floor kicks in)
```

**Delhi worker pays ₹66 vs Pune worker pays ₹60 — same tier, same earnings, different city risk.**

---

## 7. Fraud Detection — How the Score is Calculated

Fraud detection runs in two layers: an **ML service** (primary) and a **rule-based fallback**.

### Step 1 — Build Feature Vector

`build_fraud_features()` in `ml/fraud_model.py` constructs this input:

| Feature | What It Measures |
|---------|-----------------|
| `claim_count_30d` | How many claims this worker has filed since registration (or 30 days, whichever is more recent) |
| `total_payout_30d` | Total ₹ paid out in that same window |
| `payout_amount` | The current claim's proposed payout |
| `payout_ratio` | `payout_amount / daily_wage` — is the claim disproportionate? |
| `daily_wage` | Worker's 7-day average daily earnings |
| `cross_platform_flag` | 1.0 if worker was detected working during the event, 0.0 if idle |
| `curfew_zone_mismatch` | 1.0 if curfew claim but worker's GPS does not match curfew zone |

**Critical:** The 30-day window is anchored to the worker's **registration date**, not just 30 days ago. If a worker re-registered using a delivery_id that had 16 historical claims, those pre-registration claims are excluded from `claim_count_30d`. This prevents inherited fraud penalties.

```python
thirty_days_ago = (date.today() - timedelta(days=30)).isoformat()
if registration_date:
    reg_date = registration_date[:10]
    cutoff = reg_date if reg_date > thirty_days_ago else thirty_days_ago
# Only count claims since max(registration_date, 30_days_ago)
```

### Step 2 — ML Service (Primary Path)

The feature vector is sent to a separate ML service on port 8001:

```
POST http://localhost:8001/predict/fraud-score
{
  "delivery_activity_detected": false,
  "duplicate_claim": false,
  "claim_count_30d": 2,
  "payout_amount": 266.0,
  "daily_wage": 700.0,
  ...
}
```

The ML service runs an **Isolation Forest** model trained on historical claims. Isolation Forest is an unsupervised anomaly detection algorithm — it isolates outliers (unusual claim patterns) using random partitioning trees. An outlier (anomaly) gets a high fraud score.

### Step 3 — Rule-Based Fallback

If the ML service is down, the rule-based fallback applies:

| Condition | Score Added | Flag |
|-----------|-------------|------|
| Worker was actively delivering during event | +0.50 | `ACTIVE_DURING_CLAIM` |
| Duplicate claim (same event, same worker) | +0.45 | `DUPLICATE_CLAIM` |
| More than 5 claims in the window | +0.25 | `HIGH_CLAIM_FREQUENCY` |
| 3–5 claims in the window | +0.10 | `ELEVATED_CLAIM_FREQUENCY` |
| Cross-platform activity detected | +0.35 | `CROSS_PLATFORM_ACTIVITY` |
| Curfew zone GPS mismatch | +0.50 | `CURFEW_ZONE_MISMATCH` |
| Payout > 120% of daily wage | +0.15 | `HIGH_PAYOUT_RATIO` |
| Total payout in window > ₹3,000 | +0.10 | `HIGH_CUMULATIVE_PAYOUT` |

Scores are summed and capped at 0.99.

### Score → Disposition

| Score | Disposition | Claim Status |
|-------|-------------|-------------|
| < 0.60 | Auto_Approved | `auto_initiated` → paid |
| 0.60–0.89 | Flagged for Manual Review | `under_review` |
| ≥ 0.90 | Auto_Rejected | `under_review` + rejected |

---

## 8. Demo Guide — How to Show High Fraud Score

### Method 1: Use the Rescore Button (Admin Portal)

1. Go to admin portal → Claims page
2. Find any existing claim
3. Click the **Rescore** button next to it
4. The system re-runs the ML fraud pipeline and updates the displayed score

### Method 2: Manually Fire Multiple Claims for One Worker

The most reliable way to show a **high fraud score (>90%)** in a live demo:

1. Open the admin portal → go to **Settings / Simulator**
2. Fire trigger `T-04` (Flood) for **Mumbai** → creates 1 claim for each registered Mumbai worker
3. Fire trigger `T-02` (Extreme Heat) for **Mumbai** → creates more claims for the same workers
4. Fire trigger `T-01` (Heavy Rain) for **Mumbai** → now each worker has 3+ claims in the window

After 5+ claims, `claim_count_30d > 5` → rule-based fallback adds +0.25 (`HIGH_CLAIM_FREQUENCY`) on top of any other signals → score likely crosses 0.60 or 0.90.

**Exact API call for manual fire:**
```bash
curl -X POST http://localhost:8000/api/triggers/fire \
  -H "Content-Type: application/json" \
  -d '{"city": "Mumbai", "trigger_id": "T-01"}'
```

Repeat 4–5 times with different trigger IDs for the same city and the same registered worker. Their `claim_count_30d` will accumulate.

### Method 3: Make a Worker Active During a Claim (Cross-Platform Flag)

If you want to show the `ACTIVE_DURING_CLAIM` flag (+0.50):

1. Insert a delivery row for a worker in the platform table for today's date
2. Fire any trigger for that worker's city
3. The cross-platform check will find the delivery row and set `cross_platform_clear = False`
4. The fraud score immediately jumps by 0.50 minimum

```sql
INSERT INTO zomato_workers (worker_id, platform, date, deliveries, earnings, active_hours, city)
VALUES ('<delivery_id>', 'zomato', CURRENT_DATE, 10, 400, 6, 'Chennai');
```

Then fire T-01 for Chennai. Fraud score will be ≥ 0.50 (usually triggers `Flagged_Manual_Review`).

### Summary: Fastest High Fraud Demo

```
1. Fire T-04 Flood for Mumbai → worker gets claim #1
2. Fire T-01 Heavy Rain for Mumbai → claim #2
3. Fire T-02 Extreme Heat for Mumbai → claim #3
4. Fire T-03 Severe AQI for Mumbai → claim #4
5. Fire T-04 Flood again (new date) → claim #5
6. Fire T-01 again → claim #6 → claim_count > 5 → score 0.85+
```

The admin portal's Claims page will show the fraud score in red with the flags listed.

---

## 9. Web App Architecture

**Location:** `web/` | **Framework:** Next.js 15 App Router | **Port:** 3000

### Structure

```
web/src/app/
├── api/                     # Server-side API routes (run on Node.js)
│   ├── auth/
│   │   ├── login/route.ts   # POST — authenticates, returns JWT
│   │   ├── register/route.ts# POST — validates city, creates account
│   │   └── me/route.ts      # GET — returns user profile with created_at
│   ├── backend/
│   │   └── [...path]/route.ts # Catch-all proxy → FastAPI :8000
│   ├── chat/route.ts        # POST → OpenAI GPT-4o-mini
│   ├── claims/
│   │   └── worker/[deliveryId]/route.ts  # Registration-date filtered claims
│   ├── payment/
│   │   ├── pay/route.ts     # POST — process premium payment
│   │   └── history/route.ts # GET — payment history
│   ├── premium/predict/     # Proxies to FastAPI
│   └── verify-id/route.ts   # Validates delivery ID against platform DB
├── dashboard/
│   ├── page.tsx             # Main worker dashboard (2700+ lines)
│   ├── page.module.css      # CSS Modules (3000+ lines)
│   └── PlatformIcons.tsx    # SVG logos for each platform
├── login/page.tsx
├── register/page.tsx        # 4-step registration wizard
├── simulator/page.tsx       # Weather disruption simulator
└── globals.css              # Design tokens (CSS custom properties)
```

### Authentication Flow

```
Worker logs in → /api/auth/login → verifies password against Supabase
→ creates JWT (HS256, 30-day expiry, secret = JWT_SECRET)
→ returns { token, user }
→ Client stores token in localStorage ("gg_token")
→ All subsequent API calls: Authorization: Bearer <token>
```

The token's `sub` claim is the `registered_workers.id` UUID (NOT the delivery_id).

### The JWT Secret Problem

The Python backend and Next.js both use the same JWT secret (`wpip-dev-secret` by default, configurable via `JWT_SECRET` env var). The Next.js `verifyToken()` function tries multiple known secrets to support tokens issued by either service.

### Dashboard State Architecture

The dashboard (`dashboard/page.tsx`) is a single large client component with:
- 15+ `useState` hooks for all data (premium, claims, payments, traffic, curfew, chat, etc.)
- `useEffect` for initial load, 15-second server refresh, and reactive re-fetches
- All API calls go through Next.js API routes (never directly to FastAPI from the browser)
- CSS Modules for styling with `var(--accent)`, `var(--primary)` etc. design tokens

### Claims Filtering

Claims are filtered server-side in `/api/claims/worker/[deliveryId]/route.ts`:
1. The Next.js route calls the FastAPI backend: `GET /api/claims/worker/{delivery_id}`
2. The FastAPI backend looks up `registered_workers.created_at` using the delivery_id
3. Appends `?after=<timestamp>` to the Supabase query
4. Only claims from registration onwards are returned
5. Client-side has a secondary filter as a safety net

---

## 10. Admin Portal Architecture

**Location:** `admin/` | **Framework:** Next.js 15 App Router | **Port:** 3001

### Pages

| Route | Purpose |
|-------|---------|
| `/workers` | All registered workers, verification controls |
| `/claims` | All claims, fraud scores, Rescore button, premium coverage badge |
| `/payments` | Transaction history, weekly analytics (80/20 split) |
| `/triggers` | Live trigger status per city, manual fire controls |
| `/reports` | Analytics and reporting |

### 80/20 Revenue Split (Payments Page)

For every ₹100 premium collected:
- **₹80** → Claims Reserve (used to pay worker claims)
- **₹20** → Platform Revenue (WPIP's gross profit)

The admin payments page shows weekly bar charts breaking down both figures.

### Rescore Endpoint

When an admin clicks "Rescore" on a claim:
```
Admin UI → POST /api/claims/rescore (admin Next.js)
→ POST http://localhost:8000/api/claims/{claim_id}/rescore (FastAPI)
→ Fetches claim from Supabase
→ Looks up worker's registration date from registered_workers
→ Re-runs build_fraud_features() with correct cutoff
→ Re-runs compute_fraud_score()
→ Updates fraud_score and fraud_flags in claims table
→ Returns updated claim to admin UI
```

---

## 11. Mobile App — Full Architecture Deep Dive

**Location:** `mobile/` | **Framework:** React Native 0.83 + Expo SDK 55 | **Type:** Native App (NOT PWA)

### Native App vs PWA

The WPIP mobile app is a **fully native mobile app** built with React Native. It is not a Progressive Web App (PWA). Key differences:

| Feature | PWA | WPIP Mobile (React Native) |
|---------|-----|---------------------------|
| Runtime | Browser (WebView) | Native iOS/Android runtime |
| Install | Browser prompt | App Store / Expo Go |
| Performance | Limited | Native-level |
| APIs | Web APIs only | Full native device APIs |
| Offline | Service Worker | AsyncStorage + React Query |
| Navigation | Browser history | Native stack navigator |
| Styling | CSS | React Native StyleSheet |

Expo is the build toolchain — it wraps React Native and provides a managed workflow for fonts, splash screens, and OTA updates without a full native build for development.

### App Entry & Bootstrap

```
index.js → App.js → LanguageProvider → ThemeProvider → AuthProvider → NavigationContainer
```

**AuthProvider** (`context/AuthContext.js`):
- On app launch, reads `gg_token` and `gg_user` from AsyncStorage
- Calls `/api/auth/me` to verify the token is still valid
- If valid, restores the session — the user lands directly on HomeScreen
- If expired, clears session and returns to LandingScreen

**ThemeProvider** (`context/ThemeContext.js`):
- Stores theme preference in AsyncStorage (`gg_theme_mode`)
- Exposes `COLORS`, `FONTS`, `SIZES`, `SHADOWS` to all components
- Two color sets: `DARK_COLORS` and `LIGHT_COLORS` defined in `constants/theme.js`

**LanguageProvider** (`context/LanguageContext.js`):
- Manages active language (English, Hindi, Tamil, Telugu, Malayalam)
- Translation function `t("key")` used throughout all screens

### Navigation Structure

React Navigation v7 with a **Native Stack Navigator** (hardware-accelerated transitions):

```
LandingScreen   → splash + onboarding
LoginScreen     → email/password
SignUpScreen    → 4-step registration wizard (mirrors web)
HomeScreen      → main dashboard (bottom tab navigation built-in)
PolicyScreen    → coverage details, tier upgrade
ClaimsScreen    → claims history with date filtering
ProfileScreen   → personal info, language, theme toggle
PaymentScreen   → premium payment, UPI/card input
```

All screens use `headerShown: false` — custom headers are built inside each screen.

### HomeScreen Architecture (The Main Screen)

HomeScreen (`screens/HomeScreen.js`) is the most complex file — 2200+ lines. It contains:

**State Management (all local useState):**
```javascript
[loading, setLoading]           // Initial data fetch
[premium, setPremium]           // Current week's premium quote
[claims, setClaims]             // Worker's claims (from registration onwards)
[triggerStatus, setTriggerStatus] // Live trigger data per city
[trafficLive, setTrafficLive]  // TomTom TTI
[curfewLive, setCurfewLive]    // Curfew risk
[showPayModal, setShowPayModal] // Payment sheet
[chatOpen, setChatOpen]         // AI chatbot visibility
[chatMessages, setChatMessages] // Conversation history
[chatInput, chatLoading]        // Chat input state
```

**Data Loading (`loadDashboard`):**
```javascript
// Called on mount and on pull-to-refresh
const [premium, claims, triggers, traffic, curfew] = await Promise.allSettled([
  api.predictPremium(deliveryId, city, tier),
  api.getWorkerClaims(token, deliveryId),
  api.getTriggerStatus(),
  api.getCityTraffic(city),
  api.getCityCurfew(city),
]);
```

All 5 API calls are parallelized using `Promise.allSettled` — if one fails, the others still succeed.

**Bottom Tab Navigation (built inside HomeScreen):**
```
[Home] [Claims] [AI Chat ↑] [Policy] [Profile]
                 (raised center button)
```
The center "AI Chat" button is a raised circular button (like Instagram's camera button) that lifts above the tab bar using `marginTop: -22`.

### API Layer (`lib/api.js`)

The mobile app never calls Supabase or FastAPI directly. All requests go through the Next.js web app's API routes:

```javascript
const WEB_URL    = process.env.WEB_API_URL    || "http://localhost:3000"
const BACKEND_URL = process.env.BACKEND_API_URL || "http://localhost:8000"
```

Every API function uses a **primary + fallback** pattern:
```javascript
async function requestWithFallback(primaryUrl, fallbackUrl, options) {
  try {
    return await request(primaryUrl, options);  // Try Next.js first
  } catch (error) {
    if (shouldFallback(error)) {
      return request(fallbackUrl, options);     // Fall back to FastAPI
    }
    throw error;
  }
}
```

This means if the Next.js server is down, the app still works by hitting FastAPI directly.

### Styling System

React Native does not use CSS. Styles are defined with `StyleSheet.create()` inside each screen, computed at render time using `useMemo`:

```javascript
const styles = useMemo(() => createStyles(COLORS, FONTS), [COLORS, FONTS]);
```

`createStyles` is a function that takes the current theme's COLORS and FONTS and returns a `StyleSheet` object. This means themes are applied reactively — switching dark/light mode instantly re-computes all styles.

### Session Persistence

```javascript
// Save on login
await AsyncStorage.setItem("gg_token", token);
await AsyncStorage.setItem("gg_user", JSON.stringify(user));

// Restore on app launch
const storedToken = await AsyncStorage.getItem("gg_token");
const storedUser  = await AsyncStorage.getItem("gg_user");

// Verify token is still valid (call /api/auth/me)
// If expired → clearSession() → back to Landing
```

`AsyncStorage` is React Native's equivalent of `localStorage` — it persists across app restarts.

### SignUpScreen — 4-Step Registration

| Step | Fields |
|------|--------|
| 1 — Platforms | Select delivery platforms (Zomato, Swiggy, etc.) |
| 2 — Personal Info | Name, phone, email, password, city (auto-detected), delivery ID |
| 3 — Documents | PAN, Aadhaar, UPI ID, bank account, consent checkboxes |
| 4 — Coverage | Choose tier (Basic/Standard/Pro), see premium estimate |

On "Verify ID": the app calls `/api/verify-id` which checks the delivery ID against the platform tables in Supabase and returns the city associated with that worker. The city field is auto-filled and locked.

### Mobile App File Map

```
mobile/
├── App.js                    # Root — providers + navigator
├── index.js                  # Expo entry point
├── app.json                  # Expo config (name, icons, splash)
├── babel.config.js           # Babel + react-native-dotenv
├── .env                      # WEB_API_URL, BACKEND_API_URL
├── context/
│   ├── AuthContext.js        # Session management, login/logout
│   ├── ThemeContext.js       # Dark/light theme + color tokens
│   └── LanguageContext.js    # i18n, t() translation function
├── constants/
│   └── theme.js              # DARK_COLORS, LIGHT_COLORS, FONTS, SIZES, SHADOWS
├── lib/
│   └── api.js                # All API calls (primary + fallback routing)
├── components/
│   ├── Button.js             # Reusable button
│   ├── Checkbox.js           # Styled checkbox
│   └── InputField.js         # Styled text input
├── screens/
│   ├── LandingScreen.js      # Onboarding / splash
│   ├── LoginScreen.js        # Email + password login
│   ├── SignUpScreen.js       # 4-step registration
│   ├── HomeScreen.js         # Main dashboard (2200+ lines)
│   ├── ClaimsScreen.js       # Claims history
│   ├── PolicyScreen.js       # Coverage details
│   ├── ProfileScreen.js      # User profile + settings
│   └── PaymentScreen.js      # Premium payment
├── android/                  # Android native project
└── ios/                      # iOS native project
```

---

## 12. AI Chatbot Integration

### Architecture

```
Mobile / Web UI
    ↓ POST { messages: [...] }
Next.js /api/chat/route.ts
    ↓ POST to OpenAI API
    { model: "gpt-4o-mini", messages: [system + last 12 messages] }
    ↓
OpenAI GPT-4o-mini
    ↓ reply text
Back to client
```

### System Prompt

The chatbot is given a WPIP-specific system prompt explaining:
- How parametric triggers work (no manual filing)
- The 3 coverage tiers and their payouts
- Premium pricing (weekly, dynamically adjusted)
- Claim settlement timeline (24–48 hours via UPI)
- AutoPay discount (5%)
- City and registration date constraints
- Fraud detection behavior

The model is instructed to respond in the user's language (Hindi, Tamil, etc. if they write in that language).

### Web Implementation
- Floating FAB button (bottom-right, hidden on mobile)
- Sliding chat modal with message bubbles, typing indicator, suggestion chips
- State: `chatMessages`, `chatInput`, `chatLoading`

### Mobile Implementation
- Center raised button in the bottom tab bar
- Bottom sheet modal (slides up from bottom) with keyboard-aware layout
- Same conversation history and OpenAI backend
- Suggestions: "How do claims work?", "What does Standard cover?", "When will I get paid?"

---

## 13. Data Flow — End to End

### Worker Registration

```
Worker opens app → selects platforms → enters delivery ID
→ clicks "Verify ID"
→ /api/verify-id hits zomato_workers / swiggy_workers etc.
→ returns detected_city (auto-fills city, locks dropdown)
→ Worker completes 4 steps → submits
→ /api/auth/register:
    1. Check email/phone uniqueness
    2. Verify delivery ID in platform tables
    3. Validate city matches platform DB city (HARD REJECT if mismatch)
    4. Hash password (bcrypt, 12 rounds)
    5. Insert into registered_workers
    6. Create JWT
→ Worker lands on dashboard
```

### Claim Auto-Creation (Every 15 Minutes)

```
Scheduler tick
→ poll_triggers() for all 6 cities
→ For Chennai: fetch_weather() → temp=43°C → T-02 fires (score=0.636)
→ _upsert_disruption_event("Chennai", T-02, today)
→ _get_workers_in_city("Chennai", today) → 50 workers found in platform tables
→ _get_registered_workers_index("Chennai") → 3 registered workers
→ Filter: only 3 workers eligible
→ For each:
    check_cross_platform_activity(worker_id, today) → True (idle, eligible)
    daily_wage = get_worker_daily_wage(worker_id, days=7) = ₹720
    payout = compute_payout(₹720, 6h, 0.636, "standard") = ₹207
    fraud_features = build_fraud_features(worker_id, ...)
    fraud_score = compute_fraud_score(features) = 0.05 (clean)
    status = "auto_initiated"
    INSERT INTO claims (...)
→ 3 claims created, workers notified on next dashboard refresh
```

### Worker Sees Claim

```
Worker opens mobile app → HomeScreen loads
→ api.getWorkerClaims(token, deliveryId)
→ GET http://localhost:3000/api/claims/worker/{deliveryId}
→ Next.js route → GET http://localhost:8000/api/claims/worker/{deliveryId}
→ FastAPI: looks up registered_workers.created_at by delivery_id
→ Queries claims WHERE worker_id = deliveryId AND created_at >= registration_date
→ Returns only post-registration claims
→ Worker sees ₹207 flood claim, status: "Approved — payout processing"
```

---

## 14. City-by-City Risk Breakdown

### Why Cities Differ

India's gig economy is concentrated in metros with vastly different climate profiles:

| City | Key Risk Drivers | Typical High-Risk Period |
|------|-----------------|------------------------|
| **Delhi** | Summer heat (45–48°C), winter smog (AQI 500+), dust storms | May–Jun (heat), Nov–Jan (AQI) |
| **Chennai** | Northeast monsoon flooding, Bay of Bengal cyclones | Oct–Dec |
| **Mumbai** | Arabian Sea monsoon (100mm+ in 6 hours), coastal flooding | Jun–Sep |
| **Hyderabad** | Flash floods during southwest monsoon, heat in summer | Jun–Oct |
| **Bengaluru** | Urban flooding due to poor drainage, mild but unpredictable | Jun–Sep |
| **Pune** | Lowest risk — Western Ghats create rain shadow | Minimal |

### How Risk Translates to Premium

The same Standard-tier worker earning ₹5,000/week pays:

| City | City Risk | Estimated Weekly Premium |
|------|-----------|------------------------|
| Delhi | 1.20× | ₹72–₹100 |
| Chennai | 1.15× | ₹69–₹95 |
| Mumbai | 1.10× | ₹66–₹90 |
| Bengaluru | 1.00× | ₹60–₹82 |
| Hyderabad | 0.95× | ₹60–₹78 |
| Pune | 0.90× | ₹60–₹74 |

(All floored at ₹60/week for Standard tier)

### Seasonal Variation

Premiums re-price every Monday based on the **current week's live weather**. During the Chennai cyclone season (Oct–Dec), a Standard worker's premium can spike to the ₹100 ceiling. In Pune's dry season, it stays at the ₹60 floor.

---

*Generated from live codebase analysis — April 2026*
*WPIP · Worker Protection Insurance Platform*
