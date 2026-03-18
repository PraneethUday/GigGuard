# 🛡️ GigGuard — AI-Powered Parametric Insurance for India's Gig Economy

> **Guidewire DEVTrails 2026** | Problem Statement: AI-Powered Insurance for India's Gig Economy

---

## 📌 Table of Contents

- [Overview](#overview)
- [The Problem](#the-problem)
- [Our Solution](#our-solution)
- [Persona & Scenarios](#persona--persona-based-scenarios)
- [Application Workflow](#application-workflow)
- [Weekly Premium Model](#weekly-premium-model--how-it-works)
- [Parametric Triggers](#parametric-triggers)
- [AI/ML Integration](#aiml-integration)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture-diagram)
- [Development Plan](#development-plan)

---

## Overview

**GigGuard** is an AI-enabled parametric insurance platform designed exclusively for platform-based delivery partners in India (Zomato, Swiggy, Zepto, Amazon, Blinkit, etc.). It protects their **income** — not their vehicles or health — against uncontrollable external disruptions such as extreme weather, severe pollution, and sudden curfews.

GigGuard operates on a **weekly pricing model** aligned with the gig worker's earnings cycle, and uses **machine learning** to dynamically compute premiums and automate the claims process with near-zero human intervention.

---

## The Problem

India's gig delivery workforce is the backbone of the digital economy. However, workers are fully exposed to external disruptions — heavy rainfall, floods, AQI spikes, curfews — that can wipe out **20–30% of their monthly earnings** with no safety net. No existing insurance product is designed around the week-to-week financial reality of a delivery partner.

---

## Our Solution

GigGuard solves this with three core pillars:

1. **AI-Powered Weekly Premiums** — Dynamic, hyper-local risk pricing computed weekly using ML models trained on weather, location, and earnings history.
2. **Parametric Automation** — Real-time monitoring of disruption triggers (rain, AQI, curfew data). When a threshold is crossed, claims are automatically initiated and payouts processed — no manual claim filing required.
3. **Intelligent Fraud Detection** — Anomaly detection, GPS validation, and duplicate prevention to ensure payout integrity.

> **Coverage Scope:** Income loss ONLY. We strictly exclude health, life, accidents, and vehicle repair coverage.

---

## Persona & Persona-Based Scenarios

### 🎯 Chosen Persona: Food Delivery Partners (Zomato / Swiggy)

Food delivery riders are the most weather-sensitive segment — two-wheeler riders who operate outdoors across all hours and cannot work in extreme rain, heat, or poor visibility. They are paid per delivery and lose income the moment deliveries stop.

---

### 📖 Scenario 1 — Heavy Rainfall (Environmental Disruption)

**Character:** Ravi, a Zomato delivery partner in Chennai, working 10 hours/day, earning ₹700/day.

**Event:** IMD issues a Red Alert for Chennai. Rainfall exceeds 64.5mm in 24 hours (Extreme Rain threshold). Zomato suspends delivery operations in affected zones.

**GigGuard Response:**
- Weather API detects rainfall threshold breach at 6:14 AM.
- GigGuard automatically creates a claim for Ravi's policy covering that disruption window.
- AI validates: Ravi's GPS confirms he was in the active zone; no deliveries were logged; no duplicate claim exists.
- Payout of ₹350 (50% of daily wage for 5 disrupted hours) is transferred to Ravi's UPI/bank in under 10 minutes.
- Ravi receives a WhatsApp/app notification: *"GigGuard has credited ₹350 to your account for today's disruption."*

---

### 📖 Scenario 2 — Severe Air Pollution (Environmental Disruption)

**Character:** Priya, a Swiggy delivery partner in Delhi, 8 hours/day, ₹600/day average.

**Event:** AQI in Priya's zone crosses 401 (Severe category). Delhi government issues a Graded Response Action Plan (GRAP) advisory halting two-wheeler movement.

**GigGuard Response:**
- AQI API detects threshold breach (>400) in Priya's registered zone.
- Parametric trigger fires; GigGuard auto-initiates claim.
- Fraud engine checks: GPS confirms Priya was not working (no delivery pings), cross-references GRAP bulletin.
- Payout processed for lost hours. Priya does not need to file anything manually.

---

### 📖 Scenario 3 — Unplanned Curfew / Section 144 (Social Disruption)

**Character:** Arjun, a Swiggy rider in Bengaluru, working evening slots (5 PM – 11 PM).

**Event:** Local authorities impose a sudden Section 144 curfew in a 3-km radius following unrest. Arjun's zone is locked down for 4 hours during his peak earning window.

**GigGuard Response:**
- GigGuard ingests government notification feeds and news APIs.
- Curfew zone is matched against Arjun's registered delivery zone.
- Parametric trigger fires; claim auto-created for the disrupted 4-hour window.
- Payout released after fraud validation confirms inactivity in Arjun's delivery logs.

---

## Application Workflow

The diagram below (see [Architecture Diagram](#architecture-diagram)) illustrates the full system flow. At a high level:

```
Delivery Partner Onboards
        │
        ▼
[Registration Process]
  ├── Partner submits details via App/Web
  ├── Platform API (Zomato/Swiggy) pulls wage & activity data
  └── AI Risk Profile generated (zone, hours, earnings history)
        │
        ▼
[Premium Computation]
  ├── ML model computes weekly premium (see below)
  ├── Partner reviews premium + policy terms
  └── Partner pays weekly premium (UPI / auto-debit)
        │
        ▼
[Policy Management]
  ├── Active policy monitored in real-time
  ├── Disruption triggers polled every 15 minutes
  └── Policy dashboard visible to partner & insurer
        │
        ▼
[Disruption Detected → Parametric Trigger Fires]
  ├── External condition threshold crossed (Weather / AQI / Curfew)
  ├── Fraud validation runs automatically
  └── Claim auto-initiated (zero manual steps for valid claims)
        │
        ▼
[Claim Management & Payout]
  ├── Payout calculated based on disrupted hours × daily wage rate
  ├── Payment disbursed via UPI / bank transfer (simulated: Razorpay test)
  └── Partner notified via app + WhatsApp
```

---

## Weekly Premium Model — How It Works

### Why Weekly?

Gig workers operate week-to-week. Daily income is variable. A monthly premium model does not align with their cash flow — they may not have ₹200 upfront but can afford ₹40–60 per week, auto-deducted from platform payouts.

---

### Dynamic Premium Calculation (ML-Driven)

The weekly premium is **not flat**. It is dynamically computed every week using our ML model based on the following factors:

| Feature | Description | Example |
|---|---|---|
| `zone_risk_score` | Historical disruption frequency in the worker's delivery zone | High-flood zone in Mumbai = higher score |
| `weather_forecast_7d` | Predicted weather severity for the upcoming week (OpenWeatherMap API) | Red alert forecast = premium ↑ |
| `aqi_forecast_7d` | Air Quality Index prediction for the zone | AQI > 300 forecast = premium ↑ |
| `worker_earnings_avg` | 4-week rolling average daily earnings (from platform API) | Higher earnings = higher coverage = adjusted premium |
| `historical_claim_rate` | Zone-level claim frequency from past data | High claim zone = slight premium ↑ |
| `worker_risk_profile` | Individual worker's hours worked, zone switches, reliability score | Stable worker = loyalty discount |
| `public_holiday_flag` | Government holiday calendar check | Holiday week = lower disruption risk |
| `platform_surge_forecast` | Predicted high-demand days (e.g., weekends, festivals) | High surge = more earnings at risk = higher coverage need |

---

### Premium Formula (Simplified)

```
Base Premium = Weekly Earnings Avg × Coverage Rate (5–8%)

Dynamic Multiplier = f(zone_risk_score, weather_forecast, aqi_forecast)

Final Weekly Premium = Base Premium × Dynamic Multiplier × Loyalty Discount

Coverage Payout = Disrupted Hours × (Weekly Earnings Avg / Active Hours in Week)
```

**Example:**
- Ravi's 4-week average earnings: ₹4,200/week
- Base coverage rate: 6% → Base Premium = ₹252
- Zone risk multiplier: 1.1 (moderate flood zone, Chennai)
- Weather forecast multiplier: 1.2 (heavy rain predicted)
- Loyalty discount: 0.95 (3+ months continuous policy)
- **Final Premium = ₹252 × 1.1 × 1.2 × 0.95 = ₹315.72 → rounded to ₹316/week**

---

### Premium Tiers (Indicative)

| Coverage Tier | Weekly Premium | Max Weekly Payout | Best For |
|---|---|---|---|
| Basic Shield | ₹30–60 | ₹500 | Part-time riders |
| Standard Guard | ₹60–120 | ₹1,200 | Full-time riders |
| Pro Protect | ₹120–220 | ₹2,500 | High-earning riders |

> The ML model selects the most appropriate tier automatically and allows the worker to upgrade/downgrade.

---

## Parametric Triggers

Parametric insurance means: **the event data triggers the claim — not a manual complaint**. When a pre-defined measurable threshold is crossed, the system acts.

### Trigger Table

| Trigger ID | Disruption Type | Data Source | Threshold | Payout Trigger |
|---|---|---|---|---|
| `T-01` | Heavy Rainfall | OpenWeatherMap / IMD API | Rainfall > 64.5mm/24h (Red Alert) OR > 35.5mm/3h | Auto-claim for affected zone |
| `T-02` | Extreme Heat | OpenWeatherMap | Temperature > 45°C AND Heat Index > 54°C | Auto-claim if active in zone |
| `T-03` | Severe Air Pollution | CPCB AQI API / OpenAQ | AQI > 400 (Severe) AND official advisory issued | Auto-claim for GRAP-restricted zones |
| `T-04` | Flood / Waterlogging | OpenWeatherMap + Google Maps Traffic API | Zone tagged as flooded / road closures detected | Auto-claim for registered zone |
| `T-05` | Curfew / Section 144 | Government notification RSS / News API | Official order detected in registered zone | Auto-claim for disrupted window |

### How Triggers Work

```
Every 15 minutes:
  ├── Poll all data sources for active worker zones
  ├── Compare against thresholds
  ├── If threshold crossed → flag disruption event
  ├── Match event zone to policies active in that zone
  ├── Run fraud validation pipeline
  └── If valid → auto-initiate claim and payout
```

---

## AI/ML Integration

### 1. Dynamic Premium Calculation (Scikit-learn / XGBoost)

- **Model Type:** Gradient Boosted Regression (XGBoost)
- **Training Data:** Synthetic + historical weather events, IMD disruption records, simulated delivery platform earnings data
- **Features:** Zone risk score, weather forecast, AQI forecast, earnings avg, claim history, holiday calendar
- **Output:** Weekly premium amount (₹) + recommended coverage tier
- **Retraining:** Model retrained weekly with new disruption event outcomes

### 2. Fraud Detection Engine (Isolation Forest + Rule-Based Layer)

- **Model Type:** Isolation Forest (unsupervised anomaly detection) + hard rule checks
- **Fraud Signals Detected:**
  - GPS location mismatch (worker not in claimed disruption zone)
  - Delivery activity detected during claimed disruption window (platform API cross-check)
  - Duplicate claim submission (same worker, same event, multiple policies)
  - Abnormal claim frequency relative to zone peers
  - GPS spoofing detection (sudden unrealistic location jumps)
- **Output:** Fraud risk score (0–1). Score > 0.75 flags for manual review; Score > 0.9 auto-rejects.

### 3. Risk Profiling (Clustering + Scoring)

- **Model Type:** K-Means clustering for zone-level risk segmentation + logistic scoring for worker profile
- **Purpose:** Assign each delivery zone a risk tier (Low / Medium / High / Extreme) based on historical disruption frequency, drainage quality, curfew history
- **Worker-level:** Individual risk score considering zone history, hours worked, account age, claim history

### 4. Predictive Analytics Dashboard (For Insurers)

- **Model:** Time-series forecasting (Prophet / LSTM) to predict next week's likely claim volume by zone
- **Output:** Insurer dashboard showing predicted loss ratio, zone-wise exposure, upcoming high-risk days

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Web App** | React (TypeScript) | Worker-facing web portal & insurer admin dashboard |
| **Mobile App** | React Native | iOS & Android app for delivery partners |
| **Backend API** | Node.js + Express + TypeScript | REST API server, business logic, policy engine |
| **AI/ML Services** | Python (FastAPI) | ML model serving — premium calculation, fraud detection, risk profiling |
| **ML Libraries** | Scikit-learn, XGBoost, Pandas, NumPy, Prophet | Model training and inference |
| **Database** | PostgreSQL | Policies, claims, worker profiles, transactions |
| **Cache / Queue** | Redis | Real-time trigger polling, job queues for claim processing |
| **External APIs** | OpenWeatherMap, CPCB AQI, IMD RSS, Google Maps | Parametric trigger data sources |
| **Payment (Mock)** | Razorpay Test Mode / UPI Simulator | Simulated payout processing |
| **Authentication** | JWT + OAuth2 | Secure worker and admin login |
| **DevOps** | Docker, GitHub Actions | Containerisation, CI/CD |
| **Cloud** | AWS (EC2, RDS, S3, Lambda) | Hosting, storage, serverless trigger workers |

---

## Architecture Diagram

The architecture below shows the complete system design for GigGuard, covering the interactions between delivery platforms (Zomato/Swiggy), the insurance software core, external data systems, and delivery partners.

![GigGuard Architecture Diagram](./architecture_diagram.png)

**Key Architectural Components:**

- **Delivery Partner App (React Native):** Onboarding, policy review, premium payment, claim status, payout notifications.
- **Web Portal (React):** Insurer admin dashboard — policy management, fraud review, analytics, zone risk maps.
- **Backend (Node.js + Express):** Core business logic — registration, policy engine, premium computation orchestration, claim management.
- **AI/ML Service (Python + FastAPI):** Separate microservice for ML inference (premium model, fraud model, risk profiling). Consumed by the backend via internal REST calls.
- **Parametric Trigger Worker (Python + Redis Queue):** Runs every 15 minutes, polls all external APIs, evaluates thresholds, publishes disruption events to the claim engine.
- **External Integrations:** Weather APIs (OpenWeatherMap / IMD), AQI APIs (CPCB / OpenAQ), Platform APIs (simulated Zomato/Swiggy data), Government advisory feeds.
- **"AI Data" Store:** Historical disruption data, zone risk scores, weather patterns — feeds the ML models for training and inference.

---

## Development Plan

### Phase 1 — Ideation & Foundation [March 4–20] *(Seed Phase)*
**Theme: "Ideate & Know Your Delivery Worker"**

| Week | Tasks |
|---|---|
| Week 1 | Finalise persona (Food Delivery — Zomato/Swiggy). Define parametric triggers. Design data models (worker, policy, claim, payout). Set up project repo and CI/CD pipeline. |
| Week 2 | Build registration flow (web + app). Set up backend scaffolding (Node/Express/TypeScript). Integrate OpenWeatherMap API. Build wireframes for worker app and admin dashboard. Produce architecture diagram. |

**Deliverable:** README (this document) + Git repository + 2-minute strategy video.

---

### Phase 2 — Automation & Protection [March 21 – April 4] *(Scale Phase)*
**Theme: "Protect Your Worker"**

| Week | Tasks |
|---|---|
| Week 3 | Build Registration Process (full flow). Build Insurance Policy Management module. Integrate platform API mock (simulated Zomato/Swiggy worker data). Train initial XGBoost premium model on synthetic data. |
| Week 4 | Build Dynamic Premium Calculation API (Python ML service). Build Claims Management module. Implement 3–5 parametric triggers (T-01 to T-05). Build basic fraud detection rules. Connect React Native app to backend APIs. |

**Deliverable:** Executable source code + 2-minute demo video showcasing Registration, Policy Management, Dynamic Premium Calculation, and Claims Management.

---

### Phase 3 — Scale & Optimise [April 5–17] *(Soar Phase)*
**Theme: "Perfect for Your Worker"**

| Week | Tasks |
|---|---|
| Week 5 | Upgrade fraud detection to Isolation Forest ML model. Add GPS spoofing detection. Integrate Razorpay test mode for simulated instant payouts. Build Worker Dashboard (earnings protected, active coverage, claim history). |
| Week 6 | Build Insurer Admin Dashboard (loss ratios, predictive analytics, zone risk map). Polish UX across web and mobile. Stress-test parametric trigger pipeline. Record 5-minute final demo video. Prepare final pitch deck (PDF). |

**Deliverable:** Advanced fraud detection + instant payout simulation + intelligent dashboards + 5-minute demo video + final pitch deck PDF.

---

## 📁 Repository Structure

```
gigguard/
├── frontend/                  # React Web App (TypeScript)
│   ├── src/
│   │   ├── pages/             # Worker portal & Admin dashboard
│   │   ├── components/
│   │   └── services/          # API clients
├── mobile/                    # React Native App (iOS + Android)
│   ├── src/
│   │   ├── screens/
│   │   └── services/
├── backend/                   # Node.js + Express + TypeScript
│   ├── src/
│   │   ├── routes/            # API routes
│   │   ├── controllers/
│   │   ├── models/            # DB models
│   │   ├── services/          # Policy engine, claim engine
│   │   └── workers/           # Parametric trigger polling workers
├── ai-service/                # Python FastAPI ML Service
│   ├── models/                # Trained ML model files
│   ├── training/              # Training scripts
│   ├── api/                   # FastAPI endpoints
│   └── data/                  # Synthetic training data
├── docs/
│   ├── architecture_diagram.png
│   └── pitch_deck.pdf         # (Phase 3 deliverable)
├── docker-compose.yml
└── README.md
```

---

## ⚠️ Critical Constraints (Per Problem Statement)

- ✅ **Coverage Scope:** Income loss ONLY — no health, life, accident, or vehicle repair coverage.
- ✅ **Weekly Pricing:** All premiums and financial models are structured on a weekly basis.
- ✅ **Persona:** Food delivery partners (Zomato / Swiggy) — a specific, well-defined sub-category.
- ✅ **Parametric Only:** Claims triggered by external data thresholds — not by self-reported incidents.

---

*Built for Guidewire DEVTrails 2026 | Team GigGuard*
