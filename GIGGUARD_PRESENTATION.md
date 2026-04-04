# GigGuard Presentation Document

## 1. Project Overview

GigGuard (WPIP: Worker Protection Insurance Platform) is an AI-enabled parametric insurance system for delivery workers in India.

Core goal:
- Protect weekly income when external disruptions stop deliveries.

Coverage focus:
- Income-loss protection only.
- Not a health, life, or vehicle damage policy.

Primary worker platforms supported today:
- Swiggy
- Zomato
- Amazon Flex
- Blinkit
- Zepto
- Meesho
- Porter
- Dunzo

## 2. Problem We Solve

Gig workers are exposed to events they cannot control:
- Heavy rain
- Flooding
- Severe air pollution
- Extreme heat
- Curfew/restriction windows

When disruptions happen, workers lose paid delivery time immediately. Traditional insurance does not match their weekly earning cycle.

GigGuard addresses this with:
- Weekly premiums
- Automated trigger-based claims
- Fast payout workflow
- Fraud-aware validations

## 3. What Is Special About GigGuard

Key differentiators:
- Parametric insurance model: event threshold triggers claim workflow.
- Multi-platform worker support: one worker can be active on multiple delivery apps.
- Dynamic premium estimation using ML and weather risk.
- Hybrid fraud engine: Isolation Forest support plus rule-based scoring.
- Automatic claim creation from live trigger polling.
- Mobile-first worker experience and web-based operational control center.

## 4. Device Support and Product Surfaces

### 4.1 Mobile App (Worker App)

Platform support:
- iOS
- Android
- Expo-based React Native stack

Main screens and capabilities:
- Landing and value proposition
- Login and multi-step registration
- Home dashboard with coverage and risk view
- Policy screen with tier scheduling for next week
- Claims screen with live trigger awareness
- Payments screen with mock payment flow and history
- Profile screen with verification and platform linkage

Mobile data routing behavior:
- Uses backend APIs directly when available.
- Falls back to web API routes when backend direct call is unavailable.

### 4.2 Web App (Worker/Operations Web)

Main routes:
- Login
- Register
- Dashboard

Capabilities:
- Authentication and profile session
- Premium prediction and plan selection
- Trigger-aware dashboard metrics
- Connected platform visualization
- Claims history view
- Payment and payment history endpoints

### 4.3 Admin Web App (Insurer/Operations)

Separate Next.js admin server (port 3001 by default).

Main admin routes:
- Admin dashboard
- Disruption monitor
- Control center

Capabilities:
- Worker verification lifecycle management
- Worker tier changes
- Claim status and payout status operations
- Trigger monitoring across cities
- Synthetic data control and ML retrain controls
- Premium test tooling and operational summaries

## 5. End-to-End System Architecture

High-level flow:
1. Worker registers and links delivery platforms.
2. Worker profile and delivery history are stored in Supabase tables.
3. FastAPI backend computes premiums using worker history plus weather/AQI signals.
4. Scheduler runs synthetic data updates and trigger polling.
5. Trigger engine detects threshold breaches and creates disruption events.
6. Claim engine auto-creates claims for affected workers.
7. Fraud checks and GPS checks adjust claim status.
8. Payment execution endpoint marks approved claims as paid (mock transaction IDs).
9. Mobile, web, and admin dashboards visualize policy, claims, and disruptions.

## 6. Tech Stack (Implemented)

### 6.1 Mobile
- React Native
- Expo
- React Navigation
- AsyncStorage

### 6.2 Web Worker App
- Next.js 16
- React 19
- TypeScript
- Supabase JS client
- JOSE (JWT)
- bcryptjs

### 6.3 Admin App
- Next.js 16
- React 19
- TypeScript
- Supabase JS client

### 6.4 Backend and ML
- FastAPI
- Uvicorn
- Supabase Python client
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- Joblib
- httpx
- bcrypt
- PyJWT

### 6.5 Data and Infrastructure
- Supabase (Postgres + APIs)
- Render deployment model:
  - Web service for FastAPI
  - Cron job every 15 minutes for automated tick

## 7. Methods and Engineering Approaches Used

### 7.1 Synthetic Data Generation Method

Current data source strategy:
- Synthetic worker activity generation for development and model training bootstrap.

How it works:
- Creates a persistent pool of about 200 workers.
- Evenly distributes workers across 8 platforms.
- Generates daily deliveries, earnings, active hours, rating, city data.
- Adds weekend boosts and occasional anomaly spikes.
- Uses deterministic worker IDs for consistent identity across runs.
- Upserts by worker_id + date to keep daily records idempotent.

### 7.2 Weather and Air-Quality Integration Method

External data source:
- OpenWeatherMap APIs:
  - Current weather
  - Air pollution

Signals ingested:
- Temperature
- Humidity
- Rain (1h and 3h)
- Wind speed
- AQI proxy mapped to a 0-500 style index
- PM2.5 and PM10

Reliability method:
- In-memory weather cache with TTL.
- Fallback default weather profile on API failure.

### 7.3 Premium Prediction Method (ML + Fallback)

Model approach:
- XGBoost regressor for weekly premium estimation.

Feature engineering includes:
- Rolling 7-day earnings, deliveries, hours, rating
- 3-day vs 7-day trend features
- Earnings volatility (coefficient of variation)
- Efficiency metrics (earn per hour, earn per delivery)
- Weather and AQI risk features
- City risk weighting

Output behavior:
- Tier-aware premium prediction.
- AutoPay discounted premium variant.
- Tier-based payout caps.

Fallback method:
- If trained model is not available, backend computes premium using rule/formula approach.

### 7.4 Retraining Method

Retrain pipeline:
- Pull latest worker histories.
- Fetch current city weather snapshots.
- Build training dataset.
- Train and evaluate XGBoost.
- Save new model only if quality is acceptable vs previous model.
- Persist metadata and top feature importance.

Scheduler behavior:
- Auto-retrain runs periodically from scheduler loop.
- Manual retrain trigger endpoint available in control center.

### 7.5 Fraud Detection Method

Hybrid strategy:
- Isolation Forest model support for anomaly detection.
- Rule-based scoring fallback for robust operation.

Fraud signals used:
- Cross-platform activity during disruption window
- Missing/failed GPS verification
- High claim frequency over recent window
- High payout ratio vs daily wage
- High cumulative payout trend

Fraud decisioning:
- Claims below threshold can stay auto-initiated/approved.
- High-risk claims move to under-review state.

### 7.6 Parametric Trigger Method

Trigger polling:
- Trigger worker evaluates city weather/AQI regularly.

Current trigger set:
- T-01 heavy rain
- T-02 extreme heat
- T-03 severe AQI
- T-04 flood risk
- T-05 curfew placeholder

Event handling method:
- Upsert one disruption event per trigger/city/date.
- Idempotent conflict key prevents duplicate event rows.

### 7.7 Automatic Claims Method

When trigger fires:
1. Identify workers in affected city.
2. Estimate daily wage from recent history.
3. Compute payout with policy caps.
4. Run cross-platform activity check.
5. Run GPS proximity validation.
6. Build fraud features and score claim.
7. Upsert claim row with status and fraud metadata.

### 7.8 Payment and Payout Method

Current implementation:
- Mock payout execution endpoint for approved claims.
- Generates transaction IDs and marks claims paid.
- Web payment endpoint supports UPI/debit/credit modes.
- Payment history endpoint exposes worker-specific records.

Note:
- Current web payment history uses an in-memory payment store, so persistence is process-lifetime.

## 8. Backend Responsibilities (What Backend Does Today)

FastAPI backend currently handles:
- Authentication (register, login, token verification, profile fetch)
- Delivery ID verification against selected platform tables
- Premium prediction and premium listing
- Weather endpoints for city and monitored set
- Model status and retrain controls
- Synthetic data generation trigger endpoint
- Data summary and income adjustment admin endpoints
- Trigger status and disruption listing
- Claims listing and worker-specific claim retrieval
- Test-fire trigger endpoint for scenario simulation
- Payout execution endpoint for approved claims
- GPS check-in endpoint
- Regional exposure summary endpoint

## 9. Database and Data Model (Current)

Major table groups:
- Platform worker tables (8 separate platform tables)
- Registered workers table
- Premium predictions table
- Weather cache table
- Disruption events table
- Claims table
- GPS check-ins table

Design patterns used:
- Upsert-heavy writes for idempotent scheduler execution
- Composite uniqueness for event and claim deduplication
- Indexed query columns for worker/date/event retrieval speed

## 10. API Surface (Presentation Summary)

Worker/auth APIs:
- login
- register
- me
- verify-id

Premium APIs:
- premium predict
- premium worker latest
- premium all

Trigger and claims APIs:
- trigger status
- trigger test-fire
- disruptions list
- claims list
- claims by worker
- execute payouts

Weather and analytics APIs:
- city weather
- all monitored weather
- admin data summary
- admin exposure

Control APIs:
- admin generate-data
- admin adjust-income
- model retrain
- model retrain logs
- model status

## 11. Current Product Flows (User Journey)

### 11.1 Worker Journey
1. Sign up with personal and platform details.
2. Verify delivery ID against selected platform records.
3. Get weekly premium estimate.
4. Select tier and activate policy intent.
5. Monitor triggers, claims, and payouts from dashboard.
6. Make weekly payment via payment interface.

### 11.2 Disruption to Claim Journey
1. Trigger engine detects disruption threshold.
2. Disruption event is recorded.
3. Affected workers are evaluated.
4. Claims are auto-created with fraud metadata.
5. Approved claims are paid through payout execution path.

### 11.3 Admin/Operations Journey
1. Monitor workers and verification statuses.
2. Monitor real-time disruption dashboard.
3. Observe claims and payout statuses.
4. Trigger manual scenario tests.
5. Retrain premium model and inspect logs.
6. Tune synthetic data multipliers for simulation/testing.

## 12. Security and Access Methods

Implemented security methods:
- Password hashing with bcrypt/bcryptjs.
- JWT token auth using HS256.
- Bearer token protected routes for sensitive operations.
- Server-side Supabase service-role usage in API routes.

## 13. Deployment and Runtime Model

Current runtime layout:
- Web app (Next.js) on local dev server.
- Admin app (Next.js) on separate local port.
- Backend FastAPI service on port 8000.
- Render cron capable of invoking periodic scheduler ticks.

Background behavior:
- Scheduler tick generates/upserts data.
- Trigger polling runs in backend cycle.
- Retraining can run periodically and on-demand.

## 14. What Is Implemented Now vs What Is Mocked

Implemented now:
- Multi-platform registration and ID checks
- Synthetic platform activity generation
- Real weather/AQI fetch
- ML-based premium prediction path
- Trigger detection and disruption event writes
- Auto-claim generation logic with fraud scoring and GPS checks
- Worker dashboards (mobile and web)
- Admin disruption/control views

Mocked or partially simulated:
- External platform integrations are represented by synthetic platform tables
- Payout gateway is mock transaction flow
- Some governance workflows remain admin-assisted

## 15. Presentation Demo Script (Suggested)

1. Start with the problem statement and weekly-income risk.
2. Show worker onboarding (web/mobile registration).
3. Show ID verification and multi-platform linkage.
4. Show premium prediction with weather-aware risk.
5. Show disruption dashboard and test-fire trigger in admin.
6. Show automatic claim creation and fraud metadata.
7. Show payment/payout flow and transaction status.
8. Close with differentiators: parametric automation, ML premium engine, fraud resilience, and multi-device support.

## 16. Elevator Pitch (Use in Presentation Opening)

GigGuard is a parametric, AI-driven income protection platform for delivery workers. It combines synthetic platform simulation, real weather and AQI signals, ML-based premium intelligence, fraud-aware auto-claims, and multi-device operations across mobile, web, and admin control surfaces.

## 17. One-Line Conclusion

GigGuard turns disruption events into automated financial protection for gig workers, with operational intelligence for insurers and a scalable architecture ready for real integrations.
