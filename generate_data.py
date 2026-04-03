"""
GigGuard — Synthetic Dataset Generator
Generates all 5 datasets needed for ML model training:
  1. zone_risk.csv
  2. workers.csv
  3. weather_events.csv
  4. policies.csv
  5. claims.csv

Run: python3 generate_data.py
"""

import numpy as np
import pandas as pd
import random
import os
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

OUTPUT_DIR = "/home/claude/gigguard_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# CONSTANTS — Indian Cities, Zones, Platforms
# ─────────────────────────────────────────────

CITIES = {
    "Chennai":    {"base_rain_prob": 0.38, "base_aqi": 95,  "base_curfew_prob": 0.02, "n_zones": 12},
    "Delhi":      {"base_rain_prob": 0.22, "base_aqi": 280, "base_curfew_prob": 0.04, "n_zones": 18},
    "Mumbai":     {"base_rain_prob": 0.42, "base_aqi": 130, "base_curfew_prob": 0.02, "n_zones": 15},
    "Bengaluru":  {"base_rain_prob": 0.30, "base_aqi": 85,  "base_curfew_prob": 0.03, "n_zones": 14},
    "Hyderabad":  {"base_rain_prob": 0.28, "base_aqi": 110, "base_curfew_prob": 0.02, "n_zones": 10},
    "Pune":       {"base_rain_prob": 0.25, "base_aqi": 90,  "base_curfew_prob": 0.01, "n_zones": 8},
    "Kolkata":    {"base_rain_prob": 0.35, "base_aqi": 175, "base_curfew_prob": 0.05, "n_zones": 10},
    "Ahmedabad":  {"base_rain_prob": 0.18, "base_aqi": 140, "base_curfew_prob": 0.02, "n_zones": 8},
}

PLATFORMS = ["Zomato", "Swiggy", "Amazon_Flex", "Blinkit", "Zepto"]

ZONE_NAMES = [
    "Anna Nagar", "Velachery", "T Nagar", "Adyar", "Tambaram",
    "Rohini", "Dwarka", "Lajpat Nagar", "Karol Bagh", "Noida Sec 18",
    "Andheri", "Bandra", "Powai", "Thane", "Dadar",
    "Koramangala", "Whitefield", "Indiranagar", "Marathahalli", "HSR Layout",
    "Hitech City", "Gachibowli", "Kukatpally", "Madhapur",
    "Kothrud", "Wakad", "Hinjewadi", "Viman Nagar",
    "Salt Lake", "New Town", "Howrah", "Jadavpur",
    "Satellite", "SG Highway", "Bopal", "Navrangpura",
    "Miyapur", "Kondapur", "Ameerpet", "Secunderabad",
    "Chembur", "Malad", "Goregaon", "Vasai",
    "Rajajinagar", "Yeshwanthpur", "Jayanagar", "BTM Layout",
]

# ─────────────────────────────────────────────
# DATASET 1 — zone_risk.csv
# ─────────────────────────────────────────────
print("Generating zone_risk.csv ...")

zone_rows = []
zone_id_counter = 1

for city, cfg in CITIES.items():
    for z in range(cfg["n_zones"]):
        zone_name = random.choice(ZONE_NAMES)
        
        # Drainage quality — affects flood risk
        drainage_quality = np.random.choice(["Poor", "Moderate", "Good"], p=[0.3, 0.45, 0.25])
        drainage_score = {"Poor": 0.2, "Moderate": 0.55, "Good": 0.9}[drainage_quality]
        
        # Historical disruption frequency (events per year)
        base_freq = cfg["base_rain_prob"] * 52  # approx weekly events
        hist_disruption_freq = round(np.random.normal(base_freq, base_freq * 0.2), 2)
        hist_disruption_freq = max(1.0, hist_disruption_freq)
        
        # Historical claim rate (claims per active policy per year)
        hist_claim_rate = round(hist_disruption_freq * np.random.uniform(0.55, 0.80), 3)
        
        # Platform activity density (riders per sq km)
        platform_density = round(np.random.uniform(12, 95), 1)
        
        # Waterlogging risk score 0–1
        waterlogging_risk = round(
            (1 - drainage_score) * 0.6 + (cfg["base_rain_prob"] * 0.4) + np.random.uniform(-0.05, 0.05), 3
        )
        waterlogging_risk = min(1.0, max(0.0, waterlogging_risk))
        
        # Composite zone risk score 0–1
        zone_risk_score = round(
            0.35 * (hist_claim_rate / 20) +
            0.25 * waterlogging_risk +
            0.20 * (cfg["base_aqi"] / 400) +
            0.10 * (cfg["base_curfew_prob"] * 10) +
            0.10 * np.random.uniform(0, 0.3),
            4
        )
        zone_risk_score = min(1.0, max(0.05, zone_risk_score))
        
        # Risk tier
        if zone_risk_score < 0.25:
            risk_tier = "Low"
        elif zone_risk_score < 0.50:
            risk_tier = "Medium"
        elif zone_risk_score < 0.75:
            risk_tier = "High"
        else:
            risk_tier = "Extreme"
        
        zone_rows.append({
            "zone_id": f"Z{zone_id_counter:04d}",
            "zone_name": zone_name,
            "city": city,
            "drainage_quality": drainage_quality,
            "drainage_score": round(drainage_score + np.random.uniform(-0.05, 0.05), 3),
            "hist_disruption_freq_per_year": hist_disruption_freq,
            "hist_claim_rate": hist_claim_rate,
            "platform_activity_density": platform_density,
            "waterlogging_risk_score": waterlogging_risk,
            "zone_risk_score": zone_risk_score,
            "risk_tier": risk_tier,
            "curfew_history_count": np.random.poisson(cfg["base_curfew_prob"] * 20),
            "avg_aqi_baseline": round(cfg["base_aqi"] + np.random.normal(0, 20), 1),
        })
        zone_id_counter += 1

df_zones = pd.DataFrame(zone_rows)
df_zones.to_csv(f"{OUTPUT_DIR}/zone_risk.csv", index=False)
print(f"  ✓ zone_risk.csv — {len(df_zones)} zones")

# ─────────────────────────────────────────────
# DATASET 2 — workers.csv
# ─────────────────────────────────────────────
print("Generating workers.csv ...")

FIRST_NAMES = [
    "Ravi", "Priya", "Arjun", "Meena", "Suresh", "Kavitha", "Manoj", "Deepa",
    "Kiran", "Anjali", "Vijay", "Lakshmi", "Rohit", "Sunita", "Arun", "Pooja",
    "Ganesh", "Nisha", "Ramesh", "Divya", "Sanjay", "Rekha", "Amit", "Geeta",
    "Prakash", "Usha", "Venkat", "Saranya", "Muthu", "Bhavani", "Selvam", "Radha",
    "Dinesh", "Chitra", "Balu", "Yamini", "Karthi", "Sindhu", "Senthil", "Padma",
]
LAST_NAMES = [
    "Kumar", "Sharma", "Singh", "Patel", "Reddy", "Nair", "Iyer", "Pillai",
    "Gupta", "Verma", "Das", "Rao", "Joshi", "Mehta", "Shah", "Mishra",
    "Murugan", "Krishnan", "Sundar", "Babu", "Raja", "Devi", "Bhat", "Shetty",
]

N_WORKERS = 5000
zone_ids = df_zones["zone_id"].tolist()
zone_city_map = df_zones.set_index("zone_id")["city"].to_dict()
zone_risk_map = df_zones.set_index("zone_id")["zone_risk_score"].to_dict()

worker_rows = []
start_date = datetime(2024, 1, 1)

for i in range(N_WORKERS):
    worker_id = f"W{i+1:06d}"
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    
    zone_id = random.choice(zone_ids)
    city = zone_city_map[zone_id]
    
    # Number of platforms — most workers on 1-2, some on 3+
    n_platforms = np.random.choice([1, 2, 3, 4], p=[0.50, 0.30, 0.15, 0.05])
    platforms = random.sample(PLATFORMS, n_platforms)
    
    # Registration date — spread over 2 years
    reg_days_ago = np.random.randint(14, 730)
    registration_date = (start_date + timedelta(days=730 - reg_days_ago)).strftime("%Y-%m-%d")
    
    # Weeks of activity
    weeks_active = reg_days_ago // 7
    
    # Eligibility
    is_eligible = weeks_active >= 2
    
    # Hours per day — part-time vs full-time
    worker_type = np.random.choice(["part_time", "full_time", "super_active"], p=[0.25, 0.60, 0.15])
    hours_per_day = {
        "part_time": round(np.random.uniform(3, 5), 1),
        "full_time": round(np.random.uniform(7, 9), 1),
        "super_active": round(np.random.uniform(10, 14), 1),
    }[worker_type]
    
    # Weekly active days
    active_days_per_week = np.random.choice([3, 4, 5, 6, 7], p=[0.10, 0.20, 0.35, 0.25, 0.10])
    
    # Earnings per hour (varies by city and platform count)
    city_earning_multiplier = {
        "Mumbai": 1.20, "Delhi": 1.15, "Bengaluru": 1.10, "Chennai": 1.05,
        "Hyderabad": 1.05, "Pune": 1.00, "Kolkata": 0.95, "Ahmedabad": 0.92,
    }[city]
    
    base_hourly = np.random.uniform(55, 95) * city_earning_multiplier
    multi_platform_bonus = 1 + (n_platforms - 1) * 0.08
    
    weekly_earnings_avg = round(
        base_hourly * hours_per_day * active_days_per_week * multi_platform_bonus
        + np.random.normal(0, 80),
        2
    )
    weekly_earnings_avg = max(400, weekly_earnings_avg)
    
    # Loyalty weeks — continuous payment streak
    loyalty_weeks = min(weeks_active, int(np.random.exponential(8)))
    loyalty_discount = min(0.15, loyalty_weeks * 0.01)
    
    # AutoPay
    autopay_enabled = np.random.choice([True, False], p=[0.55, 0.45])
    autopay_discount = 0.05 if autopay_enabled else 0.0
    
    # Multi-platform score 0-1
    multi_platform_score = round(min(1.0, (n_platforms / 5) + np.random.uniform(0, 0.1)), 3)
    
    # Historical claim count
    hist_claim_count = np.random.poisson(weeks_active * 0.04)
    
    # Phone and UPI (masked)
    phone = f"+91-9{np.random.randint(100000000, 999999999)}"
    
    worker_rows.append({
        "worker_id": worker_id,
        "name": name,
        "phone": phone,
        "city": city,
        "zone_id": zone_id,
        "registration_date": registration_date,
        "weeks_active": weeks_active,
        "is_eligible": is_eligible,
        "worker_type": worker_type,
        "platforms": "|".join(platforms),
        "n_platforms": n_platforms,
        "hours_per_day": hours_per_day,
        "active_days_per_week": active_days_per_week,
        "weekly_earnings_avg": weekly_earnings_avg,
        "loyalty_weeks": loyalty_weeks,
        "loyalty_discount": round(loyalty_discount, 4),
        "autopay_enabled": autopay_enabled,
        "autopay_discount": autopay_discount,
        "multi_platform_score": multi_platform_score,
        "hist_claim_count": hist_claim_count,
        "zone_risk_score": zone_risk_map[zone_id],
    })

df_workers = pd.DataFrame(worker_rows)
df_workers.to_csv(f"{OUTPUT_DIR}/workers.csv", index=False)
print(f"  ✓ workers.csv — {len(df_workers)} workers")

# ─────────────────────────────────────────────
# DATASET 3 — weather_events.csv
# ─────────────────────────────────────────────
print("Generating weather_events.csv ...")

event_rows = []
event_id_counter = 1

# Generate 2 years of weekly weather data per city
for city, cfg in CITIES.items():
    city_zones = df_zones[df_zones["city"] == city]["zone_id"].tolist()
    
    for week in range(104):  # 2 years = 104 weeks
        event_date = (start_date + timedelta(weeks=week)).strftime("%Y-%m-%d")
        month = (start_date + timedelta(weeks=week)).month
        
        # Seasonal modifiers (Indian monsoon pattern)
        monsoon_months = [6, 7, 8, 9, 10]
        winter_months = [11, 12, 1, 2]
        is_monsoon = month in monsoon_months
        is_winter = month in winter_months
        
        rain_modifier = 2.5 if is_monsoon else (0.3 if not is_winter else 0.6)
        aqi_modifier = 1.8 if is_winter else (0.7 if is_monsoon else 1.0)
        
        # Rainfall mm/24h
        rain_probability = min(0.95, cfg["base_rain_prob"] * rain_modifier)
        rain_occurred = np.random.random() < rain_probability
        rainfall_mm = 0.0
        if rain_occurred:
            # Most rain events are moderate; some extreme
            rain_severity = np.random.choice(["light", "moderate", "heavy", "extreme"],
                                              p=[0.35, 0.35, 0.20, 0.10])
            rainfall_mm = {
                "light": round(np.random.uniform(5, 35), 1),
                "moderate": round(np.random.uniform(35, 64), 1),
                "heavy": round(np.random.uniform(64.5, 115), 1),
                "extreme": round(np.random.uniform(115, 200), 1),
            }[rain_severity]
        
        # Temperature
        base_temp = 28 + 6 * np.sin((month - 4) * np.pi / 6)
        temperature = round(base_temp + np.random.normal(0, 3), 1)
        heat_index = round(temperature + np.random.uniform(2, 10), 1)
        
        # AQI
        base_aqi = cfg["base_aqi"] * aqi_modifier
        aqi_value = round(max(30, np.random.normal(base_aqi, base_aqi * 0.25)), 1)
        
        # Curfew
        curfew_imposed = np.random.random() < cfg["base_curfew_prob"]
        curfew_hours = int(np.random.uniform(4, 24)) if curfew_imposed else 0
        
        # Flood / Waterlogging
        flood_detected = rainfall_mm > 64.5 and np.random.random() < 0.60
        
        # Which zones affected
        affected_zones = []
        for zone_id in city_zones:
            zone_rain_threshold = 64.5
            zone_aqi_threshold = 400
            zone_affected = (
                rainfall_mm > zone_rain_threshold or
                aqi_value > zone_aqi_threshold or
                curfew_imposed or
                flood_detected
            )
            if zone_affected:
                affected_zones.append(zone_id)
        
        # Parametric trigger flags
        trigger_rain = rainfall_mm >= 64.5
        trigger_heat = temperature >= 45 and heat_index >= 54
        trigger_aqi = aqi_value >= 400
        trigger_flood = flood_detected
        trigger_curfew = curfew_imposed
        any_trigger = trigger_rain or trigger_heat or trigger_aqi or trigger_flood or trigger_curfew
        
        # Disrupted hours (0–24 depending on severity)
        disrupted_hours = 0
        if trigger_rain:
            disrupted_hours = max(disrupted_hours, int(np.random.uniform(3, 10)))
        if trigger_aqi:
            disrupted_hours = max(disrupted_hours, curfew_hours if curfew_imposed else int(np.random.uniform(4, 12)))
        if trigger_curfew:
            disrupted_hours = max(disrupted_hours, curfew_hours)
        
        event_rows.append({
            "event_id": f"E{event_id_counter:06d}",
            "city": city,
            "event_date": event_date,
            "week_number": week + 1,
            "month": month,
            "is_monsoon_season": is_monsoon,
            "rainfall_mm_24h": rainfall_mm,
            "temperature_celsius": temperature,
            "heat_index_celsius": heat_index,
            "aqi_value": aqi_value,
            "curfew_imposed": curfew_imposed,
            "curfew_hours": curfew_hours,
            "flood_detected": flood_detected,
            "trigger_rain": trigger_rain,
            "trigger_heat": trigger_heat,
            "trigger_aqi": trigger_aqi,
            "trigger_flood": trigger_flood,
            "trigger_curfew": trigger_curfew,
            "any_trigger_fired": any_trigger,
            "disrupted_hours": disrupted_hours,
            "n_affected_zones": len(affected_zones),
            "affected_zone_ids": "|".join(affected_zones) if affected_zones else "",
        })
        event_id_counter += 1

df_events = pd.DataFrame(event_rows)
df_events.to_csv(f"{OUTPUT_DIR}/weather_events.csv", index=False)
print(f"  ✓ weather_events.csv — {len(df_events)} events ({df_events['any_trigger_fired'].sum()} triggered)")

# ─────────────────────────────────────────────
# DATASET 4 — policies.csv
# ─────────────────────────────────────────────
print("Generating policies.csv ...")

policy_rows = []
policy_id_counter = 1

# Only eligible workers get policies
eligible_workers = df_workers[df_workers["is_eligible"] == True].copy()

# Each worker gets policies for some weeks (not necessarily all)
for _, worker in eligible_workers.iterrows():
    n_policy_weeks = min(worker["weeks_active"] - 2, int(np.random.exponential(20)))
    n_policy_weeks = max(1, n_policy_weeks)
    
    city_cfg = CITIES[worker["city"]]
    
    for week_offset in range(n_policy_weeks):
        policy_week = week_offset + 1
        policy_date = (datetime(2024, 1, 1) + timedelta(weeks=week_offset)).strftime("%Y-%m-%d")
        month = (datetime(2024, 1, 1) + timedelta(weeks=week_offset)).month
        
        is_monsoon = month in [6, 7, 8, 9, 10]
        is_winter = month in [11, 12, 1, 2]
        
        # Weather forecast for upcoming week (simulated)
        rain_forecast_prob = city_cfg["base_rain_prob"] * (2.5 if is_monsoon else 0.5)
        rain_forecast_prob = min(0.95, rain_forecast_prob + np.random.uniform(-0.05, 0.10))
        
        aqi_forecast = city_cfg["base_aqi"] * (1.8 if is_winter else 0.8)
        aqi_forecast = max(30, aqi_forecast + np.random.normal(0, 30))
        
        # Zone risk score
        zone_risk = worker["zone_risk_score"]
        
        # Regional exposure factor
        expected_claims_region = rain_forecast_prob * 0.7 * 500  # approx active workers
        regional_exposure_factor = round(1.0 + (rain_forecast_prob * 0.5) + (aqi_forecast / 4000), 4)
        
        # Premium calculation
        base_premium = round(worker["weekly_earnings_avg"] * 0.035, 2)
        
        zone_multiplier = round(1.0 + zone_risk * 0.15, 4)
        weather_multiplier = round(1.0 + rain_forecast_prob * 0.20, 4)
        aqi_multiplier = round(1.0 + max(0, (aqi_forecast - 200) / 2000), 4)
        
        dynamic_multiplier = round(zone_multiplier * weather_multiplier * aqi_multiplier * regional_exposure_factor, 4)
        
        loyalty_discount = worker["loyalty_discount"]
        autopay_discount = worker["autopay_discount"]
        
        final_premium = round(
            base_premium * dynamic_multiplier * (1 - loyalty_discount) * (1 - autopay_discount), 2
        )
        final_premium = max(15.0, min(200.0, final_premium))
        
        # Coverage tier
        earnings = worker["weekly_earnings_avg"]
        if earnings < 1500:
            tier = "Basic Shield"
            max_payout = 500
        elif earnings < 3000:
            tier = "Standard Guard"
            max_payout = 1200
        else:
            tier = "Pro Protect"
            max_payout = 2500
        
        # Payment status
        payment_status = np.random.choice(["Paid", "Failed", "Pending"], p=[0.88, 0.07, 0.05])
        
        # Policy status
        if payment_status == "Paid":
            policy_status = "Active"
        elif payment_status == "Failed":
            policy_status = "Lapsed"
        else:
            policy_status = "Pending"
        
        # Holiday week
        public_holiday_flag = np.random.random() < 0.08
        
        policy_rows.append({
            "policy_id": f"P{policy_id_counter:07d}",
            "worker_id": worker["worker_id"],
            "city": worker["city"],
            "zone_id": worker["zone_id"],
            "policy_week": policy_week,
            "policy_start_date": policy_date,
            "coverage_tier": tier,
            "base_premium": base_premium,
            "zone_risk_score": round(zone_risk, 4),
            "zone_multiplier": zone_multiplier,
            "rain_forecast_prob_7d": round(rain_forecast_prob, 4),
            "weather_multiplier": weather_multiplier,
            "aqi_forecast_7d": round(aqi_forecast, 1),
            "aqi_multiplier": aqi_multiplier,
            "regional_exposure_factor": regional_exposure_factor,
            "dynamic_multiplier": dynamic_multiplier,
            "loyalty_weeks_at_purchase": worker["loyalty_weeks"],
            "loyalty_discount_applied": round(loyalty_discount, 4),
            "autopay_enabled": worker["autopay_enabled"],
            "autopay_discount_applied": round(autopay_discount, 4),
            "final_weekly_premium": final_premium,
            "max_weekly_payout": max_payout,
            "weekly_earnings_avg": worker["weekly_earnings_avg"],
            "n_platforms": worker["n_platforms"],
            "multi_platform_score": worker["multi_platform_score"],
            "public_holiday_flag": public_holiday_flag,
            "payment_status": payment_status,
            "policy_status": policy_status,
        })
        policy_id_counter += 1

df_policies = pd.DataFrame(policy_rows)
df_policies.to_csv(f"{OUTPUT_DIR}/policies.csv", index=False)
print(f"  ✓ policies.csv — {len(df_policies)} policies | Active: {(df_policies['policy_status']=='Active').sum()}")

# ─────────────────────────────────────────────
# DATASET 5 — claims.csv
# ─────────────────────────────────────────────
print("Generating claims.csv ...")

claim_rows = []
claim_id_counter = 1

# Only active policies can generate claims
active_policies = df_policies[df_policies["policy_status"] == "Active"].copy()

# Build a city-week → event lookup
event_lookup = {}
for _, ev in df_events[df_events["any_trigger_fired"] == True].iterrows():
    key = (ev["city"], ev["week_number"])
    event_lookup[key] = ev

worker_map = df_workers.set_index("worker_id").to_dict("index")

for _, policy in active_policies.iterrows():
    city = policy["city"]
    week = policy["policy_week"]
    worker_id = policy["worker_id"]
    
    # Check if a trigger event exists for this city-week
    event = event_lookup.get((city, week))
    if event is None:
        continue
    
    worker = worker_map.get(worker_id)
    if not worker:
        continue
    
    # Probability that worker is affected (not all workers claim every event)
    claim_probability = 0.65 + (policy["zone_risk_score"] * 0.20)
    if np.random.random() > claim_probability:
        continue
    
    # ── Fraud Simulation ──
    # 5% of claims are fraudulent
    is_fraud = np.random.random() < 0.05
    
    # GPS location match
    if is_fraud and np.random.random() < 0.50:
        gps_in_affected_zone = False  # GPS spoofing / wrong zone
        gps_spoof_detected = np.random.random() < 0.70
    else:
        gps_in_affected_zone = True
        gps_spoof_detected = False
    
    # Delivery activity during disruption (fraud = working while claiming)
    if is_fraud and np.random.random() < 0.40:
        delivery_activity_detected = True
    else:
        delivery_activity_detected = False
    
    # Duplicate claim
    duplicate_claim = is_fraud and np.random.random() < 0.20
    
    # Eligibility check
    eligibility_passed = worker["weeks_active"] >= 2
    
    # New registration fraud (registered < 2 weeks before event)
    new_registration_fraud = worker["weeks_active"] < 2
    
    # Abnormal claim frequency
    zone_avg_claim_rate = df_zones[df_zones["zone_id"] == policy["zone_id"]]["hist_claim_rate"].values
    zone_avg = zone_avg_claim_rate[0] if len(zone_avg_claim_rate) > 0 else 0.05
    abnormal_claim_freq = worker["hist_claim_count"] > (zone_avg * 2.5)
    
    # Fraud risk score (rule-based for Phase 2)
    fraud_signals = sum([
        not gps_in_affected_zone,
        delivery_activity_detected,
        duplicate_claim,
        not eligibility_passed,
        new_registration_fraud,
        gps_spoof_detected,
        abnormal_claim_freq,
    ])
    fraud_risk_score = round(min(0.99, fraud_signals * 0.18 + np.random.uniform(0, 0.08)), 4)
    
    # Claim disposition
    if fraud_risk_score > 0.90:
        claim_status = "Auto_Rejected"
        payout_amount = 0.0
    elif fraud_risk_score > 0.75:
        claim_status = "Flagged_Manual_Review"
        payout_amount = 0.0
    else:
        claim_status = "Auto_Approved"
        # Payout = disrupted hours × hourly wage
        hourly_wage = worker["weekly_earnings_avg"] / (worker["hours_per_day"] * worker["active_days_per_week"])
        disrupted_hours = min(event["disrupted_hours"], worker["hours_per_day"])
        payout_amount = round(hourly_wage * disrupted_hours, 2)
        payout_amount = min(payout_amount, policy["max_weekly_payout"])
    
    # Processing time (minutes)
    if claim_status == "Auto_Approved":
        processing_time_mins = round(np.random.uniform(2, 12), 1)
    elif claim_status == "Flagged_Manual_Review":
        processing_time_mins = round(np.random.uniform(120, 4320), 1)  # hours to days
    else:
        processing_time_mins = round(np.random.uniform(1, 5), 1)
    
    # Trigger type
    trigger_type = []
    if event["trigger_rain"]:   trigger_type.append("Rain")
    if event["trigger_aqi"]:    trigger_type.append("AQI")
    if event["trigger_curfew"]: trigger_type.append("Curfew")
    if event["trigger_flood"]:  trigger_type.append("Flood")
    if event["trigger_heat"]:   trigger_type.append("Heat")
    
    claim_rows.append({
        "claim_id": f"C{claim_id_counter:07d}",
        "policy_id": policy["policy_id"],
        "worker_id": worker_id,
        "event_id": event["event_id"],
        "city": city,
        "zone_id": policy["zone_id"],
        "claim_date": event["event_date"],
        "week_number": week,
        "trigger_type": "|".join(trigger_type) if trigger_type else "Unknown",
        "rainfall_mm": event["rainfall_mm_24h"],
        "aqi_value": event["aqi_value"],
        "temperature": event["temperature_celsius"],
        "curfew_imposed": event["curfew_imposed"],
        "disrupted_hours": event["disrupted_hours"],
        # Fraud fields
        "gps_in_affected_zone": gps_in_affected_zone,
        "gps_spoof_detected": gps_spoof_detected,
        "delivery_activity_detected": delivery_activity_detected,
        "duplicate_claim": duplicate_claim,
        "eligibility_passed": eligibility_passed,
        "new_registration_fraud": new_registration_fraud,
        "abnormal_claim_freq": abnormal_claim_freq,
        "fraud_risk_score": fraud_risk_score,
        "is_fraud_ground_truth": is_fraud,
        # Outcome
        "claim_status": claim_status,
        "payout_amount": payout_amount,
        "processing_time_mins": processing_time_mins,
        "n_platforms_verified": worker["n_platforms"],
        "weekly_earnings_avg": worker["weekly_earnings_avg"],
        "premium_paid": policy["final_weekly_premium"],
    })
    claim_id_counter += 1

df_claims = pd.DataFrame(claim_rows)
df_claims.to_csv(f"{OUTPUT_DIR}/claims.csv", index=False)
print(f"  ✓ claims.csv — {len(df_claims)} claims")
print(f"     Auto Approved : {(df_claims['claim_status']=='Auto_Approved').sum()}")
print(f"     Manual Review : {(df_claims['claim_status']=='Flagged_Manual_Review').sum()}")
print(f"     Auto Rejected : {(df_claims['claim_status']=='Auto_Rejected').sum()}")
print(f"     Fraud (ground truth): {df_claims['is_fraud_ground_truth'].sum()}")

# ─────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  GIGGUARD SYNTHETIC DATASET — GENERATION COMPLETE")
print("="*55)

datasets = {
    "zone_risk.csv":      df_zones,
    "workers.csv":        df_workers,
    "weather_events.csv": df_events,
    "policies.csv":       df_policies,
    "claims.csv":         df_claims,
}

total_rows = 0
for name, df in datasets.items():
    size_kb = os.path.getsize(f"{OUTPUT_DIR}/{name}") / 1024
    print(f"  {name:<25} {len(df):>7,} rows   {size_kb:>7.1f} KB")
    total_rows += len(df)

print(f"  {'─'*45}")
print(f"  {'TOTAL':.<25} {total_rows:>7,} rows")
print("="*55)
print(f"\n  All files saved to: {OUTPUT_DIR}/\n")
