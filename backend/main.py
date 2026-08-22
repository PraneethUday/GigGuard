"""
FastAPI application for GigGuard — Delivery Platform Data API + ML Premium Engine.

Exposes endpoints for:
  - Worker activity data (per platform)
  - ML-powered premium prediction
  - Weather data
  - Admin premium management
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import io
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Query, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from db import fetch_workers, fetch_worker_history, supabase
from generator import generate_daily_rows
from auth_utils import verify_password, hash_password, create_token, verify_token
from scheduler import tick, run_retrain, INTERVAL_SECONDS
from ml.weather import fetch_weather, fetch_all_cities
from ml.premium_model import (
    build_features_from_history,
    predict_premium,
    load_model,
    load_metadata,
    compute_target_premium,
    clamp_premium,
    TIER_CONFIG,
)
from triggers import poll_triggers, get_trigger_status, test_fire_trigger
import cooldown as city_cooldown
from ml.traffic import fetch_traffic_tti, compute_traffic_risk
from ml.curfew import evaluate_curfew_risk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retrain state — shared between the background task and the logs endpoint
# ---------------------------------------------------------------------------
retrain_state: dict = {
    "running": False,
    "completed": False,
    "error": None,
    "logs": [],
}


class _LogCapture(io.TextIOBase):
    """Tee: write to both a list and the original stdout."""

    def __init__(self, log_list: list, original: object) -> None:
        self._log_list = log_list
        self._original = original

    def write(self, s: str) -> int:
        stripped = s.rstrip("\n")
        if stripped:
            self._log_list.append(stripped)
        self._original.write(s)  # type: ignore[attr-defined]
        return len(s)

    def flush(self) -> None:
        self._original.flush()  # type: ignore[attr-defined]


async def _retrain_with_logs() -> None:
    retrain_state["running"] = True
    retrain_state["completed"] = False
    retrain_state["error"] = None
    retrain_state["logs"] = []

    original_stdout = sys.stdout
    sys.stdout = _LogCapture(retrain_state["logs"], original_stdout)
    try:
        await run_retrain()
        retrain_state["completed"] = True
    except Exception as exc:  # pylint: disable=broad-except
        retrain_state["error"] = str(exc)
        retrain_state["logs"].append(f"[retrain] ERROR: {exc}")
    finally:
        sys.stdout = original_stdout
        retrain_state["running"] = False


# ---------------------------------------------------------------------------
# Background scheduler as a lifespan task
# ---------------------------------------------------------------------------

async def _scheduler_loop() -> None:
    """Background loop that generates data every 15 minutes."""
    logger.info("Background scheduler started (interval=%ds). Waiting 30s before first tick...", INTERVAL_SECONDS)
    await asyncio.sleep(30)  # Let the server start serving requests first
    while True:
        await tick()
        await asyncio.sleep(INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the background scheduler when the app boots."""
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(_scheduler_loop())
    logger.info("FastAPI lifespan: scheduler task created.")
    
    yield
    
    task.cancel()
    
    # Prevent Uvicorn from hanging on background Supabase/GDELT threads
    if hasattr(loop, '_default_executor') and loop._default_executor:
        loop._default_executor.shutdown(wait=False)
        # Hide the executor so the event loop doesn't try to shut it down again with wait=True
        loop._default_executor = None
        
    # Launch a "sniper" thread to forcefully kill this process after 1 second
    # This guarantees Uvicorn has just enough time to cleanly surrender port 8000
    # but forcefully ends it before Python gets stuck waiting on non-daemon threads forever.
    def _sniper():
        import time, os, signal
        time.sleep(1.0)
        os.kill(os.getpid(), signal.SIGKILL)
    import threading
    threading.Thread(target=_sniper, daemon=True).start()
    
    logger.info("FastAPI lifespan: scheduler task cancelled.")


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GigGuard — Delivery Platform Data API",
    description=(
        "Synthetic delivery-worker data simulator for premium calculation, "
        "fraud detection, and worker risk profiling."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


PLATFORM_TABLE = {
    "swiggy": "swiggy_workers",
    "zomato": "zomato_workers",
    "amazon": "amazon_flex_workers",
    "blinkit": "blinkit_workers",
    "zepto": "zepto_workers",
    "meesho": "meesho_workers",
    "porter": "porter_workers",
    "dunzo": "dunzo_workers",
}


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    name: str
    age: int | None = None
    phone: str
    email: str
    password: str
    city: str
    area: str | None = None
    deliveryId: str
    platforms: list[str] = []
    pan: str | None = None
    aadhaar: str | None = None
    upi: str | None = None
    bank: str | None = None
    consent: bool = False

    autopay: bool = False
    tier: str = "standard"


class VerifyIdRequest(BaseModel):
    deliveryId: str
    platforms: list[str]


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    if not authorization.lower().startswith("bearer "):
        return None
    token = authorization[7:].strip()
    return token or None


def _safe_user(user_row: dict) -> dict:
    allowed_fields = [
        "id",
        "name",
        "email",
        "phone",
        "platforms",
        "tier",
        "verification_status",
        "city",
        "area",
        "delivery_id",
        "autopay",
        "upi",
        "is_active",
    ]
    return {k: user_row.get(k) for k in allowed_fields if k in user_row}


def _normalize_city(city: str | None) -> str:
    value = " ".join(str(city or "").strip().lower().split())
    aliases = {
        "bangalore": "bengaluru",
        "bangaluru": "bengaluru",
        "banglore": "bengaluru",
        "bengaluru": "bengaluru",
        "new delhi": "delhi",
    }
    return aliases.get(value, value)


def _get_registered_claim_user_index() -> dict[str, str]:
    """Return delivery_id -> normalized registered city for claim visibility.

    Only active registrations are included. Rejected registrations are excluded.
    """
    try:
        resp = (
            supabase.table("registered_workers")
            .select("delivery_id, city")
            .eq("is_active", True)
            .neq("verification_status", "rejected")
            .execute()
        )
    except Exception as exc:
        logger.error("Failed to load registered claim users: %s", exc)
        return {}

    index: dict[str, str] = {}
    for row in resp.data or []:
        delivery_id = str(row.get("delivery_id", "")).strip().lower()
        city = _normalize_city(row.get("city"))
        if delivery_id and city:
            index[delivery_id] = city
    return index


def _filter_claim_rows_for_registered_city(
    rows: list[dict],
    registered_user_city_index: dict[str, str],
) -> list[dict]:
    """Keep claims where worker is registered and claim city matches registration city."""
    filtered: list[dict] = []
    for row in rows:
        worker_id = str(row.get("worker_id", "")).strip().lower()
        if not worker_id:
            continue

        registered_city = registered_user_city_index.get(worker_id)
        if not registered_city:
            continue

        claim_city = _normalize_city(row.get("city"))
        if claim_city == registered_city:
            filtered.append(row)

    return filtered


@app.post("/api/auth/login")
def auth_login(req: LoginRequest):
    email = (req.email or "").strip().lower()
    password = req.password or ""

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required.")

    try:
        resp = (
            supabase.table("registered_workers")
            .select("id, name, email, phone, platforms, tier, verification_status, city, area, delivery_id, autopay, upi, is_active, password_hash")
            .eq("email", email)
            .eq("is_active", True)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        logger.error("Auth login query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error.")

    users = resp.data or []
    if not users:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    user = users[0]
    if not verify_password(password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_token(user["id"])
    user.pop("password_hash", None)

    return {"token": token, "user": _safe_user(user)}


@app.post("/api/auth/register")
def auth_register(req: RegisterRequest):
    email = (req.email or "").strip().lower()
    phone = (req.phone or "").strip()
    delivery_id = (req.deliveryId or "").strip()
    name = (req.name or "").strip()

    if not name or not email or not phone or not req.password or not req.platforms or not delivery_id:
        raise HTTPException(status_code=400, detail="Missing required fields.")

    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    try:
        email_exists = (
            supabase.table("registered_workers")
            .select("id")
            .eq("email", email)
            .limit(1)
            .execute()
        )
        if email_exists.data:
            raise HTTPException(status_code=409, detail="An account with this email already exists.")

        phone_exists = (
            supabase.table("registered_workers")
            .select("id")
            .eq("phone", phone)
            .limit(1)
            .execute()
        )
        if phone_exists.data:
            raise HTTPException(status_code=409, detail="An account with this phone already exists.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Auth register duplicate check failed: %s", exc)
        raise HTTPException(status_code=500, detail="Registration failed. Please try again.")

    matched_platforms: list[str] = []
    platform_city: str | None = None
    for platform in req.platforms:
        table = PLATFORM_TABLE.get(platform)
        if not table:
            continue
        try:
            check = (
                supabase.table(table)
                .select("worker_id, city")
                .eq("worker_id", delivery_id)
                .limit(1)
                .execute()
            )
            if check.data:
                matched_platforms.append(platform)
                if not platform_city:
                    platform_city = check.data[0].get("city") or None
        except Exception:
            continue

    # Reject if the selected city doesn't match what the platform has on record.
    if matched_platforms and platform_city:
        if _normalize_city(req.city) != _normalize_city(platform_city):
            raise HTTPException(
                status_code=400,
                detail=f"City mismatch: your delivery ID is registered in {platform_city}, but you selected {req.city}. Please select the correct city.",
            )

    password_hash = hash_password(req.password)
    payload = {
        "name": name,
        "age": req.age,
        "phone": phone,
        "email": email,
        "password_hash": password_hash,
        "city": req.city,
        "area": req.area,
        "delivery_id": delivery_id,
        "platforms": req.platforms,
        "pan": req.pan,
        "aadhaar": req.aadhaar,
        "upi": req.upi,
        "bank": req.bank,
        "consent": bool(req.consent),
        "autopay": bool(req.autopay),
        "tier": req.tier or "standard",
        "verification_status": "pending",
    }

    try:
        insert_resp = (
            supabase.table("registered_workers")
            .insert(payload)
            .select("id, name, email, phone, platforms, tier, verification_status, city, area, delivery_id, autopay, upi, is_active")
            .single()
            .execute()
        )
    except Exception as exc:
        logger.error("Auth register insert failed: %s", exc)
        raise HTTPException(status_code=500, detail="Registration failed. Please try again.")

    new_user = insert_resp.data or {}
    if not new_user:
        raise HTTPException(status_code=500, detail="Registration failed. Please try again.")

    token = create_token(new_user["id"])
    return {
        "token": token,
        "user": _safe_user(new_user),
        "verified": False,
        "auto_verified": len(matched_platforms) > 0,
        "auto_verified_platforms": matched_platforms,
    }


@app.get("/api/auth/me")
def auth_me(authorization: str | None = Header(default=None)):
    token = _extract_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    user_id = verify_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    try:
        resp = (
            supabase.table("registered_workers")
            .select("id, name, email, phone, platforms, tier, verification_status, city, area, delivery_id, autopay, upi, is_active")
            .eq("id", user_id)
            .eq("is_active", True)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        logger.error("Auth me query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error.")

    users = resp.data or []
    if not users:
        raise HTTPException(status_code=404, detail="User not found")

    return {"user": _safe_user(users[0])}


@app.post("/api/verify-id")
def verify_delivery_id(req: VerifyIdRequest):
    delivery_id = (req.deliveryId or "").strip()
    platforms = req.platforms or []

    if not delivery_id or not platforms:
        raise HTTPException(
            status_code=400,
            detail="Delivery ID and at least one platform are required.",
        )

    matched_platforms: list[str] = []
    detected_city: str | None = None
    for platform in platforms:
        table = PLATFORM_TABLE.get(platform)
        if not table:
            continue
        try:
            check = (
                supabase.table(table)
                .select("worker_id, city")
                .eq("worker_id", delivery_id)
                .limit(1)
                .execute()
            )
            if check.data:
                matched_platforms.append(platform)
                if not detected_city:
                    detected_city = check.data[0].get("city") or None
        except Exception:
            continue

    return {
        "verified": len(matched_platforms) > 0,
        "matched_platforms": matched_platforms,
        "checked_platforms": platforms,
        "detected_city": detected_city,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "service": "GigGuard Delivery Platform Data API",
        "version": "2.0.0",
        "platforms": [
            "swiggy", "zomato", "amazon_flex", "blinkit",
            "zepto", "meesho", "porter", "dunzo",
        ],
        "endpoints": [
            "/api/workers",
            "/api/swiggy/workers",
            "/api/zomato/workers",
            "/api/amazon_flex/workers",
            "/api/blinkit/workers",
            "/api/zepto/workers",
            "/api/meesho/workers",
            "/api/porter/workers",
            "/api/dunzo/workers",
            "/api/workers/{worker_id}/history",
            "/api/health",
        ],
    }


@app.get("/api/workers")
async def get_all_workers(limit: int = Query(500, ge=1, le=5000)):
    """Return latest delivery data for **all** platforms combined."""
    data = fetch_workers(platform=None, limit=limit)
    return {"count": len(data), "platform": "all", "data": data}


@app.get("/api/swiggy/workers")
async def get_swiggy_workers(limit: int = Query(500, ge=1, le=5000)):
    """Return latest delivery data for **Swiggy** workers."""
    data = fetch_workers(platform="swiggy", limit=limit)
    return {"count": len(data), "platform": "swiggy", "data": data}


@app.get("/api/zomato/workers")
async def get_zomato_workers(limit: int = Query(500, ge=1, le=5000)):
    """Return latest delivery data for **Zomato** workers."""
    data = fetch_workers(platform="zomato", limit=limit)
    return {"count": len(data), "platform": "zomato", "data": data}


@app.get("/api/amazon_flex/workers")
async def get_amazon_flex_workers(limit: int = Query(500, ge=1, le=5000)):
    """Return latest delivery data for **Amazon Flex** workers."""
    data = fetch_workers(platform="amazon_flex", limit=limit)
    return {"count": len(data), "platform": "amazon_flex", "data": data}


@app.get("/api/blinkit/workers")
async def get_blinkit_workers(limit: int = Query(500, ge=1, le=5000)):
    """Return latest delivery data for **Blinkit** workers."""
    data = fetch_workers(platform="blinkit", limit=limit)
    return {"count": len(data), "platform": "blinkit", "data": data}


@app.get("/api/zepto/workers")
async def get_zepto_workers(limit: int = Query(500, ge=1, le=5000)):
    """Return latest delivery data for **Zepto** workers."""
    data = fetch_workers(platform="zepto", limit=limit)
    return {"count": len(data), "platform": "zepto", "data": data}


@app.get("/api/meesho/workers")
async def get_meesho_workers(limit: int = Query(500, ge=1, le=5000)):
    """Return latest delivery data for **Meesho** workers."""
    data = fetch_workers(platform="meesho", limit=limit)
    return {"count": len(data), "platform": "meesho", "data": data}


@app.get("/api/porter/workers")
async def get_porter_workers(limit: int = Query(500, ge=1, le=5000)):
    """Return latest delivery data for **Porter** workers."""
    data = fetch_workers(platform="porter", limit=limit)
    return {"count": len(data), "platform": "porter", "data": data}


@app.get("/api/dunzo/workers")
async def get_dunzo_workers(limit: int = Query(500, ge=1, le=5000)):
    """Return latest delivery data for **Dunzo** workers."""
    data = fetch_workers(platform="dunzo", limit=limit)
    return {"count": len(data), "platform": "dunzo", "data": data}


@app.get("/api/workers/{worker_id}/history")
def get_worker_history(
    worker_id: str,
    days: int = Query(30, ge=1, le=365),
):
    """Return historical data for a specific worker (for ML features)."""
    data = fetch_worker_history(worker_id=worker_id, days=days)
    return {"count": len(data), "worker_id": worker_id, "days": days, "data": data}


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# ML Premium Endpoints
# ---------------------------------------------------------------------------

class PremiumRequest(BaseModel):
    delivery_id: str
    city: str = "Unknown"
    tier: str = "standard"


@app.post("/api/premium/predict")
def predict_worker_premium(req: PremiumRequest):
    """Predict weekly premium for a single worker using trained ML model."""
    # Fetch worker history
    history = fetch_worker_history(req.delivery_id, days=30)
    weather = fetch_weather(req.city)
    features = build_features_from_history(history, weather)

    model = load_model()
    if model:
        result = predict_premium(model, features, req.tier)
    else:
        raw_premium = compute_target_premium(features, req.tier)
        premium = clamp_premium(raw_premium, req.tier)
        cfg = TIER_CONFIG.get(req.tier, TIER_CONFIG["standard"])
        result = {
            "weekly_premium": premium,
            "weekly_premium_autopay": round(premium * 0.95, 2),
            "raw_prediction": round(raw_premium, 2),
            "tier": req.tier,
            "max_payout": cfg["max_payout"],
            "weather_risk": features.get("weather_risk", 0),
            "city_risk": features.get("city_risk", 1.0),
            "weekly_earnings_est": features.get("weekly_earnings_est", 0),
        }

    result["history_days"] = len(history)
    result["weather"] = weather
    return result


@app.get("/api/premium/worker/{worker_id}")
async def get_worker_premium(worker_id: str):
    """Get the latest computed premium for a registered worker from DB."""
    try:
        resp = (
            supabase.table("premium_predictions")
            .select("*")
            .eq("worker_id", worker_id)
            .order("week_start", desc=True)
            .limit(1)
            .execute()
        )
        if resp.data:
            return {"found": True, "premium": resp.data[0]}
        return {"found": False, "message": "No premium computed yet for this worker."}
    except Exception as e:
        return {"found": False, "error": str(e)}


@app.get("/api/premium/all")
async def get_all_premiums(limit: int = Query(100, ge=1, le=1000)):
    """Get latest computed premiums for all workers."""
    try:
        resp = (
            supabase.table("premium_predictions")
            .select("*")
            .order("computed_at", desc=True)
            .limit(limit)
            .execute()
        )
        return {"count": len(resp.data), "data": resp.data}
    except Exception as e:
        return {"count": 0, "error": str(e), "data": []}


@app.get("/api/weather/{city}")
def get_city_weather(city: str):
    """Get real-time weather + AQI for a city from OpenWeatherMap."""
    data = fetch_weather(city)
    return {"city": city, "weather": data}


@app.get("/api/weather")
def get_all_weather():
    """Get weather for all supported cities."""
    data = fetch_all_cities()
    return {"cities": data}


@app.get("/api/traffic/{city}")
async def get_city_traffic(city: str):
    """Get real-time traffic TTI for a city from TomTom API."""
    data = await fetch_traffic_tti(city)
    risk = compute_traffic_risk(data.get("tti", 1.0))
    return {
        "city": city,
        "traffic": {
            **data,
            "traffic_risk": round(risk, 4),
        },
    }


@app.get("/api/curfew/{city}")
async def get_city_curfew(city: str):
    """Get curfew/unrest risk assessment for a city via GDELT + NLP."""
    data = await evaluate_curfew_risk(city)
    return {"city": city, "curfew": data}


@app.get("/api/model/status")
async def model_status():
    """Get info about the currently loaded ML model."""
    meta = load_metadata()
    model = load_model()
    return {
        "model_loaded": model is not None,
        "metadata": meta,
        "tiers": TIER_CONFIG,
    }


@app.post("/api/model/retrain")
async def trigger_retrain():
    """Start model retraining as a background task and return immediately."""
    if retrain_state["running"]:
        return {"status": "already_running", "message": "Retrain is already in progress."}
    asyncio.create_task(_retrain_with_logs())
    return {"status": "started", "message": "Retrain started in background."}


@app.get("/api/model/retrain/logs")
async def get_retrain_logs():
    """Return current retrain progress: running flag, logs, error, completed."""
    meta = load_metadata() if retrain_state["completed"] else None
    return {
        "running": retrain_state["running"],
        "completed": retrain_state["completed"],
        "error": retrain_state["error"],
        "logs": retrain_state["logs"],
        "metadata": meta,
    }


class AdjustIncomeRequest(BaseModel):
    platform: str | None = None
    worker_id: str | None = None
    earnings_multiplier: float = 1.0
    deliveries_multiplier: float = 1.0


@app.post("/api/admin/adjust-income")
async def adjust_income(req: AdjustIncomeRequest):
    """Adjust worker income data by multiplier for testing the ML model.

    Modifies the last 7 days of data so the change fully affects the
    7-day rolling average used by the ML model. Uses batch upsert (one DB
    call per table) instead of row-by-row updates for speed.
    """
    from datetime import date as date_type, timedelta
    today = date_type.today()
    start_date = (today - timedelta(days=6)).isoformat()
    end_date = today.isoformat()

    valid_tables = {
        "swiggy_workers", "zomato_workers", "amazon_flex_workers",
        "blinkit_workers", "zepto_workers", "meesho_workers",
        "porter_workers", "dunzo_workers",
    }

    tables = []
    if req.platform:
        table_name = f"{req.platform}_workers"
        if table_name in valid_tables:
            tables = [table_name]
    else:
        from db import PLATFORM_TABLES
        tables = PLATFORM_TABLES

    updated_count = 0
    for table in tables:
        try:
            query = (
                supabase.table(table)
                .select("id, earnings, deliveries")
                .gte("date", start_date)
                .lte("date", end_date)
            )
            if req.worker_id:
                query = query.eq("worker_id", req.worker_id)
            resp = query.limit(2000).execute()

            if not resp.data:
                continue

            # Build batch upsert payload — one call per table instead of per row
            updates = [
                {
                    "id": row["id"],
                    "earnings": round(row["earnings"] * req.earnings_multiplier, 2),
                    "deliveries": max(1, int(row["deliveries"] * req.deliveries_multiplier)),
                }
                for row in resp.data
            ]
            supabase.table(table).upsert(updates, on_conflict="id").execute()
            updated_count += len(updates)

        except Exception as e:
            logger.error(f"Error adjusting {table}: {e}")

    return {
        "status": "ok",
        "updated_rows": updated_count,
        "days_modified": 7,
        "earnings_multiplier": req.earnings_multiplier,
        "deliveries_multiplier": req.deliveries_multiplier,
    }


@app.get("/api/admin/data-summary")
async def data_summary():
    """Get summary statistics of current worker data for the control center."""
    from db import PLATFORM_TABLES
    from datetime import date as date_type
    today = date_type.today().isoformat()

    platform_stats = []
    for table in PLATFORM_TABLES:
        try:
            resp = (
                supabase.table(table)
                .select("earnings, deliveries, active_hours, rating")
                .eq("date", today)
                .limit(500)
                .execute()
            )
            rows = resp.data
            if rows:
                earnings = [r["earnings"] for r in rows]
                deliveries = [r["deliveries"] for r in rows]
                platform_stats.append({
                    "platform": table.replace("_workers", ""),
                    "worker_count": len(rows),
                    "avg_earnings": round(sum(earnings) / len(earnings), 2),
                    "min_earnings": round(min(earnings), 2),
                    "max_earnings": round(max(earnings), 2),
                    "avg_deliveries": round(sum(deliveries) / len(deliveries), 1),
                    "total_earnings": round(sum(earnings), 2),
                })
        except Exception:
            pass

    return {
        "date": today,
        "platforms": platform_stats,
        "total_workers": sum(p["worker_count"] for p in platform_stats),
        "overall_avg_earnings": round(
            sum(p["avg_earnings"] * p["worker_count"] for p in platform_stats) /
            max(1, sum(p["worker_count"] for p in platform_stats)), 2
        ) if platform_stats else 0,
    }


@app.post("/api/admin/generate-data")
async def trigger_data_generation():
    """Manually trigger one cycle of data generation.

    Inserts/updates today's synthetic rows for all workers immediately.
    Trigger polling (which requires external HTTP calls to OpenWeatherMap)
    is dispatched as a background task so this endpoint returns fast.
    """
    from datetime import date as date_type
    from db import upsert_rows as _upsert
    today = date_type.today()
    rows = await asyncio.to_thread(generate_daily_rows, target_date=today)
    await asyncio.to_thread(_upsert, rows)
    # Fire trigger polling in the background — don't block the response
    asyncio.create_task(poll_triggers())
    return {
        "status": "ok",
        "message": f"Generated and upserted {len(rows)} rows for {today.isoformat()}.",
        "rows_generated": len(rows),
    }


# ---------------------------------------------------------------------------
# Parametric Trigger & Claims Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/triggers/status")
def triggers_status():
    """Get current weather + trigger evaluation for all monitored cities."""
    return get_trigger_status()


@app.get("/api/disruptions")
async def get_disruptions(limit: int = Query(50, ge=1, le=500)):
    """Get recent disruption events."""
    try:
        resp = (
            supabase.table("disruption_events")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return {"count": len(resp.data), "data": resp.data}
    except Exception as e:
        return {"count": 0, "error": str(e), "data": []}


@app.get("/api/claims")
async def get_claims(limit: int = Query(100, ge=1, le=1000)):
    """Get recent auto-generated claims."""
    try:
        registered_user_city_index = _get_registered_claim_user_index()
        if not registered_user_city_index:
            return {"count": 0, "data": []}

        # Fetch extra rows because city-mismatch rows are filtered in-memory.
        query_limit = min(5000, max(limit * 5, limit))
        resp = (
            supabase.table("claims")
            .select("*")
            .in_("worker_id", sorted(registered_user_city_index.keys()))
            .order("created_at", desc=True)
            .limit(query_limit)
            .execute()
        )
        filtered = _filter_claim_rows_for_registered_city(
            resp.data or [], registered_user_city_index
        )[:limit]
        return {"count": len(filtered), "data": filtered}
    except Exception as e:
        return {"count": 0, "error": str(e), "data": []}


@app.get("/api/claims/worker/{worker_id}")
def get_worker_claims(worker_id: str, after: str | None = None):
    """Get all claims for a specific worker, filtered to on/after their registration date."""
    try:
        registered_user_city_index = _get_registered_claim_user_index()
        worker_key = str(worker_id).strip().lower()
        if worker_key not in registered_user_city_index:
            return {"count": 0, "worker_id": worker_id, "data": []}

        # Always use the registration date from the DB as the lower bound.
        # The caller may pass `after` as an override, but we look it up directly
        # from registered_workers using delivery_id so the filter is authoritative.
        if not after:
            try:
                reg_resp = (
                    supabase.table("registered_workers")
                    .select("created_at")
                    .eq("delivery_id", worker_id)
                    .eq("is_active", True)
                    .limit(1)
                    .execute()
                )
                if reg_resp.data:
                    after = str(reg_resp.data[0].get("created_at", "") or "")
            except Exception:
                pass

        query = (
            supabase.table("claims")
            .select("*")
            .eq("worker_id", worker_id)
        )
        if after:
            query = query.gte("created_at", after)
        query = query.order("created_at", desc=True)
        resp = query.execute()

        filtered = _filter_claim_rows_for_registered_city(
            resp.data or [], registered_user_city_index
        )
        return {"count": len(filtered), "worker_id": worker_id, "data": filtered}
    except Exception as e:
        return {"count": 0, "error": str(e), "data": []}


class ClaimUpdateRequest(BaseModel):
    status: str  # approved, rejected, paid
    admin_notes: str | None = None
    approved_amount: float | None = None


@app.patch("/api/claims/{claim_id}")
def update_claim(claim_id: str, req: ClaimUpdateRequest):
    """Admin endpoint: approve, reject, or mark a claim as paid."""
    allowed = {"approved", "rejected", "paid"}
    if req.status not in allowed:
        return {"error": f"status must be one of {allowed}"}, 400

    now = datetime.utcnow().isoformat()
    update_payload: dict = {
        "status": req.status,
        "payout_status": req.status,
        "reviewed_at": now,
    }

    if req.status == "approved":
        if req.approved_amount is not None:
            update_payload["payout_amount"] = req.approved_amount

    if req.status == "paid":
        import uuid as _uuid
        update_payload["transaction_id"] = f"TXN_MOCK_{_uuid.uuid4().hex[:8].upper()}"
        update_payload["paid_at"] = now

    if req.status == "rejected":
        update_payload["payout_amount"] = 0

    try:
        resp = (
            supabase.table("claims")
            .update(update_payload)
            .eq("id", claim_id)
            .execute()
        )
        if resp.data:
            return {"claim": resp.data[0]}
        return {"error": "Claim not found", "claim": None}
    except Exception as e:
        return {"error": str(e), "claim": None}


@app.post("/api/claims/{claim_id}/rescore")
def rescore_claim(claim_id: str):
    """Re-evaluate fraud score for an existing claim using the worker's registration date as cutoff."""
    from ml.fraud_model import build_fraud_features, compute_fraud_score

    try:
        # Fetch the claim
        resp = supabase.table("claims").select("*").eq("id", claim_id).limit(1).execute()
        if not resp.data:
            return {"error": "Claim not found"}, 404
        claim = resp.data[0]

        worker_id = claim.get("worker_id", "")
        payout_amount = float(claim.get("payout_amount") or 0)
        daily_wage = float(claim.get("daily_wage_est") or 0)
        cross_clear = bool(claim.get("cross_platform_clear", True))
        trigger_type = claim.get("trigger_type", "")

        # Fetch worker registration date for cutoff
        reg_resp = (
            supabase.table("registered_workers")
            .select("created_at")
            .eq("delivery_id", worker_id)
            .eq("is_active", True)
            .limit(1)
            .execute()
        )
        registration_date = None
        if reg_resp.data:
            registration_date = reg_resp.data[0].get("created_at")

        features = build_fraud_features(
            worker_id=worker_id,
            city=claim.get("city", ""),
            payout_amount=payout_amount,
            daily_wage=daily_wage,
            cross_platform_clear=cross_clear,
            supabase_client=supabase,
            trigger_type=trigger_type,
            registration_date=registration_date,
        )
        fraud_score, fraud_flags = compute_fraud_score(features, trigger_type=trigger_type)

        # Update the claim in DB
        new_status = "auto_initiated" if fraud_score < 0.75 else "under_review"
        new_payout_status = "approved" if fraud_score < 0.75 else "pending"
        update_resp = (
            supabase.table("claims")
            .update({
                "fraud_score": fraud_score,
                "fraud_flags": fraud_flags,
                "status": new_status,
                "payout_status": new_payout_status,
            })
            .eq("id", claim_id)
            .execute()
        )
        if update_resp.data:
            return {"claim": update_resp.data[0], "fraud_score": fraud_score, "fraud_flags": fraud_flags}
        return {"error": "Update failed"}
    except Exception as e:
        logger.error("[rescore] Failed for claim %s: %s", claim_id, e)
        return {"error": str(e)}


@app.get("/api/claims/all")
async def get_claims_all(limit: int = Query(200, ge=1, le=1000)):
    """Admin-facing endpoint: returns all claims across all workers."""
    try:
        registered_user_city_index = _get_registered_claim_user_index()
        if not registered_user_city_index:
            return {"count": 0, "claims": [], "data": []}

        # Fetch extra rows because city-mismatch rows are filtered in-memory.
        query_limit = min(5000, max(limit * 5, limit))
        resp = (
            supabase.table("claims")
            .select("*")
            .in_("worker_id", sorted(registered_user_city_index.keys()))
            .order("created_at", desc=True)
            .limit(query_limit)
            .execute()
        )
        filtered = _filter_claim_rows_for_registered_city(
            resp.data or [], registered_user_city_index
        )[:limit]
        return {"count": len(filtered), "claims": filtered, "data": filtered}
    except Exception as e:
        return {"count": 0, "error": str(e), "claims": [], "data": []}


class TestFireRequest(BaseModel):
    city: str = "Chennai"
    trigger_id: str = "T-01"


@app.post("/api/triggers/test-fire")
async def test_fire(req: TestFireRequest, background_tasks: BackgroundTasks):
    """Manually fire a trigger for testing (creates mock disruption + claims).

    Returns immediately. Claim creation runs in the background so the
    UI button doesn\'t freeze during the 10-20s of Supabase queries.
    """
    background_tasks.add_task(
        test_fire_trigger, city=req.city, trigger_id=req.trigger_id
    )
    return {
        "status": "fired",
        "trigger_id": req.trigger_id,
        "city": req.city,
        "claims_created": 0,
        "processing": True,
    }


# ---------------------------------------------------------------------------
# Phase 3: Payouts & Exposure
# ---------------------------------------------------------------------------

class PayoutRequest(BaseModel):
    claim_ids: list[str] = []  # If empty, execute all 'approved' claims

@app.post("/api/claims/execute-payouts")
def execute_payouts(req: PayoutRequest):
    """Mock integration with external payment gateway. Pays out approved claims."""
    try:
        # Fetch approved claims that are not yet paid
        query = supabase.table("claims").select("id").eq("payout_status", "approved")
        if req.claim_ids:
            query = query.in_("id", req.claim_ids)
        
        resp = query.execute()
        claims = resp.data

        if not claims:
            return {"status": "ok", "paid_count": 0, "message": "No approved claims found to pay."}

        paid_count = 0
        import time
        import uuid
        
        # Simulate network delay to payment gateway
        time.sleep(0.5)

        for c in claims:
            txn_id = f"TXN_MOCK_{uuid.uuid4().hex[:8].upper()}"
            supabase.table("claims").update({
                "payout_status": "paid",
                "status": "paid",
                "transaction_id": txn_id,
                "paid_at": datetime.utcnow().isoformat(),
            }).eq("id", c["id"]).execute()
            paid_count += 1

        return {
            "status": "ok",
            "paid_count": paid_count,
            "message": f"Successfully paid {paid_count} claims.",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}




@app.get("/api/admin/exposure")
async def regional_exposure():
    """Get aggregated risk exposure by city for the heatmap."""
    try:
        # Fetch all active workers
        resp = supabase.table("registered_workers").select("id, city, coverage_tier").eq("status", "active").execute()
        workers = resp.data

        # Aggregate by city
        exposure = {}
        for w in workers:
            city = w.get("city", "Unknown")
            tier = w.get("coverage_tier", "basic")
            
            # Max payout estimations based on tier
            max_payout = 500 if tier == "basic" else (1200 if tier == "standard" else 2500)
            
            if city not in exposure:
                exposure[city] = {"worker_count": 0, "max_exposure_inr": 0}
            
            exposure[city]["worker_count"] += 1
            exposure[city]["max_exposure_inr"] += max_payout

        # Also get current weather risk for these cities
        weather = fetch_all_cities()
        for city, stats in exposure.items():
            stats["active_alerts"] = 0
            w = weather.get(city, {})
            if w.get("is_heavy_rain") or w.get("is_extreme_heat") or w.get("is_severe_aqi"):
                stats["active_alerts"] = 1

        return {"status": "ok", "exposure": exposure}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ---------------------------------------------------------------------------
# Admin: city claim cooldown feature flag
# ---------------------------------------------------------------------------

@app.get("/api/admin/cooldown")
def get_cooldown_status():
    """Return the current cooldown feature state and per-city cooldown windows."""
    return city_cooldown.get_status()


class CooldownToggleRequest(BaseModel):
    enabled: bool


@app.post("/api/admin/cooldown/toggle")
def toggle_cooldown(req: CooldownToggleRequest):
    """Enable or disable the city claim cooldown feature."""
    city_cooldown.set_enabled(req.enabled)
    return city_cooldown.get_status()


class SimulatorRequest(BaseModel):
    daily_wage: float = 500.0
    tier: str = "standard"
    temperature: float = 30.0
    rain_1h: float = 0.0
    wind_speed: float = 5.0
    aqi_index: int = 100
    tti: float = 1.0
    unrest_confidence: float = 0.0

@app.post("/api/simulator/evaluate")
async def evaluate_simulator_scenario(req: SimulatorRequest):
    """Evaluate a hypothetical scenario bypassing live metrics."""
    from datetime import date, timedelta
    from ml.premium_model import build_features_from_history, predict_premium
    from fuzzy import compute_trigger_severity
    from claims import compute_payout
    
    # 1. Mock a 7-day worker history
    history = []
    base_date = date.today()
    for i in range(7):
        history.append({
            "date": (base_date - timedelta(days=i)).isoformat(),
            "earnings": req.daily_wage,
            "deliveries": max(1, req.daily_wage / 40.0),
            "active_hours": 8.0,
            "rating": 4.8,
            "city": "SimulatorCity"
        })
        
    weather_dict = {
        "temperature": req.temperature,
        "rain_1h": req.rain_1h,
        "wind_speed": req.wind_speed,
        "aqi_index": req.aqi_index,
        "is_heavy_rain": req.rain_1h >= 15.0,
        "is_extreme_heat": req.temperature >= 45.0,
        "is_severe_aqi": req.aqi_index >= 400,
    }
    traffic_dict = {"tti": req.tti}
    curfew_dict = {"confidence": req.unrest_confidence, "fired": req.unrest_confidence >= 0.8}
    
    features = build_features_from_history(history, weather_dict, traffic_dict, curfew_dict)
    model = load_model()
    premium = predict_premium(model, features, req.tier) if model else 0.0
    
    ctx = {
        **weather_dict,
        "traffic": traffic_dict,
        "tti": req.tti,
        "curfew": curfew_dict,
        "curfew_confidence": req.unrest_confidence
    }
    
    # Evaluate ALL triggers using fuzzy calibration directly (not boolean flags).
    # This lets the simulator show partial severity for values within the
    # fuzzy range (e.g. AQI 390 is within the 300-450 calibration for T-03).
    trigger_ids = ["T-01", "T-02", "T-03", "T-04", "T-05", "T-06"]
    max_severity = 0.0
    active_trigger = "None"
    all_triggers = []

    for tid in trigger_ids:
        _, severity_score = compute_trigger_severity(tid, ctx)
        if severity_score > 0:
            all_triggers.append({"trigger_id": tid, "severity": round(severity_score, 3)})
        if severity_score > max_severity:
            max_severity = severity_score
            active_trigger = tid

    payout = compute_payout(req.daily_wage, 6.0, max_severity, req.tier) if max_severity > 0 else 0.0

    return {
        "predicted_premium": premium,
        "severity": round(max_severity, 2),
        "active_trigger": active_trigger,
        "claim_payout": payout,
        "all_triggers": all_triggers,
    }
