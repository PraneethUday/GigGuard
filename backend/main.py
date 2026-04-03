"""
FastAPI application for GigGuard — Delivery Platform Data API + ML Premium Engine.

Exposes endpoints for:
  - Worker activity data (per platform)
  - ML-powered premium prediction
  - Weather data
  - Admin premium management
"""

import asyncio
import io
import logging
import random
import string
import sys
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from db import fetch_workers, fetch_worker_history, supabase
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
    logger.info("Background scheduler started (interval=%ds).", INTERVAL_SECONDS)
    while True:
        await tick()
        await asyncio.sleep(INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the background scheduler when the app boots."""
    task = asyncio.create_task(_scheduler_loop())
    logger.info("FastAPI lifespan: scheduler task created.")
    yield
    task.cancel()
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
async def get_worker_history(
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
async def predict_worker_premium(req: PremiumRequest):
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
async def get_city_weather(city: str):
    """Get real-time weather + AQI for a city from OpenWeatherMap."""
    data = fetch_weather(city)
    return {"city": city, "weather": data}


@app.get("/api/weather")
async def get_all_weather():
    """Get weather for all supported cities."""
    data = fetch_all_cities()
    return {"cities": data}


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

    This modifies TODAY's data for the specified platform or worker,
    letting the admin see how the model responds to different income levels.
    """
    from datetime import date as date_type
    today = date_type.today().isoformat()

    tables = []
    if req.platform:
        table_name = f"{req.platform}_workers"
        if table_name in [
            "swiggy_workers", "zomato_workers", "amazon_flex_workers",
            "blinkit_workers", "zepto_workers", "meesho_workers",
            "porter_workers", "dunzo_workers",
        ]:
            tables = [table_name]
    else:
        from db import PLATFORM_TABLES
        tables = PLATFORM_TABLES

    updated_count = 0
    for table in tables:
        try:
            query = supabase.table(table).select("id, earnings, deliveries").eq("date", today)
            if req.worker_id:
                query = query.eq("worker_id", req.worker_id)
            resp = query.limit(500).execute()

            for row in resp.data:
                new_earnings = round(row["earnings"] * req.earnings_multiplier, 2)
                new_deliveries = max(1, int(row["deliveries"] * req.deliveries_multiplier))
                supabase.table(table).update({
                    "earnings": new_earnings,
                    "deliveries": new_deliveries,
                }).eq("id", row["id"]).execute()
                updated_count += 1
        except Exception as e:
            logger.error(f"Error adjusting {table}: {e}")

    return {
        "status": "ok",
        "updated_rows": updated_count,
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
    """Manually trigger one cycle of data generation."""
    await tick()
    return {"status": "ok", "message": "Data generation cycle completed."}


# ---------------------------------------------------------------------------
# Mock Payment helpers
# ---------------------------------------------------------------------------

def _mock_payment_id() -> str:
    chars = string.ascii_letters + string.digits
    return "rzp_mock_" + "".join(random.choices(chars, k=14))


def _mock_payout_id() -> str:
    chars = string.ascii_letters + string.digits
    return "pout_mock_" + "".join(random.choices(chars, k=14))


def _current_week() -> tuple[date, date]:
    today = date.today()
    monday = today - timedelta(days=today.weekday())
    sunday = monday + timedelta(days=6)
    return monday, sunday


# ---------------------------------------------------------------------------
# Payment endpoints
# ---------------------------------------------------------------------------

class PayPremiumRequest(BaseModel):
    worker_id: str
    amount: float
    tier: str
    payment_method: str = "upi"


@app.post("/api/payment/pay-premium")
async def pay_premium(req: PayPremiumRequest):
    """Record a mock Razorpay premium payment for the current week."""
    if req.tier not in ("basic", "standard", "pro"):
        return {"error": "Invalid tier."}, 400
    if req.payment_method not in ("upi", "card", "netbanking"):
        return {"error": "Invalid payment method."}, 400

    week_start, week_end = _current_week()

    # Prevent double-payment for same week
    existing = (
        supabase.table("worker_payments")
        .select("id")
        .eq("worker_id", req.worker_id)
        .eq("week_start", week_start.isoformat())
        .eq("status", "success")
        .limit(1)
        .execute()
    )
    if existing.data:
        return {"error": "Premium already paid for this week.", "already_paid": True}

    payment_id = _mock_payment_id()
    resp = supabase.table("worker_payments").insert({
        "worker_id": req.worker_id,
        "amount": round(req.amount, 2),
        "tier": req.tier,
        "payment_method": req.payment_method,
        "razorpay_payment_id": payment_id,
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "status": "success",
    }).execute()

    if not resp.data:
        return {"error": "Failed to record payment."}

    return {
        "status": "success",
        "payment_id": payment_id,
        "amount": round(req.amount, 2),
        "tier": req.tier,
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "message": f"Payment of ₹{req.amount:.0f} recorded successfully.",
    }


@app.get("/api/payment/history/{worker_id}")
async def payment_history(worker_id: str):
    """Fetch premium payment history for a worker."""
    resp = (
        supabase.table("worker_payments")
        .select("*")
        .eq("worker_id", worker_id)
        .order("created_at", desc=True)
        .limit(20)
        .execute()
    )
    # Check if this week's premium is already paid
    week_start, _ = _current_week()
    paid_this_week = any(
        p["week_start"] == week_start.isoformat() and p["status"] == "success"
        for p in resp.data
    )
    return {"payments": resp.data, "paid_this_week": paid_this_week}


# ---------------------------------------------------------------------------
# Claims endpoints
# ---------------------------------------------------------------------------

class FileClaimRequest(BaseModel):
    worker_id: str
    worker_name: str
    claim_type: str
    description: str
    incident_date: str
    claim_amount: float
    upi: str | None = None


@app.post("/api/claims/file")
async def file_claim(req: FileClaimRequest):
    """File a new insurance claim."""
    valid_types = ["accident", "income_loss", "weather_disruption", "other"]
    if req.claim_type not in valid_types:
        return {"error": "Invalid claim type."}
    if req.claim_amount <= 0:
        return {"error": "Claim amount must be positive."}

    resp = supabase.table("insurance_claims").insert({
        "worker_id": req.worker_id,
        "worker_name": req.worker_name,
        "claim_type": req.claim_type,
        "description": req.description,
        "incident_date": req.incident_date,
        "claim_amount": round(req.claim_amount, 2),
        "upi": req.upi or None,
        "status": "pending",
    }).execute()

    if not resp.data:
        return {"error": "Failed to file claim."}

    return {"status": "ok", "claim": resp.data[0], "message": "Claim filed successfully. We will review it within 24 hours."}


@app.get("/api/claims/worker/{worker_id}")
async def worker_claims(worker_id: str):
    """Get all claims filed by a specific worker."""
    resp = (
        supabase.table("insurance_claims")
        .select("*")
        .eq("worker_id", worker_id)
        .order("created_at", desc=True)
        .limit(20)
        .execute()
    )
    return {"claims": resp.data}


@app.get("/api/claims/all")
async def all_claims(limit: int = Query(100, ge=1, le=500)):
    """Admin: get all insurance claims."""
    resp = (
        supabase.table("insurance_claims")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return {"count": len(resp.data), "claims": resp.data}


class UpdateClaimRequest(BaseModel):
    status: str
    approved_amount: float | None = None
    admin_notes: str | None = None


@app.patch("/api/claims/{claim_id}")
async def update_claim(claim_id: str, req: UpdateClaimRequest):
    """Admin: approve, reject, or mark a claim as paid."""
    valid_statuses = ["pending", "under_review", "approved", "rejected", "paid"]
    if req.status not in valid_statuses:
        return {"error": "Invalid status."}

    update_data: dict = {
        "status": req.status,
        "updated_at": datetime.utcnow().isoformat(),
    }
    if req.approved_amount is not None:
        update_data["approved_amount"] = round(req.approved_amount, 2)
    if req.admin_notes is not None:
        update_data["admin_notes"] = req.admin_notes
    if req.status == "paid":
        update_data["razorpay_payout_id"] = _mock_payout_id()

    resp = (
        supabase.table("insurance_claims")
        .update(update_data)
        .eq("id", claim_id)
        .execute()
    )

    if not resp.data:
        return {"error": "Claim not found or update failed."}

    return {"status": "ok", "claim": resp.data[0]}
