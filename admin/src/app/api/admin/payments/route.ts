import { NextResponse } from "next/server";
import { createServerClient } from "@/lib/supabase";

function weekStart(date: Date): string {
  const d = new Date(date);
  d.setDate(d.getDate() - ((d.getDay() + 6) % 7)); // Monday
  d.setHours(0, 0, 0, 0);
  return d.toISOString().slice(0, 10);
}

export async function GET() {
  try {
    const db = createServerClient();

    const [paymentsResult, workersResult] = await Promise.all([
      db
        .from("worker_payments")
        .select("transaction_id, worker_id, amount, method, tier, status, created_at")
        .order("created_at", { ascending: false })
        .limit(500),
      db
        .from("registered_workers")
        .select("id, name, email, city, delivery_id"),
    ]);

    if (paymentsResult.error) {
      console.error("Payments fetch error:", paymentsResult.error);
      return NextResponse.json({ error: "Failed to fetch payments." }, { status: 500 });
    }

    const workerMap = new Map(
      (workersResult.data || []).map((w) => [w.id, w]),
    );

    const payments = (paymentsResult.data || []).map((p) => ({
      ...p,
      worker: workerMap.get(p.worker_id) ?? null,
    }));

    const successful = payments.filter((p) => p.status === "success");
    const totalReceived = successful.reduce((s, p) => s + (p.amount || 0), 0);

    // Current week (Mon–Sun)
    const todayKey = weekStart(new Date());
    const thisWeekTotal = successful
      .filter((p) => weekStart(new Date(p.created_at)) === todayKey)
      .reduce((s, p) => s + (p.amount || 0), 0);

    // Weekly breakdown (last 8 weeks)
    const weekMap = new Map<string, { week: string; total: number; count: number }>();
    successful.forEach((p) => {
      const key = weekStart(new Date(p.created_at));
      if (!weekMap.has(key)) weekMap.set(key, { week: key, total: 0, count: 0 });
      const e = weekMap.get(key)!;
      e.total += p.amount || 0;
      e.count += 1;
    });
    const weeklyBreakdown = [...weekMap.values()]
      .sort((a, b) => b.week.localeCompare(a.week))
      .slice(0, 8);

    return NextResponse.json({
      payments,
      analytics: {
        totalReceived,
        thisWeekTotal,
        transactionCount: successful.length,
        weeklyBreakdown,
      },
    });
  } catch (err) {
    console.error("Admin payments error:", err);
    return NextResponse.json({ error: "Internal server error." }, { status: 500 });
  }
}
