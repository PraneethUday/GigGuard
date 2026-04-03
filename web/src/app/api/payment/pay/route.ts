import { NextRequest, NextResponse } from "next/server";
import { verifyToken } from "@/lib/auth";

const BACKEND = process.env.BACKEND_URL || "http://localhost:8000";

function getToken(req: NextRequest): string | null {
  const h = req.headers.get("authorization") || "";
  return h.toLowerCase().startsWith("bearer ") ? h.slice(7).trim() : null;
}

export async function POST(req: NextRequest) {
  const token = getToken(req);
  if (!token) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const workerId = await verifyToken(token);
  if (!workerId) return NextResponse.json({ error: "Invalid token" }, { status: 401 });

  try {
    const body = await req.json();
    const res = await fetch(`${BACKEND}/api/payment/pay-premium`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...body, worker_id: workerId }),
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch {
    return NextResponse.json({ error: "Backend unavailable." }, { status: 502 });
  }
}
