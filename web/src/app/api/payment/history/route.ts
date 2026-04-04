import { NextRequest, NextResponse } from "next/server";
import { verifyToken } from "@/lib/auth";
import { getPayments } from "@/lib/payment-store";

function getToken(req: NextRequest): string | null {
  const h = req.headers.get("authorization") || "";
  return h.toLowerCase().startsWith("bearer ") ? h.slice(7).trim() : null;
}

export async function GET(req: NextRequest) {
  const token = getToken(req);
  if (!token) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const workerId = await verifyToken(token);
  if (!workerId) return NextResponse.json({ error: "Invalid token" }, { status: 401 });

  const payments = getPayments(workerId);
  return NextResponse.json({ payments });
}
