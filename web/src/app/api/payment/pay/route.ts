import { NextRequest, NextResponse } from "next/server";
import { verifyToken } from "@/lib/auth";
import { addPayment } from "@/lib/payment-store";

function getToken(req: NextRequest): string | null {
  const h = req.headers.get("authorization") || "";
  return h.toLowerCase().startsWith("bearer ") ? h.slice(7).trim() : null;
}

function generateTxId(): string {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  const rand = Array.from({ length: 8 }, () =>
    chars[Math.floor(Math.random() * chars.length)]
  ).join("");
  return `GG-${rand}-${Date.now().toString(36).toUpperCase()}`;
}

export async function POST(req: NextRequest) {
  const token = getToken(req);
  if (!token)
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const workerId = await verifyToken(token);
  if (!workerId)
    return NextResponse.json({ error: "Invalid token" }, { status: 401 });

  try {
    const body = await req.json();
    const { amount, method, tier } = body;

    if (!amount || !method) {
      return NextResponse.json(
        { error: "Amount and payment method are required." },
        { status: 400 }
      );
    }

    if (!["upi", "debit", "credit"].includes(method)) {
      return NextResponse.json(
        { error: "Invalid payment method. Use upi, debit, or credit." },
        { status: 400 }
      );
    }

    const record = {
      transaction_id: generateTxId(),
      amount: Number(amount),
      method,
      tier: tier || "standard",
      worker_id: workerId,
      status: "success",
      timestamp: new Date().toISOString(),
    };

    // Persist server-side so both web and mobile can retrieve it via /history
    addPayment(record);

    return NextResponse.json({
      ...record,
      message: "Payment processed successfully.",
    });
  } catch {
    return NextResponse.json(
      { error: "Payment processing failed." },
      { status: 500 }
    );
  }
}
