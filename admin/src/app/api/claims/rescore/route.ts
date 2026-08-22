import { NextRequest, NextResponse } from "next/server";

const BACKEND = process.env.BACKEND_URL || "http://localhost:8000";

export async function POST(req: NextRequest) {
  try {
    const { claimId } = await req.json();
    if (!claimId)
      return NextResponse.json({ error: "claimId required" }, { status: 400 });

    const res = await fetch(`${BACKEND}/api/claims/${claimId}/rescore`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch {
    return NextResponse.json({ error: "Backend unavailable." }, { status: 502 });
  }
}
